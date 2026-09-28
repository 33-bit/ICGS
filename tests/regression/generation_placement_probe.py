"""Explicit bounded simulator/archive regression; never publishes or trains.

Run with the generation environment, --runtime-config and a fresh --output-root.
The first two former failing seeds plus middle/upper scale strata are selected.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from icgs.data.collection.generation.batch import AttemptPlanner, bounds_from_row
from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig, GenerationJob, RunConfig
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.distributed_validation import validate_closed_result
from scripts.generation_launch import build_process_environment, build_worker_commands, persist_run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    # The base run may not exist: this probe creates its own isolated run root.
    base = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    repo = Path(base.machine.repo_root)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = replace(base,
        machine=replace(base.machine, worker_ids=('000',), simulator_slots=1, worker_timeout_s=600),
        run=replace(base.run, run_id=root.name, run_root=str(root), worker_count=1,
                    publication_enabled=False, resume_from_hf=False, validation_mode=True,
                    publication_verify_threads=1),
        max_result_bytes=1024**3, max_staging_bytes=16*1024**3, staging_reserve_bytes=2*1024**3)
    if config.archive_profile is None:
        raise ValueError('probe requires the canonical archive profile')
    approved = repo/'artifacts/composition/approved_composition_manifest.json'
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    run_path = persist_run_config(config, approved, code_revision=revision)
    run = RunConfig.from_dict(json.loads(run_path.read_text())['run'])
    catalog = {r['program_id']: r for r in json.loads(approved.read_text())['catalog']}
    queue = FilesystemJobQueue(root/'queue')
    queue.configure_runtime_staging(config)
    jobs = []
    selections = {'T06': (0, 1, 12, 24), 'V02': (0, 1, 12, 24), 'T08': (0, 1), 'T01': (0, 1)}
    if os.environ.get('ICGS_PLACEMENT_SELECTIONS'):
        selections = json.loads(os.environ['ICGS_PLACEMENT_SELECTIONS'])
    for program, indices in selections.items():
        planner = AttemptPlanner(program, bounds=bounds_from_row(catalog[program]),
                                 asset_family_id=catalog[program].get('asset_family_id'))
        for index in range(max(indices)+1):
            plan = planner.next_plan('nominal')
            if index not in indices:
                continue
            job = GenerationJob.create(job_id=f'job-{root.name}-{program}-{index}', run_id=run.run_id,
                attempt_id=f'att-{plan.episode_id}', episode_id=plan.episode_id, program_id=program,
                plan=plan, code_revision=revision, manifest_sha256=run.approved_manifest_sha256,
                output_root=str(root/'staging'))
            queue.enqueue(job)
            jobs.append(job)
    started = time.monotonic()
    # Worker leases are intentionally durable after process exit.  A bounded
    # sequential probe therefore reuses the one configured worker identity
    # and passes one stable instance identity on every --once invocation.
    # This keeps lease ownership explicit without weakening the queue's
    # orphan/duplicate-worker protections.
    worker_instance_id = f'placement-probe-{root.name}'
    for index in range(len(jobs)):
        worker_config = config
        environment = build_process_environment(worker_config)
        # Keep diagnostics opt-in; normal generation does not capture a trace.
        environment['ICGS_GENERATION_TRACE'] = os.environ.get('ICGS_PLACEMENT_TRACE', '0')
        command = build_worker_commands(worker_config, approved, runtime_config_path=root/'control/runtime_config.json')[0]
        with (root/f'worker-{index:02}.log').open('w') as log:
            result = subprocess.run(command+['--once', '--worker-instance-id', worker_instance_id], cwd=repo, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=750)
        if result.returncode:
            raise RuntimeError(f'worker {index} exited {result.returncode}; see its log')
        print('WORKER_COMPLETED', index, flush=True)
    results = []
    for result in queue.iter_ready():
        job = next(j for j in jobs if j.job_id == result.job_id)
        validate_closed_result(job, result, archive_profile=config.archive_profile)
        results.append({'program_id': job.program_id, 'episode_index': job.plan.episode_index,
                        'scene_seed': job.plan.scene_seed, 'outcome': result.outcome,
                        'scale': job.plan.randomization['scale'], 'archive_validation': 'PASS'})
        print('VALIDATED', json.dumps(results[-1]), flush=True)
    proof = {
        'status': 'PASS' if len(results) == len(jobs) and all(r['outcome']=='success' for r in results) else 'FAIL',
        'code_revision': revision,
        'source_sha256': {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
                          ('scripts/generation_episode_worker.py', 'scripts/generation_worker.py',
                           'src/icgs/data/collection/generation/expert.py',
                           'tests/regression/generation_placement_probe.py')},
        'git_status': subprocess.check_output(['git','status','--short'], cwd=repo,text=True),
        'elapsed_s': time.monotonic()-started, 'expected_results': len(jobs), 'results': results,
        'trace_enabled': environment['ICGS_GENERATION_TRACE'],
        'publication': 'NOT RUN', 'training': 'NOT RUN',
    }
    (root/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)
    return int(proof['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
