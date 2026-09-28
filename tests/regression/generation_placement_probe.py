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
from concurrent.futures import ThreadPoolExecutor

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
    parallelism = max(1, int(os.environ.get('ICGS_PLACEMENT_PARALLELISM', '1')))
    selections = {'T06': (0, 1, 12, 24), 'V02': (0, 1, 12, 24), 'T08': (0, 1), 'T01': (0, 1)}
    if os.environ.get('ICGS_PLACEMENT_SELECTIONS'):
        selections = json.loads(os.environ['ICGS_PLACEMENT_SELECTIONS'])
    job_count = sum(len(indices) for indices in selections.values())
    if job_count < 1:
        raise ValueError('placement probe requires at least one selected job')
    # A --once process owns exactly one immutable claim.  Unique identities per
    # job avoid the heartbeat/claim handoff window when a slot immediately
    # starts another job while other workers are still writing archive scratch.
    worker_ids = tuple(f'{index:03}' for index in range(job_count))
    max_result_bytes = 1024**3
    staging_reserve_bytes = 2 * 1024**3
    # The probe retains every closed archive locally until validation finishes;
    # size admission for the selected workload instead of stopping after the
    # first fixed number of reservations.
    max_staging_bytes = max(
        16 * 1024**3,
        (2 * job_count + 2) * max_result_bytes + staging_reserve_bytes,
    )
    config = replace(base,
        machine=replace(base.machine, worker_ids=worker_ids, simulator_slots=parallelism, worker_timeout_s=600),
        run=replace(base.run, run_id=root.name, run_root=str(root), worker_count=job_count,
                    publication_enabled=False, resume_from_hf=False, validation_mode=True,
                    publication_verify_threads=1),
        max_result_bytes=max_result_bytes,
        max_staging_bytes=max_staging_bytes,
        staging_reserve_bytes=staging_reserve_bytes)
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
    # A slot may process many jobs over the lifetime of this probe.  Let
    # xvfb-run allocate a fresh free display for each invocation so stale
    # X11 locks from an interrupted simulator cannot poison later jobs.
    commands = build_worker_commands(
        config,
        approved,
        runtime_config_path=root/'control/runtime_config.json',
        auto_servernum=True,
    )
    def run_one(index: int) -> None:
        worker_index = index
        environment = build_process_environment(config)
        environment['ICGS_GENERATION_TRACE'] = os.environ.get('ICGS_PLACEMENT_TRACE', '0')
        worker_instance_id = f'placement-probe-{root.name}-{worker_ids[worker_index]}'
        with (root/f'worker-{index:02}.log').open('w') as log:
            result = subprocess.run(commands[worker_index] + ['--once', '--worker-instance-id', worker_instance_id],
                                    cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=750)
        if result.returncode:
            raise RuntimeError(f'worker {index} exited {result.returncode}; see its log')
        print('WORKER_COMPLETED', index, flush=True)
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        list(executor.map(run_one, range(len(jobs))))
    ready_results = queue.iter_ready()
    if len(ready_results) != len(jobs):
        raise RuntimeError(f'expected {len(jobs)} ready results, found {len(ready_results)}')

    def validate_one(result):
        job = next(j for j in jobs if j.job_id == result.job_id)
        validate_closed_result(job, result, archive_profile=config.archive_profile)
        return {'program_id': job.program_id, 'episode_index': job.plan.episode_index,
                'scene_seed': job.plan.scene_seed, 'outcome': result.outcome,
                'scale': job.plan.randomization['scale'], 'archive_validation': 'PASS'}

    validation_parallelism = max(1, int(os.environ.get(
        'ICGS_PLACEMENT_VALIDATION_PARALLELISM', str(parallelism))))
    with ThreadPoolExecutor(max_workers=validation_parallelism) as executor:
        results = list(executor.map(validate_one, ready_results))
    for row in results:
        print('VALIDATED', json.dumps(row), flush=True)
    proof = {
        'status': 'PASS' if len(results) == len(jobs) and all(r['outcome']=='success' for r in results) else 'FAIL',
        'code_revision': revision,
        'source_sha256': {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
                          ('scripts/generation_episode_worker.py', 'scripts/generation_worker.py',
                           'src/icgs/data/collection/generation/expert.py',
                           'tests/regression/generation_placement_probe.py')},
        'git_status': subprocess.check_output(['git','status','--short'], cwd=repo,text=True),
        'elapsed_s': time.monotonic()-started, 'expected_results': len(jobs), 'results': results,
        'trace_enabled': os.environ.get('ICGS_PLACEMENT_TRACE', '0'),
        'parallelism': parallelism, 'validation_parallelism': validation_parallelism,
        'publication': 'NOT RUN', 'training': 'NOT RUN',
    }
    (root/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)
    return int(proof['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
