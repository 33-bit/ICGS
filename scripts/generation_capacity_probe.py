"""Preflight a bounded real-generation capacity probe.

Execution is deliberately explicit: this command validates the probe contract
and prints the staged limits. Operators then invoke the canonical launcher with
one stage at a time; no production quota is started implicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.generation.capacity_probe import CapacityProbeConfig
from icgs.data.collection.generation.distributed_contracts import (
    GenerationRuntimeConfig,
    RunConfig,
)
from icgs.data.collection.generation.distributed_planner import DistributedPlanner
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.distributed_validation import validate_closed_result


def enqueue_fixed_jobs(planner, queue, cap: int) -> list[str]:
    """Plan exactly one finite batch; never refill while workers are running."""
    job_ids = []
    for _ in range(cap):
        job = planner.next_job()
        if job is None:
            raise RuntimeError("planner exhausted before capacity stage cap")
        queue.enqueue(job)
        job_ids.append(job.job_id)
    return job_ids


def stop_processes(processes: list[subprocess.Popen]) -> None:
    """Stop only process groups started by this probe."""
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in active:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def _stage_runtime(base: GenerationRuntimeConfig, probe: CapacityProbeConfig, stage, root: Path) -> GenerationRuntimeConfig:
    run = replace(
        base.run,
        run_id=f"{probe.run_id}-{stage.name}",
        run_root=str(root),
        worker_count=stage.worker_count,
        hf_subfolder=probe.hf_subfolder,
        publication_enabled=False,
        validation_mode=True,
        resume_from_hf=False,
    )
    machine = replace(base.machine, simulator_slots=stage.simulator_slots, worker_ids=())
    return GenerationRuntimeConfig(machine=machine, run=run)


def _free_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def run_stage(
    base: GenerationRuntimeConfig,
    probe: CapacityProbeConfig,
    stage,
    *,
    output_root: Path,
    approved_manifest: Path,
    code_revision: str,
    deadline: float,
) -> dict:
    root = output_root / stage.name
    if root.exists():
        raise ValueError(f"capacity stage root already exists: {root}")
    root.mkdir(parents=True)
    runtime = _stage_runtime(base, probe, stage, root)
    runtime_path = root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime.as_dict(), indent=2) + "\n", encoding="utf-8")
    approved_bytes = approved_manifest.read_bytes()
    approved_rows = json.loads(approved_bytes)["catalog"]
    run = RunConfig(
        run_id=runtime.run.run_id,
        run_root=str(root),
        code_revision=code_revision,
        approved_manifest_sha256=hashlib.sha256(approved_bytes).hexdigest(),
        worker_count=stage.worker_count,
        hf_subfolder=probe.hf_subfolder,
        validation_mode=True,
        validation_max_jobs=stage.max_jobs,
    )
    queue = FilesystemJobQueue(root / "queue")
    planner = DistributedPlanner.from_manifest(
        run, {row["program_id"]: row for row in approved_rows},
        {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    job_ids = enqueue_fixed_jobs(planner, queue, stage.max_jobs)
    environment = runtime.resolved_environment()
    for key in ("ICGS_HF_TOKEN_PATH", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_ACCESS_TOKEN"):
        environment.pop(key, None)
    processes = []
    minimum_free = _free_memory_bytes()
    started = time.monotonic()
    try:
        for worker_id in runtime.machine.worker_ids:
            if time.monotonic() >= deadline:
                raise TimeoutError("capacity probe global deadline reached during worker launch")
            command = [
                "xvfb-run", "--server-num", str(runtime.machine.display_base + int(worker_id)),
                "-s", f"-screen 0 {runtime.machine.display_width}x{runtime.machine.display_height}x24 +extension GLX +render -noreset",
                runtime.machine.python_executable, "-B",
                str(Path(runtime.machine.repo_root) / "scripts" / "generation_worker.py"),
                "--worker-id", worker_id, "--runtime-config", str(runtime_path),
                "--approved-manifest", str(approved_manifest), "--once",
            ]
            log = (root / f"worker-{worker_id}.log").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            processes.append(process)
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("capacity probe global deadline reached")
            available = _free_memory_bytes()
            if available is not None:
                minimum_free = available if minimum_free is None else min(minimum_free, available)
                if available < 8 * 1024**3:
                    raise RuntimeError("capacity probe stopped: less than 8 GiB memory available")
            time.sleep(1)
        returncodes = [process.returncode for process in processes]
    finally:
        stop_processes(processes)
    elapsed = time.monotonic() - started
    ready = queue.iter_ready()
    outcomes = Counter()
    artifact_bytes = 0
    invalid = []
    for result in ready:
        job_path = queue.root / "ready" / result.job_id / "job.json"
        try:
            from icgs.data.collection.generation.distributed_contracts import GenerationJob
            job = GenerationJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
            validated = validate_closed_result(job, result)
            outcomes[validated.outcome] += 1
            artifact_bytes += sum(path.stat().st_size for path in Path(result.result_dir).rglob("*") if path.is_file())
        except Exception as error:
            invalid.append({"job_id": result.job_id, "error": f"{type(error).__name__}: {error}"})
    counts = queue.counts()
    receipt = {
        "stage": stage.name,
        "worker_count": stage.worker_count,
        "simulator_slots": stage.simulator_slots,
        "job_cap": stage.max_jobs,
        "enqueued": len(job_ids),
        "elapsed_s": elapsed,
        "attempts_per_hour": round(len(ready) * 3600 / elapsed, 2) if elapsed else 0,
        "outcomes": dict(outcomes),
        "invalid_results": invalid,
        "artifact_bytes": artifact_bytes,
        "minimum_available_memory_bytes": minimum_free,
        "worker_returncodes": returncodes,
        "queue": counts.__dict__,
        "status": "PASS" if len(ready) == len(job_ids) and not invalid and all(code == 0 for code in returncodes) else "FAIL",
    }
    (root / "capacity_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-config")
    parser.add_argument("--approved-manifest")
    parser.add_argument("--code-revision")
    parser.add_argument("--output-root")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = CapacityProbeConfig.from_file(args.config)
    if not args.execute:
        print(json.dumps({"status": "PASS", **config.as_dict()}, indent=2, sort_keys=True))
        return 0
    if not all((args.runtime_config, args.approved_manifest, args.code_revision, args.output_root)):
        parser.error("--execute requires --runtime-config, --approved-manifest, --code-revision and --output-root")
    base = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    base.machine.validate_paths()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise ValueError(f"capacity output root already exists: {output_root}")
    if base.run.publication_enabled or not base.run.validation_mode:
        raise ValueError("capacity runtime must be validation_mode=true and publication_enabled=false")
    if base.run.hf_subfolder != config.hf_subfolder:
        raise ValueError("capacity runtime HF prefix does not match probe config")
    output_root.mkdir(parents=True)
    deadline = time.monotonic() + config.max_runtime_s
    receipts = []
    try:
        for stage in config.stages:
            receipt = run_stage(
                base, config, stage, output_root=output_root,
                approved_manifest=Path(args.approved_manifest).resolve(),
                code_revision=args.code_revision, deadline=deadline,
            )
            receipts.append(receipt)
            (output_root / "capacity_summary.json").write_text(
                json.dumps({"status": "IN_PROGRESS", "stages": receipts}, indent=2) + "\n",
                encoding="utf-8",
            )
            if receipt["status"] != "PASS":
                break
    except Exception as error:
        receipt = {"status": "FAIL", "error": f"{type(error).__name__}: {error}", "stages": receipts}
        (output_root / "capacity_summary.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
        )
        raise
    status = "PASS" if len(receipts) == len(config.stages) and all(row["status"] == "PASS" for row in receipts) else "FAIL"
    (output_root / "capacity_summary.json").write_text(
        json.dumps({"status": status, "stages": receipts}, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"status": status, "stages": receipts}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
