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
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.generation.capacity_probe import (
    CapacityProbeConfig,
    extract_result_metrics,
    measure_staging_tree_bytes,
    preflight_stage_capacity,
)
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
    active = [process for process in processes if process is not None and process.poll() is None]
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
    machine = replace(
        base.machine,
        simulator_slots=stage.simulator_slots,
        worker_timeout_s=stage.worker_timeout_s,
        worker_ids=(),
    )
    return GenerationRuntimeConfig(
        machine=machine,
        run=run,
        archive_profile=base.archive_profile,
        max_result_bytes=(
            stage.max_result_bytes if stage.max_result_bytes is not None
            else probe.max_result_bytes
        ),
    )


def _free_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _trip_probe_safety_stop(
    queue: FilesystemJobQueue,
    *,
    run_id: str,
    stage_name: str,
    stage_peak_bytes: int | None,
    total_peak_bytes: int | None,
    measurement_errors: list[str],
    capacity_violations: list[dict],
) -> None:
    receipt = {
        "schema_version": "icgs_generation_safety_stop_v1",
        "run_id": run_id,
        "stage": stage_name,
        "reason": "capacity_probe_staging_failure",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_peak_bytes_sampled_lower_bound": stage_peak_bytes,
        "total_peak_bytes_sampled_lower_bound": total_peak_bytes,
        "measurement_errors": measurement_errors,
        "capacity_violations": capacity_violations,
        "recovery_action": (
            "Stop this run and preserve its queue, staging, spool, partial, and final bytes. "
            "Inspect the measured cap breach or staging measurement failure and every claimed "
            "job before recovery. Reconcile archived bytes against pinned publication receipts "
            "and hashes; resume only through a separately reviewed recovery path."
        ),
    }
    try:
        queue.trip_safety_stop(receipt)
    except Exception as error:
        raise RuntimeError(
            f"fatal capacity probe FAIL: {type(error).__name__}: {error}; "
            f"could not write generation safety stop at {queue.safety_stop_path}"
        ) from error


def wait_for_ready_results(
    queue,
    processes,
    *,
    expected_jobs: int,
    deadline: float,
    stage_root: Path | None = None,
    stage_cap_bytes: int | None = None,
    total_root: Path | None = None,
    total_cap_bytes: int | None = None,
    poll_interval_s: float = 1.0,
    stage_name: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Wait for results while sampling visible staging bytes as a lower bound."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    minimum_free = _free_memory_bytes()
    stage_peak = 0
    total_peak = 0
    stage_writer_scratch_peak = 0

    def sample_tree(
        root: Path | None,
        previous_peak: int,
        previous_scratch_peak: int = 0,
    ) -> tuple[int, int, list[str]]:
        if root is None:
            return previous_peak, previous_scratch_peak, []
        measured = measure_staging_tree_bytes(root)
        value = int(measured["bytes"])
        scratch = int(measured["writer_scratch_bytes"])
        return max(previous_peak, value), max(previous_scratch_peak, scratch), list(measured["errors"])

    while True:
        stage_peak, stage_writer_scratch_peak, stage_errors = sample_tree(
            stage_root, stage_peak, stage_writer_scratch_peak,
        )
        total_peak, _total_scratch_peak, total_errors = sample_tree(total_root, total_peak)
        measurement_errors = stage_errors + total_errors
        capacity_violations = []
        if stage_cap_bytes is not None and stage_peak > stage_cap_bytes:
            capacity_violations.append({
                "scope": "stage",
                "measured_bytes": stage_peak,
                "cap_bytes": stage_cap_bytes,
                "semantic": "sampled_lower_bound",
            })
        if total_cap_bytes is not None and total_peak > total_cap_bytes:
            capacity_violations.append({
                "scope": "total",
                "measured_bytes": total_peak,
                "cap_bytes": total_cap_bytes,
                "semantic": "sampled_lower_bound",
            })
        if measurement_errors or capacity_violations:
            try:
                _trip_probe_safety_stop(
                    queue,
                    run_id=run_id or stage_name or "unknown",
                    stage_name=stage_name or (stage_root.name if stage_root is not None else "unknown"),
                    stage_peak_bytes=stage_peak if stage_root is not None else None,
                    total_peak_bytes=total_peak if total_root is not None else None,
                    measurement_errors=measurement_errors,
                    capacity_violations=capacity_violations,
                )
            except Exception as marker_error:
                try:
                    stop_processes(processes)
                except Exception as cleanup_error:
                    raise RuntimeError(
                        f"{marker_error}; process cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    ) from marker_error
                raise
            stop_processes(processes)
            counts = queue.counts()
            return {
                "ready": counts.ready,
                "active_workers_at_completion": sum(
                    process.poll() is None for process in processes if process is not None
                ),
                "minimum_available_memory_bytes": minimum_free,
                "stage_peak_bytes_sampled_lower_bound": stage_peak if stage_root is not None else None,
                "stage_writer_scratch_peak_bytes_sampled_lower_bound": (
                    stage_writer_scratch_peak if stage_root is not None else None
                ),
                "total_peak_bytes_sampled_lower_bound": total_peak if total_root is not None else None,
                "sample_interval_s": poll_interval_s,
                "measurement_errors": measurement_errors,
                "capacity_violations": capacity_violations,
            }
        counts = queue.counts()
        stop_path = getattr(queue, "safety_stop_path", None)
        if stop_path is not None:
            stop_path = Path(stop_path)
            if stop_path.exists() or stop_path.is_symlink():
                try:
                    if stop_path.is_symlink() or not stop_path.is_file():
                        raise ValueError("safety stop marker is not a regular file")
                    stop_payload = json.loads(stop_path.read_text(encoding="utf-8"))
                    stop_reason = stop_payload["reason"]
                    if not isinstance(stop_reason, str) or not stop_reason.strip():
                        raise ValueError("safety stop marker has no reason")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    stop_reason = f"unreadable safety stop marker: {type(error).__name__}: {error}"
                return {
                    "ready": counts.ready,
                    "active_workers_at_completion": sum(
                        process.poll() is None for process in processes if process is not None
                    ),
                    "minimum_available_memory_bytes": minimum_free,
                    "stage_peak_bytes_sampled_lower_bound": stage_peak if stage_root is not None else None,
                    "stage_writer_scratch_peak_bytes_sampled_lower_bound": (
                        stage_writer_scratch_peak if stage_root is not None else None
                    ),
                    "total_peak_bytes_sampled_lower_bound": total_peak if total_root is not None else None,
                    "sample_interval_s": poll_interval_s,
                    "measurement_errors": [],
                    "capacity_violations": [],
                    "safety_stop": {"path": str(stop_path), "reason": stop_reason},
                }
        if counts.ready == expected_jobs:
            return {
                "ready": counts.ready,
                "active_workers_at_completion": sum(process.poll() is None for process in processes),
                "minimum_available_memory_bytes": minimum_free,
                "stage_peak_bytes_sampled_lower_bound": stage_peak if stage_root is not None else None,
                "stage_writer_scratch_peak_bytes_sampled_lower_bound": (
                    stage_writer_scratch_peak if stage_root is not None else None
                ),
                "total_peak_bytes_sampled_lower_bound": total_peak if total_root is not None else None,
                "sample_interval_s": poll_interval_s,
                "measurement_errors": [],
                "capacity_violations": [],
            }
        if counts.ready > expected_jobs:
            raise RuntimeError("capacity queue exceeded fixed job cap")
        if time.monotonic() >= deadline:
            raise TimeoutError("capacity probe global deadline reached")
        available = _free_memory_bytes()
        if available is not None:
            minimum_free = available if minimum_free is None else min(minimum_free, available)
            if available < 8 * 1024**3:
                raise RuntimeError("capacity probe stopped: less than 8 GiB memory available")
        if all(process.poll() is not None for process in processes if process is not None):
            raise RuntimeError("all workers exited before all results became ready")
        time.sleep(poll_interval_s)


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
    preflight_stage_capacity(
        stage,
        probe,
        root,
        output_root=output_root,
        archive_profile=base.archive_profile.as_dict() if base.archive_profile is not None else None,
    )
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
    effective_max_result_bytes = (
        stage.max_result_bytes
        if stage.max_result_bytes is not None
        else probe.max_result_bytes
    )
    effective_max_staging_bytes = (
        stage.max_staging_bytes
        if stage.max_staging_bytes is not None
        else probe.max_total_staging_bytes
    )
    environment = runtime.resolved_environment()
    for key in ("ICGS_HF_TOKEN_PATH", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_ACCESS_TOKEN"):
        environment.pop(key, None)
    processes = []
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
        completion = wait_for_ready_results(
            queue, processes, expected_jobs=len(job_ids), deadline=deadline,
            stage_root=root,
            stage_cap_bytes=effective_max_staging_bytes,
            total_root=output_root,
            total_cap_bytes=probe.max_total_staging_bytes,
            stage_name=stage.name,
            run_id=runtime.run.run_id,
        )
        results_elapsed = time.monotonic() - started
    finally:
        pending_error = sys.exc_info()[1]
        try:
            stop_processes(processes)
        except Exception as cleanup_error:
            if pending_error is None:
                raise
            raise RuntimeError(
                f"{pending_error}; process cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            ) from pending_error
    cleanup_elapsed = time.monotonic() - started - results_elapsed
    returncodes = [process.poll() for process in processes]
    final_stage = measure_staging_tree_bytes(root)
    final_total = measure_staging_tree_bytes(output_root)
    stage_peak_bytes = max(
        int(completion.get("stage_peak_bytes_sampled_lower_bound") or 0),
        int(final_stage["bytes"]),
    )
    stage_writer_scratch_peak_bytes = max(
        int(completion.get("stage_writer_scratch_peak_bytes_sampled_lower_bound") or 0),
        int(final_stage["writer_scratch_bytes"]),
    )
    total_peak_bytes = max(
        int(completion.get("total_peak_bytes_sampled_lower_bound") or 0),
        int(final_total["bytes"]),
    )
    ready = queue.iter_ready()
    outcomes = Counter()
    artifact_bytes = 0
    invalid = []
    staging_measurement_errors = (
        list(completion.get("measurement_errors", []))
        + list(final_stage["errors"])
        + list(final_total["errors"])
    )
    for error in staging_measurement_errors:
        invalid.append({"stage": stage.name, "error": f"StagingMeasurementFailed: {error}"})
    staging_capacity_violations = list(completion.get("capacity_violations", []))
    safety_stop = completion.get("safety_stop")
    if safety_stop is not None:
        invalid.append({
            "stage": stage.name,
            "error": f"GenerationSafetyStop: {safety_stop['reason']}; inspect {safety_stop['path']}",
        })
    for violation in staging_capacity_violations:
        invalid.append({
            "stage": stage.name,
            "error": (
                f"Sampled{violation['scope'].title()}StagingCapExceeded: observed at least "
                f"{violation['measured_bytes']} bytes, exceeding {violation['cap_bytes']} bytes"
            ),
            "measured_bytes": violation["measured_bytes"],
            "cap_bytes": violation["cap_bytes"],
            "semantic": "sampled_lower_bound",
        })
    final_violations = []
    if effective_max_staging_bytes is not None and stage_peak_bytes > effective_max_staging_bytes:
        final_violations.append(("stage", stage_peak_bytes, effective_max_staging_bytes))
    if probe.max_total_staging_bytes is not None and total_peak_bytes > probe.max_total_staging_bytes:
        final_violations.append(("total", total_peak_bytes, probe.max_total_staging_bytes))
    reported_scopes = {item.get("scope") for item in completion.get("capacity_violations", [])}
    new_final_violations = []
    for scope, measured_bytes, cap_bytes in final_violations:
        if scope not in reported_scopes:
            violation = {
                "scope": scope,
                "measured_bytes": measured_bytes,
                "cap_bytes": cap_bytes,
                "semantic": "sampled_lower_bound",
            }
            staging_capacity_violations.append(violation)
            new_final_violations.append(violation)
            invalid.append({
                "stage": stage.name,
                "error": (
                    f"Sampled{scope.title()}StagingCapExceeded: observed at least "
                    f"{measured_bytes} bytes, exceeding {cap_bytes} bytes"
                ),
                "measured_bytes": measured_bytes,
                "cap_bytes": cap_bytes,
                "semantic": "sampled_lower_bound",
            })
    new_final_errors = [
        error for error in list(final_stage["errors"]) + list(final_total["errors"])
        if error not in completion.get("measurement_errors", [])
    ]
    if new_final_errors or new_final_violations:
        _trip_probe_safety_stop(
            queue,
            run_id=runtime.run.run_id,
            stage_name=stage.name,
            stage_peak_bytes=stage_peak_bytes,
            total_peak_bytes=total_peak_bytes,
            measurement_errors=staging_measurement_errors,
            capacity_violations=staging_capacity_violations,
        )
    stage_bytes_by_category: dict[str, int] = {
        "chunks": 0,
        "manifests": 0,
        "debug": 0,
        "views": 0,
        "receipts": 0,
    }
    per_results: list[dict[str, Any]] = []
    for result in ready:
        job_path = queue.root / "ready" / result.job_id / "job.json"
        try:
            from icgs.data.collection.generation.distributed_contracts import GenerationJob
            job = GenerationJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
            validated = (
                validate_closed_result(job, result, archive_profile=runtime.archive_profile)
                if runtime.archive_profile is not None
                else validate_closed_result(job, result)
            )
            outcomes[validated.outcome] += 1
            metrics = extract_result_metrics(
                result.result_dir,
                fallback_timeline=result.timeline,
                fallback_profile=runtime.archive_profile.as_dict() if runtime.archive_profile is not None else None,
            )
            res_bytes = metrics["artifact_bytes"]
            artifact_bytes += res_bytes
            for cat, val in metrics["bytes_by_category"].items():
                stage_bytes_by_category[cat] = stage_bytes_by_category.get(cat, 0) + val

            if effective_max_result_bytes is not None and res_bytes > effective_max_result_bytes:
                invalid.append({
                    "job_id": result.job_id,
                    "error": f"ByteCapExceeded: result size {res_bytes} bytes exceeds max_result_bytes {effective_max_result_bytes}",
                })

            per_results.append({
                "job_id": result.job_id,
                "attempt_id": result.attempt_id,
                "episode_id": result.episode_id,
                "outcome": validated.outcome,
                "artifact_bytes": res_bytes,
                "bytes_by_category": metrics["bytes_by_category"],
                "boundaries": metrics["boundaries"],
                "raw_points": metrics["raw_points"],
                "archive_profile": metrics["archive_profile"],
                "writer_peak_bytes_upper_bound": metrics["writer_peak_bytes_upper_bound"],
                "writer_peak_bytes_semantic": metrics["writer_peak_bytes_semantic"],
            })
        except Exception as error:
            invalid.append({"job_id": result.job_id, "error": f"{type(error).__name__}: {error}"})

    if effective_max_staging_bytes is not None and artifact_bytes > effective_max_staging_bytes:
        invalid.append({
            "stage": stage.name,
            "error": f"StagingCapExceeded: stage artifact bytes {artifact_bytes} exceeds max_staging_bytes {effective_max_staging_bytes}",
        })

    counts = queue.counts()
    receipt = {
        "stage": stage.name,
        "worker_count": stage.worker_count,
        "simulator_slots": stage.simulator_slots,
        "worker_timeout_s": stage.worker_timeout_s,
        "job_cap": stage.max_jobs,
        "enqueued": len(job_ids),
        "elapsed_s": results_elapsed,
        "cleanup_elapsed_s": cleanup_elapsed,
        "active_workers_at_completion": completion["active_workers_at_completion"],
        "attempts_per_hour": round(len(ready) * 3600 / results_elapsed, 2) if results_elapsed else 0,
        "outcomes": dict(outcomes),
        "invalid_results": invalid,
        "artifact_bytes": artifact_bytes,
        "bytes_by_category": stage_bytes_by_category,
        "archive_profile": runtime.archive_profile.as_dict() if runtime.archive_profile is not None else None,
        "stage_peak_bytes_sampled_lower_bound": stage_peak_bytes,
        "stage_writer_scratch_peak_bytes_sampled_lower_bound": stage_writer_scratch_peak_bytes,
        "total_peak_bytes_sampled_lower_bound": total_peak_bytes,
        "staging_peak_semantic": "sampled_lower_bound",
        "staging_sample_interval_s": completion.get("sample_interval_s", 1.0),
        "staging_measurement_errors": staging_measurement_errors,
        "staging_capacity_violations": staging_capacity_violations,
        "safety_stop": safety_stop,
        "minimum_available_memory_bytes": completion["minimum_available_memory_bytes"],
        "worker_returncodes": returncodes,
        "queue": counts.__dict__,
        "results": per_results,
        "per_result": per_results,
        "status": "PASS" if len(ready) == len(job_ids) and not invalid
        and sum(outcomes.values()) == len(job_ids)
        and not (outcomes["simulator_crash"] or outcomes["invalid_observation"])
        else "FAIL",
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
    if base.archive_profile is not None:
        if config.max_total_staging_bytes is None or config.max_total_staging_bytes <= 0:
            raise ValueError("archive profile capacity probe requires explicit positive max_total_staging_bytes")
        for stage in config.stages:
            eff_r = stage.max_result_bytes if stage.max_result_bytes is not None else config.max_result_bytes
            if eff_r is None or eff_r <= 0:
                raise ValueError("archive profile capacity probe requires explicit positive max_result_bytes")
    if config.max_total_staging_bytes is not None:
        check_dir = output_root
        while not check_dir.exists() and check_dir != check_dir.parent:
            check_dir = check_dir.parent
        usage = shutil.disk_usage(check_dir)
        if usage.free < config.max_total_staging_bytes:
            raise RuntimeError(
                f"preflight capacity insufficient: required {config.max_total_staging_bytes} bytes "
                f"total staging capacity, but only {usage.free} bytes are free on {check_dir}"
            )
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
