"""Preflight and launch the configured generation workers and coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from icgs.data.collection.generation.distributed_contracts import (
    GenerationRuntimeConfig,
    RunConfig,
)
from icgs.data.collection.generation.distributed_validation import (
    build_validation_receipt,
    load_validation_plan,
    write_validation_receipt,
)
from icgs.data.collection.generation.steps import GENERATION_PROGRAMS


def validate_smoke_receipt(path: str | Path, *, expected_program_ids) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("smoke receipt must contain results")
    expected = {str(program_id) for program_id in expected_program_ids}
    seen = [str(row.get("program_id")) for row in results if isinstance(row, dict)]
    if len(seen) != len(expected) or set(seen) != expected:
        raise ValueError("smoke receipt must contain each expected program exactly once")
    blocking = [
        str(row.get("program_id"))
        for row in results
        if row.get("result_class") not in {"success", "valid_failure"}
        or row.get("timeline_ok") is not True
    ]
    if blocking:
        raise ValueError("blocking smoke outcomes: " + ", ".join(sorted(blocking)))


def build_worker_commands(
    config: GenerationRuntimeConfig,
    approved_manifest: str | Path,
) -> list[list[str]]:
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    machine = config.machine
    worker_count = config.run.worker_count
    worker_id_width = max(3, len(str(worker_count - 1)))
    runtime_config_path = Path(config.run.run_root) / "control" / "runtime_config.json"
    worker_script = Path(machine.repo_root) / "scripts" / "generation_worker.py"
    commands = []
    for index in range(worker_count):
        worker_id = f"{index:0{worker_id_width}d}"
        commands.append([
            "xvfb-run",
            "--server-num", str(machine.display_base + index),
            "-s", f"-screen 0 {machine.display_width}x{machine.display_height}x24",
            machine.python_executable,
            "-B",
            str(worker_script),
            "--worker-id", worker_id,
            "--runtime-config", str(runtime_config_path),
            "--approved-manifest", str(approved_manifest),
        ])
    return commands


def build_process_environment(
    config: GenerationRuntimeConfig,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    return config.resolved_environment(base=base)


def persist_validation_receipt(
    validation_plan_path: str | Path,
    receipt_path: str | Path,
    *,
    runtime: dict | None = None,
) -> "ValidationReceipt":
    """Create the explicit NOT_RUN receipt for a JSON validation plan."""
    plan = load_validation_plan(validation_plan_path)
    receipt = build_validation_receipt(plan, runtime=runtime)
    write_validation_receipt(receipt_path, receipt, plan)
    return receipt


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def persist_run_config(
    config: GenerationRuntimeConfig,
    approved_manifest: str | Path,
    *,
    code_revision: str,
    validation_max_jobs: int | None = None,
) -> Path:
    """Persist the worker-readable config and its exact digest before launch."""
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    control_root = Path(config.run.run_root) / "control"
    runtime_payload = config.as_dict()
    runtime_bytes = (
        json.dumps(runtime_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
    runtime_path = control_root / "runtime_config.json"
    _atomic_write(runtime_path, runtime_bytes)

    manifest_path = Path(approved_manifest)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    hf_repo = config.run.hf_repo or "33bit/icgs"
    hf_subfolder = config.run.hf_subfolder or "generation"
    run = RunConfig(
        run_id=config.run.run_id,
        run_root=config.run.run_root,
        code_revision=code_revision,
        approved_manifest_sha256=manifest_digest,
        worker_count=config.run.worker_count,
        publish_interval_s=config.run.publish_interval_s,
        hf_repo=hf_repo,
        hf_subfolder=hf_subfolder,
        publication_enabled=config.run.publication_enabled,
        validation_mode=config.run.validation_mode,
        validation_max_jobs=validation_max_jobs,
    )
    run_payload = {
        "run": run.as_dict(),
        "approved_manifest": str(manifest_path),
        "manifest": {"manifest_version": 3, "episodes": [], "failure_attempts": []},
        "runtime_config": runtime_payload,
        "runtime_config_path": str(runtime_path),
        "runtime_config_sha256": runtime_digest,
    }
    run_path = control_root / "run.json"
    serialized_run = (json.dumps(run_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(run_path, serialized_run)
    return run_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--smoke-receipt", required=True)
    parser.add_argument("--validation-plan")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)

    unchecked = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    Path(unchecked.run.run_root).mkdir(parents=True, exist_ok=True)
    config = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=True)
    if not config.run.publication_enabled:
        raise ValueError("full launch requires publication_enabled in runtime config")
    validate_smoke_receipt(args.smoke_receipt, expected_program_ids=GENERATION_PROGRAMS)

    plan = None
    if args.validation_plan:
        if not config.run.validation_mode:
            raise ValueError("--validation-plan requires validation_mode=true")
        plan = load_validation_plan(
            args.validation_plan,
            validation_mode=config.run.validation_mode,
        )
        if config.run.worker_count > plan.worker_count:
            raise ValueError("runtime config worker_count exceeds validation plan bound")

    run_config_path = persist_run_config(
        config,
        args.approved_manifest,
        code_revision=args.code_revision,
        validation_max_jobs=plan.max_jobs if plan is not None else None,
    )
    root = Path(config.run.run_root)
    if plan is not None:
        persist_validation_receipt(
            args.validation_plan,
            root / "control" / "validation_receipt.json",
            runtime={
                "worker_count": config.run.worker_count,
                "max_jobs": plan.max_jobs,
                "max_episodes": plan.max_episodes,
                "max_attempts": plan.max_attempts,
                "hf_subfolder": config.run.hf_subfolder,
            },
        )
    commands = build_worker_commands(config, args.approved_manifest)
    environment = build_process_environment(config)
    processes = []
    worker_pids = {}
    for command in commands:
        process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        worker_id = command[command.index("--worker-id") + 1]
        worker_pids[worker_id] = process.pid
        processes.append(process)

    machine = config.machine
    coordinator = [
        machine.python_executable,
        "-B",
        str(Path(machine.repo_root) / "scripts" / "generation_coordinator.py"),
        "--run-config", str(run_config_path),
    ]
    coordinator_log = root / "control" / "coordinator.log"
    coordinator_stream = coordinator_log.open("a", encoding="utf-8")
    coordinator_process = subprocess.Popen(
        coordinator,
        env=environment,
        start_new_session=True,
        stdout=coordinator_stream,
        stderr=subprocess.STDOUT,
    )
    processes.append(coordinator_process)

    watchdog = [
        machine.python_executable,
        "-B",
        str(Path(machine.repo_root) / "scripts" / "generation_watchdog.py"),
        "--run-root", str(root),
    ]
    watchdog_process = subprocess.Popen(
        watchdog,
        env=environment,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    processes.append(watchdog_process)

    launch_receipt = {
        "workers": config.run.worker_count,
        "worker_pids": worker_pids,
        "coordinator_pid": coordinator_process.pid,
        "watchdog_pid": watchdog_process.pid,
        "restart_counts": {
            "coordinator": 0,
            "workers": {worker_id: 0 for worker_id in worker_pids},
        },
        "launched_at_s": time.time(),
    }
    receipt_path = root / "control" / "launch.json"
    temporary = receipt_path.with_name(receipt_path.name + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(json.dumps(launch_receipt, sort_keys=True))
    if args.detach:
        return 0
    return coordinator_process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
