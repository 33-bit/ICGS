"""Preflight and launch the configured generation workers and coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import hf_hub_download

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
from scripts.generation_coordinator import _validate_resume_manifest


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
    *,
    runtime_config_path: str | Path | None = None,
) -> list[list[str]]:
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    machine = config.machine
    worker_ids = config.machine.worker_ids
    runtime_config_path = Path(runtime_config_path) if runtime_config_path is not None else (
        Path(config.run.run_root) / "control" / "runtime_config.json"
    )
    worker_script = Path(machine.repo_root) / "scripts" / "generation_worker.py"
    commands = []
    for worker_id in worker_ids:
        commands.append([
            "xvfb-run",
            "--server-num", str(machine.display_base + int(worker_id)),
            "-s", (
                f"-screen 0 {machine.display_width}x{machine.display_height}x24 "
                "+extension GLX +render -noreset"
            ),
            machine.python_executable,
            "-B",
            str(worker_script),
            "--worker-id", worker_id,
            "--host-id", config.machine.host_id,
            "--runtime-config", str(runtime_config_path),
            "--approved-manifest", str(approved_manifest),
        ])
    return commands


def build_process_environment(
    config: GenerationRuntimeConfig,
    base: dict[str, str] | None = None,
    *,
    include_hf_credential: bool = False,
) -> dict[str, str]:
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    environment = config.resolved_environment(base=base)
    if not include_hf_credential:
        for key in (
            "ICGS_HF_TOKEN_PATH",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HF_ACCESS_TOKEN",
        ):
            environment.pop(key, None)
    return environment


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


def persist_runtime_snapshot(config: GenerationRuntimeConfig, path: str | Path) -> tuple[Path, str]:
    """Write a host-local immutable runtime snapshot and return its digest."""
    payload = (json.dumps(config.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    target = Path(path)
    _atomic_write(target, payload)
    return target, hashlib.sha256(payload).hexdigest()


def _resume_credential_path(token_path: str | Path | None) -> Path:
    configured = token_path or os.environ.get("ICGS_HF_TOKEN_PATH")
    if not configured:
        raise ValueError("resume_from_hf requires --hf-token-path or ICGS_HF_TOKEN_PATH")
    path = Path(configured)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"HF credential path must be a regular file: {path}")
    return path


def preflight_resume_manifest(
    config: GenerationRuntimeConfig,
    *,
    token_path: str | Path | None,
    run_root: str | Path,
    approved_manifest: str | Path,
    downloader=None,
) -> dict[str, object] | None:
    """Fetch and validate the immutable HF manifest before child processes start."""
    if not config.run.resume_from_hf:
        return None
    downloader = hf_hub_download if downloader is None else downloader
    credential = _resume_credential_path(token_path)
    token = credential.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("resume_from_hf credential file is empty")
    filename = f"{config.run.hf_subfolder}/dataset_manifest.json"
    try:
        remote_path = downloader(
            repo_id=config.run.hf_repo,
            repo_type="dataset",
            filename=filename,
            token=token,
            force_download=True,
        )
    except Exception as error:
        raise RuntimeError("resume_from_hf requires a readable remote dataset manifest") from error
    remote_bytes = Path(remote_path).read_bytes()
    try:
        approved_payload = json.loads(Path(approved_manifest).read_text(encoding="utf-8"))
        catalog = approved_payload.get("catalog")
        if not isinstance(catalog, list):
            raise ValueError("approved manifest catalog must be a list")
        approved_program_ids = {
            str(row["program_id"])
            for row in catalog
            if isinstance(row, dict) and isinstance(row.get("program_id"), str)
        }
        manifest = _validate_resume_manifest(json.loads(remote_bytes), RunConfig(
            run_id=config.run.run_id,
            run_root=config.run.run_root,
            code_revision="a" * 40,
            approved_manifest_sha256="b" * 64,
            worker_count=config.run.worker_count,
            publish_interval_s=config.run.publish_interval_s,
            hf_repo=config.run.hf_repo or "33bit/icgs",
            hf_subfolder=config.run.hf_subfolder or "generation",
            publication_enabled=config.run.publication_enabled,
            validation_mode=config.run.validation_mode,
            distribution_mode=config.run.distribution_mode,
            resume_from_hf=config.run.resume_from_hf,
        ), approved_program_ids=approved_program_ids)
    except Exception as error:
        raise RuntimeError("remote resume manifest failed immutable validation") from error
    parts = Path(remote_path).parts
    revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    bootstrap = {
        "run_id": config.run.run_id,
        "remote_manifest_sha256": hashlib.sha256(remote_bytes).hexdigest(),
        "remote_revision": revision,
        "source_run_ids": list(manifest.get("source_run_ids") or ()),
        "fetched_at_s": time.time(),
    }
    bootstrap_path = Path(run_root) / "control" / "resume_bootstrap.json"
    _atomic_write(
        bootstrap_path,
        (json.dumps(bootstrap, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return bootstrap


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"approved manifest must be a regular file: {manifest_path}")
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
        distribution_mode=config.run.distribution_mode,
        resume_from_hf=config.run.resume_from_hf,
    )
    run_payload = {
        "run": run.as_dict(),
        "approved_manifest": str(manifest_path),
        "manifest": {"manifest_version": 3, "episodes": [], "failure_attempts": []},
        "runtime_config": runtime_payload,
        "runtime_config_path": str(runtime_path),
        "runtime_config_sha256": runtime_digest,
        "coordinator_worker_ids": list(config.machine.worker_ids),
    }
    run_path = control_root / "run.json"
    serialized_run = (json.dumps(run_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(run_path, serialized_run)
    return run_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--code-revision")
    parser.add_argument("--smoke-receipt")
    parser.add_argument("--validation-plan")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--workers-only", action="store_true")
    parser.add_argument("--run-config")
    parser.add_argument("--hf-token-path")
    args = parser.parse_args(argv)

    unchecked = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    if args.workers_only:
        if not args.run_config:
            raise ValueError("--workers-only requires --run-config")
        config = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=True)
        existing_run = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
        if existing_run.get("run", {}).get("run_root") != config.run.run_root:
            raise ValueError("workers-only runtime config run_root does not match run config")
        existing_run_id = existing_run.get("run", {}).get("run_id")
        if existing_run_id is not None and existing_run_id != config.run.run_id:
            raise ValueError("workers-only runtime config run_id does not match run config")
        if config.run.distribution_mode != "shared_filesystem":
            raise ValueError("workers-only requires distribution_mode=shared_filesystem")
        coordinator_worker_ids = existing_run.get("coordinator_worker_ids")
        if not isinstance(coordinator_worker_ids, list):
            raise ValueError("run config must record coordinator_worker_ids for workers-only attach")
        if set(coordinator_worker_ids) & set(config.machine.worker_ids):
            raise ValueError("workers-only worker_ids overlap coordinator worker scope")
        approved_path = Path(args.approved_manifest)
        if not approved_path.is_file() or approved_path.is_symlink():
            raise ValueError("approved manifest must be a regular file")
        expected_manifest_sha = existing_run.get("run", {}).get("approved_manifest_sha256")
        actual_manifest_sha = hashlib.sha256(approved_path.read_bytes()).hexdigest()
        if expected_manifest_sha is not None and expected_manifest_sha != actual_manifest_sha:
            raise ValueError("approved manifest digest does not match run config")
        root = Path(config.run.run_root)
        host_runtime_path, runtime_digest = persist_runtime_snapshot(
            config,
            root / "control" / f"runtime_config-{config.machine.host_id}.json",
        )
        receipt_path = root / "control" / f"worker-launch-{config.machine.host_id}.json"
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            previous_pids = [int(pid) for pid in dict(previous.get("worker_pids") or {}).values()]
            if any(_pid_is_alive(pid) for pid in previous_pids):
                raise RuntimeError(f"worker host already has an active launch: {config.machine.host_id}")
        commands = build_worker_commands(
            config,
            args.approved_manifest,
            runtime_config_path=host_runtime_path,
        )
        environment = build_process_environment(config)
        environment = build_process_environment(config)
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
        receipt = {
            "host_id": config.machine.host_id,
            "run_id": config.run.run_id,
            "runtime_config_path": str(host_runtime_path),
            "runtime_config_sha256": runtime_digest,
            "approved_manifest": str(approved_path),
            "approved_manifest_sha256": actual_manifest_sha,
            "workers": len(worker_pids),
            "worker_ids": list(worker_pids),
            "worker_pids": worker_pids,
            "launched_at_s": time.time(),
        }
        temporary = receipt_path.with_name(receipt_path.name + f".partial-{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
        watchdog = [
            config.machine.python_executable,
            "-B",
            str(Path(config.machine.repo_root) / "scripts" / "generation_watchdog.py"),
            "--run-root", str(root),
            "--workers-only",
            "--host-id", config.machine.host_id,
        ]
        subprocess.Popen(
            watchdog,
            env=environment,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if not args.code_revision or not args.smoke_receipt:
        raise ValueError("normal launch requires --code-revision and --smoke-receipt")
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
    preflight_resume_manifest(
        config,
        token_path=args.hf_token_path,
        run_root=root,
        approved_manifest=args.approved_manifest,
    )
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
    coordinator_environment = build_process_environment(config, include_hf_credential=True)
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
        "--runtime-config", str(Path(config.run.run_root) / "control" / "runtime_config.json"),
    ]
    coordinator_token_path = args.hf_token_path or os.environ.get("ICGS_HF_TOKEN_PATH")
    if coordinator_token_path:
        coordinator.extend(["--hf-token-path", coordinator_token_path])
    coordinator_log = root / "control" / "coordinator.log"
    coordinator_stream = coordinator_log.open("a", encoding="utf-8")
    coordinator_process = subprocess.Popen(
        coordinator,
        env=coordinator_environment,
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
        "host_id": config.machine.host_id,
        "workers": len(worker_pids),
        "worker_ids": list(worker_pids),
        "worker_pids": worker_pids,
        "coordinator_pid": coordinator_process.pid,
        "watchdog_pid": watchdog_process.pid,
        "hf_token_path": coordinator_token_path,
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
