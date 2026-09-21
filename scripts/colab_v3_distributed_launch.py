"""Preflight and launch exactly 200 v3 workers plus one coordinator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from icgs.data.collection.v3.distributed_contracts import RunConfig
from icgs.data.collection.v3.steps import V3_PROGRAMS


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


def build_worker_commands(*, workers: int, run_root: str, approved_manifest: str) -> list[list[str]]:
    if workers != 200:
        raise ValueError("workers must be 200")
    commands = []
    for index in range(workers):
        worker_id = f"{index:03d}"
        commands.append([
            "xvfb-run", "--server-num", str(200 + index),
            "-s", "-screen 0 1280x1024x24",
            "/content/icgs-data-env/bin/python", "-B",
            "/content/ICGS/scripts/colab_v3_distributed_worker.py",
            "--worker-id", worker_id, "--run-root", run_root,
            "--approved-manifest", approved_manifest,
        ])
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--workers", type=int, default=200)
    parser.add_argument("--publish-interval-s", type=int, default=300)
    parser.add_argument("--hf-repo", default="33bit/icgs")
    parser.add_argument("--hf-subfolder", default="primary_v3")
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--smoke-receipt", required=True)
    parser.add_argument("--publication-enabled", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.publish_interval_s != 300:
        raise ValueError("publish interval must be 300 seconds")
    if not args.publication_enabled:
        raise ValueError("full launch requires --publication-enabled")
    if not Path("/content/.icgs_hf_token").is_file():
        raise RuntimeError("/content/.icgs_hf_token is required for coordinator")
    validate_smoke_receipt(args.smoke_receipt, expected_program_ids=V3_PROGRAMS)
    root = Path(args.run_root)
    (root / "control").mkdir(parents=True, exist_ok=True)
    commands = build_worker_commands(workers=args.workers, run_root=str(root), approved_manifest=args.approved_manifest)
    env = os.environ.copy()
    simulator_root = "/content/icgs-ephemeral/CoppeliaSim"
    rlbench_root = "/content/icgs-ephemeral/RLBench"
    env.update({
        "ICGS_HF_REPO": args.hf_repo,
        "ICGS_HF_SUBFOLDER": args.hf_subfolder,
        "COPPELIASIM_ROOT": simulator_root,
        "LD_LIBRARY_PATH": simulator_root + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "QT_QPA_PLATFORM_PLUGIN_PATH": simulator_root,
        "PYTHONPATH": ":".join(filter(None, ["/content/ICGS/src", rlbench_root, env.get("PYTHONPATH")])),
    })
    coordinator = [
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/colab_v3_distributed_coordinator.py",
        "--run-config", str(root / "control" / "run.json"),
    ]
    import hashlib
    approved_digest = hashlib.sha256(Path(args.approved_manifest).read_bytes()).hexdigest()
    run_id = root.name
    run_config = RunConfig(
        run_id=run_id,
        run_root=str(root),
        code_revision=args.code_revision,
        approved_manifest_sha256=approved_digest,
        worker_count=args.workers,
        publish_interval_s=args.publish_interval_s,
        hf_repo=args.hf_repo,
        hf_subfolder=args.hf_subfolder,
    )
    (root / "control" / "run.json").write_text(json.dumps({
        "run": run_config.as_dict(),
        "approved_manifest": args.approved_manifest,
        "manifest": {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    }, indent=2) + "\n", encoding="utf-8")
    processes = []
    worker_pids = {}
    for command in commands:
        process = subprocess.Popen(
            command, env=env, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        worker_pids[command[command.index("--worker-id") + 1]] = process.pid
        processes.append(process)
    coordinator_log = root / "control" / "coordinator.log"
    coordinator_stream = coordinator_log.open("a", encoding="utf-8")
    coordinator_process = subprocess.Popen(
        coordinator, env=env, start_new_session=True,
        stdout=coordinator_stream, stderr=subprocess.STDOUT,
    )
    processes.append(coordinator_process)
    watchdog = [
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/colab_v3_distributed_watchdog.py",
        "--run-root", str(root),
    ]
    watchdog_process = subprocess.Popen(
        watchdog, env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    processes.append(watchdog_process)
    launch_receipt = {
        "workers": 200,
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
