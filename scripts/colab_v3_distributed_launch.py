"""Preflight and launch exactly 200 v3 workers plus one coordinator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


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
    parser.add_argument("--publication-enabled", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.publish_interval_s != 300:
        raise ValueError("publish interval must be 300 seconds")
    if not args.publication_enabled:
        raise ValueError("full launch requires --publication-enabled")
    if not Path("/content/.icgs_hf_token").is_file():
        raise RuntimeError("/content/.icgs_hf_token is required for coordinator")
    root = Path(args.run_root)
    (root / "control").mkdir(parents=True, exist_ok=True)
    commands = build_worker_commands(workers=args.workers, run_root=str(root), approved_manifest=args.approved_manifest)
    env = os.environ.copy()
    env.update({"ICGS_HF_REPO": args.hf_repo, "ICGS_HF_SUBFOLDER": args.hf_subfolder})
    coordinator = [
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/colab_v3_distributed_coordinator.py",
        "--run-config", str(root / "control" / "run.json"),
    ]
    import hashlib
    approved_digest = hashlib.sha256(Path(args.approved_manifest).read_bytes()).hexdigest()
    run_id = root.name
    (root / "control" / "run.json").write_text(json.dumps({
        "run": {
            "run_id": run_id, "run_root": str(root),
            "code_revision": "unknown", "approved_manifest_sha256": approved_digest,
            "worker_count": 200, "publish_interval_s": 300,
            "hf_repo": args.hf_repo, "hf_subfolder": args.hf_subfolder,
        },
        "approved_manifest": args.approved_manifest,
        "manifest": {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    }, indent=2) + "\n", encoding="utf-8")
    processes = []
    for command in commands:
        processes.append(subprocess.Popen(command, env=env, start_new_session=True))
    processes.append(subprocess.Popen(coordinator, env=env, start_new_session=True))
    watchdog = [
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/colab_v3_distributed_watchdog.py",
        "--run-root", str(root),
    ]
    processes.append(subprocess.Popen(watchdog, env=env, start_new_session=True))
    (root / "control" / "launch.json").write_text(
        json.dumps({"workers": 200, "coordinator_pid": processes[-2].pid, "watchdog_pid": processes[-1].pid}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"workers": 200, "coordinator_pid": processes[-2].pid, "watchdog_pid": processes[-1].pid}))
    if args.detach:
        return 0
    return processes[-2].wait()


if __name__ == "__main__":
    import json
    raise SystemExit(main())
