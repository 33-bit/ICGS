"""Keepalive and bounded process replacement for one generation run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

WORKER_COUNT = 200
MAX_RESTARTS_DEFAULT = 20


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _coordinator_lock_is_free(lock_path: str | Path) -> bool:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return True


def _pid_matches(pid: int, expected_tokens: tuple[str, ...], *, cmdline_reader=None) -> bool:
    if pid <= 0:
        return False
    reader = cmdline_reader or (lambda value: Path(f"/proc/{value}/cmdline").read_bytes())
    try:
        raw = reader(pid)
    except (FileNotFoundError, PermissionError, OSError):
        return False
    text = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    return all(token in text for token in expected_tokens)


def _worker_command(worker_id: str, run_root: Path, approved_manifest: str) -> list[str]:
    return [
        "xvfb-run", "--server-num", str(200 + int(worker_id)),
        "-s", "-screen 0 1280x1024x24",
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/generation_worker.py",
        "--worker-id", worker_id, "--run-root", str(run_root),
        "--approved-manifest", approved_manifest,
    ]


def _coordinator_command(run_root: Path) -> list[str]:
    return [
        "/content/icgs-data-env/bin/python", "-B",
        "/content/ICGS/scripts/generation_coordinator.py",
        "--run-config", str(run_root / "control" / "run.json"),
    ]


def reconcile_processes(
    run_root: str | Path,
    *,
    popen=subprocess.Popen,
    cmdline_reader=None,
    max_restarts: int = MAX_RESTARTS_DEFAULT,
) -> dict:
    """Replace dead/mis-owned slots and atomically persist the launch receipt."""
    root = Path(run_root)
    control = root / "control"
    launch_path = control / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    run_payload = json.loads((control / "run.json").read_text(encoding="utf-8"))
    approved_manifest = str(run_payload["approved_manifest"])
    launch.setdefault("restart_counts", {"coordinator": 0, "workers": {}})
    launch["restart_counts"].setdefault("workers", {})
    for index in range(WORKER_COUNT):
        launch["restart_counts"]["workers"].setdefault(f"{index:03d}", 0)
    launch.setdefault("worker_pids", {})

    coordinator_pid = int(launch.get("coordinator_pid", 0) or 0)
    coordinator_tokens = ("generation_coordinator.py", str(control / "run.json"))
    if not _pid_matches(coordinator_pid, coordinator_tokens, cmdline_reader=cmdline_reader):
        if not _coordinator_lock_is_free(control / "coordinator.lock"):
            return launch
        count = int(launch["restart_counts"]["coordinator"])
        if count >= max_restarts:
            raise RuntimeError("coordinator restart limit exceeded")
        log_stream = (control / "coordinator.log").open("a")
        process = popen(_coordinator_command(root), start_new_session=True,
                        stdout=log_stream, stderr=subprocess.STDOUT)
        launch["coordinator_pid"] = process.pid
        launch["restart_counts"]["coordinator"] = count + 1

    for index in range(WORKER_COUNT):
        worker_id = f"{index:03d}"
        pid = int(launch["worker_pids"].get(worker_id, 0) or 0)
        tokens = ("generation_worker.py", worker_id, str(root))
        if _pid_matches(pid, tokens, cmdline_reader=cmdline_reader):
            continue
        count = int(launch["restart_counts"]["workers"][worker_id])
        if count >= max_restarts:
            raise RuntimeError(f"worker {worker_id} restart limit exceeded")
        process = popen(_worker_command(worker_id, root, approved_manifest),
                        start_new_session=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        launch["worker_pids"][worker_id] = process.pid
        launch["restart_counts"]["workers"][worker_id] = count + 1

    _atomic_json(launch_path, launch)
    return launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS_DEFAULT)
    args = parser.parse_args()
    root = Path(args.run_root)
    while True:
        heartbeat_path = root / "control" / "coordinator-heartbeat.json"
        if heartbeat_path.is_file():
            payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            if payload.get("status") in {"COMPLETE", "FAILED", "INCOMPLETE"}:
                return 0
        try:
            reconcile_processes(root, max_restarts=args.max_restarts)
        except FileNotFoundError:
            # Launcher writes run/launch receipts immediately after spawning;
            # tolerate that short hand-off window without creating untracked slots.
            pass
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
