"""Keepalive and bounded process replacement for one generation run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import Literal

from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig


MAX_RESTARTS_DEFAULT = 20
DEFAULT_STALE_AFTER_S = 900.0
_TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED", "INCOMPLETE"})
_SHELL_PROGRAMS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})


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


def _read_process_cmdline(pid: int) -> bytes | str:
    """Read a process command line on procfs hosts and POSIX fallbacks."""
    proc_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_path.read_bytes()
        if raw:
            return raw
    except (FileNotFoundError, PermissionError, OSError):
        pass

    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as error:
        raise OSError(f"cannot inspect process {pid}") from error
    if result.returncode != 0 or not result.stdout:
        raise OSError(f"process {pid} is not inspectable")
    return result.stdout


def _command_line_args(raw: object) -> tuple[str, ...]:
    if isinstance(raw, bytes):
        if b"\0" in raw:
            return tuple(
                item.decode("utf-8", errors="replace")
                for item in raw.split(b"\0")
                if item
            )
        raw = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, bytearray):
        return _command_line_args(bytes(raw))
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        return tuple(str(item) for item in raw if str(item))
    if not isinstance(raw, str) or not raw.strip():
        return ()
    try:
        return tuple(shlex.split(raw))
    except ValueError:
        return ()


def _flatten_command(command: Sequence[str]) -> tuple[str, ...]:
    """Normalize procfs and ps forms of xvfb-run's grouped -s argument."""
    flattened: list[str] = []
    expand_next = False
    for value in command:
        text = str(value)
        if expand_next:
            try:
                flattened.extend(shlex.split(text))
            except ValueError:
                flattened.append(text)
            expand_next = False
        else:
            flattened.append(text)
        if text == "-s":
            expand_next = True
    return tuple(flattened)


def _program_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if Path(expected).is_absolute():
        return os.path.realpath(actual) == os.path.realpath(expected)
    resolved = shutil.which(expected)
    if resolved is not None:
        return os.path.realpath(actual) == os.path.realpath(resolved)
    return Path(actual).name == expected


def _is_shell_program(value: str) -> bool:
    return Path(value).name in _SHELL_PROGRAMS


def _command_identity_matches(
    actual: Sequence[str],
    expected_command: Sequence[str],
    wrapper_command: Sequence[str] | None,
) -> bool:
    actual_args = _flatten_command(actual)
    expected_args = _flatten_command(expected_command)
    if not expected_args:
        return False
    if (
        len(actual_args) == len(expected_args)
        and actual_args[1:] == expected_args[1:]
        and _program_matches(actual_args[0], expected_args[0])
    ):
        return True
    if wrapper_command is None:
        return False

    wrapper_args = _flatten_command(wrapper_command)
    if not wrapper_args:
        return False
    wrapped_suffix = wrapper_args[1:] + expected_args
    if (
        len(actual_args) == len(wrapped_suffix) + 1
        and _program_matches(actual_args[0], wrapper_args[0])
        and actual_args[1:] == wrapped_suffix
    ):
        return True
    return (
        len(actual_args) == len(wrapped_suffix) + 2
        and _is_shell_program(actual_args[0])
        and _program_matches(actual_args[1], wrapper_args[0])
        and actual_args[2:] == wrapped_suffix
    )


def _pid_matches(
    pid: int,
    expected_tokens: tuple[str, ...] = (),
    *,
    cmdline_reader=None,
    expected_command: Sequence[str] | None = None,
    wrapper_command: Sequence[str] | None = None,
) -> bool:
    if pid <= 0:
        return False
    reader = cmdline_reader or _read_process_cmdline
    try:
        actual = _command_line_args(reader(pid))
    except (FileNotFoundError, PermissionError, OSError):
        return False
    if expected_command is not None:
        return _command_identity_matches(actual, expected_command, wrapper_command)
    return bool(actual) and all(token in actual for token in expected_tokens)


def heartbeat_health(
    payload: Mapping[str, object],
    *,
    now_s: float,
    stale_after_s: float,
) -> Literal["healthy", "busy", "stale", "terminal"]:
    """Classify coordinator liveness from terminal state and durable progress."""
    if type(now_s) not in (int, float) or type(stale_after_s) not in (int, float):
        raise ValueError("now_s and stale_after_s must be numbers")
    if stale_after_s <= 0:
        raise ValueError("stale_after_s must be positive")
    if not isinstance(payload, Mapping):
        return "stale"
    if payload.get("status") in _TERMINAL_STATUSES:
        return "terminal"

    phase = payload.get("phase")
    publication = payload.get("publication")
    publication_phase = publication.get("phase") if isinstance(publication, Mapping) else None
    if phase == "publication_in_progress" or publication_phase == "publication_in_progress":
        return "busy"

    timestamp = payload.get("timestamp_s")
    if type(timestamp) not in (int, float):
        return "stale"
    if now_s - float(timestamp) > stale_after_s:
        return "stale"

    if phase in {"tick_start", "validation", "refill", "publication", "publication_error"}:
        return "busy"

    progress = payload.get("last_progress_at_s")
    if type(progress) not in (int, float):
        return "busy"
    if now_s - float(progress) > stale_after_s:
        return "stale"
    return "healthy"


def _load_runtime_config(
    run_payload: Mapping[str, object],
    *,
    root: Path,
) -> tuple[GenerationRuntimeConfig, Path]:
    runtime_value = run_payload.get("runtime_config_path")
    if not isinstance(runtime_value, str) or not runtime_value.strip():
        raise ValueError("run receipt must contain runtime_config_path")
    runtime_path = Path(runtime_value)
    if not runtime_path.is_absolute():
        runtime_path = (root / "control" / runtime_path).resolve()
    if runtime_path.is_symlink() or not runtime_path.is_file():
        raise ValueError(f"runtime config must be a regular file: {runtime_path}")
    config = GenerationRuntimeConfig.from_file(runtime_path, check_paths=False)
    if Path(config.run.run_root).resolve() != root.resolve():
        raise ValueError("runtime config run_root does not match watchdog run root")
    return config, runtime_path


def _worker_command(
    worker_id: str,
    config: GenerationRuntimeConfig,
    runtime_config_path: Path,
    approved_manifest: str,
) -> list[str]:
    width = max(3, len(str(config.run.worker_count - 1)))
    if worker_id != f"{int(worker_id):0{width}d}":
        raise ValueError(f"invalid worker id: {worker_id}")
    return [
        "xvfb-run",
        "--server-num", str(config.machine.display_base + int(worker_id)),
        "-s", (
            f"-screen 0 {config.machine.display_width}x{config.machine.display_height}x24 "
            "+extension GLX +render -noreset"
        ),
        config.machine.python_executable,
        "-B",
        str(Path(config.machine.repo_root) / "scripts" / "generation_worker.py"),
        "--worker-id", worker_id,
        "--runtime-config", str(runtime_config_path),
        "--approved-manifest", approved_manifest,
    ]


def _coordinator_command(
    config: GenerationRuntimeConfig,
    runtime_config_path: Path,
    run_config_path: Path,
) -> list[str]:
    return [
        config.machine.python_executable,
        "-B",
        str(Path(config.machine.repo_root) / "scripts" / "generation_coordinator.py"),
        "--run-config", str(run_config_path),
        "--runtime-config", str(runtime_config_path),
    ]


def _record_restart(
    launch: dict,
    *,
    component: str,
    reason: str,
    previous_pid: int,
    new_pid: int,
    now_s: float,
) -> None:
    launch.setdefault("restart_history", []).append({
        "component": component,
        "reason": reason,
        "previous_pid": previous_pid,
        "new_pid": new_pid,
        "at_s": now_s,
    })


def reconcile_processes(
    run_root: str | Path,
    *,
    popen=subprocess.Popen,
    cmdline_reader=None,
    max_restarts: int = MAX_RESTARTS_DEFAULT,
    now_s: float | None = None,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
) -> dict:
    """Replace mis-owned slots and stale coordinators, then atomically persist receipt."""
    if max_restarts < 0:
        raise ValueError("max_restarts must be nonnegative")
    root = Path(run_root)
    control = root / "control"
    launch_path = control / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    run_payload = json.loads((control / "run.json").read_text(encoding="utf-8"))
    config, runtime_config_path = _load_runtime_config(run_payload, root=root)
    approved_manifest = str(run_payload["approved_manifest"])
    now = time.time() if now_s is None else now_s
    launch.setdefault("restart_counts", {"coordinator": 0, "workers": {}})
    launch["restart_counts"].setdefault("workers", {})
    launch.setdefault("worker_pids", {})
    launch.setdefault("restart_history", [])
    for index in range(config.run.worker_count):
        launch["restart_counts"]["workers"].setdefault(f"{index:03d}", 0)

    heartbeat_path = control / "coordinator-heartbeat.json"
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        heartbeat = {}
    health = heartbeat_health(heartbeat, now_s=now, stale_after_s=stale_after_s)
    coordinator_pid = int(launch.get("coordinator_pid", 0) or 0)
    coordinator_command = tuple(
        _coordinator_command(config, runtime_config_path, control / "run.json")
    )
    coordinator_tokens = (
        str(Path(config.machine.repo_root) / "scripts" / "generation_coordinator.py"),
        str(control / "run.json"),
        str(runtime_config_path),
    )
    pid_valid = _pid_matches(
        coordinator_pid,
        coordinator_tokens,
        cmdline_reader=cmdline_reader,
        expected_command=coordinator_command,
    )
    coordinator_reason = None
    if health != "terminal":
        if not pid_valid:
            coordinator_reason = "pid_invalid"
        elif health == "stale" and heartbeat:
            # The launcher starts the watchdog immediately after the coordinator.
            # Give a valid coordinator time to publish its first heartbeat instead
            # of creating a duplicate that will lose the coordinator lock.
            coordinator_reason = "heartbeat_stale"

    if coordinator_reason is not None:
        if not _coordinator_lock_is_free(control / "coordinator.lock"):
            return launch
        count = int(launch["restart_counts"]["coordinator"])
        if count >= max_restarts:
            raise RuntimeError("coordinator restart limit exceeded")
        log_stream = (control / "coordinator.log").open("a")
        try:
            process = popen(
                _coordinator_command(config, runtime_config_path, control / "run.json"),
                start_new_session=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_stream.close()
        launch["coordinator_pid"] = process.pid
        launch["restart_counts"]["coordinator"] = count + 1
        _record_restart(
            launch,
            component="coordinator",
            reason=coordinator_reason,
            previous_pid=coordinator_pid,
            new_pid=process.pid,
            now_s=now,
        )

    for index in range(config.run.worker_count):
        worker_id = f"{index:03d}"
        pid = int(launch["worker_pids"].get(worker_id, 0) or 0)
        worker_command = tuple(
            _worker_command(worker_id, config, runtime_config_path, approved_manifest)
        )
        python_index = worker_command.index(config.machine.python_executable)
        worker_wrapper = worker_command[:python_index]
        worker_invocation = worker_command[python_index:]
        tokens = (
            str(Path(config.machine.repo_root) / "scripts" / "generation_worker.py"),
            worker_id,
            "--runtime-config",
            str(runtime_config_path),
        )
        if _pid_matches(
            pid,
            tokens,
            cmdline_reader=cmdline_reader,
            expected_command=worker_invocation,
            wrapper_command=worker_wrapper,
        ):
            continue
        count = int(launch["restart_counts"]["workers"][worker_id])
        if count >= max_restarts:
            raise RuntimeError(f"worker {worker_id} restart limit exceeded")
        process = popen(
            list(worker_command),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        launch["worker_pids"][worker_id] = process.pid
        launch["restart_counts"]["workers"][worker_id] = count + 1
        _record_restart(
            launch,
            component=f"worker:{worker_id}",
            reason="pid_invalid",
            previous_pid=pid,
            new_pid=process.pid,
            now_s=now,
        )

    _atomic_json(launch_path, launch)
    return launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--stale-after-s", type=float, default=DEFAULT_STALE_AFTER_S)
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS_DEFAULT)
    args = parser.parse_args()
    root = Path(args.run_root)
    while True:
        heartbeat_path = root / "control" / "coordinator-heartbeat.json"
        if heartbeat_path.is_file():
            payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            if payload.get("status") in _TERMINAL_STATUSES:
                return 0
        try:
            reconcile_processes(
                root,
                max_restarts=args.max_restarts,
                stale_after_s=args.stale_after_s,
            )
        except FileNotFoundError:
            # Launcher writes run/launch receipts immediately after spawning;
            # tolerate that short hand-off window without creating untracked slots.
            pass
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
