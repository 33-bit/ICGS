from __future__ import annotations

from pathlib import Path
import json
import os

from scripts.colab_v3_distributed_launch import build_worker_commands, validate_smoke_receipt
from scripts.colab_v3_distributed_coordinator import CoordinatorLock, _inflight_jobs_from_queue
from scripts.colab_v3_distributed_watchdog import (
    _coordinator_lock_is_free,
    _pid_matches,
    reconcile_processes,
)
from icgs.data.collection.v3.distributed_contracts import GenerationJob, RunConfig
from icgs.data.collection.v3.distributed_planner import DistributedPlanner
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


def test_launcher_builds_exactly_200_fixed_display_workers():
    commands = build_worker_commands(
        workers=200,
        run_root="/content/run",
        approved_manifest="/content/manifest.json",
    )
    assert len(commands) == 200
    assert commands[0][commands[0].index("--server-num") + 1] == "200"
    assert commands[-1][commands[-1].index("--server-num") + 1] == "399"
    assert all("-a" not in command for command in commands)


def test_launcher_rejects_non_200_workers():
    import pytest
    with pytest.raises(ValueError, match="workers must be 200"):
        build_worker_commands(workers=199, run_root="/content/run", approved_manifest="/content/manifest.json")


def test_no_stop_call_in_control_sources():
    for name in (
        "scripts/colab_v3_distributed_launch.py",
        "scripts/colab_v3_distributed_coordinator.py",
        "scripts/colab_v3_distributed_watchdog.py",
    ):
        assert "colab stop" not in Path(name).read_text(encoding="utf-8")


def test_launcher_exports_pinned_simulator_environment_to_workers():
    text = Path("scripts/colab_v3_distributed_launch.py").read_text(encoding="utf-8")
    assert "COPPELIASIM_ROOT" in text
    assert "LD_LIBRARY_PATH" in text
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" in text
    assert "PYTHONPATH" in text


def test_coordinator_bounds_ingestion_per_tick():
    text = Path("scripts/colab_v3_distributed_coordinator.py").read_text(encoding="utf-8")
    assert "MAX_READY_PER_TICK = 100" in text
    assert "self.queue.iter_ready()[:MAX_READY_PER_TICK]" in text


def test_launcher_requires_real_code_revision_for_run_contract():
    text = Path("scripts/colab_v3_distributed_launch.py").read_text(encoding="utf-8")
    assert '"--code-revision"' in text
    assert '"code_revision": "unknown"' not in text


def test_launcher_detaches_child_output_from_control_pipe():
    text = Path("scripts/colab_v3_distributed_launch.py").read_text(encoding="utf-8")
    assert "stdout=subprocess.DEVNULL" in text
    assert "coordinator.log" in text


def test_coordinator_lock_rejects_second_owner_and_becomes_free(tmp_path):
    lock_path = tmp_path / "control" / "coordinator.lock"
    with CoordinatorLock(lock_path):
        assert _coordinator_lock_is_free(lock_path) is False
        import pytest
        with pytest.raises(RuntimeError, match="coordinator lock is already held"):
            with CoordinatorLock(lock_path):
                pass
    assert _coordinator_lock_is_free(lock_path) is True


def test_pid_identity_requires_every_expected_cmdline_token():
    reader = lambda pid: b"python\0worker.py\0--worker-id\0" + b"007" + b"\0--run-root\0/content/run\0"
    assert _pid_matches(12, ("worker.py", "007", "/content/run"), cmdline_reader=reader)
    assert not _pid_matches(12, ("worker.py", "008", "/content/run"), cmdline_reader=reader)
    assert not _pid_matches(12, ("worker.py", "007", "/content/other"), cmdline_reader=reader)


def test_watchdog_restarts_only_missing_slots_and_updates_receipt(tmp_path):
    root = tmp_path / "run"
    control = root / "control"
    control.mkdir(parents=True)
    approved = tmp_path / "approved.json"
    approved.write_text("{}", encoding="utf-8")
    (control / "run.json").write_text(json.dumps({
        "run": {"worker_count": 200},
        "approved_manifest": str(approved),
    }), encoding="utf-8")
    worker_pids = {f"{index:03d}": 1000 + index for index in range(200)}
    launch = {
        "workers": 200,
        "worker_pids": worker_pids,
        "coordinator_pid": 900,
        "watchdog_pid": os.getpid(),
        "restart_counts": {
            "coordinator": 0,
            "workers": {f"{index:03d}": 0 for index in range(200)},
        },
    }
    (control / "launch.json").write_text(json.dumps(launch), encoding="utf-8")

    def reader(pid: int) -> bytes:
        if pid == 900:
            return f"python\0colab_v3_distributed_coordinator.py\0--run-config\0{control / 'run.json'}\0".encode()
        if pid == worker_pids["007"]:
            return b""
        worker_id = f"{pid - 1000:03d}"
        return f"python\0colab_v3_distributed_worker.py\0--worker-id\0{worker_id}\0--run-root\0{root}\0".encode()

    class Process:
        pid = 7777

    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    updated = reconcile_processes(root, popen=popen, cmdline_reader=reader)
    assert len(calls) == 1
    command = calls[0][0]
    assert command[command.index("--worker-id") + 1] == "007"
    assert command[command.index("--server-num") + 1] == "207"
    assert updated["worker_pids"]["007"] == 7777
    assert updated["restart_counts"]["workers"]["007"] == 1
    assert updated["coordinator_pid"] == 900


def test_launch_smoke_accepts_retained_valid_failure(tmp_path):
    import json

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "valid_failure" if index == 0 else "success",
                "timeline_ok": True,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_smoke_receipt(path, expected_program_ids=programs)


def test_launch_smoke_rejects_crash_or_missing_timeline(tmp_path):
    import json
    import pytest

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "simulator_crash" if index == 0 else "success",
                "timeline_ok": index != 1,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="blocking smoke outcomes"):
        validate_smoke_receipt(path, expected_program_ids=programs)


def test_coordinator_resume_loads_pending_and_ready_jobs(tmp_path):
    import hashlib
    import json

    manifest_path = Path("artifacts/composition/approved_composition_manifest_v3.json")
    rows = {row["program_id"]: row for row in json.loads(manifest_path.read_text())["catalog"]}
    run = RunConfig(
        run_id="resume-test", run_root=str(tmp_path), code_revision="a" * 40,
        approved_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    planner = DistributedPlanner.from_manifest(run, rows, {"episodes": [], "failure_attempts": []})
    queue = FilesystemJobQueue(tmp_path / "queue")
    first, second = planner.next_job(), planner.next_job()
    queue.enqueue(first)
    queue.enqueue(second)
    assert queue.claim("000") == first
    loaded = _inflight_jobs_from_queue(queue)
    assert {job.job_id for job in loaded} == {first.job_id, second.job_id}
