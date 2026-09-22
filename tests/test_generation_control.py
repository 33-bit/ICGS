from __future__ import annotations

from pathlib import Path
import json
import os
import hashlib
from types import SimpleNamespace

import pytest

from scripts import generation_launch
from scripts import generation_coordinator as generation_coordinator_module
from scripts.generation_launch import validate_smoke_receipt
from scripts.generation_coordinator import (
    CoordinatorControlPlane,
    CoordinatorLock,
    _inflight_jobs_from_queue,
)
from scripts.generation_watchdog import (
    _coordinator_lock_is_free,
    _pid_matches,
    reconcile_processes,
)
from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    GenerationRuntimeConfig,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_planner import DistributedPlanner
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.distributed_validation import ValidatedResult


def _runtime_config(tmp_path: Path, *, publication_enabled: bool = False):
    repo_root = tmp_path / "repo"
    simulator_root = tmp_path / "simulator"
    rlbench_root = tmp_path / "rlbench"
    run_root = tmp_path / "run"
    python_executable = tmp_path / "venv" / "bin" / "python"
    for directory in (repo_root, simulator_root, rlbench_root, run_root, python_executable.parent):
        directory.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    return GenerationRuntimeConfig.from_dict({
        "machine": {
            "repo_root": str(repo_root),
            "python_executable": str(python_executable),
            "simulator_root": str(simulator_root),
            "rlbench_root": str(rlbench_root),
            "display_base": 41,
            "display_width": 1366,
            "display_height": 768,
            "simulator_slots": 2,
            "worker_timeout_s": 47,
        },
        "run": {
            "run_id": "control-test",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 91,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/control-test",
            "publication_enabled": publication_enabled,
            "validation_mode": True,
        },
    })


def _required_api(name: str):
    assert hasattr(generation_launch, name), f"generation_launch.{name} is required"
    return getattr(generation_launch, name)


def test_launcher_builds_workers_from_configured_limits_and_paths(tmp_path: Path):
    config = _runtime_config(tmp_path)
    approved_manifest = str(tmp_path / "approved.json")
    try:
        commands = _required_api("build_worker_commands")(config, approved_manifest)
    except TypeError as exc:
        pytest.fail(f"build_worker_commands must accept a runtime config: {exc}")

    assert len(commands) == 2
    assert [command[command.index("--worker-id") + 1] for command in commands] == ["000", "001"]
    assert [command[command.index("--server-num") + 1] for command in commands] == ["41", "42"]
    assert all(command[command.index("-s") + 1] == "-screen 0 1366x768x24" for command in commands)
    assert all(config.machine.python_executable in command for command in commands)
    assert all(
        str(Path(config.machine.repo_root) / "scripts" / "generation_worker.py") in command
        for command in commands
    )
    assert all(
        command[command.index("--runtime-config") + 1]
        == str(Path(config.run.run_root) / "control" / "runtime_config.json")
        for command in commands
    )
    assert all(command[command.index("--approved-manifest") + 1] == approved_manifest for command in commands)
    assert all("/content" not in " ".join(command) for command in commands)


def test_launcher_persists_runtime_config_digest_before_starting_children(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    runtime_config_path = tmp_path / "runtime.json"
    runtime_config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved_manifest = tmp_path / "approved.json"
    approved_manifest.write_text("{}\n", encoding="utf-8")
    smoke_receipt = tmp_path / "smoke.json"
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS

    smoke_receipt.write_text(json.dumps({
        "results": [
            {"program_id": program_id, "result_class": "success", "timeline_ok": True}
            for program_id in GENERATION_PROGRAMS
        ],
    }), encoding="utf-8")
    run_json = Path(config.run.run_root) / "control" / "run.json"
    runtime_snapshot = run_json.parent / "runtime_config.json"
    calls = []

    class Process:
        def __init__(self, pid: int):
            self.pid = pid

        def wait(self):
            return 0

    def popen(command, **kwargs):
        assert run_json.is_file()
        assert runtime_snapshot.is_file()
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        snapshot_bytes = runtime_snapshot.read_bytes()
        assert payload["runtime_config"] == json.loads(snapshot_bytes)
        assert payload["runtime_config_sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
        calls.append(command)
        return Process(1000 + len(calls))

    monkeypatch.setattr(generation_launch.subprocess, "Popen", popen)
    args = [
        "--runtime-config", str(runtime_config_path),
        "--approved-manifest", str(approved_manifest),
        "--code-revision", "a" * 40,
        "--smoke-receipt", str(smoke_receipt),
        "--detach",
    ]
    try:
        result = generation_launch.main(args)
    except TypeError as exc:
        pytest.fail(f"generation_launch.main must accept runtime-config arguments: {exc}")
    except SystemExit as exc:
        pytest.fail(f"generation_launch rejected runtime-config arguments: {exc}")

    assert result == 0
    assert len(calls) == 4
    stored = json.loads(run_json.read_text(encoding="utf-8"))
    assert stored["run"]["worker_count"] == 2
    assert stored["runtime_config"] == config.as_dict()


def test_no_stop_call_in_control_sources():
    for name in (
        "scripts/generation_launch.py",
        "scripts/generation_coordinator.py",
        "scripts/generation_watchdog.py",
    ):
        assert "colab stop" not in Path(name).read_text(encoding="utf-8")


def test_launcher_exports_pinned_simulator_environment_to_workers(tmp_path: Path):
    config = _runtime_config(tmp_path)
    environment = _required_api("build_process_environment")(config, base={})
    assert environment["COPPELIASIM_ROOT"] == config.machine.simulator_root
    assert environment["LD_LIBRARY_PATH"] == config.machine.simulator_root
    assert environment["QT_QPA_PLATFORM_PLUGIN_PATH"] == config.machine.simulator_root
    assert str(Path(config.machine.repo_root) / "src") in environment["PYTHONPATH"]


def test_coordinator_bounds_ingestion_per_tick():
    text = Path("scripts/generation_coordinator.py").read_text(encoding="utf-8")
    assert "MAX_READY_PER_TICK = 100" in text
    assert "self.queue.iter_ready()[:MAX_READY_PER_TICK]" in text


@pytest.mark.parametrize(
    "malformed_result_json",
    [False, True],
    ids=["validation-failure", "malformed-result-json"],
)
def test_tick_quarantines_bad_result_and_ingests_following_valid_result(
    tmp_path: Path,
    monkeypatch,
    malformed_result_json: bool,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    run = RunConfig(
        run_id="quarantine-run",
        run_root=str(run_root),
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        worker_count=2,
    )
    queue = FilesystemJobQueue(run_root / "queue")

    def make_job(job_id: str, index: int) -> GenerationJob:
        plan = AttemptPlan(
            program_id="T01", split="train", episode_index=index,
            episode_id=f"episode-t01-{index:06d}", episode_kind="nominal",
            scene_seed=index, collection_seed=20260920,
            randomization={"scene_signature": f"sig-{index}", "asset_instance_id": f"asset-{index}"},
            intervention=None,
        )
        return GenerationJob.create(
            job_id=job_id, run_id=run.run_id, attempt_id=f"att-{plan.episode_id}",
            episode_id=plan.episode_id, program_id="T01", plan=plan,
            code_revision="a" * 40, manifest_sha256="b" * 64,
            output_root=str(run_root / "staging"),
        )

    malformed_job = make_job("job-a-malformed", 1)
    valid_job = make_job("job-b-valid", 2)
    ready_results = []
    for worker_id, job, content in (
        ("000", malformed_job, b"malformed artifact"),
        ("001", valid_job, b"valid artifact"),
    ):
        result_dir = run_root / "staging" / "worker-results" / job.job_id / "T01"
        result_dir.mkdir(parents=True)
        (result_dir / "artifact.bin").write_bytes(content)
        queue.enqueue(job)
        assert queue.claim(worker_id) == job
        result = WorkerResult(
            job_id=job.job_id, attempt_id=job.attempt_id, episode_id=job.episode_id,
            program_id=job.program_id, outcome="success", result_dir=str(result_dir),
            file_sha256={}, timeline={"actions": 0, "observations": 1, "durations": 0},
        )
        ready_path = queue.publish_ready(worker_id, result)
        if malformed_result_json and job.job_id == malformed_job.job_id:
            (ready_path / "result.json").write_text("{ malformed", encoding="utf-8")
        ready_results.append(result)

    class Planner:
        def __init__(self):
            self.recorded = []

        def next_job(self):
            return None

        def record_result(self, result, provenance):
            self.recorded.append((result, provenance))

        def quota_complete(self):
            return False

        def snapshot(self):
            return SimpleNamespace(as_dict=lambda: {})

    class Publisher:
        def publish_due(self, **kwargs):
            return None

    planner = Planner()
    control = CoordinatorControlPlane(run, queue, planner, Publisher())

    def validate(job, result):
        if job.job_id == malformed_job.job_id:
            raise ValueError("invalid artifact inventory")
        return ValidatedResult(
            job=job,
            result=result,
            provenance={"episode_kind": "nominal"},
            episode_entry={
                "attempt_id": job.attempt_id,
                "episode_id": job.episode_id,
                "program_id": job.program_id,
                "outcome": "success",
            },
            attempt_entry=None,
        )

    def ingest(manifest, validated):
        updated = dict(manifest)
        updated["episodes"] = list(updated.get("episodes") or []) + [validated.episode_entry]
        updated["failure_attempts"] = list(updated.get("failure_attempts") or [])
        return updated

    monkeypatch.setattr(generation_coordinator_module, "validate_closed_result", validate)
    monkeypatch.setattr(generation_coordinator_module, "ingest_validated_result", ingest)
    assert hasattr(control, "process_ready_result"), "process_ready_result is required"
    process_ready_result = control.process_ready_result
    outcomes = []

    def record_processing(result):
        outcome = process_ready_result(result)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(control, "process_ready_result", record_processing)
    control.tick(now_s=10.0)

    assert outcomes == ["quarantined", "ingested"]
    assert control.status == "RUNNING"
    assert [result.job_id for result, _ in planner.recorded] == [valid_job.job_id]
    assert [row["episode_id"] for row in control.manifest["episodes"]] == [valid_job.episode_id]
    assert control.manifest["failure_attempts"] == []
    counts = queue.counts()
    assert counts.quarantined == 1
    assert counts.ingested == 1
    assert tuple(job.job_id for job in _inflight_jobs_from_queue(queue)) == (
        malformed_job.job_id,
    )
    failure = json.loads((queue.root / "quarantined" / malformed_job.job_id / "validation_failure.json").read_text())
    if malformed_result_json:
        assert failure["result_identity"]["job_id"] == malformed_job.job_id
        assert failure["result_identity"]["result_json_sha256"] == hashlib.sha256(
            b"{ malformed"
        ).hexdigest()
        assert failure["exception_type"] == "JSONDecodeError"
        assert "artifact.bin" in failure["actual_file_inventory"]
    else:
        assert failure["result_identity"]["outcome"] == "success"
        assert failure["exception_message"] == "invalid artifact inventory"


def test_launcher_requires_real_code_revision_for_run_contract():
    text = Path("scripts/generation_launch.py").read_text(encoding="utf-8")
    assert '"--code-revision"' in text
    assert '"code_revision": "unknown"' not in text


def test_launcher_detaches_child_output_from_control_pipe():
    text = Path("scripts/generation_launch.py").read_text(encoding="utf-8")
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
            return f"python\0generation_coordinator.py\0--run-config\0{control / 'run.json'}\0".encode()
        if pid == worker_pids["007"]:
            return b""
        worker_id = f"{pid - 1000:03d}"
        return f"python\0generation_worker.py\0--worker-id\0{worker_id}\0--run-root\0{root}\0".encode()

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

    manifest_path = Path("artifacts/composition/approved_composition_manifest.json")
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
