from pathlib import Path

import pytest

from icgs.data.collection.generation.capacity_probe import (
    CapacityProbeConfig,
    CapacityStage,
    categorize_artifact_bytes,
    extract_result_metrics,
    measure_staging_tree_bytes,
    preflight_stage_capacity,
)
from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    AttemptPlan,
    GenerationJob,
    GenerationRuntimeConfig,
)
from scripts import generation_capacity_probe


def _make_job(job_id: str, episode_id: str = "episode-001") -> GenerationJob:
    plan = AttemptPlan(
        program_id="T01",
        split="train",
        episode_index=1,
        episode_id=episode_id,
        episode_kind="nominal",
        scene_seed=1000,
        collection_seed=20260920,
        randomization={"scene_signature": "sig-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    return GenerationJob.create(
        job_id=job_id,
        run_id="run-1",
        attempt_id=f"att-{episode_id}",
        episode_id=episode_id,
        program_id="T01",
        plan=plan,
        code_revision="a" * 40,
        manifest_sha256="b" * 64,
        output_root="/tmp/out",
    )


def _payload():
    return {
        "run_id": "capacity-test",
        "hf_subfolder": "validation/capacity-test",
        "max_total_jobs": 40,
        "max_runtime_s": 600,
        "stages": [
            {"name": "low", "worker_count": 8, "simulator_slots": 2, "max_jobs": 8, "worker_timeout_s": 180},
            {"name": "high", "worker_count": 16, "simulator_slots": 4, "max_jobs": 16, "worker_timeout_s": 180},
        ],
    }


def test_capacity_probe_config_accepts_bounded_stages():
    config = CapacityProbeConfig.from_dict(_payload())
    assert config.stages[1].worker_count == 16
    assert config.as_dict()["stages"][0]["max_jobs"] == 8


@pytest.mark.parametrize("field,value", [("max_total_jobs", 401), ("max_runtime_s", 5401)])
def test_capacity_probe_rejects_global_limits(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_stage_sum_over_global_cap():
    payload = _payload()
    payload["max_total_jobs"] = 10
    with pytest.raises(ValueError, match="exceed"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_non_validation_prefix():
    payload = _payload()
    payload["hf_subfolder"] = "generation/production"
    with pytest.raises(ValueError, match="validation"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_unsafe_stage_path():
    payload = _payload()
    payload["stages"][0]["name"] = "../escape"
    with pytest.raises(ValueError, match="stage name"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_unsafe_run_identity():
    payload = _payload()
    payload["run_id"] = "../escape"
    with pytest.raises(ValueError, match="run_id"):
        CapacityProbeConfig.from_dict(payload)


def test_stage_process_cleanup_terminates_only_owned_process_groups(monkeypatch):
    calls = []

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise generation_capacity_probe.subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(generation_capacity_probe.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    generation_capacity_probe.stop_processes([Process()])
    assert calls == [(1234, generation_capacity_probe.signal.SIGTERM), (1234, generation_capacity_probe.signal.SIGKILL)]


def test_stage_planning_never_enqueues_past_stage_cap(tmp_path):
    class Planner:
        def __init__(self):
            self.calls = 0

        def next_job(self):
            self.calls += 1
            return type("Job", (), {"job_id": f"job-{self.calls}"})()

    class Queue:
        def __init__(self):
            self.jobs = []

        def enqueue(self, job):
            self.jobs.append(job.job_id)

    planner, queue = Planner(), Queue()
    generation_capacity_probe.enqueue_fixed_jobs(planner, queue, 8)
    assert planner.calls == 8
    assert queue.jobs == [f"job-{index}" for index in range(1, 9)]


def test_stage_runtime_disables_publication_and_rewrites_identity(tmp_path):
    payload = {
        "machine": {
            "repo_root": str(tmp_path / "repo"),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path / "sim"),
            "rlbench_root": str(tmp_path / "rlbench"),
            "display_base": 20,
            "display_width": 1024,
            "display_height": 768,
            "simulator_slots": 2,
            "worker_timeout_s": 1200,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path / "base"),
            "worker_count": 2,
            "hf_subfolder": "validation/capacity-test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(payload)
    probe = CapacityProbeConfig.from_dict(_payload())
    runtime = generation_capacity_probe._stage_runtime(base, probe, probe.stages[1], tmp_path / "high")
    assert runtime.run.worker_count == 16
    assert runtime.machine.simulator_slots == 4
    assert runtime.run.run_id == "capacity-test-high"
    assert runtime.run.publication_enabled is False
    assert runtime.machine.worker_timeout_s == 180
    assert len(runtime.machine.worker_ids) == 16
    assert runtime.archive_profile == base.archive_profile


def test_archive_stage_runtime_carries_effective_result_cap_to_worker_snapshot(tmp_path):
    payload = {
        "machine": {"repo_root": str(tmp_path), "python_executable": str(tmp_path / "python"),
                    "simulator_root": str(tmp_path), "rlbench_root": str(tmp_path),
                    "display_base": 1, "display_width": 100, "display_height": 100,
                    "simulator_slots": 1, "worker_timeout_s": 60},
        "run": {"run_id": "base", "run_root": str(tmp_path), "worker_count": 2,
                "hf_subfolder": "validation/test", "publication_enabled": False,
                "validation_mode": True},
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(payload)
    stage = CapacityStage(name="small", worker_count=2, simulator_slots=1, max_jobs=2,
                          max_result_bytes=1000, max_staging_bytes=3000)
    probe = CapacityProbeConfig(run_id="probe", hf_subfolder="validation/test", max_total_jobs=2,
                                max_runtime_s=300, stages=(stage,), max_result_bytes=2000,
                                max_total_staging_bytes=5000)
    runtime = generation_capacity_probe._stage_runtime(base, probe, stage, tmp_path / "small")
    restored = GenerationRuntimeConfig.from_dict(runtime.as_dict())
    assert restored.max_result_bytes == 1000
    assert restored.resolved_environment(base={})["ICGS_GENERATION_MAX_RESULT_BYTES"] == "1000"
    assert restored.archive_profile.as_dict() == base.archive_profile.as_dict()


def test_wait_for_ready_results_finishes_before_idle_workers_exit(monkeypatch):
    class Counts:
        def __init__(self, ready):
            self.ready = ready
            self.pending = 2 - ready
            self.claimed = 0

    class Queue:
        def __init__(self):
            self.calls = 0

        def counts(self):
            self.calls += 1
            return Counts(min(self.calls - 1, 2))

    class Process:
        def poll(self):
            return None

    monkeypatch.setattr(generation_capacity_probe.time, "sleep", lambda _: None)
    monkeypatch.setattr(generation_capacity_probe, "_free_memory_bytes", lambda: 100 * 1024**3)
    result = generation_capacity_probe.wait_for_ready_results(
        Queue(), [Process(), Process(), Process()], expected_jobs=2,
        deadline=generation_capacity_probe.time.monotonic() + 10,
    )
    assert result["ready"] == 2
    assert result["active_workers_at_completion"] == 3


def test_wait_for_ready_results_rejects_workers_exiting_with_unfinished_jobs(monkeypatch):
    class Counts:
        pending = 1
        claimed = 0
        ready = 0

    class Queue:
        def counts(self):
            return Counts()

    class Process:
        def poll(self):
            return 0

    monkeypatch.setattr(generation_capacity_probe, "_free_memory_bytes", lambda: 100 * 1024**3)
    with pytest.raises(RuntimeError, match="before all results"):
        generation_capacity_probe.wait_for_ready_results(
            Queue(), [Process()], expected_jobs=1,
            deadline=generation_capacity_probe.time.monotonic() + 10,
        )


def test_capacity_probe_config_staging_caps():
    payload = _payload()
    payload["max_result_bytes"] = 10 * 1024 * 1024
    payload["max_total_staging_bytes"] = 500 * 1024 * 1024
    payload["stages"][0]["max_result_bytes"] = 10 * 1024 * 1024
    payload["stages"][0]["max_staging_bytes"] = 100 * 1024 * 1024
    config = CapacityProbeConfig.from_dict(payload)
    assert config.max_result_bytes == 10 * 1024 * 1024
    assert config.max_total_staging_bytes == 500 * 1024 * 1024
    assert config.stages[0].max_result_bytes == 10 * 1024 * 1024
    assert config.stages[0].max_staging_bytes == 100 * 1024 * 1024
    dumped = config.as_dict()
    assert dumped["max_result_bytes"] == 10 * 1024 * 1024
    assert dumped["max_total_staging_bytes"] == 500 * 1024 * 1024
    assert dumped["stages"][0]["max_result_bytes"] == 10 * 1024 * 1024
    assert dumped["stages"][0]["max_staging_bytes"] == 100 * 1024 * 1024


@pytest.mark.parametrize("field,value", [
    ("max_result_bytes", -1),
    ("max_result_bytes", 0),
    ("max_result_bytes", "100"),
    ("max_total_staging_bytes", 0),
    ("max_total_staging_bytes", -50),
])
def test_capacity_probe_rejects_invalid_caps(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_stage_cap_exceeding_total():
    payload = _payload()
    payload["max_total_staging_bytes"] = 100
    payload["stages"][0]["max_staging_bytes"] = 200
    with pytest.raises(ValueError, match="staging"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_stage_rejects_per_result_exceeding_stage_staging():
    with pytest.raises(ValueError, match="exceed"):
        CapacityStage(
            name="test",
            worker_count=2,
            simulator_slots=1,
            max_jobs=2,
            max_result_bytes=500,
            max_staging_bytes=200,
        )


def test_category_accounting_categorizes_archive_files(tmp_path):
    result_dir = tmp_path / "result_001"
    (result_dir / "data").mkdir(parents=True)
    (result_dir / "views").mkdir(parents=True)

    (result_dir / "data" / "chunk-00000.npz").write_bytes(b"C" * 100)
    (result_dir / "data" / "chunk-00001.npz").write_bytes(b"C" * 200)
    (result_dir / "episode.manifest.json").write_bytes(b"M" * 50)
    (result_dir / "artifact_manifest.json").write_bytes(b"M" * 30)
    (result_dir / "debug.json").write_bytes(b"D" * 40)
    (result_dir / "views" / "geom.view.json").write_bytes(b"V" * 25)
    (result_dir / "worker_receipt.json").write_bytes(b"R" * 15)

    categories = categorize_artifact_bytes(result_dir)
    assert categories["chunks"] == 300
    assert categories["manifests"] == 80
    assert categories["debug"] == 40
    assert categories["views"] == 25
    assert categories["receipts"] == 15
    assert sum(categories.values()) == 460


def test_extract_result_metrics_without_loading_prior_episodes(tmp_path):
    import json
    result_dir = tmp_path / "result_ep"
    result_dir.mkdir(parents=True)
    manifest = {
        "episode_id": "ep-001",
        "attempt_id": "att-001",
        "archive_profile": ArchiveProfileConfig().as_dict(),
        "timeline": {"observations": 55, "transitions": 54},
        "array_specs": {
            "online_points": {
                "shape": [112640, 3],
                "semantic_role": "online_observations/points",
            },
        },
        "local_write_limits": {
            "max_chunk_bytes": 268435456,
            "spool_peak_bytes": 1000000,
            "local_peak_bytes_upper_bound": 2500000,
        },
    }
    (result_dir / "episode.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (result_dir / "debug.json").write_text("{}", encoding="utf-8")

    metrics = extract_result_metrics(result_dir)
    assert metrics["boundaries"] == 55
    assert metrics["raw_points"] == 112640
    assert metrics["writer_peak_bytes_upper_bound"] == 2500000
    assert metrics["writer_peak_bytes_semantic"] == "archive_writer_reported_upper_bound"
    assert metrics["archive_profile"]["dataset_identity"] == "icgs-primary-v3-archive-v1"
    assert metrics["bytes_by_category"]["manifests"] > 0
    assert metrics["bytes_by_category"]["debug"] > 0


def test_preflight_no_worker_behavior_when_capacity_insufficient(tmp_path):
    stage = CapacityStage(
        name="bounded",
        worker_count=4,
        simulator_slots=2,
        max_jobs=4,
        max_staging_bytes=1000 * 1024 * 1024,
    )
    probe = CapacityProbeConfig(
        run_id="cap-run",
        hf_subfolder="validation/probe",
        max_total_jobs=10,
        max_runtime_s=300,
        stages=(stage,),
        max_total_staging_bytes=2000 * 1024 * 1024,
    )

    class FakeDiskUsage:
        free = 100 * 1024 * 1024  # only 100 MB free, but 1000 MB required

    with pytest.raises(RuntimeError, match="preflight capacity insufficient"):
        preflight_stage_capacity(stage, probe, tmp_path, disk_usage_fn=lambda _: FakeDiskUsage())


def test_byte_cap_rejection_in_stage_results(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock
    from icgs.data.collection.generation.distributed_contracts import WorkerResult

    stage = CapacityStage(
        name="low",
        worker_count=2,
        simulator_slots=1,
        max_jobs=1,
        max_result_bytes=500,  # Cap at 500 bytes
        max_staging_bytes=1000,
    )
    probe = CapacityProbeConfig(
        run_id="cap-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage,),
        max_result_bytes=500,
        max_total_staging_bytes=2000,
    )
    res_dir = tmp_path / "res_001"
    res_dir.mkdir(parents=True)
    (res_dir / "large_file.dat").write_bytes(b"X" * 1500)  # 1500 bytes exceeds 500 cap
    (res_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")

    from types import SimpleNamespace

    class MockQueue:
        root = tmp_path / "queue"
        def counts(self):
            return SimpleNamespace(ready=1, pending=0, claimed=0)
        def iter_ready(self):
            return [
                WorkerResult(
                    job_id="j1",
                    attempt_id="att-ep1",
                    episode_id="ep1",
                    program_id="T01",
                    outcome="success",
                    result_dir=str(res_dir),
                    file_sha256={"large_file.dat": "0" * 64, "artifact_manifest.json": "0" * 64},
                    timeline={"observations": 2, "actions": 1, "durations": 1},
                )
            ]

    # Preflight should pass if disk space is sufficient
    class GoodDisk:
        free = 10**9
    monkeypatch.setattr("icgs.data.collection.generation.capacity_probe.shutil.disk_usage", lambda _: GoodDisk())
    monkeypatch.setattr(generation_capacity_probe.shutil, "disk_usage", lambda _: GoodDisk())
    monkeypatch.setattr(generation_capacity_probe, "validate_closed_result", lambda job, res, **kw: SimpleNamespace(outcome="success"))
    monkeypatch.setattr(generation_capacity_probe, "wait_for_ready_results", lambda *a, **kw: {"active_workers_at_completion": 0, "minimum_available_memory_bytes": 10**9})
    monkeypatch.setattr(generation_capacity_probe, "stop_processes", lambda _: None)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", lambda *a, **kw: mock_proc)

    # Mock queue directory
    job = _make_job("j1", "ep1")
    job_file = tmp_path / "queue" / "ready" / "j1" / "job.json"
    job_file.parent.mkdir(parents=True)
    job_file.write_text(json.dumps(job.as_dict()), encoding="utf-8")

    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path),
            "worker_count": 1,
            "hf_subfolder": "validation/test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)

    manifest_file = Path("artifacts/composition/approved_composition_manifest.json").resolve()

    # Patch FilesystemJobQueue and enqueue_fixed_jobs
    monkeypatch.setattr(generation_capacity_probe, "FilesystemJobQueue", lambda _: MockQueue())
    monkeypatch.setattr(generation_capacity_probe, "enqueue_fixed_jobs", lambda *a: ["j1"])

    receipt = generation_capacity_probe.run_stage(
        base, probe, stage,
        output_root=tmp_path / "stage_out",
        approved_manifest=manifest_file,
        code_revision="a" * 40,
        deadline=10**10,
    )
    assert receipt["status"] == "FAIL"
    assert any("ByteCapExceeded" in str(item) for item in receipt["invalid_results"])
    assert "results" in receipt
    assert len(receipt["results"]) == 1
    assert receipt["results"][0]["artifact_bytes"] > 500


def test_stage_preflight_blocks_worker_launch_on_insufficient_capacity(tmp_path, monkeypatch):
    import json
    stage = CapacityStage(
        name="low",
        worker_count=2,
        simulator_slots=1,
        max_jobs=1,
        max_result_bytes=500 * 1024 * 1024,
        max_staging_bytes=1000 * 1024 * 1024,
    )
    probe = CapacityProbeConfig(
        run_id="cap-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage,),
        max_result_bytes=500 * 1024 * 1024,
        max_total_staging_bytes=1000 * 1024 * 1024,
    )
    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path),
            "worker_count": 2,
            "hf_subfolder": "validation/test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)

    # Mock low disk space
    class LowDisk:
        free = 1024  # only 1 KB free
    monkeypatch.setattr("icgs.data.collection.generation.capacity_probe.shutil.disk_usage", lambda _: LowDisk())
    monkeypatch.setattr(generation_capacity_probe.shutil, "disk_usage", lambda _: LowDisk())

    # Ensure subprocess.Popen is never called
    spawned = []
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", lambda *a, **kw: spawned.append(a))

    manifest_file = Path("artifacts/composition/approved_composition_manifest.json").resolve()

    with pytest.raises(RuntimeError, match="preflight capacity insufficient"):
        generation_capacity_probe.run_stage(
            base, probe, stage,
            output_root=tmp_path / "stage_out",
            approved_manifest=manifest_file,
            code_revision="a" * 40,
            deadline=10**10,
        )
    assert len(spawned) == 0, "No workers should be launched when preflight capacity is insufficient"


def test_compact_profile_validation_and_receipt_instrumentation(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock
    from icgs.data.collection.generation.distributed_contracts import WorkerResult

    stage = CapacityStage(
        name="compact_stage",
        worker_count=1,
        simulator_slots=1,
        max_jobs=1,
        max_result_bytes=10 * 1024 * 1024,
        max_staging_bytes=50 * 1024 * 1024,
    )
    probe = CapacityProbeConfig(
        run_id="probe-compact",
        hf_subfolder="validation/test",
        max_total_jobs=2,
        max_runtime_s=300,
        stages=(stage,),
        max_result_bytes=10 * 1024 * 1024,
        max_total_staging_bytes=50 * 1024 * 1024,
    )

    res_dir = tmp_path / "result_ep"
    (res_dir / "data").mkdir(parents=True)
    (res_dir / "views").mkdir(parents=True)
    (res_dir / "data" / "chunk-00000.npz").write_bytes(b"npz_content")
    (res_dir / "debug.json").write_text("{}", encoding="utf-8")
    (res_dir / "views" / "d_geom.view.json").write_text("{}", encoding="utf-8")
    (res_dir / "worker_receipt.json").write_text("{}", encoding="utf-8")

    archive_prof = ArchiveProfileConfig().as_dict()
    ep_manifest = {
        "episode_id": "ep-comp-1",
        "attempt_id": "att-comp-1",
        "archive_profile": archive_prof,
        "timeline": {"observations": 32, "transitions": 31},
        "array_specs": {
            "online_points": {
                "shape": [65536, 3],
                "semantic_role": "online_observations/points",
            },
        },
        "local_write_limits": {
            "max_chunk_bytes": 268435456,
            "spool_peak_bytes": 500000,
            "local_peak_bytes_upper_bound": 1200000,
        },
    }
    (res_dir / "episode.manifest.json").write_text(json.dumps(ep_manifest), encoding="utf-8")
    (res_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")

    from types import SimpleNamespace

    class MockQueue:
        root = tmp_path / "queue"
        def counts(self):
            return SimpleNamespace(ready=1, pending=0, claimed=0)
        def iter_ready(self):
            return [
                WorkerResult(
                    job_id="j_compact",
                    attempt_id="att-comp-1",
                    episode_id="ep-comp-1",
                    program_id="T01",
                    outcome="success",
                    result_dir=str(res_dir),
                    file_sha256={str(p.relative_to(res_dir)): "0" * 64 for p in res_dir.rglob("*") if p.is_file()},
                    timeline={"observations": 32, "actions": 31, "durations": 31},
                )
            ]

    class GoodDisk:
        free = 10**9
    monkeypatch.setattr("icgs.data.collection.generation.capacity_probe.shutil.disk_usage", lambda _: GoodDisk())
    monkeypatch.setattr(generation_capacity_probe.shutil, "disk_usage", lambda _: GoodDisk())

    validated_profiles = []
    def mock_validate_closed(job, res, *, archive_profile=None):
        validated_profiles.append(archive_profile)
        return SimpleNamespace(outcome="success")

    monkeypatch.setattr(generation_capacity_probe, "validate_closed_result", mock_validate_closed)
    monkeypatch.setattr(generation_capacity_probe, "wait_for_ready_results", lambda *a, **kw: {"active_workers_at_completion": 0, "minimum_available_memory_bytes": 10**9})
    monkeypatch.setattr(generation_capacity_probe, "stop_processes", lambda _: None)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", lambda *a, **kw: mock_proc)

    job = _make_job("j_compact", "ep-comp-1")
    job_file = tmp_path / "queue" / "ready" / "j_compact" / "job.json"
    job_file.parent.mkdir(parents=True)
    job_file.write_text(json.dumps(job.as_dict()), encoding="utf-8")

    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path),
            "worker_count": 1,
            "hf_subfolder": "validation/test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)

    manifest_file = Path("artifacts/composition/approved_composition_manifest.json").resolve()

    monkeypatch.setattr(generation_capacity_probe, "FilesystemJobQueue", lambda _: MockQueue())
    monkeypatch.setattr(generation_capacity_probe, "enqueue_fixed_jobs", lambda *a: ["j_compact"])

    receipt = generation_capacity_probe.run_stage(
        base, probe, stage,
        output_root=tmp_path / "stage_compact_out",
        approved_manifest=manifest_file,
        code_revision="a" * 40,
        deadline=10**10,
    )
    assert receipt["status"] == "PASS"
    assert validated_profiles == [base.archive_profile]
    assert receipt["archive_profile"] == archive_prof
    assert receipt["staging_peak_semantic"] == "sampled_lower_bound"
    assert receipt["staging_sample_interval_s"] == 1.0
    assert "bytes_by_category" in receipt
    assert receipt["bytes_by_category"]["chunks"] > 0
    assert receipt["bytes_by_category"]["manifests"] > 0
    assert receipt["bytes_by_category"]["debug"] > 0
    assert receipt["bytes_by_category"]["views"] > 0
    assert receipt["bytes_by_category"]["receipts"] > 0

    assert len(receipt["results"]) == 1
    res0 = receipt["results"][0]
    assert res0["boundaries"] == 32
    assert res0["raw_points"] == 65536
    assert res0["writer_peak_bytes_upper_bound"] == 1200000
    assert res0["writer_peak_bytes_semantic"] == "archive_writer_reported_upper_bound"
    assert res0["archive_profile"] == archive_prof
    assert res0["bytes_by_category"]["chunks"] > 0
    assert res0["bytes_by_category"]["manifests"] > 0
    assert res0["bytes_by_category"]["debug"] > 0
    assert res0["bytes_by_category"]["views"] > 0
    assert res0["bytes_by_category"]["receipts"] > 0


def test_stage_tree_measurement_includes_hidden_writer_scratch(tmp_path):
    stage_root = tmp_path / "stage"
    result = stage_root / "worker-results" / "job-1" / "retry-0" / "T01"
    result.mkdir(parents=True)
    (result / "debug.json").write_bytes(b"{}")
    spool = result.parent / ".T01.archive-spool-orphan" / "chunk-00000" / "piece.npy"
    spool.parent.mkdir(parents=True)
    spool.write_bytes(b"captured-but-unarchived")

    measured = measure_staging_tree_bytes(stage_root)

    assert measured == {"bytes": 25, "writer_scratch_bytes": 23, "errors": []}


def test_capacity_stage_fails_when_sibling_writer_spool_exceeds_cap(
    tmp_path: Path, monkeypatch,
):
    import json
    import time
    from types import SimpleNamespace
    from icgs.data.collection.generation.distributed_contracts import WorkerResult

    stage = CapacityStage(
        name="sampled-cap-stage",
        worker_count=1,
        simulator_slots=1,
        max_jobs=1,
        max_result_bytes=100_000,
        max_staging_bytes=2_000_000,
    )
    probe = CapacityProbeConfig(
        run_id="sampled-cap",
        hf_subfolder="validation/sampled-cap",
        max_total_jobs=1,
        max_runtime_s=300,
        stages=(stage,),
        max_result_bytes=100_000,
        max_total_staging_bytes=3_000_000,
    )
    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path / "base-run"),
            "worker_count": 1,
            "hf_subfolder": "validation/sampled-cap",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)
    job = _make_job("job-tiny-result", "episode-tiny-result")
    result_dir_holder = {}
    queue_root_holder = {}

    class Queue:
        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True)
            queue_root_holder["root"] = self.root

        def counts(self):
            return SimpleNamespace(ready=1, pending=0, claimed=0)

        def iter_ready(self):
            return [WorkerResult(
                job_id=job.job_id,
                attempt_id=job.attempt_id,
                episode_id=job.episode_id,
                program_id=job.program_id,
                outcome="success",
                result_dir=str(result_dir_holder["path"]),
                file_sha256={"debug.json": "0" * 64},
                timeline={"observations": 1, "actions": 0, "durations": 0},
            )]

    class FinishedProcess:
        def poll(self):
            return 0

    class GoodDisk:
        free = 100_000_000

    def launch_worker(command, **kwargs):
        runtime_path = Path(command[command.index("--runtime-config") + 1])
        stage_root = Path(json.loads(runtime_path.read_text())["run"]["run_root"])
        result_dir = stage_root / "worker-results" / job.job_id / "retry-0" / "T01"
        result_dir.mkdir(parents=True)
        (result_dir / "debug.json").write_bytes(b"{}")
        result_dir_holder["path"] = result_dir
        job_path = queue_root_holder["root"] / "ready" / job.job_id / "job.json"
        job_path.parent.mkdir(parents=True)
        job_path.write_text(json.dumps(job.as_dict()), encoding="utf-8")
        orphan = result_dir.parent / ".T01.archive-spool-orphan" / "chunk-00000" / "piece.npy"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"x" * 2_000_001)
        return FinishedProcess()

    monkeypatch.setattr("icgs.data.collection.generation.capacity_probe.shutil.disk_usage", lambda _: GoodDisk())
    monkeypatch.setattr(generation_capacity_probe.shutil, "disk_usage", lambda _: GoodDisk())
    monkeypatch.setattr(generation_capacity_probe, "FilesystemJobQueue", Queue)
    monkeypatch.setattr(generation_capacity_probe, "enqueue_fixed_jobs", lambda *args: [job.job_id])
    monkeypatch.setattr(generation_capacity_probe.DistributedPlanner, "from_manifest", lambda *args: object())
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", launch_worker)
    monkeypatch.setattr(generation_capacity_probe, "_free_memory_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(
        generation_capacity_probe,
        "validate_closed_result",
        lambda *args, **kwargs: SimpleNamespace(outcome="success"),
    )

    receipt = generation_capacity_probe.run_stage(
        base,
        probe,
        stage,
        output_root=tmp_path / "capacity-output",
        approved_manifest=Path("artifacts/composition/approved_composition_manifest.json").resolve(),
        code_revision="a" * 40,
        deadline=time.monotonic() + 60,
    )

    assert receipt["artifact_bytes"] == 2
    assert receipt["stage_peak_bytes_sampled_lower_bound"] > stage.max_staging_bytes
    assert receipt["stage_writer_scratch_peak_bytes_sampled_lower_bound"] > stage.max_staging_bytes
    assert receipt["staging_peak_semantic"] == "sampled_lower_bound"
    assert receipt["status"] == "FAIL"
    assert any("SampledStageStagingCapExceeded" in row["error"] for row in receipt["invalid_results"])


def test_archive_profile_requires_explicit_caps_before_worker_launch(tmp_path, monkeypatch):
    stage = CapacityStage(
        name="arch_stage",
        worker_count=2,
        simulator_slots=1,
        max_jobs=2,
    )
    probe = CapacityProbeConfig(
        run_id="arch-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage,),
        # Omit all caps
    )
    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path),
            "worker_count": 2,
            "hf_subfolder": "validation/test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)

    spawned = []
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", lambda *a, **kw: spawned.append(a))
    manifest_file = Path("artifacts/composition/approved_composition_manifest.json").resolve()
    stage_dir = tmp_path / "stage_missing_caps" / stage.name

    with pytest.raises(ValueError, match="archive profile.*requires explicit positive"):
        generation_capacity_probe.run_stage(
            base, probe, stage,
            output_root=tmp_path / "stage_missing_caps",
            approved_manifest=manifest_file,
            code_revision="a" * 40,
            deadline=10**10,
        )
    assert len(spawned) == 0, "No workers should be launched when archive caps are missing"
    assert not stage_dir.exists(), "Stage directory should not be created if preflight fails"

    # Positive bounded fixture: with explicit positive caps and free disk, preflight passes
    valid_stage = CapacityStage(
        name="valid_stage",
        worker_count=2,
        simulator_slots=1,
        max_jobs=2,
        max_result_bytes=1000,
        max_staging_bytes=3000,
    )
    valid_probe = CapacityProbeConfig(
        run_id="valid-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(valid_stage,),
        max_result_bytes=1000,
        max_total_staging_bytes=5000,
    )

    class PlentyDisk:
        free = 10**8

    preflight_stage_capacity(
        valid_stage,
        valid_probe,
        tmp_path / "valid_stage_out",
        output_root=tmp_path,
        archive_profile=base.archive_profile.as_dict(),
        disk_usage_fn=lambda _: PlentyDisk(),
    )


def test_legacy_non_archive_no_cap_compatibility(tmp_path):
    stage = CapacityStage(
        name="legacy_stage",
        worker_count=2,
        simulator_slots=1,
        max_jobs=2,
    )
    probe = CapacityProbeConfig(
        run_id="legacy-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage,),
    )
    # Preflight should pass for legacy non-archive runs without caps
    preflight_stage_capacity(
        stage,
        probe,
        tmp_path / "legacy_stage_out",
        output_root=tmp_path,
        archive_profile=None,
    )


def test_capacity_probe_rejects_product_overflow():
    # 1. Stage product overflow: max_result_bytes * max_jobs > max_staging_bytes
    with pytest.raises(ValueError, match="exceeds stage max_staging_bytes"):
        CapacityStage(
            name="overflow_stage",
            worker_count=2,
            simulator_slots=1,
            max_jobs=2,
            max_result_bytes=100,
            max_staging_bytes=150,  # 2 * 100 = 200 > 150
        )

    # 2. Probe product overflow: eff_res * stage.max_jobs > max_total_staging_bytes
    stage = CapacityStage(
        name="s1",
        worker_count=2,
        simulator_slots=1,
        max_jobs=2,
    )
    with pytest.raises(ValueError, match="exceeds max_total_staging_bytes"):
        CapacityProbeConfig(
            run_id="probe-overflow",
            hf_subfolder="validation/test",
            max_total_jobs=5,
            max_runtime_s=300,
            stages=(stage,),
            max_result_bytes=100,
            max_total_staging_bytes=150,  # 2 * 100 = 200 > 150
        )

    # 3. Sum of stages overflow: stage 1 + stage 2 budgets exceed max_total_staging_bytes
    s1 = CapacityStage(name="s1", worker_count=1, simulator_slots=1, max_jobs=1, max_result_bytes=100, max_staging_bytes=100)
    s2 = CapacityStage(name="s2", worker_count=1, simulator_slots=1, max_jobs=1, max_result_bytes=100, max_staging_bytes=100)
    with pytest.raises(ValueError, match="exceeds max_total_staging_bytes"):
        CapacityProbeConfig(
            run_id="probe-sum-overflow",
            hf_subfolder="validation/test",
            max_total_jobs=5,
            max_runtime_s=300,
            stages=(s1, s2),
            max_total_staging_bytes=150,  # 100 + 100 = 200 > 150
        )


def test_preflight_accounts_for_prior_stage_retained_bytes(tmp_path):
    output_root = tmp_path / "multi_stage_run"
    prior_stage = output_root / "stage_1"
    prior_stage.mkdir(parents=True)
    (prior_stage / "chunk.npz").write_bytes(b"X" * 400)  # 400 bytes retained

    stage_2 = CapacityStage(
        name="stage_2",
        worker_count=2,
        simulator_slots=1,
        max_jobs=2,
        max_result_bytes=100,
        max_staging_bytes=200,
    )
    probe = CapacityProbeConfig(
        run_id="multi-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage_2,),
        max_result_bytes=100,
        max_total_staging_bytes=500,  # 400 retained + 200 stage = 600 > 500!
    )

    class GoodDisk:
        free = 10**9

    with pytest.raises(RuntimeError, match="retained bytes.*exceeds max_total_staging_bytes"):
        preflight_stage_capacity(
            stage_2,
            probe,
            output_root / "stage_2",
            output_root=output_root,
            disk_usage_fn=lambda _: GoodDisk(),
        )


def test_stage_preflight_blocks_worker_launch_on_insufficient_worst_case_reservation(tmp_path, monkeypatch):
    stage = CapacityStage(
        name="worst_case_stage",
        worker_count=2,
        simulator_slots=1,
        max_jobs=4,
        max_result_bytes=500,  # Worst case: 4 * 500 = 2000 bytes
        max_staging_bytes=2000,
    )
    probe = CapacityProbeConfig(
        run_id="cap-probe",
        hf_subfolder="validation/test",
        max_total_jobs=5,
        max_runtime_s=300,
        stages=(stage,),
        max_result_bytes=500,
        max_total_staging_bytes=5000,
    )
    base_payload = {
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 1,
            "display_width": 100,
            "display_height": 100,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path),
            "worker_count": 2,
            "hf_subfolder": "validation/test",
            "publication_enabled": False,
            "validation_mode": True,
        },
        "archive_profile": ArchiveProfileConfig().as_dict(),
    }
    base = GenerationRuntimeConfig.from_dict(base_payload)

    class LowDisk:
        free = 1500  # Only 1500 free, but stage worst case is 2000 bytes

    monkeypatch.setattr("icgs.data.collection.generation.capacity_probe.shutil.disk_usage", lambda _: LowDisk())
    monkeypatch.setattr(generation_capacity_probe.shutil, "disk_usage", lambda _: LowDisk())

    spawned = []
    monkeypatch.setattr(generation_capacity_probe.subprocess, "Popen", lambda *a, **kw: spawned.append(a))
    manifest_file = Path("artifacts/composition/approved_composition_manifest.json").resolve()

    with pytest.raises(RuntimeError, match="preflight capacity insufficient"):
        generation_capacity_probe.run_stage(
            base, probe, stage,
            output_root=tmp_path / "stage_worst_case_out",
            approved_manifest=manifest_file,
            code_revision="a" * 40,
            deadline=10**10,
        )
    assert len(spawned) == 0, "No workers should be launched when worst-case reservation exceeds free disk"
