from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    GenerationRuntimeConfig,
)
from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.episode_archive import (
    EpisodeArchiveReader,
    EpisodeArchiveWriter,
    validate_archive_manifest,
)
from scripts import generation_worker


def _runtime_config(tmp_path: Path, *, worker_timeout_s: int = 47):
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
            "display_base": 37,
            "display_width": 1280,
            "display_height": 720,
            "simulator_slots": 2,
            "worker_timeout_s": worker_timeout_s,
        },
        "run": {
            "run_id": "worker-test",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 91,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/worker-test",
            "publication_enabled": False,
            "validation_mode": True,
        },
    })


def test_simulator_slot_pool_uses_runtime_config(tmp_path, monkeypatch):
    config = _runtime_config(tmp_path)
    monkeypatch.setenv("ICGS_SIMULATOR_SLOTS", "99")
    try:
        pool = generation_worker.SimulatorSlotPool(tmp_path, config)
    except TypeError as exc:
        pytest.fail(f"SimulatorSlotPool must accept runtime config: {exc}")
    assert pool.slot_count == 2
    with pool:
        assert pool.acquired_slot is not None
        assert pool.acquired_slot.is_file()


def test_shared_filesystem_slot_pool_is_host_local(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["machine"].update({"host_id": "host-a", "worker_ids": ["000"], "simulator_slots": 1})
    payload["run"]["distribution_mode"] = "shared_filesystem"
    host_config = GenerationRuntimeConfig.from_dict(payload)

    pool = generation_worker.SimulatorSlotPool(tmp_path, host_config)

    assert pool.root == tmp_path / "control" / "simulator-slots" / "host-a"


def test_worker_source_has_no_hf_token_or_api_access():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert ".icgs_hf_token" not in text
    assert "HfApi" not in text
    assert "huggingface_hub" not in text


def test_worker_display_number_uses_configured_base_and_worker_limit(tmp_path: Path):
    config = _runtime_config(tmp_path)
    try:
        assert generation_worker.display_number("001", config) == 38
    except TypeError as exc:
        pytest.fail(f"display_number must accept runtime config: {exc}")
    with pytest.raises(ValueError, match="worker_id"):
        generation_worker.display_number("002", config)


def test_worker_rejects_id_outside_host_scope_before_claim(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["machine"].update({"host_id": "host-a", "worker_ids": ["001"], "simulator_slots": 1})
    payload["run"]["distribution_mode"] = "shared_filesystem"
    host_config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(host_config.run.run_root) / "queue")

    with pytest.raises(ValueError, match="host worker scope"):
        generation_worker.run_worker(
            "000", queue, config=host_config,
            approved_manifest=str(tmp_path / "approved.json"), once=True,
        )


def test_worker_cli_requires_runtime_config(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "generation_worker.py",
        "--worker-id", "000",
        "--approved-manifest", "approved.json",
        "--once",
    ])

    with pytest.raises(SystemExit) as exc:
        generation_worker.main()

    assert exc.value.code == 2
    assert "--runtime-config" in capsys.readouterr().err


def test_worker_subprocess_uses_configured_paths_timeout_and_slots(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, worker_timeout_s=33)
    monkeypatch.setenv("ICGS_HF_TOKEN_PATH", "/secret/token")
    monkeypatch.setenv("HF_TOKEN", "secret")
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-1", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    try:
        generation_worker.run_worker(
            "000",
            queue,
            config=config,
            approved_manifest=str(approved_manifest),
            once=True,
            runner=runner,
        )
    except TypeError as exc:
        pytest.fail(f"run_worker must accept runtime config and injected runner: {exc}")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        config.machine.python_executable,
        "-B",
        str(Path(config.machine.repo_root) / "scripts" / "generation_episode_worker.py"),
        "T01",
    ]
    assert kwargs["timeout"] == 33
    assert kwargs["env"]["ICGS_SIMULATOR_SLOTS"] == "2"
    assert "ICGS_HF_TOKEN_PATH" not in kwargs["env"]
    assert "HF_TOKEN" not in kwargs["env"]
    assert queue.counts().ready == 1


def test_worker_stages_recovered_retry_in_a_disjoint_result_directory(tmp_path: Path):
    config = _runtime_config(tmp_path)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-retry", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"), retry_generation=1,
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest), once=True,
        runner=lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="failed", stderr=""),
    )

    result = json.loads((queue.root / "ready" / "job-retry" / "result.json").read_text())
    assert "/worker-results/job-retry/retry-1/T01" in result["result_dir"]


def test_archive_result_detection_requires_canonical_manifest_without_legacy_fallback(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["archive_profile"] = ArchiveProfileConfig(chunk_boundaries=2).as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "episode.manifest.json").write_text(json.dumps({
        "archive_format_id": config.archive_profile.archive_format_id,
        "episode_schema_version": config.archive_profile.episode_schema_version,
        "dataset_identity": config.archive_profile.dataset_identity,
        "archive_kind": "episode",
        "outcome": "valid_failure",
        "timeline": {"observations": 2, "transitions": 1},
    }), encoding="utf-8")

    detected = generation_worker._archive_result_payload(candidate, config)
    assert detected["archive_kind"] == "episode"
    assert detected["outcome"] == "valid_failure"

    (candidate / "episode.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        generation_worker._archive_result_payload(candidate, config)


def test_archive_profile_worker_propagates_identity_and_writes_canonical_failure(tmp_path: Path):
    base = _runtime_config(tmp_path)
    payload = base.as_dict()
    payload["archive_profile"] = ArchiveProfileConfig(chunk_boundaries=2).as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-archive-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920, randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-archive", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=1, stdout="worker stdout", stderr="worker stderr")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest), once=True, runner=runner
    )

    assert calls[0]["env"]["ICGS_GENERATION_RUN_ID"] == job.run_id
    assert calls[0]["env"]["ICGS_GENERATION_CODE_REVISION"] == job.code_revision
    assert calls[0]["env"]["ICGS_GENERATION_ARCHIVE_PROFILE"]
    result = json.loads((queue.root / "ready" / job.job_id / "result.json").read_text())
    assert Path(result["result_dir"], "attempt.manifest.json").is_file()
    assert not Path(result["result_dir"], "attempt.json").exists()


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_worker_preserves_closed_attempt_when_runner_raises(
    tmp_path: Path, outcome: str
):
    base = _runtime_config(tmp_path)
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    payload = base.as_dict()
    payload["archive_profile"] = profile.as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-closed-attempt", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-closed", "asset_instance_id": "asset-closed"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-closed-attempt", run_id=config.run.run_id,
        attempt_id=f"att-{plan.episode_id}", episode_id=plan.episode_id, program_id="T01",
        plan=plan, code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    measured_actions = np.asarray([[0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    measured_points = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    def runner(command, **kwargs):
        candidate = Path(kwargs["env"]["ICGS_GENERATION_WRITE_EPISODE"]) / job.program_id
        EpisodeArchiveWriter(profile).write_attempt(
            {
                "attempt_id": job.attempt_id,
                "episode_id": None,
                "program_id": job.program_id,
                "outcome": outcome,
                "valid_observation_until": 1,
            },
            prefix_arrays={
                "actions": measured_actions,
                "points": measured_points,
                "point_offsets": np.asarray([0, 1, 2], dtype=np.int64),
                "T_w_e": np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
            },
            debug_metadata={
                "source_run_id": job.run_id,
                "code_revision": job.code_revision,
                "preprocessing_identity": "runner_closed_attempt_v1",
                "stdout": "attempt archive closed before the runner exception",
            },
            output_dir=candidate,
        )
        raise subprocess.TimeoutExpired(command, 33, output="timed out after close")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest),
        once=True, runner=runner,
    )

    result = json.loads((queue.root / "ready" / job.job_id / "result.json").read_text())
    result_dir = Path(result["result_dir"])
    manifest_path = result_dir / "attempt.manifest.json"
    assert result["outcome"] == outcome
    assert manifest_path.is_file()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    archive = EpisodeArchiveReader(manifest_path)
    np.testing.assert_array_equal(archive.raw_arrays["actions"], measured_actions)
    np.testing.assert_array_equal(archive.raw_arrays["points"], measured_points)
    assert archive.debug_metadata["stdout"] == "attempt archive closed before the runner exception"


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_worker_keeps_closed_attempt_when_publication_fails(
    tmp_path: Path, monkeypatch, outcome: str
):
    base = _runtime_config(tmp_path)
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    payload = base.as_dict()
    payload["archive_profile"] = profile.as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-publish-failure", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-publish", "asset_instance_id": "asset-publish"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-publish-failure", run_id=config.run.run_id,
        attempt_id=f"att-{plan.episode_id}", episode_id=plan.episode_id, program_id="T01",
        plan=plan, code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    measured_actions = np.asarray([[0.25, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    measured_points = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    def runner(command, **kwargs):
        candidate = Path(kwargs["env"]["ICGS_GENERATION_WRITE_EPISODE"]) / job.program_id
        EpisodeArchiveWriter(profile).write_attempt(
            {
                "attempt_id": job.attempt_id,
                "episode_id": None,
                "program_id": job.program_id,
                "outcome": outcome,
                "valid_observation_until": 0,
            },
            prefix_arrays={
                "actions": measured_actions,
                "points": measured_points,
                "point_offsets": np.asarray([0, 1], dtype=np.int64),
                "T_w_e": np.eye(4, dtype=np.float64)[None],
            },
            debug_metadata={
                "source_run_id": job.run_id,
                "code_revision": job.code_revision,
                "preprocessing_identity": "runner_publish_failure_v1",
                "traceback": "original closed-attempt traceback",
            },
            output_dir=candidate,
        )
        raise subprocess.TimeoutExpired(command, 33, output="timed out after close")

    def fail_publish(*args, **kwargs):
        raise RuntimeError("publish unavailable")

    monkeypatch.setattr(queue, "publish_ready", fail_publish)

    with pytest.raises(RuntimeError, match="publish unavailable"):
        generation_worker.run_worker(
            "000", queue, config=config, approved_manifest=str(approved_manifest),
            once=True, runner=runner,
        )

    candidate = Path(job.output_root) / "worker-results" / job.job_id / "retry-0" / job.program_id
    manifest_path = candidate / "attempt.manifest.json"
    assert manifest_path.is_file()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    archive = EpisodeArchiveReader(manifest_path)
    assert archive.manifest.payload["outcome"] == outcome
    np.testing.assert_array_equal(archive.raw_arrays["actions"], measured_actions)
    assert archive.debug_metadata["traceback"] == "original closed-attempt traceback"


def test_generation_subprocess_refreshes_heartbeat_until_exit(monkeypatch):
    events = []

    class Process:
        def __init__(self):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def communicate(self):
            return "stdout", "stderr"

        def kill(self):
            self.poll_count = 2

    process = Process()
    monkeypatch.setattr(generation_worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(generation_worker.time, "sleep", lambda _: None)

    result = generation_worker._run_generation_process(
        ["python", "episode.py"], env={}, timeout_s=10,
        heartbeat=lambda: events.append("heartbeat"), heartbeat_interval_s=1,
    )

    assert result.returncode == 0
    assert len(events) >= 2


def test_worker_invalid_result_hashes_the_candidate_directory():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert "file_sha256=_file_hashes(candidate)" in text
    assert "file_sha256=_file_hashes(result_dir)" in text


def test_worker_exception_path_does_not_publish_empty_inventory():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert "file_sha256=_file_hashes(result_dir)" in text
    assert "stale candidate" in text
