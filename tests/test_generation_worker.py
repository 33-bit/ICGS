from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    GenerationRuntimeConfig,
)
from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
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
