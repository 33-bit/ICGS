from __future__ import annotations

from dataclasses import replace
import fcntl
import json
import multiprocessing as multiprocessing
from pathlib import Path
import queue as queue_module
import time

import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation import distributed_validation as validation


def _register_worker_blocked_by_queue_lock(root: str, results) -> None:
    queue = FilesystemJobQueue(root)
    results.put("started")
    try:
        queue.register_worker("000", "host-a", "instance-a", now_s=10.0, lease_s=120.0)
    except Exception as error:  # pragma: no cover - assertion reports the child error
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put("done")


def _plan(index: int = 17, kind: str = "perturbed") -> AttemptPlan:
    return AttemptPlan(
        program_id="T01",
        split="train",
        episode_index=index,
        episode_id=f"episode-t01-{index:06d}",
        episode_kind=kind,
        scene_seed=1234 + index,
        collection_seed=20260920,
        randomization={
            "scene_signature": f"signature-{index}",
            "asset_instance_id": f"asset-{index}",
        },
        intervention=None if kind == "nominal" else {
            "kind": "pause_hold",
            "intervention_id": "pause_hold_v1",
        },
    )


def _job(job_id: str = "job-1", index: int = 17) -> GenerationJob:
    return GenerationJob.create(
        job_id=job_id,
        run_id="run-1",
        attempt_id=f"att-{_plan(index).episode_id}",
        episode_id=_plan(index).episode_id,
        program_id="T01",
        plan=_plan(index),
        code_revision="a" * 40,
        manifest_sha256="b" * 64,
        output_root="/content/run/staging",
    )


def _result(job: GenerationJob, result_dir: Path) -> WorkerResult:
    return WorkerResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        episode_id=job.episode_id,
        program_id=job.program_id,
        outcome="success",
        result_dir=str(result_dir),
        file_sha256={},
        timeline={"actions": 0, "observations": 1, "durations": 0},
    )


def _failure(
    job: GenerationJob,
    result: WorkerResult,
    source_path: Path,
    message: str = "bad result",
    *,
    allowed_result_root: Path | None = None,
):
    assert hasattr(validation, "ValidationFailure"), "ValidationFailure is required"
    failure_type = validation.ValidationFailure
    try:
        raise ValueError(message)
    except ValueError as exc:
        return failure_type.capture(
            job,
            result,
            exc,
            source_path=source_path,
            allowed_result_root=allowed_result_root,
        )


def test_run_config_accepts_configurable_worker_count_and_publish_interval():
    run = RunConfig(
        run_id="run-1",
        run_root="/content/run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        worker_count=2,
        publish_interval_s=45,
    )
    assert run.worker_count == 2
    assert run.publish_interval_s == 45
    with pytest.raises(ValueError, match="worker_count must be a positive integer"):
        RunConfig(
            run_id="run-1",
            run_root="/content/run",
            code_revision="a" * 40,
            approved_manifest_sha256="b" * 64,
            worker_count=0,
        )


def test_job_roundtrip_preserves_exact_attempt_plan():
    job = _job()
    restored = GenerationJob.from_dict(job.as_dict())
    assert restored == job
    assert restored.plan == job.plan


def test_job_rejects_attempt_identity_that_disagrees_with_plan():
    with pytest.raises(ValueError, match="attempt_id must match"):
        GenerationJob.create(
            job_id="job-bad", run_id="run-1", attempt_id="different",
            episode_id=_plan().episode_id, program_id="T01", plan=_plan(),
            code_revision="a" * 40, manifest_sha256="b" * 64,
            output_root="/content/run/staging",
        )


def test_attempt_outcome_requires_null_episode_id():
    with pytest.raises(ValueError, match="episode_id must be null"):
        WorkerResult(
            job_id="job-1",
            attempt_id="attempt-1",
            episode_id="episode-1",
            program_id="T01",
            outcome="simulator_crash",
            result_dir="/content/result",
            file_sha256={},
            timeline=None,
        )


def test_two_workers_cannot_claim_same_job(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())
    first = queue.claim("000", now_s=10.0)
    second = queue.claim("001", now_s=10.0)
    assert first is not None and first.job_id == "job-1"
    assert second is None
    assert queue.counts().claimed == 1


def test_active_worker_lease_rejects_duplicate_host_instance(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())

    queue.register_worker("000", "host-a", "instance-a", now_s=10.0, lease_s=120.0)
    with pytest.raises(RuntimeError, match="worker lease"):
        queue.register_worker("000", "host-b", "instance-b", now_s=20.0, lease_s=120.0)


def test_worker_lease_registration_serializes_with_other_queue_operations(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    lock_path = queue.root / "control" / "queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
    process = context.Process(
        target=_register_worker_blocked_by_queue_lock,
        args=(str(queue.root), results),
    )
    process.start()
    assert results.get(timeout=5.0) == "started"
    with pytest.raises(queue_module.Empty):
        results.get(timeout=0.2)
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    lock_stream.close()
    assert results.get(timeout=5.0) == "done"
    process.join(timeout=5.0)
    assert process.exitcode == 0


def test_expired_worker_lease_can_be_replaced(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.register_worker("000", "host-a", "instance-a", now_s=10.0, lease_s=20.0)

    queue.register_worker("000", "host-b", "instance-b", now_s=31.0, lease_s=20.0)
    heartbeat = json.loads((queue.root / "heartbeats" / "worker-000.json").read_text())
    assert heartbeat["host_id"] == "host-b"
    assert heartbeat["worker_instance_id"] == "instance-b"


def test_stale_claim_requeues_without_changing_job_identity(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())
    claimed = queue.claim("000", now_s=10.0)
    assert claimed is not None
    recovered = queue.recover_stale(now_s=71.0, stale_after_s=60.0)
    assert recovered == ["job-1"]
    recovered_payload = json.loads((queue.root / "pending" / "job-1.json").read_text())
    assert recovered_payload["retry_generation"] == 1
    reclaimed = queue.claim("001", now_s=72.0)
    assert reclaimed is not None
    assert reclaimed.attempt_id == claimed.attempt_id
    assert reclaimed.plan == claimed.plan


def test_fresh_worker_heartbeat_prevents_stale_claim_recovery(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())
    claimed = queue.claim("000", now_s=10.0)
    assert claimed is not None
    queue.register_worker("000", "host-a", "instance-a", now_s=10.0, lease_s=120.0)
    queue.write_heartbeat(
        "000", claimed.job_id, now_s=65.0, host_id="host-a",
        worker_instance_id="instance-a", lease_s=120.0,
    )

    assert queue.recover_stale(now_s=71.0, stale_after_s=60.0) == []


def test_expired_worker_lease_allows_claim_recovery_even_before_stale_timeout(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())
    claimed = queue.claim(
        "000", now_s=10.0, host_id="host-a", worker_instance_id="instance-a", lease_s=20.0,
    )
    assert claimed is not None
    queue.write_heartbeat(
        "000", claimed.job_id, now_s=10.0, host_id="host-a",
        worker_instance_id="instance-a", lease_s=20.0,
    )

    assert queue.recover_stale(now_s=31.0, stale_after_s=60.0) == ["job-1"]


def test_late_result_from_replaced_worker_lease_is_fenced(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    job = _job(job_id="job-fenced")
    queue.enqueue(job)
    claimed = queue.claim(
        "000", now_s=10.0, host_id="host-a", worker_instance_id="instance-a", lease_s=20.0,
    )
    assert claimed is not None
    queue.recover_stale(now_s=31.0, stale_after_s=10.0)
    queue.register_worker("000", "host-b", "instance-b", now_s=31.0, lease_s=120.0)
    with pytest.raises(RuntimeError, match="another instance|expired"):
        queue.publish_ready(
            "000", _result(job, tmp_path / "result"), worker_instance_id="instance-a",
        )


def test_ready_ingested_published_state_machine(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    job = _job()
    queue.enqueue(job)
    assert queue.claim("000") == job
    result = WorkerResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        episode_id=job.episode_id,
        program_id=job.program_id,
        outcome="valid_failure",
        result_dir="/content/result",
        file_sha256={"episode.json": "c" * 64},
        timeline={"actions": 3, "observations": 4, "durations": 3},
    )
    queue.publish_ready("000", result)
    assert queue.iter_ready() == (result,)
    queue.mark_ingested(result)
    assert queue.counts().ingested == 1
    queue.mark_published(job.job_id)
    counts = queue.counts()
    assert counts.published == 1
    assert counts.ingested == 0


def test_iter_ready_ignores_atomic_partial_directories(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    job = _job(job_id="job-closed")
    result = WorkerResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        episode_id=job.episode_id,
        program_id=job.program_id,
        outcome="valid_failure",
        result_dir="/content/result",
        file_sha256={"episode.json": "c" * 64},
        timeline={"actions": 3, "observations": 4, "durations": 3},
    )

    partial = queue.root / "ready" / "job-in-flight.partial-123-456"
    partial.mkdir(parents=True)
    (partial / "job.json").write_text(json.dumps(job.as_dict()), encoding="utf-8")
    closed = queue.root / "ready" / job.job_id
    closed.mkdir()
    (closed / "result.json").write_text(json.dumps(result.as_dict()), encoding="utf-8")

    assert queue.iter_ready() == (result,)


def test_quarantine_ready_moves_result_writes_diagnostic_and_counts_state(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job(job_id="job-quarantined")
    result_dir = tmp_path / "artifacts" / job.job_id
    result_dir.mkdir(parents=True)
    (result_dir / "broken.bin").write_bytes(b"corrupt payload")
    queue.enqueue(job)
    assert queue.claim("000") == job
    result = _result(job, result_dir)
    ready_path = queue.publish_ready("000", result)
    failure = _failure(job, result, ready_path, allowed_result_root=tmp_path)

    quarantined = queue.quarantine_ready(job.job_id, failure)

    assert quarantined == queue.root / "quarantined" / job.job_id
    assert not ready_path.exists()
    assert (quarantined / "job.json").is_file()
    assert (quarantined / "result.json").is_file()
    diagnostic = json.loads((quarantined / "validation_failure.json").read_text(encoding="utf-8"))
    assert diagnostic["job_identity"]["job_id"] == job.job_id
    assert diagnostic["result_identity"]["outcome"] == "success"
    assert diagnostic["source_path"] == str(ready_path)
    assert diagnostic["actual_file_inventory"]["broken.bin"]["bytes"] == len(b"corrupt payload")
    assert queue.iter_ready() == ()
    assert queue._find_job("quarantined", job.job_id) == quarantined
    assert queue.enqueue(job) == quarantined
    counts = queue.counts()
    assert hasattr(counts, "quarantined")
    assert counts.quarantined == 1
    assert counts.ready == 0


def test_quarantine_ready_is_idempotent_for_equal_failure_and_rejects_conflicts(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job(job_id="job-idempotent")
    result_dir = tmp_path / "artifacts"
    result_dir.mkdir()
    queue.enqueue(job)
    assert queue.claim("000") == job
    result = _result(job, result_dir)
    ready_path = queue.publish_ready("000", result)
    failure = _failure(job, result, ready_path, allowed_result_root=tmp_path)

    first = queue.quarantine_ready(job.job_id, failure)
    repeated = queue.quarantine_ready(job.job_id, failure)

    assert repeated == first
    conflicting = replace(failure, exception_message="different validation error")
    with pytest.raises(ValueError, match="quarantine conflict"):
        queue.quarantine_ready(job.job_id, conflicting)


def test_quarantine_ready_rejects_path_traversal_job_ids(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job()
    result = _result(job, tmp_path / "missing-result")
    failure = _failure(job, result, queue.root / "ready" / job.job_id)

    with pytest.raises(ValueError, match="job_id"):
        queue.quarantine_ready("../escape", failure)

    assert not (queue.root.parent / "escape").exists()


def test_quarantine_ready_rejects_dangling_symlink_target_without_moving_ready(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job(job_id="job-symlink-target")
    result_dir = tmp_path / "artifacts"
    result_dir.mkdir()
    queue.enqueue(job)
    assert queue.claim("000") == job
    result = _result(job, result_dir)
    ready_path = queue.publish_ready("000", result)
    failure = _failure(job, result, ready_path, allowed_result_root=tmp_path)
    target = queue.root / "quarantined" / job.job_id
    target.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(ValueError, match="quarantine target"):
        queue.quarantine_ready(job.job_id, failure)

    assert ready_path.is_dir()
    assert target.is_symlink()


def test_claimed_job_lookup_treats_glob_characters_as_literal(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job(job_id="job-one")
    queue.enqueue(job)
    assert queue.claim("worker") == job

    assert queue._find_job("claimed", "job-*") is None
