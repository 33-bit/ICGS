from __future__ import annotations

import json
from pathlib import Path

import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue


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


def test_run_config_freezes_worker_count_and_publish_interval():
    run = RunConfig(
        run_id="run-1",
        run_root="/content/run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
    )
    assert run.worker_count == 200
    assert run.publish_interval_s == 300
    with pytest.raises(ValueError, match="worker_count must be 200"):
        RunConfig(
            run_id="run-1",
            run_root="/content/run",
            code_revision="a" * 40,
            approved_manifest_sha256="b" * 64,
            worker_count=199,
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


def test_stale_claim_requeues_without_changing_job_identity(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(_job())
    claimed = queue.claim("000", now_s=10.0)
    assert claimed is not None
    recovered = queue.recover_stale(now_s=71.0, stale_after_s=60.0)
    assert recovered == ["job-1"]
    reclaimed = queue.claim("001", now_s=72.0)
    assert reclaimed is not None
    assert reclaimed.attempt_id == claimed.attempt_id
    assert reclaimed.plan == claimed.plan


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
