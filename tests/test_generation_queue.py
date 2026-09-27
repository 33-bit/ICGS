from __future__ import annotations

from dataclasses import replace
import fcntl
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import queue as queue_module
import stat
import time

import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation import distributed_queue
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


def test_run_safety_stop_forbids_new_claims_and_ready_publication(tmp_path: Path):
    run_root = tmp_path / "run"
    queue = FilesystemJobQueue(run_root / "queue")
    job = _job(job_id="job-stop")
    queue.enqueue(job)
    claimed = queue.claim(
        "000", host_id="host-a", worker_instance_id="instance-a", lease_s=120.0,
    )
    assert claimed == job
    queue.enqueue(_job(job_id="job-pending"))

    queue.trip_safety_stop({
        "reason": "orphan_archive_scratch_after_timeout",
        "recovery_action": "Preserve the scratch files until HF fate is verified.",
    })
    queue.trip_safety_stop({
        "reason": "orphan_archive_scratch_after_worker_error",
        "job_id": "job-concurrent-orphan",
        "recovery_action": "Preserve the additional scratch files until HF fate is verified.",
        "preserved_files": [{"path": "worker-results/job-concurrent/piece.npy", "bytes": 4, "kind": "file"}],
    })

    with pytest.raises(RuntimeError, match="generation safety stop"):
        queue.publish_ready(
            "000", _result(job, tmp_path / "result"), worker_instance_id="instance-a",
        )
    assert not list((queue.root / "ready").iterdir())
    receipt = json.loads(queue.safety_stop_path.read_text(encoding="utf-8"))
    assert receipt["incident_count"] == 2
    assert receipt["preserved_bytes"] == 4
    assert receipt["preserved_files"][0]["path"] == "worker-results/job-concurrent/piece.npy"
    with pytest.raises(RuntimeError, match="generation safety stop"):
        queue.claim("001")
    assert queue.counts().pending == 1


def test_safety_stop_create_and_incident_update_fsync_directory_entries(tmp_path: Path, monkeypatch):
    # Catches a marker whose JSON data is synced but whose rename can vanish after a crash.
    queue = FilesystemJobQueue(tmp_path / "run" / "queue")
    marker = queue.safety_stop_path
    original_fsync = os.fsync
    synced = []

    def observe_fsync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            for directory in (marker.parent, marker.parent.parent):
                if directory.is_dir():
                    target = directory.stat()
                    if (target.st_dev, target.st_ino) == (info.st_dev, info.st_ino):
                        synced.append((directory, json.loads(marker.read_text())["incident_count"]))
        return original_fsync(fd)

    monkeypatch.setattr(distributed_queue.os, "fsync", observe_fsync)
    queue.trip_safety_stop({"reason": "first", "recovery_action": "Inspect first incident."})
    assert (marker.parent, 1) in synced
    assert (marker.parent.parent, 1) in synced
    queue.trip_safety_stop({"reason": "second", "recovery_action": "Inspect second incident."})
    assert (marker.parent, 2) in synced


def test_claim_discovery_stops_on_abandoned_archive_scratch(tmp_path: Path):
    run_root = tmp_path / "run"
    queue = FilesystemJobQueue(run_root / "queue")
    job = replace(_job(job_id="job-abandoned"), output_root=str(run_root / "staging"))
    queue.enqueue(job)
    orphan = (
        run_root / "staging" / "worker-results" / job.job_id / "retry-0"
        / ".T01.archive-spool-abandoned" / "chunk-00000" / "piece.npy"
    )
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"unarchived")

    with pytest.raises(RuntimeError, match="generation safety stop"):
        queue.claim("000")

    assert orphan.read_bytes() == b"unarchived"
    assert queue.counts().pending == 1
    receipt = json.loads(queue.safety_stop_path.read_text(encoding="utf-8"))
    assert receipt["reason"] == "discoverable_orphan_archive_scratch"
    assert receipt["preserved_bytes"] == len(b"unarchived")


def test_claim_discovery_allows_scratch_for_a_live_claim(tmp_path: Path):
    run_root = tmp_path / "run"
    queue = FilesystemJobQueue(run_root / "queue")
    first = replace(_job(job_id="job-live"), output_root=str(run_root / "staging"))
    second = replace(_job(job_id="job-next", index=18), output_root=str(run_root / "staging"))
    queue.enqueue(first)
    assert queue.claim(
        "000", host_id="host-a", worker_instance_id="instance-a", lease_s=120.0,
    ) == first
    queue.write_heartbeat(
        "000", first.job_id, host_id="host-a", worker_instance_id="instance-a", lease_s=120.0,
    )
    scratch = (
        run_root / "staging" / "worker-results" / first.job_id / "retry-0"
        / ".T01.archive-spool-live" / "piece.npy"
    )
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"active")
    queue.enqueue(second)

    assert queue.claim("001", host_id="host-b", worker_instance_id="instance-b") == second
    assert not queue.safety_stop_path.exists()


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


def _receipt_only_queue_fixture(tmp_path: Path, *, job_id: str = "job-receipt-only"):
    import hashlib

    run_root = tmp_path / "run"
    queue = FilesystemJobQueue(run_root / "queue")
    job = _job(job_id=job_id)
    result_dir = run_root / "results" / job_id
    result_dir.mkdir(parents=True)
    (result_dir / "episode_manifest.json").write_bytes(b"canonical archive manifest\n")
    relative = "episode_manifest.json"
    digest = hashlib.sha256((result_dir / relative).read_bytes()).hexdigest()
    result = replace(
        _result(job, result_dir),
        file_sha256={relative: digest},
    )
    queue.enqueue(job)
    assert queue.claim("000") == job
    queue.publish_ready("000", result)
    queue.mark_ingested(result)
    return queue, job, result, result_dir


def _write_verified_archive_receipt(queue, job, result, *, wrong_artifact_hash: bool = False):
    import hashlib
    from icgs.data.collection.generation.diversity import train_subset_for_sample

    profile = {
        "dataset_identity": "icgs-primary-v3-archive-v1",
        "archive_format_id": "icgs_npz_chunked_v1",
        "episode_schema_version": "icgs_episode_archive_v1",
        "local_artifact_retention": "receipt_only",
    }
    row = {
        "archive_ref": f"episodes/{job.program_id}/{job.episode_id}",
        "archive_manifest": (
            f"episodes/{job.program_id}/{job.episode_id}/episode.manifest.json"
        ),
        "archive_format_id": profile["archive_format_id"],
        "episode_schema_version": profile["episode_schema_version"],
        "dataset_identity": profile["dataset_identity"],
        "job_id": job.job_id,
        "run_id": job.run_id,
        "manifest_sha256": job.manifest_sha256,
        "retry_generation": job.retry_generation,
        "source_run_id": job.run_id,
        "code_revision": job.code_revision,
        "preprocessing_identity": "receipt_only_queue_fixture_v1",
        "attempt_id": job.attempt_id,
        "episode_id": job.episode_id,
        "program_id": job.program_id,
        "split": "dev" if job.plan.split == "development" else job.plan.split,
        "subset": (
            train_subset_for_sample(job.plan.randomization)
            if job.plan.split == "train" else None
        ),
        "episode_kind": job.plan.episode_kind,
        "scene_signature": job.plan.randomization.get("scene_signature"),
        "scene_seed": job.plan.scene_seed,
        "episode_index": job.plan.episode_index,
        "asset_instance_id": job.plan.randomization.get("asset_instance_id"),
        "asset_family_id": job.plan.randomization.get("asset_family_id"),
        "source_lineage_id": "lineage-1",
        "intervention_id": (job.plan.intervention or {}).get("intervention_id"),
        "source_episode_id": (job.plan.intervention or {}).get("source_episode_id"),
        "base_episode_id": (job.plan.intervention or {}).get("base_episode_id"),
        "outcome": result.outcome,
        "attempt_plan": job.plan.as_dict(),
        "file_sha256": dict(result.file_sha256),
    }
    manifest = {
        "manifest_version": 3,
        "source_run_ids": [job.run_id],
        "archive_profile": profile,
        "dataset_identity": profile["dataset_identity"],
        "archive_format_id": profile["archive_format_id"],
        "episode_schema_version": profile["episode_schema_version"],
        "episodes": [row],
        "failure_attempts": [],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (queue.root / "publication_manifest.json").write_bytes(manifest_bytes)
    artifact_hash = "0" * 64 if wrong_artifact_hash else result.file_sha256["episode_manifest.json"]
    receipt = {
        "status": "VERIFIED",
        "run_id": job.run_id,
        "source_run_id": job.run_id,
        "job_ids": [job.job_id],
        "data_commit_oid": "d" * 40,
        "commit_oid": "c" * 40,
        "artifact_hashes": {f"{job.job_id}/episode_manifest.json": artifact_hash},
        **profile,
        "archive_profile": profile,
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    (queue.root / "publication_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8",
    )


def test_receipt_only_retention_refuses_to_prune_without_verified_remote_receipt(
    tmp_path: Path,
):
    queue, job, result, result_dir = _receipt_only_queue_fixture(tmp_path)

    with pytest.raises(ValueError, match="verified publication receipt"):
        queue.mark_published(job.job_id, retention="receipt_only")

    assert (result_dir / "episode_manifest.json").is_file()
    assert (queue.root / "ingested" / job.job_id / "result.json").is_file()


def test_receipt_only_retention_keeps_job_receipt_and_prunes_only_verified_payload(
    tmp_path: Path,
):
    queue, job, result, result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, result)

    published = queue.mark_published(job.job_id, retention="receipt_only")

    assert published == queue.root / "published" / job.job_id
    assert not result_dir.exists()
    per_job = json.loads((published / "publication_receipt.json").read_text(encoding="utf-8"))
    assert per_job["source_run_id"] == job.run_id
    assert per_job["commit_oid"] == "c" * 40
    assert per_job["artifact_hashes"] == {
        "episode_manifest.json": result.file_sha256["episode_manifest.json"]
    }
    assert per_job["manifest_row"]["archive_ref"] == "episodes/T01/episode-t01-000017"
    assert (published / "job.json").is_file()
    assert not (published / "episode_manifest.json").exists()
    assert queue.mark_published(job.job_id, retention="receipt_only") == published


def test_receipt_only_retention_rejects_receipt_hash_mismatch_without_pruning(
    tmp_path: Path,
):
    queue, job, result, result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, result, wrong_artifact_hash=True)

    with pytest.raises(ValueError, match="artifact hash|verified publication receipt"):
        queue.mark_published(job.job_id, retention="receipt_only")

    assert (result_dir / "episode_manifest.json").is_file()


def test_receipt_only_retention_never_prunes_payload_outside_run_root(tmp_path: Path):
    queue, job, result, _result_dir = _receipt_only_queue_fixture(tmp_path)
    outside_root = tmp_path / "outside-run"
    outside_root.mkdir()
    outside_file = outside_root / "episode_manifest.json"
    outside_file.write_bytes(b"canonical archive manifest\n")
    _write_verified_archive_receipt(queue, job, result)
    stored_result_path = queue.root / "ingested" / job.job_id / "result.json"
    stored_result = json.loads(stored_result_path.read_text(encoding="utf-8"))
    stored_result["result_dir"] = str(outside_root)
    stored_result_path.write_text(json.dumps(stored_result), encoding="utf-8")

    with pytest.raises(ValueError, match="beneath the run root"):
        queue.mark_published(job.job_id, retention="receipt_only")

    assert outside_file.is_file()
    assert (queue.root / "ingested" / job.job_id / "result.json").is_file()


def test_receipt_only_existing_job_receipt_rejects_missing_result_path_outside_run_root(
    tmp_path: Path,
):
    queue, job, result, _result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, result)
    source = queue.root / "ingested" / job.job_id
    _stored_job, _stored_result, per_job_receipt = queue._receipt_only_binding(
        job.job_id, source,
    )
    (source / "publication_receipt.json").write_text(
        json.dumps(per_job_receipt), encoding="utf-8",
    )
    stored_result_path = source / "result.json"
    stored_result = json.loads(stored_result_path.read_text(encoding="utf-8"))
    outside_missing_result = tmp_path / "outside-run" / "already-pruned-result"
    stored_result["result_dir"] = str(outside_missing_result)
    stored_result_path.write_text(json.dumps(stored_result), encoding="utf-8")

    with pytest.raises(ValueError, match="run root"):
        queue.mark_published(job.job_id, retention="receipt_only")

    assert source.is_dir()
    assert not (queue.root / "published" / job.job_id).exists()


@pytest.mark.parametrize(
    "field",
    [
        "attempt_plan",
        "split",
        "subset",
        "episode_kind",
        "scene_signature",
        "scene_seed",
        "episode_index",
        "asset_instance_id",
        "asset_family_id",
        "intervention_id",
        "source_episode_id",
        "base_episode_id",
        "preprocessing_identity",
    ],
)
def test_receipt_only_retention_rejects_manifest_row_with_tampered_planner_identity(
    tmp_path: Path,
    field: str,
):
    import hashlib

    queue, job, result, result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, result)
    manifest_path = queue.root / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "attempt_plan":
        manifest["episodes"][0][field]["episode_index"] += 1
    elif field == "episode_index":
        manifest["episodes"][0][field] += 1
    elif field in {"scene_seed"}:
        manifest["episodes"][0][field] += 1
    elif field in {"source_episode_id", "base_episode_id", "intervention_id"}:
        manifest["episodes"][0][field] = "forged-id"
    elif field == "preprocessing_identity":
        manifest["episodes"][0][field] = " "
    else:
        manifest["episodes"][0][field] = "forged-value"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    receipt_path = queue.root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dataset_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest row.*identity|attempt_plan|preprocessing"):
        queue.mark_published(job.job_id, retention="receipt_only")

    assert (result_dir / "episode_manifest.json").is_file()


def test_receipt_only_idempotent_publish_rejects_corrupted_saved_queue_receipt(
    tmp_path: Path,
):
    queue, job, result, _result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, result)
    published = queue.mark_published(job.job_id, retention="receipt_only")
    per_job_path = published / "publication_receipt.json"
    per_job = json.loads(per_job_path.read_text(encoding="utf-8"))
    per_job["artifact_hashes"]["episode_manifest.json"] = "0" * 64
    per_job_path.write_text(json.dumps(per_job), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt|hash|inventory"):
        queue.mark_published(job.job_id, retention="receipt_only")


def test_receipt_only_prune_resumes_after_interrupted_partial_directory_removal(
    tmp_path: Path,
):
    queue, job, _result, result_dir = _receipt_only_queue_fixture(tmp_path)
    _write_verified_archive_receipt(queue, job, _result)
    source = queue.root / "ingested" / job.job_id
    _stored_job, _payload, per_job_receipt = queue._receipt_only_binding(job.job_id, source)
    (source / "publication_receipt.json").write_text(
        json.dumps(per_job_receipt), encoding="utf-8",
    )
    (result_dir / "episode_manifest.json").unlink()

    published = queue.mark_published(job.job_id, retention="receipt_only")

    assert published == queue.root / "published" / job.job_id
    assert not result_dir.exists()
    assert (published / "publication_receipt.json").is_file()


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
