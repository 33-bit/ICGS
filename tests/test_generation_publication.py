from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_publication import (
    HuggingFaceBatchPublisher,
    PublicationConfig,
    PublicationReceipt,
    reconcile_publication,
)
from icgs.data.collection.generation.episode_archive import EpisodeArchiveWriter
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue


class FakeApi:
    def __init__(self, *, fail_create_commit: bool = False):
        self.fail_create_commit = fail_create_commit
        self.calls = []
        self.revision = "" * 64

    def list_repo_files(self, **kwargs):
        return []

    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_create_commit:
            raise RuntimeError("429 rate limit")
        self.revision = "c" * 40
        return type("Commit", (), {"oid": self.revision})()

    def hf_hub_download(self, **kwargs):
        raise AssertionError("fake publisher must use injected verifier")


class TransientTimeoutApi(FakeApi):
    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError(
            "Error while uploading 'episode.json': "
            "<Error><Code>RequestTimeout</Code></Error>"
        )


class DataCommitThenTimeoutApi(FakeApi):
    def create_commit(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            self.revision = "d" * 40
            return type("Commit", (), {"oid": self.revision})()
        raise RuntimeError("RequestTimeout while committing publication receipt")


class ArchiveCapturingApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.committed_files = []

    def create_commit(self, **kwargs):
        committed_files = {}
        for operation in kwargs["operations"]:
            content = Path(operation.path_or_fileobj).read_bytes()
            committed_files[operation.path_in_repo] = content
        self.committed_files.append(committed_files)
        return super().create_commit(**kwargs)


def _run() -> RunConfig:
    return RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
    )


def _job() -> GenerationJob:
    from icgs.data.collection.generation.batch import AttemptPlan
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-00001", episode_kind="nominal", scene_seed=1,
        collection_seed=20260920,
        randomization={
            "scene_signature": "sig-1",
            "asset_instance_id": "asset-1",
            "asset_family_id": "family-1",
        },
        intervention=None,
    )
    return GenerationJob.create(
        job_id="job-run-1-episode-t01-00001", run_id="run-1",
        attempt_id="att-episode-t01-00001", episode_id=plan.episode_id,
        program_id="T01", plan=plan, code_revision="a" * 40,
        manifest_sha256="b" * 64, output_root="/content/run/staging",
    )


def _queue(tmp_path: Path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    job = _job()
    queue.enqueue(job)
    assert queue.claim("000") == job
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "episode.json").write_text(
        '{"episode_id": "episode-t01-00001", "program_id": "T01", "outcome": "success"}\n',
        encoding="utf-8",
    )
    import hashlib
    digest = hashlib.sha256((result_dir / "episode.json").read_bytes()).hexdigest()
    result = WorkerResult(
        job_id=job.job_id, attempt_id=job.attempt_id, episode_id=job.episode_id,
        program_id=job.program_id, outcome="success", result_dir=str(result_dir),
        file_sha256={"episode.json": digest},
        timeline={"actions": 0, "observations": 1, "durations": 0},
    )
    queue.publish_ready("000", result)
    queue.mark_ingested(result)
    return queue, job


def _archive_queue(
    tmp_path: Path,
    *,
    retention: str = "keep",
    outcome: str = "success",
    job: GenerationJob | None = None,
):
    from icgs.data.collection.generation.episode_record import assemble_episode

    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention=retention)
    queue = FilesystemJobQueue(tmp_path / "archive-queue")
    job = _job() if job is None else job
    queue.enqueue(job)
    assert queue.claim("000") == job
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[0, 3] = 0.1
    record = assemble_episode(
        plan=job.plan,
        binding={
            "program_id": job.program_id,
            "split": "train",
            "asset_family_id": "family-1",
            "source_lineage_id": "lineage-1",
        },
        observations=[
            {
                "points": np.asarray([[0.1, 0.0, 0.8]], dtype=np.float32),
                "point_valid": np.asarray([True]), "T_w_e": pose0, "grip": 0,
            },
            {
                "points": np.asarray([[0.2, 0.0, 0.8]], dtype=np.float32),
                "point_valid": np.asarray([True]), "T_w_e": pose1, "grip": 1,
            },
        ],
        transitions=[{
            "command": {"T_w_e": pose1, "grip": 1, "duration_s": 0.05},
            "achieved_duration_s": 0.05,
            "physics_substeps": 1,
            "before_boundary": 0,
            "after_boundary": 1,
        }],
        outcome=outcome,
    )
    result_dir = tmp_path / "archive-result"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays={},
        debug_metadata={
            "source_run_id": job.run_id,
            "code_revision": job.code_revision,
            "preprocessing_identity": "publication_fixture_v1",
            "job": job.as_dict(),
        },
        output_dir=result_dir,
    )
    hashes = {
        str(path.relative_to(result_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_dir.rglob("*") if path.is_file()
    }
    result = WorkerResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        episode_id=job.episode_id,
        program_id=job.program_id,
        outcome=outcome,
        result_dir=str(result_dir),
        file_sha256=hashes,
        timeline={"actions": 1, "observations": 2, "durations": 1},
    )
    queue.publish_ready("000", result)
    queue.mark_ingested(result)
    return queue, job, result, profile


def _enqueue_second_archive_job(
    queue: FilesystemJobQueue,
    first_job: GenerationJob,
    profile: ArchiveProfileConfig,
    tmp_path: Path,
) -> GenerationJob:
    from dataclasses import replace
    from icgs.data.collection.generation.episode_record import assemble_episode

    second_plan = replace(
        first_job.plan,
        episode_index=2,
        episode_id="episode-t01-00002",
        randomization={
            **first_job.plan.randomization,
            "scene_signature": "sig-2",
            "asset_instance_id": "asset-2",
        },
    )
    second_job = GenerationJob.create(
        job_id="job-run-1-episode-t01-00002",
        run_id=first_job.run_id,
        attempt_id="att-episode-t01-00002",
        episode_id=second_plan.episode_id,
        program_id=first_job.program_id,
        plan=second_plan,
        code_revision=first_job.code_revision,
        manifest_sha256=first_job.manifest_sha256,
        output_root=first_job.output_root,
    )
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[0, 3] = 0.1
    record = assemble_episode(
        plan=second_plan,
        binding={
            "program_id": second_job.program_id,
            "split": "train",
            "asset_family_id": "family-1",
            "source_lineage_id": "lineage-1",
        },
        observations=[
            {
                "points": np.asarray([[0.1, 0.0, 0.8]], dtype=np.float32),
                "point_valid": np.asarray([True]), "T_w_e": pose0, "grip": 0,
            },
            {
                "points": np.asarray([[0.2, 0.0, 0.8]], dtype=np.float32),
                "point_valid": np.asarray([True]), "T_w_e": pose1, "grip": 1,
            },
        ],
        transitions=[{
            "command": {"T_w_e": pose1, "grip": 1, "duration_s": 0.05},
            "achieved_duration_s": 0.05,
            "physics_substeps": 1,
            "before_boundary": 0,
            "after_boundary": 1,
        }],
        outcome="success",
    )
    result_dir = tmp_path / "archive-result-second"
    EpisodeArchiveWriter(profile).write_episode(
        record,
        raw_arrays={},
        debug_metadata={
            "source_run_id": second_job.run_id,
            "code_revision": second_job.code_revision,
            "preprocessing_identity": "publication_fixture_v1",
            "job": second_job.as_dict(),
        },
        output_dir=result_dir,
    )
    hashes = {
        str(path.relative_to(result_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_dir.rglob("*") if path.is_file()
    }
    result = WorkerResult(
        job_id=second_job.job_id,
        attempt_id=second_job.attempt_id,
        episode_id=second_job.episode_id,
        program_id=second_job.program_id,
        outcome="success",
        result_dir=str(result_dir),
        file_sha256=hashes,
        timeline={"actions": 1, "observations": 2, "durations": 1},
    )
    queue.enqueue(second_job)
    assert queue.claim("001") == second_job
    queue.publish_ready("001", result)
    queue.mark_ingested(result)
    return second_job


def test_publish_due_only_after_300_seconds_or_force(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue, last_success_s=100.0)
    assert publisher.publish_due(now_s=399.9, force=False) is None
    receipt = publisher.publish_due(now_s=400.0, force=False)
    assert receipt is not None
    assert receipt.job_ids == (job.job_id,)
    assert publisher.remote_manifest["episodes"][0]["episode_id"] == job.episode_id


def test_legacy_receipt_commit_uses_a_stable_verification_snapshot(tmp_path: Path):
    queue, job = _queue(tmp_path)

    class SnapshotApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.receipt_snapshot_bytes = None
            self.receipt_paths = []

        def create_commit(self, **kwargs):
            for operation in kwargs["operations"]:
                if operation.path_in_repo.endswith("/publication_receipt.json"):
                    self.receipt_paths.append(Path(operation.path_or_fileobj))
                    self.receipt_snapshot_bytes = Path(operation.path_or_fileobj).read_bytes()
            return super().create_commit(**kwargs)

    revisions = []
    api = SnapshotApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        remote_verify=lambda job_ids, revision, _token: revisions.append((job_ids, revision)),
    )

    completed = publisher.publish_due(now_s=300.0, force=False)

    assert completed is not None and completed.status == "COMPLETE"
    assert revisions == [((job.job_id,), "c" * 40)]
    assert api.receipt_snapshot_bytes is not None
    assert json.loads(api.receipt_snapshot_bytes)["status"] == "DATA_COMMITTED"
    assert api.receipt_paths[-1] == queue.root / "publication_receipt_to_verify.json"
    assert not (queue.root / "publication_receipt_to_verify.json").exists()


def test_complete_receipt_does_not_block_the_next_ingested_batch(tmp_path: Path):
    from dataclasses import replace
    import hashlib

    queue, first_job = _queue(tmp_path)
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(_run(), api, "secret", queue, last_success_s=0.0)
    first_receipt = publisher.publish_due(now_s=300.0, force=False)
    assert first_receipt is not None
    assert queue.counts().published == 1

    second_plan = replace(
        first_job.plan,
        episode_index=2,
        episode_id="episode-t01-00002",
        randomization={"scene_signature": "sig-2", "asset_instance_id": "asset-2"},
    )
    second_job = GenerationJob.create(
        job_id="job-run-1-episode-t01-00002",
        run_id="run-1",
        attempt_id="att-episode-t01-00002",
        episode_id=second_plan.episode_id,
        program_id="T01",
        plan=second_plan,
        code_revision="a" * 40,
        manifest_sha256="b" * 64,
        output_root="/content/run/staging",
    )
    queue.enqueue(second_job)
    assert queue.claim("000") == second_job
    result_dir = tmp_path / "result-second"
    result_dir.mkdir()
    episode_path = result_dir / "episode.json"
    episode_path.write_text(
        '{"episode_id": "episode-t01-00002", "program_id": "T01", "outcome": "success"}\n',
        encoding="utf-8",
    )
    second_result = WorkerResult(
        job_id=second_job.job_id,
        attempt_id=second_job.attempt_id,
        episode_id=second_job.episode_id,
        program_id=second_job.program_id,
        outcome="success",
        result_dir=str(result_dir),
        file_sha256={"episode.json": hashlib.sha256(episode_path.read_bytes()).hexdigest()},
        timeline={"actions": 0, "observations": 1, "durations": 0},
    )
    queue.publish_ready("000", second_result)
    queue.mark_ingested(second_result)

    second_receipt = publisher.publish_due(now_s=600.0, force=False)
    assert second_receipt is not None
    assert second_receipt.job_ids == (second_job.job_id,)
    assert queue.counts().published == 2
    assert len(api.calls) == 4


def test_pending_job_limit_must_be_positive(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue)
    with pytest.raises(ValueError, match="limit must be positive"):
        publisher.pending_job_ids(limit=0)


def test_rate_limit_commit_is_deferred_and_keeps_jobs_unpublished(tmp_path: Path, monkeypatch):
    queue, job = _queue(tmp_path)
    monkeypatch.setattr(
        "icgs.data.collection.generation.distributed_publication.time.sleep",
        lambda _seconds: None,
    )
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(fail_create_commit=True), "secret", queue, last_success_s=0.0)
    assert publisher.publish_due(now_s=300.0, force=False) is None
    assert publisher.pending_job_ids() == (job.job_id,)
    assert queue.counts().ingested == 1
    receipt = json.loads((queue.root / "publication_receipt.json").read_text())
    assert receipt["status"] == "DEFERRED"


def test_transient_lfs_timeout_is_deferred_without_crashing_coordinator(tmp_path: Path, monkeypatch):
    queue, job = _queue(tmp_path)
    api = TransientTimeoutApi()
    publisher = HuggingFaceBatchPublisher(_run(), api, "secret", queue, last_success_s=0.0)
    monkeypatch.setattr("icgs.data.collection.generation.distributed_publication.time.sleep", lambda _seconds: None)

    assert publisher.publish_due(now_s=300.0, force=False) is None
    assert len(api.calls) == publisher.MAX_TRANSIENT_ATTEMPTS
    assert publisher.pending_job_ids() == (job.job_id,)
    assert queue.counts().ingested == 1

    assert publisher.publish_due(now_s=301.0, force=False) is None
    assert len(api.calls) == publisher.MAX_TRANSIENT_ATTEMPTS


def test_publish_uses_supplied_compact_manifest(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue, last_success_s=0.0)
    compact = {
        "manifest_version": 3,
        "episodes": [{"episode_id": job.episode_id, "program_id": job.program_id, "outcome": "success"}],
        "failure_attempts": [],
    }
    publisher.publish_due(now_s=300.0, force=False, local_manifest=compact)
    assert publisher.remote_manifest["episodes"] == compact["episodes"]


def test_operations_use_meaningful_episode_path(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue, last_success_s=0.0)
    ops = publisher._operations((job.job_id,), {"manifest_version": 3, "episodes": [], "failure_attempts": []})
    paths = [op.path_in_repo for op in ops]
    assert any(f"episodes/{job.program_id}/{job.episode_id}/" in path for path in paths)
    assert not any(f"results/{job.job_id}/" in path for path in paths)
    assert any(path.endswith("/resume_receipt.json") for path in paths)
    assert any("/views/" in path for path in paths)


def test_publication_refuses_result_directory_outside_run_root(tmp_path: Path):
    queue, job = _queue(tmp_path)
    stored_result_path = queue.root / "ingested" / job.job_id / "result.json"
    stored_result = json.loads(stored_result_path.read_text(encoding="utf-8"))
    outside_root = tmp_path.parent / f"{tmp_path.name}-outside-result"
    outside_root.mkdir()
    original_root = Path(stored_result["result_dir"])
    for path in original_root.iterdir():
        (outside_root / path.name).write_bytes(path.read_bytes())
    stored_result["result_dir"] = str(outside_root)
    stored_result_path.write_text(json.dumps(stored_result), encoding="utf-8")
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue)

    with pytest.raises(ValueError, match="beneath the run root"):
        publisher._operations(
            (job.job_id,),
            {"manifest_version": 3, "episodes": [], "failure_attempts": []},
        )

    assert (queue.root / "ingested" / job.job_id / "result.json").is_file()


def test_legacy_publication_receipt_omits_archive_only_identity_fields(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue)

    publisher._operations(
        (_job_value.job_id,),
        {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )

    receipt = json.loads((queue.root / "publication_receipt.json").read_text(encoding="utf-8"))
    assert set(receipt).isdisjoint({
        "source_run_id", "dataset_identity", "archive_format_id",
        "episode_schema_version", "archive_profile", "dataset_manifest_sha256",
    })


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_archive_operations_revalidate_complete_episode_inventory_before_publication(
    tmp_path: Path,
    outcome: str,
):
    queue, job, result, profile = _archive_queue(tmp_path, outcome=outcome)
    assert result.outcome == outcome
    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, last_success_s=0.0,
        archive_profile=profile,
    )
    manifest = publisher._manifest_from_states(("ingested",))
    assert "result_dir" not in manifest["episodes"][0]

    operations = publisher._operations((job.job_id,), manifest)
    result_paths = {
        operation.path_in_repo
        for operation in operations
        if operation.path_in_repo.startswith(
            f"{_run().hf_subfolder}/episodes/{job.program_id}/{job.episode_id}/"
        )
    }
    assert result_paths == {
        f"{_run().hf_subfolder}/episodes/{job.program_id}/{job.episode_id}/{relative}"
        for relative in result.file_sha256
    }

    chunk = next(Path(result.result_dir).glob("data/chunk-*.npz"))
    chunk.write_bytes(chunk.read_bytes() + b"tampered")
    result_path = queue.root / "ingested" / job.job_id / "result.json"
    stored_result = json.loads(result_path.read_text(encoding="utf-8"))
    stored_result["file_sha256"][str(chunk.relative_to(result.result_dir))] = hashlib.sha256(
        chunk.read_bytes()
    ).hexdigest()
    result_path.write_text(json.dumps(stored_result), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        publisher._operations((job.job_id,), manifest)


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_operations_upload_complete_attempt_inventory(
    tmp_path: Path,
    outcome: str,
):
    profile = ArchiveProfileConfig(chunk_boundaries=2)
    queue = FilesystemJobQueue(tmp_path / "attempt-queue")
    job = _job()
    queue.enqueue(job)
    assert queue.claim("000") == job
    result_dir = tmp_path / "attempt-result"
    EpisodeArchiveWriter(profile).write_attempt(
        {
            "schema_version": "icgs_episode_v2",
            "attempt_id": job.attempt_id,
            "episode_id": None,
            "program_id": job.program_id,
            "split": job.plan.split,
            "episode_kind": job.plan.episode_kind,
            "outcome": outcome,
            "valid_observation_until": None,
        },
        prefix_arrays=(
            {"actions": np.asarray([[0.01]], dtype=np.float64)}
            if outcome == "simulator_crash"
            else {}
        ),
        debug_metadata={
            "source_run_id": job.run_id,
            "code_revision": job.code_revision,
            "preprocessing_identity": "publication_attempt_fixture_v1",
            "job": job.as_dict(),
        },
        output_dir=result_dir,
    )
    hashes = {
        str(path.relative_to(result_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_dir.rglob("*") if path.is_file()
    }
    result = WorkerResult(
        job_id=job.job_id,
        attempt_id=job.attempt_id,
        episode_id=None,
        program_id=job.program_id,
        outcome=outcome,
        result_dir=str(result_dir),
        file_sha256=hashes,
        timeline=None,
    )
    queue.publish_ready("000", result)
    queue.mark_ingested(result)

    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, archive_profile=profile,
    )
    manifest = publisher._manifest_from_states(("ingested",))
    operations = publisher._operations((job.job_id,), manifest)
    prefix = f"{_run().hf_subfolder}/attempts/{job.program_id}/{job.attempt_id}/"
    attempt_paths = {
        operation.path_in_repo
        for operation in operations
        if operation.path_in_repo.startswith(prefix)
    }

    assert manifest["episodes"] == []
    assert manifest["failure_attempts"][0]["outcome"] == outcome
    assert attempt_paths == {prefix + relative for relative in hashes}


@pytest.mark.parametrize(
    "remote_mismatch",
    [
        "none",
        "archive",
        "dataset_manifest",
        "resume_receipt",
        "publication_receipt",
        "view_D_geom",
        "view_D_temporal",
        "view_D_dyn",
        "view_D_task",
        "episode_view_D_geom",
        "episode_view_missing",
    ],
)
def test_coordinator_verifies_each_archive_file_in_a_fresh_revision_pinned_cache(
    tmp_path: Path, monkeypatch, remote_mismatch: str,
):
    from scripts import generation_coordinator as coordinator

    queue, job, result, profile = _archive_queue(tmp_path)
    run = _run()
    publisher = HuggingFaceBatchPublisher(
        run, FakeApi(), "secret", queue, archive_profile=profile,
    )
    local = publisher._manifest_from_states(("ingested",))
    manifest = publisher._merge_manifest(
        local, {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    publisher._operations((job.job_id,), manifest)
    prefix = f"{run.hf_subfolder}/episodes/{job.program_id}/{job.episode_id}"
    remote_sources = {
        f"{prefix}/{relative}": Path(result.result_dir) / relative
        for relative in result.file_sha256
    }
    remote_sources[f"{run.hf_subfolder}/dataset_manifest.json"] = (
        queue.root / "publication_manifest.json"
    )
    remote_sources[f"{run.hf_subfolder}/resume_receipt.json"] = (
        queue.root / "resume_receipt.json"
    )
    receipt_snapshot = queue.root / "publication_receipt_to_verify.json"
    receipt_snapshot.write_bytes((queue.root / "publication_receipt.json").read_bytes())
    remote_sources[f"{run.hf_subfolder}/publication_receipt.json"] = receipt_snapshot
    mutable_receipt_path = queue.root / "publication_receipt.json"
    mutable_receipt = json.loads(mutable_receipt_path.read_text(encoding="utf-8"))
    mutable_receipt["status"] = "VERIFIED"
    mutable_receipt["commit_oid"] = "c" * 40
    mutable_receipt_path.write_text(json.dumps(mutable_receipt), encoding="utf-8")
    for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
        remote_sources[f"{run.hf_subfolder}/views/{view}.json"] = (
            queue.root / "views" / f"{view}.json"
        )
        remote_sources[
            f"{run.hf_subfolder}/views/provisional/episodes/"
            f"{job.program_id}/{job.episode_id}/{view}.json"
        ] = (
            queue.root / "views" / "provisional" / "episodes"
            / job.program_id / job.episode_id / f"{view}.json"
        )
    first_archive_filename = sorted(
        filename for filename in remote_sources
        if filename.startswith(f"{run.hf_subfolder}/episodes/")
        or filename.startswith(f"{run.hf_subfolder}/attempts/")
    )[0]
    metadata_paths = {
        "dataset_manifest": f"{run.hf_subfolder}/dataset_manifest.json",
        "resume_receipt": f"{run.hf_subfolder}/resume_receipt.json",
        "publication_receipt": f"{run.hf_subfolder}/publication_receipt.json",
        **{
            f"view_{view}": f"{run.hf_subfolder}/views/{view}.json"
            for view in ("D_geom", "D_temporal", "D_dyn", "D_task")
        },
        "episode_view_D_geom": (
            f"{run.hf_subfolder}/views/provisional/episodes/"
            f"{job.program_id}/{job.episode_id}/D_geom.json"
        ),
    }
    changed_filename = (
        first_archive_filename if remote_mismatch == "archive"
        else metadata_paths.get("episode_view_D_geom")
        if remote_mismatch == "episode_view_missing"
        else metadata_paths.get(remote_mismatch)
    )
    calls = []

    def fake_download(**kwargs):
        cache_dir = Path(kwargs["cache_dir"])
        assert kwargs["revision"] == "c" * 40
        if remote_mismatch == "episode_view_missing" and "/views/provisional/episodes/" in kwargs["filename"]:
            calls.append((kwargs["filename"], kwargs["revision"], cache_dir))
            raise FileNotFoundError(kwargs["filename"])
        source = remote_sources[kwargs["filename"]]
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded = cache_dir / "downloaded.bin"
        contents = source.read_bytes()
        if remote_mismatch != "none" and kwargs["filename"] == changed_filename:
            contents += b"remote mutation"
        downloaded.write_bytes(contents)
        calls.append((kwargs["filename"], kwargs["revision"], cache_dir))
        return str(downloaded)

    monkeypatch.setattr(coordinator, "hf_hub_download", fake_download)

    if remote_mismatch == "episode_view_missing":
        with pytest.raises(FileNotFoundError):
            coordinator._verify_remote_batch(
                queue, run, (job.job_id,), "c" * 40, "local-test-token",
            )
    elif remote_mismatch != "none":
        with pytest.raises(ValueError, match="remote hash mismatch"):
            coordinator._verify_remote_batch(
                queue, run, (job.job_id,), "c" * 40, "local-test-token",
            )
    else:
        coordinator._verify_remote_batch(
            queue, run, (job.job_id,), "c" * 40, "local-test-token",
        )

    if remote_mismatch == "archive":
        assert len(calls) == 1
    elif remote_mismatch != "none":
        assert calls[-1][0] == changed_filename
    else:
        assert {filename for filename, _revision, _cache in calls} == set(remote_sources)
    assert len({cache for _filename, _revision, cache in calls}) == len(calls)
    assert all(not cache.exists() for _filename, _revision, cache in calls)


def test_published_receipt_only_job_reconstructs_manifest_without_episode_payload(
    tmp_path: Path,
):
    from scripts import generation_coordinator as coordinator

    queue, job, _result, profile = _archive_queue(tmp_path, retention="receipt_only")
    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, archive_profile=profile,
    )
    local = publisher._manifest_from_states(("ingested",))
    manifest = publisher._merge_manifest(
        local, {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    publisher._operations(
        (job.job_id,),
        manifest,
        PublicationReceipt(
            run_id=job.run_id,
            source_run_id=job.run_id,
            job_ids=(job.job_id,),
            status="VERIFIED",
            data_commit_oid="d" * 40,
            commit_oid="c" * 40,
            artifact_hashes=publisher._artifact_hashes((job.job_id,)),
            dataset_identity=profile.dataset_identity,
            archive_format_id=profile.archive_format_id,
            episode_schema_version=profile.episode_schema_version,
            archive_profile=profile.as_dict(),
            prefix=_run().hf_subfolder,
        ),
    )

    queue.mark_published(job.job_id, retention="receipt_only")
    assert not Path(_result.result_dir).exists()

    reconstructed = coordinator._manifest_from_closed_queue(
        queue, states=("published",), archive_profile=profile,
    )
    resumed = coordinator._merge_manifests(manifest, reconstructed)

    assert reconstructed["episodes"] == manifest["episodes"]
    assert resumed["episodes"] == manifest["episodes"]
    assert not Path(_result.result_dir).exists()


def test_coordinator_open_recovers_verified_prune_then_fails_closed_on_remote_error(
    tmp_path: Path,
    monkeypatch,
):
    from dataclasses import replace
    from scripts import generation_coordinator
    from scripts.generation_launch import persist_run_config
    from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig
    import icgs.data.collection.generation.distributed_queue as queue_module

    job = _job()
    job = replace(
        job,
        plan=replace(
            job.plan,
            randomization={
                **job.plan.randomization,
                "scene_seed": job.plan.scene_seed,
            },
        ),
    )
    queue, job, result, profile = _archive_queue(
        tmp_path, retention="receipt_only", job=job,
    )
    queue.root.rename(tmp_path / "queue")
    queue = FilesystemJobQueue(tmp_path / "queue")

    runtime = GenerationRuntimeConfig.from_dict({
        "machine": {
            "repo_root": str(tmp_path),
            "python_executable": "/usr/bin/python3",
            "simulator_root": str(tmp_path),
            "rlbench_root": str(tmp_path),
            "display_base": 41,
            "display_width": 1366,
            "display_height": 768,
            "simulator_slots": 1,
            "worker_timeout_s": 60,
        },
        "run": {
            "run_id": job.run_id,
            "run_root": str(tmp_path),
            "worker_count": 1,
            "publish_interval_s": 300,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "generation/recovery-test",
            "publication_enabled": True,
            "validation_mode": False,
        },
        "archive_profile": profile.as_dict(),
    })
    approved_manifest = tmp_path / "approved.json"
    approved_manifest.write_bytes(
        Path("artifacts/composition/approved_composition_manifest.json").read_bytes()
    )
    run_config_path = persist_run_config(
        runtime, approved_manifest, code_revision=job.code_revision,
    )
    token_path = tmp_path / "hf-token"
    token_path.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token_path)
    run_config_path.write_text(json.dumps(run_payload), encoding="utf-8")

    run = RunConfig.from_dict(run_payload["run"])
    publisher = HuggingFaceBatchPublisher(
        run,
        FakeApi(),
        "secret",
        queue,
        archive_profile=profile,
        remote_verify=lambda *_args: None,
        publication_config=PublicationConfig(
            batch_size=1,
            retry_attempts=1,
            retry_cooldown_s=0.0,
            rate_limit_cooldown_s=0.0,
        ),
    )
    relative_to_remove = next(iter(result.file_sha256))
    original_rmtree = queue_module.shutil.rmtree

    def interrupt_prune(path, *args, **kwargs):
        if Path(path) == Path(result.result_dir):
            (Path(result.result_dir) / relative_to_remove).unlink()
            raise OSError("simulated interruption during receipt-only pruning")
        return original_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(queue_module.shutil, "rmtree", interrupt_prune)
        with pytest.raises(OSError, match="interruption during receipt-only pruning"):
            publisher.publish_due(now_s=300.0, force=True)

    source = queue.root / "ingested" / job.job_id
    assert source.is_dir()
    assert (source / "publication_receipt.json").is_file()
    assert Path(result.result_dir).is_dir()
    assert not (Path(result.result_dir) / relative_to_remove).exists()

    cache_roots = []

    def download(**kwargs):
        cache = Path(kwargs["cache_dir"])
        cache_roots.append(cache)
        (cache / "partial-download.tmp").write_bytes(b"temporary cache bytes")
        raise OSError("remote unavailable")

    monkeypatch.setattr(generation_coordinator, "hf_hub_download", download)
    with pytest.raises(
        RuntimeError, match="archive startup requires a valid remote dataset manifest"
    ) as failure:
        generation_coordinator.CoordinatorControlPlane.open(
            run_config_path,
            api_factory=lambda: pytest.fail("publisher must not be constructed"),
            token_path=token_path,
        )

    assert isinstance(failure.value.__cause__, OSError)
    assert "remote unavailable" in str(failure.value.__cause__)
    assert queue.counts().ingested == 0
    assert queue.counts().published == 1
    assert not Path(result.result_dir).exists()
    assert (queue.root / "published" / job.job_id / "publication_receipt.json").is_file()
    assert len(cache_roots) == 1 and not cache_roots[0].exists()


@pytest.mark.parametrize("field", ["split", "preprocessing_identity", "source_lineage_id"])
def test_coordinator_rejects_tampered_published_manifest_row_after_pruning(
    tmp_path: Path,
    field: str,
):
    from scripts import generation_coordinator as coordinator

    queue, job, _result, profile = _archive_queue(tmp_path, retention="receipt_only")
    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, archive_profile=profile,
    )
    local = publisher._manifest_from_states(("ingested",))
    manifest = publisher._merge_manifest(
        local, {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    publisher._operations(
        (job.job_id,),
        manifest,
        PublicationReceipt(
            run_id=job.run_id,
            job_ids=(job.job_id,),
            status="VERIFIED",
            data_commit_oid="d" * 40,
            commit_oid="c" * 40,
            artifact_hashes=publisher._artifact_hashes((job.job_id,)),
            source_run_id=job.run_id,
            dataset_identity=profile.dataset_identity,
            archive_format_id=profile.archive_format_id,
            episode_schema_version=profile.episode_schema_version,
            archive_profile=profile.as_dict(),
            prefix=_run().hf_subfolder,
        ),
    )
    published = queue.mark_published(job.job_id, retention="receipt_only")
    receipt_path = published / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_row"][field] = "tampered-value"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest row|identity|preprocessing"):
        coordinator._manifest_from_closed_queue(
            queue, states=("published",), archive_profile=profile,
        )


def test_archive_verification_failure_keeps_payload_and_retries_same_revision(
    tmp_path: Path,
):
    queue, job, result, profile = _archive_queue(tmp_path, retention="receipt_only")

    class SequencedApi(FakeApi):
        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            oid = "d" * 40 if len(self.calls) == 1 else "e" * 40
            return type("Commit", (), {"oid": oid})()

    revisions = []

    def verify(_job_ids, revision, _token):
        revisions.append(revision)
        if len(revisions) == 1:
            raise ValueError("remote hash mismatch")

    api = SequencedApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile, remote_verify=verify,
    )

    with pytest.raises(ValueError, match="remote hash mismatch"):
        publisher.publish_due(now_s=300.0, force=False)

    receipt_after_failure = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_after_failure["status"] == "DATA_COMMITTED"
    assert receipt_after_failure["data_commit_oid"] == "d" * 40
    assert receipt_after_failure["commit_oid"] == "e" * 40
    assert queue.counts().ingested == 1
    assert queue.counts().published == 0
    assert Path(result.result_dir).is_dir()
    snapshot_path = queue.root / "publication_receipt_to_verify.json"
    assert snapshot_path.is_file()
    assert api.calls[1]["operations"][0].path_or_fileobj == str(snapshot_path)

    completed = publisher.publish_due(now_s=301.0, force=False)

    assert completed is not None and completed.status == "COMPLETE"
    assert revisions == ["e" * 40, "e" * 40]
    assert len(api.calls) == 3
    operation_paths = [
        [operation.path_in_repo for operation in call["operations"]]
        for call in api.calls
    ]
    assert any("/episodes/" in path for path in operation_paths[0])
    assert all(not any("/episodes/" in path for path in paths) for paths in operation_paths[1:])
    assert queue.counts().published == 1
    assert not Path(result.result_dir).exists()
    completed_receipt = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert completed_receipt["source_run_id"] == job.run_id
    assert completed_receipt["artifact_hashes"] == {
        f"{job.job_id}/{relative}": digest
        for relative, digest in result.file_sha256.items()
    }
    assert completed_receipt["dataset_manifest_sha256"] == hashlib.sha256(
        (queue.root / "publication_manifest.json").read_bytes()
    ).hexdigest()
    assert not snapshot_path.exists()
    resume_receipt = json.loads(
        (queue.root / "resume_receipt.json").read_text(encoding="utf-8")
    )
    assert resume_receipt["dataset_manifest_revision"] == "d" * 40
    assert publisher.publish_due(now_s=302.0, force=True).status == "COMPLETE"
    assert len(api.calls) == 3


def test_archive_retry_advances_remote_manifest_before_next_batch(tmp_path: Path):
    queue, first_job, _first_result, profile = _archive_queue(
        tmp_path, retention="receipt_only",
    )

    class CapturingApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.committed_files = []

        def create_commit(self, **kwargs):
            self.committed_files.append({
                operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes()
                for operation in kwargs["operations"]
            })
            return super().create_commit(**kwargs)

    verification_calls = []
    fail_once = True

    def verify(job_ids, revision, _token):
        nonlocal fail_once
        verification_calls.append((job_ids, revision))
        if fail_once:
            fail_once = False
            raise TimeoutError("temporary remote verification timeout")

    api = CapturingApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=verify,
        publication_config=PublicationConfig(
            batch_size=1,
            retry_attempts=1,
            retry_cooldown_s=0.0,
            rate_limit_cooldown_s=0.0,
        ),
    )

    assert publisher.publish_due(now_s=300.0, force=False) is None
    assert queue.counts().published == 0
    assert publisher.publish_due(now_s=301.0, force=False).status == "COMPLETE"
    assert queue.counts().published == 1

    second_job = _enqueue_second_archive_job(queue, first_job, profile, tmp_path)

    assert publisher.publish_due(now_s=302.0, force=True).status == "COMPLETE"

    second_manifest_bytes = next(
        files[f"{_run().hf_subfolder}/dataset_manifest.json"]
        for files in reversed(api.committed_files)
        if f"{_run().hf_subfolder}/dataset_manifest.json" in files
    )
    committed_manifest = json.loads(second_manifest_bytes)
    assert {row["episode_id"] for row in committed_manifest["episodes"]} == {
        first_job.episode_id,
        second_job.episode_id,
    }
    assert verification_calls[0][0] == (first_job.job_id,)
    assert verification_calls[1][0] == (first_job.job_id,)
    assert queue.counts().published == 2


def test_archive_new_batch_clears_stale_verification_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    queue, first_job, _first_result, profile = _archive_queue(
        tmp_path, retention="receipt_only",
    )
    api = ArchiveCapturingApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=lambda *_args: None,
        publication_config=PublicationConfig(
            batch_size=1,
            retry_attempts=1,
            retry_cooldown_s=0.0,
            rate_limit_cooldown_s=0.0,
        ),
    )
    original_unlink = Path.unlink
    interrupted = False

    def interrupt_snapshot_unlink(path, *args, **kwargs):
        nonlocal interrupted
        if path == queue.root / "publication_receipt_to_verify.json" and not interrupted:
            interrupted = True
            raise OSError("simulated crash after COMPLETE")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt_snapshot_unlink)
        with pytest.raises(OSError, match="simulated crash after COMPLETE"):
            publisher.publish_due(now_s=300.0, force=True)

    stale_snapshot = queue.root / "publication_receipt_to_verify.json"
    assert stale_snapshot.is_file()
    assert json.loads((queue.root / "publication_receipt.json").read_text())["status"] == "COMPLETE"

    # Model the remote manifest already containing the completed first batch;
    # the regression is the stale local receipt snapshot blocking the disjoint
    # second batch from replacing it.
    publisher.remote_manifest = json.loads(
        (queue.root / "publication_manifest.json").read_text(encoding="utf-8")
    )
    second_job = _enqueue_second_archive_job(queue, first_job, profile, tmp_path)

    completed = publisher.publish_due(now_s=301.0, force=True)

    assert completed is not None and completed.status == "COMPLETE"
    assert not stale_snapshot.exists()
    manifest_path = f"{_run().hf_subfolder}/dataset_manifest.json"
    second_manifest = json.loads(
        next(files[manifest_path] for files in reversed(api.committed_files) if manifest_path in files)
    )
    assert {row["episode_id"] for row in second_manifest["episodes"]} == {
        first_job.episode_id,
        second_job.episode_id,
    }


def test_archive_verified_completion_retry_advances_manifest_before_next_batch(
    tmp_path: Path,
    monkeypatch,
):
    queue, first_job, _first_result, profile = _archive_queue(
        tmp_path, retention="receipt_only",
    )
    api = ArchiveCapturingApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=lambda *_args: None,
        publication_config=PublicationConfig(
            batch_size=1,
            retry_attempts=1,
            retry_cooldown_s=0.0,
            rate_limit_cooldown_s=0.0,
        ),
    )
    original_mark_published = queue.mark_published
    failed = True

    def fail_mark_published(job_id, **kwargs):
        nonlocal failed
        if failed:
            failed = False
            raise OSError("simulated prune interruption")
        return original_mark_published(job_id, **kwargs)

    monkeypatch.setattr(queue, "mark_published", fail_mark_published)
    with pytest.raises(OSError, match="simulated prune interruption"):
        publisher.publish_due(now_s=300.0, force=True)
    monkeypatch.setattr(queue, "mark_published", original_mark_published)

    retried = publisher.publish_due(now_s=301.0, force=True)
    assert retried is not None and retried.status == "COMPLETE"
    assert publisher.remote_manifest["episodes"][0]["episode_id"] == first_job.episode_id

    second_job = _enqueue_second_archive_job(queue, first_job, profile, tmp_path)
    completed = publisher.publish_due(now_s=302.0, force=True)

    assert completed is not None and completed.status == "COMPLETE"
    manifest_path = f"{_run().hf_subfolder}/dataset_manifest.json"
    second_manifest = json.loads(
        next(files[manifest_path] for files in reversed(api.committed_files) if manifest_path in files)
    )
    assert {row["episode_id"] for row in second_manifest["episodes"]} == {
        first_job.episode_id,
        second_job.episode_id,
    }


def test_archive_receipt_retry_reuses_committed_data_revision_without_reupload(
    tmp_path: Path,
):
    queue, job, result, profile = _archive_queue(tmp_path, retention="receipt_only")

    class ReceiptTimeoutApi(FakeApi):
        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                oid = "d" * 40
            elif len(self.calls) == 2:
                raise RuntimeError("RequestTimeout while committing receipt")
            else:
                oid = ("e" if len(self.calls) == 3 else "f") * 40
            return type("Commit", (), {"oid": oid})()

    revisions = []
    api = ReceiptTimeoutApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        publication_config=PublicationConfig(
            retry_attempts=1, retry_cooldown_s=0.0, rate_limit_cooldown_s=0.0,
        ),
        remote_verify=lambda _job_ids, revision, _token: revisions.append(revision),
    )

    assert publisher.publish_due(now_s=300.0, force=False) is None
    deferred = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert deferred["status"] == "DEFERRED"
    assert deferred["data_commit_oid"] == "d" * 40
    assert Path(result.result_dir).is_dir()
    assert queue.counts().published == 0

    completed = publisher.publish_due(now_s=301.0, force=False)

    assert completed is not None and completed.status == "COMPLETE"
    assert deferred["data_commit_oid"] == "d" * 40
    assert revisions == ["e" * 40]
    assert len(api.calls) == 4
    operation_paths = [
        [operation.path_in_repo for operation in call["operations"]]
        for call in api.calls
    ]
    assert any("/episodes/" in path for path in operation_paths[0])
    assert all(not any("/episodes/" in path for path in paths) for paths in operation_paths[1:])
    assert not Path(result.result_dir).exists()
    assert queue.counts().published == 1
    assert publisher.pending_job_ids() == ()


def test_archive_data_commit_timeout_retries_same_ingested_batch(tmp_path: Path):
    queue, job, result, profile = _archive_queue(tmp_path, retention="receipt_only")

    class DataTimeoutApi(FakeApi):
        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("RequestTimeout while committing archive")
            oid = {2: "d", 3: "e"}.get(len(self.calls), "f") * 40
            return type("Commit", (), {"oid": oid})()

    revisions = []
    api = DataTimeoutApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        publication_config=PublicationConfig(
            retry_attempts=1, retry_cooldown_s=0.0, rate_limit_cooldown_s=0.0,
        ),
        remote_verify=lambda _job_ids, revision, _token: revisions.append(revision),
    )

    assert publisher.publish_due(now_s=300.0, force=False) is None
    deferred = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert deferred["status"] == "DEFERRED"
    assert deferred["data_commit_oid"] is None
    assert Path(result.result_dir).is_dir()

    completed = publisher.publish_due(now_s=301.0, force=False)

    assert completed is not None and completed.status == "COMPLETE"
    assert revisions == ["e" * 40]
    assert len(api.calls) == 4
    assert queue.counts().published == 1
    assert not Path(result.result_dir).exists()
    assert completed.job_ids == (job.job_id,)


def test_coordinator_recovers_data_revision_after_local_receipt_write_crash(
    tmp_path: Path,
):
    from scripts import generation_coordinator as coordinator

    queue, job, result, profile = _archive_queue(tmp_path, retention="receipt_only")
    remote_root = tmp_path / "remote" / "snapshots" / ("d" * 40)

    class CommittingApi(FakeApi):
        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            for operation in kwargs["operations"]:
                destination = remote_root / operation.path_in_repo
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(operation.path_or_fileobj).read_bytes())
            return type("Commit", (), {"oid": "d" * 40})()

    first_api = CommittingApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), first_api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=lambda _job_ids, _revision, _token: None,
        publication_config=PublicationConfig(
            retry_attempts=1, retry_cooldown_s=0.0, rate_limit_cooldown_s=0.0,
        ),
    )
    original_write = publisher._write_publication_receipt

    def interrupt_local_oid_persistence(receipt):
        if receipt.status == "DATA_COMMITTED" and receipt.data_commit_oid == "d" * 40:
            raise RuntimeError("simulated stop before local OID persistence")
        original_write(receipt)

    publisher._write_publication_receipt = interrupt_local_oid_persistence
    with pytest.raises(RuntimeError, match="local OID persistence"):
        publisher.publish_due(now_s=300.0, force=False)

    assert len(first_api.calls) == 1
    assert any(
        "/episodes/" in operation.path_in_repo
        for operation in first_api.calls[0]["operations"]
    )
    pending = json.loads((queue.root / "publication_receipt.json").read_text())
    assert pending["status"] == "PREPARED"
    assert pending["data_commit_oid"] is None
    remote_manifest = remote_root / _run().hf_subfolder / "dataset_manifest.json"
    download_calls = []
    cache_roots = []
    returned_paths = []

    def download(**kwargs):
        download_calls.append(kwargs)
        cache_arg = kwargs.get("cache_dir")
        if cache_arg is None:
            if kwargs["filename"].endswith("/dataset_manifest.json"):
                return str(remote_manifest)
            if kwargs["filename"].endswith("/publication_receipt.json"):
                return str(remote_root / kwargs["filename"])
            raise FileNotFoundError(kwargs["filename"])
        cache = Path(cache_arg)
        cache_roots.append(cache)
        source = remote_root / kwargs["filename"]
        cached = cache / "snapshots" / ("d" * 40) / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            (cache / "partial-download.tmp").write_bytes(b"partial control bytes")
            raise FileNotFoundError(source)
        payload = source.read_bytes()
        blob = cache / "blobs" / hashlib.sha256(payload).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        cached.symlink_to(blob)
        returned_paths.append(cached)
        return str(cached)

    coordinator._recover_interrupted_archive_publication(
        queue,
        _run(),
        profile,
        token="secret",
        downloader=download,
    )

    recovered = json.loads((queue.root / "publication_receipt.json").read_text())
    assert recovered["status"] == "DATA_COMMITTED"
    assert recovered["data_commit_oid"] == "d" * 40
    assert len(download_calls) == 2
    assert all(call.get("cache_dir") for call in download_calls)
    assert len(cache_roots) == 2 and len(set(cache_roots)) == 2
    assert all(not cache.exists() for cache in cache_roots)
    assert len(returned_paths) == 2
    assert coordinator._snapshot_revision(returned_paths[0]) == "d" * 40

    api = FakeApi()
    verified_revisions = []
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=lambda _job_ids, revision, _token: verified_revisions.append(revision),
        publication_config=PublicationConfig(
            retry_attempts=1, retry_cooldown_s=0.0, rate_limit_cooldown_s=0.0,
        ),
    )
    completed = publisher.publish_due(now_s=300.0, force=False)
    assert completed is not None and completed.status == "COMPLETE"
    assert len(api.calls) == 2
    assert verified_revisions == ["c" * 40]
    assert all(
        not any("/episodes/" in operation.path_in_repo for operation in call["operations"])
        for call in api.calls
    )
    assert recovered["data_commit_oid"] == "d" * 40
    assert not Path(result.result_dir).exists()


def test_coordinator_recovers_receipt_commit_oid_from_exact_remote_snapshot(
    tmp_path: Path,
):
    from dataclasses import replace
    from scripts import generation_coordinator as coordinator

    queue, job, result, profile = _archive_queue(tmp_path, retention="receipt_only")
    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, archive_profile=profile,
    )
    local = publisher._manifest_from_states(("ingested",))
    manifest = publisher._merge_manifest(
        local, {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    prepared = PublicationReceipt(
        run_id=job.run_id,
        job_ids=(job.job_id,),
        status="PREPARED",
        artifact_hashes=publisher._artifact_hashes((job.job_id,)),
        prefix=_run().hf_subfolder,
    )
    publisher._operations((job.job_id,), manifest, prepared)
    current = PublicationReceipt.from_dict(
        json.loads((queue.root / "publication_receipt.json").read_text())
    )
    candidate = replace(current, status="DATA_COMMITTED", data_commit_oid="d" * 40)
    publisher._write_publication_receipt(candidate)
    snapshot_path = publisher._ensure_verification_receipt_snapshot(candidate)
    remote_root = tmp_path / "remote" / "snapshots" / ("e" * 40)
    remote_receipt = remote_root / _run().hf_subfolder / "publication_receipt.json"
    remote_receipt.parent.mkdir(parents=True)
    remote_receipt.write_bytes(snapshot_path.read_bytes())
    download_calls = []
    cache_roots = []
    returned_paths = []

    def download(**kwargs):
        download_calls.append(kwargs)
        cache_arg = kwargs.get("cache_dir")
        if cache_arg is None:
            return str(remote_receipt)
        cache = Path(cache_arg)
        cache_roots.append(cache)
        cached = cache / "snapshots" / ("e" * 40) / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(remote_receipt.read_bytes())
        returned_paths.append(cached)
        return str(cached)

    coordinator._recover_interrupted_archive_publication(
        queue,
        _run(),
        profile,
        token="secret",
        downloader=download,
    )

    recovered = PublicationReceipt.from_dict(
        json.loads((queue.root / "publication_receipt.json").read_text())
    )
    assert recovered.status == "DATA_COMMITTED"
    assert recovered.data_commit_oid == "d" * 40
    assert recovered.commit_oid == "e" * 40
    assert len(download_calls) == 1
    assert download_calls[0].get("cache_dir")
    assert len(cache_roots) == 1 and not cache_roots[0].exists()
    assert len(returned_paths) == 1
    assert coordinator._snapshot_revision(returned_paths[0]) == "e" * 40

    api = FakeApi()
    verified_revisions = []
    resumed_publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        archive_profile=profile,
        remote_verify=lambda _job_ids, revision, _token: verified_revisions.append(revision),
    )
    completed = resumed_publisher.publish_due(now_s=300.0, force=False)

    assert completed is not None and completed.status == "COMPLETE"
    assert verified_revisions == ["e" * 40]
    assert len(api.calls) == 1
    assert all(
        "/episodes/" not in operation.path_in_repo
        for operation in api.calls[0]["operations"]
    )
    assert not Path(result.result_dir).exists()


def test_interrupted_archive_recovery_rejects_control_file_outside_owned_cache(
    tmp_path: Path,
):
    from scripts import generation_coordinator as coordinator

    queue, job, _result, profile = _archive_queue(tmp_path, retention="receipt_only")
    receipt = PublicationReceipt(
        run_id=job.run_id,
        job_ids=(job.job_id,),
        status="PREPARED",
        artifact_hashes={f"{job.job_id}/artifact_manifest.json": "a" * 64},
        source_run_id=job.run_id,
        dataset_identity=profile.dataset_identity,
        archive_format_id=profile.archive_format_id,
        episode_schema_version=profile.episode_schema_version,
        archive_profile=profile.as_dict(),
        dataset_manifest_sha256="b" * 64,
        prefix=_run().hf_subfolder,
    )
    (queue.root / "publication_receipt.json").write_text(
        json.dumps(receipt.as_dict()), encoding="utf-8"
    )
    outside = tmp_path / "outside-dataset-manifest.json"
    outside.write_bytes(b"remote control file")
    cache_roots = []
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        cache_arg = kwargs.get("cache_dir")
        if cache_arg is not None:
            cache = Path(cache_arg)
            cache_roots.append(cache)
            (cache / "downloaded-control-file.tmp").write_bytes(b"cache bytes")
        return str(outside)

    with pytest.raises(
        RuntimeError, match="cannot reconcile pending archive data commit"
    ) as failure:
        coordinator._recover_interrupted_archive_publication(
            queue,
            _run(),
            profile,
            token="secret",
            downloader=download,
        )

    assert len(calls) == 1
    assert calls[0].get("cache_dir")
    assert isinstance(failure.value.__cause__, ValueError)
    assert "outside owned scratch" in str(failure.value.__cause__)
    assert len(cache_roots) == 1 and not cache_roots[0].exists()


@pytest.mark.parametrize(
    ("data_commit_oid", "filename", "error_message"),
    [
        (None, "dataset_manifest.json", "pending archive data commit"),
        ("d" * 40, "publication_receipt.json", "pending archive receipt commit"),
    ],
)
def test_interrupted_archive_recovery_rejects_returned_directory_as_absence(
    tmp_path: Path,
    data_commit_oid: str | None,
    filename: str,
    error_message: str,
):
    from scripts import generation_coordinator as coordinator

    queue, job, _result, profile = _archive_queue(tmp_path, retention="receipt_only")
    receipt = PublicationReceipt(
        run_id=job.run_id,
        job_ids=(job.job_id,),
        status="PREPARED" if data_commit_oid is None else "DATA_COMMITTED",
        data_commit_oid=data_commit_oid,
        artifact_hashes={f"{job.job_id}/artifact_manifest.json": "a" * 64},
        source_run_id=job.run_id,
        dataset_identity=profile.dataset_identity,
        archive_format_id=profile.archive_format_id,
        episode_schema_version=profile.episode_schema_version,
        archive_profile=profile.as_dict(),
        dataset_manifest_sha256="b" * 64,
        prefix=_run().hf_subfolder,
    )
    (queue.root / "publication_receipt.json").write_text(
        json.dumps(receipt.as_dict()), encoding="utf-8"
    )
    cache_roots = []
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        cache = Path(kwargs["cache_dir"])
        cache_roots.append(cache)
        returned_directory = cache / "snapshots" / ("e" * 40) / kwargs["filename"]
        returned_directory.mkdir(parents=True)
        return str(returned_directory)

    with pytest.raises(RuntimeError, match=error_message) as failure:
        coordinator._recover_interrupted_archive_publication(
            queue,
            _run(),
            profile,
            token="secret",
            downloader=download,
        )

    assert len(calls) == 1
    assert calls[0]["filename"].endswith(filename)
    assert calls[0].get("cache_dir")
    assert isinstance(failure.value.__cause__, ValueError)
    assert "not a regular file" in str(failure.value.__cause__)
    assert len(cache_roots) == 1 and not cache_roots[0].exists()


def test_interrupted_archive_recovery_rejects_external_symlink_alias_into_scratch(
    tmp_path: Path,
):
    from scripts import generation_coordinator as coordinator

    queue, job, _result, profile = _archive_queue(tmp_path, retention="receipt_only")
    receipt = PublicationReceipt(
        run_id=job.run_id,
        job_ids=(job.job_id,),
        status="PREPARED",
        artifact_hashes={f"{job.job_id}/artifact_manifest.json": "a" * 64},
        source_run_id=job.run_id,
        dataset_identity=profile.dataset_identity,
        archive_format_id=profile.archive_format_id,
        episode_schema_version=profile.episode_schema_version,
        archive_profile=profile.as_dict(),
        dataset_manifest_sha256="b" * 64,
        prefix=_run().hf_subfolder,
    )
    (queue.root / "publication_receipt.json").write_text(
        json.dumps(receipt.as_dict()), encoding="utf-8"
    )
    cache_roots = []
    aliases = []
    snapshot_oid = "f" * 40

    def download(**kwargs):
        cache = Path(kwargs["cache_dir"])
        cache_roots.append(cache)
        target = cache / "snapshots" / snapshot_oid / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"temporary remote bytes")
        alias = tmp_path / "outside" / "snapshots" / snapshot_oid / kwargs["filename"]
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target)
        aliases.append(alias)
        return str(alias)

    with pytest.raises(
        RuntimeError, match="cannot reconcile pending archive data commit"
    ) as failure:
        coordinator._recover_interrupted_archive_publication(
            queue,
            _run(),
            profile,
            token="secret",
            downloader=download,
        )

    assert len(aliases) == 1
    assert aliases[0].resolve().is_relative_to(cache_roots[0].resolve())
    assert not aliases[0].absolute().is_relative_to(cache_roots[0].absolute())
    assert coordinator._snapshot_revision(aliases[0]) == snapshot_oid
    assert isinstance(failure.value.__cause__, ValueError)
    assert "outside owned scratch" in str(failure.value.__cause__)
    assert len(cache_roots) == 1 and not cache_roots[0].exists()


def test_archive_metadata_and_prefix_views_remain_provisional_and_revision_bound(
    tmp_path: Path,
):
    queue, job, result, profile = _archive_queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(
        _run(), FakeApi(), "secret", queue, last_success_s=0.0,
        archive_profile=profile,
    )
    local = publisher._manifest_from_states(("ingested",))
    manifest = publisher._merge_manifest(
        local,
        {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )
    receipt = PublicationReceipt(
        run_id=job.run_id,
        job_ids=(job.job_id,),
        artifact_hashes=publisher._artifact_hashes((job.job_id,)),
        prefix=_run().hf_subfolder,
        created_at_s=1.0,
        updated_at_s=1.0,
    )

    publisher._operations((job.job_id,), manifest, receipt)

    dataset_path = queue.root / "publication_manifest.json"
    dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset_manifest["archive_profile"] == profile.as_dict()
    assert dataset_manifest["dataset_identity"] == profile.dataset_identity
    assert dataset_manifest["archive_format_id"] == profile.archive_format_id
    assert dataset_manifest["episode_schema_version"] == profile.episode_schema_version
    assert dataset_manifest["view_status"] == "PROVISIONAL"
    assert dataset_manifest["source_snapshot_created_at_s"] == receipt.created_at_s
    assert "result_dir" not in dataset_manifest["episodes"][0]
    assert dataset_manifest["episodes"][0]["archive_ref"] == (
        f"episodes/{job.program_id}/{job.episode_id}"
    )

    resume = json.loads((queue.root / "resume_receipt.json").read_text(encoding="utf-8"))
    assert resume["source_run_ids"] == [job.run_id]
    assert resume["dataset_identity"] == profile.dataset_identity
    assert resume["archive_format_id"] == profile.archive_format_id
    assert resume["episode_schema_version"] == profile.episode_schema_version
    assert resume["dataset_manifest_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    remote_receipt = json.loads((queue.root / "publication_receipt.json").read_text(encoding="utf-8"))
    assert remote_receipt["source_run_id"] == job.run_id
    assert remote_receipt["dataset_identity"] == profile.dataset_identity
    assert remote_receipt["archive_format_id"] == profile.archive_format_id
    assert remote_receipt["episode_schema_version"] == profile.episode_schema_version
    assert remote_receipt["artifact_hashes"] == receipt.artifact_hashes

    for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
        view_payload = json.loads((queue.root / "views" / f"{view}.json").read_text(encoding="utf-8"))
        assert view_payload["status"] == "PROVISIONAL"
        assert view_payload["archive_refs"] == [
            f"episodes/{job.program_id}/{job.episode_id}"
        ]
        assert "episode_ids" not in view_payload
        assert "schema_version" not in view_payload

        episode_pointer_path = (
            queue.root / "views" / "provisional" / "episodes"
            / job.program_id / job.episode_id / f"{view}.json"
        )
        episode_pointer = json.loads(episode_pointer_path.read_text(encoding="utf-8"))
        assert episode_pointer["status"] == "PROVISIONAL"
        assert episode_pointer["role"] == "all"
        assert episode_pointer["mix"] is False
        assert episode_pointer["archive_ref"] == f"episodes/{job.program_id}/{job.episode_id}"


def test_validation_publisher_rejects_receipt_only_archive_retention(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    run = RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
        hf_subfolder="validation/validation-test", validation_mode=True,
    )
    profile = ArchiveProfileConfig(local_artifact_retention="receipt_only")

    with pytest.raises(ValueError, match="validation mode.*keep"):
        HuggingFaceBatchPublisher(
            run, FakeApi(), "secret", queue, archive_profile=profile,
        )


def test_publication_limits_large_lfs_commit_concurrency(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(_run(), api, "secret", queue, last_success_s=0.0)
    publisher.publish_due(now_s=300.0, force=False)
    assert api.calls[0]["num_threads"] == 1
    assert publisher.MAX_JOBS_PER_COMMIT == 1


def test_conflicting_remote_manifest_refuses_before_api_call(tmp_path: Path):
    queue, job = _queue(tmp_path)
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(_run(), api, "secret", queue, last_success_s=0.0)
    with pytest.raises(ValueError, match="remote immutable conflict"):
        publisher.publish_due(
            now_s=300.0, force=False,
            remote_manifest={"episodes": [{"episode_id": job.episode_id, "program_id": "other"}]},
        )
    assert not api.calls


def test_publication_config_controls_upload_threads_and_retry_policy(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    config = PublicationConfig(
        batch_size=3,
        upload_threads=4,
        retry_attempts=2,
        retry_cooldown_s=17.0,
        rate_limit_cooldown_s=23.0,
    )
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        publication_config=config,
    )

    publisher.publish_due(now_s=300.0, force=False)

    assert publisher.publication_config == config
    assert api.calls[0]["num_threads"] == 4
    assert publisher._next_retry_s == 0.0


def test_data_commit_timeout_persists_deferred_receipt_without_publishing(tmp_path: Path, monkeypatch):
    queue, job = _queue(tmp_path)
    monkeypatch.setattr(
        "icgs.data.collection.generation.distributed_publication.time.sleep",
        lambda _seconds: None,
    )
    api = DataCommitThenTimeoutApi()
    config = PublicationConfig(
        batch_size=1,
        upload_threads=2,
        retry_attempts=1,
        retry_cooldown_s=31.0,
        rate_limit_cooldown_s=47.0,
    )
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        publication_config=config,
    )

    assert publisher.publish_due(now_s=300.0, force=False) is None
    receipt = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "DEFERRED"
    assert receipt["data_commit_oid"] == "d" * 40
    assert receipt["job_ids"] == [job.job_id]
    assert queue.counts().ingested == 1
    assert queue.counts().published == 0
    assert len(api.calls) == 2


def test_matching_remote_hashes_reconcile_deferred_receipt_without_reupload(tmp_path: Path, monkeypatch):
    queue, job = _queue(tmp_path)
    monkeypatch.setattr(
        "icgs.data.collection.generation.distributed_publication.time.sleep",
        lambda _seconds: None,
    )
    api = DataCommitThenTimeoutApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        publication_config=PublicationConfig(retry_attempts=1),
    )
    assert publisher.publish_due(now_s=300.0, force=False) is None
    local = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    remote = {
        "publication_receipt": {
            "status": "DATA_COMMITTED",
            "run_id": "run-1",
            "job_ids": [job.job_id],
            "data_commit_oid": "d" * 40,
            "prefix": "generation",
            "artifact_hashes": local["artifact_hashes"],
        },
        "artifact_hashes": local["artifact_hashes"],
    }

    receipt = publisher.publish_due(
        now_s=400.0,
        force=True,
        remote_manifest=remote,
    )

    assert receipt is not None
    assert receipt.status == "COMPLETE"
    assert len(api.calls) == 2
    assert queue.counts().ingested == 0
    assert queue.counts().published == 1
    resolved = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert resolved["status"] == "COMPLETE"


def test_conflicting_remote_data_commit_fails_closed_without_retry(tmp_path: Path, monkeypatch):
    queue, _job_value = _queue(tmp_path)
    monkeypatch.setattr(
        "icgs.data.collection.generation.distributed_publication.time.sleep",
        lambda _seconds: None,
    )
    api = DataCommitThenTimeoutApi()
    publisher = HuggingFaceBatchPublisher(
        _run(), api, "secret", queue, last_success_s=0.0,
        publication_config=PublicationConfig(retry_attempts=1),
    )
    assert publisher.publish_due(now_s=300.0, force=False) is None
    local = json.loads(
        (queue.root / "publication_receipt.json").read_text(encoding="utf-8")
    )
    remote = {
        "publication_receipt": {
            "status": "DATA_COMMITTED",
            "run_id": "run-1",
            "job_ids": local["job_ids"],
            "data_commit_oid": "x" * 40,
            "prefix": "generation",
            "artifact_hashes": local["artifact_hashes"],
        },
        "artifact_hashes": local["artifact_hashes"],
    }

    with pytest.raises(ValueError, match="immutable publication identity"):
        publisher.publish_due(now_s=400.0, force=True, remote_manifest=remote)
    assert len(api.calls) == 2
    assert queue.counts().ingested == 1


def test_validation_mode_publication_prefix_isolated_to_validation_cpu_run(tmp_path: Path):
    queue, job = _queue(tmp_path)
    run = RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
        hf_subfolder="validation/validation-cpu-20260922/run-1",
        validation_mode=True,
    )
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(run, api, "secret", queue, last_success_s=0.0)
    publisher.publish_due(now_s=300.0, force=False)

    paths = [operation.path_in_repo for operation in api.calls[0]["operations"]]
    assert all(path.startswith("validation/validation-cpu-20260922/") for path in paths)
    assert any(f"episodes/{job.program_id}/{job.episode_id}/" in path for path in paths)


def test_validation_mode_rejects_historical_publication_prefix_before_api_call(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    run = RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
        hf_subfolder="validation/old-run",
        validation_mode=True,
    )
    api = FakeApi()
    publisher = HuggingFaceBatchPublisher(run, api, "secret", queue, last_success_s=0.0)

    with pytest.raises(ValueError, match="validation-cpu-20260922"):
        publisher.publish_due(now_s=300.0, force=False)
    assert api.calls == []


def test_reconcile_publication_rejects_remote_job_identity_conflict():
    receipt = PublicationReceipt.from_dict({
        "receipt_version": 2,
        "run_id": "run-1",
        "job_ids": ["job-1"],
        "status": "DATA_COMMITTED",
        "data_commit_oid": "d" * 40,
        "artifact_hashes": {"job-1/episode.json": "a" * 64},
        "prefix": "generation",
        "path_count": 1,
        "updated_at_s": 300.0,
    })
    with pytest.raises(ValueError, match="immutable publication identity"):
        reconcile_publication(receipt, {
            "publication_receipt": {
                "status": "DATA_COMMITTED",
                "run_id": "run-1",
                "job_ids": ["job-2"],
                "data_commit_oid": "d" * 40,
                "artifact_hashes": {"job-1/episode.json": "a" * 64},
                "prefix": "generation",
            },
            "artifact_hashes": {"job-1/episode.json": "a" * 64},
        })
