from __future__ import annotations

from pathlib import Path

import pytest

from icgs.data.collection.generation.distributed_contracts import GenerationJob, RunConfig, WorkerResult
from icgs.data.collection.generation.distributed_publication import HuggingFaceBatchPublisher
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
        randomization={"scene_signature": "sig-1", "asset_instance_id": "asset-1"},
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


def test_publish_due_only_after_300_seconds_or_force(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue, last_success_s=100.0)
    assert publisher.publish_due(now_s=399.9, force=False) is None
    receipt = publisher.publish_due(now_s=400.0, force=False)
    assert receipt is not None
    assert receipt.job_ids == (job.job_id,)
    assert publisher.remote_manifest["episodes"][0]["episode_id"] == job.episode_id


def test_pending_job_limit_must_be_positive(tmp_path: Path):
    queue, _job_value = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(), "secret", queue)
    with pytest.raises(ValueError, match="limit must be positive"):
        publisher.pending_job_ids(limit=0)


def test_failed_commit_keeps_jobs_unpublished(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(fail_create_commit=True), "secret", queue, last_success_s=0.0)
    with pytest.raises(RuntimeError, match="429"):
        publisher.publish_due(now_s=300.0, force=False)
    assert publisher.pending_job_ids() == (job.job_id,)
    assert queue.counts().ingested == 1


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
