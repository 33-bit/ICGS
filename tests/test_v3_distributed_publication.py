from __future__ import annotations

from pathlib import Path

import pytest

from icgs.data.collection.v3.distributed_contracts import GenerationJob, RunConfig, WorkerResult
from icgs.data.collection.v3.distributed_publication import HuggingFaceBatchPublisher
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


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


def _run() -> RunConfig:
    return RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
    )


def _job() -> GenerationJob:
    from icgs.data.collection.v3.batch import AttemptPlan
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="v3-t01-00001", episode_kind="nominal", scene_seed=1,
        collection_seed=20260920,
        randomization={"scene_signature": "sig-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    return GenerationJob.create(
        job_id="job-run-1-v3-t01-00001", run_id="run-1",
        attempt_id="att-v3-t01-00001", episode_id=plan.episode_id,
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
        '{"episode_id": "v3-t01-00001", "program_id": "T01", "outcome": "success"}\n',
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


def test_failed_commit_keeps_jobs_unpublished(tmp_path: Path):
    queue, job = _queue(tmp_path)
    publisher = HuggingFaceBatchPublisher(_run(), FakeApi(fail_create_commit=True), "secret", queue, last_success_s=0.0)
    with pytest.raises(RuntimeError, match="429"):
        publisher.publish_due(now_s=300.0, force=False)
    assert publisher.pending_job_ids() == (job.job_id,)
    assert queue.counts().ingested == 1


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
