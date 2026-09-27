"""Run-wide admission must outlive worker leases and publication outages."""
from dataclasses import replace
import multiprocessing
from pathlib import Path

import pytest

from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from test_generation_queue import _job, _result, _write_verified_archive_receipt

MIB = 1024 * 1024


def _queue(root, *, slots=1):
    queue = FilesystemJobQueue(root / "queue")
    queue.configure_staging(
        max_result_bytes=MIB,
        max_staging_bytes=(slots + 3) * MIB,
        staging_reserve_bytes=2 * MIB,
    )
    for index in range(4):
        queue.enqueue(replace(_job(f"job-{index}", index), output_root=str(root / "staging")))
    return queue


def _claim(root, worker, result):
    queue = FilesystemJobQueue(Path(root) / "queue")
    job = queue.claim(worker)
    result.put(job.job_id if job else None)


def test_staging_blocks_concurrent_claims_and_survives_reopen(tmp_path):
    queue = _queue(tmp_path)
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    children = [context.Process(target=_claim, args=(str(tmp_path), str(i), result)) for i in range(4)]
    for child in children:
        child.start()
    for child in children:
        child.join(10)
        assert child.exitcode == 0
    assert sum(result.get(timeout=2) is not None for _ in children) == 1
    assert FilesystemJobQueue(queue.root).claim("another") is None
    assert queue.staging_status()["reserved_bytes"] == MIB


def test_expired_claim_does_not_release_storage(tmp_path):
    queue = _queue(tmp_path)
    job = queue.claim("000", now_s=1)
    assert job is not None
    assert queue.recover_stale(now_s=200, stale_after_s=10) == [job.job_id]
    assert queue.claim("001") is None
    assert queue.staging_status()["reserved_bytes"] == MIB


def test_keep_retention_and_ready_backlog_keep_reservation(tmp_path):
    queue = _queue(tmp_path)
    job = queue.claim("000")
    root = Path(job.output_root) / "worker-results" / job.job_id / "retry-0" / "result"
    root.mkdir(parents=True)
    result = _result(job, root)
    queue.publish_ready("000", result)
    assert queue.claim("001") is None
    queue.mark_ingested(result)
    queue.mark_published(job.job_id, retention="keep")
    assert queue.claim("001") is None


def test_verified_pruning_releases_capacity_but_failed_verification_does_not(tmp_path):
    import hashlib
    queue = _queue(tmp_path)
    job = queue.claim("000")
    root = Path(job.output_root) / "worker-results" / job.job_id / "retry-0" / "result"
    root.mkdir(parents=True)
    data = b"canonical archive manifest\n"
    (root / "episode_manifest.json").write_bytes(data)
    result = replace(_result(job, root), file_sha256={"episode_manifest.json": hashlib.sha256(data).hexdigest()})
    queue.publish_ready("000", result)
    queue.mark_ingested(result)
    with pytest.raises(ValueError, match="verified publication receipt"):
        queue.mark_published(job.job_id, retention="receipt_only")
    assert queue.claim("001") is None
    _write_verified_archive_receipt(queue, job, result)
    queue.mark_published(job.job_id, retention="receipt_only")
    assert not root.exists()
    assert queue.staging_status()["reserved_bytes"] == 0
    assert queue.claim("001") is not None


def test_unreserved_bytes_and_free_disk_block_claims(tmp_path, monkeypatch):
    from collections import namedtuple
    from icgs.data.collection.generation import distributed_queue
    queue = _queue(tmp_path, slots=2)
    (tmp_path / "retained.bin").write_bytes(b"x" * (2 * MIB))
    assert queue.claim("000") is None
    (tmp_path / "retained.bin").unlink()
    disk = namedtuple("usage", "total used free")
    monkeypatch.setattr(distributed_queue.shutil, "disk_usage", lambda _: disk(100 * MIB, 99 * MIB, MIB))
    assert queue.claim("000") is None


def test_staging_policy_cannot_change_on_attach(tmp_path):
    queue = _queue(tmp_path)
    with pytest.raises(ValueError, match="staging.*conflict"):
        queue.configure_staging(max_result_bytes=MIB, max_staging_bytes=8 * MIB, staging_reserve_bytes=2 * MIB)


def test_reservation_written_before_claim_move_is_not_released_on_restart(tmp_path, monkeypatch):
    from icgs.data.collection.generation import distributed_queue
    queue = _queue(tmp_path)
    original = distributed_queue.os.replace
    def interrupt(source, target):
        if Path(source).parent.name == "pending":
            raise OSError("crash before claim rename")
        return original(source, target)
    monkeypatch.setattr(distributed_queue.os, "replace", interrupt)
    with pytest.raises(OSError, match="crash"):
        queue.claim("000")
    monkeypatch.setattr(distributed_queue.os, "replace", original)
    assert FilesystemJobQueue(queue.root).claim("001") is None


def test_staging_policy_rejects_unsafe_limits(tmp_path):
    queue = FilesystemJobQueue(tmp_path / "queue")
    for cap, reserve in [(True, 2 * MIB), (0, 2 * MIB), (4 * MIB, MIB), (3 * MIB, 2 * MIB)]:
        with pytest.raises(ValueError, match="staging"):
            queue.configure_staging(max_result_bytes=MIB, max_staging_bytes=cap, staging_reserve_bytes=reserve)


def test_runtime_budget_roundtrip_and_worker_admission(tmp_path):
    from test_generation_config import _portable_payload
    from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig, ArchiveProfileConfig
    from scripts.generation_worker import run_worker
    payload = _portable_payload(tmp_path)
    payload.update(archive_profile=ArchiveProfileConfig().as_dict(), max_result_bytes=MIB,
                   max_staging_bytes=4 * MIB, staging_reserve_bytes=2 * MIB)
    config = GenerationRuntimeConfig.from_dict(payload)
    assert GenerationRuntimeConfig.from_dict(config.as_dict()) == config
    root = Path(config.run.run_root)
    queue = FilesystemJobQueue(root / "queue")
    for index in range(2):
        queue.enqueue(replace(_job(f"job-{index}", index), output_root=str(root / "staging")))
    # Existing retained bytes consume the available admission capacity. A worker
    # must not even read the manifest/start the supplied runner in this state.
    (root / "retained").write_bytes(b"x" * (2 * MIB))
    def forbidden(*args, **kwargs):
        pytest.fail("backpressured worker started a simulator")
    assert run_worker("000", queue, config=config, approved_manifest="absent.json",
                      once=True, runner=forbidden) == 0
    assert queue.counts().claimed == 0
    assert queue.staging_status()["can_claim"] is False


@pytest.mark.parametrize("entry", ["launch", "worker"])
def test_production_entry_requires_shared_budget_before_start(tmp_path, entry):
    import json
    from test_generation_config import _portable_payload
    from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig
    from scripts import generation_launch, generation_worker
    payload = _portable_payload(tmp_path)
    payload.update(archive_profile=ArchiveProfileConfig().as_dict(), max_result_bytes=MIB)
    payload["run"]["validation_mode"] = False
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(payload))
    args = ["--runtime-config", str(config_path), "--approved-manifest", "absent.json"]
    if entry == "worker":
        args += ["--worker-id", "000", "--once"]
    with pytest.raises(ValueError, match="max_staging_bytes"):
        (generation_launch if entry == "launch" else generation_worker).main(args)


def test_core_worker_rejects_production_without_budget(tmp_path):
    from test_generation_config import _portable_payload
    from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig, ArchiveProfileConfig
    from scripts.generation_worker import run_worker
    payload = _portable_payload(tmp_path)
    payload.update(archive_profile=ArchiveProfileConfig().as_dict(), max_result_bytes=MIB)
    payload["run"]["validation_mode"] = False
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    with pytest.raises(ValueError, match="max_staging_bytes"):
        run_worker("000", queue, config=config, approved_manifest="absent.json", once=True)


def test_budgeted_remote_scratch_uses_reserved_run_not_system_tmp(tmp_path, monkeypatch):
    from scripts import generation_coordinator as coordinator
    import tempfile
    queue = _queue(tmp_path / "run")
    system_tmp = tmp_path / "other-volume"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    run = type("Run", (), {"run_root": str(queue.root.parent)})()
    assert hasattr(coordinator, "_run_scratch"), "remote scratch must be bound to its reserved filesystem"
    with coordinator._run_scratch(run, prefix="test-") as scratch:
        assert Path(scratch).is_relative_to(queue.root.parent)
    assert not Path(scratch).exists()


def test_contained_hf_cache_links_do_not_block_admission(tmp_path):
    queue = _queue(tmp_path, slots=2)
    scratch = tmp_path / "hf-scratch" / "cache"
    scratch.mkdir(parents=True)
    (scratch / "blob").write_bytes(b"verified bytes")
    (scratch / "snapshot").symlink_to("blob")
    assert queue.claim("000") is not None
    (scratch / "external").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="symlink"):
        queue.claim("001")
