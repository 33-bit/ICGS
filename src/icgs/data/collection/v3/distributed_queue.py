"""Atomic same-filesystem queue for distributed primary-v3 collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from icgs.data.collection.v3.distributed_contracts import GenerationJob, QueueCounts, WorkerResult


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    data = _canonical_json(payload)
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class FilesystemJobQueue:
    _STATES = ("pending", "claimed", "ready", "ingested", "published")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for state in self._STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        (self.root / "heartbeats").mkdir(parents=True, exist_ok=True)

    def _find_job(self, state: str, job_id: str) -> Path | None:
        base = self.root / state
        if state == "claimed":
            matches = sorted(base.glob(f"*/{job_id}.json"))
            return matches[0] if matches else None
        path = base / (job_id if state in {"ready", "ingested", "published"} else f"{job_id}.json")
        return path if path.exists() else None

    def enqueue(self, job: GenerationJob) -> Path:
        expected = _canonical_json(job.as_dict())
        for state in self._STATES:
            existing = self._find_job(state, job.job_id)
            if existing is None:
                continue
            job_file = existing / "job.json" if existing.is_dir() else existing
            if job_file.read_bytes() != expected:
                raise ValueError(f"immutable job conflict: {job.job_id}")
            return existing
        path = self.root / "pending" / f"{job.job_id}.json"
        _atomic_write(path, job.as_dict())
        return path

    def claim(self, worker_id: str, *, now_s: float | None = None) -> GenerationJob | None:
        worker_dir = self.root / "claimed" / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted((self.root / "pending").glob("*.json")):
            target = worker_dir / source.name
            try:
                os.replace(source, target)
            except FileNotFoundError:
                continue
            job = GenerationJob.from_dict(_read_json(target))
            _atomic_write(
                worker_dir / f"{job.job_id}.claim.json",
                {"worker_id": worker_id, "job_id": job.job_id, "claimed_at_s": time.time() if now_s is None else now_s},
            )
            return job
        return None

    def write_heartbeat(self, worker_id: str, job_id: str | None, *, now_s: float | None = None) -> Path:
        path = self.root / "heartbeats" / f"worker-{worker_id}.json"
        _atomic_write(path, {"worker_id": worker_id, "job_id": job_id, "timestamp_s": time.time() if now_s is None else now_s})
        return path

    def publish_ready(self, worker_id: str, result: WorkerResult) -> Path:
        worker_dir = self.root / "claimed" / worker_id
        claimed = worker_dir / f"{result.job_id}.json"
        if not claimed.is_file():
            raise FileNotFoundError(f"worker {worker_id} does not own {result.job_id}")
        job = GenerationJob.from_dict(_read_json(claimed))
        if (job.attempt_id, job.program_id) != (result.attempt_id, result.program_id):
            raise ValueError("worker result identity does not match claimed job")
        partial = self.root / "ready" / f"{result.job_id}.partial-{os.getpid()}-{time.time_ns()}"
        partial.mkdir(parents=True)
        _atomic_write(partial / "job.json", job.as_dict())
        _atomic_write(partial / "result.json", result.as_dict())
        target = self.root / "ready" / result.job_id
        if target.exists():
            raise ValueError(f"ready result already exists: {result.job_id}")
        os.replace(partial, target)
        claimed.unlink()
        claim_meta = worker_dir / f"{result.job_id}.claim.json"
        claim_meta.unlink(missing_ok=True)
        return target

    def iter_ready(self) -> tuple[WorkerResult, ...]:
        return tuple(
            WorkerResult.from_dict(_read_json(directory / "result.json"))
            for directory in sorted((self.root / "ready").iterdir())
            if directory.is_dir() and ".partial-" not in directory.name
        )

    def mark_ingested(self, result: WorkerResult) -> Path:
        source = self.root / "ready" / result.job_id
        target = self.root / "ingested" / result.job_id
        if target.exists():
            return target
        os.replace(source, target)
        return target

    def mark_published(self, job_id: str) -> Path:
        source = self.root / "ingested" / job_id
        target = self.root / "published" / job_id
        if target.exists():
            return target
        os.replace(source, target)
        return target

    def recover_stale(self, *, now_s: float, stale_after_s: float) -> list[str]:
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        recovered: list[str] = []
        for job_path in sorted((self.root / "claimed").glob("*/*.json")):
            if job_path.name.endswith(".claim.json"):
                continue
            claim_path = job_path.with_name(job_path.stem + ".claim.json")
            claimed_at = _read_json(claim_path).get("claimed_at_s") if claim_path.is_file() else job_path.stat().st_mtime
            if now_s - float(claimed_at) < stale_after_s:
                continue
            target = self.root / "pending" / job_path.name
            if target.exists():
                raise ValueError(f"pending job already exists while recovering {job_path.stem}")
            os.replace(job_path, target)
            claim_path.unlink(missing_ok=True)
            recovered.append(job_path.stem)
        return recovered

    def counts(self) -> QueueCounts:
        return QueueCounts(
            pending=sum(1 for _ in (self.root / "pending").glob("*.json")),
            claimed=sum(1 for path in (self.root / "claimed").glob("*/*.json") if not path.name.endswith(".claim.json")),
            ready=sum(1 for path in (self.root / "ready").iterdir() if path.is_dir() and ".partial-" not in path.name),
            ingested=sum(1 for path in (self.root / "ingested").iterdir() if path.is_dir()),
            published=sum(1 for path in (self.root / "published").iterdir() if path.is_dir()),
        )


__all__ = ["FilesystemJobQueue"]
