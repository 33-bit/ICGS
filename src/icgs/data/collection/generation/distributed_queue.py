"""Atomic same-filesystem queue for distributed generation collection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from icgs.data.collection.generation.distributed_contracts import GenerationJob, QueueCounts, WorkerResult
from icgs.data.collection.generation.distributed_validation import ValidationFailure


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


@dataclass(frozen=True)
class QueueCountsWithQuarantine(QueueCounts):
    quarantined: int


@dataclass(frozen=True)
class MalformedReadyResult:
    job_id: str
    result_identity: dict[str, Any]
    error: Exception


class FilesystemJobQueue:
    _STATES = ("pending", "claimed", "ready", "ingested", "published", "quarantined")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for state in self._STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        (self.root / "heartbeats").mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _operation_lock(self):
        """Serialize multi-file queue transitions across hosts on one POSIX mount."""
        lock_path = self.root / "control" / "queue.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def _find_job(self, state: str, job_id: str) -> Path | None:
        if state not in self._STATES:
            raise ValueError(f"unknown queue state: {state}")
        self._validate_job_component(job_id)
        base = self.root / state
        if base.is_symlink() or not base.is_dir():
            raise ValueError(f"queue state must be a real directory: {base}")
        if state == "claimed":
            matches = []
            for worker_directory in sorted(base.iterdir()):
                try:
                    if not stat.S_ISDIR(worker_directory.lstat().st_mode):
                        continue
                except FileNotFoundError:
                    continue
                candidate = worker_directory / f"{job_id}.json"
                if candidate.is_symlink():
                    raise ValueError(f"queue job path must not be a symlink: {candidate}")
                if candidate.exists():
                    matches.append(candidate)
            return matches[0] if matches else None
        path = base / (
            job_id
            if state in {"ready", "ingested", "published", "quarantined"}
            else f"{job_id}.json"
        )
        if path.is_symlink():
            raise ValueError(f"queue job path must not be a symlink: {path}")
        return path if path.exists() else None

    def _worker_lease_path(self, worker_id: str) -> Path:
        self._validate_job_component(worker_id)
        return self.root / "heartbeats" / f"worker-{worker_id}.json"

    def _register_worker_unlocked(
        self,
        worker_id: str,
        host_id: str,
        worker_instance_id: str,
        *,
        now_s: float | None = None,
        lease_s: float = 1800.0,
    ) -> Path:
        if not host_id or not worker_instance_id:
            raise ValueError("host_id and worker_instance_id are required")
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        now = time.time() if now_s is None else float(now_s)
        path = self._worker_lease_path(worker_id)
        if path.is_file():
            existing = _read_json(path)
            expires = float(existing.get("lease_expires_at_s", 0.0))
            current = str(existing.get("worker_instance_id", ""))
            if expires > now and current and current != worker_instance_id:
                raise RuntimeError(f"worker lease is active for {worker_id}")
        _atomic_write(path, {
            "worker_id": worker_id,
            "host_id": host_id,
            "worker_instance_id": worker_instance_id,
            "job_id": None,
            "timestamp_s": now,
            "lease_expires_at_s": now + float(lease_s),
        })
        return path

    def register_worker(
        self,
        worker_id: str,
        host_id: str,
        worker_instance_id: str,
        *,
        now_s: float | None = None,
        lease_s: float = 1800.0,
    ) -> Path:
        with self._operation_lock():
            return self._register_worker_unlocked(
                worker_id,
                host_id,
                worker_instance_id,
                now_s=now_s,
                lease_s=lease_s,
            )

    def _assert_worker_lease(
        self,
        worker_id: str,
        worker_instance_id: str | None,
        *,
        now_s: float | None = None,
    ) -> None:
        if worker_instance_id is None:
            return
        path = self._worker_lease_path(worker_id)
        if not path.is_file():
            raise RuntimeError(f"worker lease is missing for {worker_id}")
        lease = _read_json(path)
        now = time.time() if now_s is None else float(now_s)
        if str(lease.get("worker_instance_id")) != worker_instance_id:
            raise RuntimeError(f"worker lease is owned by another instance: {worker_id}")
        if float(lease.get("lease_expires_at_s", 0.0)) <= now:
            raise RuntimeError(f"worker lease expired: {worker_id}")

    @staticmethod
    def _validate_job_component(job_id: str) -> None:
        if (
            not isinstance(job_id, str)
            or not job_id
            or job_id in {".", ".."}
            or Path(job_id).is_absolute()
            or "/" in job_id
            or "\\" in job_id
            or "\x00" in job_id
        ):
            raise ValueError("job_id must be a safe path component")

    def enqueue(self, job: GenerationJob) -> Path:
        expected = _canonical_json(job.as_dict())
        with self._operation_lock():
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

    def claim(
        self,
        worker_id: str,
        *,
        now_s: float | None = None,
        worker_instance_id: str | None = None,
        host_id: str | None = None,
        lease_s: float = 1800.0,
    ) -> GenerationJob | None:
        with self._operation_lock():
            if worker_instance_id is not None:
                self._register_worker_unlocked(
                    worker_id,
                    host_id or "unknown",
                    worker_instance_id,
                    now_s=now_s,
                    lease_s=lease_s,
                )
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
                    {
                        "worker_id": worker_id,
                        "host_id": host_id,
                        "worker_instance_id": worker_instance_id,
                        "job_id": job.job_id,
                        "claimed_at_s": time.time() if now_s is None else now_s,
                    },
                )
                return job
            return None

    def write_heartbeat(
        self,
        worker_id: str,
        job_id: str | None,
        *,
        now_s: float | None = None,
        host_id: str | None = None,
        worker_instance_id: str | None = None,
        lease_s: float = 1800.0,
    ) -> Path:
        now = time.time() if now_s is None else float(now_s)
        with self._operation_lock():
            if worker_instance_id is not None:
                self._assert_worker_lease(worker_id, worker_instance_id, now_s=now)
            path = self._worker_lease_path(worker_id)
            _atomic_write(path, {
                "worker_id": worker_id,
                "host_id": host_id,
                "worker_instance_id": worker_instance_id,
                "job_id": job_id,
                "timestamp_s": now,
                "lease_expires_at_s": now + float(lease_s),
            })
            return path

    def publish_ready(
        self,
        worker_id: str,
        result: WorkerResult,
        *,
        worker_instance_id: str | None = None,
    ) -> Path:
        with self._operation_lock():
            self._assert_worker_lease(worker_id, worker_instance_id)
            worker_dir = self.root / "claimed" / worker_id
            claimed = worker_dir / f"{result.job_id}.json"
            if not claimed.is_file():
                raise FileNotFoundError(f"worker {worker_id} does not own {result.job_id}")
            job = GenerationJob.from_dict(_read_json(claimed))
            if (job.attempt_id, job.program_id) != (result.attempt_id, result.program_id):
                raise ValueError("worker result identity does not match claimed job")
            claim_meta = worker_dir / f"{result.job_id}.claim.json"
            if worker_instance_id is not None and claim_meta.is_file():
                metadata = _read_json(claim_meta)
                if metadata.get("worker_instance_id") != worker_instance_id:
                    raise RuntimeError("worker claim is owned by another instance")
            partial = self.root / "ready" / f"{result.job_id}.partial-{os.getpid()}-{time.time_ns()}"
            partial.mkdir(parents=True)
            _atomic_write(partial / "job.json", job.as_dict())
            _atomic_write(partial / "result.json", result.as_dict())
            target = self.root / "ready" / result.job_id
            if target.exists():
                raise ValueError(f"ready result already exists: {result.job_id}")
            os.replace(partial, target)
            claimed.unlink()
            claim_meta.unlink(missing_ok=True)
            return target

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"expected a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"expected a regular file: {path}")
            stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            with stream:
                return stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def iter_ready(self) -> tuple[WorkerResult | MalformedReadyResult, ...]:
        results: list[WorkerResult | MalformedReadyResult] = []
        ready_root = self.root / "ready"
        if ready_root.is_symlink() or not ready_root.is_dir():
            raise ValueError(f"queue state must be a real directory: {ready_root}")
        for directory in sorted(ready_root.iterdir()):
            if ".partial-" in directory.name:
                continue
            raw_result = b""
            payload: Any = None
            try:
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    continue
                result_path = directory / "result.json"
                raw_result = self._read_regular_file(result_path)
                payload = json.loads(raw_result)
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object: {result_path}")
                result = WorkerResult.from_dict(payload)
                if result.job_id != directory.name:
                    raise ValueError("ready directory/result job identity mismatch")
            except Exception as error:
                if not raw_result:
                    try:
                        raw_result = self._read_regular_file(directory / "result.json")
                    except (OSError, ValueError):
                        raw_result = b""
                identity: dict[str, Any] = {
                    "job_id": directory.name,
                    "result_json_sha256": hashlib.sha256(raw_result).hexdigest(),
                }
                if isinstance(payload, dict):
                    for key in ("attempt_id", "episode_id", "program_id", "outcome", "result_dir"):
                        if key in payload and (
                            payload[key] is None or isinstance(payload[key], str)
                        ):
                            identity[key] = payload[key]
                    declared_job_id = payload.get("job_id")
                    if (
                        isinstance(declared_job_id, str)
                        and declared_job_id != directory.name
                    ):
                        identity["declared_job_id"] = declared_job_id
                identity["parse_error_type"] = type(error).__name__
                identity["parse_error_message"] = str(error)
                results.append(MalformedReadyResult(directory.name, identity, error))
            else:
                results.append(result)
        return tuple(results)

    def mark_ingested(self, result: WorkerResult) -> Path:
        with self._operation_lock():
            source = self.root / "ready" / result.job_id
            target = self.root / "ingested" / result.job_id
            if target.exists():
                return target
            os.replace(source, target)
            return target

    def quarantine_ready(self, job_id: str, failure: ValidationFailure) -> Path:
        self._validate_job_component(job_id)
        if not isinstance(failure, ValidationFailure):
            raise TypeError("failure must be a ValidationFailure")
        if failure.job_id != job_id:
            raise ValueError("quarantine failure identity does not match job_id")
        if failure.result_identity.get("job_id") != job_id:
            raise ValueError("quarantine result identity does not match job_id")

        with self._operation_lock():
            ready_root = self.root / "ready"
            quarantine_root = self.root / "quarantined"
            for state_root in (ready_root, quarantine_root):
                if state_root.is_symlink() or not state_root.is_dir():
                    raise ValueError(f"queue state must be a real directory: {state_root}")

            source = ready_root / job_id
            target = quarantine_root / job_id
            if failure.source_path != str(source):
                raise ValueError("quarantine source_path does not match ready directory")
            if source.is_symlink():
                raise ValueError(f"ready result must be a real directory: {source}")

            expected = _canonical_json(failure.as_dict())
            if target.is_symlink() or target.exists():
                if target.is_symlink() or not target.is_dir():
                    raise ValueError(f"quarantine target must be a real directory: {target}")
                if source.exists():
                    raise ValueError(f"ready and quarantined results both exist: {job_id}")
                diagnostic = target / "validation_failure.json"
                if diagnostic.is_symlink():
                    raise ValueError(f"quarantine diagnostic must not be a symlink: {diagnostic}")
                if not diagnostic.exists():
                    _atomic_write(diagnostic, failure.as_dict())
                    return target
                if diagnostic.read_bytes() != expected:
                    raise ValueError(f"quarantine conflict: {job_id}")
                return target

            if not source.is_dir():
                raise FileNotFoundError(f"ready result does not exist: {job_id}")
            os.replace(source, target)
            _atomic_write(target / "validation_failure.json", failure.as_dict())
            return target

    def mark_published(self, job_id: str) -> Path:
        with self._operation_lock():
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
        with self._operation_lock():
            for job_path in sorted((self.root / "claimed").glob("*/*.json")):
                if job_path.name.endswith(".claim.json"):
                    continue
                claim_path = job_path.with_name(job_path.stem + ".claim.json")
                claim = _read_json(claim_path) if claim_path.is_file() else {}
                claimed_at = claim.get("claimed_at_s", job_path.stat().st_mtime)
                worker_id = job_path.parent.name
                heartbeat_path = self._worker_lease_path(worker_id)
                heartbeat = _read_json(heartbeat_path) if heartbeat_path.is_file() else {}
                heartbeat_job_id = heartbeat.get("job_id")
                if heartbeat_job_id == job_path.stem:
                    lease_expires = float(heartbeat.get("lease_expires_at_s", 0.0))
                    if lease_expires > now_s:
                        claimed_at = max(float(claimed_at), float(heartbeat.get("timestamp_s", claimed_at)))
                    else:
                        claimed_at = now_s - stale_after_s
                if now_s - float(claimed_at) < stale_after_s:
                    continue
                target = self.root / "pending" / job_path.name
                if target.exists():
                    raise ValueError(f"pending job already exists while recovering {job_path.stem}")
                job = GenerationJob.from_dict(_read_json(job_path))
                retry = replace(job, retry_generation=job.retry_generation + 1)
                _atomic_write(target, retry.as_dict())
                job_path.unlink()
                claim_path.unlink(missing_ok=True)
                recovered.append(job_path.stem)
        return recovered

    def counts(self) -> QueueCountsWithQuarantine:
        def directory_count(state: str, *, exclude_partial: bool = False) -> int:
            state_root = self.root / state
            if state_root.is_symlink() or not state_root.is_dir():
                raise ValueError(f"queue state must be a real directory: {state_root}")
            count = 0
            for path in state_root.iterdir():
                if exclude_partial and ".partial-" in path.name:
                    continue
                try:
                    count += stat.S_ISDIR(path.lstat().st_mode)
                except FileNotFoundError:
                    continue
            return count

        return QueueCountsWithQuarantine(
            pending=sum(1 for _ in (self.root / "pending").glob("*.json")),
            claimed=sum(1 for path in (self.root / "claimed").glob("*/*.json") if not path.name.endswith(".claim.json")),
            ready=directory_count("ready", exclude_partial=True),
            ingested=directory_count("ingested"),
            published=directory_count("published"),
            quarantined=directory_count("quarantined"),
        )


__all__ = ["FilesystemJobQueue", "MalformedReadyResult"]
