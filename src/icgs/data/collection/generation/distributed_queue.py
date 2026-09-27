"""Atomic same-filesystem queue for distributed generation collection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any, Literal

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    QueueCounts,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_validation import (
    ValidationFailure,
    validate_archive_manifest_row_identity,
)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    data = _canonical_json(payload)
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_durable_safety_record(path: Path, payload: dict[str, Any]) -> None:
    """Persist the safety record and each directory entry leading to it."""
    _atomic_write(path, payload)
    directory = path.parent.absolute()
    while True:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if directory == directory.parent:
            break
        directory = directory.parent


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


class GenerationSafetyStop(RuntimeError):
    """Raised when a run-root recovery marker blocks further automatic work."""


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

    @property
    def safety_stop_path(self) -> Path:
        return self.root.parent / "control" / "generation-safety-stop.json"

    def _assert_run_active_unlocked(self) -> None:
        marker = self.safety_stop_path
        if not marker.exists() and not marker.is_symlink():
            return
        try:
            if marker.is_symlink() or not marker.is_file():
                raise ValueError("safety stop marker is not a regular file")
            receipt = _read_json(marker)
            action = receipt.get("recovery_action")
            if not isinstance(action, str) or not action.strip():
                raise ValueError("safety stop marker has no recovery action")
        except Exception as error:
            raise GenerationSafetyStop(
                f"generation safety stop is present but unreadable at {marker}; "
                "preserve the run tree and inspect the marker before resuming"
            ) from error
        raise GenerationSafetyStop(f"generation safety stop at {marker}: {action}")

    def assert_run_active(self) -> None:
        with self._operation_lock():
            self._assert_run_active_unlocked()

    def _active_claim_generations_unlocked(self, now_s: float) -> set[tuple[str, int]]:
        active: set[tuple[str, int]] = set()
        claimed_root = self.root / "claimed"
        if claimed_root.is_symlink() or not claimed_root.is_dir():
            raise ValueError(f"queue state must be a real directory: {claimed_root}")
        for worker_root in sorted(claimed_root.iterdir()):
            if worker_root.is_symlink() or not worker_root.is_dir():
                continue
            lease_path = self._worker_lease_path(worker_root.name)
            if lease_path.is_symlink() or not lease_path.is_file():
                continue
            try:
                lease = _read_json(lease_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if float(lease.get("lease_expires_at_s", 0.0)) <= now_s:
                continue
            for job_path in sorted(worker_root.glob("*.json")):
                if job_path.name.endswith(".claim.json") or job_path.is_symlink():
                    continue
                try:
                    job = GenerationJob.from_dict(_read_json(job_path))
                    claim = _read_json(worker_root / f"{job.job_id}.claim.json")
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                worker_instance_id = claim.get("worker_instance_id")
                if (
                    claim.get("job_id") == job.job_id
                    and claim.get("worker_id") == worker_root.name
                    and worker_instance_id
                    and lease.get("worker_instance_id") == worker_instance_id
                    and lease.get("job_id") == job.job_id
                ):
                    active.add((job.job_id, job.retry_generation))
        return active

    @staticmethod
    def _inventory_preserved_paths(paths: list[Path], run_root: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for root in paths:
            items = [root]
            if root.is_dir() and not root.is_symlink():
                items.extend(sorted(root.rglob("*")))
            for path in items:
                absolute = path.absolute()
                if absolute in seen:
                    continue
                seen.add(absolute)
                try:
                    relative = path.relative_to(run_root).as_posix()
                    info = path.lstat()
                except (OSError, ValueError):
                    relative = os.path.relpath(path, run_root).replace(os.sep, "/")
                    entries.append({"path": relative, "bytes": 0, "kind": "unreadable"})
                    continue
                if path.is_symlink():
                    link_bytes = os.fsencode(os.readlink(path))
                    entries.append({
                        "path": relative,
                        "bytes": len(link_bytes),
                        "sha256": hashlib.sha256(link_bytes).hexdigest(),
                        "kind": "symlink",
                    })
                elif path.is_file():
                    digest = hashlib.sha256()
                    byte_count = 0
                    try:
                        with path.open("rb") as stream:
                            while chunk := stream.read(1024 * 1024):
                                digest.update(chunk)
                                byte_count += len(chunk)
                    except OSError:
                        entries.append({"path": relative, "bytes": int(info.st_size), "kind": "unreadable"})
                    else:
                        entries.append({
                            "path": relative,
                            "bytes": byte_count,
                            "sha256": digest.hexdigest(),
                            "kind": "file",
                        })
                elif path.is_dir():
                    entries.append({"path": relative, "bytes": 0, "kind": "directory"})
                else:
                    entries.append({"path": relative, "bytes": int(info.st_size), "kind": "other"})
        return entries

    def assert_no_orphan_archive_scratch(self) -> None:
        """Trip the run stop before new work if abandoned writer scratch is discoverable."""
        with self._operation_lock():
            self._assert_run_active_unlocked()
            run_root = self.root.parent.resolve()
            output_root = run_root / "staging" / "worker-results"
            if output_root.is_symlink():
                raise GenerationSafetyStop(
                    f"generation safety stop: staging result root is a symlink: {output_root}"
                )
            if not output_root.is_dir():
                return
            active_generations = self._active_claim_generations_unlocked(time.time())
            orphan_roots: list[Path] = []
            orphan_jobs: set[str] = set()
            for job_root in sorted(output_root.iterdir()):
                if job_root.is_symlink() or not job_root.is_dir():
                    continue
                for retry_root in sorted(job_root.iterdir()):
                    if retry_root.is_symlink() or not retry_root.is_dir() or not retry_root.name.startswith("retry-"):
                        continue
                    try:
                        retry_generation = int(retry_root.name.removeprefix("retry-"))
                    except ValueError:
                        retry_generation = -1
                    if (job_root.name, retry_generation) in active_generations:
                        continue
                    for child in sorted(retry_root.iterdir()):
                        if child.name.startswith(".") and any(
                            token in child.name
                            for token in (
                                ".archive-spool-",
                                ".partial-",
                                ".archive-write-in-progress-",
                            )
                        ):
                            orphan_roots.append(child)
                            orphan_jobs.add(job_root.name)
            if not orphan_roots:
                return
            entries = self._inventory_preserved_paths(orphan_roots, run_root)
            receipt = {
                "schema_version": "icgs_generation_safety_stop_v1",
                "reason": "discoverable_orphan_archive_scratch",
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job_ids": sorted(orphan_jobs),
                "preserved_bytes": sum(item["bytes"] for item in entries if item["kind"] == "file"),
                "preserved_files": entries,
                "recovery_action": (
                    "Keep every listed path unchanged. Verify its fate against the pinned Hugging Face "
                    "publication receipts and artifact hashes; recover or archive unmatched bytes before "
                    "starting a disjoint run. Do not delete listed paths until their HF fate is known."
                ),
            }
            try:
                self._trip_safety_stop_unlocked(receipt)
            except Exception as error:
                raise GenerationSafetyStop(
                    "generation safety stop: abandoned archive scratch was found but its receipt could "
                    f"not be persisted; preserve the run tree and inspect {output_root} before resuming"
                ) from error
            raise GenerationSafetyStop(
                f"generation safety stop: found {receipt['preserved_bytes']} bytes of abandoned archive "
                f"scratch; inspect {self.safety_stop_path} before resuming"
            )

    def _trip_safety_stop_unlocked(self, receipt: dict[str, Any]) -> Path:
        if not isinstance(receipt, dict) or not receipt:
            raise ValueError("safety stop receipt must be a nonempty object")
        marker = self.safety_stop_path
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists() or marker.is_symlink():
            try:
                if marker.is_symlink() or not marker.is_file():
                    raise ValueError("safety stop marker is not a regular file")
                existing = _read_json(marker)
                if not isinstance(existing.get("recovery_action"), str):
                    raise ValueError("safety stop marker has no recovery action")
            except Exception as error:
                incident_root = marker.parent / "generation-safety-stop-incidents"
                incident_root.mkdir(parents=True, exist_ok=True)
                job_id = str(receipt.get("job_id", "unknown"))
                self._validate_job_component(job_id)
                retry = receipt.get("retry_generation", 0)
                if type(retry) is not int or retry < 0:
                    retry = 0
                suffix = f"{job_id}.retry-{retry}.{time.time_ns()}.json"
                incident_path = incident_root / suffix
                _write_durable_safety_record(incident_path, receipt)
                raise GenerationSafetyStop(
                    f"generation safety stop marker at {marker} is unreadable; "
                    f"orphan incident receipt was preserved at {incident_path}"
                ) from error
            incidents = existing.get("incidents")
            if not isinstance(incidents, list):
                incidents = [{
                    key: value for key, value in existing.items()
                    if key not in {"incidents", "incident_count"}
                }]
            incidents = [item for item in incidents if isinstance(item, dict)]
            incidents.append(dict(receipt))
            preserved_files = []
            for incident in incidents:
                incident_files = incident.get("preserved_files", [])
                if isinstance(incident_files, list):
                    preserved_files.extend(item for item in incident_files if isinstance(item, dict))

            def regular_file_bytes(item: dict[str, Any]) -> int:
                if item.get("kind") != "file":
                    return 0
                value = item.get("bytes", 0)
                return value if type(value) is int and value >= 0 else 0

            existing.update({
                "incidents": incidents,
                "incident_count": len(incidents),
                "preserved_files": preserved_files,
                "preserved_bytes": sum(regular_file_bytes(item) for item in preserved_files),
            })
            _write_durable_safety_record(marker, existing)
            return marker
        receipt = dict(receipt)
        receipt.setdefault("incident_count", 1)
        receipt.setdefault("incidents", [dict(receipt)])
        _write_durable_safety_record(marker, receipt)
        return marker

    def trip_safety_stop(self, receipt: dict[str, Any]) -> Path:
        with self._operation_lock():
            return self._trip_safety_stop_unlocked(receipt)

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
            self._assert_run_active_unlocked()
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
        self.assert_no_orphan_archive_scratch()
        with self._operation_lock():
            self._assert_run_active_unlocked()
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
            self._assert_run_active_unlocked()
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
            self._assert_run_active_unlocked()
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
            self._assert_run_active_unlocked()
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

    def _receipt_only_binding(
        self,
        job_id: str,
        source: Path,
    ) -> tuple[GenerationJob, dict[str, Any], dict[str, Any]]:
        """Validate the local batch receipt and return its immutable job row."""
        for name in ("job.json", "result.json"):
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"receipt-only publication requires a regular {name}")
        job = GenerationJob.from_dict(_read_json(source / "job.json"))
        result = WorkerResult.from_dict(_read_json(source / "result.json"))
        if job.job_id != job_id or result.job_id != job_id:
            raise ValueError("receipt-only publication job identity mismatch")
        if result.attempt_id != job.attempt_id or result.program_id != job.program_id:
            raise ValueError("receipt-only publication result identity mismatch")

        receipt_path = self.root / "publication_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("receipt-only pruning requires a verified publication receipt")
        receipt = _read_json(receipt_path)
        if receipt.get("status") not in {"VERIFIED", "COMPLETE"}:
            raise ValueError("receipt-only pruning requires a verified publication receipt")
        if receipt.get("run_id") != job.run_id or receipt.get("source_run_id") != job.run_id:
            raise ValueError("verified publication receipt run/source identity mismatch")
        job_ids = receipt.get("job_ids")
        if not isinstance(job_ids, list) or job_ids.count(job_id) != 1:
            raise ValueError("verified publication receipt does not bind this job")
        for field in ("dataset_identity", "archive_format_id", "episode_schema_version"):
            if not isinstance(receipt.get(field), str) or not receipt[field].strip():
                raise ValueError(f"verified publication receipt is missing {field}")
        data_oid = receipt.get("data_commit_oid")
        commit_oid = receipt.get("commit_oid")
        for field, value in (("data_commit_oid", data_oid), ("commit_oid", commit_oid)):
            if (
                not isinstance(value, str)
                or len(value) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"verified publication receipt is missing a valid {field}")

        declared_hashes = result.file_sha256
        if not declared_hashes:
            raise ValueError("receipt-only publication requires a nonempty artifact inventory")
        batch_hashes = receipt.get("artifact_hashes")
        if not isinstance(batch_hashes, dict):
            raise ValueError("verified publication receipt is missing artifact hashes")
        job_prefix = f"{job_id}/"
        receipt_hashes = {
            key[len(job_prefix):]: digest
            for key, digest in batch_hashes.items()
            if isinstance(key, str) and key.startswith(job_prefix)
        }
        if receipt_hashes != declared_hashes:
            raise ValueError("verified publication receipt artifact hashes do not match the job")

        profile = receipt.get("archive_profile")
        if not isinstance(profile, dict):
            raise ValueError("verified publication receipt is missing archive identity")
        if profile.get("local_artifact_retention") != "receipt_only":
            raise ValueError("receipt-only pruning requires the receipt-only archive profile")
        for field in ("dataset_identity", "archive_format_id", "episode_schema_version"):
            if profile.get(field) != receipt.get(field):
                raise ValueError(f"verified publication receipt archive identity mismatch: {field}")
        manifest_digest = receipt.get("dataset_manifest_sha256")
        if (
            not isinstance(manifest_digest, str)
            or len(manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in manifest_digest)
        ):
            raise ValueError("verified publication receipt is missing dataset manifest SHA256")
        manifest_path = self.root / "publication_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("receipt-only pruning requires the published dataset manifest")
        manifest_bytes = self._read_regular_file(manifest_path)
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest:
            raise ValueError("verified publication receipt dataset manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("published dataset manifest must be an object")
        if manifest.get("dataset_identity") != receipt["dataset_identity"]:
            raise ValueError("published dataset manifest identity mismatch")
        if manifest.get("archive_format_id") != receipt["archive_format_id"]:
            raise ValueError("published dataset manifest archive format mismatch")
        if manifest.get("episode_schema_version") != receipt["episode_schema_version"]:
            raise ValueError("published dataset manifest episode schema mismatch")
        if manifest.get("archive_profile") != profile:
            raise ValueError("published dataset manifest archive profile mismatch")
        source_runs = manifest.get("source_run_ids")
        if not isinstance(source_runs, list) or job.run_id not in source_runs:
            raise ValueError("published dataset manifest does not bind the source run")

        collection, identity_key, identity = (
            ("episodes", "episode_id", result.episode_id)
            if result.episode_id is not None
            else ("failure_attempts", "attempt_id", result.attempt_id)
        )
        rows = manifest.get(collection)
        if not isinstance(rows, list):
            raise ValueError(f"published dataset manifest {collection} must be a list")
        matching = [
            row for row in rows
            if isinstance(row, dict) and row.get(identity_key) == identity
        ]
        if len(matching) != 1:
            raise ValueError("published dataset manifest does not uniquely bind the job row")
        row = matching[0]
        archive_profile = ArchiveProfileConfig.from_dict(profile)
        validate_archive_manifest_row_identity(job, result, row, archive_profile)
        if row.get("job_id") != job_id or row.get("run_id") != job.run_id:
            raise ValueError("published dataset manifest row job/run identity mismatch")
        if row.get("source_run_id") != job.run_id or row.get("file_sha256") != declared_hashes:
            raise ValueError("published dataset manifest row source/hash identity mismatch")

        return job, result.as_dict(), {
            "receipt_version": 1,
            "status": "VERIFIED",
            "job_id": job_id,
            "run_id": job.run_id,
            "source_run_id": job.run_id,
            "dataset_identity": receipt["dataset_identity"],
            "archive_format_id": receipt["archive_format_id"],
            "episode_schema_version": receipt["episode_schema_version"],
            "archive_profile": dict(profile),
            "data_commit_oid": data_oid,
            "commit_oid": commit_oid,
            "dataset_manifest_sha256": manifest_digest,
            "artifact_hashes": dict(declared_hashes),
            "manifest_row_sha256": _canonical_json_sha256(dict(row)),
            "manifest_row": row,
        }

    def resolve_result_root(
        self,
        result_dir: str,
        *,
        require_exists: bool,
    ) -> Path:
        result_root = Path(result_dir)
        if not result_root.is_absolute() or ".." in result_root.parts:
            raise ValueError("result_dir must be absolute and contained")
        run_root = self.root.parent.resolve(strict=True)
        lexical_root = Path(os.path.abspath(result_root))
        if not lexical_root.is_relative_to(run_root) or lexical_root == run_root:
            raise ValueError("receipt-only result_dir must remain beneath the run root")
        current = run_root
        for part in lexical_root.relative_to(run_root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("result_dir may not traverse symlinks")
        if result_root.is_symlink():
            raise ValueError("result_dir may not be a symlink")
        resolved = result_root.resolve(strict=require_exists)
        if not resolved.is_relative_to(run_root) or resolved == run_root:
            raise ValueError("result_dir must remain beneath the run root")
        if require_exists and not resolved.is_dir():
            raise ValueError("result_dir must be a real directory")
        if not require_exists and (result_root.exists() or result_root.is_symlink()):
            raise ValueError("result payload unexpectedly exists after publication")
        return resolved

    def _validate_saved_receipt_only_record(
        self,
        directory: Path,
        job_id: str,
        *,
        archive_profile: ArchiveProfileConfig | None = None,
        allow_remaining_payload: bool = False,
    ) -> dict[str, Any]:
        for name in ("job.json", "result.json", "publication_receipt.json"):
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("published result is missing its receipt-only queue record")
        job = GenerationJob.from_dict(_read_json(directory / "job.json"))
        result = WorkerResult.from_dict(_read_json(directory / "result.json"))
        receipt = _read_json(directory / "publication_receipt.json")
        if (
            job.job_id != job_id
            or result.job_id != job_id
            or result.attempt_id != job.attempt_id
            or result.program_id != job.program_id
        ):
            raise ValueError("published receipt-only queue record job identity mismatch")
        if (
            receipt.get("status") != "VERIFIED"
            or receipt.get("job_id") != job_id
            or receipt.get("run_id") != job.run_id
            or receipt.get("source_run_id") != job.run_id
        ):
            raise ValueError("published receipt-only queue record identity mismatch")
        profile_payload = receipt.get("archive_profile")
        if not isinstance(profile_payload, dict):
            raise ValueError("published receipt-only queue record is missing archive profile")
        profile = ArchiveProfileConfig.from_dict(profile_payload)
        if archive_profile is not None and profile != archive_profile:
            raise ValueError("published receipt-only queue record archive profile mismatch")
        if profile.local_artifact_retention != "receipt_only":
            raise ValueError("published receipt-only queue record has the wrong retention profile")
        for field in ("dataset_identity", "archive_format_id", "episode_schema_version"):
            if receipt.get(field) != getattr(profile, field):
                raise ValueError(f"published receipt-only queue record {field} mismatch")
        for field in ("data_commit_oid", "commit_oid"):
            value = receipt.get(field)
            if (
                not isinstance(value, str)
                or len(value) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"published receipt-only queue record has invalid {field}")
        manifest_digest = receipt.get("dataset_manifest_sha256")
        if (
            not isinstance(manifest_digest, str)
            or len(manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in manifest_digest)
        ):
            raise ValueError("published receipt-only queue record has invalid dataset manifest SHA256")
        if not result.file_sha256 or receipt.get("artifact_hashes") != result.file_sha256:
            raise ValueError("published receipt-only queue record artifact hash mismatch")
        row = receipt.get("manifest_row")
        if not isinstance(row, dict):
            raise ValueError("published receipt-only queue record has no manifest row")
        if receipt.get("manifest_row_sha256") != _canonical_json_sha256(row):
            raise ValueError("published receipt-only queue record manifest row SHA256 mismatch")
        validate_archive_manifest_row_identity(job, result, row, profile)
        if allow_remaining_payload and (
            Path(result.result_dir).exists() or Path(result.result_dir).is_symlink()
        ):
            self._verify_local_result_inventory(result, allow_missing_files=True)
        else:
            self.resolve_result_root(result.result_dir, require_exists=False)
        return dict(row)

    def _verify_local_result_inventory(
        self,
        result: WorkerResult,
        *,
        allow_missing_files: bool = False,
    ) -> Path:
        resolved = self.resolve_result_root(result.result_dir, require_exists=True)

        actual: dict[str, str] = {}
        pending = [resolved]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISLNK(mode):
                        raise ValueError("receipt-only result inventory may not contain symlinks")
                    if stat.S_ISDIR(mode):
                        pending.append(path)
                    elif stat.S_ISREG(mode):
                        relative = path.relative_to(resolved).as_posix()
                        actual[relative] = hashlib.sha256(self._read_regular_file(path)).hexdigest()
                    else:
                        raise ValueError("receipt-only result inventory must contain regular files only")
        if allow_missing_files:
            if any(
                relative not in result.file_sha256
                or result.file_sha256[relative] != digest
                for relative, digest in actual.items()
            ):
                raise ValueError("remaining receipt-only files do not match the verified artifact inventory")
        elif actual != result.file_sha256:
            raise ValueError("receipt-only result files do not match the verified artifact inventory")
        return resolved

    def mark_published(
        self,
        job_id: str,
        *,
        retention: Literal["keep", "receipt_only"] = "keep",
    ) -> Path:
        if retention not in {"keep", "receipt_only"}:
            raise ValueError("retention must be keep or receipt_only")
        self._validate_job_component(job_id)
        with self._operation_lock():
            self._assert_run_active_unlocked()
            for state in ("ingested", "published"):
                state_root = self.root / state
                if state_root.is_symlink() or not state_root.is_dir():
                    raise ValueError(f"queue state must be a real directory: {state_root}")
            source = self.root / "ingested" / job_id
            target = self.root / "published" / job_id
            if target.is_symlink():
                raise ValueError(f"published job must not be a symlink: {target}")
            if target.exists():
                if source.exists():
                    raise ValueError(f"ingested and published results both exist: {job_id}")
                if retention == "receipt_only":
                    self._validate_saved_receipt_only_record(target, job_id)
                return target
            if source.is_symlink() or not source.is_dir():
                raise FileNotFoundError(f"ingested result does not exist: {job_id}")
            if retention == "keep":
                os.replace(source, target)
                return target

            _job, result_payload, per_job_receipt = self._receipt_only_binding(job_id, source)
            result = WorkerResult.from_dict(result_payload)
            result_root = Path(result.result_dir)
            local_receipt_path = source / "publication_receipt.json"
            if local_receipt_path.exists() or local_receipt_path.is_symlink():
                if local_receipt_path.is_symlink() or not local_receipt_path.is_file():
                    raise ValueError("per-job publication receipt must be a regular file")
                if _read_json(local_receipt_path) != per_job_receipt:
                    raise ValueError("per-job publication receipt conflicts with verified batch")
                if result_root.is_symlink():
                    raise ValueError("receipt-only result_dir may not be a symlink")
                if result_root.exists():
                    self.resolve_result_root(result.result_dir, require_exists=True)
                    self._verify_local_result_inventory(result, allow_missing_files=True)
                    shutil.rmtree(result_root)
                else:
                    # A durable per-job receipt can make pruning restartable after
                    # the payload has already disappeared.  It does not make an
                    # untrusted result_dir safe: still validate its run-root
                    # containment before accepting the receipt-only transition.
                    self.resolve_result_root(result.result_dir, require_exists=False)
            elif result_root.exists():
                self._verify_local_result_inventory(result)
                if local_receipt_path.is_symlink():
                    raise ValueError("per-job publication receipt must not be a symlink")
                else:
                    _atomic_write(local_receipt_path, per_job_receipt)
                shutil.rmtree(result_root)
            else:
                raise ValueError("missing local payload lacks a matching verified per-job receipt")
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


__all__ = ["FilesystemJobQueue", "GenerationSafetyStop", "MalformedReadyResult"]
