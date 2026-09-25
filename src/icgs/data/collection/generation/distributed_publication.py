"""Five-minute, conflict-safe Hugging Face publication for generation results."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from huggingface_hub import CommitOperationAdd

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_validation import validate_closed_result
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue


PUBLICATION_STATES = frozenset({
    "PREPARED",
    "DATA_COMMITTED",
    "VERIFIED",
    "COMPLETE",
    "DEFERRED",
})


@dataclass(frozen=True)
class PublicationConfig:
    batch_size: int = 1
    upload_threads: int = 1
    retry_attempts: int = 3
    retry_base_s: float = 5.0
    retry_cooldown_s: float = 300.0
    rate_limit_cooldown_s: float = 3600.0

    def __post_init__(self) -> None:
        for name in ("batch_size", "upload_threads", "retry_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("retry_base_s", "retry_cooldown_s", "rate_limit_cooldown_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value < 0:
                raise ValueError(f"{name} must be a nonnegative number")


@dataclass(frozen=True)
class PublicationReceipt:
    receipt_version: int = 2
    run_id: str = ""
    job_ids: tuple[str, ...] = ()
    status: str = "PREPARED"
    data_commit_oid: str | None = None
    commit_oid: str | None = None
    artifact_hashes: dict[str, str] | None = None
    source_run_id: str | None = None
    dataset_identity: str | None = None
    archive_format_id: str | None = None
    episode_schema_version: str | None = None
    archive_profile: dict[str, Any] | None = None
    dataset_manifest_sha256: str | None = None
    prefix: str = ""
    path_count: int = 0
    created_at_s: float = 0.0
    updated_at_s: float = 0.0
    published_at_s: float | None = None
    last_error: str | None = None
    next_retry_s: float | None = None

    def __post_init__(self) -> None:
        if self.status not in PUBLICATION_STATES:
            raise ValueError(f"unsupported publication state: {self.status}")
        if not isinstance(self.job_ids, tuple):
            raise ValueError("job_ids must be a tuple")
        if self.artifact_hashes is not None and not isinstance(self.artifact_hashes, dict):
            raise ValueError("artifact_hashes must be an object or null")
        if self.archive_profile is not None and not isinstance(self.archive_profile, dict):
            raise ValueError("archive_profile must be an object or null")
        identity = (
            self.source_run_id,
            self.dataset_identity,
            self.archive_format_id,
            self.episode_schema_version,
        )
        if any(value is not None for value in identity):
            if any(not isinstance(value, str) or not value.strip() for value in identity):
                raise ValueError("archive publication identity fields must be nonblank strings")
            if self.source_run_id != self.run_id:
                raise ValueError("publication source_run_id must match run_id")
        if self.dataset_manifest_sha256 is not None and (
            len(self.dataset_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.dataset_manifest_sha256)
        ):
            raise ValueError("dataset_manifest_sha256 must be a lowercase SHA256 digest")
        if self.archive_profile is not None:
            for key in ("dataset_identity", "archive_format_id", "episode_schema_version"):
                if self.archive_profile.get(key) != getattr(self, key):
                    raise ValueError(f"publication archive profile disagrees with {key}")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "receipt_version": self.receipt_version,
            "run_id": self.run_id,
            "job_ids": list(self.job_ids),
            "status": self.status,
            "data_commit_oid": self.data_commit_oid,
            "commit_oid": self.commit_oid,
            "artifact_hashes": dict(self.artifact_hashes or {}),
            "prefix": self.prefix,
            "path_count": self.path_count,
            "created_at_s": self.created_at_s,
            "updated_at_s": self.updated_at_s,
            "published_at_s": self.published_at_s,
            "last_error": self.last_error,
            "next_retry_s": self.next_retry_s,
        }
        if any((
            self.source_run_id,
            self.dataset_identity,
            self.archive_format_id,
            self.episode_schema_version,
            self.archive_profile,
            self.dataset_manifest_sha256,
        )):
            payload.update({
                "source_run_id": self.source_run_id,
                "dataset_identity": self.dataset_identity,
                "archive_format_id": self.archive_format_id,
                "episode_schema_version": self.episode_schema_version,
                "archive_profile": (
                    dict(self.archive_profile) if self.archive_profile is not None else None
                ),
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
            })
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicationReceipt":
        values = dict(payload)
        return cls(
            receipt_version=int(values.get("receipt_version", 2)),
            run_id=str(values.get("run_id", "")),
            job_ids=tuple(str(item) for item in values.get("job_ids", ())),
            status={
                "COMMIT_PENDING": "PREPARED",
                "COMMITTED": "DATA_COMMITTED",
            }.get(str(values.get("status", "PREPARED")), str(values.get("status", "PREPARED"))),
            data_commit_oid=values.get("data_commit_oid") or values.get("commit_oid"),
            commit_oid=values.get("commit_oid"),
            artifact_hashes=dict(values.get("artifact_hashes") or {}),
            source_run_id=values.get("source_run_id"),
            dataset_identity=values.get("dataset_identity"),
            archive_format_id=values.get("archive_format_id"),
            episode_schema_version=values.get("episode_schema_version"),
            archive_profile=(
                dict(values["archive_profile"])
                if isinstance(values.get("archive_profile"), Mapping)
                else None
            ),
            dataset_manifest_sha256=values.get("dataset_manifest_sha256"),
            prefix=str(values.get("prefix", "")),
            path_count=int(values.get("path_count", 0)),
            created_at_s=float(values.get("created_at_s", 0.0)),
            updated_at_s=float(values.get("updated_at_s", values.get("created_at_s", 0.0))),
            published_at_s=(
                None if values.get("published_at_s") is None else float(values["published_at_s"])
            ),
            last_error=values.get("last_error"),
            next_retry_s=(
                None if values.get("next_retry_s") is None else float(values["next_retry_s"])
            ),
        )


class _TransientPublicationDeferred(RuntimeError):
    """Internal signal that a temporary HF failure was safely deferred."""


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reconcile_publication(
    receipt: PublicationReceipt | Mapping[str, Any],
    remote_manifest: Mapping[str, Any],
) -> PublicationReceipt:
    """Resolve an immutable local receipt against a remote publication identity."""
    local = receipt if isinstance(receipt, PublicationReceipt) else PublicationReceipt.from_dict(receipt)
    remote_value = remote_manifest.get("publication_receipt", remote_manifest)
    if not isinstance(remote_value, Mapping):
        return local
    remote = dict(remote_value)
    remote_job_ids = tuple(str(item) for item in remote.get("job_ids", ()))
    if remote_job_ids and remote_job_ids != local.job_ids:
        raise ValueError("immutable publication identity conflict: job_ids")
    for key, local_value in (
        ("run_id", local.run_id),
        ("source_run_id", local.source_run_id),
        ("prefix", local.prefix),
        ("data_commit_oid", local.data_commit_oid),
        ("dataset_identity", local.dataset_identity),
        ("archive_format_id", local.archive_format_id),
        ("episode_schema_version", local.episode_schema_version),
        ("dataset_manifest_sha256", local.dataset_manifest_sha256),
    ):
        remote_value_for_key = remote.get(key) or remote_manifest.get(key)
        if remote_value_for_key is None and isinstance(remote.get("archive_profile"), Mapping):
            remote_value_for_key = remote["archive_profile"].get(key)
        if remote_value_for_key is None and isinstance(remote_manifest.get("archive_profile"), Mapping):
            remote_value_for_key = remote_manifest["archive_profile"].get(key)
        if local_value and remote_value_for_key and remote_value_for_key != local_value:
            raise ValueError(f"immutable publication identity conflict: {key}")
    expected_hashes = dict(local.artifact_hashes or {})
    remote_hashes = dict(
        remote.get("artifact_hashes")
        or remote_manifest.get("artifact_hashes")
        or remote.get("file_sha256")
        or {}
    )
    if expected_hashes and remote_hashes and remote_hashes != expected_hashes:
        raise ValueError("immutable publication identity conflict: artifact_hashes")
    matching_hashes = bool(expected_hashes) and remote_hashes == expected_hashes
    remote_status = str(remote.get("status", ""))
    if matching_hashes or (not expected_hashes and remote_status in {"VERIFIED", "COMPLETE"}):
        commit_oid = remote.get("commit_oid") or local.commit_oid or local.data_commit_oid
        return replace(
            local,
            status="VERIFIED",
            commit_oid=commit_oid,
            updated_at_s=max(local.updated_at_s, float(remote.get("updated_at_s", local.updated_at_s))),
            last_error=None,
            next_retry_s=None,
        )
    if remote_status == "DATA_COMMITTED":
        return replace(local, status="DATA_COMMITTED", updated_at_s=local.updated_at_s)
    return local


class HuggingFaceBatchPublisher:
    # Episode JSON contains dense point-cloud observations and can exceed 1 GB;
    # serialize one result per commit and one LFS upload thread so an idle S3
    # multipart connection cannot time out while other large uploads compete.
    MAX_JOBS_PER_COMMIT = 1
    # A single LFS upload can fail with an S3 RequestTimeout after several
    # minutes. Retry a bounded number of times in this tick, then leave the
    # immutable queue entry in ``ingested`` and defer the next attempt. This
    # keeps a transient HF outage from terminating the coordinator.
    MAX_TRANSIENT_ATTEMPTS = 3
    TRANSIENT_RETRY_BASE_S = 5.0
    TRANSIENT_RETRY_COOLDOWN_S = 300.0
    RATE_LIMIT_COOLDOWN_S = 3600.0
    def __init__(
        self,
        run: RunConfig,
        api: Any,
        token: str,
        queue: FilesystemJobQueue,
        *,
        last_success_s: float | None = None,
        remote_manifest: Mapping[str, Any] | None = None,
        remote_verify: Any | None = None,
        publication_config: PublicationConfig | None = None,
        archive_profile: ArchiveProfileConfig | None = None,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Hugging Face token is required")
        self.run = run
        if archive_profile is not None and not isinstance(archive_profile, ArchiveProfileConfig):
            raise TypeError("archive_profile must be an ArchiveProfileConfig or None")
        if (
            run.validation_mode
            and archive_profile is not None
            and archive_profile.local_artifact_retention != "keep"
        ):
            raise ValueError("validation mode requires local_artifact_retention=keep")
        self.archive_profile = archive_profile
        self.api = api
        self._token = token
        self.queue = queue
        self.publication_config = publication_config or PublicationConfig()
        self.MAX_JOBS_PER_COMMIT = self.publication_config.batch_size
        self.MAX_TRANSIENT_ATTEMPTS = self.publication_config.retry_attempts
        self.last_success_s = time.time() if last_success_s is None else float(last_success_s)
        self.remote_manifest = dict(remote_manifest or {"manifest_version": 3, "episodes": [], "failure_attempts": []})
        self.remote_verify = remote_verify
        self.receipts: list[PublicationReceipt] = []
        self._retry_state_path = self.queue.root / "publication_retry.json"
        self._next_retry_s = self._load_retry_state()

    def _load_retry_state(self) -> float:
        try:
            payload = json.loads(self._retry_state_path.read_text(encoding="utf-8"))
            return max(0.0, float(payload.get("next_retry_s", 0.0)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0

    def _write_retry_state(self, *, next_retry_s: float, error: str) -> None:
        self._retry_state_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(self._retry_state_path, _canonical_json_bytes({
            "next_retry_s": next_retry_s,
            "error": error[:2000],
            "updated_at_s": time.time(),
        }))

    def _defer_transient_failure(self, now_s: float, error: BaseException) -> None:
        status = getattr(getattr(error.__cause__, "response", None), "status_code", None)
        if status is None and "429" in str(error):
            status = 429
        cooldown_s = (
            self.publication_config.rate_limit_cooldown_s
            if status == 429
            else self.publication_config.retry_cooldown_s
        )
        self._next_retry_s = now_s + cooldown_s
        self._write_retry_state(next_retry_s=self._next_retry_s, error=str(error))

    def _clear_retry_state(self) -> None:
        self._next_retry_s = 0.0
        self._retry_state_path.unlink(missing_ok=True)

    @staticmethod
    def _is_transient_upload_error(error: BaseException) -> bool:
        """Recognize retryable transport/S3 failures without swallowing 4xx errors."""
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in {429, 500, 502, 503, 504}:
            return True
        message = str(error).lower().replace("_", "")
        return any(marker in message for marker in (
            "429",
            "ratelimit",
            "requesttimeout",
            "readtimeout",
            "connecttimeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "<code>500</code>",
            "<code>502</code>",
            "<code>503</code>",
            "<code>504</code>",
        ))

    def _with_transient_retry(self, operation):
        delay_s = self.publication_config.retry_base_s
        for attempt in range(1, self.publication_config.retry_attempts + 1):
            try:
                return operation()
            except Exception as error:
                if not self._is_transient_upload_error(error):
                    raise
                if attempt >= self.MAX_TRANSIENT_ATTEMPTS:
                    raise _TransientPublicationDeferred(str(error)) from error
                time.sleep(delay_s)
                delay_s *= 2.0

    def pending_job_ids(self, *, limit: int | None = None) -> tuple[str, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        values = tuple(sorted(path.name for path in (self.queue.root / "ingested").iterdir() if path.is_dir()))
        return values if limit is None else values[:limit]

    @property
    def _publication_receipt_path(self) -> Path:
        return self.queue.root / "publication_receipt.json"

    @property
    def _verification_receipt_path(self) -> Path:
        return self.queue.root / "publication_receipt_to_verify.json"

    def _write_publication_receipt(self, receipt: PublicationReceipt) -> None:
        _atomic_write_bytes(
            self._publication_receipt_path,
            _canonical_json_bytes(receipt.as_dict()),
        )

    def _ensure_verification_receipt_snapshot(self, receipt: PublicationReceipt) -> Path:
        """Return the immutable receipt bytes used by pinned remote verification."""
        path = self._verification_receipt_path
        if path.is_symlink():
            raise ValueError("publication receipt verification snapshot may not be a symlink")
        if path.exists():
            if not path.is_file():
                raise ValueError("publication receipt verification snapshot must be a regular file")
            try:
                snapshot = PublicationReceipt.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("publication receipt verification snapshot is malformed") from error
            immutable_fields = (
                "run_id",
                "job_ids",
                "data_commit_oid",
                "artifact_hashes",
                "source_run_id",
                "dataset_identity",
                "archive_format_id",
                "episode_schema_version",
                "archive_profile",
                "dataset_manifest_sha256",
                "prefix",
            )
            if any(getattr(snapshot, name) != getattr(receipt, name) for name in immutable_fields):
                raise ValueError("publication receipt verification snapshot identity conflict")
            return path
        _atomic_write_bytes(path, _canonical_json_bytes(receipt.as_dict()))
        return path

    def _clear_stale_verification_receipt_snapshot(self) -> None:
        path = self._verification_receipt_path
        if path.is_symlink():
            raise ValueError("publication receipt verification snapshot may not be a symlink")
        if path.exists() and not path.is_file():
            raise ValueError("publication receipt verification snapshot must be a regular file")
        path.unlink(missing_ok=True)

    def _read_publication_receipt(self) -> PublicationReceipt | None:
        try:
            return PublicationReceipt.from_dict(
                json.loads(self._publication_receipt_path.read_text(encoding="utf-8"))
            )
        except FileNotFoundError:
            return None

    def _read_completed_manifest_snapshot(
        self,
        receipt: PublicationReceipt,
    ) -> dict[str, Any]:
        manifest_path = self.queue.root / "publication_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("completed publication is missing its manifest snapshot")
        manifest_bytes = manifest_path.read_bytes()
        if (
            receipt.dataset_manifest_sha256 is not None
            and hashlib.sha256(manifest_bytes).hexdigest() != receipt.dataset_manifest_sha256
        ):
            raise ValueError("completed publication manifest snapshot hash mismatch")
        try:
            manifest = json.loads(manifest_bytes)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("completed publication manifest snapshot is malformed") from error
        if not isinstance(manifest, dict):
            raise ValueError("completed publication manifest snapshot must be an object")
        return manifest

    def _validate_prefix(self) -> None:
        if self.run.validation_mode:
            required = "validation/validation-cpu-20260922/"
            if not self.run.hf_subfolder.startswith(required):
                raise ValueError(
                    "validation publication requires hf_subfolder under "
                    "validation/validation-cpu-20260922/"
                )

    def _artifact_hashes(self, job_ids: tuple[str, ...]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for job_id in job_ids:
            result = json.loads(
                (self.queue.root / "ingested" / job_id / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            for relative, digest in dict(result.get("file_sha256") or {}).items():
                hashes[f"{job_id}/{relative}"] = str(digest)
        return hashes

    def _bind_archive_receipt(
        self,
        receipt: PublicationReceipt,
        manifest: Mapping[str, Any],
    ) -> PublicationReceipt:
        if self.archive_profile is None:
            return receipt
        manifest_sha256 = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
        return replace(
            receipt,
            source_run_id=self.run.run_id,
            dataset_identity=self.archive_profile.dataset_identity,
            archive_format_id=self.archive_profile.archive_format_id,
            episode_schema_version=self.archive_profile.episode_schema_version,
            archive_profile=self.archive_profile.as_dict(),
            dataset_manifest_sha256=manifest_sha256,
        )

    def _select_manifest(self, manifest: Mapping[str, Any], job_ids: tuple[str, ...]) -> dict[str, Any]:
        episode_ids: set[str] = set()
        attempt_ids: set[str] = set()
        for job_id in job_ids:
            result = json.loads((self.queue.root / "ingested" / job_id / "result.json").read_text(encoding="utf-8"))
            if result.get("episode_id"):
                episode_ids.add(str(result["episode_id"]))
            else:
                attempt_ids.add(str(result["attempt_id"]))
        return {
            "manifest_version": manifest.get("manifest_version", 3),
            "episodes": [row for row in manifest.get("episodes", ()) if str(row.get("episode_id")) in episode_ids],
            "failure_attempts": [row for row in manifest.get("failure_attempts", ()) if str(row.get("attempt_id")) in attempt_ids],
        }

    def _merge_manifest(self, local: Mapping[str, Any], remote: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(remote)
        for key in ("episodes", "failure_attempts"):
            by_id: dict[str, dict[str, Any]] = {}
            for row in list(remote.get(key) or []) + list(local.get(key) or []):
                identity = str(row.get("episode_id") or row.get("attempt_id") or "")
                if not identity:
                    raise ValueError(f"{key} row has no immutable identity")
                previous = by_id.get(identity)
                if previous is not None and not _same_identity(previous, row):
                    raise ValueError(f"remote immutable conflict: {identity}")
                by_id[identity] = dict(row)
            merged[key] = [by_id[key] for key in sorted(by_id)]
        merged["manifest_version"] = local.get("manifest_version", remote.get("manifest_version", 3))
        source_run_ids = {
            str(value)
            for source in (remote, local)
            for value in (source.get("source_run_ids") or ())
            if isinstance(value, str) and value.strip()
        }
        for source in (remote, local):
            value = source.get("run_id") or source.get("source_run_id")
            if isinstance(value, str) and value.strip():
                source_run_ids.add(value)
        source_run_ids.add(self.run.run_id)
        merged["source_run_ids"] = sorted(source_run_ids)
        merged["total_episodes"] = len(merged["episodes"])
        merged["total_failure_attempts"] = len(merged["failure_attempts"])
        if self.archive_profile is not None:
            archive_identity = {
                "archive_profile": self.archive_profile.as_dict(),
                "dataset_identity": self.archive_profile.dataset_identity,
                "archive_format_id": self.archive_profile.archive_format_id,
                "episode_schema_version": self.archive_profile.episode_schema_version,
                "view_status": "PROVISIONAL",
            }
            for key, expected in archive_identity.items():
                previous = remote.get(key)
                if previous is not None and previous != expected:
                    raise ValueError(f"remote archive identity conflict: {key}")
            merged.update(archive_identity)
        return merged

    def _manifest_from_states(self, states: tuple[str, ...]) -> dict[str, Any]:
        episodes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for state in states:
            for path in sorted((self.queue.root / state).iterdir()):
                if not path.is_dir():
                    continue
                result = json.loads((path / "result.json").read_text(encoding="utf-8"))
                result_root = self.queue.resolve_result_root(
                    str(result["result_dir"]), require_exists=True,
                )
                if self.archive_profile is not None:
                    job = GenerationJob.from_dict(
                        json.loads((path / "job.json").read_text(encoding="utf-8"))
                    )
                    validated = validate_closed_result(
                        job,
                        WorkerResult.from_dict(result),
                        archive_profile=self.archive_profile,
                    )
                    if validated.episode_entry is not None:
                        episodes.append(validated.episode_entry)
                    elif validated.attempt_entry is not None:
                        failures.append(validated.attempt_entry)
                    continue
                for filename, target in (("episode.json", episodes), ("attempt.json", failures)):
                    candidate = result_root / filename
                    if candidate.is_file():
                        target.append(json.loads(candidate.read_text(encoding="utf-8")))
        return {"manifest_version": 3, "episodes": episodes, "failure_attempts": failures}

    def _batch_manifest(self) -> dict[str, Any]:
        return self._manifest_from_states(("ingested",))

    def plan_batch(self, local_manifest: Mapping[str, Any], remote_manifest: Mapping[str, Any]) -> dict[str, Any]:
        merged = self._merge_manifest(local_manifest, remote_manifest)
        return {
            "manifest": merged,
            "job_ids": self.pending_job_ids(limit=self.publication_config.batch_size),
        }

    def _operations(
        self,
        job_ids: tuple[str, ...],
        manifest: Mapping[str, Any],
        receipt: PublicationReceipt | None = None,
    ) -> list[Any]:
        self._validate_prefix()
        operations: list[Any] = []
        for job_id in job_ids:
            directory = self.queue.root / "ingested" / job_id
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            result_root = self.queue.resolve_result_root(
                str(result["result_dir"]), require_exists=True,
            )
            if self.archive_profile is not None:
                job = GenerationJob.from_dict(
                    json.loads((directory / "job.json").read_text(encoding="utf-8"))
                )
                validated = validate_closed_result(
                    job,
                    WorkerResult.from_dict(result),
                    archive_profile=self.archive_profile,
                )
                if validated.episode_entry is not None:
                    collection, identity_key, expected_entry = (
                        "episodes", "episode_id", validated.episode_entry
                    )
                else:
                    collection, identity_key, expected_entry = (
                        "failure_attempts", "attempt_id", validated.attempt_entry
                    )
                if expected_entry is None:
                    raise ValueError("validated archive result did not produce a manifest row")
                rows = manifest.get(collection, ())
                if not isinstance(rows, (list, tuple)):
                    raise ValueError(f"dataset manifest {collection} must be a list")
                matching_rows = [
                    row for row in rows
                    if isinstance(row, Mapping)
                    and row.get(identity_key) == expected_entry[identity_key]
                ]
                if len(matching_rows) != 1 or dict(matching_rows[0]) != expected_entry:
                    raise ValueError(
                        f"dataset manifest archive row disagrees with validated result: {expected_entry[identity_key]}"
                    )
            program_id = str(result["program_id"])
            if result.get("episode_id"):
                record_prefix = f"episodes/{program_id}/{result['episode_id']}"
            else:
                record_prefix = f"attempts/{program_id}/{result['attempt_id']}"
            # The queue result already carries the validated immutable file
            # inventory. Reuse it instead of recursively stat'ing the entire
            # overlay filesystem for every large episode during publication.
            # Validation has already rejected missing/extra files before a job
            # reaches ``ingested``.
            for relative in sorted(result.get("file_sha256", {})):
                path = result_root / relative
                if not path.is_file():
                    raise FileNotFoundError(f"ingested artifact is missing: {path}")
                operations.append(CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/{record_prefix}/{relative}",
                    path_or_fileobj=str(path),
                ))
        manifest_path = self.queue.root / "publication_manifest.json"
        manifest_bytes = _canonical_json_bytes(manifest)
        _atomic_write_bytes(manifest_path, manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.run.hf_subfolder}/dataset_manifest.json",
            path_or_fileobj=str(manifest_path),
        ))
        resume_path = self.queue.root / "resume_receipt.json"
        counts: dict[str, dict[str, int]] = {}
        for row in list(manifest.get("episodes") or []):
            program = str(row.get("program_id")); outcome = str(row.get("outcome", "success"))
            counts.setdefault(program, {}).setdefault(outcome, 0)
            counts[program][outcome] += 1
        resume_payload: dict[str, Any] = {
            "receipt_version": 1,
            "status": "RUNNING",
            "run_id": self.run.run_id,
            "code_revision": self.run.code_revision,
            "episodes": len(manifest.get("episodes") or []),
            "failure_attempts": len(manifest.get("failure_attempts") or []),
            "per_program_outcomes": counts,
            "updated_at_s": time.time(),
        }
        if self.archive_profile is not None:
            resume_payload.update({
                "source_run_id": self.run.run_id,
                "source_run_ids": list(manifest.get("source_run_ids") or (self.run.run_id,)),
                "dataset_identity": self.archive_profile.dataset_identity,
                "archive_format_id": self.archive_profile.archive_format_id,
                "episode_schema_version": self.archive_profile.episode_schema_version,
                "archive_profile": self.archive_profile.as_dict(),
                "dataset_manifest_sha256": manifest_sha256,
                "dataset_manifest_revision": receipt.data_commit_oid if receipt is not None else None,
            })
        _atomic_write_bytes(resume_path, _canonical_json_bytes(resume_payload))
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.run.hf_subfolder}/resume_receipt.json",
            path_or_fileobj=str(resume_path),
        ))
        view_root = self.queue.root / "views"
        archive_refs = [
            str(row["archive_ref"])
            for row in manifest.get("episodes") or []
            if isinstance(row, Mapping) and isinstance(row.get("archive_ref"), str)
        ]
        for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
            view_path = view_root / f"{view}.json"
            view_path.parent.mkdir(parents=True, exist_ok=True)
            if self.archive_profile is None:
                view_payload = {
                    "view": view,
                    "schema_version": "icgs_episode_v2",
                    "episode_ids": [row.get("episode_id") for row in manifest.get("episodes") or []],
                    "pointers_only": True,
                }
            else:
                view_payload = {
                    "view": view,
                    "status": "PROVISIONAL",
                    "view_status": "PROVISIONAL",
                    "dataset_identity": self.archive_profile.dataset_identity,
                    "archive_format_id": self.archive_profile.archive_format_id,
                    "episode_schema_version": self.archive_profile.episode_schema_version,
                    "dataset_manifest_sha256": manifest_sha256,
                    "archive_refs": archive_refs,
                    "pointers_only": True,
                }
            _atomic_write_bytes(view_path, _canonical_json_bytes(view_payload))
            operations.append(CommitOperationAdd(
                path_in_repo=f"{self.run.hf_subfolder}/views/{view}.json",
                path_or_fileobj=str(view_path),
            ))
        publication_path = self._publication_receipt_path
        bound_receipt = self._bind_archive_receipt(
            receipt
            if receipt is not None
            else PublicationReceipt(
                run_id=self.run.run_id,
                job_ids=job_ids,
                prefix=self.run.hf_subfolder,
                created_at_s=time.time(),
                updated_at_s=time.time(),
            ),
            manifest,
        )
        if (
            self.archive_profile is not None
            and bound_receipt.dataset_manifest_sha256 != manifest_sha256
        ):
            raise ValueError("publication receipt dataset manifest hash does not match operations")
        _atomic_write_bytes(publication_path, _canonical_json_bytes(bound_receipt.as_dict()))
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
            path_or_fileobj=str(publication_path),
        ))
        return operations

    def publish_due(
        self,
        *,
        now_s: float,
        force: bool,
        complete: bool = False,
        remote_manifest: Mapping[str, Any] | None = None,
        local_manifest: Mapping[str, Any] | None = None,
    ) -> PublicationReceipt | None:
        self._validate_prefix()
        remote = self.remote_manifest if remote_manifest is None else dict(remote_manifest)
        local_receipt = self._read_publication_receipt()
        if local_receipt is not None and local_receipt.status == "COMPLETE":
            pending_ids = set(self.pending_job_ids())
            completed_ids = set(local_receipt.job_ids)
            if pending_ids - completed_ids:
                # A COMPLETE receipt belongs to an earlier batch. Keep its
                # immutable identity for reconciliation only when no newer
                # ingested jobs are waiting; otherwise plan a fresh batch.
                self._clear_stale_verification_receipt_snapshot()
                local_receipt = None
        if local_receipt is not None:
            reconciled = reconcile_publication(local_receipt, remote)
            if (
                self.archive_profile is not None
                and local_receipt.status not in {"VERIFIED", "COMPLETE"}
                and reconciled.status == "VERIFIED"
            ):
                # Matching receipt metadata is not a remote byte verification.
                # Keep the archived batch in its committed state until the
                # coordinator verifies every file at the pinned revision.
                reconciled = replace(
                    local_receipt,
                    status=(
                        "DATA_COMMITTED"
                        if local_receipt.data_commit_oid
                        else local_receipt.status
                    ),
                )
            if reconciled.status == "VERIFIED":
                return self._complete_verified(reconciled, now_s)
            if local_receipt.status == "COMPLETE":
                return local_receipt
            if local_receipt.status in {"DATA_COMMITTED", "DEFERRED"}:
                if self.archive_profile is not None and self.remote_verify is None:
                    raise ValueError("archive publication requires remote hash verification")
                next_retry_s = local_receipt.next_retry_s or 0.0
                if now_s < next_retry_s and reconciled.status != "VERIFIED":
                    return None
                if self.archive_profile is None or local_receipt.data_commit_oid:
                    completed = self._commit_receipt(reconciled, now_s)
                    if completed is not None and completed.status == "COMPLETE":
                        self.remote_manifest = self._read_completed_manifest_snapshot(completed)
                    return completed
                # A transient failure before Hugging Face returned a data OID
                # has no immutable revision to verify. Retry the same ingested
                # batch; once an OID is recorded, subsequent retries use it.
                local_receipt = replace(
                    local_receipt,
                    status="PREPARED",
                    commit_oid=None,
                    last_error=None,
                    next_retry_s=None,
                )

        if (
            local_receipt is None
            and not force
            and not complete
            and now_s - self.last_success_s < self.run.publish_interval_s
        ):
            return None
        if local_receipt is None and now_s < self._next_retry_s:
            return None

        job_ids = (
            local_receipt.job_ids
            if local_receipt is not None
            else self.pending_job_ids(limit=self.publication_config.batch_size)
        )
        if not job_ids:
            return None
        if self.archive_profile is not None and self.remote_verify is None:
            raise ValueError("archive publication requires remote hash verification")
        source_manifest = self._batch_manifest() if local_manifest is None else local_manifest
        local_manifest_slice = self._select_manifest(source_manifest, job_ids)
        plan = self.plan_batch(local_manifest_slice, remote)
        now = time.time()
        prepared = local_receipt or PublicationReceipt(
            run_id=self.run.run_id,
            job_ids=job_ids,
            status="PREPARED",
            artifact_hashes=self._artifact_hashes(job_ids),
            prefix=self.run.hf_subfolder,
            created_at_s=now,
            updated_at_s=now,
        )
        prepared = self._bind_archive_receipt(prepared, plan["manifest"])
        operations = self._operations(job_ids, plan["manifest"], prepared)
        prepared = replace(prepared, path_count=len(operations), updated_at_s=now)
        self._write_publication_receipt(prepared)
        try:
            data_commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add generation batch ({len(job_ids)} results)",
                token=self._token,
                num_threads=self.publication_config.upload_threads,
            ))
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            self._write_publication_receipt(self._deferred_receipt(prepared, now_s, error))
            return None
        data_oid = str(getattr(data_commit, "oid", getattr(data_commit, "commit_hash", "")))
        if not data_oid:
            raise RuntimeError("Hugging Face data commit returned no revision")
        data_committed = replace(
            prepared,
            status="DATA_COMMITTED",
            data_commit_oid=data_oid,
            updated_at_s=now_s,
            last_error=None,
            next_retry_s=None,
        )
        self._write_publication_receipt(data_committed)
        resolved = self._commit_receipt(data_committed, now_s)
        if resolved is not None and resolved.status == "COMPLETE":
            self.remote_manifest = self._read_completed_manifest_snapshot(resolved)
        return resolved

    def _deferred_receipt(
        self,
        receipt: PublicationReceipt,
        now_s: float,
        error: BaseException,
    ) -> PublicationReceipt:
        return replace(
            receipt,
            status="DEFERRED",
            updated_at_s=now_s,
            last_error=str(error)[:2000],
            next_retry_s=self._next_retry_s,
        )

    def _commit_receipt(
        self,
        receipt: PublicationReceipt,
        now_s: float,
    ) -> PublicationReceipt | None:
        if self.archive_profile is not None:
            return self._commit_archive_receipt(receipt, now_s)
        if receipt.status == "VERIFIED":
            completed = self._complete_verified(receipt, now_s)
            self._verification_receipt_path.unlink(missing_ok=True)
            return completed
        publication_path = self._publication_receipt_path
        self._write_publication_receipt(receipt)
        verification_path = (
            self._ensure_verification_receipt_snapshot(receipt)
            if self.remote_verify is not None
            else publication_path
        )
        try:
            final_commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=[CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
                    path_or_fileobj=str(verification_path),
                )],
                commit_message=f"Finalize generation publication receipt ({len(receipt.job_ids)} results)",
                token=self._token,
                num_threads=self.publication_config.upload_threads,
            ))
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            self._write_publication_receipt(self._deferred_receipt(receipt, now_s, error))
            return None
        receipt_oid = str(
            getattr(final_commit, "oid", getattr(final_commit, "commit_hash", ""))
        )
        if not receipt_oid:
            raise RuntimeError("Hugging Face receipt commit returned no revision")
        verified = replace(
            receipt,
            status="VERIFIED",
            commit_oid=receipt_oid,
            updated_at_s=now_s,
            last_error=None,
            next_retry_s=None,
        )
        self._write_publication_receipt(verified)
        if self.remote_verify is not None:
            if (
                self.archive_profile is not None
                and (
                    self._verification_receipt_path.is_symlink()
                    or not self._verification_receipt_path.is_file()
                )
            ):
                raise ValueError(
                    "archive publication is missing the exact receipt bytes used for verification"
                )
            try:
                self._with_transient_retry(
                    lambda: self.remote_verify(receipt.job_ids, receipt_oid, self._token)
                )
            except _TransientPublicationDeferred as error:
                self._defer_transient_failure(now_s, error)
                self._write_publication_receipt(self._deferred_receipt(verified, now_s, error))
                return None
        completed = self._complete_verified(verified, now_s)
        self._verification_receipt_path.unlink(missing_ok=True)
        return completed

    def _commit_archive_receipt(
        self,
        receipt: PublicationReceipt,
        now_s: float,
    ) -> PublicationReceipt | None:
        """Verify immutable archive bytes before publishing verified state or pruning."""
        if receipt.status == "VERIFIED":
            completed = self._complete_verified(receipt, now_s)
            self._verification_receipt_path.unlink(missing_ok=True)
            return completed
        if self.remote_verify is None:
            raise ValueError("archive publication requires remote hash verification")
        if not receipt.data_commit_oid:
            raise ValueError("archive receipt is missing its data commit OID")

        candidate = replace(
            receipt,
            status="DATA_COMMITTED",
            updated_at_s=now_s,
            last_error=None,
            next_retry_s=None,
        )
        if not candidate.commit_oid:
            publication_path = self._publication_receipt_path
            resume_path = self.queue.root / "resume_receipt.json"
            try:
                resume_payload = json.loads(resume_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("archive publication is missing its resume receipt") from error
            if resume_payload.get("dataset_manifest_sha256") != candidate.dataset_manifest_sha256:
                raise ValueError("resume receipt dataset manifest hash mismatch")
            resume_payload["dataset_manifest_revision"] = candidate.data_commit_oid
            _atomic_write_bytes(resume_path, _canonical_json_bytes(resume_payload))
            self._write_publication_receipt(candidate)
            verification_path = self._ensure_verification_receipt_snapshot(candidate)
            metadata_operations = [
                CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
                    path_or_fileobj=str(verification_path),
                ),
                CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/resume_receipt.json",
                    path_or_fileobj=str(resume_path),
                ),
            ]
            try:
                receipt_commit = self._with_transient_retry(lambda: self.api.create_commit(
                    repo_id=self.run.hf_repo,
                    repo_type="dataset",
                    operations=metadata_operations,
                    commit_message=f"Record committed generation archive ({len(receipt.job_ids)} results)",
                    token=self._token,
                    num_threads=self.publication_config.upload_threads,
                ))
            except _TransientPublicationDeferred as error:
                self._defer_transient_failure(now_s, error)
                self._write_publication_receipt(self._deferred_receipt(candidate, now_s, error))
                return None
            receipt_oid = str(
                getattr(receipt_commit, "oid", getattr(receipt_commit, "commit_hash", ""))
            )
            if not receipt_oid:
                raise RuntimeError("Hugging Face archive receipt commit returned no revision")
            candidate = replace(candidate, commit_oid=receipt_oid)
            self._write_publication_receipt(candidate)

        try:
            if (
                self._verification_receipt_path.is_symlink()
                or not self._verification_receipt_path.is_file()
            ):
                raise ValueError(
                    "archive publication is missing the exact receipt bytes used for verification"
                )
            self._with_transient_retry(
                lambda: self.remote_verify(candidate.job_ids, candidate.commit_oid, self._token)
            )
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            self._write_publication_receipt(self._deferred_receipt(candidate, now_s, error))
            return None
        except Exception as error:
            failed = replace(
                candidate,
                status="DATA_COMMITTED",
                updated_at_s=now_s,
                last_error=str(error)[:2000],
                next_retry_s=None,
            )
            self._write_publication_receipt(failed)
            raise

        verified = replace(
            candidate,
            status="VERIFIED",
            updated_at_s=now_s,
            last_error=None,
            next_retry_s=None,
        )
        # The verified receipt is a metadata-only commit. Its commit_oid names
        # the earlier pinned revision whose complete archive inventory passed
        # verification; dataset_manifest_revision remains the data commit OID.
        snapshot_path = self.queue.root / (
            f".publication-receipt-verified-{os.getpid()}-{time.time_ns()}.json"
        )
        _atomic_write_bytes(snapshot_path, _canonical_json_bytes(verified.as_dict()))
        try:
            final_commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=[CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
                    path_or_fileobj=str(snapshot_path),
                )],
                commit_message=f"Verify generation archive receipt ({len(receipt.job_ids)} results)",
                token=self._token,
                num_threads=self.publication_config.upload_threads,
            ))
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            self._write_publication_receipt(self._deferred_receipt(candidate, now_s, error))
            return None
        except Exception as error:
            failed = replace(
                candidate,
                status="DATA_COMMITTED",
                updated_at_s=now_s,
                last_error=str(error)[:2000],
                next_retry_s=None,
            )
            self._write_publication_receipt(failed)
            raise
        finally:
            snapshot_path.unlink(missing_ok=True)
        final_oid = str(
            getattr(final_commit, "oid", getattr(final_commit, "commit_hash", ""))
        )
        if not final_oid:
            raise RuntimeError("Hugging Face verified receipt commit returned no revision")
        self._write_publication_receipt(verified)
        completed = self._complete_verified(verified, now_s)
        self._verification_receipt_path.unlink(missing_ok=True)
        return completed

    def _complete_verified(
        self,
        receipt: PublicationReceipt,
        now_s: float,
    ) -> PublicationReceipt:
        verified = replace(receipt, status="VERIFIED", updated_at_s=now_s)
        self._write_publication_receipt(verified)
        for job_id in verified.job_ids:
            retention = (
                self.archive_profile.local_artifact_retention
                if self.archive_profile is not None
                else "keep"
            )
            self.queue.mark_published(job_id, retention=retention)
        complete = replace(
            verified,
            status="COMPLETE",
            updated_at_s=now_s,
            published_at_s=now_s,
            next_retry_s=None,
        )
        self._write_publication_receipt(complete)
        if self.archive_profile is not None:
            self.remote_manifest = self._read_completed_manifest_snapshot(complete)
        self.last_success_s = now_s
        self.receipts.append(complete)
        self._clear_retry_state()
        self._verification_receipt_path.unlink(missing_ok=True)
        return complete


__all__ = [
    "HuggingFaceBatchPublisher",
    "PublicationConfig",
    "PublicationReceipt",
    "PUBLICATION_STATES",
    "reconcile_publication",
]
