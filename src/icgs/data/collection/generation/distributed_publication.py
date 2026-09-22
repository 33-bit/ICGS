"""Five-minute, conflict-safe Hugging Face publication for generation results."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from huggingface_hub import CommitOperationAdd

from icgs.data.collection.generation.distributed_contracts import RunConfig
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
        ("prefix", local.prefix),
        ("data_commit_oid", local.data_commit_oid),
    ):
        remote_value_for_key = remote.get(key)
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
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Hugging Face token is required")
        self.run = run
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
        temporary = self._retry_state_path.with_name(
            self._retry_state_path.name + f".tmp-{os.getpid()}-{time.time_ns()}"
        )
        temporary.write_text(json.dumps({
            "next_retry_s": next_retry_s,
            "error": error[:2000],
            "updated_at_s": time.time(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self._retry_state_path)

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

    def _write_publication_receipt(self, receipt: PublicationReceipt) -> None:
        path = self._publication_receipt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".partial-{os.getpid()}-{time.time_ns()}")
        temporary.write_text(
            json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _read_publication_receipt(self) -> PublicationReceipt | None:
        try:
            return PublicationReceipt.from_dict(
                json.loads(self._publication_receipt_path.read_text(encoding="utf-8"))
            )
        except FileNotFoundError:
            return None

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
        merged["total_episodes"] = len(merged["episodes"])
        merged["total_failure_attempts"] = len(merged["failure_attempts"])
        return merged

    def _manifest_from_states(self, states: tuple[str, ...]) -> dict[str, Any]:
        episodes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for state in states:
            for path in sorted((self.queue.root / state).iterdir()):
                if not path.is_dir():
                    continue
                result = json.loads((path / "result.json").read_text(encoding="utf-8"))
                result_root = Path(result["result_dir"])
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
            result_root = Path(result["result_dir"])
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
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        resume_path.write_text(json.dumps({
            "receipt_version": 1,
            "status": "RUNNING",
            "run_id": self.run.run_id,
            "code_revision": self.run.code_revision,
            "episodes": len(manifest.get("episodes") or []),
            "failure_attempts": len(manifest.get("failure_attempts") or []),
            "per_program_outcomes": counts,
            "updated_at_s": time.time(),
        }, indent=2) + "\n", encoding="utf-8")
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.run.hf_subfolder}/resume_receipt.json",
            path_or_fileobj=str(resume_path),
        ))
        view_root = self.queue.root / "views"
        for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
            view_path = view_root / f"{view}.json"
            view_path.parent.mkdir(parents=True, exist_ok=True)
            view_path.write_text(json.dumps({
                "view": view,
                "schema_version": "icgs_episode_v2",
                "episode_ids": [row.get("episode_id") for row in manifest.get("episodes") or []],
                "pointers_only": True,
            }, indent=2) + "\n", encoding="utf-8")
            operations.append(CommitOperationAdd(
                path_in_repo=f"{self.run.hf_subfolder}/views/{view}.json",
                path_or_fileobj=str(view_path),
            ))
        publication_path = self._publication_receipt_path
        publication_payload = (
            receipt.as_dict()
            if receipt is not None
            else PublicationReceipt(
                run_id=self.run.run_id,
                job_ids=job_ids,
                prefix=self.run.hf_subfolder,
                created_at_s=time.time(),
                updated_at_s=time.time(),
            ).as_dict()
        )
        publication_path.write_text(
            json.dumps(publication_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
        if local_receipt is not None:
            reconciled = reconcile_publication(local_receipt, remote)
            if reconciled.status == "VERIFIED":
                return self._complete_verified(reconciled, now_s)
            if local_receipt.status == "COMPLETE":
                return local_receipt
            if local_receipt.status in {"DATA_COMMITTED", "DEFERRED"}:
                next_retry_s = local_receipt.next_retry_s or 0.0
                if now_s < next_retry_s and reconciled.status != "VERIFIED":
                    return None
                return self._commit_receipt(reconciled, now_s)

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
        if resolved is not None:
            self.remote_manifest = dict(plan["manifest"])
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
        if receipt.status == "VERIFIED":
            return self._complete_verified(receipt, now_s)
        publication_path = self._publication_receipt_path
        self._write_publication_receipt(receipt)
        try:
            final_commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=[CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
                    path_or_fileobj=str(publication_path),
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
            try:
                self._with_transient_retry(
                    lambda: self.remote_verify(receipt.job_ids, receipt_oid, self._token)
                )
            except _TransientPublicationDeferred as error:
                self._defer_transient_failure(now_s, error)
                self._write_publication_receipt(self._deferred_receipt(verified, now_s, error))
                return None
        return self._complete_verified(verified, now_s)

    def _complete_verified(
        self,
        receipt: PublicationReceipt,
        now_s: float,
    ) -> PublicationReceipt:
        verified = replace(receipt, status="VERIFIED", updated_at_s=now_s)
        self._write_publication_receipt(verified)
        for job_id in verified.job_ids:
            self.queue.mark_published(job_id)
        complete = replace(
            verified,
            status="COMPLETE",
            updated_at_s=now_s,
            published_at_s=now_s,
            next_retry_s=None,
        )
        self._write_publication_receipt(complete)
        self.last_success_s = now_s
        self.receipts.append(complete)
        self._clear_retry_state()
        return complete


__all__ = [
    "HuggingFaceBatchPublisher",
    "PublicationConfig",
    "PublicationReceipt",
    "PUBLICATION_STATES",
    "reconcile_publication",
]
