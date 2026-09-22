"""Five-minute, conflict-safe Hugging Face publication for generation results."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from huggingface_hub import CommitOperationAdd

from icgs.data.collection.generation.distributed_contracts import RunConfig
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue


@dataclass(frozen=True)
class PublicationReceipt:
    commit_oid: str
    job_ids: tuple[str, ...]
    published_at_s: float
    path_count: int


class _TransientPublicationDeferred(RuntimeError):
    """Internal signal that a temporary HF failure was safely deferred."""


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


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
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Hugging Face token is required")
        self.run = run
        self.api = api
        self._token = token
        self.queue = queue
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
        cooldown_s = self.RATE_LIMIT_COOLDOWN_S if status == 429 else self.TRANSIENT_RETRY_COOLDOWN_S
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
        delay_s = self.TRANSIENT_RETRY_BASE_S
        for attempt in range(1, self.MAX_TRANSIENT_ATTEMPTS + 1):
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
        return {"manifest": merged, "job_ids": self.pending_job_ids()}

    def _operations(self, job_ids: tuple[str, ...], manifest: Mapping[str, Any]) -> list[Any]:
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
        publication_path = self.queue.root / "publication_receipt.json"
        publication_path.write_text(json.dumps({
            "receipt_version": 1,
            "run_id": self.run.run_id,
            "job_ids": list(job_ids),
            "commit_oid": None,
            "status": "COMMIT_PENDING",
            "created_at_s": time.time(),
        }, indent=2) + "\n", encoding="utf-8")
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
        if not force and not complete and now_s - self.last_success_s < self.run.publish_interval_s:
            return None
        if now_s < self._next_retry_s:
            return None
        job_ids = self.pending_job_ids(limit=self.MAX_JOBS_PER_COMMIT)
        if not job_ids:
            return None
        remote = self.remote_manifest if remote_manifest is None else dict(remote_manifest)
        source_manifest = self._batch_manifest() if local_manifest is None else local_manifest
        local = self._select_manifest(source_manifest, job_ids)
        plan = self.plan_batch(local, remote)
        operations = self._operations(job_ids, plan["manifest"])
        try:
            commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add generation batch ({len(job_ids)} results)",
                token=self._token,
                num_threads=1,
            ))
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            return None
        oid = str(getattr(commit, "oid", getattr(commit, "commit_hash", "")))
        if not oid:
            raise RuntimeError("Hugging Face commit returned no revision")
        publication_path = self.queue.root / "publication_receipt.json"
        publication_path.write_text(json.dumps({
            "receipt_version": 1,
            "run_id": self.run.run_id,
            "job_ids": list(job_ids),
            "commit_oid": oid,
            "status": "COMMITTED",
            "created_at_s": time.time(),
        }, indent=2) + "\n", encoding="utf-8")
        try:
            final_commit = self._with_transient_retry(lambda: self.api.create_commit(
                repo_id=self.run.hf_repo,
                repo_type="dataset",
                operations=[CommitOperationAdd(
                    path_in_repo=f"{self.run.hf_subfolder}/publication_receipt.json",
                    path_or_fileobj=str(publication_path),
                )],
                commit_message=f"Finalize generation publication receipt ({len(job_ids)} results)",
                token=self._token,
            ))
        except _TransientPublicationDeferred as error:
            self._defer_transient_failure(now_s, error)
            return None
        oid = str(getattr(final_commit, "oid", getattr(final_commit, "commit_hash", oid)))
        if self.remote_verify is not None:
            try:
                self._with_transient_retry(lambda: self.remote_verify(job_ids, oid, self._token))
            except _TransientPublicationDeferred as error:
                self._defer_transient_failure(now_s, error)
                return None
        self.remote_manifest = dict(plan["manifest"])
        receipt = PublicationReceipt(oid, job_ids, now_s, len(operations))
        for job_id in job_ids:
            self.queue.mark_published(job_id)
        self.last_success_s = now_s
        self.receipts.append(receipt)
        self._clear_retry_state()
        return receipt


__all__ = ["HuggingFaceBatchPublisher", "PublicationReceipt"]
