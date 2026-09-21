"""Five-minute, conflict-safe Hugging Face publication for v3 results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from huggingface_hub import CommitOperationAdd

from icgs.data.collection.v3.distributed_contracts import RunConfig
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


@dataclass(frozen=True)
class PublicationReceipt:
    commit_oid: str
    job_ids: tuple[str, ...]
    published_at_s: float
    path_count: int


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class HuggingFaceBatchPublisher:
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

    def pending_job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in (self.queue.root / "ingested").iterdir() if path.is_dir()))

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
            for path in sorted(result_root.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(result_root).as_posix()
                    operations.append(CommitOperationAdd(
                        path_in_repo=f"{self.run.hf_subfolder}/results/{job_id}/{relative}",
                        path_or_fileobj=str(path),
                    ))
        manifest_path = self.queue.root / "publication_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.run.hf_subfolder}/dataset_manifest.json",
            path_or_fileobj=str(manifest_path),
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
        job_ids = self.pending_job_ids()
        if not job_ids:
            return None
        remote = self.remote_manifest if remote_manifest is None else dict(remote_manifest)
        local = self._batch_manifest() if local_manifest is None else dict(local_manifest)
        plan = self.plan_batch(local, remote)
        operations = self._operations(job_ids, plan["manifest"])
        commit = self.api.create_commit(
            repo_id=self.run.hf_repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add primary v3 batch ({len(job_ids)} results)",
            token=self._token,
        )
        oid = str(getattr(commit, "oid", getattr(commit, "commit_hash", "")))
        if not oid:
            raise RuntimeError("Hugging Face commit returned no revision")
        if self.remote_verify is not None:
            self.remote_verify(job_ids, oid, self._token)
        self.remote_manifest = dict(plan["manifest"])
        receipt = PublicationReceipt(oid, job_ids, now_s, len(operations))
        for job_id in job_ids:
            self.queue.mark_published(job_id)
        self.last_success_s = now_s
        self.receipts.append(receipt)
        return receipt


__all__ = ["HuggingFaceBatchPublisher", "PublicationReceipt"]
