"""Single-owner coordinator for distributed generation generation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download

from icgs.data.collection.generation.distributed_contracts import (
    CoordinatorHeartbeat,
    GenerationJob,
    GenerationRuntimeConfig,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_planner import DistributedPlanner
from icgs.data.collection.generation.distributed_publication import HuggingFaceBatchPublisher
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue, MalformedReadyResult
from icgs.data.collection.generation.distributed_validation import ingest_validated_result, validate_closed_result
from icgs.data.collection.generation.distributed_validation import ValidationFailure


MAX_READY_PER_TICK = 100


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CoordinatorLock:
    """Non-blocking filesystem lock held for the complete coordinator lifetime."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("coordinator lock is already held") from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"{os.getpid()}\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None
        return False


def _inflight_jobs_from_queue(queue: FilesystemJobQueue):
    from icgs.data.collection.generation.distributed_contracts import GenerationJob

    paths = list((queue.root / "pending").glob("*.json"))
    paths.extend(
        path for path in (queue.root / "claimed").glob("*/*.json")
        if not path.name.endswith(".claim.json")
    )
    jobs = [GenerationJob.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    for state in ("ready", "quarantined"):
        state_root = queue.root / state
        if state_root.is_symlink() or not state_root.is_dir():
            raise ValueError(f"queue state must be a real directory: {state_root}")
        for directory in sorted(state_root.iterdir()):
            if ".partial-" in directory.name:
                continue
            try:
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    continue
                job_path = directory / "job.json"
                if job_path.is_symlink() or not job_path.is_file():
                    continue
                jobs.append(
                    GenerationJob.from_dict(
                        json.loads(job_path.read_text(encoding="utf-8"))
                    )
                )
            except (OSError, TypeError, ValueError, KeyError):
                continue
    by_id = {job.job_id: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate in-flight jobs while resuming coordinator")
    return tuple(by_id[key] for key in sorted(by_id))


def _manifest_from_closed_queue(queue: FilesystemJobQueue, states=("ingested", "published")) -> dict:
    from icgs.data.collection.generation.distributed_contracts import GenerationJob, WorkerResult

    manifest = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
    for state in states:
        for directory in sorted((queue.root / state).iterdir()):
            if not directory.is_dir():
                continue
            job = GenerationJob.from_dict(json.loads((directory / "job.json").read_text(encoding="utf-8")))
            result = WorkerResult.from_dict(json.loads((directory / "result.json").read_text(encoding="utf-8")))
            manifest = ingest_validated_result(manifest, validate_closed_result(job, result))
    return manifest


def _verify_remote_batch(queue: FilesystemJobQueue, run: RunConfig, job_ids: tuple[str, ...], revision: str, token: str) -> None:
    for job_id in job_ids:
        directory = queue.root / "ingested" / job_id
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        result_root = Path(result["result_dir"])
        prefix = (
            f"{run.hf_subfolder}/episodes/{result['program_id']}/{result['episode_id']}"
            if result.get("episode_id")
            else f"{run.hf_subfolder}/attempts/{result['program_id']}/{result['attempt_id']}"
        )
        for path in sorted(result_root.rglob("*")):
            if not path.is_file():
                continue
            remote = hf_hub_download(
                repo_id=run.hf_repo,
                repo_type="dataset",
                filename=f"{prefix}/{path.relative_to(result_root).as_posix()}",
                revision=revision,
                token=token,
                force_download=True,
            )
            local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            remote_hash = hashlib.sha256(Path(remote).read_bytes()).hexdigest()
            if local_hash != remote_hash:
                raise ValueError(f"remote hash mismatch for {job_id}/{path.name}")


class CoordinatorControlPlane:
    def __init__(self, run: RunConfig, queue: FilesystemJobQueue, planner: DistributedPlanner, publisher: HuggingFaceBatchPublisher, manifest=None):
        self.run = run
        self.queue = queue
        self.planner = planner
        self.publisher = publisher
        self.manifest = dict(manifest or {"manifest_version": 3, "episodes": [], "failure_attempts": []})
        self.status = "PREFLIGHT"
        self._tick_started_at_s: float | None = None
        self._last_progress_at_s: float | None = None
        self._last_progress_kind: str | None = None
        self._last_validation_error: dict | None = None
        self._publication: dict = {"phase": "idle"}

    @classmethod
    def open(cls, run_config_path: str | Path, *, api_factory):
        payload = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
        run = RunConfig.from_dict(payload["run"])
        queue = FilesystemJobQueue(run.run_root + "/queue")
        approved = json.loads(Path(payload["approved_manifest"]).read_text(encoding="utf-8"))
        rows = {row["program_id"]: row for row in approved["catalog"]}
        manifest = _manifest_from_closed_queue(queue)
        if not manifest["episodes"] and not manifest["failure_attempts"]:
            manifest = payload.get("manifest", manifest)
        planner = DistributedPlanner.from_manifest(run, rows, manifest, _inflight_jobs_from_queue(queue))
        api = api_factory()
        token = Path("/content/.icgs_hf_token").read_text(encoding="utf-8").strip()
        publisher = HuggingFaceBatchPublisher(
            run, api, token, queue,
            remote_verify=lambda job_ids, revision, _token: _verify_remote_batch(
                queue, run, job_ids, revision, token
            ),
        )
        try:
            remote_path = hf_hub_download(
                repo_id=run.hf_repo,
                repo_type="dataset",
                filename=f"{run.hf_subfolder}/dataset_manifest.json",
                token=token,
                force_download=True,
            )
            publisher.remote_manifest = json.loads(Path(remote_path).read_text(encoding="utf-8"))
        except Exception:
            publisher.remote_manifest = _manifest_from_closed_queue(queue, states=("published",))
        return cls(run, queue, planner, publisher, manifest)

    def _refill(self, target: int = 400) -> int:
        refilled = 0
        while self.queue.counts().pending + self.queue.counts().claimed < target:
            job = self.planner.next_job()
            if job is None:
                break
            self.queue.enqueue(job)
            refilled += 1
        return refilled

    def _record_progress(self, now_s: float, kind: str) -> None:
        self._last_progress_at_s = now_s
        self._last_progress_kind = kind

    def process_ready_result(
        self,
        result: WorkerResult | MalformedReadyResult,
        *,
        now_s: float | None = None,
    ) -> Literal["ingested", "quarantined"]:
        now = (
            time.time()
            if now_s is None and self._tick_started_at_s is None
            else self._tick_started_at_s
            if now_s is None
            else now_s
        )
        source_path = self.queue.root / "ready" / result.job_id
        job = None
        inventory_root = source_path
        self._write_heartbeat(now, phase="validation")
        try:
            job_path = source_path / "job.json"
            if job_path.is_symlink() or not job_path.is_file():
                raise ValueError("ready job.json must be a regular file")
            job = GenerationJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
            if job.job_id != result.job_id:
                raise ValueError("ready job/result identity mismatch")
            if isinstance(result, MalformedReadyResult):
                inventory_root = (
                    Path(job.output_root)
                    / "worker-results"
                    / job.job_id
                    / job.program_id
                )
                raise result.error
            inventory_root = Path(result.result_dir)
            allowed_result_root = self.queue.root.parent.resolve()
            if (
                inventory_root.is_symlink()
                or not inventory_root.resolve().is_relative_to(allowed_result_root)
            ):
                raise ValueError("result_dir must be contained within the run root")
            validated = validate_closed_result(job, result)
            updated_manifest = ingest_validated_result(self.manifest, validated)
        except Exception as error:
            failure = ValidationFailure.capture(
                job,
                result if isinstance(result, WorkerResult) else None,
                error,
                source_path=source_path,
                allowed_result_root=self.queue.root.parent,
                job_id=result.job_id,
                result_identity=(
                    result.result_identity
                    if isinstance(result, MalformedReadyResult)
                    else None
                ),
                result_dir=inventory_root,
            )
            self.queue.quarantine_ready(result.job_id, failure)
            self._last_validation_error = {
                "job_id": failure.job_id,
                "exception_type": failure.exception_type,
                "exception_message": failure.exception_message,
            }
            self._write_heartbeat(now, phase="validation")
            return "quarantined"

        self.planner.record_result(result, validated.provenance)
        self.queue.mark_ingested(result)
        self.manifest = updated_manifest
        self._record_progress(now, "result_ingested")
        self._write_heartbeat(now, phase="validation")
        return "ingested"

    def tick(self, *, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else now_s
        self._tick_started_at_s = now
        self._write_heartbeat(now, phase="tick_start")
        self.queue.recover_stale(now_s=now, stale_after_s=1800.0)
        for result in self.queue.iter_ready()[:MAX_READY_PER_TICK]:
            self.process_ready_result(result)
        self._write_heartbeat(now, phase="refill")
        refilled = self._refill()
        if refilled:
            self._record_progress(now, "jobs_refilled")
        self._write_heartbeat(now, phase="refill")
        complete = self.planner.quota_complete()
        self._publication = {"phase": "publication_in_progress", "started_at_s": now}
        self._write_heartbeat(now, phase="publication_in_progress")
        try:
            receipt = self.publisher.publish_due(
                now_s=now, force=complete, complete=complete,
                local_manifest=self.manifest,
            )
        except Exception as error:
            self._publication = {
                "phase": "error",
                "started_at_s": now,
                "last_error": {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            }
            self._write_heartbeat(now, phase="publication_error")
            raise
        if receipt is None:
            self._publication = {"phase": "idle", "last_attempt_at_s": now}
        else:
            self._publication = {
                "phase": "complete",
                "started_at_s": now,
                "completed_at_s": getattr(receipt, "published_at_s", now),
                "job_ids": list(getattr(receipt, "job_ids", ())),
            }
            self._record_progress(now, "publication_complete")
        self._write_heartbeat(now, phase="publication")
        self.status = "COMPLETE" if complete and not self.queue.counts().claimed else "RUNNING"
        self._write_heartbeat(now, phase="tick_end")

    def _write_heartbeat(self, now: float, *, phase: str) -> None:
        control = self.queue.root.parent / "control"
        heartbeat = CoordinatorHeartbeat(
            status=self.status,
            phase=phase,
            timestamp_s=now,
            pid=os.getpid(),
            tick_started_at_s=self._tick_started_at_s,
            last_progress_at_s=self._last_progress_at_s,
            last_progress_kind=self._last_progress_kind,
            last_validation_error=self._last_validation_error,
            publication=dict(self._publication),
            queue=dict(self.queue.counts().__dict__),
            planner=dict(self.planner.snapshot().as_dict()),
        )
        _atomic_json(control / "coordinator-heartbeat.json", heartbeat.as_dict())

    def run_forever(self, *, poll_s: float = 1.0) -> int:
        control = self.queue.root.parent / "control"
        with CoordinatorLock(control / "coordinator.lock"):
            _atomic_json(control / "coordinator.pid", {"pid": os.getpid(), "run_id": self.run.run_id})
            try:
                self.status = "RUNNING"
                while self.status not in {"COMPLETE", "INCOMPLETE", "FAILED"}:
                    self.tick()
                    if self.status not in {"COMPLETE", "INCOMPLETE", "FAILED"}:
                        time.sleep(poll_s)
                return 0 if self.status == "COMPLETE" else 1
            finally:
                try:
                    (control / "coordinator.pid").unlink()
                except FileNotFoundError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--runtime-config")
    args = parser.parse_args()
    if args.runtime_config:
        GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    api = HfApi(token=Path("/content/.icgs_hf_token").read_text(encoding="utf-8").strip())
    control = CoordinatorControlPlane.open(args.run_config, api_factory=lambda: api)
    return control.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
