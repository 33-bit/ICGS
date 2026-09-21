"""Single-owner coordinator for distributed primary-v3 generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi

from icgs.data.collection.v3.distributed_contracts import RunConfig
from icgs.data.collection.v3.distributed_planner import DistributedPlanner
from icgs.data.collection.v3.distributed_publication import HuggingFaceBatchPublisher
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue
from icgs.data.collection.v3.distributed_validation import ingest_validated_result, validate_closed_result


MAX_READY_PER_TICK = 200


def _inflight_jobs_from_queue(queue: FilesystemJobQueue):
    from icgs.data.collection.v3.distributed_contracts import GenerationJob

    paths = list((queue.root / "pending").glob("*.json"))
    paths.extend(
        path for path in (queue.root / "claimed").glob("*/*.json")
        if not path.name.endswith(".claim.json")
    )
    paths.extend(
        path / "job.json" for path in (queue.root / "ready").iterdir()
        if path.is_dir() and ".partial-" not in path.name
    )
    jobs = [GenerationJob.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    by_id = {job.job_id: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate in-flight jobs while resuming coordinator")
    return tuple(by_id[key] for key in sorted(by_id))


def _manifest_from_closed_queue(queue: FilesystemJobQueue, states=("ingested", "published")) -> dict:
    from icgs.data.collection.v3.distributed_contracts import GenerationJob, WorkerResult

    manifest = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
    for state in states:
        for directory in sorted((queue.root / state).iterdir()):
            if not directory.is_dir():
                continue
            job = GenerationJob.from_dict(json.loads((directory / "job.json").read_text(encoding="utf-8")))
            result = WorkerResult.from_dict(json.loads((directory / "result.json").read_text(encoding="utf-8")))
            manifest = ingest_validated_result(manifest, validate_closed_result(job, result))
    return manifest


class CoordinatorControlPlane:
    def __init__(self, run: RunConfig, queue: FilesystemJobQueue, planner: DistributedPlanner, publisher: HuggingFaceBatchPublisher, manifest=None):
        self.run = run
        self.queue = queue
        self.planner = planner
        self.publisher = publisher
        self.manifest = dict(manifest or {"manifest_version": 3, "episodes": [], "failure_attempts": []})
        self.status = "PREFLIGHT"

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
        publisher = HuggingFaceBatchPublisher(run, api, token, queue)
        publisher.remote_manifest = _manifest_from_closed_queue(queue, states=("published",))
        return cls(run, queue, planner, publisher, manifest)

    def _refill(self, target: int = 400) -> None:
        while self.queue.counts().pending + self.queue.counts().claimed < target:
            job = self.planner.next_job()
            if job is None:
                break
            self.queue.enqueue(job)

    def tick(self, *, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else now_s
        self.queue.recover_stale(now_s=now, stale_after_s=1800.0)
        for result in self.queue.iter_ready()[:MAX_READY_PER_TICK]:
            job_path = self.queue.root / "ready" / result.job_id / "job.json"
            from icgs.data.collection.v3.distributed_contracts import GenerationJob
            job = GenerationJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
            validated = validate_closed_result(job, result)
            self.manifest = ingest_validated_result(self.manifest, validated)
            self.planner.record_result(result, validated.provenance)
            self.queue.mark_ingested(result)
        self._refill()
        complete = self.planner.quota_complete()
        receipt = self.publisher.publish_due(
            now_s=now, force=complete, complete=complete,
            local_manifest=self.manifest,
        )
        self.status = "COMPLETE" if complete and not self.queue.counts().claimed else "RUNNING"
        control = self.queue.root.parent / "control"
        control.mkdir(parents=True, exist_ok=True)
        (control / "coordinator-heartbeat.json").write_text(json.dumps({
            "status": self.status, "timestamp_s": now,
            "queue": self.queue.counts().__dict__, "planner": self.planner.snapshot().as_dict(),
        }, indent=2) + "\n", encoding="utf-8")

    def run_forever(self, *, poll_s: float = 1.0) -> int:
        self.status = "RUNNING"
        while self.status not in {"COMPLETE", "INCOMPLETE", "FAILED"}:
            self.tick()
            time.sleep(poll_s)
        return 0 if self.status == "COMPLETE" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()
    api = HfApi(token=Path("/content/.icgs_hf_token").read_text(encoding="utf-8").strip())
    control = CoordinatorControlPlane.open(args.run_config, api_factory=lambda: api)
    return control.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
