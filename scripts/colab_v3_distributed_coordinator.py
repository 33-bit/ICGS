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


class CoordinatorControlPlane:
    def __init__(self, run: RunConfig, queue: FilesystemJobQueue, planner: DistributedPlanner, publisher: HuggingFaceBatchPublisher):
        self.run = run
        self.queue = queue
        self.planner = planner
        self.publisher = publisher
        self.manifest = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
        self.status = "PREFLIGHT"

    @classmethod
    def open(cls, run_config_path: str | Path, *, api_factory):
        payload = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
        run = RunConfig.from_dict(payload["run"])
        queue = FilesystemJobQueue(run.run_root + "/queue")
        approved = json.loads(Path(payload["approved_manifest"]).read_text(encoding="utf-8"))
        rows = {row["program_id"]: row for row in approved["catalog"]}
        planner = DistributedPlanner.from_manifest(run, rows, payload.get("manifest", {"episodes": [], "failure_attempts": []}))
        api = api_factory()
        token = Path("/content/.icgs_hf_token").read_text(encoding="utf-8").strip()
        publisher = HuggingFaceBatchPublisher(run, api, token, queue)
        return cls(run, queue, planner, publisher)

    def _refill(self, target: int = 400) -> None:
        while self.queue.counts().pending + self.queue.counts().claimed < target:
            job = self.planner.next_job()
            if job is None:
                break
            self.queue.enqueue(job)

    def tick(self, *, now_s: float | None = None) -> None:
        now = time.time() if now_s is None else now_s
        self.queue.recover_stale(now_s=now, stale_after_s=1800.0)
        for result in self.queue.iter_ready():
            job_path = self.queue.root / "ready" / result.job_id / "job.json"
            from icgs.data.collection.v3.distributed_contracts import GenerationJob
            job = GenerationJob.from_dict(json.loads(job_path.read_text(encoding="utf-8")))
            validated = validate_closed_result(job, result)
            self.manifest = ingest_validated_result(self.manifest, validated)
            self.planner.record_result(result, validated.provenance)
            self.queue.mark_ingested(result)
        self._refill()
        complete = self.planner.quota_complete()
        receipt = self.publisher.publish_due(now_s=now, force=complete, complete=complete)
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
