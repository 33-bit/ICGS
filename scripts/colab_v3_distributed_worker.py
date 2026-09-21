"""Credential-free persistent primary-v3 worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from icgs.data.collection.v3.distributed_contracts import GenerationJob, WorkerResult
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


def display_number(worker_id: str) -> int:
    if worker_id not in {f"{index:03d}" for index in range(200)}:
        raise ValueError("worker_id must be 000..199")
    # The launcher maps each slot to display 200 + worker_id.
    return 200 + int(worker_id)


def _file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path.relative_to(root))] = digest
    return hashes


def run_worker(worker_id: str, queue: FilesystemJobQueue, *, once: bool = False, idle_poll_s: float = 1.0) -> int:
    display_number(worker_id)
    while True:
        queue.write_heartbeat(worker_id, None)
        job = queue.claim(worker_id)
        if job is None:
            if once:
                return 0
            time.sleep(idle_poll_s)
            continue
        queue.write_heartbeat(worker_id, job.job_id)
        try:
            plan_path = Path(job.output_root) / "plans" / f"{job.job_id}.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(job.plan.as_dict(), indent=2) + "\n", encoding="utf-8")
            output_root = Path(job.output_root) / "worker-results" / job.job_id
            env = os.environ.copy()
            env.update({
                "ICGS_V3_ATTEMPT_JSON": str(plan_path),
                "ICGS_V3_WRITE_EPISODE": str(output_root),
                "PYTHONUNBUFFERED": "1",
            })
            command = [
                "/content/icgs-data-env/bin/python", "-B",
                "/content/ICGS/scripts/colab_v3_pilot_episodes_worker.py", job.program_id,
            ]
            process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=1200)
            log_path = output_root / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
            candidate = output_root / job.program_id
            episode_path = candidate / "episode.json"
            if episode_path.is_file():
                payload = json.loads(episode_path.read_text(encoding="utf-8"))
                outcome = str(payload.get("provenance", {}).get("outcome", "success"))
                result = WorkerResult(
                    job_id=job.job_id, attempt_id=job.attempt_id, episode_id=job.episode_id,
                    program_id=job.program_id, outcome=outcome, result_dir=str(candidate),
                    file_sha256=_file_hashes(candidate), timeline={
                        "actions": len(payload.get("transitions", [])),
                        "observations": len(payload.get("online_observations", [])),
                        "durations": len(payload.get("dt", [])),
                    },
                )
            else:
                sidecar = candidate / "quarantine.json"
                outcome = "simulator_crash" if process.returncode else "invalid_observation"
                result = WorkerResult(
                    job_id=job.job_id, attempt_id=job.attempt_id, episode_id=None,
                    program_id=job.program_id, outcome=outcome, result_dir=str(candidate),
                    file_sha256=_file_hashes(result_dir), timeline=None,
                )
            queue.publish_ready(worker_id, result)
        except Exception as exc:
            result_dir = Path(job.output_root) / "worker-results" / job.job_id / job.program_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "attempt.json").write_text(json.dumps({
                "attempt_id": job.attempt_id, "episode_id": None, "program_id": job.program_id,
                "outcome": "simulator_crash", "error": str(exc),
            }, indent=2) + "\n", encoding="utf-8")
            result = WorkerResult(
                job_id=job.job_id, attempt_id=job.attempt_id, episode_id=None,
                program_id=job.program_id, outcome="simulator_crash", result_dir=str(result_dir),
                file_sha256={}, timeline=None,
            )
            queue.publish_ready(worker_id, result)
        if once:
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    queue = FilesystemJobQueue(Path(args.run_root) / "queue")
    return run_worker(args.worker_id, queue, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
