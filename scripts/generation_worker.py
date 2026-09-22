"""Credential-free persistent generation worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from icgs.data.collection.generation.distributed_contracts import GenerationJob, WorkerResult
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue


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


class SimulatorSlotPool:
    """Cross-process semaphore for memory-heavy simulator launches.

    The worker count remains 200, but a VM can configure a smaller number of
    simultaneous CoppeliaSim processes with ``ICGS_SIMULATOR_SLOTS``. POSIX
    flock releases a slot automatically if a worker is killed, so recovery
    never depends on stale lease metadata.
    """

    def __init__(self, run_root: str | Path):
        try:
            slot_count = int(os.environ.get("ICGS_SIMULATOR_SLOTS", "200"))
        except ValueError as exc:
            raise ValueError("ICGS_SIMULATOR_SLOTS must be an integer") from exc
        if slot_count <= 0:
            raise ValueError("ICGS_SIMULATOR_SLOTS must be positive")
        self.slot_count = slot_count
        self.root = Path(run_root) / "control" / "simulator-slots"
        self.root.mkdir(parents=True, exist_ok=True)
        self._stream = None
        self.acquired_slot: Path | None = None

    def __enter__(self):
        while True:
            for index in range(self.slot_count):
                path = self.root / f"slot-{index:03d}.lock"
                stream = path.open("a+")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                self._stream = stream
                self.acquired_slot = path
                return self
            time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None
            self.acquired_slot = None
        return False


def run_worker(
    worker_id: str,
    queue: FilesystemJobQueue,
    *,
    approved_manifest: str = "/content/ICGS/artifacts/composition/approved_composition_manifest.json",
    once: bool = False,
    idle_poll_s: float = 1.0,
) -> int:
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
            approved = json.loads(Path(approved_manifest).read_text(encoding="utf-8"))
            binding = next(row for row in approved["catalog"] if row["program_id"] == job.program_id)
            binding_path = Path(job.output_root) / "plans" / f"{job.job_id}.binding.json"
            binding_path.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
            output_root = Path(job.output_root) / "worker-results" / job.job_id
            env = os.environ.copy()
            env.update({
                "ICGS_GENERATION_ATTEMPT_JSON": str(plan_path),
                "ICGS_GENERATION_WRITE_EPISODE": str(output_root),
                "ICGS_GENERATION_BINDING_JSON": str(binding_path),
                "ICGS_GENERATION_APPROVED_MANIFEST": str(approved_manifest),
                "PYTHONUNBUFFERED": "1",
            })
            command = [
                "/content/icgs-data-env/bin/python", "-B",
                "/content/ICGS/scripts/generation_episode_worker.py", job.program_id,
            ]
            with SimulatorSlotPool(Path(job.output_root).parent):
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
                attempt_path = candidate / "attempt.json"
                if attempt_path.is_file():
                    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                    outcome = str(attempt.get("outcome", "simulator_crash"))
                else:
                    outcome = "simulator_crash" if process.returncode else "invalid_observation"
                    candidate.mkdir(parents=True, exist_ok=True)
                    attempt_path.write_text(json.dumps({
                        "attempt_id": job.attempt_id, "episode_id": None, "program_id": job.program_id,
                        "outcome": outcome, "error": "worker did not produce a closed attempt",
                    }, indent=2) + "\n", encoding="utf-8")
                    file_inventory = {
                        "attempt.json": {
                            "sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                            "bytes": attempt_path.stat().st_size,
                        }
                    }
                    (candidate / "artifact_manifest.json").write_text(json.dumps({
                        "attempt_id": job.attempt_id, "episode_id": None,
                        "program_id": job.program_id, "outcome": outcome,
                        "files": file_inventory,
                    }, indent=2) + "\n", encoding="utf-8")
                result = WorkerResult(
                    job_id=job.job_id, attempt_id=job.attempt_id, episode_id=None,
                    program_id=job.program_id, outcome=outcome, result_dir=str(candidate),
                    file_sha256=_file_hashes(candidate), timeline=None,
                )
            queue.publish_ready(worker_id, result)
        except Exception as exc:
            result_dir = Path(job.output_root) / "worker-results" / job.job_id / job.program_id
            episode_path = result_dir / "episode.json"
            artifact_manifest_path = result_dir / "artifact_manifest.json"
            # A subprocess timeout can happen after the pilot has atomically
            # closed a complete episode. Preserve that valid data instead of
            # overlaying an attempt record onto the same directory.
            if episode_path.is_file() and artifact_manifest_path.is_file():
                try:
                    payload = json.loads(episode_path.read_text(encoding="utf-8"))
                    outcome = str(payload.get("provenance", {}).get("outcome", ""))
                    if outcome in {"success", "valid_failure"}:
                        queue.publish_ready(worker_id, WorkerResult(
                            job_id=job.job_id,
                            attempt_id=job.attempt_id,
                            episode_id=job.episode_id,
                            program_id=job.program_id,
                            outcome=outcome,
                            result_dir=str(result_dir),
                            file_sha256=_file_hashes(result_dir),
                            timeline={
                                "actions": len(payload.get("transitions", [])),
                                "observations": len(payload.get("online_observations", [])),
                                "durations": len(payload.get("dt", [])),
                            },
                        ))
                        if once:
                            return 0
                        continue
                except Exception:
                    # Fall through to a closed crash attempt if the candidate
                    # cannot satisfy the episode contract.
                    pass
            # No closed episode survived. Remove the stale candidate so the
            # crash attempt has an immutable, self-consistent inventory.
            if result_dir.exists():
                shutil.rmtree(result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "attempt.json").write_text(json.dumps({
                "attempt_id": job.attempt_id, "episode_id": None, "program_id": job.program_id,
                "outcome": "simulator_crash", "error": str(exc),
            }, indent=2) + "\n", encoding="utf-8")
            result = WorkerResult(
                job_id=job.job_id, attempt_id=job.attempt_id, episode_id=None,
                program_id=job.program_id, outcome="simulator_crash", result_dir=str(result_dir),
                file_sha256=_file_hashes(result_dir), timeline=None,
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
    return run_worker(args.worker_id, queue, approved_manifest=args.approved_manifest, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
