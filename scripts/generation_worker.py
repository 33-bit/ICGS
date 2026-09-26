"""Credential-free persistent generation worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import uuid
from typing import Any
from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    GenerationRuntimeConfig,
    WorkerResult,
)
from icgs.data.collection.generation.distributed_queue import (
    FilesystemJobQueue,
    GenerationSafetyStop,
)
from icgs.data.collection.generation.diversity import train_subset_for_sample
from icgs.data.collection.generation.episode_archive import EpisodeArchiveWriter


def display_number(worker_id: str, config: GenerationRuntimeConfig) -> int:
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")
    width = max(3, len(str(config.run.worker_count - 1)))
    if not worker_id.isdecimal():
        raise ValueError("worker_id must be a zero-padded worker index")
    index = int(worker_id)
    if index >= config.run.worker_count or worker_id != f"{index:0{width}d}":
        raise ValueError(f"worker_id must identify a worker in 0..{config.run.worker_count - 1}")
    return config.machine.display_base + index


def _file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[str(path.relative_to(root))] = digest
    return hashes


def _attempt_output_root(job: GenerationJob) -> Path:
    """Return a retry-fenced staging directory for one immutable job."""
    return (
        Path(job.output_root)
        / "worker-results"
        / job.job_id
        / f"retry-{job.retry_generation}"
    )


def _credential_free_environment(config: GenerationRuntimeConfig) -> dict[str, str]:
    environment = config.resolved_environment()
    for key in (
        "ICGS_HF_TOKEN_PATH",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HF_ACCESS_TOKEN",
    ):
        environment.pop(key, None)
    return environment


def _archive_result_payload(
    candidate: Path,
    config: GenerationRuntimeConfig,
    *,
    job: GenerationJob | None = None,
) -> dict | None:
    """Return one canonical archive manifest, with no legacy fallback in archive mode."""
    if config.archive_profile is None:
        return None
    manifest_paths = [
        path for path in (candidate / "episode.manifest.json", candidate / "attempt.manifest.json")
        if path.is_file()
    ]
    if (candidate / "episode.json").exists() or (candidate / "attempt.json").exists():
        raise ValueError("archive profile result must not contain legacy episode/attempt JSON")
    if len(manifest_paths) > 1:
        raise ValueError("archive profile result must contain exactly one archive manifest")
    if not manifest_paths:
        return None
    payload = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    if payload.get("archive_format_id") != config.archive_profile.archive_format_id:
        raise ValueError("archive result format identity disagrees with runtime profile")
    if payload.get("episode_schema_version") != config.archive_profile.episode_schema_version:
        raise ValueError("archive result schema identity disagrees with runtime profile")
    if payload.get("dataset_identity") != config.archive_profile.dataset_identity:
        raise ValueError("archive result dataset identity disagrees with runtime profile")
    expected_kind = "episode" if manifest_paths[0].name == "episode.manifest.json" else "attempt"
    if payload.get("archive_kind") != expected_kind:
        raise ValueError("archive manifest kind disagrees with its filename")
    outcome = payload.get("outcome")
    valid_outcomes = {"success", "valid_failure"} if expected_kind == "episode" else {"simulator_crash", "invalid_observation"}
    if outcome not in valid_outcomes:
        raise ValueError("archive manifest outcome disagrees with its kind")
    if job is not None:
        if payload.get("attempt_id") != job.attempt_id or payload.get("program_id") != job.program_id:
            raise ValueError("archive manifest identity disagrees with the claimed job")
        expected_episode_id = job.episode_id if expected_kind == "episode" else None
        if payload.get("episode_id") != expected_episode_id:
            raise ValueError("archive manifest episode identity disagrees with the claimed job")
    return payload


def _write_archive_worker_attempt(
    candidate: Path,
    job: GenerationJob,
    config: GenerationRuntimeConfig,
    *,
    outcome: str,
    error: str,
    stdout: str = "",
    stderr: str = "",
    traceback_text: str = "",
    exit_code: int | None = None,
    timeout: bool = False,
) -> None:
    if candidate.exists():
        if any(candidate.iterdir()):
            raise FileExistsError(
                "archive worker attempt candidate must be classified before overwrite"
            )
        candidate.rmdir()
    attempt = {
        "attempt_id": job.attempt_id,
        "episode_id": None,
        "program_id": job.program_id,
        "split": "dev" if job.plan.split == "development" else job.plan.split,
        "subset": (
            job.plan.randomization.get("train_subset")
            or train_subset_for_sample(job.plan.randomization)
            if job.plan.split == "train" else None
        ),
        "outcome": outcome,
        "error": error[:8192],
        "valid_observation_until": None,
    }
    debug = {
        "source_run_id": job.run_id,
        "code_revision": job.code_revision,
        "preprocessing_identity": "generation_worker_failure_v1",
        "job_identity": {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "attempt_id": job.attempt_id,
            "episode_id": job.episode_id,
            "program_id": job.program_id,
            "code_revision": job.code_revision,
            "manifest_sha256": job.manifest_sha256,
            "retry_generation": job.retry_generation,
            "plan": job.plan.as_dict(),
        },
        "exit_code": exit_code,
        "timeout": timeout,
        "stdout": (stdout or "")[-8192:],
        "stderr": (stderr or "")[-8192:],
        "traceback": (traceback_text or "")[-8192:],
    }
    EpisodeArchiveWriter(config.archive_profile).write_attempt(
        attempt,
        prefix_arrays={},
        debug_metadata=debug,
        output_dir=candidate,
    )


def _hash_file_stream(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _orphan_archive_scratch_inventory(
    attempt_root: Path,
    candidate: Path,
    *,
    program_id: str,
    run_root: Path,
    archive_closed: bool,
) -> list[dict[str, Any]]:
    """Inventory unclosed writer scratch without changing or deleting any bytes."""
    roots = []
    if attempt_root.is_dir() and not attempt_root.is_symlink():
        roots.extend(
            child for child in sorted(attempt_root.iterdir())
            if child.name.startswith((
                f".{program_id}.archive-spool-",
                f".{program_id}.partial-",
                f".{program_id}.archive-write-in-progress-",
            ))
        )
    if not archive_closed and candidate.is_dir() and not candidate.is_symlink():
        if any(candidate.iterdir()):
            roots.append(candidate)

    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        paths = [root]
        if root.is_dir() and not root.is_symlink():
            paths.extend(sorted(root.rglob("*")))
        for path in paths:
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
                try:
                    byte_count, digest = _hash_file_stream(path)
                except OSError:
                    entries.append({"path": relative, "bytes": int(info.st_size), "kind": "unreadable"})
                else:
                    entries.append({
                        "path": relative,
                        "bytes": byte_count,
                        "sha256": digest,
                        "kind": "file",
                    })
            elif path.is_dir():
                entries.append({"path": relative, "bytes": 0, "kind": "directory"})
            else:
                entries.append({"path": relative, "bytes": int(info.st_size), "kind": "other"})
    return entries


def _stop_run_for_orphan_scratch(
    queue: FilesystemJobQueue,
    job: GenerationJob,
    attempt_root: Path,
    candidate: Path,
    *,
    archive_closed: bool,
    timeout: bool,
    on_marker_failure=None,
) -> None:
    run_root = queue.root.parent.resolve()
    entries = _orphan_archive_scratch_inventory(
        attempt_root,
        candidate,
        program_id=job.program_id,
        run_root=run_root,
        archive_closed=archive_closed,
    )
    if not entries:
        return
    receipt = {
        "schema_version": "icgs_generation_safety_stop_v1",
        "run_id": job.run_id,
        "job_id": job.job_id,
        "attempt_id": job.attempt_id,
        "retry_generation": job.retry_generation,
        "program_id": job.program_id,
        "reason": (
            "orphan_archive_scratch_after_timeout"
            if timeout else "orphan_archive_scratch_after_worker_error"
        ),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preserved_bytes": sum(int(item["bytes"]) for item in entries if item["kind"] == "file"),
        "preserved_files": entries,
        "recovery_action": (
            "Keep every listed path unchanged. Stop this run, establish whether each byte was committed "
            "to the pinned Hugging Face revision using publication receipts and hashes, and recover or "
            "archive any unmatched bytes before starting a disjoint run. Do not delete these paths "
            "until their HF fate is known."
        ),
    }
    try:
        queue.trip_safety_stop(receipt)
    except Exception as error:
        if on_marker_failure is not None:
            try:
                on_marker_failure()
            except Exception:
                pass
        raise GenerationSafetyStop(
            "generation safety stop: orphan scratch was preserved but the run receipt could not "
            f"be written at {queue.safety_stop_path}; recovery scan is required before any new claim"
        ) from error
    raise GenerationSafetyStop(
        f"generation safety stop: preserved {receipt['preserved_bytes']} orphan archive bytes; "
        f"inspect {queue.safety_stop_path} before resuming"
    )


class SimulatorSlotPool:
    """Cross-process semaphore for memory-heavy simulator launches.

    POSIX flock releases a slot automatically if a worker is killed, so
    recovery never depends on stale lease metadata.
    """

    def __init__(self, run_root: str | Path, config: GenerationRuntimeConfig):
        if not isinstance(config, GenerationRuntimeConfig):
            raise TypeError("config must be a GenerationRuntimeConfig")
        self.slot_count = config.machine.simulator_slots
        self.root = Path(run_root) / "control" / "simulator-slots"
        if config.run.distribution_mode == "shared_filesystem":
            self.root /= config.machine.host_id
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


def _run_generation_process(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_s: int,
    heartbeat,
    heartbeat_interval_s: float,
    timeout_grace_s: float = 0.0,
    stop_check=None,
    stop_check_interval_s: float = 1.0,
):
    """Run one simulator process while renewing the owning worker lease."""
    process = subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    next_heartbeat = started
    next_stop_check = started
    if stop_check is not None and stop_check_interval_s <= 0:
        raise ValueError("stop_check_interval_s must be positive")

    def terminate_bounded() -> None:
        if process.poll() is not None:
            return
        if timeout_grace_s > 0:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout_grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.kill()
            process.wait()

    try:
        while True:
            now = time.monotonic()
            if stop_check is not None and now >= next_stop_check:
                stop_check()
                next_stop_check = now + stop_check_interval_s
            if now >= next_heartbeat:
                heartbeat()
                next_heartbeat = now + heartbeat_interval_s
            returncode = process.poll()
            if returncode is not None:
                heartbeat()
                stdout, stderr = process.communicate()
                return subprocess.CompletedProcess(command, returncode, stdout, stderr)
            if now - started >= timeout_s:
                terminate_bounded()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command,
                    timeout_s,
                    output=stdout,
                    stderr=stderr,
                )
            time.sleep(min(heartbeat_interval_s, max(0.01, timeout_s - (now - started))))
    except BaseException:
        if process.poll() is None:
            terminate_bounded()
            process.communicate()
        raise


def run_worker(
    worker_id: str,
    queue: FilesystemJobQueue,
    *,
    config: GenerationRuntimeConfig,
    approved_manifest: str,
    once: bool = False,
    idle_poll_s: float = 1.0,
    runner=None,
    worker_instance_id: str | None = None,
) -> int:
    display_number(worker_id, config)
    if worker_id not in config.machine.worker_ids:
        raise ValueError(f"worker_id is outside host worker scope: {worker_id}")
    worker_instance_id = worker_instance_id or (
        f"{config.machine.host_id}:{worker_id}:{os.getpid()}:{uuid.uuid4().hex}"
    )
    lease_s = float(config.machine.worker_timeout_s) + 600.0
    queue.register_worker(
        worker_id,
        config.machine.host_id,
        worker_instance_id,
        lease_s=lease_s,
    )
    while True:
        queue.write_heartbeat(
            worker_id,
            None,
            host_id=config.machine.host_id,
            worker_instance_id=worker_instance_id,
            lease_s=lease_s,
        )
        job = queue.claim(
            worker_id,
            worker_instance_id=worker_instance_id,
            host_id=config.machine.host_id,
            lease_s=lease_s,
        )
        if job is None:
            if once:
                return 0
            time.sleep(idle_poll_s)
            continue
        queue.write_heartbeat(
            worker_id,
            job.job_id,
            host_id=config.machine.host_id,
            worker_instance_id=worker_instance_id,
            lease_s=lease_s,
        )

        def release_claim_after_stop_write_failure() -> None:
            queue.write_heartbeat(
                worker_id,
                None,
                host_id=config.machine.host_id,
                worker_instance_id=worker_instance_id,
                lease_s=lease_s,
            )

        try:
            plan_path = Path(job.output_root) / "plans" / f"{job.job_id}.retry-{job.retry_generation}.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(job.plan.as_dict(), indent=2) + "\n", encoding="utf-8")
            approved = json.loads(Path(approved_manifest).read_text(encoding="utf-8"))
            binding = next(row for row in approved["catalog"] if row["program_id"] == job.program_id)
            binding_path = Path(job.output_root) / "plans" / f"{job.job_id}.retry-{job.retry_generation}.binding.json"
            binding_path.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
            output_root = _attempt_output_root(job)
            env = _credential_free_environment(config)
            env.update({
                "ICGS_GENERATION_ATTEMPT_JSON": str(plan_path),
                "ICGS_GENERATION_WRITE_EPISODE": str(output_root),
                "ICGS_GENERATION_BINDING_JSON": str(binding_path),
                "ICGS_GENERATION_APPROVED_MANIFEST": str(approved_manifest),
                "ICGS_GENERATION_RUN_ID": job.run_id,
                "ICGS_GENERATION_CODE_REVISION": job.code_revision,
                "PYTHONUNBUFFERED": "1",
            })
            if config.archive_profile is not None:
                env["ICGS_GENERATION_JOB_IDENTITY"] = json.dumps({
                    "job_id": job.job_id,
                    "run_id": job.run_id,
                    "attempt_id": job.attempt_id,
                    "episode_id": job.episode_id,
                    "program_id": job.program_id,
                    "code_revision": job.code_revision,
                    "manifest_sha256": job.manifest_sha256,
                    "retry_generation": job.retry_generation,
                    "plan": job.plan.as_dict(),
                }, sort_keys=True, separators=(",", ":"))
            command = [
                config.machine.python_executable,
                "-B",
                str(Path(config.machine.repo_root) / "scripts" / "generation_episode_worker.py"),
                job.program_id,
            ]
            with SimulatorSlotPool(Path(job.output_root).parent, config):
                if runner is None:
                    heartbeat_job_id = job.job_id

                    def heartbeat_owned_job() -> None:
                        queue.write_heartbeat(
                            worker_id,
                            heartbeat_job_id,
                            host_id=config.machine.host_id,
                            worker_instance_id=worker_instance_id,
                            lease_s=lease_s,
                        )

                    process = _run_generation_process(
                        command,
                        env=env,
                        timeout_s=config.machine.worker_timeout_s,
                        heartbeat=heartbeat_owned_job,
                        heartbeat_interval_s=max(
                            1.0,
                            min(30.0, lease_s / 3.0),
                        ),
                        timeout_grace_s=10.0 if config.archive_profile is not None else 0.0,
                        stop_check=(queue.assert_run_active if config.archive_profile is not None else None),
                        stop_check_interval_s=1.0,
                    )
                else:
                    process = runner(
                        command,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=config.machine.worker_timeout_s,
                    )
            log_path = output_root / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
            candidate = output_root / job.program_id
            archive_payload = _archive_result_payload(candidate, config, job=job)
            if config.archive_profile is not None:
                _stop_run_for_orphan_scratch(
                    queue,
                    job,
                    output_root,
                    candidate,
                    archive_closed=archive_payload is not None,
                    timeout=False,
                    on_marker_failure=release_claim_after_stop_write_failure,
                )
                if archive_payload is None:
                    outcome = "simulator_crash" if process.returncode else "invalid_observation"
                    _write_archive_worker_attempt(
                        candidate,
                        job,
                        config,
                        outcome=outcome,
                        error="worker did not produce a closed archive",
                        stdout=process.stdout or "",
                        stderr=process.stderr or "",
                        exit_code=process.returncode,
                    )
                    archive_payload = _archive_result_payload(candidate, config, job=job)
                else:
                    outcome = str(archive_payload.get("outcome"))
                timeline_payload = archive_payload.get("timeline") or {}
                result = WorkerResult(
                    job_id=job.job_id,
                    attempt_id=job.attempt_id,
                    episode_id=job.episode_id if outcome in {"success", "valid_failure"} else None,
                    program_id=job.program_id, outcome=outcome, result_dir=str(candidate),
                    file_sha256=_file_hashes(candidate),
                    timeline=(
                        {
                            "actions": int(timeline_payload.get("transitions", 0)),
                            "observations": int(timeline_payload.get("observations", 0)),
                            "durations": int(timeline_payload.get("transitions", 0)),
                        }
                        if outcome in {"success", "valid_failure"} else None
                    ),
                )
            else:
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
                    queue.publish_ready(worker_id, result, worker_instance_id=worker_instance_id)
                    if once:
                        return 0
                    continue
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
            queue.publish_ready(worker_id, result, worker_instance_id=worker_instance_id)
        except Exception as exc:
            result_dir = _attempt_output_root(job) / job.program_id
            if config.archive_profile is not None:
                archive_payload = None
                try:
                    archive_payload = _archive_result_payload(result_dir, config, job=job)
                except Exception:
                    pass
                _stop_run_for_orphan_scratch(
                    queue,
                    job,
                    result_dir.parent,
                    result_dir,
                    archive_closed=archive_payload is not None,
                    timeout=isinstance(exc, subprocess.TimeoutExpired),
                    on_marker_failure=release_claim_after_stop_write_failure,
                )
                try:
                    queue.assert_run_active()
                except GenerationSafetyStop:
                    raise
                if archive_payload is not None and archive_payload.get("archive_kind") in {"episode", "attempt"}:
                    outcome = str(archive_payload.get("outcome"))
                    if archive_payload.get("archive_kind") == "episode" and outcome in {"success", "valid_failure"}:
                        timeline_payload = archive_payload.get("timeline") or {}
                        queue.publish_ready(worker_id, WorkerResult(
                            job_id=job.job_id,
                            attempt_id=job.attempt_id,
                            episode_id=job.episode_id,
                            program_id=job.program_id,
                            outcome=outcome,
                            result_dir=str(result_dir),
                            file_sha256=_file_hashes(result_dir),
                            timeline={
                                "actions": int(timeline_payload.get("transitions", 0)),
                                "observations": int(timeline_payload.get("observations", 0)),
                                "durations": int(timeline_payload.get("transitions", 0)),
                            },
                        ), worker_instance_id=worker_instance_id)
                        if once:
                            return 0
                        continue
                    if (
                        archive_payload.get("archive_kind") == "attempt"
                        and outcome in {"simulator_crash", "invalid_observation"}
                    ):
                        queue.publish_ready(worker_id, WorkerResult(
                            job_id=job.job_id,
                            attempt_id=job.attempt_id,
                            episode_id=None,
                            program_id=job.program_id,
                            outcome=outcome,
                            result_dir=str(result_dir),
                            file_sha256=_file_hashes(result_dir),
                            timeline=None,
                        ), worker_instance_id=worker_instance_id)
                        if once:
                            return 0
                        continue
                _write_archive_worker_attempt(
                    result_dir,
                    job,
                    config,
                    outcome="simulator_crash",
                    error=str(exc),
                    stdout=getattr(exc, "output", "") or "",
                    stderr=getattr(exc, "stderr", "") or "",
                    traceback_text=traceback.format_exc(),
                    timeout=isinstance(exc, subprocess.TimeoutExpired),
                )
                queue.publish_ready(worker_id, WorkerResult(
                    job_id=job.job_id,
                    attempt_id=job.attempt_id,
                    episode_id=None,
                    program_id=job.program_id,
                    outcome="simulator_crash",
                    result_dir=str(result_dir),
                    file_sha256=_file_hashes(result_dir),
                    timeline=None,
                ), worker_instance_id=worker_instance_id)
                if once:
                    return 0
                continue
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
                        ), worker_instance_id=worker_instance_id)
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
            queue.publish_ready(worker_id, result, worker_instance_id=worker_instance_id)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--host-id")
    parser.add_argument("--worker-instance-id")
    args = parser.parse_args(argv)
    config = GenerationRuntimeConfig.from_file(args.runtime_config)
    if args.host_id is not None and args.host_id != config.machine.host_id:
        raise ValueError("--host-id does not match runtime config host_id")
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    return run_worker(
        args.worker_id,
        queue,
        config=config,
        approved_manifest=args.approved_manifest,
        once=args.once,
        worker_instance_id=args.worker_instance_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
