"""Single-owner coordinator for distributed generation generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
import time
from typing import Any, Literal, Mapping

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    CoordinatorHeartbeat,
    GenerationJob,
    GenerationRuntimeConfig,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.batch import attempt_from_dict
from icgs.data.collection.generation.distributed_planner import DistributedPlanner
from icgs.data.collection.generation.distributed_publication import (
    HuggingFaceBatchPublisher,
    PublicationReceipt,
)
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue, MalformedReadyResult
from icgs.data.collection.generation.distributed_validation import (
    ingest_validated_result,
    validate_closed_result,
)
from icgs.data.collection.generation.distributed_validation import ValidationFailure
from icgs.data.collection.generation.diversity import scene_signature, train_subset_for_sample
from icgs.data.collection.generation.steps import get_generation_program
from icgs.data.collection.generation.perturbations import applicable_perturbations
from icgs.data.collection.generation.episode_archive import validate_archive_manifest


MAX_READY_PER_TICK = 100


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _credential_path(
    run_config_path: str | Path,
    payload: dict,
    token_path: str | Path | None = None,
) -> Path:
    configured = token_path or payload.get("hf_token_path") or os.environ.get("ICGS_HF_TOKEN_PATH")
    if not isinstance(configured, (str, Path)) or not str(configured).strip():
        raise ValueError("an HF credential path must be configured")
    path = Path(configured)
    if not path.is_absolute():
        path = Path(run_config_path).parent / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"HF credential path must be a regular file: {path}")
    return path


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


def _manifest_from_closed_queue(
    queue: FilesystemJobQueue,
    states=("ingested", "published"),
    *,
    archive_profile=None,
) -> dict:
    from icgs.data.collection.generation.distributed_contracts import GenerationJob, WorkerResult

    manifest = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
    source_run_ids: set[str] = set()
    for state in states:
        for directory in sorted((queue.root / state).iterdir()):
            if not stat.S_ISDIR(directory.lstat().st_mode):
                continue
            for name in ("job.json", "result.json"):
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"closed queue {name} must be a regular file: {path}")
            job = GenerationJob.from_dict(json.loads((directory / "job.json").read_text(encoding="utf-8")))
            result = WorkerResult.from_dict(json.loads((directory / "result.json").read_text(encoding="utf-8")))
            receipt_path = directory / "publication_receipt.json"
            receipt_marker = receipt_path.exists() or receipt_path.is_symlink()
            if (
                state == "ingested"
                and archive_profile is not None
                and receipt_marker
            ):
                row = queue._validate_saved_receipt_only_record(
                    directory,
                    job.job_id,
                    archive_profile=archive_profile,
                    allow_remaining_payload=True,
                )
                collection, identity_key, identity = (
                    ("episodes", "episode_id", result.episode_id)
                    if result.episode_id is not None
                    else ("failure_attempts", "attempt_id", result.attempt_id)
                )
                if row.get(identity_key) != identity:
                    raise ValueError("interrupted per-job receipt manifest row identity mismatch")
                rows = manifest[collection]
                previous = next((item for item in rows if item.get(identity_key) == identity), None)
                if previous is not None and previous != row:
                    raise ValueError(f"immutable published manifest row conflict: {identity}")
                if previous is None:
                    rows.append(row)
                published = queue.mark_published(job.job_id, retention="receipt_only")
                queue._validate_saved_receipt_only_record(
                    published,
                    job.job_id,
                    archive_profile=archive_profile,
                )
                source_run_ids.add(job.run_id)
                continue
            if state == "published" and archive_profile is not None and receipt_marker:
                row = queue._validate_saved_receipt_only_record(
                    directory,
                    job.job_id,
                    archive_profile=archive_profile,
                )
                collection, identity_key, identity = (
                    ("episodes", "episode_id", result.episode_id)
                    if result.episode_id is not None
                    else ("failure_attempts", "attempt_id", result.attempt_id)
                )
                if (
                    row.get(identity_key) != identity
                ):
                    raise ValueError("published per-job receipt manifest row identity mismatch")
                rows = manifest[collection]
                previous = next((item for item in rows if item.get(identity_key) == identity), None)
                if previous is not None and previous != row:
                    raise ValueError(f"immutable published manifest row conflict: {identity}")
                if previous is None:
                    rows.append(row)
                source_run_ids.add(job.run_id)
                continue
            validated = (
                validate_closed_result(job, result, archive_profile=archive_profile)
                if archive_profile is not None
                else validate_closed_result(job, result)
            )
            manifest = ingest_validated_result(manifest, validated)
            source_run_ids.add(job.run_id)
    if archive_profile is not None:
        manifest.update({
            "source_run_ids": sorted(source_run_ids),
            "dataset_identity": archive_profile.dataset_identity,
            "archive_format_id": archive_profile.archive_format_id,
            "episode_schema_version": archive_profile.episode_schema_version,
            "archive_profile": archive_profile.as_dict(),
            "view_status": "PROVISIONAL",
            "total_episodes": len(manifest["episodes"]),
            "total_failure_attempts": len(manifest["failure_attempts"]),
        })
    return manifest


def _merge_manifests(remote: Mapping[str, object], local: Mapping[str, object]) -> dict:
    """Merge immutable remote and surviving local rows for a resumed planner."""
    merged = {
        "manifest_version": local.get("manifest_version", remote.get("manifest_version", 3)),
        "episodes": [],
        "failure_attempts": [],
    }
    all_identities: set[str] = set()
    for key in ("episodes", "failure_attempts"):
        by_id: dict[str, dict] = {}
        for source in (remote, local):
            for row in list(source.get(key) or []):
                if not isinstance(row, Mapping):
                    raise ValueError(f"remote resume {key} row must be an object")
                identity = str(row.get("episode_id") or row.get("attempt_id") or "")
                if not identity:
                    raise ValueError(f"remote resume {key} row has no immutable identity")
                if identity in all_identities and identity not in by_id:
                    raise ValueError(f"remote resume duplicate immutable identity: {identity}")
                previous = by_id.get(identity)
                current = dict(row)
                if previous is not None and previous != current:
                    raise ValueError(f"remote resume immutable conflict: {identity}")
                by_id[identity] = current
                all_identities.add(identity)
        merged[key] = [by_id[key] for key in sorted(by_id)]
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
    if source_run_ids:
        merged["source_run_ids"] = sorted(source_run_ids)
    for key in (
        "dataset_identity",
        "archive_format_id",
        "episode_schema_version",
        "archive_profile",
        "view_status",
    ):
        values = [source[key] for source in (remote, local) if key in source]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"remote resume immutable conflict: {key}")
        if values:
            merged[key] = values[0]
    return merged


def _validate_runtime_run_binding(run: RunConfig, runtime: GenerationRuntimeConfig) -> None:
    runtime_values = {
        "run_id": runtime.run.run_id,
        "run_root": runtime.run.run_root,
        "worker_count": runtime.run.worker_count,
        "publish_interval_s": runtime.run.publish_interval_s,
        "hf_repo": runtime.run.hf_repo or "33bit/icgs",
        "hf_subfolder": runtime.run.hf_subfolder or "generation",
        "publication_enabled": runtime.run.publication_enabled,
        "validation_mode": runtime.run.validation_mode,
        "distribution_mode": runtime.run.distribution_mode,
        "resume_from_hf": runtime.run.resume_from_hf,
    }
    for name, expected in runtime_values.items():
        if getattr(run, name) != expected:
            raise ValueError(f"runtime config {name} does not match run config")


def _snapshot_revision(path: str | Path) -> str | None:
    parts = Path(path).parts
    try:
        index = parts.index("snapshots")
        value = parts[index + 1]
    except (ValueError, IndexError):
        return None
    if (
        len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def _require_pinned_revision(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("archive resume requires a pinned lowercase HF commit OID")
    return value


def _download_pinned_file(
    downloader, run: RunConfig, token: str, revision: str, filename: str,
    scratch: Path, expected_sha256: str | None = None,
) -> Path:
    _require_pinned_revision(revision)
    remote = Path(downloader(
        repo_id=run.hf_repo, repo_type="dataset", filename=filename,
        token=token, force_download=True, revision=revision, cache_dir=str(scratch),
    ))
    if not remote.is_file() or not remote.resolve().is_relative_to(scratch.resolve()):
        raise ValueError(f"remote archive file is missing or outside owned scratch: {filename}")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with remote.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"remote archive hash mismatch: {filename}")
    return remote


def _download_archive_control_file(
    downloader, run: RunConfig, token: str, filename: str,
    scratch: Path, revision: str | None = None,
) -> Path:
    remote = Path(downloader(
        repo_id=run.hf_repo, repo_type="dataset", filename=filename,
        token=token, force_download=True, revision=revision, cache_dir=str(scratch),
    ))
    remote_lexical = Path(os.path.abspath(remote))
    scratch_lexical = Path(os.path.abspath(scratch))
    if (
        not remote_lexical.is_relative_to(scratch_lexical)
        or not remote.resolve().is_relative_to(scratch.resolve())
    ):
        raise ValueError(f"remote archive control file is outside owned scratch: {filename}")
    if not remote.is_file():
        raise ValueError(f"remote archive control file is not a regular file: {filename}")
    return remote


def _verify_archive_resume_files(
    manifest: Mapping[str, Any], run: RunConfig, profile: ArchiveProfileConfig,
    *, token: str, revision: str, downloader, manifest_sha256: str,
) -> None:
    """Validate one complete archive at a time, releasing scratch after each row."""
    prefix = run.hf_subfolder
    with tempfile.TemporaryDirectory(prefix="icgs-hf-resume-control-") as owned:
        scratch = Path(owned)
        resume = json.loads(_download_pinned_file(
            downloader, run, token, revision, f"{prefix}/resume_receipt.json", scratch,
        ).read_bytes())
        publication = PublicationReceipt.from_dict(json.loads(_download_pinned_file(
            downloader, run, token, revision, f"{prefix}/publication_receipt.json", scratch,
        ).read_bytes()))
        expected_identity = {
            "dataset_identity": profile.dataset_identity,
            "archive_format_id": profile.archive_format_id,
            "episode_schema_version": profile.episode_schema_version,
            "archive_profile": profile.as_dict(),
            "dataset_manifest_sha256": manifest_sha256,
        }
        if not isinstance(resume, Mapping):
            raise ValueError("remote archive resume receipt must be an object")
        for field, expected in expected_identity.items():
            if resume.get(field) != expected or getattr(publication, field) != expected:
                raise ValueError(f"remote archive receipt {field} mismatch")
        if (
            publication.status not in {"VERIFIED", "COMPLETE"}
            or publication.run_id not in manifest["source_run_ids"]
            or publication.source_run_id != publication.run_id
            or publication.prefix != prefix
            or resume.get("run_id") != publication.run_id
            or resume.get("source_run_id") != publication.run_id
            or resume.get("source_run_ids") != manifest["source_run_ids"]
            or resume.get("episodes") != len(manifest["episodes"])
            or resume.get("failure_attempts") != len(manifest["failure_attempts"])
            or _require_pinned_revision(resume.get("dataset_manifest_revision")) != publication.data_commit_oid
        ):
            raise ValueError("remote archive resume/publication receipt identity mismatch")
        _require_pinned_revision(publication.commit_oid)
        rows = list(manifest["episodes"]) + list(manifest["failure_attempts"])
        by_job = {row["job_id"]: row for row in rows}
        if not publication.job_ids or any(job not in by_job for job in publication.job_ids):
            raise ValueError("remote publication receipt job_ids are not in dataset manifest")
        receipt_hashes = {
            f"{job}/{relative}": digest
            for job in publication.job_ids
            for relative, digest in by_job[job]["file_sha256"].items()
        }
        if publication.artifact_hashes != receipt_hashes:
            raise ValueError("remote publication receipt artifact hashes mismatch")

    for row in rows:
        with tempfile.TemporaryDirectory(prefix="icgs-hf-resume-archive-") as owned:
            scratch = Path(owned)
            archive = scratch / "archive"
            archive.mkdir()
            for relative, digest in sorted(row["file_sha256"].items()):
                filename = f"{prefix}/{row['archive_ref']}/{relative}"
                target = archive / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="icgs-hf-resume-file-") as fetch:
                    remote = _download_pinned_file(
                        downloader, run, token, revision, filename, Path(fetch), digest,
                    )
                    shutil.copyfile(remote, target)
            manifest_name = "episode.manifest.json" if row["episode_id"] is not None else "attempt.manifest.json"
            validate_archive_manifest(archive / manifest_name)
            payload = json.loads((archive / manifest_name).read_text(encoding="utf-8"))
            if payload.get("archive_profile") != profile.as_dict():
                raise ValueError("remote archive manifest archive_profile disagrees with dataset profile")
            for field in ("archive_format_id", "episode_schema_version", "dataset_identity", "archive_kind", "episode_id", "attempt_id", "program_id", "outcome", "source_run_id", "code_revision", "preprocessing_identity", "split", "subset"):
                expected = (
                    ("episode" if row["episode_id"] is not None else "attempt")
                    if field == "archive_kind" else row.get(field)
                )
                if field in {"archive_format_id", "episode_schema_version", "dataset_identity"}:
                    expected = getattr(profile, field)
                if payload.get(field) != expected:
                    raise ValueError(f"remote archive {field} disagrees with dataset row")
            debug = json.loads((archive / "debug.json").read_text(encoding="utf-8"))
            if not isinstance(debug, Mapping):
                raise ValueError("remote archive debug metadata must be an object")
            job = debug.get("job_identity") or debug.get("job")
            embedded_plan = (
                job.get("plan") if isinstance(job, Mapping) else None
            ) or debug.get("_plan") or debug.get("plan")
            if embedded_plan != row["attempt_plan"]:
                raise ValueError("remote archive attempt_plan disagrees with debug provenance")
            if isinstance(job, Mapping) and job.get("job_id") != row["job_id"]:
                raise ValueError("remote archive job_id disagrees with debug provenance")


def _recover_interrupted_archive_publication(
    queue: FilesystemJobQueue,
    run: RunConfig,
    archive_profile,
    *,
    token: str,
    downloader,
) -> None:
    """Recover commit OIDs after HF accepted a commit but the process stopped before persisting them."""
    local_receipt_path = queue.root / "publication_receipt.json"
    if local_receipt_path.is_symlink():
        raise ValueError("pending publication receipt may not be a symlink")
    if not local_receipt_path.exists():
        return None
    if not local_receipt_path.is_file():
        raise ValueError("pending publication receipt must be a regular file")
    try:
        local_receipt = PublicationReceipt.from_dict(
            json.loads(local_receipt_path.read_text(encoding="utf-8"))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("pending publication receipt is malformed")
    if local_receipt.status not in {"PREPARED", "DATA_COMMITTED", "DEFERRED"}:
        return None
    if local_receipt.archive_profile != archive_profile.as_dict():
        raise ValueError("pending publication receipt archive profile mismatch")
    if (
        local_receipt.run_id != run.run_id
        or local_receipt.source_run_id != run.run_id
        or local_receipt.prefix != run.hf_subfolder
        or not local_receipt.job_ids
    ):
        raise ValueError("pending publication receipt run/job identity mismatch")
    for name in ("data_commit_oid", "commit_oid"):
        value = getattr(local_receipt, name)
        if value is not None and (
            not isinstance(value, str)
            or len(value) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"pending publication receipt has invalid {name}")
    if (
        local_receipt.dataset_manifest_sha256 is None
        or local_receipt.artifact_hashes is None
        or not local_receipt.artifact_hashes
    ):
        raise ValueError("pending archive publication receipt is missing immutable hashes")

    local_manifest_path = queue.root / "publication_manifest.json"
    if local_receipt.data_commit_oid is None:
        try:
            with tempfile.TemporaryDirectory(prefix="icgs-hf-recovery-manifest-") as owned:
                remote_manifest_path = _download_archive_control_file(
                    downloader, run, token,
                    f"{run.hf_subfolder}/dataset_manifest.json", Path(owned),
                    revision=None,
                )
                remote_manifest_bytes = remote_manifest_path.read_bytes()
        except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
            return None
        except Exception as error:
            raise RuntimeError(
                "cannot reconcile pending archive data commit; refusing a potentially duplicate upload"
            ) from error
        remote_manifest_sha = hashlib.sha256(remote_manifest_bytes).hexdigest()
        if remote_manifest_sha == local_receipt.dataset_manifest_sha256:
            revision = _snapshot_revision(remote_manifest_path)
            if revision is None:
                raise ValueError(
                    "remote dataset manifest matches the pending batch but its commit OID is unavailable"
                )
            if local_manifest_path.is_symlink() or not local_manifest_path.is_file():
                raise ValueError("pending publication is missing its local dataset manifest")
            if hashlib.sha256(local_manifest_path.read_bytes()).hexdigest() != remote_manifest_sha:
                raise ValueError("pending local dataset manifest changed during commit recovery")
            local_receipt = replace(
                local_receipt,
                status="DATA_COMMITTED",
                data_commit_oid=revision,
                updated_at_s=time.time(),
            )
            _atomic_json(local_receipt_path, local_receipt.as_dict())

    if local_receipt.data_commit_oid is None or local_receipt.commit_oid is not None:
        return None

    snapshot_path = queue.root / "publication_receipt_to_verify.json"
    try:
        with tempfile.TemporaryDirectory(prefix="icgs-hf-recovery-receipt-") as owned:
            remote_receipt_path = _download_archive_control_file(
                downloader, run, token,
                f"{run.hf_subfolder}/publication_receipt.json", Path(owned),
                revision=None,
            )
            remote_receipt_bytes = remote_receipt_path.read_bytes()
    except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
        return None
    except Exception as error:
        raise RuntimeError(
            "cannot reconcile pending archive receipt commit; refusing a potentially duplicate commit"
        ) from error

    receipt_revision = None
    if (
        snapshot_path.is_file()
        and not snapshot_path.is_symlink()
        and remote_receipt_bytes == snapshot_path.read_bytes()
    ):
        try:
            snapshot_receipt = PublicationReceipt.from_dict(
                json.loads(remote_receipt_bytes)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("pending receipt verification snapshot is malformed") from error
        fields = (
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
        if any(
            getattr(snapshot_receipt, field) != getattr(local_receipt, field)
            for field in fields
        ):
            raise ValueError("remote receipt snapshot identity mismatch")
        receipt_revision = _snapshot_revision(remote_receipt_path)
        if receipt_revision is None:
            raise ValueError("remote publication receipt snapshot has no commit OID")
    else:
        try:
            remote_receipt = PublicationReceipt.from_dict(json.loads(remote_receipt_bytes))
        except (TypeError, ValueError, json.JSONDecodeError):
            remote_receipt = None
        if remote_receipt is not None and (
            remote_receipt.status in {"VERIFIED", "COMPLETE"}
            and remote_receipt.run_id == local_receipt.run_id
            and remote_receipt.job_ids == local_receipt.job_ids
            and remote_receipt.data_commit_oid == local_receipt.data_commit_oid
            and remote_receipt.dataset_manifest_sha256 == local_receipt.dataset_manifest_sha256
            and remote_receipt.artifact_hashes == local_receipt.artifact_hashes
            and remote_receipt.source_run_id == local_receipt.source_run_id
            and remote_receipt.dataset_identity == local_receipt.dataset_identity
            and remote_receipt.archive_format_id == local_receipt.archive_format_id
            and remote_receipt.episode_schema_version == local_receipt.episode_schema_version
            and remote_receipt.archive_profile == local_receipt.archive_profile
            and remote_receipt.prefix == local_receipt.prefix
        ):
            candidate_revision = remote_receipt.commit_oid
            if (
                isinstance(candidate_revision, str)
                and len(candidate_revision) in {40, 64}
                and all(character in "0123456789abcdef" for character in candidate_revision)
            ):
                receipt_revision = candidate_revision
    if receipt_revision is None:
        return None
    local_receipt = replace(
        local_receipt,
        status="DATA_COMMITTED",
        commit_oid=receipt_revision,
        updated_at_s=time.time(),
    )
    _atomic_json(local_receipt_path, local_receipt.as_dict())
    return None


def _validate_resume_manifest(
    manifest: Mapping[str, object],
    run: RunConfig,
    *,
    approved_program_ids: set[str] | None = None,
) -> dict:
    """Validate the immutable portion of a remote manifest before planning."""
    if not isinstance(manifest, Mapping):
        raise ValueError("remote resume manifest must be an object")
    if manifest.get("manifest_version", 3) != 3:
        raise ValueError("remote resume manifest_version must be 3")
    source_ids: set[str] = set()
    for key in ("run_id", "source_run_id"):
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"remote resume {key} must be a nonblank string")
            source_ids.add(value)
    for value in manifest.get("source_run_ids", ()) or ():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("remote resume source_run_ids must contain strings")
        source_ids.add(value)
    if run.resume_from_hf and not source_ids:
        raise ValueError("remote resume manifest must identify its source run")
    if run.run_id in source_ids:
        raise ValueError("resumed run_id must be disjoint from remote source runs")

    identities: set[str] = set()
    validated = dict(manifest)
    for collection, identity_key in (("episodes", "episode_id"), ("failure_attempts", "attempt_id")):
        rows = manifest.get(collection, [])
        if not isinstance(rows, list):
            raise ValueError(f"remote resume {collection} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"remote resume {collection} row must be an object")
            identity = row.get(identity_key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"remote resume {collection} row missing {identity_key}")
            if identity in identities:
                raise ValueError(f"remote resume duplicate immutable identity: {identity}")
            identities.add(identity)
            for required in ("program_id", "outcome"):
                if not isinstance(row.get(required), str) or not str(row[required]).strip():
                    raise ValueError(f"remote resume {collection} row missing {required}")
            if approved_program_ids is not None and str(row["program_id"]) not in approved_program_ids:
                raise ValueError(f"remote resume row uses unapproved program: {row['program_id']}")
            row_run_id = row.get("run_id")
            if row_run_id is not None and row_run_id == run.run_id:
                raise ValueError("resumed row run_id must be disjoint from current run")
            attempt_plan = row.get("attempt_plan")
            if attempt_plan is not None:
                if not isinstance(attempt_plan, Mapping):
                    raise ValueError("remote resume attempt_plan must be an object")
                if attempt_plan.get("episode_id") != row.get("episode_id"):
                    raise ValueError("remote resume attempt_plan episode identity mismatch")
                if attempt_plan.get("program_id") != row.get("program_id"):
                    raise ValueError("remote resume attempt_plan program identity mismatch")
    return validated


def _resume_archive_file_path(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"archive {field} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"archive {field} must be a portable relative path")
    return path.as_posix()


def _resume_archive_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"archive {field} must be a lowercase SHA256 digest")
    return value


def _remote_file_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Path):
        return value.read_bytes()
    if isinstance(value, str):
        return Path(value).read_bytes()
    raise ValueError("remote archive file must be bytes or a regular file path")


def _validate_remote_archive_manifest_bytes(
    row: Mapping[str, Any],
    content: bytes,
    profile: ArchiveProfileConfig,
) -> None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("remote archive manifest is malformed") from error
    if not isinstance(payload, Mapping):
        raise ValueError("remote archive manifest must be an object")
    archive_kind = "episode" if row.get("episode_id") is not None else "attempt"
    expected = {
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "dataset_identity": profile.dataset_identity,
        "archive_kind": archive_kind,
        "episode_id": row.get("episode_id"),
        "attempt_id": row.get("attempt_id"),
        "program_id": row.get("program_id"),
        "outcome": row.get("outcome"),
        "source_run_id": row.get("source_run_id"),
        "code_revision": row.get("code_revision"),
        "preprocessing_identity": row.get("preprocessing_identity"),
    }
    archive_profile = payload.get("archive_profile")
    if archive_profile != profile.as_dict():
        raise ValueError("remote archive manifest profile disagrees with dataset manifest")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"remote archive manifest identity mismatch for {field}")
    chunks = payload.get("chunk_inventory")
    if not isinstance(chunks, list):
        raise ValueError("remote archive manifest chunk inventory must be a list")
    if archive_kind == "episode" and not chunks:
        raise ValueError("remote archive manifest chunk inventory is empty")
    declared_files = row["file_sha256"]
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise ValueError("remote archive manifest chunk entry must be an object")
        path = _resume_archive_file_path(chunk.get("path"), field="chunk reference")
        if not path.startswith("data/chunk-") or not path.endswith(".npz"):
            raise ValueError("remote archive manifest chunk reference is not canonical")
        chunk_hash = _resume_archive_sha(chunk.get("sha256"), field=f"chunk {path}")
        if declared_files.get(path) != chunk_hash:
            raise ValueError(f"remote archive manifest chunk reference is missing from row: {path}")


def _validate_archive_resume_manifest(
    manifest: Mapping[str, Any],
    run: RunConfig,
    profile: ArchiveProfileConfig,
    *,
    approved_program_ids: set[str] | None = None,
    approved_rows: Mapping[str, Mapping[str, Any]] | None = None,
    remote_files: Mapping[str, Any] | None = None,
    remote_manifest_sha256: str | None = None,
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not isinstance(profile, ArchiveProfileConfig):
        raise TypeError("archive_profile must be an ArchiveProfileConfig")
    if manifest_bytes is not None and remote_manifest_sha256 is not None:
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != remote_manifest_sha256:
            raise ValueError("remote dataset manifest SHA256 does not match the pinned hash")
    if manifest.get("manifest_version", 3) != 3:
        raise ValueError("remote resume manifest_version must be 3")
    for field, expected in (
        ("dataset_identity", profile.dataset_identity),
        ("archive_format_id", profile.archive_format_id),
        ("episode_schema_version", profile.episode_schema_version),
        ("archive_profile", profile.as_dict()),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"remote archive manifest {field} mismatch")
    source_ids: set[str] = set()
    source_run_ids = manifest.get("source_run_ids")
    if not isinstance(source_run_ids, list) or not source_run_ids:
        raise ValueError("remote archive manifest must identify source_run_ids")
    for value in source_run_ids:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("remote archive source_run_ids must contain strings")
        source_ids.add(value)
    for key in ("run_id", "source_run_id"):
        value = manifest.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"remote archive {key} must be a nonblank string")
            source_ids.add(value)
    if run.resume_from_hf and not source_ids:
        raise ValueError("remote resume manifest must identify its source run")
    if run.run_id in source_ids:
        raise ValueError("resumed run_id must be disjoint from remote source runs")

    identities: set[str] = set()
    attempt_ids: set[str] = set()
    job_ids: set[str] = set()
    validated = dict(manifest)
    remote_files_supplied = remote_files is not None
    remote_file_inventory: dict[str, Any] = {}
    expected_remote_files: set[str] = set()
    if remote_files_supplied:
        prefix = f"{run.hf_subfolder}/" if run.hf_subfolder else None
        for key, value in dict(remote_files).items():
            relative = _resume_archive_file_path(key, field="remote file reference")
            if prefix and relative.startswith(prefix):
                relative = _resume_archive_file_path(
                    relative[len(prefix):], field="remote file reference"
                )
            if relative in remote_file_inventory:
                raise ValueError(f"remote archive file is supplied more than once: {relative}")
            remote_file_inventory[relative] = value
    for collection, identity_key, allowed_outcomes in (
        ("episodes", "episode_id", {"success", "valid_failure"}),
        ("failure_attempts", "attempt_id", {"simulator_crash", "invalid_observation"}),
    ):
        rows = manifest.get(collection, [])
        if not isinstance(rows, list):
            raise ValueError(f"remote archive {collection} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"remote archive {collection} row must be an object")
            row = dict(row)
            identity = row.get(identity_key)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"remote archive row missing {identity_key}")
            if identity in identities:
                raise ValueError(f"remote resume duplicate immutable identity: {identity}")
            identities.add(identity)
            for field in (
                "job_id",
                "run_id",
                "source_run_id",
                "code_revision",
                "preprocessing_identity",
                "program_id",
                "attempt_id",
            ):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"remote archive row {field} must be a nonblank string")
            if row["job_id"] in job_ids:
                raise ValueError(f"remote archive duplicate job identity: {row['job_id']}")
            if row["attempt_id"] in attempt_ids:
                raise ValueError(f"remote archive duplicate attempt identity: {row['attempt_id']}")
            job_ids.add(row["job_id"])
            attempt_ids.add(row["attempt_id"])
            if row.get("episode_id") is not None and (
                not isinstance(row.get("episode_id"), str) or not row["episode_id"].strip()
            ):
                raise ValueError("remote archive row episode_id must be a nonblank string or null")
            if approved_program_ids is not None and row.get("program_id") not in approved_program_ids:
                raise ValueError(f"remote resume row uses unapproved program: {row.get('program_id')}")
            if row.get("outcome") not in allowed_outcomes:
                raise ValueError(f"remote archive {collection} has invalid outcome")
            if row.get("run_id") != row.get("source_run_id") or row.get("source_run_id") not in source_ids:
                raise ValueError("remote archive row source_run_id is not bound to the manifest")
            if row.get("run_id") == run.run_id:
                raise ValueError("resumed row run_id must be disjoint from current run")
            if (
                not isinstance(row.get("manifest_sha256"), str)
                or len(row["manifest_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in row["manifest_sha256"])
            ):
                raise ValueError("remote archive row manifest_sha256 must be a lowercase SHA256 digest")
            if type(row.get("retry_generation")) is not int or row["retry_generation"] < 0:
                raise ValueError("remote archive row retry_generation must be nonnegative")
            for field, expected in (
                ("archive_format_id", profile.archive_format_id),
                ("episode_schema_version", profile.episode_schema_version),
                ("dataset_identity", profile.dataset_identity),
            ):
                if row.get(field) != expected:
                    raise ValueError(f"remote archive row {field} mismatch")
            archive_ref = _resume_archive_file_path(row.get("archive_ref"), field="archive_ref")
            expected_prefix = "episodes/" if collection == "episodes" else "attempts/"
            if not archive_ref.startswith(expected_prefix) or len(PurePosixPath(archive_ref).parts) != 3:
                raise ValueError("remote archive_ref is not canonical")
            expected_ref = (
                f"episodes/{row['program_id']}/{row['episode_id']}"
                if collection == "episodes"
                else f"attempts/{row['program_id']}/{row['attempt_id']}"
            )
            if archive_ref != expected_ref:
                raise ValueError("remote archive_ref does not match row identity")
            manifest_name = "episode.manifest.json" if collection == "episodes" else "attempt.manifest.json"
            archive_manifest = _resume_archive_file_path(row.get("archive_manifest"), field="manifest reference")
            if archive_manifest != f"{archive_ref}/{manifest_name}":
                raise ValueError("remote archive manifest reference does not match archive_ref")
            inventory = row.get("file_sha256")
            if not isinstance(inventory, Mapping) or not inventory:
                raise ValueError("remote archive row file_sha256 must be nonempty")
            inventory = dict(inventory)
            required = {manifest_name, "artifact_manifest.json", "debug.json"}
            if collection == "episodes":
                required.add("data/chunk-00000.npz")
            if not required.issubset(inventory):
                missing = sorted(required - set(inventory))
                raise ValueError(f"remote archive row is missing archive/chunk reference: {missing}")
            for relative, digest in inventory.items():
                relative = _resume_archive_file_path(relative, field="file reference")
                _resume_archive_sha(digest, field=f"file {relative}")
            try:
                plan = attempt_from_dict(row["attempt_plan"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("remote archive row attempt_plan is malformed") from error
            if plan.program_id != row.get("program_id"):
                raise ValueError("remote archive attempt_plan program identity mismatch")
            try:
                catalog_split = get_generation_program(plan.program_id).split
            except (KeyError, ValueError) as error:
                raise ValueError("remote archive attempt_plan uses unknown catalog program") from error
            if plan.split != catalog_split:
                raise ValueError("remote archive attempt_plan split disagrees with catalog")
            if plan.episode_kind not in {"nominal", "perturbed"}:
                raise ValueError("remote archive attempt_plan episode_kind is invalid")
            if not isinstance(plan.randomization, Mapping):
                raise ValueError("remote archive attempt_plan randomization must be an object")
            for field in ("asset_instance_id", "asset_family_id"):
                asset = plan.randomization.get(field)
                if not isinstance(asset, str) or not asset.strip():
                    raise ValueError(f"remote archive attempt_plan {field} is invalid")
            if plan.randomization["asset_instance_id"] != f"{plan.program_id}-asset-{plan.scene_seed}":
                raise ValueError("remote archive attempt_plan asset_instance_id disagrees with planner seed")
            if plan.randomization.get("scene_seed") != plan.scene_seed:
                raise ValueError("remote archive attempt_plan randomization scene_seed mismatch")
            for field, expected in (("program_id", plan.program_id), ("split", plan.split)):
                declared = plan.randomization.get(field)
                if declared is not None and declared != expected:
                    raise ValueError(f"remote archive attempt_plan randomization {field} mismatch")
            if approved_rows is not None:
                approved_row = approved_rows.get(plan.program_id)
                if approved_row is None:
                    raise ValueError("remote archive attempt_plan program is not approved")
                expected_family = approved_row.get("asset_family_id")
                if plan.randomization["asset_family_id"] != expected_family:
                    raise ValueError("remote archive attempt_plan asset_family_id disagrees with approved catalog")
            if plan.episode_kind == "nominal" and plan.intervention is not None:
                raise ValueError("remote archive nominal attempt_plan cannot have intervention")
            if plan.episode_kind == "perturbed":
                intervention = plan.intervention
                if not isinstance(intervention, Mapping):
                    raise ValueError("remote archive perturbed attempt_plan requires intervention")
                kind = intervention.get("kind") or intervention.get("intervention_type")
                if kind not in applicable_perturbations(plan.program_id):
                    raise ValueError("remote archive attempt_plan intervention kind is invalid for program")
                if intervention.get("intervention_id") != f"{kind}_v1":
                    raise ValueError("remote archive attempt_plan intervention_id mismatch")
            if f"att-{plan.episode_id}" != row.get("attempt_id"):
                raise ValueError("remote archive attempt_plan attempt identity mismatch")
            if collection == "episodes" and plan.episode_id != row.get("episode_id"):
                raise ValueError("remote archive attempt_plan episode identity mismatch")
            if collection == "failure_attempts" and row.get("episode_id") is not None:
                raise ValueError("remote archive failure attempt episode_id must be null")
            expected_split = "dev" if plan.split == "development" else plan.split
            if row.get("split") != expected_split:
                raise ValueError("remote archive row split disagrees with attempt_plan")
            if row.get("episode_kind") != plan.episode_kind:
                raise ValueError("remote archive row episode_kind disagrees with attempt_plan")
            planned_subset = (
                train_subset_for_sample(plan.randomization) if plan.split == "train" else None
            )
            if row.get("subset") != planned_subset:
                raise ValueError("remote archive row subset disagrees with attempt_plan")
            expected_fields = {
                "scene_signature": plan.randomization.get("scene_signature") or scene_signature(plan.randomization),
                "scene_seed": plan.scene_seed,
                "episode_index": plan.episode_index,
                "asset_instance_id": plan.randomization.get("asset_instance_id"),
                "asset_family_id": plan.randomization.get("asset_family_id"),
                "intervention_id": (plan.intervention or {}).get("intervention_id"),
                "source_episode_id": (plan.intervention or {}).get("source_episode_id"),
                "base_episode_id": (plan.intervention or {}).get("base_episode_id"),
            }
            for field, expected in expected_fields.items():
                if row.get(field) != expected:
                    raise ValueError(f"remote archive row {field} disagrees with attempt_plan")
            if collection == "episodes" and (
                not isinstance(row.get("source_lineage_id"), str)
                or not row["source_lineage_id"].strip()
            ):
                raise ValueError("remote archive episode source_lineage_id is required")

            for relative, digest in inventory.items():
                remote_key = f"{archive_ref}/{relative}"
                expected_remote_files.add(remote_key)
                remote_value = remote_file_inventory.get(remote_key)
                if remote_files_supplied and remote_value is None:
                    raise ValueError(f"remote archive file is missing: {remote_key}")
                if remote_value is not None:
                    content = _remote_file_bytes(remote_value)
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError(f"remote archive hash mismatch: {remote_key}")
                    if relative == manifest_name:
                        _validate_remote_archive_manifest_bytes(row, content, profile)
        validated[collection] = [dict(row) for row in rows]
    if remote_files_supplied:
        unexpected = sorted(set(remote_file_inventory) - expected_remote_files)
        if unexpected:
            raise ValueError(f"remote archive contains files outside the declared inventory: {unexpected}")
        for collection in ("episodes", "failure_attempts"):
            for row in validated[collection]:
                with tempfile.TemporaryDirectory(prefix="icgs-resume-fixture-") as owned:
                    root = Path(owned)
                    for relative in row["file_sha256"]:
                        source_key = f"{row['archive_ref']}/{relative}"
                        target = root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(_remote_file_bytes(remote_file_inventory[source_key]))
                    name = "episode.manifest.json" if collection == "episodes" else "attempt.manifest.json"
                    validate_archive_manifest(root / name)
    return validated


def load_remote_manifest(
    manifest: Mapping[str, Any] | str | Path,
    run: RunConfig,
    *,
    archive_profile: ArchiveProfileConfig | None = None,
    approved_program_ids: set[str] | None = None,
    approved_rows: Mapping[str, Mapping[str, Any]] | None = None,
    remote_files: Mapping[str, Any] | None = None,
    remote_manifest_sha256: str | None = None,
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Parse and validate a legacy or lossless archive resume manifest.

    ``remote_files`` is a complete local fixture seam keyed by each row's
    HF-relative ``archive_ref/<file>`` path. The coordinator downloads every
    declared archive file at one pinned revision and invokes the canonical
    validator before using the returned rows for planning.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        raw_manifest_bytes = manifest_path.read_bytes()
        payload = json.loads(raw_manifest_bytes)
    else:
        payload = dict(manifest)
        raw_manifest_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if manifest_bytes is None:
        manifest_bytes = raw_manifest_bytes
    if archive_profile is None:
        return _validate_resume_manifest(payload, run, approved_program_ids=approved_program_ids)
    return _validate_archive_resume_manifest(
        payload,
        run,
        archive_profile,
        approved_program_ids=approved_program_ids,
        approved_rows=approved_rows,
        remote_files=remote_files,
        remote_manifest_sha256=remote_manifest_sha256,
        manifest_bytes=manifest_bytes,
    )


def _verify_remote_batch(queue: FilesystemJobQueue, run: RunConfig, job_ids: tuple[str, ...], revision: str, token: str) -> None:
    if (
        not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("remote verification requires a pinned commit OID")

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    def verify_file(filename: str, local_path: Path, expected_hash: str) -> None:
        if local_path.is_symlink() or not local_path.is_file():
            raise ValueError(f"local publication artifact is not a regular file: {filename}")
        local_hash = digest(local_path)
        if local_hash != expected_hash:
            raise ValueError(f"local hash mismatch for {filename}")
        # A fresh cache per file bounds disk lifetime and prevents this
        # verification pass from reading stale entries in the shared HF cache.
        with tempfile.TemporaryDirectory(prefix="icgs-hf-verify-") as scratch:
            remote_path = hf_hub_download(
                repo_id=run.hf_repo,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                token=token,
                force_download=True,
                cache_dir=scratch,
            )
            remote = Path(remote_path)
            if not remote.is_file():
                raise ValueError(f"remote artifact download is not a regular file: {filename}")
            if digest(remote) != expected_hash:
                raise ValueError(f"remote hash mismatch for {filename}")

    run_root = queue.root.parent.resolve(strict=True)
    for job_id in job_ids:
        directory = queue.root / "ingested" / job_id
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        worker_result = WorkerResult.from_dict(result)
        result_root = Path(result["result_dir"])
        if result_root.is_symlink() or not result_root.is_absolute() or ".." in result_root.parts:
            raise ValueError(f"remote verification result_dir is not a real absolute path: {job_id}")
        lexical_root = Path(os.path.abspath(result_root))
        if not lexical_root.is_relative_to(run_root) or lexical_root == run_root:
            raise ValueError(f"remote verification result_dir escapes the run root: {job_id}")
        current = run_root
        for part in lexical_root.relative_to(run_root).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"remote verification result_dir traverses a symlink: {job_id}")
        resolved_root = result_root.resolve(strict=True)
        if not resolved_root.is_relative_to(run_root) or resolved_root == run_root:
            raise ValueError(f"remote verification result_dir escapes the run root: {job_id}")
        prefix = (
            f"{run.hf_subfolder}/episodes/{result['program_id']}/{result['episode_id']}"
            if result.get("episode_id")
            else f"{run.hf_subfolder}/attempts/{result['program_id']}/{result['attempt_id']}"
        )
        if not worker_result.file_sha256:
            raise ValueError(f"remote verification artifact inventory is empty: {job_id}")
        for relative, expected_hash in sorted(worker_result.file_sha256.items()):
            local_path = resolved_root / relative
            verify_file(f"{prefix}/{relative}", local_path, expected_hash)
        if worker_result.episode_id is not None and worker_result.outcome in {"success", "valid_failure"}:
            for view in ("D_geom", "D_temporal", "D_dyn", "D_task"):
                pointer_path = (
                    queue.root / "views" / "provisional" / "episodes"
                    / str(result["program_id"]) / str(worker_result.episode_id)
                    / f"{view}.json"
                )
                pointer_filename = (
                    f"{run.hf_subfolder}/views/provisional/episodes/"
                    f"{result['program_id']}/{worker_result.episode_id}/{view}.json"
                )
                if pointer_path.is_symlink() or not pointer_path.is_file():
                    raise ValueError(
                        f"local publication artifact is not a regular file: {pointer_filename}"
                    )
                verify_file(pointer_filename, pointer_path, digest(pointer_path))

    manifest_path = queue.root / "publication_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("remote verification requires the local dataset manifest")
    manifest_hash = digest(manifest_path)
    receipt_path = queue.root / "publication_receipt.json"
    if receipt_path.is_file() and not receipt_path.is_symlink():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_manifest_hash = receipt.get("dataset_manifest_sha256")
        if expected_manifest_hash is not None and expected_manifest_hash != manifest_hash:
            raise ValueError("local dataset manifest hash disagrees with publication receipt")
    verify_file(
        f"{run.hf_subfolder}/dataset_manifest.json",
        manifest_path,
        manifest_hash,
    )
    control_files = [
        (
            f"{run.hf_subfolder}/resume_receipt.json",
            queue.root / "resume_receipt.json",
        ),
        *[
            (
                f"{run.hf_subfolder}/views/{view}.json",
                queue.root / "views" / f"{view}.json",
            )
            for view in ("D_geom", "D_temporal", "D_dyn", "D_task")
        ],
        (
            f"{run.hf_subfolder}/publication_receipt.json",
            queue.root / "publication_receipt_to_verify.json",
        ),
    ]
    for filename, local_path in control_files:
        verify_file(filename, local_path, digest(local_path))


class CoordinatorControlPlane:
    def __init__(
        self,
        run: RunConfig,
        queue: FilesystemJobQueue,
        planner: DistributedPlanner,
        publisher: HuggingFaceBatchPublisher,
        manifest=None,
        *,
        archive_profile=None,
    ):
        self.run = run
        self.queue = queue
        self.planner = planner
        self.publisher = publisher
        self.archive_profile = archive_profile
        self.manifest = dict(manifest or {"manifest_version": 3, "episodes": [], "failure_attempts": []})
        self.status = "PREFLIGHT"
        self._tick_started_at_s: float | None = None
        self._last_progress_at_s: float | None = None
        self._last_progress_kind: str | None = None
        self._last_validation_error: dict | None = None
        self._publication: dict = {"phase": "idle"}

    @classmethod
    def open(
        cls,
        run_config_path: str | Path,
        *,
        api_factory,
        token_path: str | Path | None = None,
        runtime_config_path: str | Path | None = None,
    ):
        payload = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
        run = RunConfig.from_dict(payload["run"])
        configured_runtime_path = runtime_config_path or payload.get("runtime_config_path")
        if configured_runtime_path is None:
            raise ValueError("run receipt must contain runtime_config_path")
        configured_runtime_path = Path(configured_runtime_path)
        if configured_runtime_path.is_symlink() or not configured_runtime_path.is_file():
            raise ValueError(f"runtime config must be a regular file: {configured_runtime_path}")
        expected_runtime_sha = payload.get("runtime_config_sha256")
        if expected_runtime_sha is not None:
            actual_runtime_sha = hashlib.sha256(configured_runtime_path.read_bytes()).hexdigest()
            if actual_runtime_sha != expected_runtime_sha:
                raise ValueError("runtime config digest mismatch")
        runtime = GenerationRuntimeConfig.from_file(configured_runtime_path, check_paths=False)
        _validate_runtime_run_binding(run, runtime)
        queue = FilesystemJobQueue(run.run_root + "/queue")
        approved = json.loads(Path(payload["approved_manifest"]).read_text(encoding="utf-8"))
        approved_bytes = Path(payload["approved_manifest"]).read_bytes()
        if hashlib.sha256(approved_bytes).hexdigest() != run.approved_manifest_sha256:
            raise ValueError("approved manifest digest does not match run config")
        rows = {row["program_id"]: row for row in approved["catalog"]}
        token = _credential_path(run_config_path, payload, token_path).read_text(
            encoding="utf-8"
        ).strip()
        local_manifest = _manifest_from_closed_queue(
            queue, archive_profile=runtime.archive_profile
        )
        if not local_manifest["episodes"] and not local_manifest["failure_attempts"]:
            local_manifest = payload.get("manifest", local_manifest)
        remote_manifest: dict = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
        remote_manifest_sha256: str | None = None
        remote_error: Exception | None = None
        remote_manifest_absent = False
        bootstrap_path = Path(run.run_root) / "control" / "resume_bootstrap.json"
        remote_revision = None
        bootstrap_manifest_sha256 = None
        if bootstrap_path.is_file():
            bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            if bootstrap.get("run_id") != run.run_id:
                raise ValueError("resume bootstrap run_id does not match run config")
            remote_revision = bootstrap.get("remote_revision")
            bootstrap_manifest_sha256 = bootstrap.get("remote_manifest_sha256")
        if runtime.archive_profile is not None and run.resume_from_hf:
            remote_revision = _require_pinned_revision(remote_revision)
        try:
            if runtime.archive_profile is not None and run.resume_from_hf:
                with tempfile.TemporaryDirectory(prefix="icgs-hf-resume-manifest-") as scratch:
                    remote_path = _download_pinned_file(
                        hf_hub_download, run, token, remote_revision,
                        f"{run.hf_subfolder}/dataset_manifest.json", Path(scratch),
                    )
                    remote_bytes = remote_path.read_bytes()
            elif runtime.archive_profile is not None:
                with tempfile.TemporaryDirectory(prefix="icgs-hf-initial-manifest-") as scratch:
                    try:
                        remote_path = _download_archive_control_file(
                            hf_hub_download, run, token,
                            f"{run.hf_subfolder}/dataset_manifest.json", Path(scratch),
                            revision=remote_revision,
                        )
                    except EntryNotFoundError as error:
                        if (
                            isinstance(error, LocalEntryNotFoundError)
                            or remote_revision is not None
                        ):
                            raise
                        remote_manifest_absent = True
                    else:
                        remote_bytes = remote_path.read_bytes()
            else:
                remote_path = hf_hub_download(
                    repo_id=run.hf_repo, repo_type="dataset",
                    filename=f"{run.hf_subfolder}/dataset_manifest.json",
                    token=token, force_download=True, revision=remote_revision,
                )
                remote_bytes = Path(remote_path).read_bytes()
            if not remote_manifest_absent:
                remote_manifest_sha256 = hashlib.sha256(remote_bytes).hexdigest()
                if bootstrap_manifest_sha256 and remote_manifest_sha256 != bootstrap_manifest_sha256:
                    raise ValueError("remote resume manifest digest changed after preflight")
                remote_manifest = load_remote_manifest(
                    json.loads(remote_bytes),
                    run,
                    archive_profile=runtime.archive_profile,
                    approved_program_ids=set(rows),
                    approved_rows=rows if runtime.archive_profile is not None else None,
                    remote_manifest_sha256=remote_manifest_sha256,
                    manifest_bytes=remote_bytes,
                )
                if runtime.archive_profile is not None and run.resume_from_hf:
                    _verify_archive_resume_files(
                        remote_manifest, run, runtime.archive_profile, token=token,
                        revision=remote_revision, downloader=hf_hub_download,
                        manifest_sha256=remote_manifest_sha256,
                    )
        except Exception as error:
            remote_error = error
        if (
            runtime.archive_profile is not None
            and not run.resume_from_hf
            and remote_error is not None
        ):
            raise RuntimeError(
                "archive startup requires a valid remote dataset manifest or confirmed absence"
            ) from remote_error
        if runtime.archive_profile is not None and run.publication_enabled:
            _recover_interrupted_archive_publication(
                queue,
                run,
                runtime.archive_profile,
                token=token,
                downloader=hf_hub_download,
            )
        if run.resume_from_hf:
            if remote_error is not None:
                raise RuntimeError(
                    "resume_from_hf requires a readable remote dataset manifest"
                ) from remote_error
            manifest = _merge_manifests(remote_manifest, local_manifest)
        else:
            manifest = local_manifest
        if remote_error is None:
            _atomic_json(
                Path(run.run_root) / "control" / "resume_bootstrap.json",
                {
                    "run_id": run.run_id,
                    "remote_manifest_sha256": remote_manifest_sha256,
                    "remote_revision": remote_revision,
                    "source_run_ids": list(remote_manifest.get("source_run_ids") or ()),
                    "fetched_at_s": time.time(),
                },
            )
        planner = DistributedPlanner.from_manifest(run, rows, manifest, _inflight_jobs_from_queue(queue))
        api = api_factory()
        publisher = HuggingFaceBatchPublisher(
            run, api, token, queue,
            archive_profile=runtime.archive_profile,
            remote_verify=lambda job_ids, revision, _token: _verify_remote_batch(
                queue, run, job_ids, revision, token
            ),
        )
        if remote_error is None:
            publisher.remote_manifest = remote_manifest
        else:
            publisher.remote_manifest = _manifest_from_closed_queue(
                queue,
                states=("published",),
                archive_profile=runtime.archive_profile,
            )
        return cls(
            run,
            queue,
            planner,
            publisher,
            manifest,
            archive_profile=runtime.archive_profile,
        )

    def _refill(self, target: int = 400) -> int:
        validation_max_jobs = getattr(self.run, "validation_max_jobs", None)
        bounded_validation = self.run.validation_mode and validation_max_jobs is not None
        if bounded_validation:
            target = min(target, validation_max_jobs)
        refilled = 0
        while True:
            counts = self.queue.counts()
            current_jobs = counts.pending + counts.claimed
            if bounded_validation:
                current_jobs += counts.ready + counts.ingested + counts.published
            if current_jobs >= target:
                break
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
            validated = (
                validate_closed_result(job, result, archive_profile=self.archive_profile)
                if self.archive_profile is not None
                else validate_closed_result(job, result)
            )
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
            try:
                publication_receipt = json.loads(
                    (self.queue.root / "publication_receipt.json").read_text(
                        encoding="utf-8"
                    )
                )
                state = str(publication_receipt.get("status", ""))
                if state in {"PREPARED", "DATA_COMMITTED", "VERIFIED", "DEFERRED", "COMPLETE"}:
                    self._publication = {
                        "phase": state.lower(),
                        "status": state,
                        "last_error": publication_receipt.get("last_error"),
                        "next_retry_s": publication_receipt.get("next_retry_s"),
                        "data_commit_oid": publication_receipt.get("data_commit_oid"),
                    }
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
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
    parser.add_argument("--hf-token-path")
    args = parser.parse_args()
    if args.runtime_config:
        GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    run_payload = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    credential_path = _credential_path(args.run_config, run_payload, args.hf_token_path)
    token = credential_path.read_text(encoding="utf-8").strip()
    api = HfApi(token=token)
    control = CoordinatorControlPlane.open(
        args.run_config,
        api_factory=lambda: api,
        token_path=credential_path,
        runtime_config_path=args.runtime_config,
    )
    return control.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
