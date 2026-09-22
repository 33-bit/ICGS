"""Validate and immutably ingest closed distributed collection results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import traceback as traceback_module
from typing import Any, Mapping

from icgs.data.collection.generation.distributed_contracts import (
    GenerationJob,
    ValidationGateResult,
    ValidationPlan,
    ValidationReceipt,
    WorkerResult,
)
from icgs.data.collection.generation.report import split_disjointness_report
from icgs.data.schemas.episode_records import validate_episode


@dataclass(frozen=True)
class ValidatedResult:
    job: GenerationJob
    result: WorkerResult
    provenance: dict[str, Any]
    episode_entry: dict[str, Any] | None
    attempt_entry: dict[str, Any] | None

    @property
    def outcome(self) -> str:
        return self.result.outcome


@dataclass(frozen=True)
class ValidationFailure:
    job_id: str
    job_identity: dict[str, Any] | None
    result_identity: dict[str, Any]
    exception_type: str
    exception_message: str
    traceback: str
    actual_file_inventory: dict[str, dict[str, Any]]
    source_path: str

    @classmethod
    def capture(
        cls,
        job: GenerationJob | None,
        result: WorkerResult | None,
        error: BaseException,
        *,
        source_path: str | Path,
        allowed_result_root: str | Path | None = None,
        job_id: str | None = None,
        result_identity: Mapping[str, Any] | None = None,
        result_dir: str | Path | None = None,
    ) -> "ValidationFailure":
        identity = dict(result_identity) if result_identity is not None else (
            result.as_dict() if result is not None else None
        )
        identity_job_id = result.job_id if result is not None else job_id
        if identity is None or identity_job_id is None:
            raise ValueError("result identity and job_id are required for validation failure")
        allowed_root = allowed_result_root
        if allowed_root is None:
            allowed_root = job.output_root if job is not None else source_path
        inventory_root = result_dir
        if inventory_root is None:
            inventory_root = result.result_dir if result is not None else source_path
        return cls(
            job_id=identity_job_id,
            job_identity=job.as_dict() if job is not None else None,
            result_identity=identity,
            exception_type=type(error).__name__,
            exception_message=str(error),
            traceback="".join(
                traceback_module.format_exception(type(error), error, error.__traceback__)
            ),
            actual_file_inventory=_safe_file_inventory(
                Path(inventory_root),
                allowed_root=Path(allowed_root),
            ),
            source_path=str(source_path),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_identity": self.job_identity,
            "result_identity": self.result_identity,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback,
            "actual_file_inventory": self.actual_file_inventory,
            "source_path": self.source_path,
        }


def load_validation_plan(
    path: str | Path,
    *,
    validation_mode: bool | None = None,
) -> ValidationPlan:
    """Load a bounded JSON validation plan; never import or execute plan content."""
    payload = _read_strict_json_object(path)
    if validation_mode is False and payload.get("fault_injection"):
        raise ValueError("fault_injection is only allowed in validation_mode")
    return ValidationPlan.from_dict(payload)


def _read_strict_json_object(path: str | Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("validation plan must be a JSON object")
    return payload


def validate_validation_receipt(
    receipt: ValidationReceipt,
    plan: ValidationPlan,
) -> ValidationReceipt:
    """Validate explicit PASS/FAIL/NOT_RUN states and their evidence contract."""
    if not isinstance(receipt, ValidationReceipt):
        raise TypeError("receipt must be a ValidationReceipt")
    if not isinstance(plan, ValidationPlan):
        raise TypeError("plan must be a ValidationPlan")
    if receipt.plan_id != plan.plan_id:
        raise ValueError("validation receipt plan_id does not match plan")
    if tuple(receipt.gates) != plan.gate_names:
        raise ValueError("validation receipt gates do not match plan")

    failures = []
    not_run = []
    for name in plan.gate_names:
        result = receipt.gates[name]
        if result.status == "PASS":
            if not result.evidence:
                raise ValueError(f"validation PASS requires evidence: {name}")
        elif result.status in {"FAIL", "NOT_RUN"}:
            if not result.reason.strip():
                raise ValueError(f"validation {result.status} requires a reason: {name}")
            if result.status == "FAIL":
                failures.append(name)
            else:
                not_run.append(name)

    expected_status = "FAIL" if failures else "NOT_RUN" if not_run else "PASS"
    if receipt.status != expected_status:
        raise ValueError(
            f"validation receipt status {receipt.status} does not match gate states {expected_status}"
        )
    if receipt.status == "PASS":
        runtime = receipt.runtime or {}
        required_runtime = {"worker_count", "max_jobs", "max_episodes", "max_attempts", "hf_subfolder"}
        missing = sorted(required_runtime - set(runtime))
        if missing:
            raise ValueError("validation PASS requires runtime evidence: " + ", ".join(missing))
        if runtime["worker_count"] > plan.worker_count:
            raise ValueError("validation receipt exceeds worker bound")
        if runtime["max_jobs"] > plan.max_jobs or runtime["max_episodes"] > plan.max_episodes:
            raise ValueError("validation receipt exceeds bounded plan")
        if not str(runtime["hf_subfolder"]).startswith(
            "validation/validation-cpu-20260922/"
        ):
            raise ValueError("validation receipt escapes isolated HF prefix")
    return receipt


def build_validation_receipt(
    plan: ValidationPlan,
    *,
    gates: Mapping[str, ValidationGateResult] | None = None,
    runtime: Mapping[str, Any] | None = None,
    now_s: float | None = None,
) -> ValidationReceipt:
    timestamp = time.time() if now_s is None else float(now_s)
    gate_results = dict(gates or {
        name: ValidationGateResult(
            status="NOT_RUN",
            reason="validation execution was not run",
            evidence={},
        )
        for name in plan.gate_names
    })
    statuses = {result.status for result in gate_results.values()}
    status = "FAIL" if "FAIL" in statuses else "NOT_RUN" if "NOT_RUN" in statuses else "PASS"
    receipt = ValidationReceipt(
        plan_id=plan.plan_id,
        status=status,
        gates=gate_results,
        created_at_s=timestamp,
        updated_at_s=timestamp,
        runtime=dict(runtime or {}),
    )
    return validate_validation_receipt(receipt, plan)


def write_validation_receipt(
    path: str | Path,
    receipt: ValidationReceipt,
    plan: ValidationPlan,
) -> Path:
    validate_validation_receipt(receipt, plan)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".partial-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(
        json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"result inventory entry must be a regular file: {path}")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _actual_files(root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"result inventory contains symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"result inventory entry is not a regular file: {relative}")
        actual[relative] = _sha256(path)
    return actual


def _safe_file_inventory(root: Path, *, allowed_root: Path) -> dict[str, dict[str, Any]]:
    try:
        root_info = root.lstat()
    except OSError:
        return {}
    if stat.S_ISLNK(root_info.st_mode):
        return {".": {"kind": "symlink", "target": os.readlink(root)}}
    if not stat.S_ISDIR(root_info.st_mode):
        return {".": {"kind": "not_directory"}}

    allowed = allowed_root.resolve()
    resolved_root = root.resolve()
    if not resolved_root.is_relative_to(allowed):
        return {".": {"kind": "outside_allowed_root", "path": str(root)}}

    inventory: dict[str, dict[str, Any]] = {}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
            except OSError as exc:
                directories.remove(name)
                inventory[relative] = {"kind": "unreadable", "error": str(exc)}
                continue
            if stat.S_ISLNK(info.st_mode):
                directories.remove(name)
                inventory[relative] = {"kind": "symlink", "target": os.readlink(path)}

        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    inventory[relative] = {"kind": "symlink", "target": os.readlink(path)}
                    continue
                if not stat.S_ISREG(info.st_mode):
                    inventory[relative] = {"kind": "not_regular_file"}
                    continue
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags)
                digest = hashlib.sha256()
                with os.fdopen(descriptor, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode):
                        inventory[relative] = {"kind": "not_regular_file"}
                        continue
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(chunk)
                inventory[relative] = {
                    "kind": "file",
                    "bytes": opened.st_size,
                    "sha256": digest.hexdigest(),
                }
            except OSError as exc:
                inventory[relative] = {"kind": "unreadable", "error": str(exc)}
    return inventory


def _verify_identity(job: GenerationJob, result: WorkerResult) -> None:
    if result.job_id != job.job_id or result.attempt_id != job.attempt_id:
        raise ValueError("result job/attempt identity mismatch")
    if result.program_id != job.program_id:
        raise ValueError("result program identity mismatch")
    expected_episode = job.episode_id if result.outcome in {"success", "valid_failure"} else None
    if result.episode_id != expected_episode:
        raise ValueError("result episode identity mismatch")


def _validate_artifact_manifest(root: Path, result: WorkerResult) -> None:
    path = root / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError("artifact manifest is required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "attempt_id": result.attempt_id,
        "episode_id": result.episode_id,
        "program_id": result.program_id,
        "outcome": result.outcome,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"artifact manifest identity mismatch for {key}")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("artifact manifest files must be an object")
    actual = _actual_files(root)
    actual.pop("artifact_manifest.json", None)
    if set(files) != set(actual):
        raise ValueError("artifact manifest file inventory mismatch")
    for relative, info in files.items():
        if not isinstance(info, dict) or info.get("sha256") != actual[relative]:
            raise ValueError(f"artifact manifest checksum mismatch: {relative}")


def validate_closed_result(job: GenerationJob, result: WorkerResult) -> ValidatedResult:
    _verify_identity(job, result)
    root = Path(result.result_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("result_dir must be a real directory")
    actual = _actual_files(root)
    if set(actual) != set(result.file_sha256):
        missing = sorted(set(result.file_sha256) - set(actual))
        extra = sorted(set(actual) - set(result.file_sha256))
        raise ValueError(f"result file inventory mismatch: missing={missing}, extra={extra}")
    for relative, digest in result.file_sha256.items():
        if actual[relative] != digest:
            raise ValueError(f"checksum mismatch: {relative}")

    if result.outcome in {"success", "valid_failure"}:
        episode_path = root / "episode.json"
        layout_path = root / "layout"
        if not (layout_path / "layout_manifest.json").is_file():
            raise ValueError("episode training layout is required")
        from icgs.data.training_layout import validate_training_episode_layout
        validate_training_episode_layout(layout_path)
        _validate_artifact_manifest(root, result)
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        validate_episode(episode)
        provenance = dict(episode["provenance"])
        expected = {
            "attempt_id": job.attempt_id,
            "episode_id": job.episode_id,
            "program_id": job.program_id,
            "outcome": result.outcome,
            "scene_signature": job.plan.randomization.get("scene_signature"),
            "scene_seed": job.plan.scene_seed,
            "episode_index": job.plan.episode_index,
            "episode_kind": job.plan.episode_kind,
        }
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f"episode provenance mismatch for {key}")
        transitions = len(episode["transitions"])
        observations = len(episode["online_observations"])
        durations = len(episode["dt"])
        timeline = {"actions": transitions, "observations": observations, "durations": durations}
        if result.timeline != timeline:
            raise ValueError("result timeline differs from episode")
        entry = {
            "attempt_id": job.attempt_id,
            "episode_id": job.episode_id,
            "program_id": job.program_id,
            "split": provenance.get("split"),
            "subset": provenance.get("subset") or provenance.get("train_subset"),
            "source_lineage_id": provenance.get("source_lineage_id"),
            "asset_family_id": provenance.get("asset_family_id"),
            "asset_instance_id": provenance.get("asset_instance_id"),
            "scene_signature": provenance.get("scene_signature"),
            "scene_seed": provenance.get("scene_seed"),
            "episode_index": provenance.get("episode_index"),
            "episode_kind": provenance.get("episode_kind"),
            "outcome": result.outcome,
            "intervention_id": provenance.get("intervention_id"),
            "source_episode_id": provenance.get("source_episode_id"),
            "base_episode_id": provenance.get("base_episode_id"),
            "result_dir": str(root),
            "file_sha256": dict(result.file_sha256),
            "attempt_plan": job.plan.as_dict(),
        }
        return ValidatedResult(job, result, provenance, entry, None)

    _validate_artifact_manifest(root, result)
    attempt_path = root / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt.get("attempt_id") != job.attempt_id or attempt.get("episode_id") is not None:
        raise ValueError("attempt identity mismatch")
    if attempt.get("program_id") != job.program_id or attempt.get("outcome") != result.outcome:
        raise ValueError("attempt classification mismatch")
    entry = {
        **attempt,
        "result_dir": str(root),
        "file_sha256": dict(result.file_sha256),
        "scene_signature": job.plan.randomization.get("scene_signature"),
        "scene_seed": job.plan.scene_seed,
        "episode_index": job.plan.episode_index,
        "attempt_plan": job.plan.as_dict(),
    }
    return ValidatedResult(job, result, entry, None, entry)


def _same_row(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def ingest_validated_result(
    manifest: Mapping[str, Any],
    validated: ValidatedResult,
) -> dict[str, Any]:
    updated = deepcopy(dict(manifest))
    episodes = list(updated.get("episodes") or [])
    attempts = list(updated.get("failure_attempts") or [])
    if validated.episode_entry is not None:
        candidate = validated.episode_entry
        for row in episodes:
            if row.get("episode_id") == candidate["episode_id"]:
                if _same_row(row, candidate):
                    return updated
                raise ValueError(f"immutable episode conflict: {candidate['episode_id']}")
        if any(row.get("attempt_id") == candidate["attempt_id"] for row in attempts):
            raise ValueError(f"attempt already recorded as non-episode: {candidate['attempt_id']}")
        proposed = episodes + [candidate]
        leakage = split_disjointness_report(proposed)
        if not leakage["disjoint"]:
            raise ValueError(
                "split leakage: "
                + json.dumps(
                    {
                        "overlaps": leakage["overlaps"],
                        "source_episode_leaks": leakage["source_episode_leaks"],
                        "train_test_asset_family_overlap": leakage["train_test_asset_family_overlap"],
                    },
                    sort_keys=True,
                )
            )
        episodes.append(candidate)
    else:
        candidate = validated.attempt_entry
        assert candidate is not None
        for row in attempts:
            if row.get("attempt_id") == candidate["attempt_id"]:
                if _same_row(row, candidate):
                    return updated
                raise ValueError(f"immutable attempt conflict: {candidate['attempt_id']}")
        if any(row.get("attempt_id") == candidate["attempt_id"] for row in episodes):
            raise ValueError(f"attempt already recorded as episode: {candidate['attempt_id']}")
        attempts.append(candidate)
    updated["episodes"] = episodes
    updated["failure_attempts"] = attempts
    updated["total_episodes"] = len(episodes)
    updated["total_failure_attempts"] = len(attempts)
    return updated


__all__ = [
    "ValidatedResult",
    "ValidationFailure",
    "ValidationGateResult",
    "ValidationPlan",
    "ValidationReceipt",
    "build_validation_receipt",
    "ingest_validated_result",
    "load_validation_plan",
    "validate_validation_receipt",
    "validate_closed_result",
    "write_validation_receipt",
]
