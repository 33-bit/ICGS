"""Validate and immutably ingest closed distributed collection results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from icgs.data.collection.v3.distributed_contracts import GenerationJob, WorkerResult
from icgs.data.collection.v3.report import split_disjointness_report
from icgs.data.schemas.episodes_v2 import validate_episode_v2


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


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
        validate_episode_v2(episode)
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


__all__ = ["ValidatedResult", "ingest_validated_result", "validate_closed_result"]
