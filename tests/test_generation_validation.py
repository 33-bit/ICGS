from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import GenerationJob
from icgs.data.collection.generation.distributed_validation import (
    ingest_validated_result,
    validate_closed_result,
)
from icgs.data.collection.generation import distributed_validation as validation
from icgs.data.collection.generation.rlbench_attempt import RawAttempt, materialize_raw_attempt, write_closed_attempt_result


@dataclass
class Obs:
    wrist_point_cloud: np.ndarray
    gripper_pose: np.ndarray
    gripper_open: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    front_rgb: np.ndarray
    wrist_depth: np.ndarray
    wrist_mask: np.ndarray
    front_mask: np.ndarray


def _job(index: int = 1, split: str = "train") -> GenerationJob:
    plan = AttemptPlan(
        program_id="T01", split=split, episode_index=index,
        episode_id=f"episode-t01-{index:05d}", episode_kind="nominal",
        scene_seed=100 + index, collection_seed=20260920,
        randomization={"scene_signature": f"sig-{index}", "asset_instance_id": f"asset-{index}"},
        intervention=None,
    )
    return GenerationJob.create(
        job_id=f"job-{index}", run_id="run-1", attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root="/content/run/staging",
    )


def _binding() -> dict:
    return {
        "program_id": "T01", "split": "train", "asset_family_id": "family-1",
        "source_lineage_id": "lineage-1", "asset_version": "v1",
        "randomization": {
            "translation_m": {"x": [-0.012, 0.012], "y": [-0.012, 0.012], "z": [0.0, 0.0]},
            "yaw_deg": [-30.0, 30.0], "scale": [0.8, 1.2],
            "camera_profile_id": "rlbench-wrist-depth-v1",
        },
    }


def _raw(success: bool) -> RawAttempt:
    observations = []
    for index in range(3):
        observations.append(Obs(
            wrist_point_cloud=np.asarray([[0.1 + index, 0.0, 0.8]], dtype=np.float32),
            gripper_pose=np.asarray([0.1 + index * 0.01, 0, 0.8, 0, 0, 0, 1], dtype=np.float64),
            gripper_open=1.0, joint_positions=np.zeros(7), joint_velocities=np.zeros(7),
            front_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            wrist_depth=np.zeros((2, 2), dtype=np.float32),
            wrist_mask=np.zeros((2, 2), dtype=np.uint8),
            front_mask=np.zeros((2, 2), dtype=np.uint8),
        ))
    actions = tuple(np.asarray([0.1, 0, 0.8, 0, 0, 0, 1, 1], dtype=np.float64) for _ in range(2))
    return RawAttempt(
        observations=tuple(observations), actions=actions,
        scene_states=({}, {}, {}), collision_events=(), sim_time_s=0.1,
        predicates_ok=success,
        terminal_reason="predicate_satisfied" if success else "predicate_failed",
    )


def _closed(tmp_path: Path, outcome: str, index: int = 1):
    job = _job(index)
    materialized = materialize_raw_attempt(_raw(outcome == "success"), job, _binding())
    result = write_closed_attempt_result(materialized, job, tmp_path / f"result-{index}")
    return job, result


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_episode_outcomes_require_complete_timeline_and_hashes(tmp_path: Path, outcome: str):
    job, result = _closed(tmp_path, outcome)
    validated = validate_closed_result(job, result)
    assert validated.outcome == outcome
    assert validated.episode_entry is not None
    assert validated.episode_entry["outcome"] == outcome
    assert validated.episode_entry["attempt_plan"] == job.plan.as_dict()


def test_hash_mismatch_is_rejected_without_mutating_files(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    episode = Path(result.result_dir) / "episode.json"
    before = episode.read_bytes()
    episode.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_closed_result(job, result)
    assert episode.read_bytes() == before + b"\n"


def test_ingest_is_immutable_and_idempotent_for_equal_episode(tmp_path: Path):
    job, result = _closed(tmp_path, "valid_failure")
    validated = validate_closed_result(job, result)
    original = {"manifest_version": 3, "episodes": [], "failure_attempts": []}
    original_bytes = json.dumps(original, sort_keys=True)
    updated = ingest_validated_result(original, validated)
    assert json.dumps(original, sort_keys=True) == original_bytes
    assert updated["episodes"][0]["outcome"] == "valid_failure"
    repeated = ingest_validated_result(updated, validated)
    assert repeated == updated


def test_episode_without_training_layout_is_rejected(tmp_path: Path):
    import hashlib
    import shutil
    from dataclasses import replace

    job, result = _closed(tmp_path, "success")
    shutil.rmtree(Path(result.result_dir) / "layout")
    hashes = {
        str(path.relative_to(result.result_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(result.result_dir).rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="training layout"):
        validate_closed_result(job, replace(result, file_sha256=hashes))


def test_ingest_rejects_conflicting_episode_id_without_mutating_manifest(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    validated = validate_closed_result(job, result)
    manifest = {
        "manifest_version": 3,
        "episodes": [{"episode_id": job.episode_id, "program_id": "T99", "outcome": "success"}],
        "failure_attempts": [],
    }
    before = json.dumps(manifest, sort_keys=True)
    with pytest.raises(ValueError, match="immutable episode conflict"):
        ingest_validated_result(manifest, validated)
    assert json.dumps(manifest, sort_keys=True) == before


def test_validation_failure_captures_identities_trace_source_and_actual_files(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    source_path = tmp_path / "queue" / "ready" / job.job_id
    try:
        raise ValueError("malformed closed episode")
    except ValueError as exc:
        assert hasattr(validation, "ValidationFailure"), "ValidationFailure is required"
        job = replace(job, output_root=str(tmp_path))
        failure = validation.ValidationFailure.capture(
            job,
            result,
            exc,
            source_path=source_path,
            allowed_result_root=tmp_path,
        )

    diagnostic = failure.as_dict()
    episode_path = Path(result.result_dir) / "episode.json"
    assert diagnostic["job_identity"]["job_id"] == job.job_id
    assert diagnostic["result_identity"]["attempt_id"] == result.attempt_id
    assert diagnostic["result_identity"]["outcome"] == "success"
    assert diagnostic["exception_type"] == "ValueError"
    assert diagnostic["exception_message"] == "malformed closed episode"
    assert "ValueError: malformed closed episode" in diagnostic["traceback"]
    assert diagnostic["source_path"] == str(source_path)
    assert diagnostic["actual_file_inventory"]["episode.json"]["sha256"] == hashlib.sha256(
        episode_path.read_bytes()
    ).hexdigest()


def test_validation_failure_inventory_does_not_follow_symlinks(tmp_path: Path):
    from dataclasses import replace

    job, result = _closed(tmp_path, "success")
    job = replace(job, output_root=str(tmp_path))
    outside_file = tmp_path / "outside-secret.txt"
    outside_file.write_text("outside", encoding="utf-8")
    link = Path(result.result_dir) / "outside-link"
    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    try:
        raise ValueError("inventory test")
    except ValueError as exc:
        assert hasattr(validation, "ValidationFailure"), "ValidationFailure is required"
        failure = validation.ValidationFailure.capture(
            job,
            result,
            exc,
            source_path=tmp_path / "ready" / job.job_id,
            allowed_result_root=tmp_path,
        )

    assert failure.actual_file_inventory["outside-link"]["kind"] == "symlink"
    assert "bytes" not in failure.actual_file_inventory["outside-link"]


def test_closed_result_rejects_dangling_symlink_artifact(tmp_path: Path):
    job, result = _closed(tmp_path, "success")
    link = Path(result.result_dir) / "dangling-artifact"
    try:
        link.symlink_to(tmp_path / "missing-outside-file")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")

    with pytest.raises(ValueError, match="symlink"):
        validate_closed_result(job, result)
