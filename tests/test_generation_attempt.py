from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig, GenerationJob
from icgs.data.collection.generation.episode_archive import EpisodeArchiveReader, validate_archive_manifest
from icgs.data.collection.generation.rlbench_attempt import (
    RawAttempt,
    materialize_raw_attempt,
    write_closed_attempt_result,
    online_observation_view,
)


@dataclass
class FakeObservation:
    wrist_point_cloud: np.ndarray
    gripper_pose: np.ndarray
    gripper_open: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    front_rgb: np.ndarray
    wrist_depth: np.ndarray
    wrist_mask: np.ndarray
    front_mask: np.ndarray


def _observation(index: int) -> FakeObservation:
    return FakeObservation(
        wrist_point_cloud=np.asarray([[index, 0.0, 0.8], [index, 0.1, 0.8]], dtype=np.float32),
        gripper_pose=np.asarray([index * 0.01, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        gripper_open=1.0 if index < 2 else 0.0,
        joint_positions=np.full((7,), index, dtype=np.float64),
        joint_velocities=np.full((7,), index / 10.0, dtype=np.float64),
        front_rgb=np.full((2, 2, 3), index, dtype=np.uint8),
        wrist_depth=np.full((2, 2), index / 100.0, dtype=np.float32),
        wrist_mask=np.full((2, 2), index, dtype=np.uint8),
        front_mask=np.full((2, 2), index + 1, dtype=np.uint8),
    )


def _job() -> GenerationJob:
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-000001", episode_kind="nominal",
        scene_seed=123, collection_seed=20260920,
        randomization={"scene_signature": "sig-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    return GenerationJob.create(
        job_id="job-1", run_id="run-1", attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root="/content/run/staging",
    )


def _binding() -> dict:
    return {
        "program_id": "T01",
        "split": "train",
        "asset_family_id": "asset-family-1",
        "source_lineage_id": "lineage-1",
        "asset_version": "v1",
        "randomization": {
            "translation_m": {"x": [-0.012, 0.012], "y": [-0.012, 0.012], "z": [0.0, 0.0]},
            "yaw_deg": [-30.0, 30.0], "scale": [0.8, 1.2],
            "camera_profile_id": "rlbench-wrist-depth-v1",
        },
        "events": [{"primitive": "grasp"}],
    }


def _raw(*, predicates_ok: bool, count: int = 5, simulator_crash: bool = False) -> RawAttempt:
    observations = tuple(_observation(index) for index in range(count))
    actions = tuple(
        np.asarray([index * 0.01, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        for index in range(max(0, count - 1))
    )
    return RawAttempt(
        observations=observations,
        actions=actions,
        scene_states=tuple({"boundary": index, "objects": []} for index in range(count)),
        collision_events=(),
        sim_time_s=max(0, count - 1) * 0.05,
        predicates_ok=predicates_ok,
        terminal_reason="predicate_satisfied" if predicates_ok else "predicate_failed",
        simulator_crash=simulator_crash,
    )


def test_valid_failure_materializes_episode_and_keeps_t_plus_one():
    materialized = materialize_raw_attempt(_raw(predicates_ok=False), _job(), _binding())
    assert materialized.outcome == "valid_failure"
    assert materialized.episode_record is not None
    assert materialized.episode_record["provenance"]["episode_id"] == _job().plan.episode_id
    assert len(materialized.episode_record["online_observations"]) == 5
    assert len(materialized.episode_record["transitions"]) == 4
    assert len(materialized.episode_record["dt"]) == 4


def test_invalid_observation_is_attempt_with_null_episode_id():
    materialized = materialize_raw_attempt(_raw(predicates_ok=False, count=1), _job(), _binding())
    assert materialized.outcome == "invalid_observation"
    assert materialized.episode_record is None
    assert materialized.attempt_record is not None
    assert materialized.attempt_record["episode_id"] is None
    assert materialized.attempt_record["failure_type"] is None


def test_simulator_crash_remains_attempt_even_with_valid_prefix():
    materialized = materialize_raw_attempt(
        _raw(predicates_ok=False, count=5, simulator_crash=True), _job(), _binding()
    )
    assert materialized.outcome == "simulator_crash"
    assert materialized.episode_record is None
    assert materialized.attempt_record["episode_id"] is None
    assert materialized.attempt_record["failure_type"] == "simulator_exception"


def test_attempt_record_keeps_failure_cause_and_prefix_boundaries():
    from icgs.data.collection.generation.episode_record import assemble_attempt_record

    record = assemble_attempt_record(
        attempt_id="att-1", program_id="T01", outcome="invalid_observation",
        error="bad point cloud", failure_type="observation_schema",
        terminal_t=2, valid_observation_until=1,
    )
    assert record["failure_type"] == "observation_schema"
    assert record["terminal_t"] == 2
    assert record["valid_observation_until"] == 1


def test_closed_valid_failure_writes_full_layout_and_hashes(tmp_path: Path):
    materialized = materialize_raw_attempt(_raw(predicates_ok=False), _job(), _binding())
    result = write_closed_attempt_result(materialized, _job(), tmp_path / "result")
    assert result.outcome == "valid_failure"
    assert result.timeline == {"actions": 4, "observations": 5, "durations": 4}
    assert (tmp_path / "result" / "episode.json").is_file()
    assert (tmp_path / "result" / "layout" / "layout_manifest.json").is_file()
    assert set(result.file_sha256) == {
        str(path.relative_to(tmp_path / "result"))
        for path in (tmp_path / "result").rglob("*") if path.is_file()
    }
    episode = json.loads((tmp_path / "result" / "episode.json").read_text())
    assert episode["provenance"]["outcome"] == "valid_failure"


def _enable_archive_profile(monkeypatch):
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(profile.as_dict()))


def test_archive_profile_writes_valid_failure_as_canonical_episode_archive(tmp_path: Path, monkeypatch):
    _enable_archive_profile(monkeypatch)
    materialized = materialize_raw_attempt(_raw(predicates_ok=False), _job(), _binding())

    result = write_closed_attempt_result(materialized, _job(), tmp_path / "archive-result")
    root = Path(result.result_dir)
    manifest_path = root / "episode.manifest.json"

    assert result.outcome == "valid_failure"
    assert manifest_path.is_file()
    assert not (root / "episode.json").exists()
    assert not (root / "layout").exists()
    assert not (root / "telemetry.npz").exists()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    assert EpisodeArchiveReader(manifest_path).manifest.payload["outcome"] == "valid_failure"
    debug = json.loads((root / "debug.json").read_text(encoding="utf-8"))
    assert "online_observations" not in debug
    assert "points" not in json.dumps(debug)


def test_archive_profile_preserves_measured_action_dtype(tmp_path: Path, monkeypatch):
    _enable_archive_profile(monkeypatch)
    raw = _raw(predicates_ok=False)
    raw = RawAttempt(
        observations=raw.observations,
        actions=tuple(np.asarray(action, dtype=np.float32) for action in raw.actions),
        scene_states=raw.scene_states,
        collision_events=raw.collision_events,
        sim_time_s=raw.sim_time_s,
        predicates_ok=False,
        terminal_reason=raw.terminal_reason,
    )
    materialized = materialize_raw_attempt(raw, _job(), _binding())
    result = write_closed_attempt_result(materialized, _job(), tmp_path / "dtype-archive")
    reader = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")

    assert reader.raw_arrays["actions"].dtype == np.dtype(np.float32)


@pytest.mark.parametrize("simulator_crash", [False, True])
def test_archive_profile_routes_closed_crash_or_invalid_to_attempt_archive(
    tmp_path: Path, monkeypatch, simulator_crash: bool
):
    _enable_archive_profile(monkeypatch)
    materialized = materialize_raw_attempt(
        _raw(predicates_ok=False, count=3 if simulator_crash else 1, simulator_crash=simulator_crash),
        _job(),
        _binding(),
    )

    result = write_closed_attempt_result(materialized, _job(), tmp_path / f"attempt-{simulator_crash}")
    root = Path(result.result_dir)
    manifest_path = root / "attempt.manifest.json"

    assert result.episode_id is None
    assert result.outcome in {"simulator_crash", "invalid_observation"}
    assert manifest_path.is_file()
    assert not (root / "attempt.json").exists()
    assert not (root / "valid_prefix.npz").exists()
    assert validate_archive_manifest(manifest_path)["valid"] is True


def test_archive_profile_omits_unavailable_optional_modalities(tmp_path: Path, monkeypatch):
    _enable_archive_profile(monkeypatch)
    observations = tuple(
        SimpleNamespace(
            wrist_point_cloud=np.asarray([[index, 0.0, 0.8]], dtype=np.float32),
            gripper_pose=np.asarray([index * 0.01, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            gripper_open=1.0,
        )
        for index in range(3)
    )
    raw = RawAttempt(
        observations=observations,
        actions=tuple(np.asarray([0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64) for _ in range(2)),
        scene_states=(),
        collision_events=(),
        sim_time_s=0.1,
        predicates_ok=True,
        terminal_reason="predicate_satisfied",
    )

    materialized = materialize_raw_attempt(raw, _job(), _binding())
    assert materialized.episode_record is not None
    result = write_closed_attempt_result(materialized, _job(), tmp_path / "minimal-archive")
    archive = EpisodeArchiveReader(Path(result.result_dir) / "episode.manifest.json")
    roles = {
        spec["semantic_role"]
        for spec in archive.manifest.payload["array_specs"].values()
    }
    assert not any(role.startswith("raw_arrays/front_rgb") for role in roles)
    assert not any(role.startswith("raw_arrays/wrist_depth") for role in roles)
    assert not any(role.startswith("raw_arrays/wrist_mask") for role in roles)
    assert not any(role.startswith("raw_arrays/front_mask") for role in roles)
    assert not any("joint_positions" in role or "joint_velocities" in role for role in roles)
    assert "object_states" not in archive.manifest.payload["record_metadata"]


def test_materializer_writes_measured_task_labels_when_steps_are_bound(tmp_path: Path):
    raw = _raw(predicates_ok=True)
    states = tuple(
        {"objects": [
            {"name": "object_a", "position": [0.2 if index else 0.0, 0.0, 0.0]},
            {"name": "target_a", "position": [0.2, 0.0, 0.0]},
        ]}
        for index in range(5)
    )
    raw = RawAttempt(
        observations=raw.observations, actions=raw.actions, scene_states=states,
        collision_events=raw.collision_events, sim_time_s=raw.sim_time_s,
        predicates_ok=True, terminal_reason=raw.terminal_reason,
    )
    binding = {
        **_binding(),
        "structured_steps": [{
            "object_role": "object_a", "target_role": "target_a",
            "postcondition": "object_a_placed", "precondition": None,
        }],
    }
    materialized = materialize_raw_attempt(raw, _job(), binding)
    assert materialized.episode_record["nu_valid"].all()
    assert materialized.episode_record["rho"][1][0] == 1.0


def test_pilot_writer_strips_robot_telemetry_from_online_observation_whitelist(tmp_path: Path):
    observation = {
            "points": np.ones((2, 3), dtype=np.float32),
            "point_valid": np.ones(2, dtype=bool),
            "T_w_e": np.eye(4),
            "grip": 0,
            "joint_positions": np.zeros(7),
            "joint_velocities": np.zeros(7),
    }
    public = online_observation_view(observation)
    assert set(public) == {"points", "point_valid", "T_w_e", "grip"}
    assert "joint_positions" not in public
