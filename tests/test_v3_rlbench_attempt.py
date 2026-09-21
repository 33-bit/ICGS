from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import inspect

import numpy as np

from icgs.data.collection.v3.batch import AttemptPlan
from icgs.data.collection.v3.distributed_contracts import GenerationJob
from icgs.data.collection.v3.rlbench_attempt import (
    RawAttempt,
    materialize_raw_attempt,
    write_closed_attempt_result,
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
        episode_id="icgs-primary-v3-t01-000001", episode_kind="nominal",
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


def test_simulator_crash_remains_attempt_even_with_valid_prefix():
    materialized = materialize_raw_attempt(
        _raw(predicates_ok=False, count=5, simulator_crash=True), _job(), _binding()
    )
    assert materialized.outcome == "simulator_crash"
    assert materialized.episode_record is None
    assert materialized.attempt_record["episode_id"] is None


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


def test_legacy_collector_exposes_opt_in_valid_failure_capture():
    from scripts.colab_g2_dataset_generator import collect_single_episode

    parameter = inspect.signature(collect_single_episode).parameters["allow_valid_failure"]
    assert parameter.default is False


def test_shared_raw_executor_has_exact_public_signature_and_wrapper_dependency():
    from scripts.colab_g2_dataset_generator import execute_raw_attempt

    signature = inspect.signature(execute_raw_attempt)
    assert list(signature.parameters) == ["task", "env", "spec"]
    assert "execute_raw_attempt" in Path("scripts/colab_g2_dataset_generator.py").read_text(encoding="utf-8")
