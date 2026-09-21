import json
from pathlib import Path

import numpy as np
import pytest


def _payload(success=True):
    boundaries = 3
    return {
        "episode": {
            "episode_id": "ep_000001",
            "program_id": "T01",
            "split": "train",
            "seed_id": "seed-1",
            "layout_version": 2,
        },
        "observations": {
            "rgb": np.zeros((boundaries, 4, 5, 3), dtype=np.uint8),
            "depth": np.ones((boundaries, 4, 5), dtype=np.float32),
            "masks": np.zeros((boundaries, 4, 5), dtype=np.uint16),
            "pointcloud": [np.zeros((2 + i, 3), dtype=np.float32) for i in range(boundaries)],
        },
        "robot": {
            "ee_pose": np.repeat(np.eye(4)[None], boundaries, axis=0),
            "joint_state": np.zeros((boundaries, 14), dtype=np.float64),
            "gripper": np.ones((boundaries,), dtype=np.float32),
        },
        "actions": np.zeros((boundaries - 1, 8), dtype=np.float64),
        "scene": {
            "object_ids": ["object_a", "target_a"],
            "position": np.zeros((boundaries, 2, 3), dtype=np.float64),
            "orientation_xyzw": np.zeros((boundaries, 2, 4), dtype=np.float64),
            "linear_velocity": np.zeros((boundaries, 2, 3), dtype=np.float64),
            "angular_velocity": np.zeros((boundaries, 2, 3), dtype=np.float64),
            "state_valid": np.ones((boundaries, 2), dtype=bool),
        },
        "task": {
            "events": [{"event_id": "event_0", "primitive": "lift"}],
            "rho": np.array([[0], [0], [1]], dtype=bool),
            "nu": np.array([[0], [0], [1]], dtype=bool),
            "epsilon": np.array([[1], [1], [0]], dtype=bool),
            "rho_valid": np.ones((boundaries, 1), dtype=bool),
            "nu_valid": np.ones((boundaries, 1), dtype=bool),
            "epsilon_valid": np.ones((boundaries, 1), dtype=bool),
            "collisions": [{"boundary": 1, "a": "gripper", "b": "object_a"}],
        },
        "result": {"success": success, "terminal_reason": "predicate_satisfied" if success else "predicate_failed"},
        "snapshot": {
            "quality": "approximate_replay",
            "anchor": {"joint_positions": np.zeros(7), "boundary": 0},
            "successor": {"joint_positions": np.ones(7), "boundary": 2},
        },
    }


def test_initializes_root_metadata_and_program_files(tmp_path):
    from icgs.data.training_layout import initialize_dataset_layout

    approved = {
        "manifest_version": 2,
        "protocol_id": "p",
        "catalog": [{
            "program_id": "T01", "split": "train", "asset_family_id": "a",
            "controller": {"protocol_id": "c"}, "predicates": {"protocol_id": "q"},
        }],
    }
    initialize_dataset_layout(tmp_path, approved)
    assert (tmp_path / "metadata/dataset.yaml").is_file()
    assert (tmp_path / "metadata/programs.json").is_file()
    assert (tmp_path / "programs/train/program_001.json").is_file()
    assert json.loads((tmp_path / "metadata/splits.json").read_text())["T01"] == "train"


def test_writes_and_validates_complete_success_episode(tmp_path):
    from icgs.data.training_layout import validate_training_episode_layout, write_training_episode_layout

    episode_dir = tmp_path / "episodes/ep_000001"
    inventory = write_training_episode_layout(episode_dir, _payload())
    assert (episode_dir / "observations/rgb/frames.npz").is_file()
    assert (episode_dir / "observations/depth/frames.npz").is_file()
    assert (episode_dir / "observations/masks/frames.npz").is_file()
    assert (episode_dir / "observations/pointcloud/frames.npz").is_file()
    assert (episode_dir / "scene/object_state.npz").is_file()
    assert (episode_dir / "task/rho.npy").is_file()
    assert (episode_dir / "result.json").is_file()
    assert inventory["layout_version"] == 2
    assert validate_training_episode_layout(episode_dir)["success"] is True


def test_failure_uses_same_layout_but_is_not_success(tmp_path):
    from icgs.data.training_layout import validate_training_episode_layout, write_training_episode_layout

    episode_dir = tmp_path / "failure_attempts/ep_000001"
    write_training_episode_layout(episode_dir, _payload(success=False))
    assert validate_training_episode_layout(episode_dir)["success"] is False


def test_rejects_misaligned_boundaries(tmp_path):
    from icgs.data.training_layout import write_training_episode_layout

    payload = _payload()
    payload["actions"] = np.zeros((3, 8))
    with pytest.raises(ValueError, match="actions must have boundaries - 1"):
        write_training_episode_layout(tmp_path / "episodes/ep_000001", payload)


def test_rejects_exact_snapshot_without_fidelity_evidence(tmp_path):
    from icgs.data.training_layout import write_training_episode_layout

    payload = _payload()
    payload["snapshot"]["quality"] = "exact"
    with pytest.raises(ValueError, match="exact snapshot requires fidelity evidence"):
        write_training_episode_layout(tmp_path / "episodes/ep_000001", payload)


def test_cache_manifest_requires_all_source_hashes(tmp_path):
    from icgs.data.training_layout import write_cache_manifest

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        write_cache_manifest(tmp_path, {"model_id": "m", "source_archive_sha256": "a"})
