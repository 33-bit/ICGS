"""Synthetic ``icgs_stage1_episode_v2`` records for the Stage-1 dataset tests (no simulator).

Not a test module: imported by ``test_stage1_robohiman.py`` and
``test_stage1_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.labels import derive_monitor_states
from icgs.data.stage1.schema import PHYSICS_TRANSITION, SCHEMA_VERSION, perturbation_families

SPLIT_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts/robohiman/icgs_robohiman_stage1_split_v1.json"
DT = 0.05
PREDICATES = ["handle_grasped", "drawer_open"]
EVENTS = [
    {"event_id": "handle_grasped", "relation": "handle_grasped", "prerequisites": [], "current_requirements": [],
     "hold_steps": 1, "count_only_when_eligible": False},
    {"event_id": "drawer_opened", "relation": "drawer_open", "prerequisites": ["handle_grasped"],
     "current_requirements": [], "hold_steps": 1, "count_only_when_eligible": False},
]
LEGEND = {
    "0": {"object": None, "instance": None, "role": "background", "category": "background"},
    "17": {"object": "drawer_frame", "instance": "drawer_frame", "role": "task_fixture", "category": "drawer_frame"},
    "23": {"object": "Panda_arm", "instance": "Panda_link7_visual", "role": "robot", "category": "Panda_link_visual"},
}


def seed_rng_array(seed: int) -> np.ndarray:
    _, keys, pos, has_gauss, _ = np.random.RandomState(seed).get_state()
    return np.concatenate([np.asarray(keys, dtype=np.uint32), np.array([pos, has_gauss], dtype=np.uint32)])


def trace(kind: str, steps: int) -> np.ndarray:
    """Predicate trace with S+1 rows: 'success' grasps at 3 and opens at 7; 'failure' only grasps."""
    rows = np.zeros((steps + 1, len(PREDICATES)), dtype=bool)
    rows[3:, 0] = True
    if kind == "success":
        rows[7:, 1] = True
    return rows


def make_episode(episode_id: str, *, provenance: dict[str, Any] | None, seed: int, task: str = "open_drawer",
                 status: str = "success", steps: int = 12, strategy: tuple[int, str] = (0, "no_variations"),
                 env_seed: int = 42, masks: bool = True, parent: str | None = None,
                 execution: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from icgs.data.stage1.dataset import seed_state_sha256

    predicates = trace(status, steps)
    frame_step = np.arange(0, steps + 1, 2)
    commanded = np.linspace(0.0, 1.0, steps)[:, None] * np.ones((1, 7))
    achieved = np.vstack([np.zeros((1, 7)), commanded - 0.01])  # the arm lags its command
    arrays: dict[str, np.ndarray] = {
        "step_sim_time": np.arange(steps + 1) * DT,
        "step_joint_positions": achieved,
        "step_joint_velocities": np.zeros((steps + 1, 7)),
        "step_tip_pose": np.tile([0, 0, 1, 0, 0, 0, 1.0], (steps + 1, 1)),
        "step_gripper_joint_positions": np.zeros((steps + 1, 2)),
        "step_predicates": predicates,
        "step_task_success": predicates[:, 1].copy() if status == "success" else np.zeros(steps + 1, bool),
        "cmd_arm_joint_target": commanded,
        "cmd_arm_joint_target_valid": np.ones(steps, dtype=bool),
        "cmd_arm_joint_velocity_valid": np.zeros(steps, dtype=bool),
        "cmd_gripper_joint_velocity": np.zeros((steps, 2)),
        "cmd_gripper_joint_velocity_valid": np.zeros((steps, 2), dtype=bool),
        "cmd_grasp_event": np.zeros(steps, dtype=np.int8),
        "cmd_arm_teleport_calls": np.zeros(steps, dtype=np.int16),
        "cmd_phase": np.ones(steps, dtype=np.int8),
        "prof_step_wall_s": np.full(steps, 0.013),
        "frame_step": frame_step,
        "rng_state_mt19937": seed_rng_array(seed),
    }
    for key, value in derive_monitor_states(predicates, PREDICATES, EVENTS).items():
        arrays[f"monitor_{key}"] = value
    if masks:
        handles = np.zeros((frame_step.shape[0], 4, 4), dtype=np.int32)
        handles[:, :2, :2] = 17
        handles[:, 3, 3] = 23
        arrays["cam_front_mask_handles"] = handles
        arrays["cam_front_depth"] = np.full((frame_step.shape[0], 4, 4), 0.5, dtype=np.float32)
    perturbation = {"benchmark_factors": [], "execution": list(execution or [])}
    perturbation["families"] = list(perturbation_families(perturbation))
    lineage: dict[str, Any] = {
        "collection_run_id": episode_id.split("-")[-2], "episode_index": int(episode_id.split("-")[-1]),
        "parent_episode_id": parent,
        "reset_rng": {"kind": "numpy MT19937 before TaskEnvironment.reset", "sha256": seed_state_sha256(seed),
                      "array": "rng_state_mt19937"},
    }
    if provenance is not None:
        lineage["split"] = dict(provenance)
    cameras: dict[str, Any] = {"names": ["front"], "masks_recorded": masks, "frame_stride": 2}
    if masks:
        cameras["mask_legend"] = {handle: dict(entry) for handle, entry in LEGEND.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": {"benchmark": "robohiman", "benchmark_revision": "33f71d3", "task": task,
                   "task_family": "atomic", "level": "A", "variation_index": 0,
                   "collection_strategy": {"index": strategy[0], "name": strategy[1]}},
        "perturbation": perturbation,
        "outcome": {"status": status, "recoverability": "unknown", "task_success_final": status == "success",
                    "reason": status},
        "lineage": lineage,
        "environment": {"env_seed": env_seed},
        "controller": {"physics_dt": DT, "native_command": "joint targets"},
        "timing": {"physics_dt_s": DT, "transition": PHYSICS_TRANSITION, "frame_stride": 2, "model_dt_s": 2 * DT},
        "code": {}, "cameras": cameras,
        "predicates": {"names": list(PREDICATES)},
        "events": {"specs": [dict(spec) for spec in EVENTS]},
        "arrays": {"sha256": "pending"},
        "counts": {"steps": steps, "frames": int(frame_step.shape[0])},
    }
    return manifest, arrays
