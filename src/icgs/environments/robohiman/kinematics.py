"""Panda forward kinematics for deriving canonical EE targets from joint commands.

Uses Franka's published modified-DH parameters. The CoppeliaSim model's base
frame and tip offset are *estimated* from recorded (joints, tip pose) pairs:
``T_world_tip = T_world_dh0 @ FK(q) @ T_flange_tip``. ``fit_fk_calibration``
fits both constants and reports the residual over every recorded boundary; a
canonical command is only emitted when that residual is below tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# (a_{i-1}, d_i, alpha_{i-1}) — Franka Emika Panda modified DH.
PANDA_MDH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
PANDA_FLANGE_D = 0.107
FK_ID = "panda-mdh-fitted-base-tip-v1"


def _mdh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def panda_fk(q: np.ndarray) -> np.ndarray:
    """Flange pose in the DH base frame for one joint vector [7]."""
    transform = np.eye(4)
    for (a, d, alpha), theta in zip(PANDA_MDH, np.asarray(q, dtype=np.float64)):
        transform = transform @ _mdh(a, d, alpha, theta)
    flange = np.eye(4)
    flange[2, 3] = PANDA_FLANGE_D
    return transform @ flange


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()])


def _average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    result[:3, :3] = Rotation.from_matrix(np.stack([t[:3, :3] for t in transforms])).mean().as_matrix()
    return result


def fit_fk_calibration(joints: np.ndarray, tip_poses: np.ndarray, base_pose: np.ndarray,
                       iterations: int = 4) -> dict[str, Any]:
    """Fit T_world_dh0 (init: arm base pose) and T_flange_tip by alternating averages."""
    flanges = [panda_fk(q) for q in joints]
    tips = [pose7_to_matrix(p) for p in tip_poses]
    world_dh0 = pose7_to_matrix(base_pose)
    flange_tip = np.eye(4)
    for _ in range(iterations):
        flange_tip = _average_transform([np.linalg.inv(world_dh0 @ f) @ t for f, t in zip(flanges, tips)])
        world_dh0 = _average_transform([t @ np.linalg.inv(f @ flange_tip) for f, t in zip(flanges, tips)])
    predicted = [world_dh0 @ f @ flange_tip for f in flanges]
    position = np.array([np.linalg.norm(p[:3, 3] - t[:3, 3]) for p, t in zip(predicted, tips)])
    angle = np.array([Rotation.from_matrix(p[:3, :3].T @ t[:3, :3]).magnitude() for p, t in zip(predicted, tips)])
    return {
        "fk_id": FK_ID,
        "world_dh0": world_dh0,
        "flange_tip": flange_tip,
        "residual_position_max_m": float(position.max()),
        "residual_position_mean_m": float(position.mean()),
        "residual_rotation_max_deg": float(np.degrees(angle.max())),
        "samples": int(len(tips)),
    }


def canonical_targets(joint_targets: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    """World tip pose [N,7] (xyz + xyzw) implied by joint target commands."""
    return np.stack([matrix_to_pose7(calibration["world_dh0"] @ panda_fk(q) @ calibration["flange_tip"])
                     for q in joint_targets])


__all__ = ["FK_ID", "canonical_targets", "fit_fk_calibration", "matrix_to_pose7", "panda_fk", "pose7_to_matrix"]
