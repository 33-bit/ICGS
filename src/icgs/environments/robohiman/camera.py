"""CoppeliaSim/RLBench camera convention, reimplemented without PyRep.

RLBench stores depth normalized to [0, 1] between the sensor's near and far
clipping planes, intrinsics with *negative* focal lengths, and extrinsics as
the camera-to-world pose. ``depth_to_world`` reproduces
``VisionSensor.pointcloud_from_depth_and_camera_params`` (PyRep 231a1ac) in
float64 so stored episodes can be converted offline; the smoke gate compares
it numerically against PyRep's own function and against scene geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


CAMERA_CONVENTION_ID = "coppeliasim-normalized-depth-negative-focal-v1"
# Pixels whose normalized depth reaches the far plane saw nothing.
FAR_PLANE_EPS = 1e-6


def metric_depth(depth_normalized: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.asarray(depth_normalized, dtype=np.float64)
    return float(near) + depth * (float(far) - float(near))


def depth_to_world(depth_m: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project metric depth [H,W] to world XYZ [H,W,3]."""
    depth_m = np.asarray(depth_m, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth_m.ndim != 2 or extrinsics.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("expected depth [H,W], extrinsics [4,4], intrinsics [3,3]")
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([u * depth_m, v * depth_m, depth_m, np.ones_like(depth_m)], axis=-1)
    rotation = extrinsics[:3, :3]
    centre = extrinsics[:3, 3:4]
    world_to_camera = np.concatenate([rotation.T, -rotation.T @ centre], axis=1)
    projection = np.vstack([intrinsics @ world_to_camera, [0.0, 0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(projection)[:3]
    return pixels @ inverse.T


def project_world(points_w: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to (u, v) pixel coordinates and camera depth."""
    points_w = np.atleast_2d(np.asarray(points_w, dtype=np.float64))
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    rotation, centre = extrinsics[:3, :3], extrinsics[:3, 3]
    camera = (points_w - centre) @ rotation
    homogeneous = camera @ np.asarray(intrinsics, dtype=np.float64).T
    depth = homogeneous[:, 2]
    uv = homogeneous[:, :2] / depth[:, None]
    return uv, depth


def valid_depth_mask(depth_normalized: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_normalized)
    return np.isfinite(depth) & (depth > 0.0) & (depth < 1.0 - FAR_PLANE_EPS)


def camera_cloud(
    depth_normalized: np.ndarray,
    near: float,
    far: float,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world points [H*W,3] and validity [H*W] for one camera frame."""
    points = depth_to_world(metric_depth(depth_normalized, near, far), extrinsics, intrinsics)
    valid = valid_depth_mask(depth_normalized) & np.isfinite(points).all(axis=-1)
    return points.reshape(-1, 3), valid.reshape(-1)


def fused_cloud(
    arrays: Mapping[str, np.ndarray],
    frame: int,
    cameras: Sequence[str],
    *,
    crop: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate calibrated clouds from stored arrays for one frame.

    Returns points [N,3], validity [N] and the camera index [N]. Cropping only
    changes validity; no point is dropped, so indices remain reproducible.
    """
    points, valid, source = [], [], []
    for index, name in enumerate(cameras):
        cloud, mask = camera_cloud(
            arrays[f"cam_{name}_depth"][frame],
            float(arrays[f"cam_{name}_near"][frame]),
            float(arrays[f"cam_{name}_far"][frame]),
            arrays[f"cam_{name}_extrinsics"][frame],
            arrays[f"cam_{name}_intrinsics"][frame],
        )
        points.append(cloud)
        valid.append(mask)
        source.append(np.full(cloud.shape[0], index, dtype=np.int16))
    points_all = np.concatenate(points)
    valid_all = np.concatenate(valid)
    if crop is not None:
        low, high = (np.asarray(bound, dtype=np.float64) for bound in crop)
        inside = np.all((points_all >= low) & (points_all <= high), axis=1)
        valid_all &= inside
    return points_all, valid_all, np.concatenate(source)


def decode_mask_handles(mask_rgb: np.ndarray) -> np.ndarray:
    """RLBench ``rgb_handles_to_mask`` without its in-place mutation.

    Accepts the float [0,1] RGB-coded mask of a live observation. Rounding is
    explicit; the upstream code multiplies by 255 without rounding, which the
    mask-validation gate checks rather than assumes.
    """
    coded = np.rint(np.asarray(mask_rgb, dtype=np.float64) * 255.0).astype(np.int64)
    return coded[..., 0] + coded[..., 1] * 256 + coded[..., 2] * 65536


def summarize_calibration(arrays: Mapping[str, np.ndarray], cameras: Sequence[str]) -> dict[str, Any]:
    """Per-camera calibration facts used by the A0 audit report."""
    summary: dict[str, Any] = {}
    for name in cameras:
        intrinsics = arrays[f"cam_{name}_intrinsics"]
        extrinsics = arrays[f"cam_{name}_extrinsics"]
        rotation = extrinsics[:, :3, :3]
        orthogonality = np.abs(np.einsum("fij,fkj->fik", rotation, rotation) - np.eye(3)).max()
        summary[name] = {
            "fx_fy_first": [float(intrinsics[0, 0, 0]), float(intrinsics[0, 1, 1])],
            "principal_first": [float(intrinsics[0, 0, 2]), float(intrinsics[0, 1, 2])],
            "intrinsics_constant": bool(np.allclose(intrinsics, intrinsics[:1])),
            "extrinsics_constant": bool(np.allclose(extrinsics, extrinsics[:1])),
            "rotation_orthogonality_max_err": float(orthogonality),
            "rotation_det_first": float(np.linalg.det(rotation[0])),
            "position_first_m": [float(value) for value in extrinsics[0, :3, 3]],
            "near_far_first": [float(arrays[f"cam_{name}_near"][0]), float(arrays[f"cam_{name}_far"][0])],
        }
    return summary


__all__ = [
    "CAMERA_CONVENTION_ID",
    "camera_cloud",
    "decode_mask_handles",
    "depth_to_world",
    "fused_cloud",
    "metric_depth",
    "project_world",
    "summarize_calibration",
    "valid_depth_mask",
]
