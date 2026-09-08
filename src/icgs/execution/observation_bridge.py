"""Bridge cached imagined world geometry to the native observation record."""

from __future__ import annotations

from typing import Any

import torch

from icgs.contracts.records import Observation


def bridge_source(origin: str) -> str:
    """Return the only accepted source label for a bridge query."""

    if origin == "real":
        return "measured"
    if origin == "imagined":
        return "cached-decoded-world"
    raise ValueError(f"unsupported observation bridge origin: {origin!r}")


def imagined_observation(state: Any) -> Observation:
    """Construct a native observation from an imagined state's world cache.

    Native preprocessing owns the world-to-end-effector conversion, so this
    adapter deliberately forwards the decoded world cloud without transforming
    it first.
    """

    if getattr(state, "origin", None) != "imagined":
        raise ValueError("imagined bridge requires imagined state")
    try:
        cloud = state.cached_world_cloud
        cloud_valid = state.cached_world_cloud_valid
        pose = state.T_w_e
        grip = state.grip
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("imagined state is missing cached cloud, pose, or grip") from exc
    try:
        cloud_tensor = torch.as_tensor(cloud)
        mask_tensor = torch.as_tensor(cloud_valid, device=cloud_tensor.device)
        pose_tensor = torch.as_tensor(pose, device=cloud_tensor.device)
        grip_tensor = torch.as_tensor(grip, device=cloud_tensor.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("imagined bridge inputs must be numeric tensors or arrays") from exc
    if cloud_tensor.ndim != 3 or cloud_tensor.shape[0] != 1 or cloud_tensor.shape[2] != 3:
        raise ValueError("cached_world_cloud must have shape [1,N,3] with B=1")
    if cloud_tensor.shape[1] == 0:
        raise ValueError("cached_world_cloud must be nonempty")
    if mask_tensor.ndim != 2 or mask_tensor.shape != cloud_tensor.shape[:2]:
        raise ValueError("cached_world_cloud_valid must have matching shape [1,N]")
    if mask_tensor.dtype != torch.bool:
        raise ValueError("cached_world_cloud_valid must be a boolean mask")
    if pose_tensor.shape != (1, 4, 4):
        raise ValueError("T_w_e must have shape [1,4,4] with B=1")
    if grip_tensor.shape != (1, 1):
        raise ValueError("grip must have shape [1,1] with B=1")
    if pose_tensor.is_complex() or not bool(torch.isfinite(pose_tensor).all().item()):
        raise ValueError("T_w_e must be finite and real")
    if not bool(torch.isfinite(grip_tensor).all().item()):
        raise ValueError("pose and grip must be finite")
    pose_check = pose_tensor if pose_tensor.is_floating_point() else pose_tensor.to(torch.get_default_dtype())
    expected_bottom = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=pose_check.dtype, device=pose_check.device)
    if not torch.allclose(pose_check[0, 3], expected_bottom, atol=1e-7, rtol=0.0):
        raise ValueError("T_w_e must be a valid SE(3) pose")
    rotation = pose_check[0, :3, :3]
    identity = torch.eye(3, dtype=pose_check.dtype, device=pose_check.device)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-6, rtol=0.0):
        raise ValueError("T_w_e must be a valid SE(3) pose")
    determinant = torch.linalg.det(rotation)
    if not bool(torch.isclose(determinant, torch.ones((), dtype=pose_check.dtype, device=pose_check.device), atol=1e-6, rtol=0.0).item()):
        raise ValueError("T_w_e must be a valid SE(3) pose")
    grip_value = grip_tensor[0, 0].item()
    if grip_tensor.is_complex() or grip_value not in (0, 1):
        raise ValueError("grip must be exactly 0 or 1")

    points = cloud_tensor[0][mask_tensor[0]]
    if points.shape[0] == 0:
        raise ValueError("cached_world_cloud_valid must select at least one point")
    if not bool(torch.isfinite(points).all().item()):
        raise ValueError("valid cached world points must be finite")
    if not torch.is_tensor(cloud):
        points = points.detach().cpu().numpy()
    return Observation(points=points, T_w_e=pose[0], grip=float(grip_tensor[0, 0].item()))


__all__ = ["bridge_source", "imagined_observation"]
