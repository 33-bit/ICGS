"""Deterministic 5-mm preprocessing for the added physical geometry path.

CPU index ties use the documented lexicographic/index order.  GPU floating
point ties can vary with device reduction order and are not claimed bitwise
identical across devices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from icgs.configuration.method import MethodConfig


@dataclass(frozen=True)
class PhysicalPreprocessConfig:
    """Fixed-size physical preprocessing parameters.

    Crop bounds are optional until a calibrated workspace profile is supplied;
    specifying only one bound is rejected rather than inventing the other.
    """

    voxel_size_m: float
    num_points: int
    num_anchors: int
    num_neighbors: int
    crop_min: tuple[float, float, float] | None = None
    crop_max: tuple[float, float, float] | None = None
    ell0: float | None = None

    def __init__(
        self,
        voxel_size_m: float | None = None,
        num_points: int | None = None,
        num_anchors: int | None = None,
        num_neighbors: int | None = None,
        crop_min: tuple[float, float, float] | None = None,
        crop_max: tuple[float, float, float] | None = None,
        ell0: float | None = None,
        *,
        method_config: MethodConfig | None = None,
    ) -> None:
        if method_config is not None and not isinstance(method_config, MethodConfig):
            raise TypeError("method_config must be a MethodConfig or None")
        if any(value is None for value in (voxel_size_m, num_points, num_anchors,
                                           num_neighbors, ell0)):
            resolved = method_config if method_config is not None else MethodConfig()
            geometry = resolved.geometry
            if voxel_size_m is None:
                voxel_size_m = geometry.voxel_size_m
            if num_points is None:
                num_points = geometry.num_points
            if num_anchors is None:
                num_anchors = geometry.num_anchors
            if num_neighbors is None:
                num_neighbors = geometry.neighbors
            if ell0 is None:
                ell0 = geometry.ell0_m
        if method_config is not None and crop_min is None and crop_max is None:
            bounds = method_config.sensors.workspace_bounds_m
            if bounds is not None:
                crop_min, crop_max = bounds
        object.__setattr__(self, "voxel_size_m", voxel_size_m)
        object.__setattr__(self, "num_points", num_points)
        object.__setattr__(self, "num_anchors", num_anchors)
        object.__setattr__(self, "num_neighbors", num_neighbors)
        object.__setattr__(self, "crop_min", crop_min)
        object.__setattr__(self, "crop_max", crop_max)
        object.__setattr__(self, "ell0", ell0)
        self.__post_init__()

    @classmethod
    def from_method_config(cls, method_config: MethodConfig) -> "PhysicalPreprocessConfig":
        """Build preprocessing settings from one already-resolved method config."""

        if not isinstance(method_config, MethodConfig):
            raise TypeError("method_config must be a MethodConfig")
        return cls(method_config=method_config)

    def __post_init__(self) -> None:
        voxel_size = float(self.voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size_m must be finite and positive")
        if not math.isclose(voxel_size, 0.005, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("P03 physical preprocessing identity requires voxel_size_m=0.005")
        object.__setattr__(self, "voxel_size_m", voxel_size)
        if not math.isfinite(float(self.ell0)) or self.ell0 <= 0:
            raise ValueError("ell0 must be finite and positive")
        if not math.isclose(float(self.ell0), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("P03 physical preprocessing identity requires ell0=1.0")
        object.__setattr__(self, "ell0", float(self.ell0))
        for name in ("num_points", "num_anchors", "num_neighbors"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected_counts = {"num_points": 2048, "num_anchors": 128, "num_neighbors": 32}
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"P03 physical preprocessing identity requires {name}={expected}")
        if (self.crop_min is None) != (self.crop_max is None):
            raise ValueError("crop_min and crop_max must be provided together")
        if self.crop_min is not None:
            low = tuple(float(value) for value in self.crop_min)
            high = tuple(float(value) for value in self.crop_max)
            if len(low) != 3 or len(high) != 3 or any(not math.isfinite(v) for v in low + high):
                raise ValueError("crop bounds must contain three finite values")
            if any(a >= b for a, b in zip(low, high)):
                raise ValueError("crop_min must be strictly below crop_max")
            object.__setattr__(self, "crop_min", low)
            object.__setattr__(self, "crop_max", high)


@dataclass(frozen=True)
class PreparedPhysicalCloud:
    """Fixed-size cloud, FPS anchors and deterministic neighbor masks."""

    points: Tensor
    point_valid: Tensor
    anchors: Tensor
    anchor_valid: Tensor
    neighbor_indices: Tensor
    neighbor_valid: Tensor

    @property
    def neighbors(self) -> Tensor:
        batch_size, num_anchors, num_neighbors = self.neighbor_indices.shape
        expanded = self.points[:, None, :, :].expand(batch_size, num_anchors, -1, -1)
        indices = self.neighbor_indices[..., None].expand(-1, -1, -1, 3)
        return torch.gather(expanded, 2, indices)


def _as_config(
    config: PhysicalPreprocessConfig | MethodConfig | Mapping[str, Any] | None,
) -> PhysicalPreprocessConfig:
    if config is None:
        return PhysicalPreprocessConfig()
    if isinstance(config, MethodConfig):
        return PhysicalPreprocessConfig.from_method_config(config)
    if isinstance(config, PhysicalPreprocessConfig):
        return config
    if isinstance(config, Mapping):
        return PhysicalPreprocessConfig(**dict(config))
    raise TypeError("config must be PhysicalPreprocessConfig, mapping, or None")


def _as_batched_points(points: Tensor | Any, valid: Tensor | Any) -> tuple[Tensor, Tensor, bool]:
    points = torch.as_tensor(points)
    valid = torch.as_tensor(valid)
    if not points.is_floating_point():
        points = points.to(torch.get_default_dtype())
    if points.ndim == 2:
        points = points.unsqueeze(0)
        single = True
    elif points.ndim == 3:
        single = False
    else:
        raise ValueError("points must have shape [N,3] or [B,N,3]")
    if points.shape[-1] != 3:
        raise ValueError("points must be XYZ with shape [N,3] or [B,N,3]")
    if valid.ndim == 1:
        valid = valid.unsqueeze(0)
    if valid.ndim != 2 or valid.shape != points.shape[:2]:
        raise ValueError("valid must have shape [N] or [B,N] matching points")
    return points, valid.to(dtype=torch.bool, device=points.device), single


def _lexicographic_order(keys: Tensor) -> list[int]:
    key_rows = keys.detach().cpu().tolist()
    return sorted(range(len(key_rows)), key=lambda index: tuple(key_rows[index]))


def _voxel_centroids(points: Tensor, valid: Tensor, config: PhysicalPreprocessConfig) -> Tensor:
    selected = points[valid]
    if selected.numel() == 0:
        raise ValueError("physical cloud contains no valid points")
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError("valid physical points must be finite")
    if config.crop_min is not None:
        crop_min = torch.tensor(config.crop_min, dtype=points.dtype, device=points.device)
        crop_max = torch.tensor(config.crop_max, dtype=points.dtype, device=points.device)
        in_crop = ((selected >= crop_min) & (selected <= crop_max)).all(dim=-1)
        selected = selected[in_crop]
        if selected.numel() == 0:
            raise ValueError("physical cloud contains no valid points inside crop")
    keys = torch.floor(selected / config.voxel_size_m).to(dtype=torch.int64)
    order = _lexicographic_order(keys)
    centroids: list[Tensor] = []
    start = 0
    while start < len(order):
        first = order[start]
        key = tuple(keys[first].detach().cpu().tolist())
        end = start + 1
        while end < len(order) and tuple(keys[order[end]].detach().cpu().tolist()) == key:
            end += 1
        indices = torch.tensor(order[start:end], dtype=torch.long, device=points.device)
        centroids.append(selected.index_select(0, indices).mean(dim=0))
        start = end
    return torch.stack(centroids, dim=0)


def _fixed_points(centroids: Tensor, count: int) -> tuple[Tensor, Tensor]:
    num_centroids = centroids.shape[0]
    if num_centroids >= count:
        if count == 1:
            indices = torch.zeros(1, dtype=torch.long, device=centroids.device)
        else:
            indices = torch.div(
                torch.arange(count, dtype=torch.long, device=centroids.device) * (num_centroids - 1),
                count - 1,
                rounding_mode="floor",
            )
        return centroids.index_select(0, indices), torch.ones(count, dtype=torch.bool, device=centroids.device)
    indices = torch.arange(count, dtype=torch.long, device=centroids.device).remainder(num_centroids)
    valid = torch.arange(count, device=centroids.device) < num_centroids
    return centroids.index_select(0, indices), valid


def _farthest_point_anchors(points: Tensor, valid: Tensor, count: int) -> tuple[Tensor, Tensor]:
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("physical cloud contains no valid points")
    num_unique = min(int(valid_indices.numel()), count)
    selected = [valid_indices[0]]
    distances = ((points - points[selected[0]]) ** 2).sum(dim=-1)
    distances = torch.where(valid, distances, torch.full_like(distances, -1.0))
    for _ in range(1, num_unique):
        next_index = distances.argmax()
        selected.append(next_index)
        next_distance = ((points - points[next_index]) ** 2).sum(dim=-1)
        distances = torch.minimum(distances, next_distance)
        distances = torch.where(valid, distances, torch.full_like(distances, -1.0))
    selected_tensor = torch.stack(selected)
    anchors = points.index_select(0, selected_tensor)
    anchor_valid = torch.ones(num_unique, dtype=torch.bool, device=points.device)
    if num_unique < count:
        repeats = torch.arange(count, dtype=torch.long, device=points.device).remainder(num_unique)
        anchors = points.index_select(0, repeats)
        anchor_valid = torch.arange(count, device=points.device) < num_unique
    return anchors, anchor_valid


def _nearest_neighbors(points: Tensor, point_valid: Tensor, anchors: Tensor,
                       anchor_valid: Tensor, count: int) -> tuple[Tensor, Tensor]:
    valid_indices = torch.nonzero(point_valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("physical cloud contains no valid points")
    indices: list[Tensor] = []
    masks: list[Tensor] = []
    for anchor, is_valid in zip(anchors, anchor_valid):
        distances = ((points.index_select(0, valid_indices) - anchor) ** 2).sum(dim=-1)
        order = torch.argsort(distances, stable=True)
        ordered = valid_indices.index_select(0, order)
        take = min(int(ordered.numel()), count)
        chosen = ordered[:take]
        neighbor_mask = torch.ones(take, dtype=torch.bool, device=points.device)
        if take < count:
            chosen = chosen[torch.arange(count, device=points.device).remainder(take)]
            neighbor_mask = torch.arange(count, device=points.device) < take
        if not bool(is_valid.item()):
            neighbor_mask = torch.zeros(count, dtype=torch.bool, device=points.device)
        indices.append(chosen)
        masks.append(neighbor_mask)
    return torch.stack(indices), torch.stack(masks)


def prepare_physical_cloud(
    points: Tensor | Any,
    valid: Tensor | Any,
    config: PhysicalPreprocessConfig | MethodConfig | Mapping[str, Any] | None = None,
) -> PreparedPhysicalCloud:
    """Voxelize, sample, FPS-anchor and kNN a world-frame point cloud.

    All index decisions are deterministic.  Repeated coordinates only fill
    fixed storage shapes; their corresponding masks remain invalid.
    """

    config = _as_config(config)
    points, valid, _ = _as_batched_points(points, valid)
    point_batches: list[Tensor] = []
    point_masks: list[Tensor] = []
    anchor_batches: list[Tensor] = []
    anchor_masks: list[Tensor] = []
    index_batches: list[Tensor] = []
    neighbor_masks: list[Tensor] = []
    for batch_points, batch_valid in zip(points, valid):
        centroids = _voxel_centroids(batch_points, batch_valid, config)
        sampled, sampled_valid = _fixed_points(centroids, config.num_points)
        anchors, anchor_valid = _farthest_point_anchors(sampled, sampled_valid, config.num_anchors)
        neighbor_indices, neighbor_valid = _nearest_neighbors(
            sampled, sampled_valid, anchors, anchor_valid, config.num_neighbors
        )
        point_batches.append(sampled)
        point_masks.append(sampled_valid)
        anchor_batches.append(anchors)
        anchor_masks.append(anchor_valid)
        index_batches.append(neighbor_indices)
        neighbor_masks.append(neighbor_valid)

    return PreparedPhysicalCloud(
        points=torch.stack(point_batches),
        point_valid=torch.stack(point_masks),
        anchors=torch.stack(anchor_batches),
        anchor_valid=torch.stack(anchor_masks),
        neighbor_indices=torch.stack(index_batches),
        neighbor_valid=torch.stack(neighbor_masks),
    )


def _axis_rotation(angle: float, gravity: Tensor) -> Tensor:
    axis = gravity / torch.linalg.vector_norm(gravity)
    x, y, z = axis
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack((
        torch.stack((zero, -z, y)),
        torch.stack((z, zero, -x)),
        torch.stack((-y, x, zero)),
    ))
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def augment_yaw(
    points_w: Tensor | Any,
    T_w_e: Tensor | Any,
    command_targets_w: Tensor | Any | None,
    angle: float,
    gravity: Tensor | Any | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Apply one calibrated-gravity yaw jointly to cloud, pose and targets."""

    points = torch.as_tensor(points_w)
    if not points.is_floating_point():
        points = points.to(torch.get_default_dtype())
    pose = torch.as_tensor(T_w_e, dtype=points.dtype, device=points.device)
    gravity_tensor = torch.tensor((0.0, 0.0, 1.0), dtype=points.dtype, device=points.device) \
        if gravity is None else torch.as_tensor(gravity, dtype=points.dtype, device=points.device)
    if gravity_tensor.shape != (3,) or not bool(torch.isfinite(gravity_tensor).all().item()):
        raise ValueError("gravity must be a finite 3-vector")
    if float(torch.linalg.vector_norm(gravity_tensor).item()) == 0.0:
        raise ValueError("gravity must be nonzero")
    rotation = _axis_rotation(float(angle), gravity_tensor)
    if points.shape[-1] != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError("points and T_w_e must have shapes [...,3] and [...,4,4]")
    rotated_points = torch.matmul(points, rotation.transpose(-1, -2))
    world_rotation = torch.eye(4, dtype=points.dtype, device=points.device)
    world_rotation[:3, :3] = rotation
    rotated_pose = world_rotation @ pose
    rotated_commands = None
    if command_targets_w is not None:
        commands = torch.as_tensor(command_targets_w, dtype=points.dtype, device=points.device)
        if commands.shape[-2:] != (4, 4):
            raise ValueError("command_targets_w must have shape [...,4,4]")
        rotated_commands = world_rotation @ commands
    return rotated_points, rotated_pose, rotated_commands


__all__ = [
    "PhysicalPreprocessConfig",
    "PreparedPhysicalCloud",
    "augment_yaw",
    "prepare_physical_cloud",
]
