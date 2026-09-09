"""Causal physical state ownership and branch lineage validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _owned_tensor(value: Tensor, shape: tuple[int, ...], name: str, *, dtype: torch.dtype | None = None) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value.clone()


def _owned_valid(value: Tensor, shape: tuple[int, ...], name: str, *, device: torch.device) -> Tensor:
    if not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.bool:
        raise ValueError(f"{name} must have shape {shape} and boolean dtype")
    if value.device != device:
        raise ValueError(f"{name} must share the physical state device")
    return value.clone()


def _validate_pose(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    bottom = value[..., 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=value.dtype, device=value.device)
    if not bool(torch.allclose(bottom, expected, atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    rotation = value[..., :3, :3]
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if not bool(torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    if not bool(torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    return value.clone()


def _lineage(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} lineage must be nonempty")
    return value


def validate_next_boundary(previous: int, current: int) -> None:
    """Require a physical update to advance exactly one causal boundary."""

    if (
        isinstance(previous, bool)
        or isinstance(current, bool)
        or not isinstance(previous, int)
        or not isinstance(current, int)
        or previous < 0
        or current < 0
    ):
        raise ValueError("boundary indices must be nonnegative integers")
    if current != previous + 1:
        raise ValueError("physical update requires exactly the next boundary")


@dataclass
class PhysicalState:
    """Owned physical tensors at one causal boundary."""

    X: Tensor
    x: Tensor
    valid: Tensor
    p: Tensor
    memory: Tensor
    T_w_e: Tensor
    grip: Tensor
    cached_world_cloud: Tensor | None
    cached_world_cloud_valid: Tensor | None
    boundary: int
    encoder_lineage: str
    memory_lineage: str
    origin: str = "real"

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.X) or self.X.ndim != 3 or self.X.shape[1:] != (128, 256):
            raise ValueError("X must have shape [B,128,256]")
        batch = self.X.shape[0]
        if batch <= 0:
            raise ValueError("X must have a nonempty batch")
        if not self.X.is_floating_point():
            raise TypeError("X must use a floating dtype")
        device, dtype = self.X.device, self.X.dtype
        X = _owned_tensor(self.X, (batch, 128, 256), "X", dtype=dtype)
        x = _owned_tensor(self.x, (batch, 128, 3), "x", dtype=dtype)
        valid = _owned_valid(self.valid, (batch, 128), "valid", device=device)
        p = _owned_tensor(self.p, (batch, 13), "p", dtype=dtype)
        memory = _owned_tensor(self.memory, (batch, 2, 256), "memory", dtype=dtype)
        pose = _owned_tensor(self.T_w_e, (batch, 4, 4), "T_w_e", dtype=dtype)
        pose = _validate_pose(pose, "T_w_e")
        grip = _owned_tensor(self.grip, (batch, 1), "grip", dtype=dtype)
        if not bool(((grip == 0) | (grip == 1)).all().item()):
            raise ValueError("grip must contain only 0 or 1")
        for name, value in (("x", x), ("p", p), ("memory", memory), ("T_w_e", pose), ("grip", grip)):
            if value.device != device:
                raise ValueError(f"{name} must share the physical state device")

        if isinstance(self.boundary, bool) or not isinstance(self.boundary, int) or self.boundary < 0:
            raise ValueError("boundary must be a nonnegative integer")
        if self.origin not in ("real", "imagined"):
            raise ValueError("origin must be real or imagined")
        encoder_lineage = _lineage(self.encoder_lineage, "encoder")
        memory_lineage = _lineage(self.memory_lineage, "physical memory")

        cached_cloud = self.cached_world_cloud
        cached_valid = self.cached_world_cloud_valid
        if (cached_cloud is None) != (cached_valid is None):
            raise ValueError("cached world cloud and mask must be provided together")
        if cached_cloud is not None and cached_valid is not None:
            if not torch.is_tensor(cached_cloud) or cached_cloud.ndim != 3 or cached_cloud.shape[0] != batch or cached_cloud.shape[-1] != 3:
                raise ValueError("cached_world_cloud must have shape [B,N,3]")
            if cached_cloud.shape[1] == 0:
                raise ValueError("cached_world_cloud must contain at least one point")
            if cached_cloud.dtype != dtype or cached_cloud.device != device:
                raise ValueError("cached_world_cloud must share physical state dtype and device")
            if not bool(torch.isfinite(cached_cloud).all().item()):
                raise ValueError("cached_world_cloud must contain only finite values")
            if not torch.is_tensor(cached_valid) or cached_valid.shape != cached_cloud.shape[:2] or cached_valid.dtype != torch.bool:
                raise ValueError("cached_world_cloud_valid must match cached_world_cloud and be boolean")
            if cached_valid.device != device:
                raise ValueError("cached_world_cloud_valid must share the physical state device")
            cached_cloud = cached_cloud.clone()
            cached_valid = cached_valid.clone()

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "T_w_e", pose)
        object.__setattr__(self, "grip", grip)
        object.__setattr__(self, "cached_world_cloud", cached_cloud)
        object.__setattr__(self, "cached_world_cloud_valid", cached_valid)
        object.__setattr__(self, "encoder_lineage", encoder_lineage)
        object.__setattr__(self, "memory_lineage", memory_lineage)

    @property
    def anchor_valid(self) -> Tensor:
        return self.valid

    @property
    def physical_memory_lineage(self) -> str:
        return self.memory_lineage

    def branch_copy(self) -> "PhysicalState":
        """Clone mutable state tensors and mark the branch as imagined."""

        return PhysicalState(
            X=self.X.clone(),
            x=self.x.clone(),
            valid=self.valid.clone(),
            p=self.p.clone(),
            memory=self.memory.clone(),
            T_w_e=self.T_w_e.clone(),
            grip=self.grip.clone(),
            cached_world_cloud=None if self.cached_world_cloud is None else self.cached_world_cloud.clone(),
            cached_world_cloud_valid=None if self.cached_world_cloud_valid is None else self.cached_world_cloud_valid.clone(),
            boundary=self.boundary,
            encoder_lineage=self.encoder_lineage,
            memory_lineage=self.memory_lineage,
            origin="imagined",
        )


def validate_real_update(previous: PhysicalState, current: PhysicalState) -> None:
    """Validate that a measured-history update preserves physical lineage."""

    if not isinstance(previous, PhysicalState) or not isinstance(current, PhysicalState):
        raise TypeError("real update requires PhysicalState records")
    if previous.origin != "real" or current.origin != "real":
        raise ValueError("real update cannot consume imagined physical state")
    validate_next_boundary(previous.boundary, current.boundary)
    if previous.encoder_lineage != current.encoder_lineage or previous.memory_lineage != current.memory_lineage:
        raise ValueError("real update requires matching physical lineage")


__all__ = ["PhysicalState", "validate_next_boundary", "validate_real_update"]
