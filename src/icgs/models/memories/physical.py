"""Causal physical features and memory modules for the added method path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from icgs.contracts.method import TimedCommand
from icgs.geometry.se3 import se3_log
from icgs.models.layers.method import masked_mean


_ELL0 = 1.0
_DT0 = 0.1


class _PhysicalTokenProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proprioception = nn.Sequential(
            nn.Linear(13, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.type_embeddings = nn.Parameter(torch.zeros(4, 256))

    def forward(self, p: Tensor) -> Tensor:
        return self.proprioception(p)


def _as_pose(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if value.ndim != 3 or value.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape [B,4,4]")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    bottom = value[..., 3, :]
    expected_bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=value.dtype, device=value.device
    )
    if not bool(torch.allclose(bottom, expected_bottom, atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    rotation = value[..., :3, :3]
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if not bool(torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    if not bool(torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-6, rtol=1e-6)):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    return value


def _as_feature(value: Tensor, shape: tuple[int, ...], name: str, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    value = torch.as_tensor(value, dtype=dtype, device=device)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def proprioception(T_w_e: Tensor, grip: Tensor, gravity: Tensor) -> Tensor:
    """Encode pose, measured grip and calibrated gravity as 13 features."""

    T_w_e = _as_pose(T_w_e, "T_w_e")
    batch = T_w_e.shape[0]
    grip = _as_feature(grip, (batch, 1), "grip", dtype=T_w_e.dtype, device=T_w_e.device)
    if not bool(((grip == 0) | (grip == 1)).all().item()):
        raise ValueError("grip must contain only 0 or 1")
    gravity = _as_feature(gravity, (batch, 3), "gravity", dtype=T_w_e.dtype, device=T_w_e.device)
    rot6 = torch.cat((T_w_e[..., :3, 0], T_w_e[..., :3, 1]), dim=-1)
    return torch.cat((T_w_e[..., :3, 3] / _ELL0, rot6, grip, gravity), dim=-1)


def action_descriptor(T_w_e: Tensor, command: TimedCommand) -> Tensor:
    """Encode an absolute timed command relative to the achieved pose."""

    T_w_e = _as_pose(T_w_e, "T_w_e")
    if not isinstance(command, TimedCommand):
        raise TypeError("command must be a TimedCommand")
    target = torch.tensor(command.target_w, dtype=T_w_e.dtype, device=T_w_e.device)
    target = target.unsqueeze(0).expand(T_w_e.shape[0], -1, -1)
    relative = torch.linalg.inv(T_w_e) @ target
    twist = se3_log(relative)
    twist = torch.cat((twist[..., :3] / _ELL0, twist[..., 3:]), dim=-1)
    commanded_grip = torch.full(
        (T_w_e.shape[0], 1), float(command.grip), dtype=T_w_e.dtype, device=T_w_e.device
    )
    log_duration_ratio = torch.full(
        (T_w_e.shape[0], 1),
        math.log(command.duration_s / _DT0),
        dtype=T_w_e.dtype,
        device=T_w_e.device,
    )
    return torch.cat((twist, commanded_grip, log_duration_ratio), dim=-1)


class PhysicalMemory(nn.Module):
    """Two-layer causal physical memory update."""

    def __init__(self):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(277, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.gru1 = nn.GRUCell(256, 256)
        self.gru2 = nn.GRUCell(256, 256)
        self.token_projector = _PhysicalTokenProjector()

    def forward(
        self,
        X: Tensor,
        valid: Tensor,
        p: Tensor,
        previous_u: Tensor,
        memory: Tensor,
    ) -> Tensor:
        if not torch.is_tensor(X) or X.ndim != 3 or X.shape[1:] != (128, 256):
            raise ValueError("X must have shape [B,128,256]")
        if not torch.is_tensor(valid) or valid.shape != X.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must have shape [B,128] and boolean dtype")
        batch = X.shape[0]
        if not torch.is_tensor(p) or p.shape != (batch, 13):
            raise ValueError("p must have shape [B,13]")
        if not torch.is_tensor(previous_u) or previous_u.shape != (batch, 8):
            raise ValueError("previous_u must have shape [B,8]")
        if not torch.is_tensor(memory) or memory.shape != (batch, 2, 256):
            raise ValueError("memory must have shape [B,2,256]")
        if not all(value.is_floating_point() for value in (X, p, previous_u, memory)):
            raise TypeError("X, p, previous_u and memory must use floating dtypes")
        if not (X.device == p.device == previous_u.device == memory.device):
            raise ValueError("physical memory tensors must share a device")
        if not (X.dtype == p.dtype == previous_u.dtype == memory.dtype):
            raise ValueError("physical memory tensors must share a dtype")

        v = self.input_mlp(torch.cat((masked_mean(X, valid, dim=1), p, previous_u), dim=-1))
        m1 = self.gru1(v, memory[:, 0])
        m2 = self.gru2(m1, memory[:, 1])
        return torch.stack((m1, m2), dim=1)

    def physical_tokens(self, state: object) -> tuple[Tensor, Tensor]:
        """Build masked geometry, proprioception and memory token rows."""

        X = getattr(state, "X", None)
        valid = getattr(state, "valid", getattr(state, "anchor_valid", None))
        p = getattr(state, "p", None)
        memory = getattr(state, "memory", None)
        if not torch.is_tensor(X) or X.ndim != 3 or X.shape[1:] != (128, 256):
            raise ValueError("state.X must have shape [B,128,256]")
        if not torch.is_tensor(valid) or valid.shape != X.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("state.valid must have shape [B,128] and boolean dtype")
        batch = X.shape[0]
        if not torch.is_tensor(p) or p.shape != (batch, 13):
            raise ValueError("state.p must have shape [B,13]")
        if not torch.is_tensor(memory) or memory.shape != (batch, 2, 256):
            raise ValueError("state.memory must have shape [B,2,256]")
        if not all(value.is_floating_point() for value in (X, p, memory)):
            raise TypeError("state tensors must use floating dtypes")
        if not (X.device == p.device == memory.device):
            raise ValueError("state tensors must share a device")
        if not (X.dtype == p.dtype == memory.dtype):
            raise ValueError("state tensors must share a dtype")

        geometry = torch.where(valid[..., None], X, torch.zeros_like(X))
        proprioception_row = self.token_projector(p).unsqueeze(1)
        memory_rows = memory
        rows = torch.cat((geometry, proprioception_row, memory_rows), dim=1)
        type_ids = torch.cat(
            (
                torch.zeros(128, dtype=torch.long, device=p.device),
                torch.tensor((1, 2, 3), dtype=torch.long, device=p.device),
            )
        )
        embeddings = self.token_projector.type_embeddings[type_ids]
        rows = rows + embeddings[None, :, :]
        rows = torch.cat(
            (
                torch.where(valid[..., None], rows[:, :128], torch.zeros_like(rows[:, :128])),
                rows[:, 128:],
            ),
            dim=1,
        )
        token_valid = torch.cat((valid, valid.new_ones((batch, 3))), dim=1)
        return rows, token_valid


__all__ = ["PhysicalMemory", "action_descriptor", "proprioception"]
