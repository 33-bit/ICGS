"""Causal physical features and memory modules for the added method path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch
from torch import Tensor, nn

from icgs.configuration.method import (
    ControlConfig,
    GeometryConfig,
    MemoryConfig,
    MethodConfig,
    NeuralConfig,
)
from icgs.contracts.method import TimedCommand
from icgs.geometry.se3 import se3_log
from icgs.models.layers.method import masked_mean


def _resolve_config(
    config: MethodConfig | Mapping[str, Any] | None = None,
    *,
    geometry: GeometryConfig | None = None,
    memory: MemoryConfig | None = None,
    neural: NeuralConfig | None = None,
    control: ControlConfig | None = None,
    ell0_m: float | None = None,
    dt0: float | None = None,
) -> MethodConfig:
    """Resolve P04 metadata once at its public construction boundary."""

    if config is None:
        resolved = MethodConfig()
    elif isinstance(config, MethodConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = MethodConfig.from_dict(dict(config))
    else:
        raise TypeError("config must be a MethodConfig, mapping, or None")

    sections = {
        "geometry": (geometry, GeometryConfig),
        "memory": (memory, MemoryConfig),
        "neural": (neural, NeuralConfig),
        "control": (control, ControlConfig),
    }
    for name, (section, section_type) in sections.items():
        if section is not None and type(section) is not section_type:
            raise TypeError(f"{name} must be {section_type.__name__}")

    if (
        geometry is None
        and memory is None
        and neural is None
        and control is None
        and ell0_m is None
        and dt0 is None
    ):
        return resolved

    selected_geometry = geometry if geometry is not None else resolved.geometry
    selected_control = control if control is not None else resolved.control
    if ell0_m is not None:
        selected_geometry = replace(selected_geometry, ell0_m=ell0_m)
    if dt0 is not None:
        selected_control = replace(selected_control, dt0=dt0)

    return replace(
        resolved,
        geometry=selected_geometry,
        memory=memory if memory is not None else resolved.memory,
        neural=neural if neural is not None else resolved.neural,
        control=selected_control,
    )


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"P04 supports only the central GELU activation, got {name!r}")


def _proprioception_dim(geometry: GeometryConfig) -> int:
    # Translation XYZ, ordered Rot6, binary grip and gravity XYZ.
    return geometry.point_dim + 2 * geometry.point_dim + 1 + geometry.point_dim


class _PhysicalTokenProjector(nn.Module):
    def __init__(self, *, input_dim: int, width: int, type_count: int, activation: str):
        super().__init__()
        self.proprioception = nn.Sequential(
            nn.Linear(input_dim, width),
            _activation(activation),
            nn.Linear(width, width),
        )
        self.type_embeddings = nn.Parameter(torch.zeros(type_count, width))

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


def proprioception(
    T_w_e: Tensor,
    grip: Tensor,
    gravity: Tensor,
    *,
    config: MethodConfig | Mapping[str, Any] | None = None,
    geometry: GeometryConfig | None = None,
    ell0_m: float | None = None,
) -> Tensor:
    """Encode pose, measured grip and calibrated gravity as 13 features."""

    resolved = _resolve_config(config, geometry=geometry, ell0_m=ell0_m)
    geometry_config = resolved.geometry
    T_w_e = _as_pose(T_w_e, "T_w_e")
    batch = T_w_e.shape[0]
    grip = _as_feature(grip, (batch, 1), "grip", dtype=T_w_e.dtype, device=T_w_e.device)
    if not bool(((grip == 0) | (grip == 1)).all().item()):
        raise ValueError("grip must contain only 0 or 1")
    gravity = _as_feature(gravity, (batch, 3), "gravity", dtype=T_w_e.dtype, device=T_w_e.device)
    rot6 = torch.cat((T_w_e[..., :3, 0], T_w_e[..., :3, 1]), dim=-1)
    return torch.cat((T_w_e[..., :3, 3] / geometry_config.ell0_m, rot6, grip, gravity), dim=-1)


def action_descriptor(
    T_w_e: Tensor,
    command: TimedCommand,
    *,
    config: MethodConfig | Mapping[str, Any] | None = None,
    geometry: GeometryConfig | None = None,
    control: ControlConfig | None = None,
    ell0_m: float | None = None,
    dt0: float | None = None,
) -> Tensor:
    """Encode an absolute timed command relative to the achieved pose."""

    resolved = _resolve_config(
        config,
        geometry=geometry,
        control=control,
        ell0_m=ell0_m,
        dt0=dt0,
    )
    geometry_config = resolved.geometry
    control_config = resolved.control
    T_w_e = _as_pose(T_w_e, "T_w_e")
    if not isinstance(command, TimedCommand):
        raise TypeError("command must be a TimedCommand")
    target = torch.tensor(command.target_w, dtype=T_w_e.dtype, device=T_w_e.device)
    target = target.unsqueeze(0).expand(T_w_e.shape[0], -1, -1)
    relative = torch.linalg.inv(T_w_e) @ target
    twist = se3_log(relative)
    twist = torch.cat((twist[..., :3] / geometry_config.ell0_m, twist[..., 3:]), dim=-1)
    commanded_grip = torch.full(
        (T_w_e.shape[0], 1), float(command.grip), dtype=T_w_e.dtype, device=T_w_e.device
    )
    log_duration_ratio = torch.full(
        (T_w_e.shape[0], 1),
        math.log(command.duration_s / control_config.dt0),
        dtype=T_w_e.dtype,
        device=T_w_e.device,
    )
    return torch.cat((twist, commanded_grip, log_duration_ratio), dim=-1)


class PhysicalMemory(nn.Module):
    """Two-layer causal physical memory update."""

    def __init__(
        self,
        config: MethodConfig | Mapping[str, Any] | None = None,
        *,
        geometry: GeometryConfig | None = None,
        memory: MemoryConfig | None = None,
        neural: NeuralConfig | None = None,
        control: ControlConfig | None = None,
    ):
        super().__init__()
        resolved = _resolve_config(
            config,
            geometry=geometry,
            memory=memory,
            neural=neural,
            control=control,
        )
        self.config = resolved
        geometry_config = resolved.geometry
        memory_config = resolved.memory
        neural_config = resolved.neural
        width = geometry_config.width
        proprioception_dim = _proprioception_dim(geometry_config)
        input_dim = width + proprioception_dim + memory_config.descriptor_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, width),
            _activation(neural_config.activation),
            nn.Linear(width, width),
        )
        self.gru1 = nn.GRUCell(width, width)
        self.gru2 = nn.GRUCell(width, width)
        self.token_projector = _PhysicalTokenProjector(
            input_dim=proprioception_dim,
            width=width,
            type_count=2 + memory_config.slots,
            activation=neural_config.activation,
        )

    def forward(
        self,
        X: Tensor,
        valid: Tensor,
        p: Tensor,
        previous_u: Tensor,
        memory: Tensor,
    ) -> Tensor:
        geometry_config = self.config.geometry
        memory_config = self.config.memory
        width = geometry_config.width
        anchor_count = geometry_config.num_anchors
        proprioception_dim = _proprioception_dim(geometry_config)
        descriptor_dim = memory_config.descriptor_dim
        slots = memory_config.slots
        if not torch.is_tensor(X) or X.ndim != 3 or X.shape[1:] != (anchor_count, width):
            raise ValueError(f"X must have shape [B,{anchor_count},{width}]")
        if not torch.is_tensor(valid) or valid.shape != X.shape[:2] or valid.dtype != torch.bool:
            raise ValueError(f"valid must have shape [B,{anchor_count}] and boolean dtype")
        batch = X.shape[0]
        if not torch.is_tensor(p) or p.shape != (batch, proprioception_dim):
            raise ValueError(f"p must have shape [B,{proprioception_dim}]")
        if not torch.is_tensor(previous_u) or previous_u.shape != (batch, descriptor_dim):
            raise ValueError(f"previous_u must have shape [B,{descriptor_dim}]")
        if not torch.is_tensor(memory) or memory.shape != (batch, slots, width):
            raise ValueError(f"memory must have shape [B,{slots},{width}]")
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

        geometry_config = self.config.geometry
        memory_config = self.config.memory
        anchor_count = geometry_config.num_anchors
        width = geometry_config.width
        proprioception_dim = _proprioception_dim(geometry_config)
        slots = memory_config.slots
        X = getattr(state, "X", None)
        valid = getattr(state, "valid", getattr(state, "anchor_valid", None))
        p = getattr(state, "p", None)
        memory = getattr(state, "memory", None)
        if not torch.is_tensor(X) or X.ndim != 3 or X.shape[1:] != (anchor_count, width):
            raise ValueError(f"state.X must have shape [B,{anchor_count},{width}]")
        if not torch.is_tensor(valid) or valid.shape != X.shape[:2] or valid.dtype != torch.bool:
            raise ValueError(f"state.valid must have shape [B,{anchor_count}] and boolean dtype")
        batch = X.shape[0]
        if not torch.is_tensor(p) or p.shape != (batch, proprioception_dim):
            raise ValueError(f"state.p must have shape [B,{proprioception_dim}]")
        if not torch.is_tensor(memory) or memory.shape != (batch, slots, width):
            raise ValueError(f"state.memory must have shape [B,{slots},{width}]")
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
                torch.zeros(anchor_count, dtype=torch.long, device=p.device),
                torch.arange(1, slots + 2, dtype=torch.long, device=p.device),
            )
        )
        embeddings = self.token_projector.type_embeddings[type_ids]
        rows = rows + embeddings[None, :, :]
        rows = torch.cat(
            (
                torch.where(
                    valid[..., None],
                    rows[:, :anchor_count],
                    torch.zeros_like(rows[:, :anchor_count]),
                ),
                rows[:, anchor_count:],
            ),
            dim=1,
        )
        token_valid = torch.cat((valid, valid.new_ones((batch, 1 + slots))), dim=1)
        return rows, token_valid


__all__ = ["PhysicalMemory", "action_descriptor", "proprioception"]
