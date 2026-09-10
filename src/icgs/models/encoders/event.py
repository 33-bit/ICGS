"""Tensor-only P05 event encoding with explicit masks and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from icgs.configuration.method import (
    EventConfig,
    GeometryConfig,
    MethodConfig,
    NeuralConfig,
)
from icgs.models.layers.method import MaskedSelfAttentionBlock


START_KIND = 0
INTERACTION_KIND = 1
END_KIND = 2
LANDMARK_TWIST_ATOL = 1e-6
_EVENT_KINDS = (START_KIND, INTERACTION_KIND, END_KIND)
_PROPRIO_WIDTH = 13
_TWIST_WIDTH = 6


@dataclass(frozen=True)
class EventEncoding:
    """Transient event tokens and their padding validity mask."""

    tokens: Tensor
    valid: Tensor


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"event encoder supports only GELU, got {name!r}")


def event_features(
    d_start: Tensor,
    d_end: Tensor,
    d_mean: Tensor,
    xi: Tensor,
    grip_start: Tensor,
    grip_end: Tensor,
) -> Tensor:
    """Concatenate unmasked event fields, rejecting every nonfinite value."""

    values = {
        "d_start": d_start,
        "d_end": d_end,
        "d_mean": d_mean,
        "xi": xi,
        "grip_start": grip_start,
        "grip_end": grip_end,
    }
    for name, value in values.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if value.ndim < 1:
            raise ValueError(f"{name} must have a feature dimension")

    leading = d_start.shape[:-1]
    descriptor_width = d_start.shape[-1]
    if descriptor_width <= 0:
        raise ValueError("event descriptors must have positive width")
    expected_shapes = {
        "d_end": leading + (descriptor_width,),
        "d_mean": leading + (descriptor_width,),
        "xi": leading + (_TWIST_WIDTH,),
        "grip_start": leading + (1,),
        "grip_end": leading + (1,),
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name, value in values.items():
        if value.device != d_start.device or value.dtype != d_start.dtype:
            raise ValueError(f"{name} must share d_start device and dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
    return torch.cat(tuple(values.values()), dim=-1)


def _validate_sections(
    geometry: GeometryConfig,
    event: EventConfig,
    neural: NeuralConfig,
) -> None:
    if geometry.width != event.width:
        raise ValueError("geometry.width must match event.width")
    if geometry.width <= 0 or event.token_hidden_dim <= 0 or event.transformer_layers <= 0:
        raise ValueError("event encoder widths and transformer layers must be positive")
    if neural.attention_heads <= 0 or event.width % neural.attention_heads:
        raise ValueError("neural attention heads must divide event.width")
    if neural.ffn_width <= 0:
        raise ValueError("neural.ffn_width must be positive")
    if not 0 <= neural.dropout < 1:
        raise ValueError("neural.dropout must be in [0,1)")
    if neural.layer_norm_eps <= 0:
        raise ValueError("neural.layer_norm_eps must be positive")
    if not neural.pre_norm:
        raise ValueError("event encoder requires neural.pre_norm=True")


def _resolve_sections(
    method_config: MethodConfig | None,
    geometry_config: GeometryConfig | None,
    event_config: EventConfig | None,
    neural_config: NeuralConfig | None,
) -> tuple[GeometryConfig, EventConfig, NeuralConfig]:
    typed = (geometry_config, event_config, neural_config)
    if method_config is not None:
        if not isinstance(method_config, MethodConfig):
            raise TypeError("method_config must be a resolved MethodConfig")
        if any(section is not None for section in typed):
            raise ValueError("method_config cannot be combined with typed sections")
        method_config.validate()
        return method_config.geometry, method_config.event, method_config.neural
    if all(section is None for section in typed):
        raise TypeError("an explicit MethodConfig or all typed config sections are required")
    if not isinstance(geometry_config, GeometryConfig):
        raise TypeError("geometry_config must be an explicit GeometryConfig section")
    if not isinstance(event_config, EventConfig):
        raise TypeError("event_config must be an explicit EventConfig section")
    if not isinstance(neural_config, NeuralConfig):
        raise TypeError("neural_config must be an explicit NeuralConfig section")
    return geometry_config, event_config, neural_config


def _validate_outer_inputs(
    *,
    frame_tokens: Tensor,
    anchor_valid: Tensor,
    proprio: Tensor,
    segment_start: Tensor,
    segment_end: Tensor,
    twist: Tensor,
    grip_start: Tensor,
    grip_end: Tensor,
    event_kind: Tensor,
    event_valid: Tensor,
    local_index: Tensor,
    local_count: Tensor,
    frame_width: int,
) -> tuple[int, int, int]:
    tensors = {
        "frame_tokens": frame_tokens,
        "anchor_valid": anchor_valid,
        "proprio": proprio,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "twist": twist,
        "grip_start": grip_start,
        "grip_end": grip_end,
        "event_kind": event_kind,
        "event_valid": event_valid,
        "local_index": local_index,
        "local_count": local_count,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
    if frame_tokens.ndim != 4 or frame_tokens.shape[-1] != frame_width:
        raise ValueError(f"frame_tokens must have shape [B,F,A,{frame_width}]")
    batch, frames, anchors, _ = frame_tokens.shape
    if batch <= 0 or frames <= 0 or anchors <= 0:
        raise ValueError("frame_tokens must have nonempty batch, frame, and anchor dimensions")
    if not frame_tokens.is_floating_point():
        raise TypeError("frame_tokens must use a floating dtype")
    if anchor_valid.shape != (batch, frames, anchors):
        raise ValueError("anchor_valid must have shape [B,F,A]")
    if anchor_valid.dtype != torch.bool:
        raise TypeError("anchor_valid must use torch.bool dtype")
    if proprio.shape != (batch, frames, _PROPRIO_WIDTH):
        raise ValueError(f"proprio must have shape [B,F,{_PROPRIO_WIDTH}]")

    if event_valid.ndim != 2 or event_valid.shape[0] != batch or event_valid.shape[1] <= 0:
        raise ValueError("event_valid must have nonempty shape [B,L]")
    events = event_valid.shape[1]
    event_shapes = {
        "segment_start": (batch, events),
        "segment_end": (batch, events),
        "twist": (batch, events, _TWIST_WIDTH),
        "grip_start": (batch, events, 1),
        "grip_end": (batch, events, 1),
        "event_kind": (batch, events),
        "local_index": (batch, events),
        "local_count": (batch, events),
    }
    for name, expected in event_shapes.items():
        if tensors[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if event_valid.dtype != torch.bool:
        raise TypeError("event_valid must use torch.bool dtype")
    for name in ("segment_start", "segment_end", "event_kind", "local_index", "local_count"):
        if tensors[name].dtype != torch.long:
            raise TypeError(f"{name} must use torch.long dtype")
    for name in ("proprio", "twist", "grip_start", "grip_end"):
        value = tensors[name]
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if value.dtype != frame_tokens.dtype:
            raise ValueError(f"{name} must share frame_tokens dtype")
    for name, value in tensors.items():
        if value.device != frame_tokens.device:
            raise ValueError(f"{name} must share frame_tokens device")
    if bool((~event_valid).all(dim=1).any().item()):
        raise ValueError("event_valid requires at least one valid event per sample")
    return batch, frames, events


def _finite_on_mask(value: Tensor, mask: Tensor, name: str) -> None:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    if not bool(torch.isfinite(value.masked_select(expanded)).all().item()):
        raise ValueError(f"{name} valid entries must contain only finite values")


def _validate_finite_inputs(
    frame_tokens: Tensor,
    anchor_valid: Tensor,
    proprio: Tensor,
    twist: Tensor,
    grip_start: Tensor,
    grip_end: Tensor,
    event_valid: Tensor,
) -> Tensor:
    frame_valid = anchor_valid.any(dim=-1)
    _finite_on_mask(frame_tokens, anchor_valid, "frame_tokens")
    _finite_on_mask(proprio, frame_valid, "proprio")
    _finite_on_mask(twist, event_valid, "twist")
    _finite_on_mask(grip_start, event_valid, "grip_start")
    _finite_on_mask(grip_end, event_valid, "grip_end")
    return frame_valid


def _validate_landmarks(
    segment_start: Tensor,
    segment_end: Tensor,
    twist: Tensor,
    event_kind: Tensor,
    event_valid: Tensor,
) -> None:
    landmark = event_valid & ((event_kind == START_KIND) | (event_kind == END_KIND))
    if bool((segment_start.masked_select(landmark) != segment_end.masked_select(landmark)).any().item()):
        raise ValueError("valid landmark segment_start must equal segment_end")
    landmark_twist = twist.masked_select(landmark[..., None].expand_as(twist)).reshape(-1, _TWIST_WIDTH)
    if landmark_twist.numel() and bool(
        (torch.linalg.vector_norm(landmark_twist, dim=-1) > LANDMARK_TWIST_ATOL).any().item()
    ):
        raise ValueError("valid landmark twist must be numerically zero")


def _validate_indices_and_order(
    *,
    segment_start: Tensor,
    segment_end: Tensor,
    event_kind: Tensor,
    event_valid: Tensor,
    local_index: Tensor,
    local_count: Tensor,
    frames: int,
) -> None:
    if bool(((segment_start < 0) & event_valid).any().item()):
        raise ValueError("valid segment_start must be nonnegative")
    if bool(((segment_start >= frames) & event_valid).any().item()):
        raise ValueError("valid segment_start must be smaller than F")
    if bool(((segment_end < 0) & event_valid).any().item()):
        raise ValueError("valid segment_end must be nonnegative")
    if bool(((segment_end >= frames) & event_valid).any().item()):
        raise ValueError("valid segment_end must be smaller than F")
    if bool(((segment_start > segment_end) & event_valid).any().item()):
        raise ValueError("valid segment_start must not exceed segment_end")
    known_kind = torch.zeros_like(event_valid)
    for kind in _EVENT_KINDS:
        known_kind |= event_kind == kind
    if bool((event_valid & ~known_kind).any().item()):
        raise ValueError("valid event_kind must be START_KIND, INTERACTION_KIND, or END_KIND")
    if bool(((local_count <= 0) & event_valid).any().item()):
        raise ValueError("valid local_count must be positive")
    if bool(((local_index < 0) & event_valid).any().item()):
        raise ValueError("valid local_index must be nonnegative")
    if bool(((local_index >= local_count) & event_valid).any().item()):
        raise ValueError("valid local_index must be smaller than local_count")


def _validate_referenced_frames(
    frame_valid: Tensor,
    segment_start: Tensor,
    segment_end: Tensor,
    event_valid: Tensor,
) -> None:
    positions = torch.arange(frame_valid.shape[1], device=frame_valid.device)
    covered = (
        (positions[None, None, :] >= segment_start[..., None])
        & (positions[None, None, :] <= segment_end[..., None])
        & event_valid[..., None]
    )
    referenced = covered.any(dim=1)
    if bool((referenced & ~frame_valid).any().item()):
        raise ValueError("anchor_valid requires at least one anchor in every referenced frame")


def _sanitize_padding(
    *,
    frame_tokens: Tensor,
    anchor_valid: Tensor,
    proprio: Tensor,
    frame_valid: Tensor,
    segment_start: Tensor,
    segment_end: Tensor,
    twist: Tensor,
    grip_start: Tensor,
    grip_end: Tensor,
    event_kind: Tensor,
    event_valid: Tensor,
    local_index: Tensor,
    local_count: Tensor,
) -> tuple[Tensor, ...]:
    event_mask = event_valid[..., None]
    safe_frames = torch.where(anchor_valid[..., None], frame_tokens, torch.zeros_like(frame_tokens))
    safe_proprio = torch.where(frame_valid[..., None], proprio, torch.zeros_like(proprio))
    safe_start = torch.where(event_valid, segment_start, torch.zeros_like(segment_start))
    safe_end = torch.where(event_valid, segment_end, torch.zeros_like(segment_end))
    safe_twist = torch.where(event_mask, twist, torch.zeros_like(twist))
    safe_grip_start = torch.where(event_mask, grip_start, torch.zeros_like(grip_start))
    safe_grip_end = torch.where(event_mask, grip_end, torch.zeros_like(grip_end))
    safe_kind = torch.where(event_valid, event_kind, torch.zeros_like(event_kind))
    safe_local_index = torch.where(event_valid, local_index, torch.zeros_like(local_index))
    safe_local_count = torch.where(event_valid, local_count, torch.ones_like(local_count))
    return (
        safe_frames,
        safe_proprio,
        safe_start,
        safe_end,
        safe_twist,
        safe_grip_start,
        safe_grip_end,
        safe_kind,
        safe_local_index,
        safe_local_count,
    )


def _encode_frames(
    frame_mlp: nn.Module,
    frame_tokens: Tensor,
    anchor_valid: Tensor,
    proprio: Tensor,
    frame_valid: Tensor,
) -> Tensor:
    counts = anchor_valid.sum(dim=2, keepdim=True).clamp_min(1).to(frame_tokens.dtype)
    pooled = frame_tokens.sum(dim=2) / counts
    descriptors = frame_mlp(torch.cat((pooled, proprio), dim=-1))
    return torch.where(frame_valid[..., None], descriptors, torch.zeros_like(descriptors))


def _gather_frames(descriptors: Tensor, indices: Tensor) -> Tensor:
    gather_index = indices[..., None].expand(-1, -1, descriptors.shape[-1])
    return descriptors.gather(dim=1, index=gather_index)


def _build_segment_features(
    descriptors: Tensor,
    segment_start: Tensor,
    segment_end: Tensor,
    event_valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    d_start = _gather_frames(descriptors, segment_start)
    d_end = _gather_frames(descriptors, segment_end)
    prefix = torch.cat(
        (torch.zeros_like(descriptors[:, :1]), descriptors.cumsum(dim=1)),
        dim=1,
    )
    segment_sum = _gather_frames(prefix, segment_end + 1) - _gather_frames(prefix, segment_start)
    length = (segment_end - segment_start + 1).to(descriptors.dtype)[..., None]
    d_mean = segment_sum / length
    event_mask = event_valid[..., None]
    return tuple(
        torch.where(event_mask, value, torch.zeros_like(value))
        for value in (d_start, d_end, d_mean)
    )


class EventEncoder(nn.Module):
    """Encode pre-tensorized measured demonstration events."""

    def __init__(
        self,
        *,
        method_config: MethodConfig | None = None,
        geometry_config: GeometryConfig | None = None,
        event_config: EventConfig | None = None,
        neural_config: NeuralConfig | None = None,
    ):
        super().__init__()
        geometry, event, neural = _resolve_sections(
            method_config,
            geometry_config,
            event_config,
            neural_config,
        )
        _validate_sections(geometry, event, neural)
        self.geometry_config = geometry
        self.event_config = event
        self.neural_config = neural
        self.frame_mlp = nn.Sequential(
            nn.Linear(geometry.width + _PROPRIO_WIDTH, event.width),
            _activation(neural.activation),
            nn.Linear(event.width, event.width),
        )
        self.event_mlp = nn.Sequential(
            nn.Linear(3 * event.width + _TWIST_WIDTH + 2, event.token_hidden_dim),
            _activation(neural.activation),
            nn.Linear(event.token_hidden_dim, event.width),
        )
        self.order_mlp = nn.Sequential(
            nn.Linear(2, event.width),
            _activation(neural.activation),
            nn.Linear(event.width, event.width),
        )
        self.kind_embedding = nn.Embedding(len(_EVENT_KINDS), event.width)
        self.blocks = nn.ModuleList(
            MaskedSelfAttentionBlock(width=event.width, neural_config=neural)
            for _ in range(event.transformer_layers)
        )

    def forward(
        self,
        frame_tokens: Tensor,
        anchor_valid: Tensor,
        proprio: Tensor,
        segment_start: Tensor,
        segment_end: Tensor,
        twist: Tensor,
        grip_start: Tensor,
        grip_end: Tensor,
        event_kind: Tensor,
        event_valid: Tensor,
        local_index: Tensor,
        local_count: Tensor,
    ) -> EventEncoding:
        _, frames, _ = _validate_outer_inputs(
            frame_tokens=frame_tokens,
            anchor_valid=anchor_valid,
            proprio=proprio,
            segment_start=segment_start,
            segment_end=segment_end,
            twist=twist,
            grip_start=grip_start,
            grip_end=grip_end,
            event_kind=event_kind,
            event_valid=event_valid,
            local_index=local_index,
            local_count=local_count,
            frame_width=self.geometry_config.width,
        )
        frame_valid = _validate_finite_inputs(
            frame_tokens,
            anchor_valid,
            proprio,
            twist,
            grip_start,
            grip_end,
            event_valid,
        )
        _validate_landmarks(segment_start, segment_end, twist, event_kind, event_valid)
        _validate_indices_and_order(
            segment_start=segment_start,
            segment_end=segment_end,
            event_kind=event_kind,
            event_valid=event_valid,
            local_index=local_index,
            local_count=local_count,
            frames=frames,
        )
        _validate_referenced_frames(
            frame_valid,
            segment_start,
            segment_end,
            event_valid,
        )
        (
            frame_tokens,
            proprio,
            segment_start,
            segment_end,
            twist,
            grip_start,
            grip_end,
            event_kind,
            local_index,
            local_count,
        ) = _sanitize_padding(
            frame_tokens=frame_tokens,
            anchor_valid=anchor_valid,
            proprio=proprio,
            frame_valid=frame_valid,
            segment_start=segment_start,
            segment_end=segment_end,
            twist=twist,
            grip_start=grip_start,
            grip_end=grip_end,
            event_kind=event_kind,
            event_valid=event_valid,
            local_index=local_index,
            local_count=local_count,
        )

        descriptors = _encode_frames(
            self.frame_mlp,
            frame_tokens,
            anchor_valid,
            proprio,
            frame_valid,
        )
        d_start, d_end, d_mean = _build_segment_features(
            descriptors,
            segment_start,
            segment_end,
            event_valid,
        )
        raw_features = event_features(
            d_start,
            d_end,
            d_mean,
            twist,
            grip_start,
            grip_end,
        )
        event_values = self.event_mlp(raw_features)
        local_index_float = local_index.to(dtype=frame_tokens.dtype)
        local_count_float = local_count.to(dtype=frame_tokens.dtype)
        order_input = torch.stack(
            (local_index_float / local_count_float, torch.ones_like(local_index_float) / local_count_float),
            dim=-1,
        )
        order_input = torch.where(event_valid[..., None], order_input, torch.zeros_like(order_input))
        order = self.order_mlp(order_input)
        kind = self.kind_embedding(event_kind)
        tokens = event_values + order + kind
        tokens = torch.where(event_valid[..., None], tokens, torch.zeros_like(tokens))
        for block in self.blocks:
            tokens = block(tokens, event_valid)
        tokens = torch.where(event_valid[..., None], tokens, torch.zeros_like(tokens))
        return EventEncoding(tokens=tokens, valid=event_valid.clone())


__all__ = [
    "END_KIND",
    "EventEncoder",
    "EventEncoding",
    "INTERACTION_KIND",
    "LANDMARK_TWIST_ATOL",
    "START_KIND",
    "event_features",
]
