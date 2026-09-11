"""Tensor-only recurrent task tracking for the added method path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from icgs.configuration.method import MethodConfig
from icgs.models.layers.method import MaskedCrossAttentionBlock, masked_mean


def task_event_features(event_tokens: Tensor, r: Tensor) -> Tensor:
    """Concatenate event, recurrent and interaction terms without masking."""

    if not torch.is_tensor(event_tokens) or event_tokens.ndim != 3:
        raise ValueError("event_tokens must have shape [B,L,W]")
    if event_tokens.shape[0] <= 0 or event_tokens.shape[1] <= 0 or event_tokens.shape[2] <= 0:
        raise ValueError("event_tokens must have nonempty shape [B,L,W]")
    if not torch.is_tensor(r) or r.shape != (event_tokens.shape[0], event_tokens.shape[2]):
        raise ValueError("r must have shape [B,W]")
    if not event_tokens.is_floating_point() or not r.is_floating_point():
        raise TypeError("event_tokens and r must use floating dtypes")
    if event_tokens.dtype != r.dtype or event_tokens.device != r.device:
        raise ValueError("event_tokens and r must share dtype and device")
    if not bool(torch.isfinite(event_tokens).all().item()) or not bool(
        torch.isfinite(r).all().item()
    ):
        raise ValueError("event_tokens and r must contain only finite values")
    expanded_r = r[:, None, :].expand_as(event_tokens)
    return torch.cat((event_tokens, expanded_r, event_tokens * expanded_r), dim=-1)


@dataclass(frozen=True)
class TaskEncoding:
    """Transient differentiable outputs; probability/state ownership is external."""

    r: Tensor
    alignment_logits: Tensor
    event_logits: Tensor
    event_valid: Tensor


class TaskTracker(nn.Module):
    """Update recurrent task features from masked physical and event tokens."""

    def __init__(self, config: MethodConfig):
        super().__init__()
        if not isinstance(config, MethodConfig):
            raise TypeError("config must be an explicit resolved MethodConfig")
        width = config.event.width
        if config.geometry.width != width or config.memory.width != width:
            raise ValueError("geometry, memory and event width must match")
        if config.tracker.memory_slots != 1:
            raise ValueError("tracker.memory_slots must be exactly one")
        if (
            isinstance(config.tracker.attention_layers, bool)
            or not isinstance(config.tracker.attention_layers, int)
            or config.tracker.attention_layers <= 0
        ):
            raise ValueError("tracker.attention_layers must be a positive integer")
        heads = config.neural.attention_heads
        if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0 or width % heads:
            raise ValueError("neural attention heads must divide tracker width")

        self.width = width
        self.config = config
        self.scene_projection = nn.Linear(width, width)
        self.cross_attention_blocks = nn.ModuleList(
            MaskedCrossAttentionBlock(width=width, neural_config=config.neural)
            for _ in range(config.tracker.attention_layers)
        )
        self.recurrent = nn.GRUCell(width, width)
        self.alignment_query = nn.Linear(width, width)
        self.alignment_key = nn.Linear(width, width)
        self.null_alignment = nn.Linear(width, 1)
        self.event_head = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, 3),
        )

    def forward(
        self,
        physical_tokens: Tensor,
        physical_valid: Tensor,
        event_tokens: Tensor,
        event_valid: Tensor,
        previous_r: Tensor,
    ) -> TaskEncoding:
        self._validate_inputs(
            physical_tokens,
            physical_valid,
            event_tokens,
            event_valid,
            previous_r,
        )
        physical_mask = physical_valid[..., None]
        event_mask = event_valid[..., None]
        physical_tokens = torch.where(
            physical_mask, physical_tokens, torch.zeros_like(physical_tokens)
        )
        event_tokens = torch.where(
            event_mask, event_tokens, torch.zeros_like(event_tokens)
        )

        scene = masked_mean(physical_tokens, physical_valid, dim=1)
        query = previous_r + self.scene_projection(scene)
        query = query[:, None, :]
        query_valid = torch.ones(
            query.shape[:2], dtype=torch.bool, device=query.device
        )
        keys = torch.cat((physical_tokens, event_tokens), dim=1)
        key_valid = torch.cat((physical_valid, event_valid), dim=1)
        for block in self.cross_attention_blocks:
            query = block(query, query_valid, keys, key_valid)
        r = self.recurrent(query[:, 0], previous_r)

        alignment_event = (
            self.alignment_key(event_tokens) * self.alignment_query(r)[:, None, :]
        ).sum(dim=-1) / math.sqrt(self.width)
        alignment_event = torch.where(
            event_valid,
            alignment_event,
            torch.full_like(alignment_event, torch.finfo(alignment_event.dtype).min),
        )
        alignment_logits = torch.cat((alignment_event, self.null_alignment(r)), dim=1)

        event_logits = self.event_head(task_event_features(event_tokens, r))
        event_logits = torch.where(event_mask, event_logits, torch.zeros_like(event_logits))
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (r, alignment_logits, event_logits)
        ):
            raise ValueError("TaskTracker produced nonfinite outputs")
        return TaskEncoding(r, alignment_logits, event_logits, event_valid)

    def _validate_inputs(
        self,
        physical_tokens: Tensor,
        physical_valid: Tensor,
        event_tokens: Tensor,
        event_valid: Tensor,
        previous_r: Tensor,
    ) -> None:
        if (
            not torch.is_tensor(physical_tokens)
            or physical_tokens.ndim != 3
            or physical_tokens.shape[0] <= 0
            or physical_tokens.shape[1] <= 0
            or physical_tokens.shape[2] != self.width
        ):
            raise ValueError(f"physical_tokens must have nonempty shape [B,S,{self.width}]")
        batch = physical_tokens.shape[0]
        if (
            not torch.is_tensor(event_tokens)
            or event_tokens.ndim != 3
            or event_tokens.shape[0] != batch
            or event_tokens.shape[1] <= 0
            or event_tokens.shape[2] != self.width
        ):
            raise ValueError(f"event_tokens must have nonempty shape [B,L,{self.width}]")
        if not torch.is_tensor(physical_valid) or physical_valid.shape != physical_tokens.shape[:2]:
            raise ValueError("physical_valid must have shape [B,S]")
        if not torch.is_tensor(event_valid) or event_valid.shape != event_tokens.shape[:2]:
            raise ValueError("event_valid must have shape [B,L]")
        if physical_valid.dtype != torch.bool or event_valid.dtype != torch.bool:
            raise TypeError("physical_valid and event_valid must use torch.bool dtype")
        if not torch.is_tensor(previous_r) or previous_r.shape != (batch, self.width):
            raise ValueError(f"previous_r must have shape [B,{self.width}]")
        if not all(
            value.is_floating_point()
            for value in (physical_tokens, event_tokens, previous_r)
        ):
            raise TypeError("tracker feature tensors must use floating dtypes")
        if not (
            physical_tokens.dtype == event_tokens.dtype == previous_r.dtype
            and physical_tokens.device == event_tokens.device == previous_r.device
        ):
            raise ValueError("tracker feature tensors must share dtype and device")
        if (
            physical_valid.device != physical_tokens.device
            or event_valid.device != event_tokens.device
        ):
            raise ValueError("tracker masks must share their feature tensor device")
        if bool((~physical_valid).all(dim=1).any().item()):
            raise ValueError("TaskTracker requires at least one valid physical token")

        physical_values = physical_tokens.masked_select(
            physical_valid[..., None].expand_as(physical_tokens)
        )
        event_values = event_tokens.masked_select(
            event_valid[..., None].expand_as(event_tokens)
        )
        if not bool(torch.isfinite(physical_values).all().item()):
            raise ValueError("valid physical_tokens must contain only finite values")
        if not bool(torch.isfinite(event_values).all().item()):
            raise ValueError("valid event_tokens must contain only finite values")
        if not bool(torch.isfinite(previous_r).all().item()):
            raise ValueError("previous_r must contain only finite values")


__all__ = ["TaskEncoding", "TaskTracker", "task_event_features"]
