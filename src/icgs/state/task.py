"""Owned causal task state derived from transient tracker outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from icgs.models.memories.task import TaskEncoding
from icgs.state.method_context import EventMemory


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _owned_float_tensor(value: Any, name: str, ndim: int) -> Tensor:
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value.clone()


def _validate_probability(value: Tensor, name: str) -> None:
    if bool(((value < 0) | (value > 1)).any().item()):
        raise ValueError(f"{name} values must be in [0,1]")


def _probability_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    # Numerical guard only: probability normalization has no configurable tolerance.
    epsilon = float(torch.finfo(dtype).eps)
    tolerance = 8.0 * epsilon
    return tolerance, tolerance


@dataclass(frozen=True)
class TaskState:
    """Owned online task probabilities with exact causal and context lineage."""

    r: Tensor
    alpha: Tensor
    rho: Tensor
    nu: Tensor
    eligible: Tensor
    boundary: int
    context_fingerprints: tuple[str, ...]
    tracker_id: str

    def __post_init__(self) -> None:
        r = _owned_float_tensor(self.r, "r", 2)
        alpha = _owned_float_tensor(self.alpha, "alpha", 2)
        rho = _owned_float_tensor(self.rho, "rho", 2)
        nu = _owned_float_tensor(self.nu, "nu", 2)
        eligible = _owned_float_tensor(self.eligible, "eligible", 2)

        batch, width = r.shape
        if batch <= 0 or width <= 0:
            raise ValueError("r must have nonempty shape [B,W]")
        if alpha.shape[0] != batch or alpha.shape[1] <= 1:
            raise ValueError("alpha must have shape [B,L+1] with nonempty L")
        event_count = alpha.shape[1] - 1
        for name, value in (("rho", rho), ("nu", nu), ("eligible", eligible)):
            if value.shape != (batch, event_count):
                raise ValueError(f"{name} must have shape [B,L] aligned with alpha")
        for name, value in (
            ("alpha", alpha),
            ("rho", rho),
            ("nu", nu),
            ("eligible", eligible),
        ):
            if value.dtype != r.dtype or value.device != r.device:
                raise ValueError(f"{name} must share r dtype and device")
            _validate_probability(value, name)

        atol, rtol = _probability_tolerance(alpha.dtype)
        if not bool(
            torch.allclose(
                alpha.sum(dim=-1),
                torch.ones(batch, dtype=alpha.dtype, device=alpha.device),
                atol=atol,
                rtol=rtol,
            )
        ):
            raise ValueError("alpha rows must sum to one")

        if isinstance(self.boundary, bool) or not isinstance(self.boundary, int):
            raise TypeError("boundary must be a nonnegative integer")
        if self.boundary < 0:
            raise ValueError("boundary must be a nonnegative integer")
        if not isinstance(self.context_fingerprints, tuple):
            raise TypeError("context_fingerprints must be a tuple")
        if len(self.context_fingerprints) != batch:
            raise ValueError(
                "context_fingerprints must have exactly the TaskState batch dimension"
            )
        context_fingerprints = tuple(
            _identifier(value, "context fingerprint")
            for value in self.context_fingerprints
        )
        tracker_id = _identifier(self.tracker_id, "tracker_id")

        object.__setattr__(self, "r", r)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "context_fingerprints", context_fingerprints)
        object.__setattr__(self, "tracker_id", tracker_id)

    def branch_copy(self) -> "TaskState":
        """Return independently stored tensors while preserving autograd paths."""

        return TaskState(
            self.r,
            self.alpha,
            self.rho,
            self.nu,
            self.eligible,
            self.boundary,
            self.context_fingerprints,
            self.tracker_id,
        )


def initial_task_memory(events: EventMemory) -> Tensor:
    """Create one zero recurrent row per owned event-memory batch row."""

    if not isinstance(events, EventMemory):
        raise TypeError("events must be an EventMemory")
    return events.tokens.new_zeros((events.tokens.shape[0], events.tokens.shape[2]))


def _masked_softmax_with_null(
    alignment_logits: Tensor,
    event_valid: Tensor,
) -> Tensor:
    """Normalize event plus final-null logits and write masked events as zero."""

    if not torch.is_tensor(alignment_logits) or alignment_logits.ndim != 2:
        raise ValueError("alignment_logits must have shape [B,L+1]")
    if not alignment_logits.is_floating_point():
        raise TypeError("alignment_logits must use a floating dtype")
    if not torch.is_tensor(event_valid) or event_valid.ndim != 2:
        raise ValueError("event_valid must have shape [B,L]")
    if event_valid.dtype != torch.bool:
        raise TypeError("event_valid must use torch.bool dtype")
    if alignment_logits.shape != (event_valid.shape[0], event_valid.shape[1] + 1):
        raise ValueError("alignment_logits must have shape [B,L+1]")
    if event_valid.device != alignment_logits.device:
        raise ValueError("event_valid must share alignment_logits device")
    if not bool(torch.isfinite(alignment_logits).all().item()):
        raise ValueError("alignment_logits must contain only finite values")

    full_valid = torch.cat(
        (
            event_valid,
            torch.ones(
                (event_valid.shape[0], 1),
                dtype=torch.bool,
                device=event_valid.device,
            ),
        ),
        dim=-1,
    )
    masked_logits = torch.where(
        full_valid,
        alignment_logits,
        torch.full_like(alignment_logits, torch.finfo(alignment_logits.dtype).min),
    )
    alpha = torch.softmax(masked_logits, dim=-1)
    alpha = torch.where(full_valid, alpha, torch.zeros_like(alpha))
    return alpha / alpha.sum(dim=-1, keepdim=True)


def _validate_encoding(encoding: TaskEncoding, events: EventMemory) -> None:
    if not isinstance(encoding, TaskEncoding):
        raise TypeError("encoding must be a TaskEncoding")
    if not isinstance(events, EventMemory):
        raise TypeError("events must be an EventMemory")
    batch, event_count, width = events.tokens.shape
    expected = {
        "r": (batch, width),
        "alignment_logits": (batch, event_count + 1),
        "event_logits": (batch, event_count, 3),
    }
    for name, shape in expected.items():
        value = getattr(encoding, name)
        if not torch.is_tensor(value) or value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if value.dtype != events.tokens.dtype or value.device != events.tokens.device:
            raise ValueError(f"{name} must share EventMemory token dtype and device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
    if (
        not torch.is_tensor(encoding.event_valid)
        or encoding.event_valid.shape != events.valid.shape
    ):
        raise ValueError("event_valid must have shape [B,L]")
    if encoding.event_valid.dtype != torch.bool:
        raise TypeError("event_valid must use torch.bool dtype")
    if encoding.event_valid.device != events.valid.device:
        raise ValueError("event_valid must share EventMemory valid device")
    if not torch.equal(encoding.event_valid, events.valid):
        raise ValueError("encoding.event_valid must exactly match EventMemory.valid")

    invalid = ~events.valid
    sentinel = torch.finfo(encoding.alignment_logits.dtype).min
    invalid_alignment = encoding.alignment_logits[:, :-1].masked_select(invalid)
    if bool((invalid_alignment != sentinel).any().item()):
        raise ValueError(
            "invalid event alignment logits must equal the finite mask sentinel"
        )
    expanded_invalid = invalid[..., None].expand_as(encoding.event_logits)
    if bool((encoding.event_logits.masked_select(expanded_invalid) != 0).any().item()):
        raise ValueError("invalid event logits must be exactly zero")


def build_task_state(
    encoding: TaskEncoding,
    events: EventMemory,
    *,
    boundary: int,
    tracker_id: str,
) -> TaskState:
    """Own one transient tracker result using EventMemory as the sole mask owner."""

    _validate_encoding(encoding, events)
    alpha = _masked_softmax_with_null(encoding.alignment_logits, events.valid)
    probabilities = torch.sigmoid(encoding.event_logits)
    zeros = torch.zeros_like(probabilities[..., 0])
    rho = torch.where(events.valid, probabilities[..., 0], zeros)
    nu = torch.where(events.valid, probabilities[..., 1], zeros)
    eligible = torch.where(events.valid, probabilities[..., 2], zeros)
    return TaskState(
        encoding.r,
        alpha,
        rho,
        nu,
        eligible,
        boundary,
        events.fingerprints,
        tracker_id,
    )


def _state_shape(state: TaskState) -> tuple[int, int, int]:
    return state.r.shape[0], state.r.shape[1], state.rho.shape[1]


def validate_task_update(previous: TaskState, current: TaskState) -> None:
    """Reject in-place lineage changes and non-successor causal updates."""

    if not isinstance(previous, TaskState) or not isinstance(current, TaskState):
        raise TypeError("previous and current must be TaskState records")
    if _state_shape(previous) != _state_shape(current):
        raise ValueError("TaskState update must preserve B/W/L shape")
    if previous.context_fingerprints != current.context_fingerprints:
        raise ValueError(
            "TaskState context lineage changed; replay full physical history instead"
        )
    if previous.tracker_id != current.tracker_id:
        raise ValueError("TaskState update must preserve tracker lineage")
    if current.boundary != previous.boundary + 1:
        raise ValueError("TaskState current boundary must be the exact successor")


__all__ = [
    "TaskState",
    "build_task_state",
    "initial_task_memory",
    "validate_task_update",
]
