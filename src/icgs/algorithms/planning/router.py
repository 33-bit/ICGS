"""Pure P06 routing arithmetic and structural native-window selection.

This module does not construct native contexts, own sessions, split seeds, or
call a policy.  Final native-window availability is injected by its state owner;
debounced gripper-transition indices are injected by P05 preprocessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral, Real

import torch
from torch import Tensor

from icgs.configuration.method import MethodConfig, RouterConfig
from icgs.contracts.method import SegmentRef


def _router_config(config: RouterConfig | MethodConfig) -> RouterConfig:
    if isinstance(config, MethodConfig):
        return config.router
    if isinstance(config, RouterConfig):
        return config
    raise TypeError("config must be RouterConfig or MethodConfig")


def _probability(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"router.{name} must be a finite probability")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"router.{name} must be a finite probability in [0,1]")
    if positive and result == 0.0:
        raise ValueError(f"router.{name} must be strictly positive")
    return result


def _neighbor_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"router.{name} must be a nonnegative integer")
    return int(value)


def _validate_probability_inputs(
    event_alpha: Tensor,
    eligible: Tensor,
    native_window_valid: Tensor,
) -> None:
    values = (
        (event_alpha, "event_alpha"),
        (eligible, "eligible"),
        (native_window_valid, "native_window_valid"),
    )
    for value, name in values:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
            raise ValueError(f"{name} must have nonempty shape [B,L]")
    if eligible.shape != event_alpha.shape or native_window_valid.shape != event_alpha.shape:
        raise ValueError("event_alpha, eligible and native_window_valid must have the same shape")
    if not event_alpha.is_floating_point():
        raise TypeError("event_alpha must use a floating dtype")
    if not eligible.is_floating_point():
        raise TypeError("eligible must use a floating dtype")
    if native_window_valid.dtype != torch.bool:
        raise TypeError("native_window_valid must use bool dtype")
    if eligible.dtype != event_alpha.dtype:
        raise TypeError("event_alpha and eligible must have the same dtype")
    if eligible.device != event_alpha.device or native_window_valid.device != event_alpha.device:
        raise ValueError("router inputs must use the same device")
    for value, name in ((event_alpha, "event_alpha"), (eligible, "eligible")):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
        if not bool(((value >= 0.0) & (value <= 1.0)).all().item()):
            raise ValueError(f"{name} must contain only values in [0,1]")


def router_probabilities(
    event_alpha: Tensor,
    eligible: Tensor,
    native_window_valid: Tensor,
    config: RouterConfig | MethodConfig,
) -> Tensor:
    """Return batched full-context then event-window route probabilities."""

    resolved = _router_config(config)
    full_probability = _probability(
        resolved.full_context_probability, "full_context_probability"
    )
    epsilon = _probability(
        resolved.probability_epsilon, "probability_epsilon", positive=True
    )
    threshold = _probability(
        resolved.fallback_threshold, "fallback_threshold", positive=True
    )
    _validate_probability_inputs(event_alpha, eligible, native_window_valid)

    values = torch.where(
        native_window_valid,
        (event_alpha + epsilon) * eligible,
        torch.zeros_like(event_alpha),
    )
    mass = values.sum(dim=-1, keepdim=True)
    fallback = mass < threshold
    safe_mass = torch.where(fallback, torch.ones_like(mass), mass)
    normalized = values / safe_mass
    routed = (1.0 - full_probability) * normalized
    probabilities = torch.cat(
        (torch.full_like(mass, full_probability), routed), dim=-1
    )
    full_fallback = torch.cat(
        (torch.ones_like(mass), torch.zeros_like(values)), dim=-1
    )
    return torch.where(fallback, full_fallback, probabilities)


def _validate_interactions(
    target: SegmentRef, demo_interactions: tuple[SegmentRef, ...]
) -> int:
    if not isinstance(target, SegmentRef):
        raise TypeError("target must be a SegmentRef")
    if target.kind != "interaction":
        raise ValueError("target must be an interaction")
    if not target.valid_action_window:
        raise ValueError("target must be a structurally eligible interaction")
    if not isinstance(demo_interactions, tuple) or not demo_interactions:
        raise TypeError("demo_interactions must be a nonempty tuple")
    if not all(isinstance(reference, SegmentRef) for reference in demo_interactions):
        raise TypeError("demo_interactions must contain only SegmentRef records")
    if not all(reference.kind == "interaction" for reference in demo_interactions):
        raise ValueError("demo_interactions must contain only interaction refs")
    if not all(
        reference.demo_content_hash == target.demo_content_hash
        for reference in demo_interactions
    ):
        raise ValueError("every demo_interactions entry must match target demo_content_hash")
    if any(reference.a >= reference.b for reference in demo_interactions):
        raise ValueError("every interaction must have positive boundary duration")
    matches = tuple(index for index, reference in enumerate(demo_interactions) if reference == target)
    if len(matches) != 1:
        raise ValueError("target must occur exactly once in demo_interactions")
    for current, following in zip(demo_interactions, demo_interactions[1:]):
        if current.b != following.a:
            raise ValueError(
                "demo_interactions must form a contiguous shared-endpoint partition"
            )
    return matches[0]


def _validate_grip_transitions(
    values: Sequence[int], *, first_boundary: int, final_boundary: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("grip_transition_indices must be a sequence")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("grip_transition_indices must contain only integer indices")
        result.append(int(value))
    transitions = tuple(result)
    if transitions != tuple(sorted(transitions)):
        raise ValueError("grip_transition_indices must be sorted")
    if len(set(transitions)) != len(transitions):
        raise ValueError("grip_transition_indices must be unique")
    if any(value < first_boundary or value > final_boundary for value in transitions):
        raise ValueError("grip_transition_indices must lie within the full demo range")
    return transitions


def _positive_waypoint_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError("native_waypoint_count must be a positive integer")
    return int(value)


def _uniform_unused_indices(unused: tuple[int, ...], count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    if count == 1:
        return (unused[(len(unused) - 1) // 2],)
    denominator = count - 1
    final_rank = len(unused) - 1
    selected = []
    for index in range(count):
        lower, remainder = divmod(index * final_rank, denominator)
        rank = lower + int(2 * remainder > denominator)
        selected.append(unused[rank])
    return tuple(selected)


def select_window_indices(
    target: SegmentRef,
    demo_interactions: tuple[SegmentRef, ...],
    grip_transition_indices: Sequence[int],
    config: RouterConfig | MethodConfig,
    *,
    native_waypoint_count: int,
) -> tuple[int, ...] | None:
    """Select one exact, unique native window or return ``None`` if infeasible."""

    resolved = _router_config(config)
    before = _neighbor_count(resolved.neighbor_events_before, "neighbor_events_before")
    after = _neighbor_count(resolved.neighbor_events_after, "neighbor_events_after")
    target_index = _validate_interactions(target, demo_interactions)
    waypoint_count = _positive_waypoint_count(native_waypoint_count)
    transitions = _validate_grip_transitions(
        grip_transition_indices,
        first_boundary=demo_interactions[0].a,
        final_boundary=demo_interactions[-1].b,
    )

    selected = demo_interactions[
        max(0, target_index - before) : min(
            len(demo_interactions), target_index + after + 1
        )
    ]
    span_start, span_end = selected[0].a, selected[-1].b
    mandatory = {
        boundary
        for reference in selected
        for boundary in (reference.a, reference.b)
    }
    mandatory.update(
        boundary for boundary in transitions if span_start <= boundary <= span_end
    )

    available_count = span_end - span_start + 1
    if len(mandatory) > waypoint_count or available_count < waypoint_count:
        return None
    unused = tuple(
        boundary
        for boundary in range(span_start, span_end + 1)
        if boundary not in mandatory
    )
    fill_count = waypoint_count - len(mandatory)
    if fill_count > len(unused):
        return None
    fill = _uniform_unused_indices(unused, fill_count)
    return tuple(sorted((*mandatory, *fill)))


__all__ = ["router_probabilities", "select_window_indices"]
