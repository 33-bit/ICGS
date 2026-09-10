"""Observable P05 demonstration segmentation.

Segmentation consumes only measured end-effector poses and gripper states from
executed transitions.  Commands, program semantics, simulator-only state, and
native Instant Policy preprocessing are outside this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, degrees, isclose, isfinite
from numbers import Real

import numpy as np

from icgs.configuration.method import EventConfig, MethodConfig
from icgs.contracts.method import ExecutedTransition, SegmentRef


class EventOverflowError(ValueError):
    """Raised when protected boundaries cannot fit the interaction cap."""


@dataclass(frozen=True)
class TimedDemoInput:
    """Validation-only immutable input for one raw timed demonstration."""

    transitions: tuple[ExecutedTransition, ...]
    demo_content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.transitions, tuple):
            raise TypeError("transitions must be a tuple")
        if not self.transitions:
            raise ValueError("transitions must be non-empty")
        if not all(isinstance(transition, ExecutedTransition) for transition in self.transitions):
            raise TypeError("all transitions must be ExecutedTransition")
        if not isinstance(self.demo_content_hash, str):
            raise TypeError("demo_content_hash must be a string")
        if not self.demo_content_hash.strip():
            raise ValueError("demo_content_hash must be non-empty")

        for transition in self.transitions:
            if transition.after.boundary != transition.before.boundary + 1:
                raise ValueError("transition boundaries must be adjacent")
        for current, following in zip(self.transitions, self.transitions[1:]):
            if current.after.boundary != following.before.boundary:
                raise ValueError("transition chain must be contiguous")


def _event_config(config: EventConfig | MethodConfig) -> EventConfig:
    if isinstance(config, MethodConfig):
        return config.event
    if isinstance(config, EventConfig):
        return config
    raise TypeError("config must be EventConfig or MethodConfig")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"event.{name} must be a positive integer")
    return value


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"event.{name} must be a positive finite number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"event.{name} must be a positive finite number")
    return result


def _validate_segmentation_config(config: EventConfig) -> None:
    _positive_integer(config.debounce_frames, "debounce_frames")
    _positive_integer(config.max_intervals, "max_intervals")
    _positive_integer(config.min_segment_intervals, "min_segment_intervals")
    _positive_integer(config.max_interactions, "max_interactions")
    _positive_real(config.translation_boundary_m, "translation_boundary_m")
    _positive_real(config.rotation_boundary_deg, "rotation_boundary_deg")
    if config.landmarks != 2:
        raise ValueError("event.landmarks must be exactly two start/end landmarks")


def _grip(value: object, index: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if not isinstance(value, Real):
        raise ValueError(f"grip {index} must be 0 or 1")
    result = float(value)
    if not isfinite(result) or result not in (0.0, 1.0):
        raise ValueError(f"grip {index} must be 0 or 1")
    return int(result)


def debounced_grip_boundaries(
    grips: Sequence[object], config: EventConfig | MethodConfig
) -> tuple[int, ...]:
    """Return confirmation-frame indices for stable measured grip changes."""

    if isinstance(grips, (str, bytes)) or not isinstance(grips, Sequence) or not grips:
        raise ValueError("grips must be a non-empty sequence")
    resolved = _event_config(config)
    debounce_frames = _positive_integer(resolved.debounce_frames, "debounce_frames")
    values = tuple(_grip(value, index) for index, value in enumerate(grips))

    stable = values[0]
    candidate: int | None = None
    candidate_count = 0
    boundaries: list[int] = []
    for index in range(1, len(values)):
        current = values[index]
        if current == stable:
            candidate = None
            candidate_count = 0
            continue
        if current == candidate:
            candidate_count += 1
        else:
            candidate = current
            candidate_count = 1
        if candidate_count >= debounce_frames:
            stable = current
            boundaries.append(index)
            candidate = None
            candidate_count = 0
    return tuple(boundaries)


def _measured_observations(demo: TimedDemoInput) -> tuple[object, ...]:
    """Extract N+1 measured observations without reading transition commands."""

    return (
        demo.transitions[0].before.observation,
        *(transition.after.observation for transition in demo.transitions),
    )


def _rotation_step_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    relative = previous.T @ current
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _strictly_exceeds(value: float, threshold: float) -> bool:
    # Numerical guard only: preserve the mathematically strict ">" threshold
    # against floating-point noise around exact boundary values.
    return value > threshold and not isclose(value, threshold, rel_tol=0.0, abs_tol=1e-9)


def _initial_boundaries(
    observations: tuple[object, ...], config: EventConfig
) -> tuple[list[int], set[int]]:
    grip_boundaries = set(
        debounced_grip_boundaries(tuple(observation.grip for observation in observations), config)
    )
    boundaries = [0]
    protected: set[int] = set()
    cumulative_translation = 0.0
    cumulative_rotation = 0.0
    previous_boundary = 0

    for index in range(1, len(observations)):
        previous_pose = observations[index - 1].T_w_e
        current_pose = observations[index].T_w_e
        cumulative_translation += float(np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3]))
        cumulative_rotation += _rotation_step_degrees(previous_pose[:3, :3], current_pose[:3, :3])

        is_grip_boundary = index in grip_boundaries
        is_motion_boundary = (
            _strictly_exceeds(cumulative_translation, config.translation_boundary_m)
            or _strictly_exceeds(cumulative_rotation, config.rotation_boundary_deg)
        )
        is_time_boundary = index - previous_boundary >= config.max_intervals
        if is_grip_boundary or is_motion_boundary or is_time_boundary:
            boundaries.append(index)
            if is_grip_boundary:
                protected.add(index)
            previous_boundary = index
            cumulative_translation = 0.0
            cumulative_rotation = 0.0

    final_boundary = len(observations) - 1
    if boundaries[-1] != final_boundary:
        boundaries.append(final_boundary)
    return boundaries, protected


def _merge_short_segments(boundaries: list[int], protected: set[int], minimum: int) -> None:
    while True:
        short_segments = sorted(
            (
                (boundaries[index + 1] - boundaries[index], boundaries[index], index)
                for index in range(len(boundaries) - 1)
                if boundaries[index + 1] - boundaries[index] < minimum
            ),
            key=lambda item: (item[0], item[1]),
        )
        merged = False
        for _, _, index in short_segments:
            choices: list[tuple[int, int, int]] = []
            if index > 0 and boundaries[index] not in protected:
                left_duration = boundaries[index] - boundaries[index - 1]
                choices.append((left_duration, boundaries[index - 1], index))
            if index < len(boundaries) - 2 and boundaries[index + 1] not in protected:
                right_duration = boundaries[index + 2] - boundaries[index + 1]
                choices.append((right_duration, boundaries[index + 1], index + 1))
            if choices:
                _, _, remove_index = min(choices, key=lambda item: (item[0], item[1]))
                boundaries.pop(remove_index)
                merged = True
                break
        if not merged:
            return


def _cap_interactions(boundaries: list[int], protected: set[int], cap: int) -> None:
    while len(boundaries) - 1 > cap:
        choices = [
            (boundaries[index + 1] - boundaries[index - 1], boundaries[index - 1], index)
            for index in range(1, len(boundaries) - 1)
            if boundaries[index] not in protected
        ]
        if not choices:
            raise EventOverflowError("event overflow: protected boundaries cannot fit interaction cap")
        _, _, remove_index = min(choices, key=lambda item: (item[0], item[1]))
        boundaries.pop(remove_index)


def segment_demo(
    demo: TimedDemoInput, config: EventConfig | MethodConfig
) -> tuple[SegmentRef, ...]:
    """Segment one timed demo using measured poses/grips and resolved settings."""

    if not isinstance(demo, TimedDemoInput):
        raise TypeError("demo must be a TimedDemoInput")
    if config is None:
        raise TypeError("config must be an explicitly resolved EventConfig or MethodConfig")
    resolved = _event_config(config)
    _validate_segmentation_config(resolved)
    observations = _measured_observations(demo)
    boundaries, protected = _initial_boundaries(observations, resolved)
    _merge_short_segments(boundaries, protected, resolved.min_segment_intervals)
    _cap_interactions(boundaries, protected, resolved.max_interactions)

    segments = [SegmentRef(demo.demo_content_hash, 0, 0, "start", False)]
    segments.extend(
        SegmentRef(demo.demo_content_hash, start, end, "interaction", True)
        for start, end in zip(boundaries, boundaries[1:])
    )
    final_boundary = len(observations) - 1
    segments.append(SegmentRef(demo.demo_content_hash, final_boundary, final_boundary, "end", False))
    return tuple(segments)


__all__ = [
    "EventOverflowError",
    "TimedDemoInput",
    "debounced_grip_boundaries",
    "segment_demo",
]
