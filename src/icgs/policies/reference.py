"""Deterministic context-preparation seed and provenance foundations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from icgs.data.preprocessing.events import TimedDemoInput
from icgs.data.preprocessing.native import sample_to_cond_demo, subsample_pcd
from icgs.geometry.transforms import transform_pcd
from icgs.state.randomness import scoped_seed


CONTEXT_RNG_PROTOCOL = "seedsequence-native-choice-v1"


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _exact_tuple(value: object, name: str) -> tuple:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    return value


def _nonempty_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def split_context_seeds(
    context_seed: int,
    *,
    demo_count: int,
    event_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive ordered full-demo and reserved window-slot seeds."""

    root_seed = _nonnegative_integer(context_seed, "context_seed")
    demos = _positive_integer(demo_count, "demo_count")
    events = _nonnegative_integer(event_count, "event_count")
    children = np.random.SeedSequence(root_seed).spawn(demos + events)
    seeds = tuple(int(child.generate_state(1)[0]) for child in children)
    return seeds[:demos], seeds[demos:]


def materialize_indexed_native_demo(
    demo: TimedDemoInput,
    boundary_indices: tuple[int, ...],
    *,
    point_seed: int,
    native_waypoint_count: int,
    native_point_count: int,
) -> Mapping[str, tuple]:
    """Materialize caller-selected measured boundaries in native demo form."""

    if not isinstance(demo, TimedDemoInput):
        raise TypeError("demo must be a TimedDemoInput")
    indices_input = _exact_tuple(boundary_indices, "boundary_indices")
    if not indices_input:
        raise ValueError("boundary_indices must be nonempty")
    indices = tuple(
        _nonnegative_integer(value, f"boundary_indices[{index}]")
        for index, value in enumerate(indices_input)
    )
    if any(current > following for current, following in zip(indices, indices[1:])):
        raise ValueError("boundary_indices must be sorted")
    if len(set(indices)) != len(indices):
        raise ValueError("boundary_indices must be unique")

    waypoint_count = _positive_integer(
        native_waypoint_count, "native_waypoint_count"
    )
    point_count = _positive_integer(native_point_count, "native_point_count")
    seed = _nonnegative_integer(point_seed, "point_seed")
    if len(indices) != waypoint_count:
        raise ValueError(
            "boundary_indices length must equal native_waypoint_count"
        )

    observations = (
        demo.transitions[0].before.observation,
        *(transition.after.observation for transition in demo.transitions),
    )
    if indices[-1] >= len(observations):
        raise ValueError("boundary_indices must be within measured observation range")
    selected = tuple(observations[index] for index in indices)

    with scoped_seed(seed, device="cpu"):
        local_points = tuple(
            transform_pcd(
                subsample_pcd(observation.points, point_count),
                np.linalg.inv(observation.T_w_e),
            )
            for observation in selected
        )

    return {
        "obs": local_points,
        "grips": tuple(observation.grip for observation in selected),
        "T_w_es": tuple(observation.T_w_e for observation in selected),
    }


def _validated_native_demo_result(
    result: object,
    *,
    native_waypoint_count: int,
) -> Mapping[str, tuple]:
    if not isinstance(result, Mapping):
        raise TypeError("native result must be a mapping")
    required = {"obs", "grips", "T_w_es"}
    if set(result) != required:
        raise ValueError("native result fields must be exactly obs, grips, T_w_es")

    values = tuple(result[field] for field in ("obs", "grips", "T_w_es"))
    if any(not isinstance(value, (list, tuple)) for value in values):
        raise TypeError("native result fields must each be a sequence")
    lengths = tuple(len(value) for value in values)
    if len(set(lengths)) != 1:
        raise ValueError("native result field cardinalities must agree")
    if lengths[0] != native_waypoint_count:
        raise ValueError(
            "native result cardinality must equal native_waypoint_count"
        )
    return {
        field: tuple(result[field])
        for field in ("obs", "grips", "T_w_es")
    }


def materialize_full_native_demo(
    demo: TimedDemoInput,
    *,
    point_seed: int,
    native_waypoint_count: int,
    native_point_count: int,
) -> Mapping[str, tuple]:
    """Materialize one measured demo through native waypoint selection once."""

    if not isinstance(demo, TimedDemoInput):
        raise TypeError("demo must be a TimedDemoInput")
    seed = _nonnegative_integer(point_seed, "point_seed")
    waypoint_count = _positive_integer(
        native_waypoint_count, "native_waypoint_count"
    )
    point_count = _positive_integer(native_point_count, "native_point_count")

    observations = (
        demo.transitions[0].before.observation,
        *(transition.after.observation for transition in demo.transitions),
    )
    raw_demo = {
        "pcds": tuple(observation.points for observation in observations),
        "grips": tuple(observation.grip for observation in observations),
        "T_w_es": tuple(observation.T_w_e for observation in observations),
    }
    with scoped_seed(seed, device="cpu"):
        result = sample_to_cond_demo(
            raw_demo,
            waypoint_count,
            num_points=point_count,
        )
    return _validated_native_demo_result(
        result,
        native_waypoint_count=waypoint_count,
    )


@dataclass(frozen=True)
class ContextPreparationRecord:
    """Immutable provenance for one prepared reference context."""

    context_seed: int
    full_demo_seeds: tuple[int, ...]
    window_slot_seeds: tuple[int, ...]
    rng_protocol: str
    full_context_id: str
    window_context_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.context_seed, "context_seed")
        full_demo_seeds = _exact_tuple(self.full_demo_seeds, "full_demo_seeds")
        window_slot_seeds = _exact_tuple(
            self.window_slot_seeds, "window_slot_seeds"
        )
        window_context_ids = _exact_tuple(
            self.window_context_ids, "window_context_ids"
        )

        if not full_demo_seeds:
            raise ValueError("full_demo_seeds must be nonempty")
        for index, seed in enumerate(full_demo_seeds):
            _nonnegative_integer(seed, f"full_demo_seeds[{index}]")
        for index, seed in enumerate(window_slot_seeds):
            _nonnegative_integer(seed, f"window_slot_seeds[{index}]")

        if (
            not isinstance(self.rng_protocol, str)
            or self.rng_protocol != CONTEXT_RNG_PROTOCOL
        ):
            raise ValueError(
                f"rng_protocol must be exactly {CONTEXT_RNG_PROTOCOL!r}"
            )
        _nonempty_identifier(self.full_context_id, "full_context_id")
        if len(window_slot_seeds) != len(window_context_ids):
            raise ValueError(
                "window_slot_seeds and window_context_ids must align exactly"
            )
        for index, identifier in enumerate(window_context_ids):
            if identifier is not None:
                _nonempty_identifier(identifier, f"window_context_ids[{index}]")


__all__ = (
    "CONTEXT_RNG_PROTOCOL",
    "ContextPreparationRecord",
    "materialize_full_native_demo",
    "materialize_indexed_native_demo",
    "split_context_seeds",
)
