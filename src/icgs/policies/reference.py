"""Deterministic context-preparation seed and provenance foundations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from icgs.configuration.schema import ExperimentConfig
from icgs.data.preprocessing.events import TimedDemoInput
from icgs.data.preprocessing.native import sample_to_cond_demo, subsample_pcd
from icgs.geometry.transforms import transform_pcd
from icgs.policies.instant_policy import InstantPolicy
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


def _canonical_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256")


def _reference_session_id(reference_id: str, role: str) -> str:
    payload = {
        "domain": "icgs.reference-session",
        "schema": 1,
        "reference_id": reference_id,
        "role": role,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_reference_policy(
    policy: object,
    config: ExperimentConfig,
    *,
    role: str,
    checkpoint_sha256: str,
) -> None:
    if not isinstance(policy, InstantPolicy):
        raise TypeError(f"{role} policy must be an InstantPolicy")
    if policy.graph_config != config.graph:
        raise ValueError(f"{role} graph config must match {role.lower()}_config.graph")
    if policy.runtime != config.runtime:
        raise ValueError(f"{role} runtime config must match {role.lower()}_config.runtime")
    if getattr(policy.sampler, "config", None) != config.sampling:
        raise ValueError(
            f"{role} sampler config must match {role.lower()}_config.sampling"
        )
    if getattr(policy.sampler, "diffusion", None) != config.diffusion:
        raise ValueError(
            f"{role} sampler diffusion must match {role.lower()}_config.diffusion"
        )
    if getattr(policy.objective, "config", None) != config.diffusion:
        raise ValueError(
            f"{role} objective config must match {role.lower()}_config.diffusion"
        )
    network_codec = getattr(policy.network, "codec", None)
    if (
        network_codec is None
        or getattr(policy.sampler, "codec", None) is not network_codec
    ):
        raise ValueError(f"{role} sampler codec must be the policy network codec")
    if getattr(policy.objective, "codec", None) is not network_codec:
        raise ValueError(f"{role} objective codec must be the policy network codec")
    if getattr(policy, "artifact_sha256", None) != checkpoint_sha256:
        raise ValueError(
            f"{role} artifact checkpoint SHA-256 must match checkpoint_sha256"
        )

    sampler_scheduler = getattr(policy.sampler, "noise_scheduler", None)
    objective_scheduler = getattr(policy.objective, "noise_scheduler", None)
    if sampler_scheduler is None or sampler_scheduler is not objective_scheduler:
        raise ValueError(
            f"{role} sampler and objective must share the same noise scheduler"
        )


def _nested_owner(owner: object, path: str, *, role: str) -> object:
    current = owner
    for component in path.split("."):
        if not hasattr(current, component):
            raise ValueError(f"{role} policy must expose {path}")
        current = getattr(current, component)
    return current


def _validate_distinct_policy_owners(d1: InstantPolicy, d2: InstantPolicy) -> None:
    for path in (
        "context_owner",
        "network",
        "network.graph",
        "network.graph.graph",
        "network.codec",
        "sampler",
        "sampler.noise_scheduler",
        "objective",
    ):
        d1_owner = _nested_owner(d1, path, role="D1")
        d2_owner = _nested_owner(d2, path, role="D2")
        if d1_owner is d2_owner:
            raise ValueError(
                f"D1/D2 {path} objects must be distinct; cross-policy alias detected"
            )


def _tensor_storage_intervals(
    network: object,
) -> tuple[tuple[str, str, int, int], ...]:
    tensors = (
        (f"parameter {name}", tensor)
        for name, tensor in network.named_parameters()
    )
    buffers = (
        (f"buffer {name}", tensor)
        for name, tensor in network.named_buffers()
    )
    intervals = []
    for name, tensor in (*tensors, *buffers):
        storage = tensor.untyped_storage()
        if storage.nbytes() > 0:
            start = storage.data_ptr()
            intervals.append(
                (name, str(tensor.device), start, start + storage.nbytes())
            )
    return tuple(intervals)


def _validate_distinct_tensor_storage(
    d1: InstantPolicy,
    d2: InstantPolicy,
) -> None:
    d1_intervals = _tensor_storage_intervals(d1.network)
    d2_intervals = _tensor_storage_intervals(d2.network)
    for d1_name, d1_device, d1_start, d1_end in d1_intervals:
        for d2_name, d2_device, d2_start, d2_end in d2_intervals:
            overlaps = (
                d1_device == d2_device
                and max(d1_start, d2_start) < min(d1_end, d2_end)
            )
            if overlaps:
                raise ValueError(
                    "D1/D2 parameter or buffer storage must not overlap; "
                    f"cross-policy alias detected between {d1_name} and {d2_name}"
                )


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
class ReferenceSessions:
    """Validated ownership record for separately constructed D1/D2 policies."""

    d1: InstantPolicy
    d2: InstantPolicy
    d1_config: ExperimentConfig
    d2_config: ExperimentConfig
    reference_id: str
    native_profile: str
    checkpoint_sha256: str
    native_point_count: int
    d1_session_id: str
    d2_session_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.d1, InstantPolicy):
            raise TypeError("D1 policy must be an InstantPolicy")
        if not isinstance(self.d2, InstantPolicy):
            raise TypeError("D2 policy must be an InstantPolicy")
        if self.d1 is self.d2:
            raise ValueError("D1/D2 policies must be distinct; policy alias detected")
        if not isinstance(self.d1_config, ExperimentConfig):
            raise TypeError("d1_config must be an ExperimentConfig")
        if not isinstance(self.d2_config, ExperimentConfig):
            raise TypeError("d2_config must be an ExperimentConfig")

        _nonempty_identifier(self.reference_id, "reference_id")
        _nonempty_identifier(self.native_profile, "native_profile")
        _canonical_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _positive_integer(self.native_point_count, "native_point_count")

        if self.d1_config.graph.num_demos != 1:
            raise ValueError("D1 graph num_demos must equal 1")
        expected_d2 = replace(
            self.d1_config,
            graph=replace(self.d1_config.graph, num_demos=2),
        )
        if self.d2_config != expected_d2:
            raise ValueError(
                "D2 config must differ from D1 config only by graph num_demos=2"
            )

        _validate_reference_policy(
            self.d1,
            self.d1_config,
            role="D1",
            checkpoint_sha256=self.checkpoint_sha256,
        )
        _validate_reference_policy(
            self.d2,
            self.d2_config,
            role="D2",
            checkpoint_sha256=self.checkpoint_sha256,
        )
        _validate_distinct_policy_owners(self.d1, self.d2)
        _validate_distinct_tensor_storage(self.d1, self.d2)

        for role, session_id in (
            ("d1", self.d1_session_id),
            ("d2", self.d2_session_id),
        ):
            expected_session_id = _reference_session_id(self.reference_id, role)
            if session_id != expected_session_id:
                raise ValueError(
                    f"{role}_session_id must equal the canonical {role.upper()} "
                    "reference-session identifier"
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
    "ReferenceSessions",
    "materialize_full_native_demo",
    "materialize_indexed_native_demo",
    "split_context_seeds",
)
