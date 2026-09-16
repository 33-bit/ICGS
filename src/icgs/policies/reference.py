"""Owned native reference-context preparation and deterministic provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from icgs.algorithms.planning.candidates import Candidate
from icgs.algorithms.planning.router import select_window_indices
from icgs.configuration.method import MethodConfig
from icgs.configuration.schema import ExperimentConfig
from icgs.data.preprocessing.events import TimedDemoInput, debounced_grip_boundaries
from icgs.data.preprocessing.native import sample_to_cond_demo, subsample_pcd
from icgs.geometry.transforms import transform_pcd
from icgs.policies.instant_policy import InstantPolicy
from icgs.state.method_context import EventMemory, MethodContext
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


def build_method_context(
    raw_demos: tuple[TimedDemoInput, ...],
    events: EventMemory,
    sessions: ReferenceSessions,
    *,
    context_seed: int,
    reference_id: str,
    config: MethodConfig,
) -> tuple[MethodContext, ContextPreparationRecord]:
    """Plan every window before preparing owned full/window native contexts.

    Only a structurally infeasible selector result becomes an absent window.
    Preparation and validation errors propagate to the setup caller.
    """

    _exact_tuple(raw_demos, "raw_demos")
    if len(raw_demos) not in (1, 2):
        raise ValueError("raw_demos must contain one or two demos (D=1/2)")
    if not all(isinstance(demo, TimedDemoInput) for demo in raw_demos):
        raise TypeError("raw_demos must contain only TimedDemoInput records")
    if not isinstance(events, EventMemory):
        raise TypeError("events must be an EventMemory")
    if events.tokens.shape[0] != 1:
        raise ValueError("method context preparation requires online B=1")
    if not isinstance(sessions, ReferenceSessions):
        raise TypeError("sessions must be ReferenceSessions")
    if not isinstance(config, MethodConfig):
        raise TypeError("config must be a MethodConfig")
    _nonempty_identifier(reference_id, "reference_id")
    if reference_id != sessions.reference_id:
        raise ValueError("reference_id must match sessions.reference_id")
    if config.native_profile != sessions.native_profile:
        raise ValueError("config.native_profile must match sessions.native_profile")
    hashes = tuple(demo.demo_content_hash for demo in raw_demos)
    if hashes != events.raw_hashes[0]:
        raise ValueError("raw demo hash order must exactly match EventMemory raw_hashes")
    _nonnegative_integer(context_seed, "context_seed")

    event_count = events.tokens.shape[1]
    full_demo_seeds, window_slot_seeds = split_context_seeds(
        context_seed,
        demo_count=len(raw_demos),
        event_count=event_count,
    )
    for seeds, count, name in (
        (full_demo_seeds, len(raw_demos), "full_demo_seeds"),
        (window_slot_seeds, event_count, "window_slot_seeds"),
    ):
        _exact_tuple(seeds, name)
        if len(seeds) != count:
            raise ValueError(f"{name} must have exactly {count} entries")
        for index, seed in enumerate(seeds):
            _nonnegative_integer(seed, f"{name}[{index}]")

    native_waypoint_count = sessions.d1.graph_config.traj_horizon
    native_point_count = sessions.native_point_count
    demos_by_hash = dict(zip(hashes, raw_demos))
    partitions = {
        content_hash: tuple(
            ref for ref in events.refs[0]
            if ref is not None
            and ref.kind == "interaction"
            and ref.demo_content_hash == content_hash
        )
        for content_hash in hashes
    }
    # Validate raw-boundary compatibility even for demos with no eligible window.
    for ref in events.refs[0]:
        if ref is not None and ref.b > len(demos_by_hash[ref.demo_content_hash].transitions):
            raise ValueError("event ref boundaries must lie within their measured demo")
    for content_hash, partition in partitions.items():
        if not partition:
            raise ValueError("interaction partition must be nonempty")
        final_boundary = len(demos_by_hash[content_hash].transitions)
        if partition[0].a != 0 or partition[-1].b != final_boundary:
            raise ValueError("interaction partition must cover the complete measured demo")
        if any(ref.a >= ref.b for ref in partition):
            raise ValueError("interaction partition must have positive durations")
        if any(current.b != nxt.a for current, nxt in zip(partition, partition[1:])):
            raise ValueError("interaction partition must be contiguous with shared endpoints")

    grip_boundaries = {}
    for demo in raw_demos:
        measured_grips = (
            demo.transitions[0].before.observation.grip,
            *(transition.after.observation.grip for transition in demo.transitions),
        )
        grip_boundaries[demo.demo_content_hash] = debounced_grip_boundaries(
            measured_grips, config.event
        )

    window_indices = [None] * event_count
    for index, ref in enumerate(events.refs[0]):
        if (
            ref is not None
            and bool(events.valid[0, index].item())
            and ref.kind == "interaction"
            and ref.valid_action_window
        ):
            window_indices[index] = select_window_indices(
                ref,
                partitions[ref.demo_content_hash],
                grip_boundaries[ref.demo_content_hash],
                config.router,
                native_waypoint_count=native_waypoint_count,
            )

    full_demos = tuple(
        materialize_full_native_demo(
            demo,
            point_seed=full_demo_seeds[index],
            native_waypoint_count=native_waypoint_count,
            native_point_count=native_point_count,
        )
        for index, demo in enumerate(raw_demos)
    )
    full_policy = sessions.d1 if len(raw_demos) == 1 else sessions.d2
    native_full = full_policy.prepare_context(full_demos, prepared=True)
    full_policy.validate_context(native_full)

    native_windows = [None] * event_count
    window_context_ids = [None] * event_count
    for index, indices in enumerate(window_indices):
        if indices is None:
            continue
        ref = events.refs[0][index]
        native_demo = materialize_indexed_native_demo(
            demos_by_hash[ref.demo_content_hash],
            indices,
            point_seed=window_slot_seeds[index],
            native_waypoint_count=native_waypoint_count,
            native_point_count=native_point_count,
        )
        prepared = sessions.d1.prepare_context((native_demo,), prepared=True)
        sessions.d1.validate_context(prepared)
        native_windows[index] = prepared
        window_context_ids[index] = prepared.source_id

    context = MethodContext(
        raw_demos=raw_demos,
        events=events,
        native_full=native_full,
        native_windows=tuple(native_windows),
        reference_id=reference_id,
    )
    record = ContextPreparationRecord(
        context_seed=context_seed,
        full_demo_seeds=full_demo_seeds,
        window_slot_seeds=window_slot_seeds,
        rng_protocol=CONTEXT_RNG_PROTOCOL,
        full_context_id=native_full.source_id,
        window_context_ids=tuple(window_context_ids),
    )
    return context, record


@dataclass(frozen=True)
class ReferenceProposal:
    """Validated route provenance retaining the native Candidate by identity.

    Freezing these fields does not freeze the Candidate's arrays or tensors.
    Correspondence with actual sessions and context slots belongs to the caller.
    """

    candidate: Candidate
    reference_id: str
    route_index: int
    event_index: int | None
    route_seed: int
    diffusion_seed: int
    native_context_id: str
    native_session_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Candidate):
            raise TypeError("candidate must be a Candidate")
        route = _nonnegative_integer(self.route_index, "route_index")
        if route == 0:
            if self.event_index is not None:
                raise ValueError("full route requires event_index=None")
        else:
            event = _nonnegative_integer(self.event_index, "event_index")
            if event != route - 1:
                raise ValueError("event_index must equal route_index - 1")

        _nonnegative_integer(self.route_seed, "route_seed")
        diffusion_seed = _nonnegative_integer(self.diffusion_seed, "diffusion_seed")
        candidate_seed = _nonnegative_integer(self.candidate.seed, "candidate.seed")
        for name in ("reference_id", "native_context_id", "native_session_id"):
            _nonempty_identifier(getattr(self, name), name)
        if self.candidate.context_id != self.native_context_id:
            raise ValueError("candidate.context_id must match native_context_id")
        if candidate_seed != diffusion_seed:
            raise ValueError("candidate.seed must match diffusion_seed")


__all__ = (
    "CONTEXT_RNG_PROTOCOL",
    "ContextPreparationRecord",
    "ReferenceProposal",
    "ReferenceSessions",
    "build_method_context",
    "materialize_full_native_demo",
    "materialize_indexed_native_demo",
    "split_context_seeds",
)
