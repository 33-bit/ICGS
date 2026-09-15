"""Validation, aggregate counts, and bounded in-memory attempt collection.

The module provides deterministic attempt counting and a bounded one-attempt
in-memory execution seam consuming an injected ``TimedEnvironment``, explicit
``MethodConfig`` or ``CollectionConfig``, and optional external ``TaskMonitor``.
It deliberately does not implement disk persistence, dataset lineage closures,
or physical simulator workloads.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from icgs.configuration.method import CollectionConfig, MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation


def attempt_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count attempts, successes, and invalid inputs without dropping failures.

    ``status`` remains recorder-owned provenance.  This boundary requires a
    nonempty canonical string with no surrounding whitespace.  It recognizes
    ``success`` and ``invalid-input`` for aggregate counters and leaves other
    valid statuses, such as ``timeout``, represented only in ``attempts``.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("attempt records must be a sequence")

    counts = {"attempts": len(records), "successes": 0, "invalid": 0}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"attempt record {index} must be a mapping")
        status = record.get("status")
        if not isinstance(status, str) or not status.strip():
            raise ValueError(f"attempt record {index} must have a nonempty status")
        if status != status.strip():
            raise ValueError(f"attempt record {index} status must be canonical")
        counts["successes"] += int(status == "success")
        counts["invalid"] += int(status == "invalid-input")
    return counts


class AttemptExecutionError(RuntimeError):
    """Raised when attempt execution fails due to an operational or infrastructure error.

    Preserves executed transitions, collected annotation snapshots, any rejected
    transition outside the accepted causal chain, and cleanup diagnostics without
    discarding the primary cause.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        cleanup_error: BaseException | None = None,
        transitions: Sequence[ExecutedTransition] = (),
        annotations: Sequence[Mapping[str, Any]] = (),
        rejected_transition: ExecutedTransition | None = None,
        initial_observation: TimedObservation | None = None,
    ) -> None:
        diagnostics = [message]
        if cause is not None:
            diagnostics.append(f"cause: {type(cause).__name__}: {cause}")
        if cleanup_error is not None:
            diagnostics.append(f"cleanup: {type(cleanup_error).__name__}: {cleanup_error}")
        if rejected_transition is not None:
            diagnostics.append("rejected transition retained")
        super().__init__("; ".join(diagnostics))
        self.cause = cause
        self.cleanup_error = cleanup_error
        self.transitions = tuple(transitions)
        self.annotations = tuple(annotations)
        self.rejected_transition = rejected_transition
        self.initial_observation = initial_observation


@dataclass(frozen=True)
class AttemptResult(Mapping[str, Any]):
    """Frozen outer record of bounded attempt execution with detached mutable snapshots.

    Contains executed transitions and detached mutable annotation mappings.
    When an external monitor is provided, ``annotations[i]`` corresponds to
    ``transitions[i]`` for each successfully recorded step; ``annotations`` is
    empty when no monitor is supplied and may be shorter than ``transitions``
    if annotation computation fails on a step.
    """

    initial_observation: TimedObservation
    transitions: tuple[ExecutedTransition, ...]
    annotations: tuple[Mapping[str, Any], ...]
    status: str

    def __getitem__(self, key: str) -> Any:
        if key == "initial_observation":
            return self.initial_observation
        if key == "transitions":
            return self.transitions
        if key == "annotations":
            return self.annotations
        if key == "status":
            return self.status
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from ("initial_observation", "transitions", "annotations", "status")

    def __len__(self) -> int:
        return 4


def _resolve_collection_bounds(config: Any) -> tuple[int, float | None]:
    """Extract max intervals and wall limit from explicit MethodConfig or CollectionConfig."""
    if isinstance(config, MethodConfig):
        collection = config.collection
    elif isinstance(config, CollectionConfig):
        collection = config
    else:
        raise TypeError(
            f"config must be an instance of MethodConfig or CollectionConfig, got {type(config).__name__}"
        )

    max_intervals = collection.max_episode_intervals
    if isinstance(max_intervals, bool) or not isinstance(max_intervals, int) or max_intervals < 1:
        raise ValueError(f"max_episode_intervals must be a positive integer, got {max_intervals!r}")

    wall_limit_s = collection.wall_limit_s
    if wall_limit_s is not None:
        if isinstance(wall_limit_s, bool) or not isinstance(wall_limit_s, (int, float)) or wall_limit_s <= 0:
            raise ValueError(f"wall_limit_s must be a positive float or None, got {wall_limit_s!r}")
        wall_limit_s = float(wall_limit_s)

    return max_intervals, wall_limit_s


def _validate_transition(
    transition: Any,
    expected_before: TimedObservation,
    expected_command: TimedCommand,
) -> None:
    """Verify full causal continuity and command matching for an executed transition."""
    if not isinstance(transition, ExecutedTransition):
        raise TypeError(
            f"environment.advance must return ExecutedTransition, got {type(transition).__name__}"
        )
    before = transition.before
    if (
        before.boundary != expected_before.boundary
        or before.simulator_timestamp != expected_before.simulator_timestamp
        or before.measured_wall_timestamp != expected_before.measured_wall_timestamp
        or before.sensor_profile_id != expected_before.sensor_profile_id
    ):
        raise ValueError(
            f"transition before metadata mismatch at boundary {expected_before.boundary}: "
            f"got ({before.boundary}, {before.simulator_timestamp}, {before.measured_wall_timestamp}, {before.sensor_profile_id!r}), "
            f"expected ({expected_before.boundary}, {expected_before.simulator_timestamp}, {expected_before.measured_wall_timestamp}, {expected_before.sensor_profile_id!r})"
        )
    if (
        before.observation.grip != expected_before.observation.grip
        or not np.array_equal(before.observation.points, expected_before.observation.points)
        or not np.array_equal(before.observation.T_w_e, expected_before.observation.T_w_e)
    ):
        raise ValueError(
            f"transition before observation does not match previous observation at boundary {expected_before.boundary}"
        )
    if transition.after.boundary != before.boundary + 1:
        raise ValueError(
            f"boundary continuity broken: after.boundary={transition.after.boundary} != before.boundary + 1 ({before.boundary + 1})"
        )
    cmd = transition.command
    if not isinstance(cmd, TimedCommand):
        raise TypeError(f"transition command must be TimedCommand, got {type(cmd).__name__}")
    if (
        cmd.grip != expected_command.grip
        or cmd.duration_s != expected_command.duration_s
        or not np.array_equal(cmd.target_w, expected_command.target_w)
    ):
        raise ValueError(
            f"transition command does not match submitted command at boundary {before.boundary}"
        )


def collect_attempt(
    environment: Any,
    commands: Iterable[TimedCommand],
    *,
    config: MethodConfig | CollectionConfig,
    monitor: Any = None,
    seed: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AttemptResult:
    """Execute one bounded attempt against an injected environment and monitor.

    This partial seam validates explicit configuration, resets the environment once,
    advances sequentially through commands up to configured resource limits,
    validates strict causal boundary continuity, and detaches annotation snapshots.
    It halts on command exhaustion or resource limit; physical terminal classification
    is deferred and not inferred from opaque monitor metadata.

    When an external monitor is provided, ``annotations[i]`` corresponds to
    ``transitions[i]`` for each step where annotation succeeded. If no monitor is
    provided, ``annotations`` is empty. If annotation computation fails on a step,
    the attempt halts with ``AttemptExecutionError`` whose ``transitions`` includes
    the executed transition while ``annotations`` contains only the shorter successful prefix.

    Limitations:
    - Only bounded in-memory execution is performed; no disk/JSON IO or physical
      simulator launching occurs.
    - Wall limits are evaluated between steps; blocking capability calls are
      not preempted. Outer multi-attempt quotas are not evaluated here.
    - Terminal success/failure classification is not implemented by this partial seam.
    """
    max_intervals, wall_limit_s = _resolve_collection_bounds(config)

    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    rejected_transition: ExecutedTransition | None = None
    transitions: list[ExecutedTransition] = []
    annotations: list[Mapping[str, Any]] = []
    initial_observation: TimedObservation | None = None
    status = "completed"

    try:
        start_time = clock()
        command_iter = iter(commands)
        initial_observation = environment.reset(seed=seed)
        if not isinstance(initial_observation, TimedObservation):
            raise TypeError(
                f"environment.reset must return TimedObservation, got {type(initial_observation).__name__}"
            )
        last_observation = initial_observation

        for _ in range(max_intervals):
            if wall_limit_s is not None and (clock() - start_time) >= wall_limit_s:
                status = "timeout"
                break

            try:
                command = next(command_iter)
            except StopIteration:
                break

            if not isinstance(command, TimedCommand):
                raise TypeError(f"command must be a TimedCommand, got {type(command).__name__}")

            transition = environment.advance(command)
            try:
                _validate_transition(transition, last_observation, command)
            except BaseException as val_err:
                if isinstance(transition, ExecutedTransition):
                    rejected_transition = transition
                raise val_err

            transitions.append(transition)
            last_observation = transition.after

            if monitor is not None:
                ann = monitor.annotate(transition)
                if not isinstance(ann, Mapping):
                    raise TypeError(f"monitor.annotate must return Mapping, got {type(ann).__name__}")
                annotations.append(copy.deepcopy(dict(ann)))
        else:
            if status == "completed":
                status = "timeout"

    except BaseException as exc:
        primary_error = exc
    finally:
        close_fn = getattr(environment, "close", None)
        if not callable(close_fn):
            cleanup_error = TypeError(
                f"environment must provide a callable close() method, got {type(close_fn).__name__}"
            )
        else:
            try:
                close_fn()
            except BaseException as exc:
                cleanup_error = exc

    if primary_error is not None:
        error = AttemptExecutionError(
            "attempt execution failed",
            cause=primary_error,
            cleanup_error=cleanup_error,
            transitions=transitions,
            annotations=annotations,
            rejected_transition=rejected_transition,
            initial_observation=initial_observation,
        )
        raise error from primary_error

    if cleanup_error is not None:
        error = AttemptExecutionError(
            "environment close failed during completion",
            cause=cleanup_error,
            cleanup_error=cleanup_error,
            transitions=transitions,
            annotations=annotations,
            initial_observation=initial_observation,
        )
        raise error from cleanup_error

    return AttemptResult(
        initial_observation=initial_observation,
        transitions=tuple(transitions),
        annotations=tuple(annotations),
        status=status,
    )


__all__ = [
    "AttemptExecutionError",
    "AttemptResult",
    "attempt_counts",
    "collect_attempt",
]
