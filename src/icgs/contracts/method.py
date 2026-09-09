"""Shared timed-method records and capability protocols.

This module deliberately contains only boundary validation and structural types.
It must remain independent of policies, models, artifacts, and simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol, TYPE_CHECKING

import numpy as np

from .records import Observation

if TYPE_CHECKING:
    from icgs.state.context_cache import PreparedContext
    from icgs.state.method_context import EventMemory, MethodContext
    from icgs.state.physical import PhysicalState
    from icgs.state.task import TaskState


def _owned(value: Any, *, shape: tuple[int, ...] | None = None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    backing = np.frombuffer(np.array(array, copy=True).tobytes(), dtype=array.dtype)
    backing = backing.reshape(array.shape)
    backing.flags.writeable = False
    return backing


def _owned_prediction_logits(value: Any) -> Any:
    """Own NumPy logits or clone torch logits without importing torch eagerly."""

    module = getattr(type(value), "__module__", "")
    if module == "torch" or module.startswith("torch."):
        import torch

        if not torch.is_tensor(value):
            raise TypeError("grip_logits must be a tensor or NumPy-compatible array")
        if value.ndim != 2 or value.shape[1] != 1:
            raise ValueError("grip_logits must have shape [B,1]")
        if not value.is_floating_point():
            raise TypeError("tensor grip_logits must use a floating dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("grip_logits must contain only finite values")
        return value.clone()

    logits = np.asarray(value)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("grip_logits must have shape [B,1]")
    return _owned(logits, name="grip_logits")


def _counter(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} lineage must be nonempty")
    return value


def _pose(value: Any, name: str) -> np.ndarray:
    pose = _owned(value, shape=(4, 4), name=name)
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a valid SE(3) pose")
    return pose


def _grip(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("grip must be 0 or 1")
    value = int(value)
    if value not in (0, 1):
        raise ValueError("grip must be 0 or 1")
    return value


def _duration(value: Any, name: str = "duration_s") -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _observation(value: Observation) -> Observation:
    points = np.asarray(value.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("observation points must have nonempty shape [N,3]")
    points = _owned(points, name="observation points")
    pose = _pose(value.T_w_e, "observation pose")
    try:
        grip = float(value.grip)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation grip must be 0 or 1") from exc
    if not isfinite(grip) or grip not in (0.0, 1.0):
        raise ValueError("observation grip must be 0 or 1")
    return Observation(points, pose, grip)


@dataclass(frozen=True)
class TimedCommand:
    target_w: np.ndarray
    grip: int
    duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_w", _pose(self.target_w, "target_w"))
        object.__setattr__(self, "grip", _grip(self.grip))
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class CommandPrefix:
    commands: tuple[TimedCommand, ...]
    proposal_root: np.ndarray
    raw_candidate_id: str

    def __post_init__(self) -> None:
        commands = tuple(self.commands)
        if not commands or not all(isinstance(command, TimedCommand) for command in commands):
            raise ValueError("commands must be a nonempty tuple of TimedCommand")
        object.__setattr__(self, "commands", commands)
        object.__setattr__(self, "proposal_root", _pose(self.proposal_root, "proposal_root"))
        if not isinstance(self.raw_candidate_id, str) or not self.raw_candidate_id:
            raise ValueError("raw_candidate_id must be nonempty")


@dataclass(frozen=True)
class TimedObservation:
    observation: Observation
    boundary: int
    simulator_timestamp: float
    measured_wall_timestamp: float
    sensor_profile_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise ValueError("observation must be the native Observation")
        object.__setattr__(self, "observation", _observation(self.observation))
        object.__setattr__(self, "boundary", _counter(self.boundary, "boundary"))
        for field in ("simulator_timestamp", "measured_wall_timestamp"):
            value = float(getattr(self, field))
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, value)
        object.__setattr__(self, "sensor_profile_id", _identifier(self.sensor_profile_id, "sensor profile"))


@dataclass(frozen=True)
class ExecutedTransition:
    before: TimedObservation
    after: TimedObservation
    command: TimedCommand
    achieved_duration_s: float
    physics_substeps: int
    controller_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.before, TimedObservation) or not isinstance(self.after, TimedObservation):
            raise ValueError("before and after must be TimedObservation records")
        if not isinstance(self.command, TimedCommand):
            raise ValueError("command must be a TimedCommand")
        if self.after.boundary <= self.before.boundary:
            raise ValueError("after boundary must follow before boundary")
        object.__setattr__(self, "achieved_duration_s", _duration(self.achieved_duration_s, "achieved_duration_s"))
        object.__setattr__(self, "physics_substeps", _counter(self.physics_substeps, "physics_substeps"))
        if not isinstance(self.controller_status, str) or not self.controller_status:
            raise ValueError("controller_status must be nonempty")


@dataclass(frozen=True)
class SegmentRef:
    demo_content_hash: str
    a: int
    b: int
    kind: str
    valid_action_window: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "demo_content_hash", _identifier(self.demo_content_hash, "demo content"))
        a, b = _counter(self.a, "a"), _counter(self.b, "b")
        if b < a:
            raise ValueError("b must not precede a")
        if self.kind not in ("interaction", "start", "end"):
            raise ValueError("kind must be interaction, start, or end")
        if not isinstance(self.valid_action_window, (bool, np.bool_)):
            raise ValueError("valid_action_window must be boolean")
        object.__setattr__(self, "a", a); object.__setattr__(self, "b", b)


@dataclass(frozen=True)
class PhysicalPrediction:
    next_state: "PhysicalState"
    grip_logits: Any
    head_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "grip_logits", _owned_prediction_logits(self.grip_logits))
        head = _counter(self.head_id, "head_id")
        if head > 2:
            raise ValueError("head_id must be one of three fixed heads")
        object.__setattr__(self, "head_id", head)


@dataclass(frozen=True)
class TerminalProbabilities:
    probabilities: np.ndarray
    event_temperature_artifact_id: str

    @property
    def probs(self) -> np.ndarray:
        return self.probabilities

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities)
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError("probabilities must have shape [B,3]")
        probabilities = _owned(probabilities, name="probabilities")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probabilities must be nonnegative and sum to one")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "event_temperature_artifact_id",
                           _identifier(self.event_temperature_artifact_id, "event temperature artifact"))


@dataclass(frozen=True)
class EvaluationOutput:
    value_logits: np.ndarray
    completion_logits: np.ndarray
    progress_logits: np.ndarray
    calibrated_value: np.ndarray
    calibrated_stop: np.ndarray
    temperatures_id: str
    horizon: int = 1

    def __post_init__(self) -> None:
        for name in ("value_logits", "completion_logits", "progress_logits", "calibrated_value", "calibrated_stop"):
            object.__setattr__(self, name, _owned(getattr(self, name), name=name))
        object.__setattr__(self, "horizon", _counter(self.horizon, "horizon"))
        if self.horizon == 0 and np.any(self.calibrated_value != 0):
            raise ValueError("calibrated value must be zero at H=0")
        object.__setattr__(self, "temperatures_id", _identifier(self.temperatures_id, "temperatures artifact"))


@dataclass(frozen=True)
class PlanningResult:
    selected_prefix: CommandPrefix | None
    completed: bool
    fallback_reason: str | None
    expected_return: float | None
    call_count: int
    timing_counters: Mapping[str, int]
    cache_counters: Mapping[str, int]
    audit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selected_prefix is not None and not isinstance(self.selected_prefix, CommandPrefix):
            raise ValueError("selected_prefix must be CommandPrefix or None")
        if self.expected_return is not None and not isfinite(float(self.expected_return)):
            raise ValueError("expected_return must be finite")
        object.__setattr__(self, "call_count", _counter(self.call_count, "call_count"))
        for name in ("timing_counters", "cache_counters"):
            counters = dict(getattr(self, name))
            for key, value in counters.items():
                counters[key] = _counter(value, f"{name}.{key}")
            object.__setattr__(self, name, dict(counters))
        object.__setattr__(self, "audit_ids", tuple(self.audit_ids))


class TimedEnvironment(Protocol):
    def reset(self, seed: int | None = None) -> TimedObservation: ...
    def advance(self, command: TimedCommand) -> ExecutedTransition: ...
    def close(self) -> None: ...


class TaskMonitor(Protocol):
    def annotate(self, transition: ExecutedTransition) -> Any: ...


class ReplayProvider(Protocol):
    def restore(self, anchor_ref: str) -> tuple[TimedObservation, Any]: ...


class MethodCapabilities(Protocol):
    def encode_cloud(self, points_w: Any, point_valid: Any) -> Any: ...
    def decode_cloud(self, encoded: Any) -> Any: ...
    def physical_update(self, encoded: Any, observation: Observation, previous_descriptor: Any, memory: Any) -> PhysicalState: ...
    def segment_demo(self, raw_demo: Any, segmentation_config: Any) -> tuple[SegmentRef, ...]: ...
    def encode_events(self, raw_demos: Any, segments: Any) -> EventMemory: ...
    def track_task(self, previous: TaskState | None, state: PhysicalState, events: EventMemory) -> TaskState: ...
    def sample_prior(self, observation: Any, task: TaskState, context: MethodContext, *, seed: int) -> Any: ...
    def materialize_prefix(self, candidate: Any, *, h: int, r: int, duration_s: float) -> CommandPrefix: ...
    def predict_step(self, state: PhysicalState, command: TimedCommand, *, head_id: int) -> PhysicalPrediction: ...
    def evaluate_state(self, state: PhysicalState, task: TaskState, events: EventMemory, H: int) -> EvaluationOutput: ...
    def predict_terminal(self, before: PhysicalState, q_before: TaskState, after: PhysicalState, q_after: TaskState, events: EventMemory, command: TimedCommand) -> TerminalProbabilities: ...
    def plan(self, state: PhysicalState, task: TaskState, context: MethodContext, *, H: int, budget: Any) -> PlanningResult: ...
