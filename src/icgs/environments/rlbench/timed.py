"""Bounded timed stepping against an explicit low-level controller capability.

This module does not import, launch, or wrap RLBench's pose-reaching task API.
The supplied controller must expose one physics-step operation and its own
simulator clock.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Protocol

import numpy as np

from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation


class UnsupportedTimedController(RuntimeError):
    """The collaborator cannot provide the explicit timed-control boundary."""


class TimedLifecycleError(RuntimeError):
    """An operation or cleanup failed without discarding either diagnostic."""

    def __init__(
        self,
        message: str,
        *,
        operation_error: Exception | None = None,
        safe_hold_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.operation_error = operation_error
        self.safe_hold_error = safe_hold_error
        self.close_error = close_error
        diagnostics = []
        if operation_error is not None:
            diagnostics.append(f"operation: {operation_error}")
        if safe_hold_error is not None:
            diagnostics.append(f"safe_hold: {safe_hold_error}")
        if close_error is not None:
            diagnostics.append(f"close: {close_error}")
        if diagnostics:
            message = f"{message}; " + "; ".join(diagnostics)
        super().__init__(message)


class TimedController(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def set_target(self, target_w: np.ndarray, grip: int) -> None: ...
    def step_physics(self) -> None: ...
    def observe(self) -> Observation: ...
    def simulator_time(self) -> float: ...
    def status(self) -> str: ...
    def safe_hold(self) -> None: ...
    def close(self) -> None: ...


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def interval_substeps(duration_s: float, physics_dt: float) -> int:
    """Return the exact integer number of physics steps in one interval."""
    duration_s = _positive_finite(duration_s, "duration_s")
    physics_dt = _positive_finite(physics_dt, "physics_dt")
    ratio = duration_s / physics_dt
    count = round(ratio)
    if count < 1 or not math.isclose(ratio, count, rel_tol=0, abs_tol=1e-9):
        raise ValueError("interval must match an integer physics-step count")
    return count


class TimedRLBenchAdapter:
    """Materialize one measured interval through an explicit controller seam."""

    def __init__(
        self,
        controller: TimedController,
        *,
        physics_dt: float,
        sensor_profile_id: str,
        max_reset_attempts: int = 3,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._require_controller(controller)
        self._controller = controller
        self._physics_dt = _positive_finite(physics_dt, "physics_dt")
        if not isinstance(sensor_profile_id, str) or not sensor_profile_id:
            raise ValueError("sensor_profile_id must be nonempty")
        self._sensor_profile_id = sensor_profile_id
        self._max_reset_attempts = _positive_integer(max_reset_attempts, "max_reset_attempts")
        if not callable(wall_clock):
            raise ValueError("wall_clock must be callable")
        self._wall_clock = wall_clock
        self._current: TimedObservation | None = None
        self._closed = False
        self._invalidated = False

    @staticmethod
    def _require_controller(controller: Any) -> None:
        required = (
            "reset",
            "set_target",
            "step_physics",
            "observe",
            "simulator_time",
            "status",
            "safe_hold",
            "close",
        )
        missing = tuple(name for name in required if not callable(getattr(controller, name, None)))
        if missing:
            names = ", ".join(missing)
            raise UnsupportedTimedController(
                f"controller lacks explicit timed capability: {names}"
            )

    def _ensure_open(self) -> None:
        if self._invalidated:
            raise RuntimeError("timed environment is invalidated")
        if self._closed:
            raise RuntimeError("timed environment is closed")

    def _cleanup(
        self,
        *,
        operation_error: Exception | None,
        reason: str,
    ) -> TimedLifecycleError | None:
        """Invalidate first, then delegate bounded cleanup in fixed order."""
        self._invalidated = True
        self._current = None

        safe_hold_error: Exception | None = None
        try:
            self._controller.safe_hold()
        except Exception as exc:
            safe_hold_error = exc

        close_error: Exception | None = None
        if not self._closed:
            try:
                self._controller.close()
            except Exception as exc:
                close_error = exc
            else:
                self._closed = True

        if safe_hold_error is None and close_error is None:
            return None
        return TimedLifecycleError(
            reason,
            operation_error=operation_error,
            safe_hold_error=safe_hold_error,
            close_error=close_error,
        )

    def _read_observation(self, boundary: int) -> TimedObservation:
        observation = self._controller.observe()
        if not isinstance(observation, Observation):
            raise UnsupportedTimedController("controller.observe must return Observation")
        simulator_timestamp = self._controller.simulator_time()
        measured_wall_timestamp = self._wall_clock()
        try:
            simulator_timestamp = float(simulator_timestamp)
            measured_wall_timestamp = float(measured_wall_timestamp)
        except (TypeError, ValueError) as exc:
            raise UnsupportedTimedController("controller clocks must return finite numbers") from exc
        if not math.isfinite(simulator_timestamp) or not math.isfinite(measured_wall_timestamp):
            raise UnsupportedTimedController("controller clocks must return finite numbers")
        return TimedObservation(
            observation,
            boundary,
            simulator_timestamp,
            measured_wall_timestamp,
            self._sensor_profile_id,
        )

    def reset(self, seed: int | None = None) -> TimedObservation:
        """Reset with a bounded retry budget and return boundary-zero state."""
        self._ensure_open()
        last_error: Exception | None = None
        for _ in range(self._max_reset_attempts):
            try:
                self._controller.reset(seed=seed)
                current = self._read_observation(0)
                self._current = current
                return current
            except Exception as exc:
                last_error = exc
        reason = f"reset failed after {self._max_reset_attempts} bounded attempts"
        cleanup_error = self._cleanup(
            operation_error=last_error,
            reason=reason,
        )
        if cleanup_error is not None:
            raise cleanup_error from last_error
        raise RuntimeError(reason) from last_error

    def _controller_status(self) -> str:
        status = self._controller.status()
        if not isinstance(status, str) or not status:
            raise UnsupportedTimedController("controller status must be a nonempty string")
        return status

    def advance(self, command: TimedCommand) -> ExecutedTransition:
        """Hold one target while stepping the controller's exact physics count."""
        self._ensure_open()
        if self._current is None:
            raise RuntimeError("timed environment must be reset before advance")
        if not isinstance(command, TimedCommand):
            raise ValueError("advance requires a TimedCommand")

        before = self._current
        substeps = interval_substeps(command.duration_s, self._physics_dt)
        try:
            self._controller.set_target(np.array(command.target_w, copy=True), command.grip)
            for _ in range(substeps):
                self._controller.step_physics()
            after = self._read_observation(before.boundary + 1)
            achieved_duration = after.simulator_timestamp - before.simulator_timestamp
            if not math.isfinite(achieved_duration) or achieved_duration <= 0:
                raise UnsupportedTimedController(
                    "simulator clock did not provide a positive achieved interval"
                )
            transition = ExecutedTransition(
                before,
                after,
                command,
                achieved_duration,
                substeps,
                self._controller_status(),
            )
        except Exception as operation_error:
            cleanup_error = self._cleanup(
                operation_error=operation_error,
                reason="timed interval failed",
            )
            if cleanup_error is not None:
                raise cleanup_error from operation_error
            raise
        self._current = after
        return transition

    def close(self) -> None:
        """Run controller-specific hold cleanup, then close; permit close retry."""
        if self._closed:
            return
        cleanup_error = self._cleanup(
            operation_error=None,
            reason="timed environment cleanup failed",
        )
        if cleanup_error is not None:
            raise cleanup_error


TimedEnvironment = TimedRLBenchAdapter


__all__ = [
    "TimedController",
    "TimedEnvironment",
    "TimedRLBenchAdapter",
    "TimedLifecycleError",
    "UnsupportedTimedController",
    "interval_substeps",
]
