"""Minimal budget and resource counter seam for search planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from icgs.contracts.method import CommandPrefix, PlanningResult
from icgs.contracts.records import Observation
from icgs.observability.recorder import NoopRecorder


@dataclass(frozen=True)
class PlanningBudget:
    """Resource bounds for online planning."""

    wall_budget_s: float | None = None
    native_call_cap: int | None = None
    model_interval_cap: int | None = None
    clock_track: str = "paused"

    def __post_init__(self) -> None:
        if self.wall_budget_s is not None:
            if isinstance(self.wall_budget_s, bool) or not isinstance(self.wall_budget_s, (int, float, np.floating, np.integer)):
                raise TypeError("wall_budget_s must be a numeric float")
            val = float(self.wall_budget_s)
            if not isfinite(val) or val <= 0:
                raise ValueError("wall_budget_s must be finite and positive")
            object.__setattr__(self, "wall_budget_s", val)

        for name in ("native_call_cap", "model_interval_cap"):
            cap = getattr(self, name)
            if cap is not None:
                if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)):
                    raise TypeError(f"{name} must be an integer")
                if cap < 0:
                    raise ValueError(f"{name} must be nonnegative")
                object.__setattr__(self, name, int(cap))

        if not isinstance(self.clock_track, str) or not self.clock_track.strip():
            raise ValueError("clock_track must be nonempty string")

        if self.wall_budget_s is None and self.native_call_cap is None:
            raise ValueError(
                "PlanningBudget must specify wall_budget_s or native_call_cap (model_interval_cap alone is insufficient)"
            )

    def check_native_call(self, count: int) -> bool:
        """Return True if another native call is permitted."""
        if self.native_call_cap is not None and count >= self.native_call_cap:
            return False
        return True

    def check_model_interval(self, count: int) -> bool:
        """Return True if another model interval is permitted."""
        if self.model_interval_cap is not None and count >= self.model_interval_cap:
            return False
        return True

    def check_wall_budget(self, elapsed_s: float) -> bool:
        """Return True if elapsed time is within wall budget."""
        if self.wall_budget_s is not None and elapsed_s >= self.wall_budget_s:
            return False
        return True


class BudgetExhausted(Exception):
    """Raised when an operation cannot be performed due to budget exhaustion."""


class NonfiniteModelError(ValueError):
    """Raised when a model prediction or probability contains nonfinite values."""


def eligible_completion(finished_at: float, deadline: float) -> bool:
    """Return True if finished_at is within deadline (equality eligible)."""
    if isinstance(finished_at, bool) or not isinstance(finished_at, (int, float, np.floating, np.integer)):
        raise TypeError("finished_at must be a numeric float")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float, np.floating, np.integer)):
        raise TypeError("deadline must be a numeric float")
    f = float(finished_at)
    d = float(deadline)
    if not isfinite(f) or not isfinite(d):
        if isfinite(f) and d == float("inf"):
            return True
        raise ValueError("finished_at and deadline must be finite")
    return f <= d



def synchronize_device(capabilities: Any = None) -> None:
    """Synchronize device operations if capability or torch device provides synchronization."""
    if capabilities is not None and hasattr(capabilities, "synchronize") and callable(capabilities.synchronize):
        capabilities.synchronize()
    else:
        import sys

        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()


def execute_fallback(
    state: Any,
    task: Any,
    context: Any,
    *,
    H: int,
    budget: PlanningBudget,
    capabilities: Any,
    cfg: Any,
    seed: int,
    clock_fn: Callable[[], float],
    start_time: float,
    deadline: float,
    earliest_candidate: Any | None,
    native_calls_count: int,
    model_intervals_count: int,
    evaluator_calls_count: int,
    iteration_count: int,
    cache_counters: Mapping[str, int] | None = None,
    audit_ids: Sequence[str] = (),
    recorder: Any = None,
    fallback_reason: str = "budget_exhausted",
    observation_fn: Callable[[Any], Observation] | None = None,
) -> PlanningResult:
    """Materialize fallback prefix using earliest sampled candidate or exactly one new reference sample."""
    rec = recorder if recorder is not None else NoopRecorder()

    if H == 0:
        return PlanningResult(
            selected_prefix=None,
            completed=False,
            fallback_reason=fallback_reason,
            expected_return=None,
            call_count=iteration_count,
            timing_counters={
                "iterations": iteration_count,
                "native_calls": native_calls_count,
                "model_intervals": model_intervals_count,
                "evaluator_calls": evaluator_calls_count,
            },
            cache_counters=dict(cache_counters) if cache_counters is not None else {"lookups": 0, "hits": 0, "misses": 0},
            audit_ids=tuple(audit_ids),
        )

    # Valid root and context required; do not fabricate geometry
    if observation_fn is not None:
        rep_obs = observation_fn(state)
    else:
        from icgs.contracts.records import Observation

        if not hasattr(state, "x") or not hasattr(state, "T_w_e") or not hasattr(state, "grip"):
            raise TypeError("state must be PhysicalState")
        points = np.ascontiguousarray(state.x[0].detach().cpu().numpy())
        pose = np.ascontiguousarray(state.T_w_e[0].detach().cpu().numpy())
        grip = float(state.grip[0].item() if hasattr(state.grip[0], "item") else state.grip[0])
        rep_obs = Observation(points=points, T_w_e=pose, grip=grip)

    if earliest_candidate is not None:
        cand = earliest_candidate
        reused = True
    else:
        reused = False
        cand = capabilities.sample_prior(rep_obs, task, context, seed=seed)
        synchronize_device(capabilities)
        native_calls_count += 1

    p_cfg = cfg.planning if hasattr(cfg, "planning") else cfg
    dt0 = float(cfg.control.dt0) if hasattr(cfg, "control") else 0.1
    step_intervals = min(p_cfg.h, H)
    r_val = min(p_cfg.r, step_intervals)
    prefix = capabilities.materialize_prefix(cand, h=step_intervals, r=r_val, duration_s=dt0)
    finished_fallback = clock_fn()
    overshoot = max(0.0, finished_fallback - deadline)
    extra_latency = max(0.0, finished_fallback - max(deadline, start_time))

    rec.event(
        "planning.fallback",
        fields={
            "reason": fallback_reason,
            "reused_prior": reused,
            "extra_latency": extra_latency,
            "overshoot": overshoot,
            "native_calls": native_calls_count,
        },
    )

    cand_id = getattr(prefix, "raw_candidate_id", "fallback_0")
    final_audit_ids = tuple(audit_ids) if audit_ids else (str(cand_id),)

    return PlanningResult(
        selected_prefix=prefix,
        completed=False,
        fallback_reason=fallback_reason,
        expected_return=None,
        call_count=iteration_count,
        timing_counters={
            "iterations": iteration_count,
            "native_calls": native_calls_count,
            "model_intervals": model_intervals_count,
            "evaluator_calls": evaluator_calls_count,
        },
        cache_counters=dict(cache_counters) if cache_counters is not None else {"lookups": 0, "hits": 0, "misses": 0},
        audit_ids=final_audit_ids,
    )


