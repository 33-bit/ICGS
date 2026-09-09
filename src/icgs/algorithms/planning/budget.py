"""Minimal budget and resource counter seam for search planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from icgs.contracts.method import CommandPrefix, EvaluationOutput, PlanningResult, TerminalProbabilities
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
    """Synchronize device operations via capability protocol (no duck typing)."""
    if capabilities is not None and hasattr(capabilities, "synchronize") and callable(capabilities.synchronize):
        capabilities.synchronize()


def _is_nonfinite_model_error(exc: Exception) -> bool:
    if isinstance(exc, NonfiniteModelError):
        return True
    if isinstance(exc, ValueError):
        msg = str(exc).lower()
        return any(
            term in msg
            for term in (
                "finite",
                "nan",
                "inf",
                "sum to one",
                "probabilities must",
                "calibrated value must be zero at h=0",
                "valid se(3) pose",
            )
        )
    return False


def timed_operation(
    op_name: str,
    op: Callable[[], Any],
    *,
    clock_fn: Callable[[], float],
    deadline: float,
    budget: PlanningBudget,
    capabilities: Any,
    recorder: Any = None,
    coverage: str = "inclusive",
    timing_counters: dict[str, int] | None = None,
    counter_key: str | None = None,
) -> tuple[Any, float, bool]:
    """Execute capability operation with immediate precheck, explicit sync, completed/overshoot audit, and counters."""
    rec = recorder if recorder is not None else NoopRecorder()
    now = clock_fn()
    if now >= deadline:
        return None, now, False

    start_t = clock_fn()
    try:
        res = op()
    except Exception as exc:
        if _is_nonfinite_model_error(exc):
            rec.event(
                "model.error",
                fields={"operation": op_name, "error": "nonfinite_model_output", "detail": str(exc)},
            )
            raise NonfiniteModelError(str(exc)) from exc
        raise

    synchronize_device(capabilities)
    if timing_counters is not None:
        timing_counters["sync_calls"] = timing_counters.get("sync_calls", 0) + 1

    finished_t = clock_fn()
    eligible = eligible_completion(finished_t, deadline)
    duration_s = max(0.0, finished_t - start_t)

    if eligible:
        if timing_counters is not None and counter_key is not None:
            timing_counters[counter_key] = timing_counters.get(counter_key, 0) + 1
        rec.event(
            "operation.completed",
            fields={
                "operation": op_name,
                "finished": finished_t,
                "duration_s": duration_s,
                "coverage": coverage,
            },
        )
        return res, finished_t, True
    else:
        overshoot = max(0.0, finished_t - deadline)
        rec.event(
            "operation.overshoot",
            fields={
                "operation": op_name,
                "finished": finished_t,
                "deadline": deadline,
                "overshoot": overshoot,
                "coverage": coverage,
            },
        )
        return res, finished_t, False


def validate_root_and_context(state: Any, context: Any, capabilities: Any) -> Observation:
    """Strictly validate root physical state geometry and context via capability protocol before any model evaluation."""
    from icgs.state.physical import PhysicalState

    if not isinstance(state, PhysicalState):
        raise TypeError("state must be PhysicalState")

    if hasattr(capabilities, "validate_context") and callable(capabilities.validate_context):
        capabilities.validate_context(context)

    pts = state.cached_world_cloud if state.cached_world_cloud is not None else state.x
    if pts is None:
        raise ValueError("state points must not be None")
    if hasattr(pts, "detach"):
        pts_np = np.ascontiguousarray(pts[0].detach().cpu().numpy())
    else:
        pts_np = np.ascontiguousarray(pts[0] if pts.ndim == 3 else pts)

    if pts_np.ndim != 2 or pts_np.shape[1] != 3 or pts_np.shape[0] == 0:
        raise ValueError("points must have shape [N, 3] with N > 0")
    if not np.all(np.isfinite(pts_np)):
        raise ValueError("points must contain only finite values")

    pose = state.T_w_e
    if hasattr(pose, "detach"):
        pose_np = np.ascontiguousarray(pose[0].detach().cpu().numpy())
    else:
        pose_np = np.ascontiguousarray(pose[0] if pose.ndim == 3 else pose)
    if pose_np.shape != (4, 4) or not np.all(np.isfinite(pose_np)):
        raise ValueError("T_w_e must be a finite 4x4 matrix")
    if not np.allclose(pose_np[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=pose_np.dtype), atol=1e-6):
        raise ValueError("T_w_e must be a valid SE(3) pose with bottom row [0, 0, 0, 1]")

    grip_val = state.grip
    if hasattr(grip_val, "detach"):
        g = float(grip_val[0].item() if hasattr(grip_val[0], "item") else grip_val[0])
    else:
        g = float(grip_val[0] if hasattr(grip_val, "__len__") else grip_val)
    if not isfinite(g) or g not in (0.0, 1.0):
        raise ValueError("grip must be 0 or 1")

    return Observation(points=pts_np, T_w_e=pose_np, grip=g)


def validate_predicted_state(state: Any) -> None:
    """Validate all selectable/cacheable state fields for finiteness and validity."""
    from icgs.state.physical import PhysicalState

    if not isinstance(state, PhysicalState):
        raise NonfiniteModelError("predicted state is not a PhysicalState")
    for name in ("x", "cached_world_cloud", "p", "memory", "T_w_e", "grip"):
        val = getattr(state, name, None)
        if val is None:
            continue
        if hasattr(val, "detach"):
            arr = val.detach().cpu().numpy()
        else:
            arr = np.asarray(val)
        if not np.all(np.isfinite(arr)):
            raise NonfiniteModelError(f"predicted state field '{name}' contains nonfinite values")


def validate_evaluation_output(eval_out: Any, name: str = "Evaluation") -> tuple[float, np.ndarray]:
    """Validate scalar stop and values from EvaluationOutput, returning (u_stop, calibrated_value)."""
    if eval_out is None:
        raise NonfiniteModelError(f"{name} output is None")
    stop_raw = getattr(eval_out, "calibrated_stop", None)
    val_raw = getattr(eval_out, "calibrated_value", None)
    if stop_raw is None or val_raw is None:
        raise NonfiniteModelError(f"{name} missing calibrated_stop or calibrated_value")

    if hasattr(stop_raw, "detach"):
        stop_np = stop_raw.detach().cpu().numpy()
    else:
        stop_np = np.asarray(stop_raw)
    if stop_np.size != 1:
        raise NonfiniteModelError(f"{name} calibrated_stop must be a scalar")
    u_val = float(stop_np.reshape(-1)[0])
    if not isfinite(u_val) or u_val < 0.0 or u_val > 1.0:
        raise NonfiniteModelError(f"{name} calibrated_stop must be finite in [0, 1], got {u_val}")

    if hasattr(val_raw, "detach"):
        val_np = val_raw.detach().cpu().numpy()
    else:
        val_np = np.asarray(val_raw)
    if val_np.size != 1:
        raise ValueError(f"{name} calibrated_value must be a single scalar, got array of shape {val_np.shape}")
    c_val = float(val_np.reshape(-1)[0])
    if not isfinite(c_val):
        raise NonfiniteModelError(f"{name} calibrated_value must be finite, got {c_val}")
    if c_val < 0.0 or c_val > 1.0:
        raise ValueError(f"{name} calibrated_value must be in [0.0, 1.0], got {c_val}")

    return u_val, c_val


def validate_terminal_probabilities(term_probs: Any) -> np.ndarray:
    """Validate TerminalProbabilities output."""
    if term_probs is None:
        raise NonfiniteModelError("TerminalProbabilities is None")
    probs = getattr(term_probs, "probabilities", None)
    if probs is None:
        raise NonfiniteModelError("TerminalProbabilities missing probabilities")
    if hasattr(probs, "detach"):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.asarray(probs)
    if probs_np.ndim != 2 or probs_np.shape[1] != 3:
        raise NonfiniteModelError("Terminal probabilities must have shape [B, 3]")
    if not np.all(np.isfinite(probs_np)):
        raise NonfiniteModelError("Terminal probabilities must be finite")
    if (probs_np < 0.0).any() or not np.allclose(probs_np.sum(axis=1), 1.0, atol=1e-5):
        raise NonfiniteModelError("Terminal probabilities must be non-negative and sum to 1.0")
    return probs_np


def audit_planning_start(
    rec: Any,
    alg_name: str,
    cfg: Any,
    budget: PlanningBudget,
) -> None:
    """Audit planning start with resolved config fingerprint, effective limits, and clock track."""
    from icgs.configuration.method import MethodConfig

    if not isinstance(cfg, MethodConfig):
        raise TypeError(f"cfg must be MethodConfig, got {type(cfg)}")

    is_wall_in_panel = budget.wall_budget_s is not None and budget.wall_budget_s in cfg.planning.wall_budgets_s
    is_native_in_panel = budget.native_call_cap is not None and budget.native_call_cap in cfg.planning.native_call_caps
    is_model_in_panel = budget.model_interval_cap is not None and budget.model_interval_cap == cfg.planning.model_interval_cap
    is_track_matching = budget.clock_track == cfg.control.clock_track

    rec.event(
        "planning.start",
        fields={
            "algorithm": alg_name,
            "config_sha256": cfg.fingerprint(),
            "schema_version": cfg.schema_version,
            "wall_budget_s": budget.wall_budget_s,
            "native_call_cap": budget.native_call_cap,
            "model_interval_cap": budget.model_interval_cap,
            "clock_track": budget.clock_track,
            "configured_clock_track": cfg.control.clock_track,
            "clock_track_override": not is_track_matching,
            "panel_matched": bool(is_wall_in_panel and is_native_in_panel and is_model_in_panel and is_track_matching),
            "wall_budget_panel_override": not is_wall_in_panel if budget.wall_budget_s is not None else False,
            "native_cap_panel_override": not is_native_in_panel if budget.native_call_cap is not None else False,
            "model_cap_panel_override": not is_model_in_panel if budget.model_interval_cap is not None else False,
        },
    )


def evaluate_root_and_check_h0(
    state: Any,
    task: Any,
    context: Any,
    *,
    H: int,
    budget: PlanningBudget,
    capabilities: Any,
    cfg: Any,
    clock_fn: Callable[[], float],
    deadline: float,
    recorder: Any,
    timing_counters: dict[str, int],
    cache_counters: Mapping[str, int],
    audit_ids: Sequence[str],
) -> tuple[PlanningResult | None, float, bool]:
    """Evaluate root state and check H=0 / absorbed root conditions.

    Returns:
        (result, u_root, should_abort_to_fallback)
    """
    events = getattr(context, "events", context)
    eval_res, finished_t, eligible = timed_operation(
        "root_eval",
        lambda: capabilities.evaluate_state(state, task, events, H=H),
        clock_fn=clock_fn,
        deadline=deadline,
        budget=budget,
        capabilities=capabilities,
        recorder=recorder,
        coverage="inclusive",
        timing_counters=timing_counters,
        counter_key="evaluator_calls",
    )

    if eval_res is None and not eligible:
        if H == 0:
            return (
                PlanningResult(
                    selected_prefix=None,
                    completed=False,
                    fallback_reason="zero_horizon_timeout",
                    expected_return=None,
                    call_count=0,
                    timing_counters=dict(timing_counters),
                    cache_counters=dict(cache_counters),
                    audit_ids=tuple(audit_ids),
                ),
                0.0,
                False,
            )
        return None, 0.0, True

    u_root, _ = validate_evaluation_output(eval_res, "Root evaluation")

    if H == 0:
        if eligible:
            return (
                PlanningResult(
                    selected_prefix=None,
                    completed=True,
                    fallback_reason="zero_horizon",
                    expected_return=float(u_root),
                    call_count=0,
                    timing_counters=dict(timing_counters),
                    cache_counters=dict(cache_counters),
                    audit_ids=tuple(audit_ids),
                ),
                u_root,
                False,
            )
        else:
            return (
                PlanningResult(
                    selected_prefix=None,
                    completed=False,
                    fallback_reason="zero_horizon_timeout",
                    expected_return=None,
                    call_count=0,
                    timing_counters=dict(timing_counters),
                    cache_counters=dict(cache_counters),
                    audit_ids=tuple(audit_ids),
                ),
                u_root,
                False,
            )

    if not eligible:
        # H > 0 and root eval overshot deadline: must trigger reference fallback!
        return None, u_root, True

    if u_root >= 1.0:
        return (
            PlanningResult(
                selected_prefix=None,
                completed=True,
                fallback_reason="absorbed_root",
                expected_return=float(u_root),
                call_count=0,
                timing_counters=dict(timing_counters),
                cache_counters=dict(cache_counters),
                audit_ids=tuple(audit_ids),
            ),
            u_root,
            False,
        )

    return None, u_root, False


def score_leaf_rollout(
    leaf_node: Any,
    events: Any,
    *,
    remaining_H: int,
    clock_fn: Callable[[], float],
    deadline: float,
    budget: PlanningBudget,
    capabilities: Any,
    recorder: Any,
    timing_counters: dict[str, int],
) -> tuple[float | None, bool]:
    """Evaluate active leaf hypotheses across 3 heads via timed_operation, returning G and eligibility."""
    from icgs.algorithms.planning.belief import leaf_return

    if remaining_H == 0 or getattr(leaf_node, "is_terminal", False):
        G = leaf_return(leaf_node.U, leaf_node.weights, np.zeros(3, dtype=np.float64), 0)
        return G, True

    values = np.zeros(3, dtype=np.float64)
    for m in range(3):
        h_m = leaf_node.hypotheses[m]
        if h_m.weight > 0.0:
            res, finished_t, eligible = timed_operation(
                "leaf_eval",
                lambda m=m: capabilities.evaluate_state(h_m.state, h_m.task, events, H=remaining_H),
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=recorder,
                coverage="inclusive",
                timing_counters=timing_counters,
                counter_key="evaluator_calls",
            )
            if not eligible or res is None:
                return None, False
            u_m, cv = validate_evaluation_output(res, f"Leaf head {m} evaluation")
            values[m] = cv

    G = leaf_return(leaf_node.U, leaf_node.weights, values, remaining_H)
    return G, True


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
    timing_counters: dict[str, int],
    cache_counters: Mapping[str, int] | None = None,
    audit_ids: Sequence[str] = (),
    recorder: Any = None,
    fallback_reason: str = "budget_exhausted",
    observation_fn: Callable[[Any], Observation] | None = None,
) -> PlanningResult:
    """Materialize fallback prefix using earliest sampled candidate or exactly one new reference sample."""
    from icgs.configuration.method import MethodConfig

    if not isinstance(cfg, MethodConfig):
        raise TypeError(f"cfg must be MethodConfig, got {type(cfg)}")

    rec = recorder if recorder is not None else NoopRecorder()

    if H == 0:
        return PlanningResult(
            selected_prefix=None,
            completed=False,
            fallback_reason=fallback_reason,
            expected_return=None,
            call_count=timing_counters.get("iterations", 0),
            timing_counters=dict(timing_counters),
            cache_counters=dict(cache_counters) if cache_counters is not None else {"lookups": 0, "hits": 0, "misses": 0},
            audit_ids=tuple(audit_ids),
        )

    if observation_fn is not None:
        rep_obs = observation_fn(state)
    else:
        rep_obs = validate_root_and_context(state, context, capabilities)

    if earliest_candidate is not None:
        cand = earliest_candidate
        reused = True
    else:
        reused = False
        start_native = clock_fn()
        try:
            cand = capabilities.sample_prior(rep_obs, task, context, seed=seed)
        except Exception as exc:
            if _is_nonfinite_model_error(exc):
                rec.event("model.error", fields={"operation": "sample_prior", "error": str(exc)})
                raise NonfiniteModelError(str(exc)) from exc
            raise
        synchronize_device(capabilities)
        timing_counters["sync_calls"] = timing_counters.get("sync_calls", 0) + 1
        finished_native = clock_fn()
        timing_counters["native_calls"] = timing_counters.get("native_calls", 0) + 1
        rec.event(
            "operation.completed",
            fields={
                "operation": "sample_prior",
                "finished": finished_native,
                "duration_s": max(0.0, finished_native - start_native),
                "coverage": "inclusive",
            },
        )

    step_intervals = min(cfg.planning.h, H)
    r_val = min(cfg.planning.r, step_intervals)
    dt0 = float(cfg.control.dt0)

    start_mat = clock_fn()
    try:
        prefix = capabilities.materialize_prefix(cand, h=step_intervals, r=r_val, duration_s=dt0)
    except Exception as exc:
        if _is_nonfinite_model_error(exc):
            rec.event("model.error", fields={"operation": "materialize_prefix", "error": str(exc)})
            raise NonfiniteModelError(str(exc)) from exc
        raise
    synchronize_device(capabilities)
    timing_counters["sync_calls"] = timing_counters.get("sync_calls", 0) + 1
    finished_mat = clock_fn()
    timing_counters["materialization_calls"] = timing_counters.get("materialization_calls", 0) + 1
    rec.event(
        "operation.completed",
        fields={
            "operation": "materialize_prefix",
            "finished": finished_mat,
            "duration_s": max(0.0, finished_mat - start_mat),
            "coverage": "inclusive",
        },
    )

    if not np.allclose(prefix.proposal_root, rep_obs.T_w_e, atol=1e-6):
        object.__setattr__(prefix, "proposal_root", np.array(rep_obs.T_w_e, copy=True))

    finished_fallback = finished_mat
    overshoot = max(0.0, finished_fallback - deadline)
    extra_latency = max(0.0, finished_fallback - max(deadline, start_time))

    rec.event(
        "planning.fallback",
        fields={
            "reason": fallback_reason,
            "reused_prior": reused,
            "extra_latency": extra_latency,
            "overshoot": overshoot,
            "native_calls": timing_counters.get("native_calls", 0),
        },
    )

    cand_id = getattr(prefix, "raw_candidate_id", "fallback_0")
    final_audit_ids = tuple(audit_ids) if audit_ids else (str(cand_id),)

    return PlanningResult(
        selected_prefix=prefix,
        completed=False,
        fallback_reason=fallback_reason,
        expected_return=None,
        call_count=timing_counters.get("iterations", 0),
        timing_counters=dict(timing_counters),
        cache_counters=dict(cache_counters) if cache_counters is not None else {"lookups": 0, "hits": 0, "misses": 0},
        audit_ids=final_audit_ids,
    )


