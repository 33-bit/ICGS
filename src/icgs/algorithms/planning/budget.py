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



def synchronize_device(capabilities: Any) -> None:
    """Synchronize device operations via capability protocol (no duck typing)."""
    capabilities.synchronize()


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
    permit_beyond_deadline: bool = False,
) -> tuple[Any, float, bool]:
    """Execute capability operation with immediate precheck, explicit sync, completed/overshoot audit, and counters."""
    rec = recorder if recorder is not None else NoopRecorder()
    start_t = clock_fn()
    if not permit_beyond_deadline and start_t >= deadline:
        return None, start_t, False

    if timing_counters is not None and counter_key is not None:
        timing_counters[counter_key] = timing_counters.get(counter_key, 0) + 1

    rec.event(
        "operation.started",
        fields={
            "operation": op_name,
            "started": start_t,
            "deadline": deadline,
            "coverage": coverage,
        },
    )

    try:
        res = op()
    except Exception as op_exc:
        if timing_counters is not None:
            timing_counters["sync_calls"] = timing_counters.get("sync_calls", 0) + 1
        sync_exc = None
        try:
            capabilities.synchronize()
        except Exception as se:
            sync_exc = se

        finished_err = clock_fn()
        duration_err = max(0.0, finished_err - start_t)
        if isinstance(op_exc, NonfiniteModelError):
            rec.event(
                "model.error",
                fields={
                    "operation": op_name,
                    "error": "nonfinite_model_output",
                    "detail": str(op_exc),
                    "duration_s": duration_err,
                    "finished": finished_err,
                    "sync_error": str(sync_exc) if sync_exc is not None else None,
                },
            )
        rec.event(
            "operation.error",
            fields={
                "operation": op_name,
                "error": type(op_exc).__name__,
                "detail": str(op_exc),
                "duration_s": duration_err,
                "finished": finished_err,
                "sync_error": str(sync_exc) if sync_exc is not None else None,
            },
        )
        raise op_exc

    if timing_counters is not None:
        timing_counters["sync_calls"] = timing_counters.get("sync_calls", 0) + 1
    try:
        capabilities.synchronize()
    except Exception as sync_exc:
        finished_err = clock_fn()
        duration_err = max(0.0, finished_err - start_t)
        rec.event(
            "operation.error",
            fields={
                "operation": op_name,
                "error": type(sync_exc).__name__,
                "detail": f"synchronization_failed: {sync_exc}",
                "duration_s": duration_err,
                "finished": finished_err,
                "sync_error": str(sync_exc),
            },
        )
        raise sync_exc

    finished_t = clock_fn()
    duration_s = max(0.0, finished_t - start_t)

    if permit_beyond_deadline:
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

    eligible = eligible_completion(finished_t, deadline)
    if eligible:
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
    """Strictly validate root physical state geometry and context via capability protocol without mutating live state."""
    import dataclasses
    from icgs.state.physical import PhysicalState

    if not isinstance(state, PhysicalState):
        raise TypeError("state must be PhysicalState")

    capabilities.validate_context(context)

    # Validate physical state on isolated replace copy without mutating caller's live state
    try:
        dataclasses.replace(state)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Root physical state validation failed: {exc}") from exc

    pts = state.cached_world_cloud if state.cached_world_cloud is not None else state.x
    mask = state.cached_world_cloud_valid if state.cached_world_cloud_valid is not None else state.valid
    if pts is None or mask is None:
        raise ValueError("state points and mask must not be None")
    if not bool(mask.any().item()):
        raise ValueError("points mask must contain at least one valid point")

    pts_valid = pts[0][mask[0]]
    pts_np = np.ascontiguousarray(pts_valid.detach().cpu().numpy())
    if pts_np.ndim != 2 or pts_np.shape[1] != 3 or pts_np.shape[0] == 0:
        raise ValueError(f"Observation points must be non-empty (N, 3) array, got shape {pts_np.shape}")
    if not np.all(np.isfinite(pts_np)):
        raise ValueError("Observation points must contain finite values")

    pose_np = np.ascontiguousarray(state.T_w_e[0].detach().cpu().numpy())
    grip_val = float(state.grip[0].item())

    return Observation(points=pts_np, T_w_e=pose_np, grip=grip_val)


def validate_predicted_state(state: Any) -> None:
    """Validate all selectable/cacheable state fields for finiteness, SE(3) validity, and shapes without mutating live state."""
    import dataclasses
    from icgs.state.physical import PhysicalState

    if not isinstance(state, PhysicalState):
        raise NonfiniteModelError("predicted state is not a PhysicalState")
    try:
        dataclasses.replace(state)
    except (ValueError, TypeError) as exc:
        raise NonfiniteModelError(f"predicted state validation failed: {exc}") from exc

    if not bool(state.valid.any().item()):
        raise NonfiniteModelError("predicted state.valid must contain at least one valid point")
    if state.cached_world_cloud_valid is not None and not bool(state.cached_world_cloud_valid.any().item()):
        raise NonfiniteModelError("predicted state cached_world_cloud_valid must contain at least one valid point")


def validate_evaluation_output(eval_out: Any, name: str = "Evaluation") -> tuple[float, float]:
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
        raise ValueError(f"{name} calibrated_stop must be a scalar")
    u_val = float(stop_np.reshape(-1)[0])
    if not isfinite(u_val):
        raise NonfiniteModelError(f"{name} calibrated_stop must be finite, got {u_val}")
    if u_val < 0.0 or u_val > 1.0:
        raise ValueError(f"{name} calibrated_stop must be in [0, 1], got {u_val}")

    if hasattr(val_raw, "detach"):
        val_np = val_raw.detach().cpu().numpy()
    else:
        val_np = np.asarray(val_raw)

    if not np.all(np.isfinite(val_np)):
        raise NonfiniteModelError(f"{name} calibrated_value contains nonfinite values")

    if val_np.size != 1:
        raise ValueError(f"{name} calibrated_value must be a single scalar, got array of shape {val_np.shape}")
    c_val = float(val_np.reshape(-1)[0])
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
    *,
    H: int,
    boundary: int = 0,
    deadline: float | None = None,
    algorithm_params: Mapping[str, Any] | None = None,
) -> None:
    """Audit planning start with resolved config envelope, effective invocation limits, and clock track."""
    from icgs.configuration.method import MethodConfig

    if not isinstance(cfg, MethodConfig):
        raise TypeError(f"cfg must be MethodConfig, got {type(cfg)}")

    wall_status = (
        "not_applied"
        if budget.wall_budget_s is None
        else ("matched" if budget.wall_budget_s in cfg.planning.wall_budgets_s else "override")
    )
    native_status = (
        "not_applied"
        if budget.native_call_cap is None
        else ("matched" if budget.native_call_cap in cfg.planning.native_call_caps else "override")
    )
    model_status = (
        "not_applied"
        if budget.model_interval_cap is None
        else ("matched" if budget.model_interval_cap == cfg.planning.model_interval_cap else "override")
    )
    clock_track_status = "matched" if budget.clock_track == cfg.control.clock_track else "override"

    applied_statuses = [s for s in (wall_status, native_status, model_status) if s != "not_applied"]
    panel_matched = bool(
        clock_track_status == "matched"
        and len(applied_statuses) > 0
        and all(s == "matched" for s in applied_statuses)
    )

    fields: dict[str, Any] = {
        "algorithm": alg_name,
        "resolved_config": cfg.resolved_config(),
        "config_sha256": cfg.fingerprint(),
        "schema_version": cfg.schema_version,
        "H": H,
        "boundary": boundary,
        "deadline": deadline,
        "wall_budget_s": budget.wall_budget_s,
        "native_call_cap": budget.native_call_cap,
        "model_interval_cap": budget.model_interval_cap,
        "clock_track": budget.clock_track,
        "configured_clock_track": cfg.control.clock_track,
        "clock_track_status": clock_track_status,
        "wall_panel_status": wall_status,
        "native_panel_status": native_status,
        "model_panel_status": model_status,
        "panel_matched": panel_matched,
    }
    if algorithm_params:
        fields["algorithm_params"] = dict(algorithm_params)
        fields.update(algorithm_params)

    rec.event("planning.start", fields=fields)


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
) -> tuple[PlanningResult | None, float, bool, str]:
    """Evaluate root state and check H=0 / absorbed root conditions.

    Returns:
        (result, u_root, should_abort_to_fallback, abort_reason)
    """
    events = getattr(context, "events", context)

    def _eval_root_op() -> tuple[float, float]:
        raw_eval = capabilities.evaluate_state(state, task, events, H=H)
        return validate_evaluation_output(raw_eval, "Root evaluation")

    try:
        eval_res, finished_t, eligible = timed_operation(
            "root_eval",
            _eval_root_op,
            clock_fn=clock_fn,
            deadline=deadline,
            budget=budget,
            capabilities=capabilities,
            recorder=recorder,
            coverage="inclusive",
            timing_counters=timing_counters,
            counter_key="evaluator_calls",
        )
    except NonfiniteModelError:
        if H == 0:
            return (
                PlanningResult(
                    selected_prefix=None,
                    completed=False,
                    fallback_reason="nonfinite_model_error",
                    expected_return=None,
                    call_count=0,
                    timing_counters=dict(timing_counters),
                    cache_counters=dict(cache_counters),
                    audit_ids=tuple(audit_ids),
                ),
                0.0,
                False,
                "nonfinite_model_error",
            )
        return None, 0.0, True, "nonfinite_model_error"

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
                "zero_horizon_timeout",
            )
        return None, 0.0, True, "budget_exhausted"

    u_root, _ = eval_res

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
                "",
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
                "zero_horizon_timeout",
            )

    if not eligible:
        # H > 0 and root eval overshot deadline: must trigger reference fallback!
        return None, u_root, True, "budget_exhausted"

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
            "",
        )

    return None, u_root, False, ""


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
            def _eval_leaf_op(m=m):
                raw = capabilities.evaluate_state(h_m.state, h_m.task, events, H=remaining_H)
                return validate_evaluation_output(raw, f"Leaf head {m} evaluation")

            eval_res, finished_t, eligible = timed_operation(
                "leaf_eval",
                _eval_leaf_op,
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=recorder,
                coverage="inclusive",
                timing_counters=timing_counters,
                counter_key="evaluator_calls",
            )
            if not eligible or eval_res is None:
                return None, False
            u_m, cv = eval_res
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
    """Execute reference fallback: reuse earliest candidate or sample once, materialize, and audit."""
    rec = recorder if recorder is not None else NoopRecorder()
    p_cfg = cfg.planning

    if observation_fn is not None:
        rep_obs = observation_fn(state)
    else:
        rep_obs = validate_root_and_context(state, context, capabilities)

    step_intervals = min(p_cfg.h, H) if H > 0 else 0
    reused = False

    if earliest_candidate is not None:
        cand = earliest_candidate
        reused = True
    else:
        cand, finished_samp, _ = timed_operation(
            "sample_prior",
            lambda: capabilities.sample_prior(rep_obs, task, context, seed=seed),
            clock_fn=clock_fn,
            deadline=deadline,
            budget=budget,
            capabilities=capabilities,
            recorder=rec,
            coverage="inclusive",
            timing_counters=timing_counters,
            counter_key="native_calls",
            permit_beyond_deadline=True,
        )

    r_val = min(p_cfg.r, step_intervals)
    dt0 = float(cfg.control.dt0)

    prefix, finished_mat, _ = timed_operation(
        "materialize_prefix",
        lambda: capabilities.materialize_prefix(cand, h=step_intervals, r=r_val, duration_s=dt0),
        clock_fn=clock_fn,
        deadline=deadline,
        budget=budget,
        capabilities=capabilities,
        recorder=rec,
        coverage="inclusive",
        timing_counters=timing_counters,
        counter_key="materialization_calls",
        permit_beyond_deadline=True,
    )

    if not np.allclose(prefix.proposal_root, rep_obs.T_w_e, atol=1e-6):
        raise ValueError(
            f"Fallback prefix proposal_root does not match root pose: expected {rep_obs.T_w_e}, got {prefix.proposal_root}"
        )

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


