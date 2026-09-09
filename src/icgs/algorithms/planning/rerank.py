"""Rerank planner (B5 baseline) evaluating single-chunk rollouts with exact cache and fallback."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable

import numpy as np

from icgs.algorithms.planning.belief import BeliefNode, Hypothesis, leaf_return
from icgs.algorithms.planning.budget import (
    BudgetExhausted,
    NonfiniteModelError,
    PlanningBudget,
    eligible_completion,
    execute_fallback,
    synchronize_device,
)
from icgs.algorithms.planning.mcts import (
    CandidateAuditRecord,
    ExactCache,
    _branch_copy_task,
    _canonical_command_bytes,
    _extract_context_identity,
    _extract_model_identities,
    _observation_from_state,
    _step_edge,
    _validate_scalar_probability,
)
from icgs.configuration.method import MethodConfig
from icgs.contracts.method import CommandPrefix, MethodCapabilities, PlanningResult
from icgs.observability.recorder import NoopRecorder
from icgs.state.physical import PhysicalState


def plan(
    state: PhysicalState,
    task: Any,
    context: Any,
    *,
    H: int,
    budget: PlanningBudget,
    capabilities: MethodCapabilities,
    cfg: MethodConfig,
    seed: int = 0,
    clock: Callable[[], float] | None = None,
    recorder: Any = None,
    horizon: int | None = None,
) -> PlanningResult:
    """Execute rerank planning evaluating candidate rollouts up to configured horizon."""
    if budget is None or not isinstance(budget, PlanningBudget):
        raise ValueError("Explicit bounded PlanningBudget must be provided to plan")
    if capabilities is None:
        raise ValueError("capabilities must be provided to execute rerank plan")
    if not isinstance(cfg, MethodConfig):
        raise TypeError(f"cfg must be MethodConfig, got {type(cfg)}")

    p_cfg = cfg.planning
    dt0 = float(cfg.control.dt0)

    if isinstance(H, bool) or not isinstance(H, (int, np.integer)) or H < 0:
        raise ValueError("H must be a nonnegative integer")
    H = int(H)
    if H > p_cfg.H:
        raise ValueError(f"H ({H}) cannot exceed cfg.planning.H ({p_cfg.H})")

    if not isinstance(state, PhysicalState):
        raise TypeError(f"state must be PhysicalState, got {type(state)}")

    rollout_bound = p_cfg.h if horizon is None else int(horizon)
    if rollout_bound <= 0:
        raise ValueError("horizon must be a positive integer")
    step_intervals = min(rollout_bound, H)

    clock_fn = clock if clock is not None else time.monotonic
    rec = recorder if recorder is not None else NoopRecorder()
    start_time = clock_fn()
    deadline = start_time + budget.wall_budget_s if budget.wall_budget_s is not None else float("inf")

    rec.event(
        "planning.start",
        fields={
            "boundary": state.boundary,
            "H": H,
            "rollout_intervals": step_intervals,
            "start_time": start_time,
            "deadline": deadline,
            "algorithm": "rerank",
        },
    )

    # Monotonic deadline check: no new operation starts at or beyond deadline
    if clock_fn() >= deadline:
        return execute_fallback(
            state,
            task,
            context,
            H=H,
            budget=budget,
            capabilities=capabilities,
            cfg=cfg,
            seed=seed,
            clock_fn=clock_fn,
            start_time=start_time,
            deadline=deadline,
            earliest_candidate=None,
            native_calls_count=0,
            model_intervals_count=0,
            evaluator_calls_count=0,
            iteration_count=0,
            cache_counters={"lookups": 0, "hits": 0, "misses": 0},
            audit_ids=(),
            recorder=rec,
            fallback_reason="budget_exhausted",
            observation_fn=_observation_from_state,
        )

    # Composite IDs
    composite_models = _extract_model_identities(capabilities)
    context_id = _extract_context_identity(context, capabilities)

    # Validate root observation without fabricating geometry
    root_obs = _observation_from_state(state)

    # Initialize root U from evaluator with strict finite [0, 1] validation
    events = getattr(context, "events", context)
    root_eval = capabilities.evaluate_state(state, task, events, H=H)
    synchronize_device(capabilities)
    evaluator_calls_count = 1
    finished_root = clock_fn()
    u_root = _validate_scalar_probability(root_eval.calibrated_stop, "Root calibrated_stop")
    f_root = 0.0
    w_root = (1.0 - u_root) / 3.0

    # Prompt termination for H=0 or fully absorbed root
    if H == 0 or u_root >= 1.0:
        if eligible_completion(finished_root, deadline):
            return PlanningResult(
                selected_prefix=None,
                completed=True,
                fallback_reason="zero_horizon" if H == 0 else "absorbed_root",
                expected_return=float(u_root),
                call_count=0,
                timing_counters={
                    "iterations": 0,
                    "native_calls": 0,
                    "model_intervals": 0,
                    "evaluator_calls": evaluator_calls_count,
                },
                cache_counters={"lookups": 0, "hits": 0, "misses": 0},
                audit_ids=(),
            )
        else:
            overshoot = max(0.0, finished_root - deadline)
            rec.event(
                "operation.overshoot",
                fields={
                    "operation": "root_eval",
                    "finished": finished_root,
                    "deadline": deadline,
                    "overshoot": overshoot,
                },
            )
            return PlanningResult(
                selected_prefix=None,
                completed=False,
                fallback_reason="zero_horizon_timeout" if H == 0 else "absorbed_root_timeout",
                expected_return=None,
                call_count=0,
                timing_counters={
                    "iterations": 0,
                    "native_calls": 0,
                    "model_intervals": 0,
                    "evaluator_calls": evaluator_calls_count,
                },
                cache_counters={"lookups": 0, "hits": 0, "misses": 0},
                audit_ids=(),
            )

    if not eligible_completion(finished_root, deadline):
        overshoot = max(0.0, finished_root - deadline)
        rec.event(
            "operation.overshoot",
            fields={
                "operation": "root_eval",
                "finished": finished_root,
                "deadline": deadline,
                "overshoot": overshoot,
            },
        )
        return execute_fallback(
            state,
            task,
            context,
            H=H,
            budget=budget,
            capabilities=capabilities,
            cfg=cfg,
            seed=seed,
            clock_fn=clock_fn,
            start_time=start_time,
            deadline=deadline,
            earliest_candidate=None,
            native_calls_count=0,
            model_intervals_count=0,
            evaluator_calls_count=evaluator_calls_count,
            iteration_count=0,
            cache_counters={"lookups": 0, "hits": 0, "misses": 0},
            audit_ids=(),
            recorder=rec,
            fallback_reason="budget_exhausted",
            observation_fn=_observation_from_state,
        )

    root_hypotheses = tuple(
        Hypothesis(
            head_id=m,
            state=state.branch_copy(),
            task=_branch_copy_task(task, capabilities),
            weight=w_root,
        )
        for m in range(3)
    )
    root_node = BeliefNode(
        tau=0,
        H_root=H,
        U=u_root,
        F=f_root,
        hypotheses=root_hypotheses,
        cfg=cfg,
    )

    # Root-scoped exact cache
    root_cache_id = f"rerank_{state.boundary}_{time.monotonic_ns()}_{seed}"
    exact_cache = ExactCache(root_cache_id)

    # Accounting and candidate tracking
    audit_records: list[CandidateAuditRecord] = []
    seen_command_hashes: set[str] = set()
    earliest_candidate: Any | None = None
    evaluated_candidates: list[tuple[float, int, CommandPrefix]] = []
    next_sample_idx = 0
    native_calls_count = 0
    model_intervals_tracker = [0]
    iteration_count = 0
    nonfinite_error = False

    while True:
        # Check all budget bounds before candidate evaluation
        if clock_fn() >= deadline:
            break
        if not budget.check_wall_budget(clock_fn() - start_time):
            break
        if not budget.check_native_call(native_calls_count):
            break
        if not budget.check_model_interval(model_intervals_tracker[0]):
            break

        # Sample prior candidate proposal
        sample_seed = (seed + next_sample_idx * 10007) & 0x7FFFFFFF
        try:
            candidate = capabilities.sample_prior(root_obs, task, context, seed=sample_seed)
        except NonfiniteModelError as exc:
            nonfinite_error = True
            rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
            break
        synchronize_device(capabilities)
        native_calls_count += 1
        if earliest_candidate is None:
            earliest_candidate = candidate
        finished_sample = clock_fn()

        prefix = capabilities.materialize_prefix(
            candidate,
            h=step_intervals,
            r=min(p_cfg.r, step_intervals),
            duration_s=dt0,
        )
        if len(prefix.commands) != step_intervals:
            raise ValueError(
                f"materialize_prefix returned {len(prefix.commands)} commands, expected exactly {step_intervals}"
            )

        cand_id = prefix.raw_candidate_id
        prefix_bytes = b"".join(_canonical_command_bytes(c) for c in prefix.commands)
        cmd_hash = hashlib.sha256(prefix_bytes).hexdigest()
        is_dup = cmd_hash in seen_command_hashes
        seen_command_hashes.add(cmd_hash)

        audit_rec = CandidateAuditRecord(
            candidate_id=str(cand_id),
            order=next_sample_idx,
            is_duplicate=is_dup,
            commands_hash=cmd_hash,
        )
        audit_records.append(audit_rec)
        rec.event(
            "candidate.audit",
            fields={
                "candidate_id": audit_rec.candidate_id,
                "order": audit_rec.order,
                "is_duplicate": audit_rec.is_duplicate,
                "commands_hash": audit_rec.commands_hash,
            },
        )
        cand_order = next_sample_idx
        next_sample_idx += 1

        if not eligible_completion(finished_sample, deadline):
            overshoot = max(0.0, finished_sample - deadline)
            rec.event(
                "operation.overshoot",
                fields={
                    "operation": "sample_prior",
                    "finished": finished_sample,
                    "deadline": deadline,
                    "overshoot": overshoot,
                },
            )
            break

        # Step rollout across all 3 heads
        try:
            leaf_node = _step_edge(
                root_node,
                prefix,
                context,
                capabilities,
                exact_cache,
                cfg,
                budget,
                clock_fn,
                start_time,
                model_intervals_tracker,
                composite_models,
                context_id,
                deadline=deadline,
                recorder=rec,
            )
        except BudgetExhausted:
            break
        except NonfiniteModelError as exc:
            nonfinite_error = True
            rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
            break

        # Leaf evaluation
        rem = leaf_node.H_root - leaf_node.tau
        if rem == 0 or leaf_node.is_terminal:
            G = leaf_return(leaf_node.U, leaf_node.weights, np.zeros(3, dtype=np.float64), 0)
        else:
            if clock_fn() >= deadline or not budget.check_wall_budget(clock_fn() - start_time):
                break
            values = np.zeros(3, dtype=np.float64)
            leaf_eval_failed = False
            for m in range(3):
                h_m = leaf_node.hypotheses[m]
                if h_m.weight > 0.0:
                    if clock_fn() >= deadline:
                        leaf_eval_failed = True
                        break
                    try:
                        eval_res = capabilities.evaluate_state(h_m.state, h_m.task, events, rem)
                    except NonfiniteModelError as exc:
                        nonfinite_error = True
                        rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
                        leaf_eval_failed = True
                        break
                    synchronize_device(capabilities)
                    evaluator_calls_count += 1
                    finished_eval = clock_fn()
                    if not eligible_completion(finished_eval, deadline):
                        overshoot = max(0.0, finished_eval - deadline)
                        rec.event(
                            "operation.overshoot",
                            fields={
                                "operation": "evaluate_state",
                                "head_id": m,
                                "finished": finished_eval,
                                "deadline": deadline,
                                "overshoot": overshoot,
                            },
                        )
                        leaf_eval_failed = True
                        break
                    cv = _validate_scalar_probability(eval_res.calibrated_value, "Leaf calibrated_value")
                    values[m] = cv
            if leaf_eval_failed:
                break
            G = leaf_return(leaf_node.U, leaf_node.weights, values, rem)

        finished_cand = clock_fn()
        if not eligible_completion(finished_cand, deadline):
            overshoot = max(0.0, finished_cand - deadline)
            rec.event(
                "operation.overshoot",
                fields={
                    "operation": "rerank.candidate",
                    "finished": finished_cand,
                    "deadline": deadline,
                    "overshoot": overshoot,
                },
            )
            break

        iteration_count += 1
        evaluated_candidates.append((G, cand_order, prefix))
        rec.event(
            "rerank.evaluated",
            fields={
                "candidate_order": cand_order,
                "G": G,
                "prefix_commands": len(prefix.commands),
            },
        )

    # Candidate selection: highest G, breaking ties to earlier sampled order
    if not evaluated_candidates:
        fb_reason = "nonfinite_model_error" if nonfinite_error else "budget_exhausted"
        return execute_fallback(
            state,
            task,
            context,
            H=H,
            budget=budget,
            capabilities=capabilities,
            cfg=cfg,
            seed=seed,
            clock_fn=clock_fn,
            start_time=start_time,
            deadline=deadline,
            earliest_candidate=earliest_candidate,
            native_calls_count=native_calls_count,
            model_intervals_count=model_intervals_tracker[0],
            evaluator_calls_count=evaluator_calls_count,
            iteration_count=iteration_count,
            cache_counters=exact_cache.counters(),
            audit_ids=tuple(r.candidate_id for r in audit_records),
            recorder=rec,
            fallback_reason=fb_reason,
            observation_fn=_observation_from_state,
        )

    best_G, best_order, best_prefix = max(
        evaluated_candidates,
        key=lambda item: (item[0], -item[1]),
    )
    rec.event(
        "rerank.choice",
        fields={
            "selected_order": best_order,
            "G": best_G,
        },
    )

    return PlanningResult(
        selected_prefix=best_prefix,
        completed=True,
        fallback_reason=None,
        expected_return=best_G,
        call_count=iteration_count,
        timing_counters={
            "iterations": iteration_count,
            "native_calls": native_calls_count,
            "model_intervals": model_intervals_tracker[0],
            "evaluator_calls": evaluator_calls_count,
        },
        cache_counters=exact_cache.counters(),
        audit_ids=tuple(r.candidate_id for r in audit_records),
    )
