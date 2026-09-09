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
    audit_planning_start,
    eligible_completion,
    evaluate_root_and_check_h0,
    execute_fallback,
    score_leaf_rollout,
    synchronize_device,
    timed_operation,
    validate_root_and_context,
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

    allowed_horizons = (p_cfg.h, 8)
    if horizon is not None:
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise TypeError(f"horizon must be an integer, got {type(horizon)}")
        if horizon not in allowed_horizons:
            raise ValueError(f"horizon must be configured planning.h ({p_cfg.h}) or native chunk 8, got {horizon}")
        rollout_bound = int(horizon)
    else:
        rollout_bound = p_cfg.h

    if rollout_bound <= 0:
        raise ValueError("horizon must be a positive integer")
    step_intervals = min(rollout_bound, H)

    clock_fn = clock if clock is not None else time.monotonic
    rec = recorder if recorder is not None else NoopRecorder()
    start_time = clock_fn()
    deadline = start_time + budget.wall_budget_s if budget.wall_budget_s is not None else float("inf")

    # Strict validation of root geometry and context before any eval
    root_obs = validate_root_and_context(state, context, capabilities)

    # Provenance audit
    audit_planning_start(rec, "rerank", cfg, budget)

    timing_counters = {
        "iterations": 0,
        "native_calls": 0,
        "model_intervals": 0,
        "evaluator_calls": 0,
        "task_tracker_calls": 0,
        "terminal_calls": 0,
        "materialization_calls": 0,
        "sync_calls": 0,
    }

    # Evaluate root and handle H=0 / absorbed root
    h0_res, u_root, should_abort = evaluate_root_and_check_h0(
        state,
        task,
        context,
        H=H,
        budget=budget,
        capabilities=capabilities,
        cfg=cfg,
        clock_fn=clock_fn,
        deadline=deadline,
        recorder=rec,
        timing_counters=timing_counters,
        cache_counters={"lookups": 0, "hits": 0, "misses": 0},
        audit_ids=(),
    )
    if h0_res is not None:
        return h0_res
    if should_abort:
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
            timing_counters=timing_counters,
            recorder=rec,
            fallback_reason="budget_exhausted",
        )

    # Composite IDs
    composite_models = _extract_model_identities(capabilities)
    context_id = _extract_context_identity(context, capabilities)

    f_root = 0.0
    w_root = (1.0 - u_root) / 3.0
    events = getattr(context, "events", context)

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
    nonfinite_error = False

    while True:
        # Check all budget bounds before candidate evaluation
        if clock_fn() >= deadline:
            break
        if not budget.check_wall_budget(clock_fn() - start_time):
            break
        if not budget.check_native_call(timing_counters["native_calls"]):
            break
        if not budget.check_model_interval(timing_counters["model_intervals"]):
            break

        sample_seed = (seed + next_sample_idx * 10007) & 0x7FFFFFFF
        try:
            candidate, _, eligible_cand = timed_operation(
                "sample_prior",
                lambda: capabilities.sample_prior(root_obs, task, context, seed=sample_seed),
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=rec,
                timing_counters=timing_counters,
                counter_key="native_calls",
            )
        except NonfiniteModelError as exc:
            nonfinite_error = True
            rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
            break

        if not eligible_cand or candidate is None:
            break

        if earliest_candidate is None:
            earliest_candidate = candidate

        try:
            prefix, _, eligible_mat = timed_operation(
                "materialize_prefix",
                lambda: capabilities.materialize_prefix(
                    candidate,
                    h=step_intervals,
                    r=min(p_cfg.r, step_intervals),
                    duration_s=dt0,
                ),
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=rec,
                timing_counters=timing_counters,
                counter_key="materialization_calls",
            )
        except NonfiniteModelError as exc:
            nonfinite_error = True
            rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
            break

        if not eligible_mat or prefix is None:
            break

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
                timing_counters,
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
        G, eligible_leaf = score_leaf_rollout(
            leaf_node,
            events,
            remaining_H=rem,
            clock_fn=clock_fn,
            deadline=deadline,
            budget=budget,
            capabilities=capabilities,
            recorder=rec,
            timing_counters=timing_counters,
        )
        if not eligible_leaf or G is None:
            break

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

        timing_counters["iterations"] += 1
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
    if not evaluated_candidates or nonfinite_error:
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
            timing_counters=timing_counters,
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
        call_count=timing_counters["iterations"],
        timing_counters=dict(timing_counters),
        cache_counters=exact_cache.counters(),
        audit_ids=tuple(r.candidate_id for r in audit_records),
    )
