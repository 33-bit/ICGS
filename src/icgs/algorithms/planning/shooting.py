"""Shooting planner (B6 baseline) resampling prior sequentially through horizon L."""

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
) -> PlanningResult:
    """Execute shooting planner sequentially resampling prior at predicted nodes up to horizon L."""
    if budget is None or not isinstance(budget, PlanningBudget):
        raise ValueError("Explicit bounded PlanningBudget must be provided to plan")
    if capabilities is None:
        raise ValueError("capabilities must be provided to execute shooting plan")
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

    clock_fn = clock if clock is not None else time.monotonic
    rec = recorder if recorder is not None else NoopRecorder()
    start_time = clock_fn()
    deadline = start_time + budget.wall_budget_s if budget.wall_budget_s is not None else float("inf")

    # Strict validation of root geometry and context before any eval
    root_obs = validate_root_and_context(state, context, capabilities)

    # Provenance audit
    audit_planning_start(
        rec,
        "shooting",
        cfg,
        budget,
        H=H,
        boundary=state.boundary,
        deadline=deadline,
        algorithm_params={"L": p_cfg.L, "h": p_cfg.h},
    )

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
    h0_res, u_root, should_abort, abort_reason = evaluate_root_and_check_h0(
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
            fallback_reason=abort_reason,
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
    root_cache_id = f"shooting_{state.boundary}_{time.monotonic_ns()}_{seed}"
    exact_cache = ExactCache(root_cache_id)

    # Accounting and candidate tracking
    audit_records: list[CandidateAuditRecord] = []
    seen_command_hashes: set[str] = set()
    earliest_candidate: Any | None = None
    evaluated_sequences: list[tuple[float, int, CommandPrefix]] = []
    next_sample_idx = 0
    seq_completed_count = 0
    nonfinite_error = False

    while True:
        # Check all budget bounds before sequence rollout
        if clock_fn() >= deadline:
            break
        if not budget.check_wall_budget(clock_fn() - start_time):
            break
        if not budget.check_native_call(timing_counters["native_calls"]):
            break
        if not budget.check_model_interval(timing_counters["model_intervals"]):
            break

        curr_node = root_node
        seq_prefixes: list[CommandPrefix] = []
        sequence_failed = False

        # Resample sequentially up to min(L, H)
        while curr_node.tau < min(p_cfg.L, H) and not curr_node.is_terminal:
            step_intervals = min(p_cfg.h, H - curr_node.tau, p_cfg.L - curr_node.tau)
            if step_intervals <= 0:
                break

            if clock_fn() >= deadline:
                sequence_failed = True
                break
            if not budget.check_native_call(timing_counters["native_calls"]):
                sequence_failed = True
                break

            # Resample prior at current predicted node representative
            rep_hyp = curr_node.select_representative()
            rep_obs = _observation_from_state(rep_hyp.state)

            sample_seed = (seed + next_sample_idx * 10007) & 0x7FFFFFFF
            try:
                candidate, _, eligible_cand = timed_operation(
                    "sample_prior",
                    lambda: capabilities.sample_prior(rep_obs, rep_hyp.task, context, seed=sample_seed),
                    clock_fn=clock_fn,
                    deadline=deadline,
                    budget=budget,
                    capabilities=capabilities,
                    recorder=rec,
                    timing_counters=timing_counters,
                    counter_key="native_calls",
                )
            except NonfiniteModelError:
                nonfinite_error = True
                sequence_failed = True
                break

            if candidate is not None and earliest_candidate is None:
                earliest_candidate = candidate

            if not eligible_cand or candidate is None:
                sequence_failed = True
                break

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
            except NonfiniteModelError:
                nonfinite_error = True
                sequence_failed = True
                break

            if not eligible_mat or prefix is None:
                sequence_failed = True
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
            next_sample_idx += 1

            # Step forward
            try:
                child_node = _step_edge(
                    curr_node,
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
                sequence_failed = True
                break
            except NonfiniteModelError:
                nonfinite_error = True
                sequence_failed = True
                break

            seq_prefixes.append(prefix)
            curr_node = child_node

        if sequence_failed or not seq_prefixes:
            break

        # Leaf evaluation for sequence leaf node
        rem = curr_node.H_root - curr_node.tau
        try:
            G, eligible_leaf = score_leaf_rollout(
                curr_node,
                events,
                remaining_H=rem,
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=rec,
                timing_counters=timing_counters,
            )
        except NonfiniteModelError:
            nonfinite_error = True
            break
        if not eligible_leaf or G is None:
            break

        finished_seq = clock_fn()
        if not eligible_completion(finished_seq, deadline):
            overshoot = max(0.0, finished_seq - deadline)
            rec.event(
                "operation.overshoot",
                fields={
                    "operation": "shooting.sequence",
                    "finished": finished_seq,
                    "deadline": deadline,
                    "overshoot": overshoot,
                },
            )
            break

        timing_counters["iterations"] += 1
        evaluated_sequences.append((G, seq_completed_count, seq_prefixes[0]))
        rec.event(
            "shooting.evaluated",
            fields={
                "seq_index": seq_completed_count,
                "G": G,
                "chunks": len(seq_prefixes),
                "total_steps": curr_node.tau,
            },
        )
        seq_completed_count += 1

    # Selection: highest G, breaking ties to earliest completed sequence order
    if not evaluated_sequences:
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

    best_G, best_seq_idx, best_prefix = max(
        evaluated_sequences,
        key=lambda item: (item[0], -item[1]),
    )
    rec.event(
        "shooting.choice",
        fields={
            "selected_seq_index": best_seq_idx,
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
