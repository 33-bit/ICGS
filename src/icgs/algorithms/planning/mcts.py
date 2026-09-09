"""Progressive-widening MCTS over common absolute prefixes with exact root-scoped cache."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from math import isfinite
import time
from typing import Any, Mapping, Sequence

import numpy as np

from icgs.algorithms.planning.belief import (
    BeliefNode,
    Hypothesis,
    leaf_return,
    propagate_mass,
    select_representative,
)
from icgs.algorithms.planning.budget import PlanningBudget
from icgs.configuration.method import MethodConfig, PlanningConfig
from icgs.contracts.method import CommandPrefix, MethodCapabilities, PlanningResult, TimedCommand
from icgs.state.physical import PhysicalState


def widening_limit(
    visits: int,
    cfg: MethodConfig | PlanningConfig | None = None,
) -> int:
    """Calculate progressive widening candidate limit for a node."""
    if cfg is None:
        cfg = MethodConfig().planning
    elif isinstance(cfg, MethodConfig):
        cfg = cfg.planning
    elif not isinstance(cfg, PlanningConfig):
        if not (hasattr(cfg, "widening_coefficient") and hasattr(cfg, "widening_exponent")):
            raise TypeError("cfg must be a MethodConfig or PlanningConfig")

    if isinstance(visits, bool) or not isinstance(visits, (int, np.integer)):
        raise TypeError("visits must be an integer")
    if visits < 0:
        raise ValueError("visits must be nonnegative")
    visits = int(visits)

    coeff = float(cfg.widening_coefficient)
    exponent = float(cfg.widening_exponent)
    limit = max(1, math.floor(coeff * ((1 + visits) ** exponent)))
    return int(limit)


def uct(
    total_return: float,
    visits: int,
    parent_visits: int,
    cfg: MethodConfig | PlanningConfig | None = None,
) -> float:
    """Calculate UCT selection value for a visited edge."""
    if cfg is None:
        cfg = MethodConfig().planning
    elif isinstance(cfg, MethodConfig):
        cfg = cfg.planning
    elif not isinstance(cfg, PlanningConfig):
        if not hasattr(cfg, "uct_exploration"):
            raise TypeError("cfg must be a MethodConfig or PlanningConfig")

    if isinstance(visits, bool) or not isinstance(visits, (int, np.integer)):
        raise TypeError("visits must be an integer")
    if isinstance(parent_visits, bool) or not isinstance(parent_visits, (int, np.integer)):
        raise TypeError("parent_visits must be an integer")
    if visits < 0:
        raise ValueError("visits must be nonnegative")
    if parent_visits < 0:
        raise ValueError("parent_visits must be nonnegative")
    visits = int(visits)
    parent_visits = int(parent_visits)

    if visits == 0:
        raise ValueError("Cannot evaluate uct on unvisited edge (visits must be positive)")

    if isinstance(total_return, bool) or not isinstance(total_return, (int, float, np.floating, np.integer)):
        raise TypeError("total_return must be a numeric float")
    total_return = float(total_return)
    if not math.isfinite(total_return):
        raise ValueError("total_return must be finite")

    c = float(cfg.uct_exploration)
    return (total_return / visits) + c * math.sqrt(math.log(1 + parent_visits) / visits)


@dataclass
class MCTSEdge:
    """Directed search edge storing absolute command prefix, child node, and visit statistics."""

    prefix: CommandPrefix
    child: BeliefNode
    visits: int = 0
    total_return: float = 0.0
    edge_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, CommandPrefix):
            raise TypeError("prefix must be a CommandPrefix")
        if not isinstance(self.child, BeliefNode):
            raise TypeError("child must be a BeliefNode")
        if isinstance(self.visits, bool) or not isinstance(self.visits, (int, np.integer)) or self.visits < 0:
            raise ValueError("visits must be a nonnegative integer")
        self.visits = int(self.visits)

        if isinstance(self.total_return, bool) or not isinstance(self.total_return, (int, float, np.floating, np.integer)):
            raise TypeError("total_return must be a float")
        self.total_return = float(self.total_return)
        if not math.isfinite(self.total_return):
            raise ValueError("total_return must be finite")

        if isinstance(self.edge_id, bool) or not isinstance(self.edge_id, (int, np.integer)) or self.edge_id < 0:
            raise ValueError("edge_id must be a nonnegative integer")
        self.edge_id = int(self.edge_id)

    @property
    def q_value(self) -> float:
        """Mean empirical return through this edge."""
        if self.visits == 0:
            return 0.0
        return self.total_return / self.visits


@dataclass(frozen=True)
class CacheKey:
    """Exact transition cache key with parent state, context, head, model, and time lineage and canonical command bytes."""

    parent_lineage: tuple[str, str, int, bytes, str]
    context_id: str
    head_id: int
    model_id: str
    tau: int
    command_bytes: bytes


class ExactCache:
    """Root-scoped exact transition cache requiring full lineage and canonical command bytes."""

    def __init__(self, root_id: str) -> None:
        self._root_id = str(root_id)
        self._cache: dict[CacheKey, tuple[PhysicalState, Any, np.ndarray]] = {}
        self.lookups: int = 0
        self.hits: int = 0
        self.misses: int = 0

    @property
    def root_id(self) -> str:
        return self._root_id

    def get(self, key: CacheKey) -> tuple[PhysicalState, Any, np.ndarray] | None:
        self.lookups += 1
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: CacheKey, transition: tuple[PhysicalState, Any, np.ndarray]) -> None:
        self._cache[key] = transition

    def counters(self) -> dict[str, int]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass(frozen=True)
class CandidateAuditRecord:
    """Audit log entry for candidate proposal with duplicate detection."""

    sample_index: int
    candidate_id: str
    is_duplicate: bool
    commands_hash: str


def _extract_task_lineage(task: Any) -> str:
    if task is None:
        return "none"
    for attr in ("lineage", "history_id", "task_id"):
        if hasattr(task, attr):
            return str(getattr(task, attr))
    if hasattr(task, "boundary"):
        return f"task_boundary_{task.boundary}"
    if isinstance(task, Mapping):
        try:
            return json.dumps(task, sort_keys=True)
        except Exception:
            pass
    return str(task)


def _extract_context_id(context: Any) -> str:
    if context is None:
        return "none"
    for attr in ("reference_id", "context_id", "source_id"):
        if hasattr(context, attr):
            return str(getattr(context, attr))
    return str(id(context))


def _canonical_command_bytes(command: TimedCommand) -> bytes:
    target_bytes = command.target_w.tobytes()
    grip_bytes = str(int(command.grip)).encode("ascii")
    dur_bytes = float(command.duration_s).hex().encode("ascii")
    return target_bytes + b":" + grip_bytes + b":" + dur_bytes


def _step_edge(
    node: BeliefNode,
    prefix: CommandPrefix,
    context: Any,
    capabilities: MethodCapabilities,
    cache: ExactCache,
    cfg: MethodConfig,
) -> BeliefNode:
    """Advance all 3 heads through common absolute prefix commands, updating state, task, and hazards each interval."""
    events = getattr(context, "events", context)
    curr_tau = node.tau
    curr_U = node.U
    curr_F = node.F
    curr_weights = np.array(node.weights, copy=True)
    curr_hypotheses = list(node.hypotheses)
    model_id = getattr(capabilities, "model_id", "default_model")
    context_id = _extract_context_id(context)

    for k, cmd in enumerate(prefix.commands):
        step_tau = curr_tau + k
        cmd_bytes = _canonical_command_bytes(cmd)
        next_states: list[PhysicalState] = []
        next_tasks: list[Any] = []
        event_probs = np.zeros((3, 3), dtype=np.float64)

        for m in range(3):
            h_m = curr_hypotheses[m]
            parent_lineage = (
                h_m.state.encoder_lineage,
                h_m.state.memory_lineage,
                h_m.state.boundary,
                h_m.state.T_w_e.detach().cpu().numpy().tobytes(),
                _extract_task_lineage(h_m.task),
            )
            key = CacheKey(
                parent_lineage=parent_lineage,
                context_id=context_id,
                head_id=m,
                model_id=model_id,
                tau=step_tau,
                command_bytes=cmd_bytes,
            )

            cached = cache.get(key)
            if cached is not None:
                next_st, next_tk, probs = cached
            else:
                pred = capabilities.predict_step(h_m.state, cmd, head_id=m)
                next_st = pred.next_state
                next_tk = capabilities.track_task(h_m.task, next_st, events)
                term = capabilities.predict_terminal(h_m.state, h_m.task, next_st, next_tk, events, cmd)
                probs = np.asarray(term.probabilities if hasattr(term, "probabilities") else term.probs, dtype=np.float64)
                if probs.ndim == 2:
                    probs = probs[0]
                cache.put(key, (next_st, next_tk, probs))

            next_states.append(next_st)
            next_tasks.append(next_tk)
            event_probs[m] = probs

        curr_U, curr_F, curr_weights = propagate_mass(curr_U, curr_F, curr_weights, event_probs, cfg)
        curr_hypotheses = [
            Hypothesis(head_id=m, state=next_states[m], task=next_tasks[m], weight=curr_weights[m])
            for m in range(3)
        ]

    return BeliefNode(
        tau=curr_tau + len(prefix.commands),
        H_root=node.H_root,
        U=curr_U,
        F=curr_F,
        hypotheses=tuple(curr_hypotheses),
        cfg=cfg,
    )


def plan(
    state: PhysicalState,
    task: Any,
    context: Any,
    *,
    H: int,
    budget: PlanningBudget | None = None,
    capabilities: MethodCapabilities | None = None,
    cfg: MethodConfig | None = None,
    seed: int = 0,
    initial_S: float | None = None,
    iterations: int | None = None,
) -> PlanningResult:
    """Execute progressive-widening MCTS over common absolute prefixes with root-scoped exact cache."""
    if capabilities is None:
        raise ValueError("capabilities must be provided to execute MCTS plan")
    if cfg is None:
        cfg = MethodConfig()
    elif not isinstance(cfg, MethodConfig):
        raise TypeError("cfg must be a MethodConfig")

    if isinstance(H, bool) or not isinstance(H, (int, np.integer)) or H < 0:
        raise ValueError("H must be a nonnegative integer")
    H = int(H)
    if H > cfg.planning.H:
        raise ValueError(f"H ({H}) cannot exceed cfg.planning.H ({cfg.planning.H})")

    if not isinstance(state, PhysicalState):
        raise TypeError(f"state must be PhysicalState, got {type(state)}")

    # Initialize root U from initial_S or evaluation
    events = getattr(context, "events", context)
    if initial_S is not None:
        if isinstance(initial_S, bool) or not isinstance(initial_S, (int, float, np.floating)):
            raise TypeError("initial_S must be a numeric float")
        u_root = float(initial_S)
    else:
        root_eval = capabilities.evaluate_state(state, task, events, H=H)
        c_stop = root_eval.calibrated_stop
        if isinstance(c_stop, np.ndarray):
            u_root = float(c_stop.flat[0])
        else:
            u_root = float(c_stop)
    u_root = float(np.clip(u_root, 0.0, 1.0))
    f_root = 0.0
    w_root = (1.0 - u_root) / 3.0

    root_hypotheses = tuple(
        Hypothesis(head_id=m, state=state, task=task, weight=w_root)
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

    # Cache is strictly scoped to this one root
    root_cache_id = f"root_{state.boundary}_{time.monotonic_ns()}_{seed}"
    exact_cache = ExactCache(root_cache_id)

    # Accounting and audit tracking
    audit_records: list[CandidateAuditRecord] = []
    seen_command_hashes: set[str] = set()
    next_edge_id = 0
    next_sample_idx = 0
    native_calls_count = 0
    model_intervals_count = 0
    iteration_count = 0

    # Determine iteration limit and budget bounds
    start_time = time.perf_counter()
    max_iters = iterations
    if max_iters is None and budget is not None:
        max_iters = budget.iterations_cap
    if max_iters is None and budget is not None and budget.wall_budget_s is None and budget.native_call_cap is None:
        max_iters = 1

    while True:
        # Check budget limits
        if max_iters is not None and iteration_count >= max_iters:
            break
        if budget is not None:
            if budget.wall_budget_s is not None and (time.perf_counter() - start_time) >= budget.wall_budget_s:
                break
            if budget.native_call_cap is not None and native_calls_count >= budget.native_call_cap:
                break
            if budget.model_interval_cap is not None and model_intervals_count >= budget.model_interval_cap:
                break

        # Selection / descent
        curr_node = root_node
        path: list[MCTSEdge] = []
        visited_nodes: list[BeliefNode] = [root_node]

        while True:
            step_intervals = min(cfg.planning.h, curr_node.H_root - curr_node.tau, cfg.planning.L - curr_node.tau)
            if curr_node.is_terminal or step_intervals <= 0:
                leaf = curr_node
                break

            limit = widening_limit(curr_node.visits, cfg)
            if len(curr_node.candidate_edges) < limit:
                # Progressive widening expands one new edge
                rep = curr_node.select_representative()
                rep_obs = rep.state.cached_world_cloud if hasattr(rep.state, "cached_world_cloud") else rep.state
                sample_seed = (seed + next_sample_idx * 10007) & 0x7FFFFFFF

                candidate = capabilities.sample_prior(rep_obs, rep.task, context, seed=sample_seed)
                native_calls_count += 1

                raw_prefix = capabilities.materialize_prefix(
                    candidate,
                    h=step_intervals,
                    r=min(cfg.planning.r, step_intervals),
                    duration_s=cfg.control.dt0,
                )
                if len(raw_prefix.commands) != step_intervals:
                    prefix = CommandPrefix(
                        commands=raw_prefix.commands[:step_intervals],
                        proposal_root=raw_prefix.proposal_root,
                        raw_candidate_id=raw_prefix.raw_candidate_id,
                    )
                else:
                    prefix = raw_prefix

                # Audit record for candidate
                prefix_bytes = b"".join(_canonical_command_bytes(c) for c in prefix.commands)
                cmd_hash = hashlib.sha256(prefix_bytes).hexdigest()
                is_dup = cmd_hash in seen_command_hashes
                seen_command_hashes.add(cmd_hash)
                cand_id = getattr(candidate, "raw_candidate_id", getattr(candidate, "index", f"cand_{next_sample_idx}"))
                audit_records.append(
                    CandidateAuditRecord(
                        sample_index=next_sample_idx,
                        candidate_id=str(cand_id),
                        is_duplicate=is_dup,
                        commands_hash=cmd_hash,
                    )
                )
                next_sample_idx += 1

                # Step child node across all 3 heads
                child_node = _step_edge(curr_node, prefix, context, capabilities, exact_cache, cfg)
                model_intervals_count += len(prefix.commands)

                new_edge = MCTSEdge(
                    prefix=prefix,
                    child=child_node,
                    visits=0,
                    total_return=0.0,
                    edge_id=next_edge_id,
                )
                next_edge_id += 1
                curr_node.candidate_edges.append(new_edge)

                path.append(new_edge)
                visited_nodes.append(child_node)
                leaf = child_node
                break
            else:
                # Selection among existing edges
                unvisited = [e for e in curr_node.candidate_edges if e.visits == 0]
                if unvisited:
                    # Tie selection by insertion ID
                    selected_edge = min(unvisited, key=lambda e: e.edge_id)
                else:
                    # Q + UCT bonus, tie selection by insertion ID
                    selected_edge = max(
                        curr_node.candidate_edges,
                        key=lambda e: (uct(e.total_return, e.visits, curr_node.visits, cfg), -e.edge_id),
                    )

                path.append(selected_edge)
                visited_nodes.append(selected_edge.child)
                curr_node = selected_edge.child

                if selected_edge.visits == 0:
                    leaf = curr_node
                    break

        # Leaf evaluation
        rem = leaf.H_root - leaf.tau
        if rem == 0 or leaf.is_terminal:
            G = leaf_return(leaf.U, leaf.weights, np.zeros(3, dtype=np.float64), 0)
        else:
            values = np.zeros(3, dtype=np.float64)
            for m in range(3):
                h_m = leaf.hypotheses[m]
                if h_m.weight > 0.0:
                    eval_res = capabilities.evaluate_state(h_m.state, h_m.task, events, rem)
                    cv = eval_res.calibrated_value
                    if isinstance(cv, np.ndarray):
                        values[m] = float(cv.flat[0])
                    else:
                        values[m] = float(cv)
            G = leaf_return(leaf.U, leaf.weights, values, rem)

        # One G backup per visited path; node visit incremented once
        for edge in path:
            edge.total_return += G
            edge.visits += 1
        for node in visited_nodes:
            node.visits += 1

        iteration_count += 1

    # Final root choice: most visits, then greater mean return, then earlier insertion ID
    if not root_node.candidate_edges:
        return PlanningResult(
            selected_prefix=None,
            completed=False,
            fallback_reason="no_eligible_candidate",
            expected_return=None,
            call_count=iteration_count,
            timing_counters={
                "iterations": iteration_count,
                "native_calls": native_calls_count,
                "model_intervals": model_intervals_count,
            },
            cache_counters=exact_cache.counters(),
            audit_ids=tuple(rec.candidate_id for rec in audit_records),
        )

    best_edge = max(
        root_node.candidate_edges,
        key=lambda e: (e.visits, e.q_value, -e.edge_id),
    )

    return PlanningResult(
        selected_prefix=best_edge.prefix,
        completed=True,
        fallback_reason=None,
        expected_return=best_edge.q_value,
        call_count=iteration_count,
        timing_counters={
            "iterations": iteration_count,
            "native_calls": native_calls_count,
            "model_intervals": model_intervals_count,
        },
        cache_counters=exact_cache.counters(),
        audit_ids=tuple(rec.candidate_id for rec in audit_records),
    )


mcts_plan = plan
