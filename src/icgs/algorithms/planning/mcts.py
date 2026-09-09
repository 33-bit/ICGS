"""Progressive-widening MCTS over common absolute prefixes with exact root-scoped cache."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from math import isfinite
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from icgs.algorithms.planning.belief import (
    BeliefNode,
    Hypothesis,
    leaf_return,
    propagate_mass,
    select_representative,
)
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
    validate_predicted_state,
    validate_root_and_context,
    validate_terminal_probabilities,
)
from icgs.configuration.method import MethodConfig, PlanningConfig
from icgs.contracts.method import CommandPrefix, MethodCapabilities, PlanningResult, TimedCommand
from icgs.contracts.records import Observation
from icgs.observability.recorder import NoopRecorder
from icgs.state.physical import PhysicalState


def _extract_planning_config(cfg: MethodConfig | PlanningConfig) -> PlanningConfig:
    """Extract PlanningConfig without duck typing or invented fallbacks."""
    if isinstance(cfg, MethodConfig):
        return cfg.planning
    if isinstance(cfg, PlanningConfig):
        return cfg
    raise TypeError(f"cfg must be MethodConfig or PlanningConfig, got {type(cfg)}")


def widening_limit(
    visits: int,
    cfg: MethodConfig | PlanningConfig,
) -> int:
    """Calculate progressive widening candidate limit for a node."""
    p_cfg = _extract_planning_config(cfg)

    if isinstance(visits, bool) or not isinstance(visits, (int, np.integer)):
        raise TypeError("visits must be an integer")
    if visits < 0:
        raise ValueError("visits must be nonnegative")
    visits = int(visits)

    coeff = float(p_cfg.widening_coefficient)
    exponent = float(p_cfg.widening_exponent)
    limit = max(1, math.floor(coeff * ((1 + visits) ** exponent)))
    return int(limit)


def uct(
    total_return: float,
    visits: int,
    parent_visits: int,
    cfg: MethodConfig | PlanningConfig,
) -> float:
    """Calculate UCT selection value for a visited edge."""
    p_cfg = _extract_planning_config(cfg)

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

    c = float(p_cfg.uct_exploration)
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


def _physical_state_identity(state: PhysicalState) -> bytes:
    """Compute complete cryptographic digest covering all owned physical state content."""
    h = hashlib.sha256()
    h.update(state.origin.encode("utf-8"))
    h.update(state.boundary.to_bytes(8, "big", signed=False))
    h.update(state.encoder_lineage.encode("utf-8"))
    h.update(state.memory_lineage.encode("utf-8"))
    h.update(state.X.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.x.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.valid.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.p.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.memory.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.T_w_e.detach().cpu().contiguous().numpy().tobytes())
    h.update(state.grip.detach().cpu().contiguous().numpy().tobytes())
    if state.cached_world_cloud is not None and state.cached_world_cloud_valid is not None:
        h.update(b"cloud:")
        h.update(state.cached_world_cloud.detach().cpu().contiguous().numpy().tobytes())
        h.update(state.cached_world_cloud_valid.detach().cpu().contiguous().numpy().tobytes())
    else:
        h.update(b"no_cloud")
    return h.digest()


@dataclass(frozen=True)
class CompositeModelIdentity:
    """Explicit composite IDs for world model, task tracker, and terminal evaluator."""

    world_model_id: str
    task_tracker_id: str
    terminal_id: str

    def __post_init__(self) -> None:
        for name in ("world_model_id", "task_tracker_id", "terminal_id"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{name} must be a nonempty string")


def _extract_model_identities(capabilities: Any) -> CompositeModelIdentity:
    """Extract explicit composite model identity from capability seam without silent default fallback."""
    if hasattr(capabilities, "model_identities"):
        m = capabilities.model_identities
        if isinstance(m, CompositeModelIdentity):
            return m
        if isinstance(m, dict):
            return CompositeModelIdentity(**m)
    if hasattr(capabilities, "get_model_identities") and callable(capabilities.get_model_identities):
        m = capabilities.get_model_identities()
        if isinstance(m, CompositeModelIdentity):
            return m
        if isinstance(m, dict):
            return CompositeModelIdentity(**m)

    w_id = getattr(capabilities, "world_model_id", getattr(capabilities, "model_id", None))
    t_id = getattr(capabilities, "task_tracker_id", None)
    term_id = getattr(capabilities, "terminal_id", None)
    if w_id is not None and t_id is not None and term_id is not None:
        return CompositeModelIdentity(world_model_id=str(w_id), task_tracker_id=str(t_id), terminal_id=str(term_id))

    raise ValueError(
        "capabilities must provide explicit CompositeModelIdentity (world_model_id, task_tracker_id, terminal_id)"
    )


def _extract_context_identity(context: Any, capabilities: Any) -> str:
    """Extract explicit context identity without lossy fallback to reference_id or id()."""
    if hasattr(capabilities, "context_identity") and callable(capabilities.context_identity):
        cid = capabilities.context_identity(context)
        if isinstance(cid, str) and cid.strip():
            return cid
    if hasattr(context, "context_identity"):
        cid = context.context_identity() if callable(context.context_identity) else context.context_identity
        if isinstance(cid, str) and cid.strip():
            return cid
    if hasattr(context, "identity"):
        cid = context.identity() if callable(context.identity) else context.identity
        if isinstance(cid, str) and cid.strip():
            return cid
    if hasattr(context, "context_id"):
        cid = context.context_id
        if isinstance(cid, str) and cid.strip():
            return cid
    if isinstance(context, dict) and "context_id" in context:
        cid = context["context_id"]
        if isinstance(cid, str) and cid.strip():
            return cid
    if isinstance(context, str) and context.strip():
        return context

    raise ValueError("context must provide an explicit identity (e.g. context_id or capabilities.context_identity)")


def _extract_task_identity(task: Any, capabilities: Any) -> str:
    """Extract explicit task history identity without lossy repr/str fallback."""
    if hasattr(capabilities, "task_identity") and callable(capabilities.task_identity):
        tid = capabilities.task_identity(task)
        if isinstance(tid, str) and tid.strip():
            return tid
    if hasattr(task, "task_identity"):
        tid = task.task_identity() if callable(task.task_identity) else task.task_identity
        if isinstance(tid, str) and tid.strip():
            return tid
    if hasattr(task, "history_id"):
        tid = task.history_id
        if isinstance(tid, str) and tid.strip():
            return tid
    if isinstance(task, dict) and "history_id" in task:
        tid = task["history_id"]
        if isinstance(tid, str) and tid.strip():
            return tid
    if isinstance(task, str) and task.strip():
        return task

    raise ValueError("task must provide an explicit history identity (e.g. history_id or capabilities.task_identity)")


def _branch_copy_task(task: Any, capabilities: Any = None) -> Any:
    """Produce an isolated branch copy of task state."""
    if capabilities is not None and hasattr(capabilities, "branch_copy_task") and callable(capabilities.branch_copy_task):
        return capabilities.branch_copy_task(task)
    if hasattr(task, "branch_copy") and callable(task.branch_copy):
        return task.branch_copy()
    if isinstance(task, dict):
        return copy.deepcopy(task)
    if hasattr(task, "__dict__"):
        return copy.deepcopy(task)
    return copy.deepcopy(task)


def _canonical_command_bytes(cmd: TimedCommand) -> bytes:
    """Serialize TimedCommand into canonical byte representation."""
    pose_bytes = np.asarray(cmd.target_w, dtype=np.float64).tobytes()
    grip_bytes = int(cmd.grip).to_bytes(1, "big")
    dur_bytes = float(cmd.duration_s).hex().encode("ascii")
    return pose_bytes + grip_bytes + dur_bytes


def _validate_scalar_probability(val: Any, name: str) -> float:
    """Validate that val is a single finite scalar within [0, 1], rejecting multi-element arrays/tensors."""
    if isinstance(val, (int, float, np.floating, np.integer)):
        f_val = float(val)
    elif isinstance(val, np.ndarray):
        if val.size != 1:
            raise ValueError(f"{name} must be a single scalar, got array of shape {val.shape}")
        f_val = float(val.item())
    elif hasattr(val, "numel") and callable(val.numel):
        if val.numel() != 1:
            raise ValueError(f"{name} must be a single scalar, got tensor of shape {tuple(val.shape)}")
        f_val = float(val.item())
    else:
        try:
            arr = np.asarray(val)
            if arr.size != 1:
                raise ValueError(f"{name} must be a single scalar, got shape {arr.shape}")
            f_val = float(arr.item())
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} cannot be converted to scalar: {e}") from e

    if not math.isfinite(f_val):
        raise ValueError(f"{name} must be finite, got {f_val}")
    if f_val < 0.0 or f_val > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {f_val}")
    return f_val


@dataclass(frozen=True)
class CacheKey:
    """Exact transition cache key requiring full state digest, lineage, and canonical command bytes."""

    parent_state_digest: bytes
    task_identity: str
    context_identity: str
    head_id: int
    composite_models: CompositeModelIdentity
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

    def get(self, key: CacheKey, capabilities: Any) -> tuple[PhysicalState, Any, np.ndarray] | None:
        """Lookup cached transition, returning branch-owned copies to avoid aliasing."""
        self.lookups += 1
        if key in self._cache:
            self.hits += 1
            st, tk, probs = self._cache[key]
            return (st.branch_copy(), _branch_copy_task(tk, capabilities), np.array(probs, copy=True))
        self.misses += 1
        return None

    def put(
        self,
        key: CacheKey,
        transition: tuple[PhysicalState, Any, np.ndarray],
        capabilities: Any = None,
    ) -> None:
        """Store transition in cache with branch isolation for all task representations."""
        st, tk, probs = transition
        self._cache[key] = (
            st.branch_copy(),
            _branch_copy_task(tk, capabilities),
            np.array(probs, copy=True),
        )

    def counters(self) -> dict[str, int]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass(frozen=True)
class CandidateAuditRecord:
    """Audit entry for each sampled candidate proposal."""

    candidate_id: str
    order: int
    is_duplicate: bool
    commands_hash: str


def _observation_from_state(state: PhysicalState) -> Observation:
    """Extract validated native Observation from PhysicalState without fabricating geometry."""
    if not isinstance(state, PhysicalState):
        raise TypeError(f"state must be PhysicalState, got {type(state)}")

    if state.cached_world_cloud is not None and state.cached_world_cloud_valid is not None:
        cloud_t = state.cached_world_cloud[0]
        valid_mask_t = state.cached_world_cloud_valid[0]
        points_t = cloud_t[valid_mask_t]
    else:
        cloud_t = state.x[0]
        valid_mask_t = state.valid[0]
        points_t = cloud_t[valid_mask_t]

    points_np = np.ascontiguousarray(points_t.detach().cpu().numpy())
    if points_np.ndim != 2 or points_np.shape[1] != 3 or points_np.shape[0] == 0:
        raise ValueError(f"Observation points must be non-empty (N, 3) array, got shape {points_np.shape}")
    if not np.all(np.isfinite(points_np)):
        raise ValueError("Observation points must contain finite values")

    pose_np = np.ascontiguousarray(state.T_w_e[0].detach().cpu().numpy())
    if pose_np.shape != (4, 4):
        raise ValueError(f"Observation pose must have shape (4, 4), got {pose_np.shape}")
    if not np.all(np.isfinite(pose_np)):
        raise ValueError("Observation pose must contain finite values")

    grip_raw = state.grip[0].item() if hasattr(state.grip[0], "item") else float(state.grip[0])
    grip_val = float(grip_raw)
    if not math.isfinite(grip_val):
        raise ValueError("Observation grip must be finite float")

    return Observation(
        points=points_np,
        T_w_e=pose_np,
        grip=grip_val,
    )


def _step_edge(
    node: BeliefNode,
    prefix: CommandPrefix,
    context: Any,
    capabilities: MethodCapabilities,
    cache: ExactCache,
    cfg: MethodConfig | PlanningConfig,
    budget: PlanningBudget,
    clock_fn: Callable[[], float],
    start_time: float,
    model_intervals_tracker: list[int] | dict[str, int],
    composite_models: CompositeModelIdentity,
    context_id: str,
    deadline: float | None = None,
    recorder: Any = None,
) -> BeliefNode:
    """Step all 3 heads forward across intervals with per-operation budget checks and branch isolation."""
    timing_counters = model_intervals_tracker
    curr_tau = node.tau
    curr_U = node.U
    curr_F = node.F
    curr_weights = list(node.weights)
    curr_hypotheses = list(node.hypotheses)
    events = getattr(context, "events", context)
    effective_deadline = (
        deadline
        if deadline is not None
        else (start_time + budget.wall_budget_s if budget.wall_budget_s is not None else float("inf"))
    )

    for step_idx, cmd in enumerate(prefix.commands):
        step_tau = curr_tau + step_idx
        cmd_bytes = _canonical_command_bytes(cmd)
        next_states: list[PhysicalState] = []
        next_tasks: list[Any] = []
        event_probs = np.zeros((3, 3), dtype=np.float64)

        for m in range(3):
            h_m = curr_hypotheses[m]
            state_digest = _physical_state_identity(h_m.state)
            task_id = _extract_task_identity(h_m.task, capabilities)
            key = CacheKey(
                parent_state_digest=state_digest,
                task_identity=task_id,
                context_identity=context_id,
                head_id=m,
                composite_models=composite_models,
                tau=step_tau,
                command_bytes=cmd_bytes,
            )

            cached = cache.get(key, capabilities)
            if cached is not None:
                next_st, next_tk, probs = cached
            else:
                curr_model_count = (
                    timing_counters["model_intervals"]
                    if isinstance(timing_counters, dict)
                    else timing_counters[0]
                )
                if (
                    clock_fn() >= effective_deadline
                    or not budget.check_model_interval(curr_model_count)
                    or not budget.check_wall_budget(clock_fn() - start_time)
                ):
                    raise BudgetExhausted("Model interval cap or wall budget reached during edge stepping")

                counters_dict = timing_counters if isinstance(timing_counters, dict) else None

                def _predict_op():
                    p = capabilities.predict_step(h_m.state, cmd, head_id=m)
                    validate_predicted_state(p.next_state)
                    return p

                pred, _, eligible_pred = timed_operation(
                    "predict_step",
                    _predict_op,
                    clock_fn=clock_fn,
                    deadline=effective_deadline,
                    budget=budget,
                    capabilities=capabilities,
                    recorder=recorder,
                    coverage="inclusive",
                    timing_counters=counters_dict,
                    counter_key="model_intervals",
                )
                if isinstance(timing_counters, list) and eligible_pred:
                    timing_counters[0] += 1
                if not eligible_pred or pred is None:
                    raise BudgetExhausted("Operation overshot deadline during predict_step")

                next_st = pred.next_state

                next_tk, _, eligible_tk = timed_operation(
                    "track_task",
                    lambda: capabilities.track_task(h_m.task, next_st, events),
                    clock_fn=clock_fn,
                    deadline=effective_deadline,
                    budget=budget,
                    capabilities=capabilities,
                    recorder=recorder,
                    coverage="inclusive",
                    timing_counters=counters_dict,
                    counter_key="task_tracker_calls",
                )
                if not eligible_tk or next_tk is None:
                    raise BudgetExhausted("Operation overshot deadline during track_task")

                def _term_op():
                    t = capabilities.predict_terminal(h_m.state, h_m.task, next_st, next_tk, events, cmd)
                    return validate_terminal_probabilities(t)

                term, _, eligible_term = timed_operation(
                    "predict_terminal",
                    _term_op,
                    clock_fn=clock_fn,
                    deadline=effective_deadline,
                    budget=budget,
                    capabilities=capabilities,
                    recorder=recorder,
                    coverage="inclusive",
                    timing_counters=counters_dict,
                    counter_key="terminal_calls",
                )
                if not eligible_term or term is None:
                    raise BudgetExhausted("Operation overshot deadline during predict_terminal")

                probs = term
                if probs.ndim == 2:
                    probs = probs[0]
                cache.put(key, (next_st, next_tk, probs), capabilities)

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
    budget: PlanningBudget,
    capabilities: MethodCapabilities,
    cfg: MethodConfig,
    seed: int = 0,
    clock: Callable[[], float] | None = None,
    recorder: Any = None,
) -> PlanningResult:
    """Execute progressive-widening MCTS over common absolute prefixes with root-scoped exact cache."""
    if budget is None or not isinstance(budget, PlanningBudget):
        raise ValueError("Explicit bounded PlanningBudget must be provided to plan")
    if capabilities is None:
        raise ValueError("capabilities must be provided to execute MCTS plan")
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
    rep_obs = validate_root_and_context(state, context, capabilities)

    # Provenance audit
    audit_planning_start(
        rec,
        "mcts",
        cfg,
        budget,
        H=H,
        boundary=state.boundary,
        deadline=deadline,
        algorithm_params={"uct_exploration": float(cfg.planning.uct_exploration), "h": cfg.planning.h},
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
    events = getattr(context, "events", context)

    f_root = 0.0
    w_root = (1.0 - u_root) / 3.0

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

    # Cache is strictly scoped to this one root
    root_cache_id = f"root_{state.boundary}_{time.monotonic_ns()}_{seed}"
    exact_cache = ExactCache(root_cache_id)

    # Accounting and audit tracking
    audit_records: list[CandidateAuditRecord] = []
    seen_command_hashes: set[str] = set()
    earliest_candidate: Any | None = None
    next_edge_id = 0
    next_sample_idx = 0
    nonfinite_error = False

    while True:
        # Check all budget bounds before iteration (no start at deadline)
        if clock_fn() >= deadline:
            break
        if not budget.check_wall_budget(clock_fn() - start_time):
            break
        if not budget.check_native_call(timing_counters["native_calls"]):
            break
        if not budget.check_model_interval(timing_counters["model_intervals"]):
            break

        # Selection / descent
        curr_node = root_node
        path: list[MCTSEdge] = []
        visited_nodes: list[BeliefNode] = [root_node]
        iteration_aborted = False

        while True:
            step_intervals = min(p_cfg.h, curr_node.H_root - curr_node.tau, p_cfg.L - curr_node.tau)
            if curr_node.is_terminal or step_intervals <= 0:
                leaf = curr_node
                break

            limit = widening_limit(curr_node.visits, cfg)
            can_expand = (
                len(curr_node.candidate_edges) < limit
                and budget.check_native_call(timing_counters["native_calls"])
                and clock_fn() < deadline
                and budget.check_wall_budget(clock_fn() - start_time)
            )

            if can_expand:
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
                except NonfiniteModelError as exc:
                    nonfinite_error = True
                    rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
                    iteration_aborted = True
                    break

                if candidate is not None and earliest_candidate is None:
                    earliest_candidate = candidate

                if not eligible_cand or candidate is None:
                    iteration_aborted = True
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
                except NonfiniteModelError as exc:
                    nonfinite_error = True
                    rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
                    iteration_aborted = True
                    break

                if not eligible_mat or prefix is None:
                    iteration_aborted = True
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

                # Step edge with operation-level checks
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
                    iteration_aborted = True
                    break
                except NonfiniteModelError as exc:
                    nonfinite_error = True
                    rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
                    iteration_aborted = True
                    break

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
                if not curr_node.candidate_edges:
                    iteration_aborted = True
                    break

                # Selection among existing edges
                unvisited = [e for e in curr_node.candidate_edges if e.visits == 0]
                if unvisited:
                    selected_edge = min(unvisited, key=lambda e: e.edge_id)
                else:
                    selected_edge = max(
                        curr_node.candidate_edges,
                        key=lambda e: (uct(e.total_return, e.visits, curr_node.visits, cfg), -e.edge_id),
                    )

                rec.event(
                    "mcts.selection",
                    fields={"parent_tau": curr_node.tau, "selected_edge_id": selected_edge.edge_id},
                )
                path.append(selected_edge)
                visited_nodes.append(selected_edge.child)
                curr_node = selected_edge.child

                if selected_edge.visits == 0:
                    leaf = curr_node
                    break

        if iteration_aborted:
            break

        # Leaf evaluation
        rem = leaf.H_root - leaf.tau
        try:
            G, eligible_leaf = score_leaf_rollout(
                leaf,
                events,
                remaining_H=rem,
                clock_fn=clock_fn,
                deadline=deadline,
                budget=budget,
                capabilities=capabilities,
                recorder=rec,
                timing_counters=timing_counters,
            )
        except NonfiniteModelError as exc:
            nonfinite_error = True
            rec.event("model.error", fields={"error": "nonfinite_model_output", "detail": str(exc)})
            break
        if not eligible_leaf or G is None:
            break

        finished_iter = clock_fn()
        if not eligible_completion(finished_iter, deadline):
            overshoot = max(0.0, finished_iter - deadline)
            rec.event(
                "operation.overshoot",
                fields={
                    "operation": "mcts.iteration",
                    "finished": finished_iter,
                    "deadline": deadline,
                    "overshoot": overshoot,
                },
            )
            break

        # One G backup per visited path; node visit incremented once
        for edge in path:
            edge.total_return += G
            edge.visits += 1
            rec.event(
                "mcts.edge_backup",
                fields={
                    "edge_id": edge.edge_id,
                    "visits": edge.visits,
                    "total_return": edge.total_return,
                    "q_value": edge.q_value,
                },
            )
        for node in visited_nodes:
            node.visits += 1
            rec.event("mcts.node_visit", fields={"tau": node.tau, "visits": node.visits})

        timing_counters["iterations"] += 1
        rec.event(
            "mcts.iteration",
            fields={
                "iteration": timing_counters["iterations"],
                "leaf_tau": leaf.tau,
                "G": G,
                "path": [e.edge_id for e in path],
            },
        )

    # Final root choice: most visits, then greater mean return, then earlier insertion ID
    eligible_edges = [e for e in root_node.candidate_edges if e.visits > 0]
    if not eligible_edges:
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

    best_edge = max(
        eligible_edges,
        key=lambda e: (e.visits, e.q_value, -e.edge_id),
    )
    rec.event(
        "mcts.root_choice",
        fields={
            "selected_edge_id": best_edge.edge_id,
            "visits": best_edge.visits,
            "q_value": best_edge.q_value,
        },
    )

    return PlanningResult(
        selected_prefix=best_edge.prefix,
        completed=True,
        fallback_reason=None,
        expected_return=best_edge.q_value,
        call_count=timing_counters["iterations"],
        timing_counters=dict(timing_counters),
        cache_counters=exact_cache.counters(),
        audit_ids=tuple(r.candidate_id for r in audit_records),
    )

