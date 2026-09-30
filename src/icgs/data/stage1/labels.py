"""Task-monitor states and context-conditioned task labels from raw predicate traces.

Inputs are per-boundary simulator predicates, never task names or language.
An event spec names the relation whose *current* truth is ``nu`` and what
makes the event eligible:

``{"event_id": str, "relation": "pred" | "pred&!other", "prerequisites": [event_id, ...],
   "current_requirements": [predicate or "!predicate", ...], "hold_steps": int,
   "count_only_when_eligible": bool, "observable": bool}``

Two layers, deliberately separate:

1. ``derive_monitor_states`` — per episode, in that episode's own event list:

   * ``nu[t, j]``       relation of event j holds at boundary t (current state).
   * ``rho[t, j]``      the relation held for ``hold_steps`` boundaries at some
                        t' <= t (causal history; stays true when later undone).
   * ``epsilon[t, j]``  prerequisites occurred and current requirements hold at
                        t (prerequisites only, not execution success).
   * ``event_id[t]``    index of the first eligible not-yet-occurred event in
                        spec order, ``L`` when none. This is a *monitor
                        diagnostic*, not the ICGS alignment ``alpha``.
   * ``*_valid``        False where a label is not observable (``observable:
                        false`` events such as free-space motions get no
                        fabricated postcondition), within ``margin`` boundaries
                        of a relation change (sensor/contact transitions are
                        uncertain), and — for ``event_id`` — where several
                        pending eligible events are unordered by prerequisites.

2. ``context_alignment`` — the ICGS alignment target is context-dependent: a
   query prefix is aligned to the ordered event tokens of an *independent*
   demonstration context of the same task (see ``icgs.data.stage1.views``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LABEL_PROTOCOL_ID = "icgs-predicate-trace-monitor-v2"
DEFAULT_MARGIN = 2  # boundaries = one 0.1 s model interval at the 0.05 s physics step


def _column(trace: np.ndarray, names: Sequence[str], token: str) -> np.ndarray:
    if "&" in token:  # conjunction of raw predicates, e.g. "in_region&!grasped"
        result = np.ones(trace.shape[0], dtype=bool)
        for part in (part.strip() for part in token.split("&")):
            result &= _column(trace, names, part)
        return result
    negate = token.startswith("!")
    name = token[1:] if negate else token
    if name not in names:
        raise KeyError(f"unknown predicate {name!r}")
    values = trace[:, list(names).index(name)].astype(bool)
    return ~values if negate else values


def validate_event_specs(specs: Sequence[Mapping[str, Any]], predicate_names: Sequence[str]) -> None:
    seen: list[str] = []
    for spec in specs:
        event_id = spec.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError(f"invalid or duplicate event_id {event_id!r}")
        relation = spec.get("relation")
        if not isinstance(relation, str) or any(part.strip().lstrip("!") not in predicate_names
                                                 for part in relation.split("&")):
            raise ValueError(f"event {event_id} relation must use recorded predicates")
        for prerequisite in spec.get("prerequisites", ()):
            if prerequisite not in seen:
                raise ValueError(f"event {event_id} prerequisite {prerequisite!r} must precede it")
        for token in spec.get("current_requirements", ()):
            if token.lstrip("!") not in predicate_names:
                raise ValueError(f"event {event_id} requirement {token!r} is not a recorded predicate")
        hold = spec.get("hold_steps", 1)
        if isinstance(hold, bool) or not isinstance(hold, int) or hold < 1:
            raise ValueError(f"event {event_id} hold_steps must be a positive integer")
        seen.append(event_id)


def _near_change(values: np.ndarray, margin: int) -> np.ndarray:
    """Boundaries within ``margin`` of a change of a boolean or integer series."""
    near = np.zeros(values.shape[0], dtype=bool)
    if margin <= 0 or values.shape[0] < 2:
        return near
    for edge in np.flatnonzero(values[1:] != values[:-1]) + 1:
        near[max(edge - margin, 0):edge + margin] = True
    return near


def _ancestors(specs: Sequence[Mapping[str, Any]]) -> list[set[int]]:
    index = {spec["event_id"]: j for j, spec in enumerate(specs)}
    result: list[set[int]] = []
    for spec in specs:
        ancestors: set[int] = set()
        for prerequisite in spec.get("prerequisites", ()):
            k = index[prerequisite]
            ancestors |= {k} | result[k]
        result.append(ancestors)
    return result


def derive_monitor_states(
    trace: np.ndarray,
    predicate_names: Sequence[str],
    specs: Sequence[Mapping[str, Any]],
    *,
    margin: int = DEFAULT_MARGIN,
) -> dict[str, np.ndarray]:
    trace = np.asarray(trace)
    if trace.ndim != 2 or trace.shape[1] != len(predicate_names):
        raise ValueError("trace must have shape [T, P] matching predicate_names")
    validate_event_specs(specs, predicate_names)
    boundaries, events = trace.shape[0], len(specs)
    index = {spec["event_id"]: j for j, spec in enumerate(specs)}
    observable = np.array([bool(spec.get("observable", True)) for spec in specs], dtype=bool)
    nu = np.zeros((boundaries, events), dtype=bool)
    rho = np.zeros((boundaries, events), dtype=bool)
    epsilon = np.zeros((boundaries, events), dtype=bool)
    nu_valid = np.zeros((boundaries, events), dtype=bool)
    rho_valid = np.zeros((boundaries, events), dtype=bool)
    epsilon_valid = np.zeros((boundaries, events), dtype=bool)
    first_occurrence = np.full(events, -1, dtype=np.int64)
    for j, spec in enumerate(specs):
        eligible = np.ones(boundaries, dtype=bool)
        eligible_known = np.ones(boundaries, dtype=bool)
        for prerequisite in spec.get("prerequisites", ()):
            eligible &= rho[:, index[prerequisite]]
            eligible_known &= rho_valid[:, index[prerequisite]]
        for token in spec.get("current_requirements", ()):
            requirement = _column(trace, predicate_names, token)
            eligible &= requirement
            eligible_known &= ~_near_change(requirement, margin)
        epsilon[:, j] = eligible
        epsilon_valid[:, j] = eligible_known
        if not observable[j]:
            continue  # no fabricated postcondition: nu/rho stay invalid
        relation = _column(trace, predicate_names, spec["relation"])
        nu[:, j] = relation
        nu_valid[:, j] = ~_near_change(relation, margin)
        # A relation already true at reset (a closed drawer) is not an
        # occurrence; such events only count while eligible.
        gate = eligible if spec.get("count_only_when_eligible", False) else np.ones(boundaries, dtype=bool)
        hold = int(spec.get("hold_steps", 1))
        run = 0
        for t in range(boundaries):
            run = run + 1 if (relation[t] and gate[t]) else 0
            if run >= hold and first_occurrence[j] < 0:
                first_occurrence[j] = t
            rho[t, j] = first_occurrence[j] >= 0
        rho_valid[:, j] = ~_near_change(rho[:, j], margin)
    ancestors = _ancestors(specs)
    event_id = np.full(boundaries, events, dtype=np.int64)
    event_id_valid = np.ones(boundaries, dtype=bool)
    for t in range(boundaries):
        pending = [j for j in range(events) if not rho[t, j] and epsilon[t, j]]
        if pending:
            event_id[t] = pending[0]
            # Unordered pending events: the monitor alone cannot say which is next.
            if any(pending[0] not in ancestors[j] for j in pending[1:]):
                event_id_valid[t] = False
        if event_id[t] < events and not observable[event_id[t]]:
            event_id_valid[t] = False
    event_id_valid &= ~_near_change(event_id, margin)
    return {
        "event_id": event_id, "event_id_valid": event_id_valid,
        "nu": nu, "nu_valid": nu_valid,
        "rho": rho, "rho_valid": rho_valid,
        "epsilon": epsilon, "epsilon_valid": epsilon_valid,
        "first_occurrence": first_occurrence,
    }


def context_event_tokens(context_first_occurrence: Mapping[str, int]) -> list[tuple[str, int]]:
    """Ordered (event_id, context boundary) tokens of events that occurred in the context."""
    occurred = [(event, int(boundary)) for event, boundary in context_first_occurrence.items() if boundary >= 0]
    return sorted(occurred, key=lambda item: (item[1], item[0]))


def context_alignment(
    query: Mapping[str, np.ndarray],
    query_event_ids: Sequence[str],
    tokens: Sequence[tuple[str, int]],
    boundaries: Sequence[int],
) -> dict[str, np.ndarray]:
    """Alignment of query prefixes to context event tokens.

    Target at boundary t: the first context token whose event has not yet
    occurred in the query; ``len(tokens)`` (null) when all have occurred.
    Invalid where any token's ``rho`` is invalid, or the query achieved a
    later token before an earlier one (order not represented by the context).
    rho/nu/epsilon and their masks are the query monitor states re-indexed to
    the context tokens.
    """
    index = {event: j for j, event in enumerate(query_event_ids)}
    missing = [event for event, _ in tokens if event not in index]
    if missing:
        raise ValueError(f"context events absent from the query monitor: {missing}")
    columns = [index[event] for event, _ in tokens]
    rows = np.asarray(list(boundaries), dtype=np.int64)
    rho = query["rho"][rows][:, columns]
    rho_valid = query["rho_valid"][rows][:, columns]
    target = np.full(rows.shape[0], len(tokens), dtype=np.int64)
    valid = np.ones(rows.shape[0], dtype=bool)
    for r in range(rows.shape[0]):
        pending = np.flatnonzero(~rho[r])
        if pending.size:
            target[r] = int(pending[0])
            if rho[r, pending[0] + 1:].any():  # a later token occurred before the target
                valid[r] = False
        if not rho_valid[r].all():
            valid[r] = False
    return {
        "alignment_target": target, "alignment_valid": valid,
        "rho": rho, "rho_valid": rho_valid,
        "nu": query["nu"][rows][:, columns], "nu_valid": query["nu_valid"][rows][:, columns],
        "epsilon": query["epsilon"][rows][:, columns],
        "epsilon_valid": query["epsilon_valid"][rows][:, columns],
    }


def label_consistency_report(states: Mapping[str, np.ndarray], specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Facts a reviewer should see: order, undone relations, ineligible occurrences."""
    rho, nu, epsilon = states["rho"], states["nu"], states["epsilon"]
    first = states["first_occurrence"]
    report: dict[str, Any] = {"events": []}
    for j, spec in enumerate(specs):
        occurred = int(first[j])
        report["events"].append({
            "event_id": spec["event_id"],
            "first_occurrence": occurred if occurred >= 0 else None,
            "occurred_while_ineligible": bool(occurred >= 0 and not epsilon[occurred, j]),
            "undone_after_occurrence": bool(occurred >= 0 and (rho[:, j] & ~nu[:, j]).any()),
            "final_nu": bool(nu[-1, j]),
        })
    order = [int(first[j]) for j in range(len(specs)) if first[j] >= 0]
    report["occurrences_in_spec_order"] = order == sorted(order)
    if "event_id_valid" in states:
        report["monitor_event_id_valid_fraction"] = float(np.mean(states["event_id_valid"]))
    return report


__all__ = ["DEFAULT_MARGIN", "LABEL_PROTOCOL_ID", "context_alignment", "context_event_tokens",
           "derive_monitor_states", "label_consistency_report", "validate_event_specs"]
