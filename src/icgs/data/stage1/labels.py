"""Derive task-memory labels (alpha, rho, nu, epsilon) from raw predicate traces.

Inputs are per-boundary simulator predicates, never task names or language.
An event spec names the relation predicate whose *current* truth is ``nu`` and
the prerequisites that make it eligible:

``{"event_id": str, "relation": str, "prerequisites": [event_id, ...],
   "current_requirements": [predicate or "!predicate", ...], "hold_steps": int,
   "count_only_when_eligible": bool}``

Label semantics (ID choices recorded in the plan):

* ``nu[t, j]``      relation of event j holds at boundary t (current state).
* ``rho[t, j]``     the relation has held for ``hold_steps`` consecutive
                    boundaries at some t' <= t (causal history; stays true
                    after the relation is later undone).
* ``epsilon[t, j]`` every prerequisite event has occurred (rho) and every
                    current requirement holds at t. It says nothing about
                    execution success.
* ``alpha[t]``      index of the first event (spec order) that has not
                    occurred and is eligible; ``L`` (null) when every event has
                    occurred or no pending event is eligible (unrepresented
                    recovery).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LABEL_PROTOCOL_ID = "icgs-predicate-trace-labels-v1"


def _column(trace: np.ndarray, names: Sequence[str], token: str) -> np.ndarray:
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
        if spec.get("relation") not in predicate_names:
            raise ValueError(f"event {event_id} relation must be a recorded predicate")
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


def derive_event_labels(
    trace: np.ndarray,
    predicate_names: Sequence[str],
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    trace = np.asarray(trace)
    if trace.ndim != 2 or trace.shape[1] != len(predicate_names):
        raise ValueError("trace must have shape [T, P] matching predicate_names")
    validate_event_specs(specs, predicate_names)
    boundaries, events = trace.shape[0], len(specs)
    index = {spec["event_id"]: j for j, spec in enumerate(specs)}
    nu = np.zeros((boundaries, events), dtype=bool)
    rho = np.zeros((boundaries, events), dtype=bool)
    epsilon = np.zeros((boundaries, events), dtype=bool)
    first_occurrence = np.full(events, -1, dtype=np.int64)
    # Specs are ordered so prerequisites precede dependants; eligibility of j
    # only needs rho of earlier events.
    for j, spec in enumerate(specs):
        eligible = np.ones(boundaries, dtype=bool)
        for prerequisite in spec.get("prerequisites", ()):
            eligible &= rho[:, index[prerequisite]]
        for token in spec.get("current_requirements", ()):
            eligible &= _column(trace, predicate_names, token)
        epsilon[:, j] = eligible
        relation = _column(trace, predicate_names, spec["relation"])
        nu[:, j] = relation
        # A relation that is already true at reset (a closed drawer) is not an
        # occurrence; such events only count while eligible.
        gate = eligible if spec.get("count_only_when_eligible", False) else np.ones(boundaries, dtype=bool)
        hold = int(spec.get("hold_steps", 1))
        run = 0
        for t in range(boundaries):
            run = run + 1 if (relation[t] and gate[t]) else 0
            if run >= hold and first_occurrence[j] < 0:
                first_occurrence[j] = t
            rho[t, j] = first_occurrence[j] >= 0
    alpha = np.full(boundaries, events, dtype=np.int64)
    for t in range(boundaries):
        for j in range(events):
            if not rho[t, j] and epsilon[t, j]:
                alpha[t] = j
                break
    return {
        "nu": nu,
        "rho": rho,
        "epsilon": epsilon,
        "alpha": alpha,
        "first_occurrence": first_occurrence,
    }


def label_consistency_report(labels: Mapping[str, np.ndarray], specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Facts a reviewer should see: order, undone relations, ineligible occurrences."""
    rho, nu, epsilon = labels["rho"], labels["nu"], labels["epsilon"]
    first = labels["first_occurrence"]
    report: dict[str, Any] = {"events": []}
    for j, spec in enumerate(specs):
        occurred = int(first[j])
        entry = {
            "event_id": spec["event_id"],
            "first_occurrence": occurred if occurred >= 0 else None,
            "occurred_while_ineligible": bool(occurred >= 0 and not epsilon[occurred, j]),
            "undone_after_occurrence": bool(occurred >= 0 and (rho[:, j] & ~nu[:, j]).any()),
            "final_nu": bool(nu[-1, j]),
        }
        report["events"].append(entry)
    order = [int(first[j]) for j in range(len(specs)) if first[j] >= 0]
    report["occurrences_in_spec_order"] = order == sorted(order)
    return report


__all__ = ["LABEL_PROTOCOL_ID", "derive_event_labels", "label_consistency_report", "validate_event_specs"]
