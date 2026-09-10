"""Deterministic, privileged P02 annotation rules.

These helpers operate on generator/monitor metadata.  They never materialize
labels into an online observation or infer an unavailable semantic predicate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


def _boundary(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer boundary")
    return value


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class Interval:
    """A half-open interval expressed in executed boundary indices."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _boundary(self.start, "start")
        end = _boundary(self.end, "end")
        if end <= start:
            raise ValueError("interval end must be greater than start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PrimitiveInterval:
    """A privileged primitive occurrence used only for annotation."""

    primitive_id: str
    start: int
    end: int
    semantic_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.primitive_id, str) or not self.primitive_id.strip():
            raise ValueError("primitive_id must be a nonempty string")
        interval = Interval(self.start, self.end)
        object.__setattr__(self, "start", interval.start)
        object.__setattr__(self, "end", interval.end)
        object.__setattr__(self, "semantic_valid", _boolean(self.semantic_valid, "semantic_valid"))

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)


@dataclass(frozen=True)
class EventAnnotation:
    """Event-to-primitive association plus independent semantic validity."""

    primitive_id: str | None
    mapping_valid: bool
    semantic_valid: bool

    def __post_init__(self) -> None:
        mapping_valid = _boolean(self.mapping_valid, "mapping_valid")
        semantic_valid = _boolean(self.semantic_valid, "semantic_valid")
        if not mapping_valid:
            if self.primitive_id is not None or semantic_valid:
                raise ValueError("masked event annotation cannot contain a primitive or semantic label")
        elif not isinstance(self.primitive_id, str) or not self.primitive_id.strip():
            raise ValueError("valid event annotation requires a primitive_id")
        object.__setattr__(self, "mapping_valid", mapping_valid)
        object.__setattr__(self, "semantic_valid", semantic_valid)


@dataclass(frozen=True)
class TaskLabels:
    """Causal task labels with per-target validity masks.

    ``rho`` records an observed historical occurrence, ``nu`` describes the
    current postcondition, and ``eligibility`` describes currently satisfied
    prerequisites.  Masked values are stored as ``False`` and must be consumed
    with their corresponding validity flag.
    """

    rho: bool
    nu: bool
    eligibility: bool
    rho_valid: bool
    nu_valid: bool
    eligibility_valid: bool

    def __post_init__(self) -> None:
        for name in ("rho", "nu", "eligibility", "rho_valid", "nu_valid", "eligibility_valid"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        for value_name, valid_name in (
            ("rho", "rho_valid"),
            ("nu", "nu_valid"),
            ("eligibility", "eligibility_valid"),
        ):
            if not getattr(self, valid_name) and getattr(self, value_name):
                raise ValueError(f"masked {value_name} must use the false placeholder")


def _label(value: bool | None, name: str) -> tuple[bool, bool]:
    if value is None:
        return False, False
    return _boolean(value, name), True


def _overlap(left: Interval, right: Interval) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def map_event_annotation(event: Interval, primitive_intervals: Sequence[PrimitiveInterval]) -> EventAnnotation:
    """Map an event to its largest temporal primitive overlap.

    A candidate must cover at least half of the **event** interval, never half
    of the primitive and never IoU.  An overlap tie selects the earlier
    primitive occurrence; equal starts remain ambiguous and are masked rather
    than resolved by input order.  A reliable free-space mapping may be kept
    while its semantic label is independently masked.
    """

    if not isinstance(event, Interval):
        raise ValueError("event must be an Interval")
    if isinstance(primitive_intervals, (str, bytes)) or not isinstance(primitive_intervals, Sequence):
        raise ValueError("primitive_intervals must be a sequence")

    candidates: list[tuple[int, PrimitiveInterval]] = []
    for index, primitive in enumerate(primitive_intervals):
        if not isinstance(primitive, PrimitiveInterval):
            raise ValueError(f"primitive interval {index} must be a PrimitiveInterval")
        overlap = _overlap(event, primitive.interval)
        if 2 * overlap >= event.duration:
            candidates.append((overlap, primitive))
    if not candidates:
        return EventAnnotation(None, False, False)

    maximum_overlap = max(overlap for overlap, _ in candidates)
    tied = [primitive for overlap, primitive in candidates if overlap == maximum_overlap]
    earliest_start = min(primitive.start for primitive in tied)
    earliest = [primitive for primitive in tied if primitive.start == earliest_start]
    if len(earliest) != 1:
        return EventAnnotation(None, False, False)
    selected = earliest[0]
    return EventAnnotation(selected.primitive_id, True, selected.semantic_valid)


def task_labels(
    *,
    occurred: bool | None,
    current_relation: bool | None,
    eligible: bool | None,
    postcondition_identifiable: bool = True,
) -> TaskLabels:
    """Construct causal occurrence/current/eligibility labels and masks.

    The caller supplies only measured history/current monitor results.  If a
    postcondition is unidentifiable (including free-space motion), only ``nu``
    is masked.  Eligibility remains independently valid when the caller can
    observe the current prerequisites, and uses ``None`` when it cannot.
    """

    rho, rho_valid = _label(occurred, "occurred")
    nu, nu_valid = _label(current_relation, "current_relation")
    eligibility, eligibility_valid = _label(eligible, "eligible")
    if not _boolean(postcondition_identifiable, "postcondition_identifiable"):
        nu, nu_valid = False, False
    return TaskLabels(rho, nu, eligibility, rho_valid, nu_valid, eligibility_valid)


def first_terminal(*, success: bool, failure: bool, absorbed_success: bool) -> Literal["success", "failure", "continue"]:
    """Resolve the proposal's deterministic first-terminal precedence."""

    success = _boolean(success, "success")
    failure = _boolean(failure, "failure")
    absorbed_success = _boolean(absorbed_success, "absorbed_success")
    if absorbed_success:
        return "success"
    if failure:
        return "failure"
    if success:
        return "success"
    return "continue"


__all__ = [
    "EventAnnotation",
    "Interval",
    "PrimitiveInterval",
    "TaskLabels",
    "first_terminal",
    "map_event_annotation",
    "task_labels",
]
