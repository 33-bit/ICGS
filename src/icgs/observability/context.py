"""Immutable correlation context carried by :mod:`contextvars`."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator


@dataclass(frozen=True)
class TraceContext:
    run_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    episode_id: str | None = None
    decision_id: str | None = None
    candidate_id: str | None = None
    branch_id: str | None = None
    transition_id: str | None = None
    boundary_index: int | None = None
    head_id: int | str | None = None
    origin: str | None = None

    def update(self, **values: Any) -> "TraceContext":
        allowed = {item.name for item in fields(self)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown trace context fields: {sorted(unknown)}")
        return replace(self, **values)

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


_CURRENT = ContextVar("icgs_observability_context", default=TraceContext())


def current_context() -> TraceContext:
    """Return the current immutable context for this execution context."""
    return _CURRENT.get()


@contextmanager
def bind_context(**values: Any) -> Iterator[TraceContext]:
    """Temporarily inherit and override correlation identifiers."""
    token = _CURRENT.set(current_context().update(**values))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


__all__ = ["TraceContext", "bind_context", "current_context"]
