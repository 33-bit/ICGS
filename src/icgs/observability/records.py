"""Small frozen value objects shared by recorder and inspection code."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EventRecord:
    """Immutable in-memory event envelope used before queue submission."""

    payload: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class CaptureResult:
    captured: bool
    capture_id: str | None = None
    path: str | None = None
    bytes_written: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class RunSummary:
    run_id: str | None
    status: str
    logs_complete: bool
    event_count: int
    metric_count: int
    capture_count: int
    dropped_count: int
    capture_skipped_count: int
    sink_error_count: int
    path: str | None = None
    error: Mapping[str, Any] | None = None


__all__ = ["SCHEMA_VERSION", "EventRecord", "CaptureResult", "RunSummary"]
