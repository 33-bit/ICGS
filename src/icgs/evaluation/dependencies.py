"""External privileged predicate boundary for P02/P12.

Task programs, roles, predicates, and simulator state are annotation metadata.
They are intentionally kept outside online observations and are supplied through
an injected capability instead of a concrete simulator dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol

from icgs.contracts.method import ExecutedTransition


class PredicateCapability(Protocol):
    """Privileged predicate evaluator supplied by an environment/benchmark owner."""

    def annotate(self, transition: ExecutedTransition, *, config: object) -> Mapping[str, Any]: ...


class TaskMonitor:
    """Inject predicate logic and its already-resolved configuration.

    This adapter contains no predicate thresholds or simulator implementation.
    A caller must provide the approved predicate capability and configuration;
    unavailable measurements are represented by that capability's annotation
    masks, not guessed at this boundary.
    """

    def __init__(self, predicates: PredicateCapability, config: object):
        annotate = getattr(predicates, "annotate", None)
        if not callable(annotate):
            raise TypeError("predicates must provide annotate(transition, *, config)")
        if config is None:
            raise ValueError("predicate config must be explicitly injected")
        self._predicates = predicates
        self._config = config

    def annotate(self, transition: ExecutedTransition) -> Mapping[str, Any]:
        """Return privileged annotation metadata separately from online fields."""

        if not isinstance(transition, ExecutedTransition):
            raise ValueError("transition must be an ExecutedTransition")
        annotation = self._predicates.annotate(transition, config=self._config)
        if not isinstance(annotation, Mapping):
            raise ValueError("predicate annotation must be a mapping")
        return MappingProxyType(dict(annotation))


__all__ = ["PredicateCapability", "TaskMonitor"]

