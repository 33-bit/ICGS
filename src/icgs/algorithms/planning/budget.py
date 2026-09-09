"""Minimal budget and resource counter seam for search planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class PlanningBudget:
    """Resource bounds for online planning."""

    wall_budget_s: float | None = None
    native_call_cap: int | None = None
    model_interval_cap: int | None = None
    clock_track: str = "paused"

    def __post_init__(self) -> None:
        if self.wall_budget_s is not None:
            if isinstance(self.wall_budget_s, bool) or not isinstance(self.wall_budget_s, (int, float)):
                raise TypeError("wall_budget_s must be a numeric float")
            val = float(self.wall_budget_s)
            if not isfinite(val) or val <= 0:
                raise ValueError("wall_budget_s must be finite and positive")
            object.__setattr__(self, "wall_budget_s", val)

        for name in ("native_call_cap", "model_interval_cap"):
            cap = getattr(self, name)
            if cap is not None:
                if isinstance(cap, bool) or not isinstance(cap, int):
                    raise TypeError(f"{name} must be an integer")
                if cap < 0:
                    raise ValueError(f"{name} must be nonnegative")
                object.__setattr__(self, name, int(cap))

        if not isinstance(self.clock_track, str) or not self.clock_track.strip():
            raise ValueError("clock_track must be nonempty string")

        if self.wall_budget_s is None and self.native_call_cap is None:
            raise ValueError(
                "PlanningBudget must specify wall_budget_s or native_call_cap (model_interval_cap alone is insufficient)"
            )

    def check_native_call(self, count: int) -> bool:
        """Return True if another native call is permitted."""
        if self.native_call_cap is not None and count >= self.native_call_cap:
            return False
        return True

    def check_model_interval(self, count: int) -> bool:
        """Return True if another model interval is permitted."""
        if self.model_interval_cap is not None and count >= self.model_interval_cap:
            return False
        return True

    def check_wall_budget(self, elapsed_s: float) -> bool:
        """Return True if elapsed time is within wall budget."""
        if self.wall_budget_s is not None and elapsed_s >= self.wall_budget_s:
            return False
        return True


class BudgetExhausted(Exception):
    """Raised when an operation cannot be performed due to budget exhaustion."""
