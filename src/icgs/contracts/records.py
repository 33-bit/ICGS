"""Public observation/action semantics; no simulator-specific classes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from icgs.state.context_cache import PreparedContext


@dataclass(frozen=True)
class Observation:
    points: Any  # segmented world XYZ [N,3]
    T_w_e: Any  # end-effector-to-world [4,4]
    grip: float  # 0 closed, 1 open


@dataclass(frozen=True)
class ActionTrajectory:
    transforms: Any  # relative to observation pose [B,P,4,4]
    grips: Any  # normalized close/open [-1,1], [B,P,1]


class PredictivePolicy(Protocol):
    def eval(self): ...
    def prepare_context(self, demos: list, *, prepared: bool = False) -> 'PreparedContext': ...
    def reset_context(self, context: 'PreparedContext') -> None: ...
    def predict(self, observation: Observation, context: PreparedContext) -> ActionTrajectory: ...


class EvaluationEnvironment(Protocol):
    def launch(self) -> None: ...
    def collect_demos(self, num_demos: int, num_traj_wp: int) -> list: ...
    def reset(self) -> None: ...
    def observe(self) -> Observation: ...
    def encode_action(self, observation: Observation, trajectory: ActionTrajectory, index: int) -> Any: ...
    def step(self, command: Any) -> tuple[float, bool]: ...
    def shutdown(self) -> None: ...
