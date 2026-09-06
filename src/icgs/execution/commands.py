"""Command targets anchored at the proposal root; time is not inferred from steps."""
from dataclasses import dataclass
import numpy as np
from icgs.data.schemas.inference import validate_pose


@dataclass(frozen=True)
class AbsolutePrefix:
    targets: np.ndarray  # [B,h,4,4] world-frame targets
    commanded_grips: np.ndarray  # [B,h,1], normalized native commands (not achieved grip)
    root_pose: np.ndarray
    command_duration_seconds: float | None = None
    achieved_duration_seconds: float | None = None
    physics_substeps: int | None = None


def absolute_prefix(trajectory, root_pose, length, *, command_duration_seconds=None):
    if isinstance(length,bool) or not isinstance(length,int):raise ValueError('prefix length must be an integer')
    actions=trajectory.transforms.detach().cpu().numpy()
    grips=trajectory.grips.detach().cpu().numpy()
    if actions.ndim!=4 or actions.shape[-2:]!=(4,4) or not 1<=length<=actions.shape[1]:
        raise ValueError('prefix length must be within native prediction horizon')
    if grips.shape!=actions.shape[:2]+(1,):raise ValueError('gripper/trajectory shape mismatch')
    root=np.array(root_pose,copy=True)
    validate_pose(root,'root pose')
    if root.shape!=(4,4):raise ValueError('one proposal root pose required')
    if command_duration_seconds is not None and (not np.isfinite(command_duration_seconds) or command_duration_seconds<=0):
        raise ValueError('known command duration must be positive')
    return AbsolutePrefix(root@actions[:,:length],grips[:,:length].copy(),root,command_duration_seconds)
