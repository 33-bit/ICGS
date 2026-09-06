"""Version1 NPZ: world-frame dense demos plus a live observation; no pickle."""
from pathlib import Path
import numpy as np
from icgs.contracts.records import Observation


def validate_pose(pose, name='pose'):
    if pose.shape[-2:]!=(4,4) or not np.isfinite(pose).all(): raise ValueError(f'invalid {name} shape/values')
    if not np.allclose(pose[...,3,:],[0,0,0,1],atol=1e-6): raise ValueError(f'invalid {name} homogeneous row')
    rotation=pose[...,:3,:3]
    if not np.allclose(rotation.swapaxes(-1,-2)@rotation,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(rotation),1,atol=1e-5):
        raise ValueError(f'invalid {name} rotation')


def load_input(path):
    with np.load(Path(path),allow_pickle=False) as archive:
        expected={'schema_version','demo_points','demo_poses','demo_grips','points','root_pose','grip'}
        if set(archive.files)!=expected: raise ValueError('inference input requires exact version1 fields')
        values={key:archive[key].copy() for key in archive.files}
    if values['schema_version'].shape!=() or values['schema_version'].item()!=1: raise ValueError('unsupported input schema')
    clouds=values['demo_points'];poses=values['demo_poses'];grips=values['demo_grips'];points=values['points']
    if clouds.ndim!=4 or clouds.shape[-1]!=3 or min(clouds.shape[:3])<1: raise ValueError('demo_points must be [D,T,N,3]')
    if poses.shape!=clouds.shape[:2]+(4,4) or grips.shape!=clouds.shape[:2]: raise ValueError('demo pose/grip cardinality mismatch')
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<1: raise ValueError('points must be world XYZ [N,3]')
    if not np.isfinite(clouds).all() or not np.isfinite(points).all(): raise ValueError('point cloud contains non-finite values')
    if not np.isin(grips,[0,1]).all() or values['grip'].shape!=() or values['grip'].item() not in (0,1): raise ValueError('observation grips must be 0 or 1')
    if values['root_pose'].shape!=(4,4):raise ValueError('root pose must be exactly [4,4]')
    validate_pose(poses,'demo pose');validate_pose(values['root_pose'],'root pose')
    demos=[dict(pcds=list(clouds[d]),T_w_es=list(poses[d]),grips=list(grips[d])) for d in range(len(clouds))]
    return Observation(points,values['root_pose'],float(values['grip'])),demos


def save_output(path, trajectory, observation, *, metadata):
    import json
    actions=trajectory.transforms.detach().cpu().numpy()
    grips=trajectory.grips.detach().cpu().numpy()
    validate_pose(actions,'output pose')
    if not np.isfinite(grips).all() or not np.isin(grips,[-1,0,1]).all(): raise ValueError('invalid output gripper commands')
    np.savez_compressed(path,schema_version=np.array(1),actions=actions,grips=grips,
                        root_pose=observation.T_w_e,absolute_targets=observation.T_w_e@actions,
                        metadata=np.array(json.dumps(metadata,sort_keys=True)))
