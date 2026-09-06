"""Deterministic geometric smoke input; not recorded robot data or task success."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
destination=Path(args.output)
destination.parent.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20260906)
cloud=np.concatenate([rng.normal(center,scale,size=(900,3)) for center,scale in
                      [([.45,.04,.12],[.025,.035,.03]),([.58,-.07,.09],[.03,.025,.02]),([.50,.12,.18],[.025,.025,.04])]])
poses=np.tile(np.eye(4),(2,10,1,1))
for d in range(2):
    for t in range(10):
        poses[d,t,:3,:3]=Rotation.from_euler('xyz',[.12+.01*d,-.08,.2+.025*t]).as_matrix()
        poses[d,t,:3,3]=[.38+.012*t,.025*d,.20-.006*t]
live=np.eye(4);live[:3,:3]=Rotation.from_euler('xyz',[.14,-.07,.26]).as_matrix();live[:3,3]=[.43,.01,.18]
grips=np.ones((2,10));grips[:,5:]=0
points=np.broadcast_to(cloud,(2,10,*cloud.shape)).copy()
np.savez_compressed(destination,demo_points=points,demo_poses=poses,demo_grips=grips,
                    points=cloud,root_pose=live,grip=np.array(1.),schema_version=np.array(1))
