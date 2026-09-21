"""Phase-1 camera calibration profiles."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from icgs.data.collection.v3.protocol import V3_PROTOCOL


CAMERA_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "rlbench-wrist-depth-v1": {
        "camera_profile_id": "rlbench-wrist-depth-v1",
        "camera_intrinsics_id": "rlbench-wrist-intrinsics-v1",
        "camera_extrinsics_id": "rlbench-wrist-extrinsics-v1",
        "action_space_id": V3_PROTOCOL.action_space_id,
        "orientation_convention": V3_PROTOCOL.orientation_convention,
        "gripper_unit": V3_PROTOCOL.gripper_unit,
        "mount": "wrist",
        "intrinsics": {
            "fx": 1120.0,
            "fy": 1120.0,
            "cx": 640.0,
            "cy": 360.0,
            "width": 1280,
            "height": 720,
            "source": "rlbench-default-unmeasured",
        },
        "distortion": {"model": "none", "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "extrinsics": {
            "frame": "end_effector",
            "T_ee_cam": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "timestamp_relation": "same_as_observation",
            "world_from": "T_world_ee @ T_ee_cam",
        },
        "depth_unit": "meter",
        "pointcloud_frame": "world",
        "rgb_depth_mask_synchronized": True,
        "timestamp_relation": "identical",
        "invalid_depth_value": 0.0,
        "mask_id_to_object_role": "episode.scene.object_roles",
    },
    "rlbench-base-cameras-v1": {
        "camera_profile_id": "rlbench-base-cameras-v1",
        "camera_intrinsics_id": "rlbench-base-intrinsics-v1",
        "camera_extrinsics_id": "rlbench-base-extrinsics-v1",
        "action_space_id": V3_PROTOCOL.action_space_id,
        "orientation_convention": V3_PROTOCOL.orientation_convention,
        "gripper_unit": V3_PROTOCOL.gripper_unit,
        "mount": "static",
        "intrinsics": {
            "fx": 1120.0,
            "fy": 1120.0,
            "cx": 640.0,
            "cy": 360.0,
            "width": 1280,
            "height": 720,
            "source": "rlbench-default-unmeasured",
        },
        "distortion": {"model": "none", "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "extrinsics": {
            "frame": "world",
            "T_world_cam": None,
            "timestamp_relation": "same_as_observation",
        },
        "depth_unit": "meter",
        "pointcloud_frame": "world",
        "rgb_depth_mask_synchronized": True,
        "timestamp_relation": "identical",
        "invalid_depth_value": 0.0,
        "mask_id_to_object_role": "episode.scene.object_roles",
    },
})


def get_camera_profile(profile_id: str | None = None) -> dict[str, Any]:
    key = profile_id or V3_PROTOCOL.camera_profile_id
    if key not in CAMERA_PROFILES:
        raise KeyError(f"unknown camera profile: {key}")
    return dict(CAMERA_PROFILES[key])


def wrist_camera_pose_world(T_world_ee: Any, profile_id: str | None = None) -> Any:
    """T_world_camera = T_world_ee @ T_ee_cam for the wrist mount."""
    import numpy as np

    profile = get_camera_profile(profile_id)
    T_ee_cam = np.asarray(profile["extrinsics"]["T_ee_cam"], dtype=np.float64)
    return np.asarray(T_world_ee, dtype=np.float64) @ T_ee_cam
