"""Concrete PyRep/RLBench low-level timed controller satisfying TimedController protocol.

Exposes single-physics-step stepping, measured simulator clocks, and conversion of
raw RLBench camera/robot states into unquantized ICGS Observation records.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from icgs.contracts.records import Observation
from icgs.environments.rlbench.timed import TimedController


DEFAULT_WORKSPACE_BOUNDS: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-0.6, 0.6),  # X
    (-0.6, 0.6),  # Y
    (0.75, 1.60),  # Z (table surface to upper reach)
)

MAX_WORKSPACE_POINTS: int = 2048
JOINT_TARGET_ATOL_RAD: float = 0.60
GRIPPER_HOLD_ATOL: float = 0.001


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to 3x3 orthonormal rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_matrix(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert 7-dim pose [x, y, z, qx, qy, qz, qw] to 4x4 homogeneous transformation matrix."""
    pose = np.asarray(pose, dtype=np.float64).flatten()
    if pose.shape[0] < 7:
        raise ValueError(f"Pose must have at least 7 elements [x, y, z, qx, qy, qz, qw], got {pose.shape}")
    pos = pose[:3]
    rot = quaternion_to_rotation_matrix(pose[3:7])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to RLBench/PyRep [x,y,z,w] order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be finite 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not math.isfinite(float(norm)) or norm <= 0.0:
        raise ValueError("rotation quaternion is invalid")
    return quaternion / norm


def filter_and_downsample_points(
    points: np.ndarray,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    max_points: int = MAX_WORKSPACE_POINTS,
) -> np.ndarray:
    """Filter point cloud to finite workspace points and downsample uniformly.

    Points are retained as float64 per ADR0011 unquantized physical measurement requirements.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim > 2:
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 1:
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        pts = pts.reshape(-1, 3)

    # Filter non-finite points
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Filter by workspace bounding box
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    in_bounds = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    pts = pts[in_bounds]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Downsample uniformly if exceeding cap
    if pts.shape[0] > max_points:
        stride = int(math.ceil(pts.shape[0] / max_points))
        pts = pts[::stride][:max_points]

    return np.ascontiguousarray(pts, dtype=np.float64)


def convert_rlbench_observation(
    raw_obs: Any,
    *,
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
    max_points: int = MAX_WORKSPACE_POINTS,
) -> Observation:
    """Convert raw RLBench observation into an unquantized ICGS Observation.

    Accepts an RLBench observation instance or a dictionary mapping of sensor arrays.
    Extracts world point cloud from wrist/shoulder cameras, computes 4x4 T_w_e pose,
    and maps gripper state to [0.0, 1.0].
    """
    pts_list = []
    # Try retrieving point clouds from standard RLBench camera attributes or dictionary keys
    for cam in ("wrist", "left_shoulder", "right_shoulder", "front", "head"):
        pc = getattr(raw_obs, f"{cam}_point_cloud", None)
        if pc is None and isinstance(raw_obs, Mapping):
            pc = raw_obs.get(f"{cam}_point_cloud")
            if pc is None:
                pc = raw_obs.get(f"{cam}_points")
            if pc is None and cam == "wrist":
                pc = raw_obs.get("points")
        if pc is not None:
            arr = np.asarray(pc, dtype=np.float64)
            if arr.size > 0:
                pts_list.append(arr.reshape(-1, 3))

    if pts_list:
        all_pts = np.concatenate(pts_list, axis=0)
    else:
        # Fallback to points attribute directly
        pc = getattr(raw_obs, "points", None)
        if pc is None and isinstance(raw_obs, Mapping):
            pc = raw_obs.get("points")
        if pc is not None:
            all_pts = np.asarray(pc, dtype=np.float64).reshape(-1, 3)
        else:
            all_pts = np.zeros((0, 3), dtype=np.float64)

    points_f64 = filter_and_downsample_points(all_pts, bounds=workspace_bounds, max_points=max_points)

    # End-effector pose
    gripper_pose = getattr(raw_obs, "gripper_pose", None)
    if gripper_pose is None and isinstance(raw_obs, Mapping):
        for k in ("gripper_pose", "T_w_e", "pose"):
            if k in raw_obs and raw_obs[k] is not None:
                gripper_pose = raw_obs[k]
                break

    if gripper_pose is not None:
        arr_pose = np.asarray(gripper_pose, dtype=np.float64)
        if arr_pose.shape == (4, 4):
            T_w_e = arr_pose
        elif arr_pose.size >= 7:
            T_w_e = pose_to_matrix(arr_pose)
        else:
            T_w_e = np.eye(4, dtype=np.float64)
    else:
        T_w_e = np.eye(4, dtype=np.float64)

    # Gripper open status (RLBench: 1.0 open, 0.0 closed)
    gripper_open = getattr(raw_obs, "gripper_open", None)
    if gripper_open is None and isinstance(raw_obs, Mapping):
        gripper_open = raw_obs.get("gripper_open")
        if gripper_open is None:
            gripper_open = raw_obs.get("grip", 0.0)

    if gripper_open is not None:
        try:
            grip_val = float(gripper_open)
            grip_f64 = 1.0 if grip_val > 0.5 else 0.0
        except (TypeError, ValueError):
            grip_f64 = 0.0
    else:
        grip_f64 = 0.0

    return Observation(points=points_f64, T_w_e=T_w_e, grip=grip_f64)


def convert_rlbench_raw_observation(
    raw_obs: Any,
    *,
    sensor_profile_id: str = "rlbench-wrist-depth-raw-v1",
) -> Observation:
    """Preserve the complete finite wrist sensor cloud for measured archival.

    Accepted schema for 'rlbench-wrist-depth-raw-v1':
    - wrist_point_cloud: finite float32/float64 array with trailing XYZ dimension
    - gripper_pose: 4x4 transform matrix or 7-vector (x, y, z, qx, qy, qz, qw)
    - gripper_open: float or int (1.0 = open, 0.0 = closed)

    This path intentionally performs no workspace crop, voxelization, or point
    count cap. Other cameras (front, shoulder, overhead) are intentionally disabled
    under the E01 wrist-only protocol; enabling them raises ValueError to prevent
    silent schema divergence or uncalibrated multi-view ingestion.
    """
    if sensor_profile_id != "rlbench-wrist-depth-raw-v1":
        raise ValueError(
            f"unsupported sensor profile: {sensor_profile_id!r}; "
            "E01 exploratory protocol only supports 'rlbench-wrist-depth-raw-v1'"
        )

    # Strictly reject extra enabled cameras to prevent uncalibrated camera scope expansion
    extra_cameras = [
        cam for cam in (
            "left_shoulder_point_cloud",
            "right_shoulder_point_cloud",
            "overhead_point_cloud",
            "front_point_cloud",
        )
        if (isinstance(raw_obs, Mapping) and raw_obs.get(cam) is not None)
        or (not isinstance(raw_obs, Mapping) and getattr(raw_obs, cam, None) is not None)
    ]
    if extra_cameras:
        raise ValueError(
            f"unsupported cameras {extra_cameras}; "
            "E01 exploratory sensor protocol is strictly wrist-only ('rlbench-wrist-depth-raw-v1')"
        )

    pc = raw_obs.get("wrist_point_cloud") if isinstance(raw_obs, Mapping) else getattr(raw_obs, "wrist_point_cloud", None)
    if pc is None:
        raise ValueError("raw observation is missing wrist_point_cloud")
    points = np.asarray(pc)
    if points.ndim < 2 or points.shape[-1] != 3 or points.size == 0:
        raise ValueError("raw wrist point cloud must be nonempty with trailing XYZ dimension")
    if points.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or not np.isfinite(points).all():
        raise ValueError("raw wrist point cloud must be finite float32/float64")
    points = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float64)
    pose = raw_obs.get("gripper_pose") if isinstance(raw_obs, Mapping) else getattr(raw_obs, "gripper_pose", None)
    grip = raw_obs.get("gripper_open") if isinstance(raw_obs, Mapping) else getattr(raw_obs, "gripper_open", None)
    if pose is None or grip is None:
        raise ValueError("raw observation must include gripper_pose and gripper_open")
    return convert_rlbench_observation(
        {
            "wrist_point_cloud": points,
            "gripper_pose": pose,
            "gripper_open": grip,
        },
        workspace_bounds=((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)),
        max_points=points.shape[0],
    )


class RLBenchTimedController(TimedController):
    """Concrete TimedController wrapping RLBench and PyRep low-level physics stepping.

    Interfaces with CoppeliaSim 4.1 / PyRep physics steps directly, bypassing
    RLBench's high-level while-loop waypoint reaching to guarantee strict 0.1s interval cadence.
    """

    def __init__(
        self,
        task_env: Any = None,
        *,
        outer_env: Any = None,
        sim_api: Any = None,
        physics_dt: float | None = None,
        workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_WORKSPACE_BOUNDS,
        max_points: int = MAX_WORKSPACE_POINTS,
    ) -> None:
        if task_env is None:
            raise RuntimeError("live RLBench TaskEnvironment is required; synthetic controller fallback is forbidden")
        self._task_env = task_env
        self._scene = getattr(task_env, "_scene", None)
        self._robot = getattr(task_env, "_robot", None)
        pyrep = (
            getattr(self._scene, "pyrep", None)
            or getattr(self._scene, "_pyrep", None)
            or getattr(task_env, "_pyrep", None)
            or getattr(task_env, "pyrep", None)
            or getattr(outer_env, "_pyrep", None)
            or getattr(outer_env, "pyrep", None)
        )
        self._pyrep = pyrep
        if self._scene is None or self._robot is None or self._pyrep is None:
            raise RuntimeError("live RLBench private _scene/_robot/pyrep handles are required")
        self._arm = getattr(self._robot, "arm", None)
        self._gripper = getattr(self._robot, "gripper", None)
        if self._arm is None or self._gripper is None:
            raise RuntimeError("live RLBench robot.arm and robot.gripper handles are required")
        self._sim = sim_api
        if self._sim is None:
            try:
                from pyrep.backend import sim as backend_sim  # type: ignore
                self._sim = backend_sim
            except Exception as exc:
                raise RuntimeError("pinned pyrep.backend.sim is required for live simulator time") from exc
        self._get_sim_time = getattr(self._sim, "simGetSimulationTime", None)
        if not callable(self._get_sim_time):
            raise RuntimeError("pyrep.backend.sim.simGetSimulationTime is required; synthetic clock forbidden")
        get_dt = getattr(self._pyrep, "get_simulation_timestep", None)
        if not callable(get_dt):
            raise RuntimeError("PyRep.get_simulation_timestep is required")
        self._physics_dt = float(get_dt())
        if not math.isfinite(self._physics_dt) or self._physics_dt <= 0:
            raise RuntimeError("live simulator timestep must be finite and positive")
        if physics_dt is not None and not math.isclose(float(physics_dt), self._physics_dt, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("requested physics_dt disagrees with live PyRep timestep")
        self._outer_env = outer_env
        if self._outer_env is None:
            self._outer_env = getattr(task_env, "_env", None)
        if self._outer_env is None or not callable(getattr(self._outer_env, "shutdown", None)):
            raise RuntimeError("outer RLBench Environment.shutdown is required for lifecycle ownership")
        self._workspace_bounds = workspace_bounds
        self._max_points = max_points
        self._target_w: np.ndarray = np.eye(4, dtype=np.float64)
        self._target_grip: int = 0
        self._joint_targets: np.ndarray | None = None
        self._step_count: int = 0
        self._status: str = "ok"
        self._closed: bool = False
        self._shutdown_called: bool = False
        self._shutdown_attempted: bool = False
        self._staged_demo: Any | None = None

    @property
    def physics_timestep(self) -> float:
        return self._physics_dt

    def stage_demo(self, demo: Any) -> None:
        """Stage an expert demonstration for the next reset_to_demo lifecycle."""
        if demo is None:
            raise ValueError("staged demonstration cannot be None")
        self._staged_demo = demo

    def reset_to_demo(self, demo: Any) -> None:
        """Reset the environment state directly to a specific demonstration."""
        self._staged_demo = demo
        self.reset()

    def reset(self, seed: int | None = None, demo: Any | None = None) -> None:
        """Reset the RLBench scene with deterministic seed or to a staged/passed demo."""
        if self._shutdown_attempted:
            raise RuntimeError("controller cannot be reset after outer Environment.shutdown")
        self._closed = False
        self._status = "ok"
        self._step_count = 0
        self._target_w = np.eye(4, dtype=np.float64)
        self._target_grip = 0
        self._joint_targets = None

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise TypeError(f"seed must be an explicit integer, got {type(seed).__name__}")
            seed_int = int(seed)
            np.random.seed(seed_int)
            import random
            random.seed(seed_int)
            set_seed = getattr(self._task_env, "set_seed", None) or getattr(self._task_env, "seed", None)
            if callable(set_seed):
                try:
                    set_seed(seed_int)
                except Exception:
                    pass

        target_demo = demo if demo is not None else self._staged_demo
        self._staged_demo = None

        try:
            if target_demo is not None:
                reset_to_demo_fn = getattr(self._task_env, "reset_to_demo", None)
                if callable(reset_to_demo_fn):
                    reset_to_demo_fn(target_demo)
                else:
                    reset_fn = getattr(self._task_env, "reset", None)
                    if not callable(reset_fn):
                        raise RuntimeError("TaskEnvironment.reset is required")
                    try:
                        reset_fn(demo=target_demo)
                    except TypeError:
                        reset_fn(target_demo)
            else:
                reset_fn = getattr(self._task_env, "reset", None)
                if not callable(reset_fn):
                    raise RuntimeError("TaskEnvironment.reset is required")
                if seed is not None:
                    try:
                        reset_fn(seed=seed)
                    except TypeError:
                        reset_fn()
                else:
                    reset_fn()
        except Exception as exc:
            self._status = "reset_failed"
            raise RuntimeError("RLBench reset failed") from exc

    def set_target(self, target_w: np.ndarray, grip: int) -> None:
        """Set end-effector target matrix and binary gripper command for subsequent substeps."""
        self._target_w = np.array(target_w, dtype=np.float64, copy=True)
        if self._target_w.shape != (4, 4) or not np.isfinite(self._target_w).all():
            raise ValueError("target_w must be a finite 4x4 matrix")
        if not np.allclose(self._target_w[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
            raise ValueError("target_w must be a valid SE(3) pose")
        rotation = self._target_w[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("target_w must be a valid SE(3) pose")
        self._target_grip = int(grip > 0)
        solve_ik = getattr(self._arm, "solve_ik_via_jacobian", None)
        set_joint_targets = getattr(self._arm, "set_joint_target_positions", None)
        if not callable(solve_ik) or not callable(set_joint_targets):
            self._status = "ik_unavailable"
            raise RuntimeError("RLBench arm Jacobian IK and joint target APIs are required")
        position = self._target_w[:3, 3].tolist()
        quaternion = _rotation_matrix_to_quaternion(self._target_w[:3, :3])
        get_tip = getattr(self._arm, "get_tip", None)
        if callable(get_tip):
            try:
                tip = get_tip()
                get_quat = getattr(tip, "get_quaternion", None)
                if callable(get_quat):
                    current_quat = np.asarray(get_quat(), dtype=np.float64)
                    if current_quat.shape == (4,) and float(np.dot(quaternion, current_quat)) < 0.0:
                        quaternion = -quaternion
            except Exception:
                pass
        quaternion_list = quaternion.tolist()

        get_joints = getattr(self._arm, "get_joint_positions", None)
        cur_joints = np.asarray(get_joints(), dtype=np.float64) if callable(get_joints) else None

        try:
            joint_targets = None
            try:
                candidate = solve_ik(position, quaternion=quaternion_list)
                if candidate is not None:
                    cand_arr = np.asarray(candidate, dtype=np.float64)
                    if cur_joints is None or float(np.max(np.abs(cand_arr - cur_joints))) <= 0.20:
                        joint_targets = cand_arr
            except Exception:
                pass

            if joint_targets is None:
                get_configs = getattr(self._arm, "get_configs_for_tip_pose", None)
                if callable(get_configs):
                    try:
                        configs = get_configs(
                            position,
                            quaternion=quaternion_list,
                            ignore_collisions=True,
                            trials=300,
                            max_configs=20,
                        )
                        if len(configs) > 0:
                            if cur_joints is not None:
                                joint_targets = min(
                                    configs,
                                    key=lambda c: float(np.max(np.abs(np.asarray(c, dtype=np.float64) - cur_joints))),
                                )
                            else:
                                joint_targets = configs[0]
                    except Exception:
                        pass

            if joint_targets is None:
                solve_sampling = getattr(self._arm, "solve_ik_via_sampling", None)
                if callable(solve_sampling):
                    try:
                        configs = solve_sampling(
                            position,
                            quaternion=quaternion_list,
                            ignore_collisions=True,
                            trials=300,
                            max_configs=10,
                            max_time_ms=100,
                        )
                        if len(configs) > 0:
                            if cur_joints is not None:
                                joint_targets = min(
                                    configs,
                                    key=lambda c: float(np.max(np.abs(np.asarray(c, dtype=np.float64) - cur_joints))),
                                )
                            else:
                                joint_targets = configs[0]
                    except Exception:
                        pass

            if joint_targets is None:
                solve_general = getattr(self._arm, "solve_ik", None)
                if callable(solve_general):
                    try:
                        joint_targets = solve_general(position, quaternion=quaternion_list)
                    except Exception:
                        pass

            if joint_targets is None:
                raise RuntimeError("RLBench IK solvers failed to find a valid configuration for target pose")

            self._joint_targets = np.asarray(joint_targets, dtype=np.float64)
            if self._joint_targets.ndim != 1 or self._joint_targets.size == 0 or not np.isfinite(self._joint_targets).all():
                raise RuntimeError("RLBench IK returned invalid joint targets")
            set_joint_targets(self._joint_targets.tolist())
            actuate = getattr(self._gripper, "actuate", None)
            if not callable(actuate):
                raise RuntimeError("RLBench gripper.actuate is required")
            actuate(float(self._target_grip), velocity=0.04)
        except Exception as exc:
            self._status = "actuation_failed"
            raise RuntimeError(f"RLBench target materialization failed: {type(exc).__name__}: {exc}") from exc

    def step_physics(self) -> None:
        """Advance CoppeliaSim simulation by exactly one physics timestep."""
        if self._closed:
            raise RuntimeError("Controller is closed")

        before = self.simulator_time()
        try:
            actuate = getattr(self._gripper, "actuate", None)
            if not callable(actuate):
                raise RuntimeError("RLBench gripper.actuate is required")
            actuate(float(self._target_grip), velocity=0.04)
            step = getattr(self._scene, "step", None)
            if not callable(step):
                raise RuntimeError("TaskEnvironment._scene.step is required")
            step()
        except Exception as exc:
            self._status = "physics_step_failed"
            raise RuntimeError("RLBench scene physics step failed") from exc

        self._step_count += 1
        after = self.simulator_time()
        if not after > before:
            self._status = "clock_stalled"
            raise RuntimeError(f"live simulator clock did not advance: before={before}, after={after}")
        if self._joint_targets is None:
            self._status = "target_missing"
            raise RuntimeError("physics step requires a materialized IK target")
        get_joints = getattr(self._arm, "get_joint_positions", None)
        if not callable(get_joints):
            self._status = "verification_unavailable"
            raise RuntimeError("RLBench joint position query is required")
        observed = np.asarray(get_joints(), dtype=np.float64)
        if observed.shape != self._joint_targets.shape or not np.isfinite(observed).all():
            self._status = "target_unverified"
            raise RuntimeError("RLBench joint target observation is invalid")
        err = float(np.max(np.abs(observed - self._joint_targets)))
        if err > JOINT_TARGET_ATOL_RAD:
            self._status = "target_unverified"
            raise RuntimeError(f"RLBench joint target hold exceeded tolerance: max error={err:.6f} rad > tol={JOINT_TARGET_ATOL_RAD}")

    def observe(self) -> Observation:
        """Return current unquantized physical observation with uncropped raw sensor cloud."""
        return convert_rlbench_raw_observation(self.observe_raw())

    def observe_raw(self) -> Any:
        """Return the unmodified pinned RLBench observation for measured archival."""
        get_observation = getattr(self._task_env, "get_observation", None)
        if not callable(get_observation):
            raise RuntimeError("TaskEnvironment.get_observation is required")
        return get_observation()

    def simulator_time(self) -> float:
        """Return the current simulation clock time in seconds."""
        value = float(self._get_sim_time())
        if not math.isfinite(value) or value < 0.0:
            self._status = "clock_invalid"
            raise RuntimeError("live simulator clock returned a non-finite or negative value")
        return value

    def status(self) -> str:
        """Return controller operational status string."""
        return self._status

    def safe_hold(self) -> None:
        """Command safe holding state: zero joint velocities, hold current pose."""
        try:
            get_joints = getattr(self._arm, "get_joint_positions", None)
            set_targets = getattr(self._arm, "set_joint_target_positions", None)
            if not callable(get_joints) or not callable(set_targets):
                raise RuntimeError("RLBench joint hold APIs are required")
            set_targets(np.asarray(get_joints(), dtype=np.float64).tolist())
            actuate = getattr(self._gripper, "actuate", None)
            get_open = getattr(self._gripper, "get_open_amount", None)
            if not callable(actuate) or not callable(get_open):
                raise RuntimeError("RLBench gripper hold APIs are required")
            amount = float(np.asarray(get_open(), dtype=np.float64).reshape(-1)[0])
            actuate(amount, velocity=0.04)
        except Exception as exc:
            self._status = "safe_hold_failed"
            raise RuntimeError("RLBench safe hold failed") from exc

    def close(self) -> None:
        """Cleanly shut down simulation and release resources."""
        if not self._closed and not self._shutdown_attempted:
            self._shutdown_attempted = True
            try:
                self._outer_env.shutdown()
            except Exception as exc:
                self._status = "shutdown_failed"
                raise RuntimeError("outer RLBench Environment.shutdown failed") from exc
            self._closed = True
            self._shutdown_called = True


def make_online_observation(step: Any) -> dict[str, Any]:
    """Extract float32 online observation with valid mask from a TimedObservation or ExecutedTransition."""
    obs = step.observation if hasattr(step, "observation") else step.after.observation
    pts = np.asarray(obs.points, dtype=np.float32)
    return {
        "points": pts,
        "point_valid": np.ones(len(pts), dtype=bool),
        "T_w_e": np.asarray(obs.T_w_e, dtype=np.float32),
        "grip": float(obs.grip),
    }


__all__ = [
    "DEFAULT_WORKSPACE_BOUNDS",
    "MAX_WORKSPACE_POINTS",
    "RLBenchTimedController",
    "convert_rlbench_observation",
    "convert_rlbench_raw_observation",
    "filter_and_downsample_points",
    "make_online_observation",
    "pose_to_matrix",
    "quaternion_to_rotation_matrix",
]
