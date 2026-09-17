"""Materialize bounded timed commands from a live RLBench expert demonstration."""

from __future__ import annotations

import math
import hashlib
import json
import random
import time
from typing import Any, Callable, Sequence

import numpy as np

from icgs.contracts.method import TimedCommand
from icgs.environments.rlbench.controller import pose_to_matrix


def interpolate_poses(
    T_start: np.ndarray,
    T_end: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    """Linearly interpolate translation and normalized rotation across num_steps intervals."""
    if num_steps <= 1:
        return [np.array(T_end, dtype=np.float64, copy=True)]

    p_start = T_start[:3, 3]
    p_end = T_end[:3, 3]
    R_start = T_start[:3, :3]
    R_end = T_end[:3, :3]

    result = []
    for step in range(1, num_steps + 1):
        alpha = step / float(num_steps)
        p = (1.0 - alpha) * p_start + alpha * p_end
        # Simple weighted rotation blending with polar decomposition / SVD orthonormalization
        R_blend = (1.0 - alpha) * R_start + alpha * R_end
        u, _, vt = np.linalg.svd(R_blend)
        R_ortho = u @ vt
        if np.linalg.det(R_ortho) < 0:
            u[:, -1] *= -1
            R_ortho = u @ vt

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_ortho
        T[:3, 3] = p
        result.append(T)
    return result


def discretize_waypoints_to_commands(
    waypoints: Sequence[tuple[np.ndarray, int]],
    *,
    interval_s: float = 0.1,
    max_intervals: int = 32,
    nominal_speed: float = 0.05,  # 5 cm/s translation speed
) -> list[TimedCommand]:
    """Discretize a sequence of (4x4 pose, grip) keyframes into fixed-duration TimedCommands.

    Args:
        waypoints: List of (4x4 pose matrix, binary grip int) keyframes.
        interval_s: Duration per command step (default: 0.1 s).
        max_intervals: Hard upper bound on returned commands (default: 32).
        nominal_speed: Translation velocity in m/s for pacing intervals.

    Returns:
        List of TimedCommand instances bounded to max_intervals.
    """
    if not waypoints:
        return []

    commands: list[TimedCommand] = []
    current_pose = np.asarray(waypoints[0][0], dtype=np.float64)
    current_grip = int(waypoints[0][1] > 0)

    for next_pose, next_grip in waypoints[1:]:
        if len(commands) >= max_intervals:
            raise ValueError(
                "expert demonstration exceeds max_intervals; truncation/padding is forbidden"
            )

        next_pose = np.asarray(next_pose, dtype=np.float64)
        next_grip = int(next_grip > 0)

        # Compute translation distance to gauge step count
        dist = float(np.linalg.norm(next_pose[:3, 3] - current_pose[:3, 3]))
        # Duration based on nominal speed, at least 1 step
        est_duration = max(dist / max(nominal_speed, 1e-4), interval_s)
        steps = max(1, round(est_duration / interval_s))

        # Never truncate a live expert trajectory to a caller cap.
        if len(commands) + steps > max_intervals:
            raise ValueError(
                "expert demonstration exceeds max_intervals; truncation/padding is forbidden"
            )

        interpolated = interpolate_poses(current_pose, next_pose, steps)
        for T in interpolated:
            commands.append(TimedCommand(
                target_w=T,
                grip=next_grip,
                duration_s=interval_s,
            ))

        current_pose = next_pose
        current_grip = next_grip

    if len(commands) > max_intervals:
        raise ValueError(
            "expert demonstration exceeds max_intervals; truncation/padding is forbidden"
        )

    return commands


def extract_rlbench_demo_waypoints(demo: Any) -> list[tuple[np.ndarray, int]]:
    """Extract (4x4 pose, grip) keyframes from an RLBench Demonstration object."""
    waypoints = []
    observations = getattr(demo, "_observations", demo)
    if not isinstance(observations, (list, tuple)):
        observations = [observations]

    if not observations:
        raise ValueError("expert demonstration contains no observations")
    for obs in observations:
        gripper_pose = getattr(obs, "gripper_pose", None)
        gripper_open = getattr(obs, "gripper_open", None)
        if gripper_pose is None:
            raise ValueError("expert demonstration observation is missing gripper_pose")
        if gripper_open is None or not math.isfinite(float(gripper_open)):
            raise ValueError("expert demonstration observation is missing finite gripper_open")
        T = pose_to_matrix(gripper_pose)
        grip = 1 if float(gripper_open) > 0.5 else 0
        waypoints.append((T, grip))

    return waypoints


class RLBenchCommandOracle:
    """Provides discretized 0.1s TimedCommand sequences for an AttemptSpec."""

    def __init__(
        self,
        waypoints: Sequence[tuple[np.ndarray, int]] | None = None,
        *,
        demo_content_hash: str | None = None,
        materialization_id: str | None = None,
        max_intervals: int = 32,
        interval_s: float = 0.1,
    ) -> None:
        if not waypoints:
            raise ValueError("live expert demonstration waypoints are required; nominal fallback is forbidden")
        if not isinstance(demo_content_hash, str) or not demo_content_hash.strip():
            raise ValueError("demo_content_hash is required for expert materialization")
        if not isinstance(materialization_id, str) or not materialization_id.strip():
            raise ValueError("materialization_id is required for expert materialization")
        self._waypoints = list(waypoints)
        self.demo_content_hash = demo_content_hash
        self.materialization_id = materialization_id
        self._max_intervals = max_intervals
        self._interval_s = interval_s

    def generate_commands(self, seed: int | None = None) -> list[TimedCommand]:
        """Generate deterministic TimedCommand trajectory for this attempt."""
        return discretize_waypoints_to_commands(
            self._waypoints,
            interval_s=self._interval_s,
            max_intervals=self._max_intervals,
        )


def materialize_expert_demo(
    demo: Any,
    *,
    source_protocol_id: str,
    max_intervals: int = 32,
    interval_s: float = 0.1,
) -> RLBenchCommandOracle:
    """Build an oracle only from a concrete RLBench demo and provenance identity."""
    if not isinstance(source_protocol_id, str) or not source_protocol_id.strip():
        raise ValueError("source_protocol_id is required for expert materialization")
    waypoints = extract_rlbench_demo_waypoints(demo)
    digest_payload = []
    for pose, grip in waypoints:
        digest_payload.append({"pose": np.asarray(pose, dtype=np.float64).round(12).tolist(), "grip": grip})
    demo_content_hash = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    materialization_id = f"{source_protocol_id}:{demo_content_hash}"
    return RLBenchCommandOracle(
        waypoints,
        demo_content_hash=demo_content_hash,
        materialization_id=materialization_id,
        max_intervals=max_intervals,
        interval_s=interval_s,
    )


def orchestrate_live_expert_demo(
    task_env: Any,
    controller: Any | None = None,
    *,
    source_protocol_id: str,
    max_intervals: int = 32,
    interval_s: float = 0.1,
    seed: int | None = None,
    timeout_s: float | None = 60.0,
) -> tuple[Any, RLBenchCommandOracle, list[TimedCommand]]:
    """Orchestrate live demonstration retrieval, state reset, and command materialization.

    1. Applies seed deterministically to numpy and random if provided.
    2. Queries task_env.get_demos(amount=1, live_demos=True).
    3. Stages demo onto controller or executes reset_to_demo directly on task_env.
    4. Materializes commands via materialize_expert_demo without padding or truncation.
    Enforces bounded wall clock timeout across setup, demo query, reset_to_demo staging,
    and command materialization.
    """
    t_start = time.monotonic()

    def _check_timeout(phase: str) -> None:
        if timeout_s is not None:
            elapsed = time.monotonic() - t_start
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"expert demonstration generation exceeded wall timeout ({timeout_s}s) during {phase}; elapsed {elapsed:.2f}s"
                )

    if task_env is None:
        raise ValueError("task_env is required for live expert demonstration")
    get_demos_fn = getattr(task_env, "get_demos", None)
    if not callable(get_demos_fn):
        raise RuntimeError("task_env.get_demos is required for live expert demonstration")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an explicit integer, got {type(seed).__name__}")
        s = int(seed)
        np.random.seed(s)
        random.seed(s)

    _check_timeout("setup")
    demos = get_demos_fn(amount=1, live_demos=True)
    _check_timeout("demo query")
    if not demos:
        raise RuntimeError("live expert demonstration query returned empty demos list")
    demo = demos[0]

    # Stage demo on controller so reset() inside collect_attempt triggers reset_to_demo
    if controller is not None and callable(getattr(controller, "stage_demo", None)):
        controller.stage_demo(demo)
    elif callable(getattr(task_env, "reset_to_demo", None)):
        task_env.reset_to_demo(demo)
    _check_timeout("demo staging")

    oracle = materialize_expert_demo(
        demo,
        source_protocol_id=source_protocol_id,
        max_intervals=max_intervals,
        interval_s=interval_s,
    )
    commands = oracle.generate_commands(seed=seed)
    _check_timeout("command generation")
    return demo, oracle, commands


def create_live_expert_command_factory(
    task_env: Any | None = None,
    *,
    source_protocol_id: str,
    max_intervals: int = 32,
    interval_s: float = 0.1,
    timeout_s: float | None = 60.0,
) -> Any:
    """Create a command factory callback conforming to runner.py interface.

    When invoked as command_factory(spec, env), discovers the underlying controller
    and task_env, queries live demo, stages reset_to_demo, and returns TimedCommands.
    Also retains the materialized oracle and demo so their hash/protocol identities
    are never discarded.
    """
    def command_factory(spec: Any, env: Any) -> Any:
        nonlocal task_env
        current_task_env = task_env
        controller = getattr(env, "controller", getattr(env, "_controller", None))
        if current_task_env is None and controller is not None:
            current_task_env = getattr(controller, "_task_env", None)
        if current_task_env is None:
            current_task_env = getattr(env, "task_env", getattr(env, "_task_env", None))
        if current_task_env is None:
            raise RuntimeError("Unable to resolve task_env from environment adapter for live expert collection")

        action_seed = getattr(spec, "action_seed", getattr(spec, "reset_seed", None))
        demo, oracle, commands = orchestrate_live_expert_demo(
            current_task_env,
            controller=controller,
            source_protocol_id=source_protocol_id,
            max_intervals=max_intervals,
            interval_s=interval_s,
            seed=action_seed,
            timeout_s=timeout_s,
        )
        if controller is not None:
            controller._last_demo = demo
            controller._last_oracle = oracle
            controller._last_demo_hash = oracle.demo_content_hash
            controller._last_materialization_id = oracle.materialization_id

        command_factory.last_demo = demo
        command_factory.last_oracle = oracle

        from icgs.data.collection.runner import MaterializedCommands
        return MaterializedCommands(
            commands=commands,
            raw_commands_id=f"{source_protocol_id}:demo:{oracle.demo_content_hash}",
            materialized_commands_id=oracle.materialization_id,
        )

    command_factory.last_demo = None
    command_factory.last_oracle = None
    return command_factory


def create_bound_attempt_spec(
    task_env: Any,
    *,
    episode_id: str,
    program_id: str = "E01",
    source_lineage_id: str = "exploratory_rlbench_dev",
    asset_family_id: str = "rlbench-exploratory",
    split: str = "dev",
    generator_seed: int = 1000,
    reset_seed: int = 2000,
    action_seed: int = 3000,
    calibration_id: str = "rlbench-wrist-depth-raw-v1",
    observation_origin: str = "measured",
    source_protocol_id: str = "rlbench-live-expert-v1",
    max_intervals: int = 32,
    interval_s: float = 0.1,
    controller: Any | None = None,
    timeout_s: float | None = 60.0,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, RLBenchCommandOracle]:
    """Create an AttemptSpec whose raw_commands_id and materialized_commands_id are concretely bound to a live demo."""
    from icgs.data.collection.runner import AttemptSpec

    demo, oracle, commands = orchestrate_live_expert_demo(
        task_env,
        controller=controller,
        source_protocol_id=source_protocol_id,
        max_intervals=max_intervals,
        interval_s=interval_s,
        seed=action_seed,
        timeout_s=timeout_s,
    )
    spec = AttemptSpec(
        episode_id=episode_id,
        program_id=program_id,
        source_lineage_id=source_lineage_id,
        asset_family_id=asset_family_id,
        split=split,
        generator_seed=generator_seed,
        reset_seed=reset_seed,
        action_seed=action_seed,
        calibration_id=calibration_id,
        observation_origin=observation_origin,
        raw_commands_id=f"{source_protocol_id}:demo:{oracle.demo_content_hash}",
        materialized_commands_id=oracle.materialization_id,
        commands=commands,
        metadata=metadata,
    )
    return spec, demo, oracle


__all__ = [
    "RLBenchCommandOracle",
    "create_bound_attempt_spec",
    "create_live_expert_command_factory",
    "discretize_waypoints_to_commands",
    "extract_rlbench_demo_waypoints",
    "interpolate_poses",
    "materialize_expert_demo",
    "orchestrate_live_expert_demo",
]
