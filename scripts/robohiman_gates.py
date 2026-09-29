"""Simulator gates for the RoboHiMan migration (run in the RoboHiMan venv under Xvfb).

  parity      upstream Scene.get_demo vs ICGS mirror (and each against itself)
              from one numpy RNG state; also proves the recorder/rendering are
              physics-neutral (upstream run records no frames at gripper steps).
  replay      anchor snapshot/restore and reset→replay drift, open and closed loop.
  a0          live geometry: numpy back-projection vs PyRep, object-box
              containment of mask pixels, cross-camera agreement, mask decoding.
  dependency  branch audit: candidate pull extents at one anchor, local outcome,
              continuations with a jittered scripted expert (NOT pi_ref).

Each subcommand writes one JSON report; thresholds are declared in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rlbench.backend.exceptions import TaskEnvironmentError

from icgs.data.stage1.store import canonical_json
from icgs.environments.rlbench.replay import validate_replay_report
from icgs.environments.robohiman.camera import (
    decode_mask_handles,
    depth_to_world,
    metric_depth,
    valid_depth_mask,
)
from icgs.environments.robohiman.episode import frame_capture
from icgs.environments.robohiman.expert import WaypointExpert, expert_control
from icgs.environments.robohiman.monitors import build_monitor
from icgs.environments.robohiman.recorder import StepRecorder
from icgs.environments.robohiman.session import RoboHiManSession
from icgs.environments.robohiman.snapshot import (
    REPLAY_ID,
    SNAPSHOT_FIELDS,
    SNAPSHOT_ID,
    capture_snapshot,
    replay_commands,
    restore_snapshot,
    trajectory_discrepancy,
)


# Declared before measurement: what "acceptable for branch collection" means.
REPLAY_THRESHOLDS = {
    "max_tip_position_m": 0.005,
    "final_object_position_m": 0.005,
    "max_articulation_m": 0.005,
    "predicate_disagreement_fraction": 0.0,
    "final_success_equal": 1.0,
}


def _session(args: argparse.Namespace, *, cameras=(), masks=False, point_cloud=False, image=(64, 64)):
    session = RoboHiManSession(args.task, strategy_index=args.strategy, image_size=image, cameras=cameras,
                               env_seed=args.env_seed, masks=masks, point_cloud=point_cloud)
    session.launch()
    return session


def _run(session, lineage, variation, *, mode: str, frames: bool = False, perturbations=None,
         stop_after=None, max_steps=3000, start_snapshot=None):
    """mode: 'upstream' (Scene.get_demo) or 'mirror' (WaypointExpert).

    With ``start_snapshot`` the run starts from a restored post-reset state
    instead of a fresh reset, removing reset-to-reset robot residuals.
    """
    if start_snapshot is None:
        session.reset(variation, lineage=lineage)
    else:
        restore_snapshot(session, start_snapshot)
        session.scene._has_init_episode = True  # what reset leaves; get_demo only checks it
    monitor = build_monitor(session.task, session.task_env, variation)
    capture = frame_capture(session) if frames else None
    recorder = StepRecorder(session, monitor, frame_stride=1 if frames else 0, capture_frame=capture)
    outcome: dict[str, Any] = {}
    with recorder:
        recorder.arm()
        with expert_control(session):
            if mode == "upstream":
                try:
                    session.scene.get_demo(record=True)
                    outcome = {"status": "success"}
                except Exception as error:  # upstream raises on unsuccessful demos
                    outcome = {"status": "exception", "error": f"{type(error).__name__}: {error}"}
            else:
                result = WaypointExpert(session, recorder, perturbations=perturbations,
                                        max_steps=max_steps).run(stop_after=stop_after)
                outcome = {"status": result.status, "reason": result.reason}
        recorder.disarm()
    outcome["task_success"] = bool(session.task_obj.success()[0])
    return recorder, recorder.arrays(), outcome


def _warm_lineage(session, args) -> dict[str, Any]:
    """Reset once so factor RNGs exist, then capture the full reset lineage."""
    np.random.seed(args.seed)
    session.reset(args.variation)
    np.random.seed(args.seed + 1000)
    return session.lineage_state()


def _compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    la, lb = a["cmd_phase"].shape[0], b["cmd_phase"].shape[0]
    length = min(la, lb)
    result: dict[str, Any] = {"steps_a": int(la), "steps_b": int(lb)}
    result.update(trajectory_discrepancy(a, b, 0, 0, length))
    for key in ("cmd_arm_joint_target", "cmd_gripper_joint_velocity"):
        valid = a[f"{key}_valid"][:length] & b[f"{key}_valid"][:length]
        diff = np.abs(a[key][:length] - b[key][:length])
        result[f"max_{key}_diff"] = float(diff[valid].max(initial=0.0))
        result[f"{key}_mask_equal"] = bool(np.array_equal(a[f"{key}_valid"][:length], b[f"{key}_valid"][:length]))
    result["bitwise_joint_trajectory_equal"] = bool(la == lb and np.array_equal(
        a["step_joint_positions"], b["step_joint_positions"]))
    result["waypoint_phase_sequence_equal"] = bool(la == lb and np.array_equal(a["cmd_waypoint"], b["cmd_waypoint"])
                                                   and np.array_equal(a["cmd_phase"], b["cmd_phase"]))
    result["final_boundary"] = trajectory_discrepancy(a, b, la, lb, 0)
    first = np.flatnonzero(np.abs(a["step_joint_positions"][:length + 1]
                                  - b["step_joint_positions"][:length + 1]).max(axis=1) > 1e-6)
    result["first_boundary_joint_diff_gt_1e-6"] = int(first[0]) if first.size else None
    return result


def cmd_parity(args: argparse.Namespace) -> dict[str, Any]:
    session = _session(args, cameras=("front", "wrist"), image=(64, 64))
    try:
        state = _warm_lineage(session, args)
        session.reset(args.variation, lineage=state)
        start = capture_snapshot(session, 0)
        report: dict[str, Any] = {"gate": "parity", "task": args.task, "variation": args.variation,
                                  "seed": args.seed}
        for label, snapshot in (("fresh_reset_each_run", None), ("restored_post_reset_snapshot", start)):
            runs = {}
            failures = {}
            for name, mode, frames in (("upstream_1", "upstream", False), ("mirror_1", "mirror", True),
                                       ("upstream_2", "upstream", False), ("mirror_2", "mirror", False)):
                started = time.time()
                try:
                    _, arrays, outcome = _run(session, state, args.variation, mode=mode, frames=frames,
                                              start_snapshot=snapshot)
                except TaskEnvironmentError as error:
                    failures[name] = f"reset failed: {error}"
                    continue
                runs[name] = (arrays, {**outcome, "steps": int(arrays["cmd_phase"].shape[0]),
                                       "frames": int(arrays.get("frame_step", np.zeros(0)).shape[0]),
                                       "wall_s": round(time.time() - started, 1)})
            if failures:
                report[label] = {"runs": {k: v[1] for k, v in runs.items()}, "reset_failures": failures}
                continue
            report[label] = {
                "runs": {k: v[1] for k, v in runs.items()},
                "comparisons": {
                    "upstream_1_vs_upstream_2 (simulator determinism)":
                        _compare(runs["upstream_1"][0], runs["upstream_2"][0]),
                    "upstream_1_vs_mirror_1 (mirror parity + per-step rendering neutrality)":
                        _compare(runs["upstream_1"][0], runs["mirror_1"][0]),
                    "mirror_1_vs_mirror_2 (rendering on vs off)": _compare(runs["mirror_1"][0], runs["mirror_2"][0]),
                },
            }
        return report
    finally:
        session.shutdown()


def _final_depth(session) -> dict[str, np.ndarray]:
    obs = session.scene.get_observation()
    out = {}
    for name in session.cameras:
        near = obs.misc[f"{name}_camera_near"]
        far = obs.misc[f"{name}_camera_far"]
        depth = np.asarray(getattr(obs, f"{name}_depth"), dtype=np.float64)
        out[name] = np.where(valid_depth_mask(depth), metric_depth(depth, near, far), np.nan)
    return out


def _depth_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    result = {}
    for name in a:
        both = np.isfinite(a[name]) & np.isfinite(b[name])
        diff = np.abs(a[name][both] - b[name][both])
        result[f"{name}_mean_abs_m"] = float(diff.mean()) if diff.size else 0.0
        result[f"{name}_frac_gt_1cm"] = float((diff > 0.01).mean()) if diff.size else 0.0
        result[f"{name}_validity_mismatch_frac"] = float((np.isfinite(a[name]) != np.isfinite(b[name])).mean())
    return result


def cmd_replay(args: argparse.Namespace) -> dict[str, Any]:
    session = _session(args, cameras=("front", "left_shoulder"), image=(64, 64))
    report: dict[str, Any] = {"gate": "replay", "task": args.task, "variation": args.variation,
                              "seed": args.seed, "anchor_after_waypoint": args.anchor_waypoint,
                              "thresholds": REPLAY_THRESHOLDS, "snapshot_id": SNAPSHOT_ID, "replay_id": REPLAY_ID}
    try:
        state0 = _warm_lineage(session, args)
        session.reset(args.variation, lineage=state0)
        monitor = build_monitor(session.task, session.task_env, args.variation)
        # Reference run: expert to the anchor, snapshot, expert to the end.
        ref = StepRecorder(session, monitor, frame_stride=0)
        with ref:
            ref.arm()
            with expert_control(session):
                first = WaypointExpert(session, ref).run(stop_after=args.anchor_waypoint)
                anchor = len(ref.steps) - 1
                snap = capture_snapshot(session, anchor)
                rest = WaypointExpert(session, ref).run(start=args.anchor_waypoint + 1, initial_step=False)
            ref.disarm()
        ref_arrays = ref.arrays()
        ref_depth = _final_depth(session)
        end = ref_arrays["cmd_phase"].shape[0]
        report["reference"] = {"anchor_boundary": anchor, "end_boundary": end, "anchor_status": first.status,
                               "final_status": rest.status, "task_success": rest.task_success}

        def after_restore(label: str, closed_loop: bool) -> dict[str, Any]:
            restore_snapshot(session, snap)
            rec = StepRecorder(session, monitor, frame_stride=0)
            with rec:
                rec.arm()
                with expert_control(session):
                    rec.phase = "replay"
                    if closed_loop:
                        result = WaypointExpert(session, rec).run(start=args.anchor_waypoint + 1, initial_step=False)
                        status = result.status
                    else:
                        replay_commands(session, ref_arrays, anchor, end)
                        status = "replayed"
                rec.disarm()
            arrays = rec.arrays()
            length = min(end - anchor, arrays["cmd_phase"].shape[0])
            metrics = trajectory_discrepancy(ref_arrays, arrays, anchor, 0, length)
            metrics.update({"steps": int(arrays["cmd_phase"].shape[0]), "status": status,
                            "restored_state_joint_err_rad": float(np.abs(
                                arrays["step_joint_positions"][0] - ref_arrays["step_joint_positions"][anchor]).max()),
                            "restored_state_object_err_m": float(np.linalg.norm(
                                arrays["step_object_poses"][0][:, :3] - ref_arrays["step_object_poses"][anchor][:, :3],
                                axis=-1).max()),
                            "restored_state_articulation_err_m": float(np.abs(
                                arrays["step_task_joint_positions"][0]
                                - ref_arrays["step_task_joint_positions"][anchor]).max())})
            metrics["final_depth"] = _depth_diff(ref_depth, _final_depth(session))
            return {"label": label, "metrics": metrics, "arrays": arrays}

        b = after_restore("restore_then_openloop_replay_1", False)
        c = after_restore("restore_then_openloop_replay_2", False)
        e = after_restore("restore_then_closedloop_expert", True)
        repeat = trajectory_discrepancy(b["arrays"], c["arrays"], 0, 0,
                                        min(b["arrays"]["cmd_phase"].shape[0], c["arrays"]["cmd_phase"].shape[0]))
        # Reset → replay the full recorded command history.
        report["reset_before_replay"] = {k: v for k, v in session.reset(args.variation, lineage=state0).items()
                                         if k.startswith("failed_placement")}
        d_rec = StepRecorder(session, monitor, frame_stride=0)
        with d_rec:
            d_rec.arm()
            with expert_control(session):
                d_rec.phase = "replay"
                replay_commands(session, ref_arrays, 0, end)
            d_rec.disarm()
        d_arrays = d_rec.arrays()
        d_anchor = trajectory_discrepancy(ref_arrays, d_arrays, 0, 0, anchor)
        d_full = trajectory_discrepancy(ref_arrays, d_arrays, 0, 0, end)
        d_full["final_depth"] = _depth_diff(ref_depth, _final_depth(session))
        # Reset → closed-loop expert from scratch (full re-execution determinism).
        try:
            _, f_arrays, f_outcome = _run(session, state0, args.variation, mode="mirror")
            f_full = _compare(ref_arrays, f_arrays)
        except TaskEnvironmentError as error:
            f_outcome, f_full = {"status": "reset_failed", "error": str(error)}, None
        report["results"] = {
            b["label"]: b["metrics"], c["label"]: c["metrics"], e["label"]: e["metrics"],
            "openloop_replay_1_vs_2 (restore repeatability)": repeat,
            "reset_then_openloop_replay_full_history@anchor": d_anchor,
            "reset_then_openloop_replay_full_history@end": d_full,
            "reset_then_closedloop_expert_full": {**(f_full or {}), **f_outcome},
        }

        def acceptable(metrics: dict[str, Any]) -> bool:
            return (metrics["max_tip_position_m"] <= REPLAY_THRESHOLDS["max_tip_position_m"]
                    and metrics["final_object_position_m"] <= REPLAY_THRESHOLDS["final_object_position_m"]
                    and metrics["max_articulation_m"] <= REPLAY_THRESHOLDS["max_articulation_m"]
                    and metrics["predicate_disagreement_fraction"] <= 0.0
                    and metrics["final_success_equal"] >= 1.0)

        report["acceptable"] = {
            "snapshot_restore_openloop": acceptable(b["metrics"]) and acceptable(c["metrics"]),
            "snapshot_restore_closedloop_expert": acceptable(e["metrics"]),
            "reset_replay_openloop": acceptable(d_anchor) and acceptable(d_full),
            "reset_closedloop_reexecution": acceptable(f_full) if f_full else None,
        }
        discrepancies = {"pose": max(b["metrics"]["max_tip_position_m"], c["metrics"]["max_tip_position_m"]),
                         "cloud": max(v for k, v in b["metrics"]["final_depth"].items() if k.endswith("mean_abs_m")),
                         "outcome": 1.0 - b["metrics"]["final_success_equal"]}
        report["replay_reports"] = {
            "snapshot": validate_replay_report({
                "mode": "approximate-reset", "protocol_id": f"{SNAPSHOT_ID}+{REPLAY_ID}",
                "stored_fields": SNAPSHOT_FIELDS["stored"], "restored_fields": SNAPSHOT_FIELDS["restored"],
                "unavailable_fields": SNAPSHOT_FIELDS["unavailable"], "discrepancies": discrepancies,
                "repetitions": 2,
                "uncertainty": {"restore_repeat_max_tip_m": repeat["max_tip_position_m"],
                                "restore_repeat_final_object_m": repeat["final_object_position_m"]}}),
            "reset_replay": validate_replay_report({
                "mode": "deterministic-replay", "protocol_id": f"numpy-rng-reset+{REPLAY_ID}",
                "stored_fields": ("rng", "command_history", "task_history", "monitor_history", "joints", "velocities",
                                  "object_poses", "grip", "articulations", "attachments", "controller_state"),
                "restored_fields": ("rng", "command_history", "task_history", "monitor_history"),
                "unavailable_fields": ("constraints", "integrators", "solver"),
                "discrepancies": {"pose": d_full["max_tip_position_m"],
                                  "cloud": max(v for k, v in d_full["final_depth"].items() if k.endswith("mean_abs_m")),
                                  "outcome": 1.0 - d_full["final_success_equal"]}}),
        }
        return report
    finally:
        session.shutdown()


def _world_box(shape) -> tuple[np.ndarray, np.ndarray]:
    bbox = np.asarray(shape.get_bounding_box(), dtype=np.float64)
    corners = np.array([[x, y, z] for x in bbox[0:2] for y in bbox[2:4] for z in bbox[4:6]])
    matrix = np.asarray(shape.get_matrix(), dtype=np.float64)
    if matrix.size == 12:
        matrix = np.vstack([matrix.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def cmd_a0(args: argparse.Namespace) -> dict[str, Any]:
    from pyrep.const import ObjectType
    from pyrep.objects.vision_sensor import VisionSensor

    cameras = ("front", "left_shoulder", "right_shoulder", "wrist")
    session = _session(args, cameras=cameras, masks=True, point_cloud=True, image=tuple(args.image_size))
    try:
        np.random.seed(args.seed)
        session.reset(args.variation)
        obs = session.scene.get_observation()
        shapes = {s.get_handle(): s for s in session.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)}
        report: dict[str, Any] = {"gate": "a0", "task": args.task, "image_size": list(args.image_size),
                                  "masks_as_one_channel": bool(session.cfg.data.masks_as_one_channel), "cameras": {}}
        centroids: dict[str, dict[str, list[float]]] = {}
        for name in cameras:
            near = float(obs.misc[f"{name}_camera_near"])
            far = float(obs.misc[f"{name}_camera_far"])
            K = np.asarray(obs.misc[f"{name}_camera_intrinsics"])
            T = np.asarray(obs.misc[f"{name}_camera_extrinsics"])
            depth = np.asarray(getattr(obs, f"{name}_depth"), dtype=np.float64)
            valid = valid_depth_mask(depth)
            ours = depth_to_world(metric_depth(depth, near, far), T, K)
            pyrep_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(near + depth * (far - near), T, K)
            live = np.asarray(getattr(obs, f"{name}_point_cloud"), dtype=np.float64)
            raw_mask = np.asarray(getattr(obs, f"{name}_mask"))
            if raw_mask.ndim == 3:
                handles = decode_mask_handles(raw_mask)
                truncating = (raw_mask.astype(np.float64) * 255).astype(np.int64)
                upstream = truncating[..., 0] + truncating[..., 1] * 256 + truncating[..., 2] * 65536
                decode_note = {"encoding": "rgb", "truncating_vs_rounding_mismatch_frac":
                               float((upstream != handles).mean())}
            else:
                handles = np.rint(raw_mask).astype(np.int64)
                decode_note = {"encoding": "one_channel", "non_integer_frac":
                               float((np.abs(raw_mask - np.rint(raw_mask)) > 1e-6).mean())}
            known = np.isin(handles, list(shapes))
            per_object = {}
            for handle in np.unique(handles[valid & known]):
                shape = shapes[int(handle)]
                pixels = valid & (handles == handle)
                if pixels.sum() < 5:
                    continue
                low, high = _world_box(shape)
                points = ours[pixels]
                inside = np.all((points >= low - 0.01) & (points <= high + 0.01), axis=1)
                per_object[shape.get_name()] = {"pixels": int(pixels.sum()),
                                                "inside_world_box_1cm_frac": float(inside.mean())}
                centroids.setdefault(shape.get_name(), {})[name] = points.mean(axis=0).tolist()
            report["cameras"][name] = {
                "near_far_m": [near, far],
                "fx_fy": [float(K[0, 0]), float(K[1, 1])],
                "valid_depth_frac": float(valid.mean()),
                "metric_depth_range_m": [float(metric_depth(depth, near, far)[valid].min()),
                                         float(metric_depth(depth, near, far)[valid].max())],
                "numpy_vs_pyrep_max_abs_m": float(np.abs(ours - pyrep_cloud).max()),
                "numpy_vs_live_pointcloud_max_abs_m": float(np.abs(ours - live)[valid].max()),
                "mask": {**decode_note, "known_handle_frac_of_valid": float(known[valid].mean()),
                         "objects": per_object},
                "extrinsics_det": float(np.linalg.det(T[:3, :3])),
                "camera_position_m": T[:3, 3].tolist(),
            }
        agreement = {}
        for obj, per_cam in centroids.items():
            if len(per_cam) >= 2:
                values = np.array(list(per_cam.values()))
                agreement[obj] = float(np.linalg.norm(values - values.mean(axis=0), axis=1).max())
        report["cross_camera_centroid_spread_m"] = agreement
        return report
    finally:
        session.shutdown()


def cmd_dependency(args: argparse.Namespace) -> dict[str, Any]:
    session = _session(args, cameras=(), image=(64, 64))
    decision = args.anchor_waypoint
    report: dict[str, Any] = {
        "gate": "dependency", "task": args.task, "variation": args.variation, "seed": args.seed,
        "decision_waypoint": decision,
        "anchor": f"after waypoint {decision - 1} (prefix replayed open loop from reset; accepted replay strategy)",
        "candidate_axis": f"waypoint {decision} target offset along its approach direction",
        "continuation_policy": ("icgs mirror of the RoboHiMan scripted expert with per-trial Gaussian "
                                f"waypoint jitter sigma={args.jitter_m} m; NOT pi_ref, diagnostic only"),
        "local_success_predicate": f"{args.local_predicate} after waypoint {decision} completes",
        "candidates": [],
    }
    try:
        state0 = _warm_lineage(session, args)
        # Reference prefix up to the anchor, executed once by the expert.
        session.reset(args.variation, lineage=state0)
        monitor = build_monitor(session.task, session.task_env, args.variation)
        prefix = StepRecorder(session, monitor, frame_stride=0)
        with prefix:
            prefix.arm()
            with expert_control(session):
                WaypointExpert(session, prefix).run(stop_after=decision - 1)
            prefix.disarm()
        prefix_arrays = prefix.arrays()
        anchor = prefix_arrays["cmd_phase"].shape[0]
        waypoints = session.task_obj.get_waypoints()
        approach = waypoints[decision]._waypoint.get_position() - waypoints[decision - 1]._waypoint.get_position()
        direction = approach / np.linalg.norm(approach)
        report["approach_direction"] = direction.tolist()
        report["anchor_boundary"] = int(anchor)
        later = list(range(decision + 1, len(waypoints)))
        rng = np.random.default_rng(args.seed)
        anchor_states = []
        for delta in args.deltas:
            offset = {"family": "execution_pose_offset", "waypoint": decision, "delta_m": (direction * delta).tolist()}
            trials = []
            for trial in range(args.trials):
                jitter = [{"family": "execution_pose_offset", "waypoint": w,
                           "delta_m": rng.normal(0.0, args.jitter_m, 3).tolist()} for w in later]
                session.reset(args.variation, lineage=state0)
                trial_monitor = build_monitor(session.task, session.task_env, args.variation)
                rec = StepRecorder(session, trial_monitor, frame_stride=0)
                with rec:
                    rec.arm()
                    with expert_control(session):
                        rec.phase = "replay"
                        replay_commands(session, prefix_arrays, 0, anchor)
                        result = WaypointExpert(session, rec, perturbations=[offset] + jitter).run(
                            start=decision, initial_step=False)
                    rec.disarm()
                arrays = rec.arrays()
                anchor_states.append(arrays["step_object_poses"][anchor][:, :3])
                names = trial_monitor.predicate_names
                after = np.flatnonzero(arrays["cmd_waypoint"] > decision)
                local_boundary = int(after[0]) if after.size else arrays["cmd_phase"].shape[0]
                joint_index = rec.joint_names.index(trial_monitor.tracked_joints[0].get_name())
                trials.append({
                    "trial": trial,
                    "prefix_replay_tip_drift_m": float(np.linalg.norm(
                        arrays["step_tip_pose"][anchor][:3] - prefix_arrays["step_tip_pose"][anchor][:3])),
                    "local_boundary": local_boundary,
                    "drawer_joint_after_decision_m": float(arrays["step_task_joint_positions"][local_boundary][joint_index]),
                    "local_success": bool(arrays["step_predicates"][local_boundary][names.index(args.local_predicate)]),
                    "final_status": result.status, "final_reason": result.reason,
                    "final_success": bool(result.task_success),
                    "final_predicates": {n: bool(v) for n, v in zip(names, arrays["step_predicates"][-1])},
                    "steps": int(arrays["cmd_phase"].shape[0]),
                })
            local = [t["local_success"] for t in trials]
            report["candidates"].append({
                "delta_m": delta, "trials": trials,
                "local_success_rate": float(np.mean(local)),
                "final_success_rate": float(np.mean([t["final_success"] for t in trials])),
                "final_success_rate_given_local": (float(np.mean([t["final_success"] for t in trials
                                                                  if t["local_success"]])) if any(local) else None),
            })
        spread = np.stack(anchor_states)
        report["anchor_object_spread_across_trials_m"] = float(np.linalg.norm(spread - spread[0], axis=-1).max())
        locals_ok = [c for c in report["candidates"] if c["local_success_rate"] == 1.0]
        rates = [c["final_success_rate"] for c in locals_ok]
        report["consequential_gap"] = (max(rates) - min(rates)) if len(rates) >= 2 else None
        report["interpretation_rule"] = ("consequential if >=2 candidates are locally successful in every trial "
                                         "and their downstream success rates differ by >=0.5; with n<=5 trials "
                                         "this is a screening signal, not an estimate")
        report["consequential"] = bool(report["consequential_gap"] is not None and report["consequential_gap"] >= 0.5)
        return report
    finally:
        session.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gate", choices=("parity", "replay", "a0", "dependency"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--strategy", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--anchor-waypoint", type=int, default=3)
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--deltas", type=float, nargs="+", default=(0.0, -0.05, -0.08, -0.11))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--jitter-m", type=float, default=0.005)
    parser.add_argument("--local-predicate", default="drawer_open")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    report = {"parity": cmd_parity, "replay": cmd_replay, "a0": cmd_a0, "dependency": cmd_dependency}[args.gate](args)
    report["wall_s"] = round(time.time() - started, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(canonical_json(report))
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
