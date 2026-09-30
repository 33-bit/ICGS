"""Pre-flight compatibility audit for one RoboHiMan task (simulator venv, Xvfb).

Bounded: at most two nominal episodes (a second seed only if the first expert
run fails) and one perturbed episode (early gripper close at the first
close_gripper waypoint) to exercise failure retention. Episodes are written to
``--store``; the task verdict to ``--out``.

Verdicts:
  PASS         every check passes on a successful nominal episode
  PARTIAL      nominal expert succeeds but a check fails or needs a caveat
  FAIL         no successful nominal episode (reset/expert) or a hard check fails
  UNSUPPORTED  the task cannot be collected faithfully (e.g. strategy disabled
               upstream, no reviewed monitor)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.dataset import make_episode_id, new_run8
from icgs.data.stage1.labels import label_consistency_report
from icgs.data.stage1.store import canonical_json, write_episode
from icgs.environments.robohiman.camera import decode_mask_handles, depth_to_world, metric_depth, valid_depth_mask
from icgs.environments.robohiman.episode import collect_episode
from icgs.environments.robohiman.monitors import SUPPORTED_TASKS
from icgs.environments.robohiman.pins import NATIVE_CAMERAS, quirks_for
from icgs.environments.robohiman.session import RoboHiManSession


def _world_box(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    bbox = np.asarray(shape.get_bounding_box(), dtype=np.float64)
    corners = np.array([[x, y, z] for x in bbox[0:2] for y in bbox[2:4] for z in bbox[4:6]])
    matrix = np.asarray(shape.get_matrix(), dtype=np.float64)
    if matrix.size == 12:
        matrix = np.vstack([matrix.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]])
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def _geometry_check(session: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Last stored frame vs the live simulator state at the same (final) boundary."""
    from pyrep.objects.vision_sensor import VisionSensor
    from pyrep.const import ObjectType

    frame = arrays["frame_step"].shape[0] - 1
    shapes = {s.get_handle(): s for s in session.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)}
    graspable = {o.get_name() for o in session.task_obj.get_graspable_objects()}
    bounds = np.asarray(session.workspace_bounds())
    out: dict[str, Any] = {"cameras": {}}
    containment = []
    for name in session.cameras:
        depth = arrays[f"cam_{name}_depth"][frame].astype(np.float64)
        near, far = float(arrays[f"cam_{name}_near"][frame]), float(arrays[f"cam_{name}_far"][frame])
        K, T = arrays[f"cam_{name}_intrinsics"][frame], arrays[f"cam_{name}_extrinsics"][frame]
        valid = valid_depth_mask(depth)
        ours = depth_to_world(metric_depth(depth, near, far), T, K)
        theirs = VisionSensor.pointcloud_from_depth_and_camera_params(near + depth * (far - near), T, K)
        inside = np.all((ours >= bounds[0] - 0.05) & (ours <= bounds[1] + 0.05), axis=-1)
        cam = {"numpy_vs_pyrep_max_abs_m": float(np.abs(ours - theirs).max()),
               "valid_frac": float(valid.mean()),
               "valid_in_workspace_5cm_frac": float(inside[valid].mean()) if valid.any() else 0.0,
               "calibration_ok": bool(near > 0 and far > near and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-4
                                      and K[2, 2] == 1.0)}
        key = f"cam_{name}_mask_handles"
        if key in arrays:
            handles = arrays[key][frame]
            per_object = {}
            for handle in np.unique(handles[valid]):
                shape = shapes.get(int(handle))
                if shape is None or shape.get_name() not in graspable:
                    continue
                pixels = valid & (handles == handle)
                if pixels.sum() < 5:
                    continue
                low, high = _world_box(shape)
                frac = float(np.all((ours[pixels] >= low - 0.01) & (ours[pixels] <= high + 0.01), axis=1).mean())
                per_object[shape.get_name()] = {"pixels": int(pixels.sum()), "inside_box_1cm": frac}
                containment.append(frac)
            cam["graspable_mask_containment"] = per_object
        out["cameras"][name] = cam
    out["min_graspable_containment"] = min(containment) if containment else None
    return out


def _checks(session: Any, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    names = manifest["predicates"]["names"]
    success_cols = [i for i, n in enumerate(names) if n.startswith("success_condition_")]
    trace = arrays["step_predicates"]
    conj = trace[:, success_cols].all(axis=1) if success_cols else np.zeros(trace.shape[0], bool)
    target = arrays["cmd_arm_joint_target_valid"]
    tracking = np.abs(arrays["cmd_arm_joint_target"][target] - arrays["step_joint_positions"][1:][target])
    specs = manifest["events"]["specs"]
    first = arrays["monitor_first_occurrence"]
    success = manifest["outcome"]["status"] == "success"
    missing_events = [spec["event_id"] for j, spec in enumerate(specs) if first[j] < 0]
    geometry = _geometry_check(session, arrays)
    cams_ok = all(c["calibration_ok"] and c["numpy_vs_pyrep_max_abs_m"] < 1e-6
                  for c in geometry["cameras"].values())
    return {
        "reset_ok": True,
        "expert_status": manifest["outcome"]["status"],
        "steps": manifest["counts"]["steps"], "frames": manifest["counts"]["frames"],
        "action_logging": {"arm_target_rows": int(target.sum()),
                           "gripper_rows": int(arrays["cmd_gripper_joint_velocity_valid"].any(axis=1).sum()),
                           "grasp_events": int((arrays["cmd_grasp_event"] != 0).sum()),
                           "ok": bool(target.sum() > 0 and arrays["cmd_gripper_joint_velocity_valid"].any())},
        "achieved_logging": {"finite": bool(np.isfinite(arrays["step_joint_positions"]).all()
                                            and np.isfinite(arrays["step_tip_pose"]).all()),
                             "command_differs_from_achieved": bool(tracking.size and tracking.max() > 0),
                             # CoppeliaSim keeps simulation time in float32; allow its spacing at t_end.
                             "sim_dt_constant": bool(np.allclose(
                                 np.diff(arrays["step_sim_time"]), manifest["controller"]["physics_dt"],
                                 atol=float(np.spacing(np.float32(arrays["step_sim_time"][-1]))) + 1e-9))},
        "cameras_and_pointcloud": {**geometry, "ok": bool(cams_ok and len(geometry["cameras"]) == len(session.cameras))},
        "predicates": {"success_conjunction_matches_task_success": float((conj == arrays["step_task_success"]).mean()),
                       "any_edge": bool((np.diff(trace.astype(np.int8), axis=0) != 0).any())},
        "monitor": {"events": [s["event_id"] for s in specs], "missing_events": missing_events,
                    "consistency": label_consistency_report(
                        {k[8:]: v for k, v in arrays.items() if k.startswith("monitor_")}, specs),
                    "all_events_occurred": not missing_events,
                    "final_monitor_event_id_null": bool(arrays["monitor_event_id"][-1] == len(specs))},
        "provenance": {"reset_rng_sha256": manifest["lineage"]["reset_rng"]["sha256"],
                       "factor_states_recorded": bool(manifest["lineage"].get("benchmark_factor_rng_before_reset")
                                                      is not None),
                       "code_revision": manifest["code"].get("icgs_revision"),
                       "robohiman_revision": manifest["environment"]["robohiman_checkout"].get("revision"),
                       "task_ttm_sha256": manifest["environment"]["task_ttm_sha256"],
                       "quirks_recorded": len(manifest["environment"]["known_upstream_quirks_preserved"])},
        "success": success,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run8", default=None, help="8 hex characters; default: random")
    args = parser.parse_args(argv)
    run8 = args.run8 or new_run8()
    started = time.time()
    report: dict[str, Any] = {"task": args.task, "known_upstream_quirks": quirks_for(args.task), "episodes": []}
    if args.task not in SUPPORTED_TASKS:
        report.update(verdict="UNSUPPORTED", reasons=["no reviewed monitor"])
        Path(args.out).write_text(canonical_json(report))
        return 0
    strategy = 0
    reasons: list[str] = []
    notes: list[str] = []
    session = RoboHiManSession(args.task, strategy_index=0, cameras=NATIVE_CAMERAS, masks=True)
    try:
        session.build_config()
    except ValueError:
        # Strategy 0 disabled upstream: audit the first enabled perturbation
        # strategy that is not withheld from TRAIN, and cap the verdict.
        import colosseum

        family = session.family.upper()
        json_dir = getattr(colosseum, f"ASSETS_{family}_JSON_FOLDER")
        strategies = json.loads(Path(json_dir, f"{args.task}.json").read_text())["strategy"]
        withheld = {"distractor", "background_texture", "all_mixed"}
        strategy = next(s["spreadsheet_idx"] for s in strategies if s["enabled"] and s["variation_name"] not in withheld)
        report["strategy_0_disabled_upstream"] = True
        reasons.append(f"strategy 0 disabled upstream; audited on AP strategy {strategy} only")
        session = RoboHiManSession(args.task, strategy_index=strategy, cameras=NATIVE_CAMERAS, masks=True)
        session.build_config()
    try:
        session.launch()
        report["strategy"] = strategy
        nominal = None
        for attempt, seed in enumerate((args.seed, args.seed + 1)):
            np.random.seed(seed)
            try:
                manifest, arrays, _ = collect_episode(session, episode_id=make_episode_id(args.task, run8, attempt),
                                                      run_id=run8, episode_index=attempt,
                                                      variation=0, rng_state=None, frame_stride=8, masks=True)
            except Exception as error:  # reset/simulator failure
                report["episodes"].append({"kind": "nominal", "seed": seed, "error": f"{type(error).__name__}: {error}"[:300]})
                continue
            write_episode(args.store, manifest, arrays)
            checks = _checks(session, manifest, arrays)
            report["episodes"].append({"kind": "nominal", "seed": seed, "episode_id": manifest["episode_id"],
                                       "checks": checks})
            if checks["success"]:
                nominal = checks
                break
        waypoints = session.task_obj.get_waypoints()
        close_index = next((i for i, w in enumerate(waypoints) if "close_gripper(" in w.get_ext()), None)
        report["first_close_gripper_waypoint"] = close_index
        failure = None
        if close_index is not None:
            np.random.seed(args.seed + 7)
            try:
                manifest, arrays, _ = collect_episode(
                    session, episode_id=make_episode_id(args.task, run8, 9), run_id=run8, episode_index=9,
                    variation=0, rng_state=None, frame_stride=8,
                    masks=True, perturbations=[{"family": "gripper_timing", "waypoint": close_index,
                                                "close_early_at_distance_m": 0.05}])
                write_episode(args.store, manifest, arrays)
                failure = {"episode_id": manifest["episode_id"], "status": manifest["outcome"]["status"],
                           "reason": manifest["outcome"]["reason"],
                           "perturbation_families": manifest["perturbation"]["families"],
                           "stored": True}
            except Exception as error:
                failure = {"error": f"{type(error).__name__}: {error}"[:300], "stored": False}
        report["failure_retention"] = failure
        if nominal is None:
            report["verdict"] = "FAIL"
            reasons.append("no successful nominal expert episode in 2 seeds")
        else:
            if not nominal["action_logging"]["ok"]:
                reasons.append("action logging")
            if not all(nominal["achieved_logging"].values()):
                reasons.append("achieved-state logging")
            if not nominal["cameras_and_pointcloud"]["ok"]:
                reasons.append("camera/point-cloud")
            containment = nominal["cameras_and_pointcloud"]["min_graspable_containment"]
            if containment is not None and containment < 0.9:
                reasons.append(f"graspable mask-box containment {containment:.2f} < 0.90")
            if nominal["predicates"]["success_conjunction_matches_task_success"] < 1.0:
                reasons.append("success-condition conjunction differs from task.success()")
            if not nominal["monitor"]["all_events_occurred"]:
                reasons.append(f"monitor events not observed in success: {nominal['monitor']['missing_events']}")
            if not nominal["monitor"]["consistency"]["occurrences_in_spec_order"]:
                reasons.append("event occurrences out of spec order")
            if failure is None or not failure.get("stored"):
                reasons.append("perturbed episode could not be collected and stored")
            elif failure["status"] == "success":
                notes.append("early-close perturbation still succeeded (retention path not exercised here)")
            if len(report["episodes"]) > 1:
                notes.append("first nominal expert attempt failed; second seed succeeded")
            report["verdict"] = "PASS" if not reasons else "PARTIAL"
            if report.get("strategy_0_disabled_upstream"):
                report["ap_only_verdict"] = report["verdict"] if len(reasons) > 1 else "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        reasons.append(f"{type(error).__name__}: {error}"[:300])
        report["traceback"] = traceback.format_exc()[-2000:]
    finally:
        session.shutdown()
    report["reasons"] = reasons
    report["notes"] = notes
    report["wall_s"] = round(time.time() - started, 1)
    Path(args.out).write_text(canonical_json(report))
    print(json.dumps({"task": args.task, "verdict": report["verdict"], "reasons": reasons, "notes": notes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
