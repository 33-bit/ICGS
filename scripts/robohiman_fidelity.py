"""Anchor-level replay fidelity for RoboHiMan (run in the simulator venv under Xvfb).

One reference expert run starts from a post-reset snapshot S0. After every
waypoint an anchor snapshot is captured and its contact state recorded. For
each anchor, strategies reproduce the anchor and replay the *recorded*
reference commands open loop to the end, ``--repeats`` times:

  anchor_restore     configuration-tree snapshot at the anchor
  anchor_regrip      anchor_restore, then the gripper finger motors are driven
                     closed for the anchor's first replayed step if the gripper
                     was closed (an explicit, recorded deviation: tests whether
                     lost grip force is the failure mode)
  contact_free       restore the latest contact-free snapshot at/before the
                     anchor, then replay the commands in between
  reset_snapshot     restore S0 and replay from boundary 0

Metrics per (anchor, strategy, repeat): anchor-state error (EE, joints,
objects, articulation, predicates, grasp set, depth) and continuation drift
(max/final EE, joints, objects, articulation, predicate-trace disagreement,
final outcome, final depth). Repeats of one strategy give its own floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.store import canonical_json
from icgs.environments.robohiman.expert import WaypointExpert, expert_control
from icgs.environments.robohiman.monitors import build_monitor
from icgs.environments.robohiman.recorder import StepRecorder
from icgs.environments.robohiman.snapshot import capture_snapshot, replay_commands, restore_snapshot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robohiman_gates import _depth_diff, _final_depth, _session, _warm_lineage  # noqa: E402


# Proposed acceptance for anchors used in branch collection (see README).
ANCHOR_STATE_LIMITS = {"ee_m": 1e-3, "joint_rad": 2e-3, "object_m": 2e-3, "articulation_m": 2e-3,
                       "depth_mean_abs_m": 1e-3}
CONTINUATION_LIMITS = {"predicate_disagreement": 0.01, "final_object_m": 0.01, "final_articulation_m": 0.01}


def _state_error(ref: dict[str, np.ndarray], b_ref: int, arr: dict[str, np.ndarray], b: int) -> dict[str, Any]:
    return {
        "ee_m": float(np.linalg.norm(ref["step_tip_pose"][b_ref, :3] - arr["step_tip_pose"][b, :3])),
        "joint_rad": float(np.abs(ref["step_joint_positions"][b_ref] - arr["step_joint_positions"][b]).max()),
        "object_m": float(np.linalg.norm(ref["step_object_poses"][b_ref][:, :3] - arr["step_object_poses"][b][:, :3],
                                         axis=-1).max()),
        "articulation_m": float(np.abs(ref["step_task_joint_positions"][b_ref]
                                       - arr["step_task_joint_positions"][b]).max(initial=0.0)),
        "predicates_equal": bool(np.array_equal(ref["step_predicates"][b_ref], arr["step_predicates"][b])),
        "grasp_set_equal": bool(np.array_equal(ref["step_grasped"][b_ref], arr["step_grasped"][b])),
    }


def _continuation(ref: dict[str, np.ndarray], a0: int, arr: dict[str, np.ndarray], b0: int) -> dict[str, Any]:
    length = min(ref["cmd_phase"].shape[0] - a0, arr["cmd_phase"].shape[0] - b0)
    window = lambda x, s, k: x[k][s:s + length + 1]  # noqa: E731
    ee = np.linalg.norm(window(ref, a0, "step_tip_pose")[:, :3] - window(arr, b0, "step_tip_pose")[:, :3], axis=-1)
    obj = np.linalg.norm(window(ref, a0, "step_object_poses")[..., :3] - window(arr, b0, "step_object_poses")[..., :3],
                         axis=-1).max(axis=-1)
    art = np.abs(window(ref, a0, "step_task_joint_positions") - window(arr, b0, "step_task_joint_positions"))
    art = art.max(axis=-1) if art.shape[-1] else np.zeros(length + 1)
    joints = np.abs(window(ref, a0, "step_joint_positions") - window(arr, b0, "step_joint_positions")).max(axis=-1)
    pred = window(ref, a0, "step_predicates") != window(arr, b0, "step_predicates")
    return {
        "steps": int(length),
        "max_ee_m": float(ee.max()), "final_ee_m": float(ee[-1]),
        "max_joint_rad": float(joints.max()),
        "max_object_m": float(obj.max()), "final_object_m": float(obj[-1]),
        "max_articulation_m": float(art.max()), "final_articulation_m": float(art[-1]),
        "predicate_disagreement": float(pred.mean()),
        "final_success_equal": bool(ref["step_task_success"][a0 + length] == arr["step_task_success"][b0 + length]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--strategy", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-anchors", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    session = _session(args, cameras=("front", "left_shoulder"), image=(64, 64))
    report: dict[str, Any] = {"task": args.task, "seed": args.seed, "repeats": args.repeats,
                              "anchor_state_limits": ANCHOR_STATE_LIMITS,
                              "continuation_limits": CONTINUATION_LIMITS, "anchors": []}
    try:
        lineage = _warm_lineage(session, args)
        session.reset(args.variation, lineage=lineage)
        s0 = capture_snapshot(session, 0)
        monitor = build_monitor(session.task, session.task_env, args.variation)
        n_waypoints = len(session.task_obj.get_waypoints())
        ref_rec = StepRecorder(session, monitor, frame_stride=0)
        anchors: list[dict[str, Any]] = []
        with ref_rec:
            ref_rec.arm()
            with expert_control(session):
                previous = -1
                for k in range(n_waypoints):
                    result = WaypointExpert(session, ref_rec).run(start=previous + 1, stop_after=k,
                                                                  initial_step=(k == 0))
                    if result.status != "anchor":
                        break
                    previous = k
                    gripper = session.robot.gripper
                    anchors.append({
                        "after_waypoint": k,
                        "boundary": len(ref_rec.steps) - 1,
                        "grasped": [o.get_name() for o in gripper.get_grasped_objects()],
                        "gripper_open_amount": float(min(gripper.get_open_amount())),
                        "snapshot": capture_snapshot(session, len(ref_rec.steps) - 1),
                        "depth": _final_depth(session),
                    })
                final = WaypointExpert(session, ref_rec).run(start=previous + 1, initial_step=False)
            ref_rec.disarm()
        ref = ref_rec.arrays()
        end = ref["cmd_phase"].shape[0]
        ref_final_depth = _final_depth(session)
        report["reference"] = {"steps": int(end), "status": final.status, "task_success": final.task_success,
                               "waypoints": n_waypoints}
        for anchor in anchors:
            anchor["contact_free"] = not anchor["grasped"] and anchor["gripper_open_amount"] > 0.95
        chosen = anchors if len(anchors) <= args.max_anchors else [
            anchors[int(i)] for i in np.linspace(0, len(anchors) - 1, args.max_anchors).round()]

        def depth_now() -> dict[str, np.ndarray] | None:
            try:
                return _final_depth(session)
            except RuntimeError:  # no joint-force value right after a restore
                return None

        def replay_from(snapshot: Any, start: int, anchor_b: int, *, regrip: bool = False) -> dict[str, Any]:
            restore_snapshot(session, snapshot)
            session.scene._has_init_episode = True
            rec = StepRecorder(session, monitor, frame_stride=0)
            with rec:
                rec.arm()
                with expert_control(session):
                    rec.phase = "replay"
                    if regrip:
                        for joint in session.robot.gripper.joints:
                            joint.set_joint_target_velocity(-0.04)
                    replay_commands(session, ref, start, anchor_b)
                    anchor_depth = depth_now()
                    replay_commands(session, ref, anchor_b, end)
                rec.disarm()
            return {"arrays": rec.arrays(), "start": start, "anchor_depth": anchor_depth,
                    "final_depth": _final_depth(session)}

        reset_runs = {anchor["boundary"]: [] for anchor in chosen}
        for _ in range(args.repeats):
            # One reset replay per repeat, rendering depth at every chosen anchor.
            restore_snapshot(session, s0)
            session.scene._has_init_episode = True
            rec = StepRecorder(session, monitor, frame_stride=0)
            depths = {}
            with rec:
                rec.arm()
                with expert_control(session):
                    rec.phase = "replay"
                    cursor = 0
                    for anchor in chosen:
                        replay_commands(session, ref, cursor, anchor["boundary"])
                        cursor = anchor["boundary"]
                        depths[cursor] = depth_now()
                    replay_commands(session, ref, cursor, end)
                rec.disarm()
            arrays, final_depth = rec.arrays(), _final_depth(session)
            for anchor in chosen:
                reset_runs[anchor["boundary"]].append({"arrays": arrays, "start": 0,
                                                       "anchor_depth": depths[anchor["boundary"]],
                                                       "final_depth": final_depth})
        for anchor in chosen:
            b = anchor["boundary"]
            free = [a for a in anchors if a["boundary"] <= b and a["contact_free"]]
            source = free[-1] if free else {"snapshot": s0, "boundary": 0}
            entry: dict[str, Any] = {k: anchor[k] for k in ("after_waypoint", "boundary", "grasped",
                                                              "gripper_open_amount", "contact_free")}
            entry["strategies"] = {}
            plans: dict[str, Any] = {
                "anchor_restore": lambda: replay_from(anchor["snapshot"], b, b),
                "contact_free": lambda: replay_from(source["snapshot"], source["boundary"], b),
                "reset_snapshot": None,
            }
            if not anchor["contact_free"]:
                plans["anchor_regrip"] = lambda: replay_from(anchor["snapshot"], b, b,
                                                             regrip=anchor["gripper_open_amount"] < 0.95)
            for name, plan in plans.items():
                runs = reset_runs[b] if plan is None else [plan() for _ in range(args.repeats)]
                repeats = []
                for run in runs:
                    arrays, local_b = run["arrays"], b - run["start"]
                    state = _state_error(ref, b, arrays, local_b)
                    state["depth"] = (_depth_diff(anchor["depth"], run["anchor_depth"])
                                      if run["anchor_depth"] is not None else None)
                    cont = _continuation(ref, b, arrays, local_b)
                    cont["final_depth"] = _depth_diff(ref_final_depth, run["final_depth"])
                    repeats.append({"anchor_state": state, "continuation": cont})
                first, second = runs[0], runs[min(1, len(runs) - 1)]
                floor = _continuation(first["arrays"], b - first["start"], second["arrays"], b - second["start"])
                worst_state = {key: max(r["anchor_state"][key] for r in repeats)
                               for key in ("ee_m", "joint_rad", "object_m", "articulation_m")}
                depth_values = [max(v for k, v in r["anchor_state"]["depth"].items() if k.endswith("mean_abs_m"))
                                for r in repeats if r["anchor_state"]["depth"]]
                worst_state["depth_mean_abs_m"] = max(depth_values) if depth_values else None
                worst_cont = {key: max(r["continuation"][key] for r in repeats)
                              for key in ("predicate_disagreement", "final_object_m", "final_articulation_m",
                                          "max_ee_m")}
                all_equal = all(r["anchor_state"]["predicates_equal"] and r["anchor_state"]["grasp_set_equal"]
                                for r in repeats)
                outcome_equal = all(r["continuation"]["final_success_equal"] for r in repeats)
                entry["strategies"][name] = {
                    "source_boundary": runs[0]["start"], "replayed_steps_to_anchor": b - runs[0]["start"],
                    "worst_anchor_state": worst_state, "anchor_predicates_and_grasps_equal": all_equal,
                    "worst_continuation": worst_cont, "all_outcomes_equal": outcome_equal,
                    "repeat_floor": floor, "repeats": repeats,
                    "anchor_accept": bool(all_equal and all(
                        worst_state[k] is not None and worst_state[k] <= ANCHOR_STATE_LIMITS[k]
                        for k in ANCHOR_STATE_LIMITS if not (k == "depth_mean_abs_m" and worst_state[k] is None))),
                    "continuation_accept": bool(outcome_equal and all(
                        worst_cont[k] <= CONTINUATION_LIMITS[k] for k in CONTINUATION_LIMITS)),
                }
            report["anchors"].append(entry)
        report["wall_s"] = round(time.time() - started, 1)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(canonical_json(report))
        summary = [{"wp": a["after_waypoint"], "b": a["boundary"], "contact_free": a["contact_free"],
                    **{name: (s["anchor_accept"], s["continuation_accept"]) for name, s in a["strategies"].items()}}
                   for a in report["anchors"]]
        print(json.dumps({"task": args.task, "reference": report["reference"], "anchors": summary}))
        return 0
    finally:
        session.shutdown()


if __name__ == "__main__":
    sys.exit(main())
