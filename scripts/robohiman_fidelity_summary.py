"""Apply the proposed branch-anchor acceptance rule to robohiman_fidelity.py reports.

Proposed after the 2026-09-30 measurements (not pre-declared): an anchor is
reliable for approximate branch collection with a given reproduction strategy
if, over every repeat,

  anchor state   EE <= 1.5 mm, arm joints <= 2e-3 rad, object and articulation
                 positions <= 2 mm (granular particles excluded, see below),
                 raw predicates and grasp set identical, depth mean |d| <= 2 mm
  continuation   (open-loop replay of the recorded commands) all final outcomes
                 equal, predicate-trace disagreement <= 1 %, final object and
                 articulation drift <= max(10 mm, 2 x the strategy's own
                 repeat floor)

Granular particles (``dirt*``) are excluded from the object terms: identical
repeats of the same strategy already disagree by 1-3 cm on them; their effect
is carried by the predicates (counts in the dustpan), which must agree.

Anchor classes: ``free`` (gripper open, nothing attached), ``kinematic_grasp``
(an object is attached by Gripper.grasp), ``physical_grip`` (gripper closed on
something without attachment, e.g. a drawer handle).

  python -B scripts/robohiman_fidelity_summary.py report.json [...] --out summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RULE = {"ee_m": 1.5e-3, "joint_rad": 2e-3, "object_m": 2e-3, "articulation_m": 2e-3, "depth_mean_abs_m": 2e-3,
        "predicate_disagreement": 0.01, "final_drift_floor_m": 0.01, "floor_factor": 2.0}


def anchor_class(anchor: dict[str, Any]) -> str:
    if anchor["grasped"]:
        return "kinematic_grasp"
    return "free" if anchor["gripper_open_amount"] > 0.95 else "physical_grip"


def evaluate(strategy: dict[str, Any], granular_task: bool) -> dict[str, Any]:
    reasons = []
    for repeat in strategy["repeats"]:
        state, cont = repeat["anchor_state"], repeat["continuation"]
        if state["ee_m"] > RULE["ee_m"]:
            reasons.append("anchor_ee")
        if state["joint_rad"] > RULE["joint_rad"]:
            reasons.append("anchor_joint")
        if not granular_task and state["object_m"] > RULE["object_m"]:
            reasons.append("anchor_object")
        if state["articulation_m"] > RULE["articulation_m"]:
            reasons.append("anchor_articulation")
        if not (state["predicates_equal"] and state["grasp_set_equal"]):
            reasons.append("anchor_predicates_or_grasp")
        depth = state.get("depth")
        if depth and max(v for k, v in depth.items() if k.endswith("mean_abs_m")) > RULE["depth_mean_abs_m"]:
            reasons.append("anchor_depth")
        if not cont["final_success_equal"]:
            reasons.append("outcome")
        if cont["predicate_disagreement"] > RULE["predicate_disagreement"]:
            reasons.append("predicate_trace")
        floor = strategy["repeat_floor"]
        for key in ("final_articulation_m",) + (() if granular_task else ("final_object_m",)):
            if cont[key] > max(RULE["final_drift_floor_m"], RULE["floor_factor"] * floor[key]):
                reasons.append(key)
    return {"accept": not reasons, "reasons": sorted(set(reasons))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    table: dict[str, Any] = {"rule": RULE, "tasks": {}, "by_class_and_strategy": {}}
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        granular = report["task"] in ("sweep_to_dustpan", "sweep_and_drop")
        rows = []
        for anchor in report["anchors"]:
            cls = anchor_class(anchor)
            verdicts = {name: evaluate(strategy, granular) for name, strategy in anchor["strategies"].items()}
            rows.append({"after_waypoint": anchor["after_waypoint"], "boundary": anchor["boundary"], "class": cls,
                         "verdicts": verdicts})
            for name, verdict in verdicts.items():
                totals[(cls, name)][0] += int(verdict["accept"])
                totals[(cls, name)][1] += 1
        table["tasks"][report["task"]] = {"reference": report["reference"], "anchors": rows,
                                          "granular_objects_excluded": granular}
    table["by_class_and_strategy"] = {f"{cls}/{name}": {"accepted": ok, "anchors": n}
                                      for (cls, name), (ok, n) in sorted(totals.items())}
    Path(args.out).write_text(json.dumps(table, indent=1, sort_keys=True))
    print(json.dumps(table["by_class_and_strategy"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
