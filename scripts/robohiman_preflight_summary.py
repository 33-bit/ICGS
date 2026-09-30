"""Tabulate robohiman_compat.py reports into one pre-flight summary (JSON + Markdown rows).

  python -B scripts/robohiman_preflight_summary.py REPORT_DIR --out summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from icgs.environments.robohiman.monitors import TASK_SPECS
from icgs.environments.robohiman.pins import task_family


def _row(report: dict[str, Any]) -> dict[str, Any]:
    nominal = next((e for e in report.get("episodes", []) if e.get("checks", {}).get("success")), None)
    checks = nominal["checks"] if nominal else {}
    failure = report.get("failure_retention") or {}
    cams = checks.get("cameras_and_pointcloud", {})
    return {
        "task": report["task"],
        "family": task_family(report["task"]),
        "semantic_family": TASK_SPECS.get(report["task"], {}).get("family"),
        "verdict": report.get("verdict"),
        "audited_strategy": report.get("strategy"),
        "reasons": report.get("reasons", []),
        "notes": report.get("notes", []),
        "nominal_attempts": len(report.get("episodes", [])),
        "steps": checks.get("steps"),
        "action_logging_ok": checks.get("action_logging", {}).get("ok"),
        "achieved_logging_ok": all(checks.get("achieved_logging", {}).values()) if checks else None,
        "pointcloud_numpy_vs_pyrep_max_m": max((c["numpy_vs_pyrep_max_abs_m"] for c in cams.get("cameras", {}).values()),
                                               default=None),
        "min_graspable_mask_containment": cams.get("min_graspable_containment"),
        "success_conjunction_match": checks.get("predicates", {}).get("success_conjunction_matches_task_success"),
        "events": checks.get("monitor", {}).get("events"),
        "missing_events": checks.get("monitor", {}).get("missing_events"),
        "failure_retention": {k: failure.get(k) for k in ("status", "reason", "stored", "error") if k in failure},
        "quirks": len(report.get("known_upstream_quirks", [])),
        "wall_s": report.get("wall_s"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = [_row(json.loads(p.read_text())) for p in sorted(Path(args.report_dir).glob("*.json"))]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    Path(args.out).write_text(json.dumps({"counts": counts, "tasks": rows}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(counts))
    for row in rows:
        print(f"| {row['task']} | {row['family']} | {row['verdict']} | {row['steps']} | "
              f"{row['failure_retention'].get('status', row['failure_retention'].get('error', '-'))} | "
              f"{'; '.join(row['reasons']) or '-'} | {'; '.join(row['notes']) or '-'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
