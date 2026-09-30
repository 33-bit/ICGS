"""Build the reference-only Stage-1 views of one dataset split and record leakage checks.

No simulator is needed. Writes ``<root>/<split>/views/{D_geom,D_dyn,D_task}.jsonl``
+ ``manifest.json`` and merges the split's pairing/leakage summary into
``<root>/reports/leakage.json``. Any query/context violation raises and
leaves the report untouched.

  python -B scripts/robohiman_views.py --root datasets/robohiman --split train
"""

from __future__ import annotations

import argparse
import json
import sys

from icgs.data.stage1.dataset import Dataset
from icgs.data.stage1.views import build_views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="datasets/robohiman")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--contexts-per-query", type=int, default=1)
    parser.add_argument("--pairing-seed", type=int, default=0)
    args = parser.parse_args(argv)
    dataset = Dataset.open(args.root)
    summary = build_views(dataset.split_dir(args.split), contexts_per_query=args.contexts_per_query,
                          pairing_seed=args.pairing_seed)
    report_path = dataset.root / "reports" / "leakage.json"
    report = json.loads(report_path.read_text())
    splits = report.get("splits", {})
    splits[args.split] = {**summary["leakage"], "view_rows": {k: v["rows"] for k, v in summary["views"].items()},
                          "view_sha256": {k: v["sha256"] for k, v in summary["views"].items()}}
    dataset.update_report("leakage.json", {"status": "views built", "kind": report.get("kind"),
                                           "split_id": dataset.manifest["split_id"], "splits": splits})
    print(json.dumps(splits[args.split]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
