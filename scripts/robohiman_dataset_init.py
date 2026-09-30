"""Create (or re-open) the RoboHiMan dataset root bound to the frozen split.

No simulator is needed. Writes ``manifest.json``, a byte copy of the locked
split as ``splits.json``, the split directories, ``preprocessing/index.json``
and the report skeletons; ``reports/reproducibility.json`` records the
upstream pins and the frozen split identity, ``reports/audit.json`` points at
the committed pre-flight compatibility summary.

  python -B scripts/robohiman_dataset_init.py --root datasets/robohiman
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from icgs.data.stage1.dataset import Dataset
from icgs.data.stage1.labels import LABEL_PROTOCOL_ID
from icgs.data.stage1.schema import SCHEMA_VERSION
from icgs.environments.robohiman.pins import GLOBAL_QUIRKS, UPSTREAM

SPLIT_MANIFEST = Path("artifacts/robohiman/icgs_robohiman_stage1_split_v1.json")
COMPAT_SUMMARY = Path("docs/experiments/robohiman-validation/v3/compat-summary.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="datasets/robohiman")
    parser.add_argument("--split-manifest", default=str(SPLIT_MANIFEST))
    args = parser.parse_args(argv)
    fresh = not (Path(args.root) / "manifest.json").exists()
    dataset = Dataset.create(args.root, args.split_manifest, reproducibility={
        "upstream_pins": {name: dict(value) for name, value in UPSTREAM.items()},
        "episode_schema_version": SCHEMA_VERSION,
        "label_protocol": LABEL_PROTOCOL_ID,
        "global_upstream_quirks": list(GLOBAL_QUIRKS),
        "init_python": platform.python_version(),
        "determinism": "per-episode numpy MT19937 state and Colosseum factor RNG states are stored in every "
                       "episode; canonical-start episodes are replayable, native-start episodes carry hidden "
                       "engine state (ADR 0016)",
    })
    if fresh and COMPAT_SUMMARY.is_file():
        dataset.update_report("audit.json", {
            "status": "pre-flight only; collection audit pending",
            "preflight_compat_summary": str(COMPAT_SUMMARY),
            "preflight_compat_summary_sha256": hashlib.sha256(COMPAT_SUMMARY.read_bytes()).hexdigest(),
        })
    print(json.dumps({"root": str(dataset.root), "created": fresh, "split_id": dataset.manifest["split_id"],
                      "split_manifest_sha256": dataset.manifest["split_manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
