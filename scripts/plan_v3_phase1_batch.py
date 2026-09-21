"""Write the phase-1 pilot attempt plan. No simulator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.v3.batch import plan_pilot_batch
from icgs.data.collection.v3.manifest import build_v3_manifest, write_v3_manifest
from icgs.data.collection.v3.protocol import V3_PROTOCOL


def main() -> int:
    v2 = ROOT / "artifacts/composition/approved_composition_manifest.json"
    v3_path = ROOT / "artifacts/composition/approved_composition_manifest_v3.json"
    write_v3_manifest(v3_path, v2)
    manifest = build_v3_manifest(v2)
    rows = {row["program_id"]: row for row in manifest["catalog"]}
    plans = plan_pilot_batch(rows)
    out = ROOT / "artifacts/composition/v3_phase1_pilot_attempts.json"
    payload = {
        "dataset_version": V3_PROTOCOL.dataset_version,
        "collection_seed": V3_PROTOCOL.collection_seed,
        "n_attempts": len(plans),
        "attempts": [plan.as_dict() for plan in plans],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_attempts": len(plans), "programs": list(V3_PROTOCOL.pilot_program_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
