"""Write the phase-1 pilot attempt plan. No simulator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.generation.batch import plan_pilot_batch
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


def main() -> int:
    manifest_path = ROOT / "artifacts/composition/approved_composition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["program_id"]: row for row in manifest["catalog"]}
    plans = plan_pilot_batch(rows)
    out = ROOT / "artifacts/composition/generation_pilot_attempts.json"
    payload = {
        "dataset_version": GENERATION_PROTOCOL.dataset_version,
        "collection_seed": GENERATION_PROTOCOL.collection_seed,
        "n_attempts": len(plans),
        "attempts": [plan.as_dict() for plan in plans],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_attempts": len(plans), "programs": list(GENERATION_PROTOCOL.pilot_program_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
