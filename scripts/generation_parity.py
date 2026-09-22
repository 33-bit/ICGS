"""Compile-only phase-1 parity check. No simulator, no publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.collection.generation.report import program_parity_report, training_eligible_ids


def main() -> int:
    compiled = compile_generation_catalog()
    report = program_parity_report(compiled)
    payload = {
        "dataset_version": GENERATION_PROTOCOL.dataset_version,
        "phase": GENERATION_PROTOCOL.phase,
        "pilot_program_ids": list(GENERATION_PROTOCOL.pilot_program_ids),
        "views": list(GENERATION_PROTOCOL.views),
        "all_match": report["all_match"],
        "training_eligible": training_eligible_ids(compiled),
        "pilot": {pid: report["programs"][pid] for pid in GENERATION_PROTOCOL.pilot_program_ids},
    }
    print(json.dumps(payload, indent=2))
    if not report["all_match"]:
        return 1
    missing = [pid for pid in GENERATION_PROTOCOL.pilot_program_ids if pid not in payload["training_eligible"]]
    if missing:
        print("pilot programs not training-eligible:", missing, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
