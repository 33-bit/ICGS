"""Preflight a bounded real-generation capacity probe.

Execution is deliberately explicit: this command validates the probe contract
and prints the staged limits. Operators then invoke the canonical launcher with
one stage at a time; no production quota is started implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from icgs.data.collection.generation.capacity_probe import CapacityProbeConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = CapacityProbeConfig.from_file(args.config)
    print(json.dumps({"status": "PASS", **config.as_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
