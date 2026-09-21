"""Keepalive/watchdog for one coordinator and 200 worker slots."""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    args = parser.parse_args()
    root = Path(args.run_root)
    while True:
        status = root / "control" / "coordinator-heartbeat.json"
        if status.is_file():
            text = status.read_text(encoding="utf-8")
            if '"status": "COMPLETE"' in text or '"status": "FAILED"' in text:
                return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
