"""Bounded instrumented RoboHiMan collection into an icgs_stage1 store.

Runs inside the isolated RoboHiMan simulator venv (see
docs/components/robohiman.md). One invocation = one task/strategy session.
Simulator errors are written as attempt records, never as episodes.

Example (smoke scale only):
  xvfb-run -a python -B scripts/robohiman_collect.py --store outputs/robohiman/store \
      --run-id smoke1 --task open_drawer --variation 0 --episodes 1 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from icgs.data.stage1.store import canonical_json, write_episode
from icgs.environments.robohiman.episode import attempt_record, collect_episode
from icgs.environments.robohiman.pins import NATIVE_CAMERAS
from icgs.environments.robohiman.session import RoboHiManSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--strategy", type=int, default=0, help="Colosseum collection-strategy index (0 = A/C)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True, help="numpy seed; full state is stored per episode")
    parser.add_argument("--env-seed", type=int, default=42, help="Colosseum factor seed (native train=42)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--cameras", nargs="+", default=list(NATIVE_CAMERAS))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--perturbation", action="append", default=[],
                        help="JSON object with family/waypoint/... (repeatable)")
    parser.add_argument("--canonical-start", action="store_true",
                        help="restore the post-reset snapshot before execution (branchable episodes; deviation)")
    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--point-cloud", action="store_true")
    parser.add_argument("--max-episodes-guard", type=int, default=20,
                        help="refuse larger runs; bulk collection is gated")
    args = parser.parse_args(argv)
    if args.episodes > args.max_episodes_guard:
        parser.error("episode count exceeds the smoke guard; bulk collection is not authorized")
    perturbations = [json.loads(item) for item in args.perturbation]
    store = Path(args.store)
    (store / "attempts").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    session = RoboHiManSession(args.task, strategy_index=args.strategy, image_size=args.image_size,
                               cameras=args.cameras, env_seed=args.env_seed, masks=args.masks,
                               point_cloud=args.point_cloud)
    session.launch()
    summary = []
    try:
        for offset in range(args.episodes):
            index = args.first_index + offset
            started = time.time()
            try:
                manifest, arrays, runtime = collect_episode(
                    session, run_id=args.run_id, episode_index=index, variation=args.variation,
                    rng_state=None, perturbations=perturbations, frame_stride=args.frame_stride,
                    max_steps=args.max_steps, masks=args.masks, point_cloud=args.point_cloud,
                    canonical_start=args.canonical_start)
            except Exception as error:  # simulator/runtime failure: attempt record only
                record = attempt_record(run_id=args.run_id, task=args.task, variation=args.variation,
                                        episode_index=index, error=error)
                path = store / "attempts" / f"{args.run_id}-{args.task}-{index:04d}.json"
                path.write_text(canonical_json(record))
                summary.append({"episode_index": index, "outcome": "simulator_error", "error": record["error"]})
                continue
            path = write_episode(store, manifest, arrays)
            summary.append({"episode_id": manifest["episode_id"], "outcome": manifest["outcome"]["status"],
                            "reason": manifest["outcome"]["reason"], "steps": manifest["counts"]["steps"],
                            "frames": manifest["counts"]["frames"],
                            "perturbation_families": manifest["perturbation"]["families"],
                            "bytes": manifest_bytes(path), "wall_s": round(time.time() - started, 1)})
            print(json.dumps(summary[-1]), flush=True)
    finally:
        session.shutdown()
    print(json.dumps({"run_id": args.run_id, "task": args.task, "episodes": summary}), flush=True)
    return 0


def manifest_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir())


if __name__ == "__main__":
    sys.exit(main())
