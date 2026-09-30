"""Bounded instrumented RoboHiMan collection.

Runs inside the isolated RoboHiMan simulator venv (see
docs/components/robohiman.md). One invocation = one task/strategy session.
Simulator errors are written as attempt records, never as episodes.

Two destinations:

* ``--dataset ROOT --split S``: the RoboHiMan dataset (``Dataset``). Every
  episode gets its own numpy seed (``--seed + offset``); the split, task,
  strategy, variation, seeds and destination are checked against the frozen
  split before simulation and again at commit. Masks default on.
* ``--store DIR``: a scratch store for gates and smoke checks (no split).

Episode IDs are ``ep-<task>-<run8>-<index>`` with a fresh random ``run8`` per
invocation unless ``--run8`` is given.

Example (smoke scale only):
  xvfb-run -a python -B scripts/robohiman_collect.py --dataset outputs/robohiman/dataset_smoke \
      --split train --task open_drawer --variation 0 --episodes 1 --seed 100000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from icgs.data.stage1.dataset import Dataset, make_episode_id, new_run8
from icgs.data.stage1.store import canonical_json, write_episode
from icgs.environments.robohiman.episode import attempt_record, collect_episode
from icgs.environments.robohiman.pins import NATIVE_CAMERAS
from icgs.environments.robohiman.session import RoboHiManSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--dataset", help="RoboHiMan dataset root (created by robohiman_dataset_init.py)")
    destination.add_argument("--store", help="scratch store for gates/smoke checks (no split)")
    parser.add_argument("--split", choices=("train", "dev", "test"), help="required with --dataset")
    parser.add_argument("--run8", help="8 hex characters; default: random per invocation")
    parser.add_argument("--task", required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--strategy", type=int, default=0, help="Colosseum collection-strategy index (0 = A/C)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True, help="numpy seed; full state is stored per episode")
    parser.add_argument("--env-seed", type=int, default=42, help="Colosseum factor seed (native train=42)")
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--cameras", nargs="+", default=list(NATIVE_CAMERAS))
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="render every k-th physics boundary (2 = 0.1 s model cadence) plus every event boundary")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--perturbation", action="append", default=[],
                        help="JSON object with family/waypoint/... (repeatable)")
    parser.add_argument("--canonical-start", action="store_true",
                        help="restore the post-reset snapshot before execution (branchable episodes; deviation)")
    parser.add_argument("--masks", action=argparse.BooleanOptionalAction, default=None,
                        help="persist foreground mask handles + legend (default: on for --dataset, off for --store)")
    parser.add_argument("--point-cloud", action="store_true")
    parser.add_argument("--max-episodes-guard", type=int, default=20,
                        help="refuse larger runs; bulk collection is gated")
    args = parser.parse_args(argv)
    if args.episodes > args.max_episodes_guard:
        parser.error("episode count exceeds the smoke guard; bulk collection is not authorized")
    if bool(args.dataset) != bool(args.split):
        parser.error("--dataset and --split go together")
    masks = bool(args.dataset) if args.masks is None else args.masks
    run8 = args.run8 or new_run8()
    perturbations = [json.loads(item) for item in args.perturbation]
    dataset = Dataset.open(args.dataset) if args.dataset else None
    provenance: list[dict | None] = []
    for offset in range(args.episodes):
        if dataset is None:
            provenance.append(None)
            continue
        # Refuse before simulating anything: every episode's seed must belong to the split.
        provenance.append(dataset.split_provenance(split=args.split, task=args.task, strategy=args.strategy,
                                                   variation=args.variation, seed=args.seed + offset,
                                                   factor_env_seed=args.env_seed))
    store = dataset.split_dir(args.split) if dataset else Path(args.store)
    (store / "attempts").mkdir(parents=True, exist_ok=True)
    session = RoboHiManSession(args.task, strategy_index=args.strategy, image_size=args.image_size,
                               cameras=args.cameras, env_seed=args.env_seed, masks=masks,
                               point_cloud=args.point_cloud)
    session.launch()
    summary = []
    try:
        for offset in range(args.episodes):
            index = args.first_index + offset
            started = time.time()
            episode_id = make_episode_id(args.task, run8, index)
            np.random.seed(args.seed + offset)  # one seed per episode: lineage is checkable per episode
            try:
                manifest, arrays, runtime = collect_episode(
                    session, episode_id=episode_id, run_id=run8, episode_index=index, variation=args.variation,
                    rng_state=None, perturbations=perturbations, frame_stride=args.frame_stride,
                    max_steps=args.max_steps, masks=masks, point_cloud=args.point_cloud,
                    canonical_start=args.canonical_start)
            except Exception as error:  # simulator/runtime failure: attempt record only
                record = attempt_record(run_id=run8, task=args.task, variation=args.variation,
                                        episode_index=index, error=error)
                record["episode_seed"] = args.seed + offset
                attempt_id = f"att-{args.task}-{run8}-{index:04d}"
                if dataset is not None:
                    dataset.record_attempt(args.split, attempt_id, record)
                else:
                    (store / "attempts" / f"{attempt_id}.json").write_text(canonical_json(record))
                summary.append({"episode_index": index, "outcome": "simulator_error", "error": record["error"]})
                continue
            if dataset is not None:
                manifest["lineage"]["split"] = provenance[offset]
                path = dataset.commit_episode(args.split, manifest, arrays, destination=store)
            else:
                path = write_episode(store, manifest, arrays)
            summary.append({"episode_id": manifest["episode_id"], "outcome": manifest["outcome"]["status"],
                            "reason": manifest["outcome"]["reason"], "steps": manifest["counts"]["steps"],
                            "frames": manifest["counts"]["frames"],
                            "perturbation_families": manifest["perturbation"]["families"],
                            "bytes": manifest_bytes(path), "wall_s": round(time.time() - started, 1)})
            print(json.dumps(summary[-1]), flush=True)
    finally:
        session.shutdown()
    print(json.dumps({"run8": run8, "task": args.task, "episodes": summary}), flush=True)
    return 0


def manifest_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir())


if __name__ == "__main__":
    sys.exit(main())
