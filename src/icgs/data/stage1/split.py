"""Frozen Stage-1 split manifests: validation, locking and episode assignment.

A split manifest (JSON) names, per split (``train``/``dev``/``test``), the
tasks, allowed collection strategies and variations, a disjoint numpy-seed
range and the benchmark factor seed. Evaluation tracks refer to split members
and state their intended claim. A lock file stores the manifest's SHA256; a
collector refuses a manifest whose hash differs from its lock, so the split
cannot change silently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")
GRADIENT_SPLITS = ("train",)


class SplitViolation(ValueError):
    pass


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Structural and leakage rules for a frozen split.

    * TRAIN, DEV and TEST own disjoint numpy seed ranges and distinct
      benchmark factor seeds, so no episode lineage can be shared.
    * A task listed by a ``held_out_task`` track never appears in TRAIN, and a
      held-out task belongs to exactly one of DEV/TEST.
    * ``seen_task`` tracks may reuse TRAIN tasks, only under their own split's
      seed range (configuration/perturbation robustness).
    * Every track states its claim and what it does not claim.
    """
    for key in ("split_id", "status", "splits", "tracks", "lineage_rules", "excluded"):
        if key not in manifest:
            raise SplitViolation(f"split manifest missing {key}")
    if manifest["status"] != "frozen":
        raise SplitViolation("only frozen split manifests may drive Stage-1 generation")
    splits = manifest["splits"]
    if set(splits) != set(SPLITS):
        raise SplitViolation(f"splits must be exactly {SPLITS}")
    ranges = []
    for name in SPLITS:
        low, high = splits[name]["numpy_seed_range"]
        if not (isinstance(low, int) and isinstance(high, int) and low <= high):
            raise SplitViolation(f"{name}: bad seed range")
        ranges.append((low, high, name))
    ranges.sort()
    for (_, high, a), (low, _, b) in zip(ranges, ranges[1:]):
        if low <= high:
            raise SplitViolation(f"seed ranges of {a} and {b} overlap")
    env_seeds = [splits[name]["factor_env_seed"] for name in SPLITS]
    if len(set(env_seeds)) != len(env_seeds):
        raise SplitViolation("each split needs its own benchmark factor seed")
    train_tasks = set(splits["train"]["tasks"])
    held_out: dict[str, str] = {}
    for track in manifest["tracks"]:
        split = track["split"]
        if split not in SPLITS:
            raise SplitViolation(f"track {track['track_id']} refers to unknown split {split}")
        if not track.get("claim") or "not_claimed" not in track:
            raise SplitViolation(f"track {track['track_id']} must state its claim and what it does not claim")
        novelty = track.get("task_novelty")
        if novelty not in ("held_out_task", "seen_task"):
            raise SplitViolation(f"track {track['track_id']} needs task_novelty held_out_task|seen_task")
        for task in track["tasks"]:
            if task not in splits[split]["tasks"]:
                raise SplitViolation(f"track {track['track_id']}: {task!r} is not in split {split}")
            if novelty == "held_out_task":
                if task in train_tasks:
                    raise SplitViolation(f"held-out task {task!r} also appears in TRAIN")
                if held_out.setdefault(task, split) != split:
                    raise SplitViolation(f"held-out task {task!r} appears in both {held_out[task]} and {split}")
            elif task not in train_tasks:
                raise SplitViolation(f"seen-task track {track['track_id']} lists non-TRAIN task {task!r}")
    for split in ("dev", "test"):
        for task in splits[split]["tasks"]:
            if task not in train_tasks and task not in held_out:
                raise SplitViolation(f"{split} task {task!r} is neither a TRAIN task nor declared held out")
    excluded = {item["task"] for item in manifest["excluded"]}
    assigned = set().union(*(set(splits[name]["tasks"]) for name in SPLITS))
    if excluded & assigned:
        raise SplitViolation(f"excluded tasks also assigned: {sorted(excluded & assigned)}")


def load_locked_manifest(path: str | Path, lock_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    lock_path = Path(lock_path) if lock_path else path.with_suffix(".lock")
    lock = json.loads(lock_path.read_text())
    actual = manifest_sha256(path)
    if lock.get("manifest_sha256") != actual:
        raise SplitViolation(f"split manifest {path} does not match its lock ({actual} != {lock.get('manifest_sha256')})")
    manifest = json.loads(path.read_text())
    if manifest["split_id"] != lock.get("split_id"):
        raise SplitViolation("split_id differs from lock")
    validate_split_manifest(manifest)
    manifest["_sha256"] = actual
    return manifest


def assert_allowed(manifest: Mapping[str, Any], *, split: str, task: str, strategy: int, variation: int,
                   seed: int, factor_env_seed: int) -> dict[str, Any]:
    """Raise unless (task, strategy, variation, seed) belongs to ``split``; return provenance."""
    if split not in SPLITS:
        raise SplitViolation(f"unknown split {split}")
    entry = manifest["splits"][split]
    if task not in entry["tasks"]:
        raise SplitViolation(f"task {task!r} is not in split {split}")
    rule = entry["tasks"][task]
    if strategy not in rule["strategies"]:
        raise SplitViolation(f"strategy {strategy} not allowed for {task} in {split}")
    if variation not in rule["variations"]:
        raise SplitViolation(f"variation {variation} not allowed for {task} in {split}")
    low, high = entry["numpy_seed_range"]
    if not low <= seed <= high:
        raise SplitViolation(f"seed {seed} outside {split} range [{low}, {high}]")
    if factor_env_seed != entry["factor_env_seed"]:
        raise SplitViolation(f"factor env seed {factor_env_seed} != {entry['factor_env_seed']} for {split}")
    return {"split_id": manifest["split_id"], "split_manifest_sha256": manifest.get("_sha256"),
            "split": split, "gradient_eligible": split in GRADIENT_SPLITS,
            "tracks": [t["track_id"] for t in manifest["tracks"] if t["split"] == split and task in t["tasks"]]}


__all__ = ["GRADIENT_SPLITS", "SPLITS", "SplitViolation", "assert_allowed", "load_locked_manifest",
           "manifest_sha256", "validate_split_manifest"]
