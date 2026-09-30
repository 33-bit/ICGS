"""The on-disk RoboHiMan dataset: layout, split enforcement and dataset metadata.

```text
datasets/robohiman/
├── manifest.json            dataset identity, schema/layout versions, split id + hash
├── splits.json              byte copy of the locked split manifest
├── preprocessing/           statistics fitted on TRAIN only
├── reports/                 audit.json, leakage.json, reproducibility.json, generation_log.jsonl
├── train/ dev/ test/
│   ├── episodes/<episode_id>/{manifest.json, arrays.npz}   immutable raw episodes
│   ├── attempts/<attempt_id>.json                          simulator errors, never episodes
│   ├── derived/<episode_id>/<name>_v<k>.{npz,json}         later computations
│   └── views/{D_geom,D_dyn,D_task}.jsonl + manifest.json   references only
```

Directory assignment is enforced here, not by convention: ``commit_episode``
rejects an episode whose split, frozen-split hash, task, strategy, variation,
seed, factor seed, perturbation family, lineage, ID or destination disagrees
with the dataset. Episode IDs are ``ep-<task>-<run8>-<index>``; everything
else lives in the episode manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.schema import EPISODE_ID_PATTERN, SCHEMA_VERSION
from icgs.data.stage1.split import SPLITS, SplitViolation, assert_allowed, load_locked_manifest, validate_split_manifest
from icgs.data.stage1.store import canonical_json, iter_episode_dirs, read_manifest, write_derived, write_episode

DATASET_NAME = "robohiman"
LAYOUT_VERSION = "robohiman-dataset-layout-v1"
REPORT_FILES = ("audit.json", "leakage.json", "reproducibility.json")
GENERATION_LOG = "generation_log.jsonl"
_RUN8 = re.compile(r"^[0-9a-f]{8}$")


def new_run8() -> str:
    return secrets.token_hex(4)


def make_episode_id(task: str, run8: str, index: int) -> str:
    if not _RUN8.match(run8):
        raise ValueError("run8 must be 8 lowercase hex characters")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("episode index must be a nonnegative integer")
    episode_id = f"ep-{task}-{run8}-{index:04d}"
    if not re.match(EPISODE_ID_PATTERN, episode_id):
        raise ValueError(f"invalid episode id {episode_id!r}")
    return episode_id


def seed_state_sha256(seed: int) -> str:
    """Hash of numpy's MT19937 state right after ``np.random.seed(seed)`` (as stored per episode)."""
    _, keys, pos, has_gauss, _ = np.random.RandomState(seed).get_state()
    array = np.concatenate([np.asarray(keys, dtype=np.uint32), np.array([pos, has_gauss], dtype=np.uint32)])
    return hashlib.sha256(array.astype(np.uint32).tobytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload))
    os.replace(temporary, path)


class Dataset:
    def __init__(self, root: Path, manifest: dict[str, Any], split_manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.splits = split_manifest

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: str | Path, split_manifest_path: str | Path, *, benchmark: Mapping[str, Any] | None = None,
               reproducibility: Mapping[str, Any] | None = None) -> "Dataset":
        root = Path(root)
        split = load_locked_manifest(split_manifest_path)
        split_bytes = Path(split_manifest_path).read_bytes()
        if (root / "manifest.json").exists():
            dataset = cls.open(root)
            if dataset.manifest["split_manifest_sha256"] != split["_sha256"]:
                raise SplitViolation("dataset already bound to a different frozen split")
            return dataset
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} exists and is not a dataset")
        for name in ("preprocessing", "reports", *SPLITS):
            (root / name).mkdir(parents=True, exist_ok=True)
        for name in SPLITS:
            for sub in ("episodes", "attempts", "derived", "views"):
                (root / name / sub).mkdir(exist_ok=True)
        (root / "splits.json").write_bytes(split_bytes)
        (root / "splits.json").chmod(0o444)
        manifest = {
            "dataset": DATASET_NAME,
            "layout_version": LAYOUT_VERSION,
            "episode_schema_version": SCHEMA_VERSION,
            "split_id": split["split_id"],
            "split_manifest_sha256": split["_sha256"],
            "created_utc": _now(),
            "splits": {name: f"{name}/" for name in SPLITS},
            "gradient_splits": ["train"],
            "benchmark": dict(benchmark or split.get("benchmark", {})),
            "policies": {
                "raw_episodes": "immutable after atomic commit; later computations go to derived/",
                "preprocessing": "statistics used for training are fitted on TRAIN episodes only",
                "episode_id": "ep-<task>-<run8>-<index>; variation/strategy/split/seed live in manifest.json",
            },
        }
        _atomic_json(root / "manifest.json", manifest)
        pending = {"status": "pending", "created_utc": _now()}
        _atomic_json(root / "reports" / "audit.json", {**pending, "kind": "compatibility and collection audit"})
        _atomic_json(root / "reports" / "leakage.json", {**pending, "kind": "split and query/context leakage checks"})
        _atomic_json(root / "reports" / "reproducibility.json",
                     {**pending, "kind": "environment pins and determinism", **dict(reproducibility or {})})
        (root / "reports" / GENERATION_LOG).touch()
        _atomic_json(root / "preprocessing" / "index.json", {"policy": "fitted on TRAIN only", "files": []})
        return cls.open(root)

    @classmethod
    def open(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("dataset") != DATASET_NAME or manifest.get("layout_version") != LAYOUT_VERSION:
            raise ValueError(f"{root} is not a {DATASET_NAME} dataset with layout {LAYOUT_VERSION}")
        actual = hashlib.sha256((root / "splits.json").read_bytes()).hexdigest()
        if actual != manifest["split_manifest_sha256"]:
            raise SplitViolation("splits.json does not match the hash recorded in manifest.json")
        split = json.loads((root / "splits.json").read_text())
        validate_split_manifest(split)
        split["_sha256"] = actual
        return cls(root, manifest, split)

    def split_dir(self, split: str) -> Path:
        if split not in SPLITS:
            raise SplitViolation(f"unknown split {split!r}")
        return self.root / split

    # ------------------------------------------------------------ queries
    def episode_index(self) -> dict[str, str]:
        """episode_id -> split, over the whole dataset."""
        index: dict[str, str] = {}
        for split in SPLITS:
            for episode_dir in iter_episode_dirs(self.split_dir(split)):
                index[episode_dir.name] = split
        return index

    def iter_episodes(self, split: str) -> Iterator[Path]:
        yield from iter_episode_dirs(self.split_dir(split))

    def test_role(self, seed: int) -> str:
        low, high = self.splits["splits"]["test"].get("context_seed_range", [1, 0])
        return "context" if low <= seed <= high else "query"

    # ------------------------------------------------------------ enforcement
    def split_provenance(self, *, split: str, task: str, strategy: int, variation: int, seed: int,
                         factor_env_seed: int) -> dict[str, Any]:
        info = assert_allowed(self.splits, split=split, task=task, strategy=strategy, variation=variation,
                              seed=seed, factor_env_seed=factor_env_seed)
        info.update(episode_seed=int(seed), factor_env_seed=int(factor_env_seed))
        if split == "test":
            info["test_role"] = self.test_role(seed)
        return info

    def check_episode(self, split: str, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                      destination: str | Path | None = None) -> None:
        """Raise SplitViolation unless the episode belongs exactly where it is being written."""
        if destination is not None and Path(destination).resolve() != self.split_dir(split).resolve():
            raise SplitViolation(f"destination {destination} is not the {split} directory of this dataset")
        lineage = manifest.get("lineage", {})
        declared = lineage.get("split")
        if not isinstance(declared, Mapping):
            raise SplitViolation("episode has no lineage.split provenance")
        if declared.get("split") != split:
            raise SplitViolation(f"episode declares split {declared.get('split')!r}, not {split!r}")
        if declared.get("split_id") != self.manifest["split_id"] or \
                declared.get("split_manifest_sha256") != self.manifest["split_manifest_sha256"]:
            raise SplitViolation("episode was collected under a different frozen split")
        source = manifest["source"]
        strategy = source["collection_strategy"]
        expected = self.split_provenance(split=split, task=source["task"], strategy=int(strategy["index"]),
                                         variation=int(source["variation_index"]),
                                         seed=int(declared["episode_seed"]),
                                         factor_env_seed=int(manifest["environment"]["env_seed"]))
        for key in ("gradient_eligible", "tracks", "test_role"):
            if key in expected and declared.get(key) != expected[key]:
                raise SplitViolation(f"lineage.split.{key} disagrees with the frozen split")
        withheld = set(self.splits.get("held_out_perturbation_strategies", ()))
        if split != "test" and strategy.get("name") in withheld:
            raise SplitViolation(f"perturbation strategy {strategy.get('name')!r} is withheld for TEST")
        seed_hash = seed_state_sha256(int(declared["episode_seed"]))
        if lineage.get("reset_rng", {}).get("sha256") != seed_hash:
            raise SplitViolation("reset RNG state does not come from the declared episode seed")
        stored = arrays.get("rng_state_mt19937")
        if stored is None or hashlib.sha256(np.asarray(stored, dtype=np.uint32).tobytes()).hexdigest() != seed_hash:
            raise SplitViolation("stored RNG state does not match the declared episode seed")
        index = self.episode_index()
        if manifest["episode_id"] in index:
            raise SplitViolation(f"episode id {manifest['episode_id']} already exists in {index[manifest['episode_id']]}")
        parent = lineage.get("parent_episode_id")
        if parent is not None and index.get(parent) != split:
            raise SplitViolation(f"parent episode {parent!r} is not a committed {split} episode")
        cameras = manifest.get("cameras", {})
        if cameras.get("masks_recorded"):
            legend = {int(handle) for handle in cameras.get("mask_legend", {})}
            seen: set[int] = set()
            for name, value in arrays.items():
                if name.endswith("_mask_handles"):
                    seen |= {int(v) for v in np.unique(value)}
            unmapped = sorted(seen - legend)
            if unmapped:
                raise SplitViolation(f"mask handles without legend entries: {unmapped[:10]}")

    def commit_episode(self, split: str, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                       destination: str | Path | None = None) -> Path:
        self.check_episode(split, manifest, arrays, destination)
        path = write_episode(self.split_dir(split), manifest, arrays)
        self.log({"event": "episode_committed", "split": split, "episode_id": manifest["episode_id"],
                  "task": manifest["source"]["task"], "outcome": manifest["outcome"]["status"],
                  "seed": manifest["lineage"]["split"]["episode_seed"],
                  "arrays_sha256": read_manifest(path)["arrays"]["sha256"]})
        return path

    def record_attempt(self, split: str, attempt_id: str, record: Mapping[str, Any]) -> Path:
        if not re.match(r"^att-[a-z0-9_]+-[0-9a-f]{8}-[0-9]{4,}$", attempt_id):
            raise ValueError("attempt ids look like att-<task>-<run8>-<index>")
        path = self.split_dir(split) / "attempts" / f"{attempt_id}.json"
        if path.exists():
            raise FileExistsError(path)
        _atomic_json(path, dict(record))
        self.log({"event": "attempt_recorded", "split": split, "attempt_id": attempt_id,
                  "outcome": record.get("outcome")})
        return path

    def write_derived(self, split: str, episode_id: str, name: str, arrays: Mapping[str, np.ndarray],
                      metadata: Mapping[str, Any]) -> Path:
        if self.episode_index().get(episode_id) != split:
            raise SplitViolation(f"{episode_id} is not a committed {split} episode")
        return write_derived(self.split_dir(split), episode_id, name, arrays, metadata)

    def write_preprocessing(self, name: str, statistics: Mapping[str, Any], *, source_episode_ids: list[str],
                            method: str) -> Path:
        """Store training statistics; every source episode must be a TRAIN episode."""
        if not re.match(r"^[a-z0-9_]+_v[0-9]+$", name):
            raise ValueError("preprocessing names must be versioned, e.g. joint_position_stats_v1")
        index = self.episode_index()
        outside = sorted(e for e in source_episode_ids if index.get(e) != "train")
        if outside or not source_episode_ids:
            raise SplitViolation(f"preprocessing statistics must be fitted on TRAIN episodes only: {outside[:5]}")
        path = self.root / "preprocessing" / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        ids = sorted(source_episode_ids)
        _atomic_json(path, {"name": name, "fitted_on": "train", "method": method, "statistics": dict(statistics),
                            "source_episode_count": len(ids),
                            "source_episode_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                            "split_id": self.manifest["split_id"], "created_utc": _now()})
        index_path = self.root / "preprocessing" / "index.json"
        index_doc = json.loads(index_path.read_text())
        index_doc["files"].append(path.name)
        _atomic_json(index_path, index_doc)
        return path

    # ------------------------------------------------------------ reports
    def log(self, event: Mapping[str, Any]) -> None:
        line = json.dumps({"utc": _now(), **dict(event)}, sort_keys=True)
        with (self.root / "reports" / GENERATION_LOG).open("a") as handle:
            handle.write(line + "\n")

    def update_report(self, name: str, payload: Mapping[str, Any]) -> None:
        if name not in REPORT_FILES:
            raise ValueError(f"unknown report {name}")
        _atomic_json(self.root / "reports" / name, {**dict(payload), "updated_utc": _now()})


def remove_scratch_dataset(root: str | Path) -> None:
    """Test/scratch helper: raw files are read-only, so restore write bits before removal."""
    for path in Path(root).rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    shutil.rmtree(root)


__all__ = ["DATASET_NAME", "Dataset", "LAYOUT_VERSION", "make_episode_id", "new_run8", "remove_scratch_dataset",
           "seed_state_sha256"]
