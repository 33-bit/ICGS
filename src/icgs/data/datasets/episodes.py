"""Pure split and causal-history helpers, manifest lineage closure, and views for executed episodes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition
from icgs.data.archives import read_episode_archive
from icgs.data.collection.programs import program_catalog
from icgs.data.schemas.episodes import validate_episode


_SPLITS = frozenset({"train", "dev", "test"})
_DATASET_TRACKS = frozenset({"primary", "exploratory"})


def validate_split_lineage(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require each declared literal ``lineage_id`` to remain in one split.

    P02 requires descendant-aware splitting too, but no approved root/parent or
    ancestry-closure field currently exists. This helper intentionally does not
    infer ancestry from identifiers, asset names, or augmentation naming.
    """
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("split lineage rows must be a sequence")
    splits_by_lineage: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"split lineage row {index} must be a mapping")
        if "lineage_id" not in row or "split" not in row:
            raise ValueError(f"split lineage row {index} must include lineage_id and split")
        lineage_id, split = row["lineage_id"], row["split"]
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError(f"split lineage row {index} has an invalid lineage_id")
        if split not in _SPLITS:
            raise ValueError(f"split lineage row {index} has an invalid split")
        previous = splits_by_lineage.setdefault(lineage_id, split)
        if previous != split:
            raise ValueError("source lineage crosses splits")


def causal_prefix(episode: Mapping[str, Any], boundary: int) -> tuple[ExecutedTransition, ...]:
    """Return only executed transitions whose successor is known at ``boundary``.

    ``boundary`` is an integer boundary index, not a transition-array position.
    Consequently the returned history ends at the current boundary and cannot
    expose the command or measured observation from a later interval.
    """
    if isinstance(boundary, bool) or not isinstance(boundary, int):
        raise ValueError("boundary must be an integer")
    validate_episode(episode)
    transitions = tuple(episode["transitions"])
    first_boundary = transitions[0].before.boundary
    final_boundary = transitions[-1].after.boundary
    if not first_boundary <= boundary <= final_boundary:
        raise ValueError("boundary must be within the episode's executed boundaries")
    return tuple(transition for transition in transitions if transition.after.boundary <= boundary)


def validate_dataset_manifest(
    manifest_data: Mapping[str, Any] | str | Path,
    *,
    dataset_root: str | Path | None = None,
    approved_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate dataset manifest schema, lineage closure, acyclicity, and split consistency."""
    if isinstance(manifest_data, (str, Path)):
        manifest_path = Path(manifest_data).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest file not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if dataset_root is None:
            dataset_root = manifest_path.parent
    elif isinstance(manifest_data, Mapping):
        data = dict(manifest_data)
    else:
        raise TypeError(f"manifest_data must be a mapping or path, got {type(manifest_data).__name__}")

    approved_manifest = None
    if approved_manifest_path is not None:
        from icgs.data.collection.approved_manifest import load_approved_manifest

        approved_manifest = load_approved_manifest(approved_manifest_path)

    if data.get("manifest_version") != 1:
        raise ValueError(f"Unsupported manifest_version: {data.get('manifest_version')!r}")
    dataset_track = data.get("dataset_track", "primary")
    if dataset_track not in _DATASET_TRACKS:
        raise ValueError(f"Unsupported dataset_track: {dataset_track!r}")
    data["dataset_track"] = dataset_track

    lineage = data.get("lineage")
    if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
        raise ValueError("Dataset manifest lineage must be a sequence")

    lineage_nodes: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(lineage):
        if not isinstance(node, Mapping):
            raise ValueError(f"Lineage entry {idx} must be a mapping")
        lid = node.get("lineage_id")
        if not isinstance(lid, str) or not lid.strip():
            raise ValueError(f"Lineage entry {idx} has invalid lineage_id")
        if lid in lineage_nodes:
            raise ValueError(f"Duplicate lineage_id in lineage: {lid}")
        split = node.get("split")
        if split not in _SPLITS:
            raise ValueError(f"Lineage entry {lid} has invalid split: {split}")
        parents = node.get("parent_ids", ())
        if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
            raise ValueError(f"Lineage entry {lid} parent_ids must be a sequence of strings")
        lineage_nodes[lid] = {"split": split, "parent_ids": tuple(parents)}

    # Check that parents exist, check acyclicity, and check split consistency
    visited: dict[str, int] = {}  # 0: visiting, 1: visited

    def _dfs(node_id: str) -> None:
        visited[node_id] = 0
        node_split = lineage_nodes[node_id]["split"]
        for p in lineage_nodes[node_id]["parent_ids"]:
            if p not in lineage_nodes:
                raise ValueError(f"Lineage node {node_id} has dangling parent: {p}")
            parent_split = lineage_nodes[p]["split"]
            if parent_split != node_split:
                raise ValueError(
                    f"Lineage cross-split violation: node {node_id} ({node_split}) != parent {p} ({parent_split})"
                )
            if visited.get(p) == 0:
                raise ValueError(f"Lineage cycle detected involving {node_id} and {p}")
            if p not in visited:
                _dfs(p)
        visited[node_id] = 1

    for lid in lineage_nodes:
        if lid not in visited:
            _dfs(lid)

    # Asset families
    asset_families_raw = data.get("asset_families", ())
    if not isinstance(asset_families_raw, Sequence) or isinstance(asset_families_raw, (str, bytes)):
        raise ValueError("Dataset manifest asset_families must be a sequence")

    asset_family_splits: dict[str, str] = {}
    for idx, fam in enumerate(asset_families_raw):
        if not isinstance(fam, Mapping):
            raise ValueError(f"Asset family entry {idx} must be a mapping")
        afid = fam.get("asset_family_id")
        if not isinstance(afid, str) or not afid.strip():
            raise ValueError(f"Asset family entry {idx} has invalid asset_family_id")
        split = fam.get("split")
        if split not in _SPLITS:
            raise ValueError(f"Asset family entry {afid} has invalid split: {split}")
        if afid in asset_family_splits:
            if asset_family_splits[afid] != split:
                raise ValueError(f"asset_family duplicate across splits: {afid}")
        asset_family_splits[afid] = split

    # Episodes
    episodes = data.get("episodes", ())
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise ValueError("Dataset manifest episodes must be a sequence")

    seen_episodes: set[str] = set()
    catalog = program_catalog()

    for idx, ep_meta in enumerate(episodes):
        if not isinstance(ep_meta, Mapping):
            raise ValueError(f"Episode entry {idx} must be a mapping")
        ep_id = ep_meta.get("episode_id")
        if not isinstance(ep_id, str) or not ep_id.strip():
            raise ValueError(f"Episode entry {idx} has invalid episode_id")
        if ep_id in seen_episodes:
            raise ValueError(f"Duplicate episode_id in dataset manifest: {ep_id}")
        seen_episodes.add(ep_id)

        # Inspect on-disk episode manifest: require safe relative path, no symlinks, and valid SHA256
        manifest_rel = ep_meta.get("manifest_path")
        if not isinstance(manifest_rel, str) or not manifest_rel.strip():
            raise ValueError(f"Episode {ep_id} missing required 'manifest_path'")

        rel_path = Path(manifest_rel)
        if rel_path.is_absolute():
            raise ValueError(f"Episode {ep_id} manifest_path must be relative, got: {manifest_rel}")
        if ".." in rel_path.parts:
            raise ValueError(f"Episode {ep_id} manifest_path contains path traversal: {manifest_rel}")

        if dataset_root is None:
            raise ValueError("dataset_root is required to validate episode manifest paths")
        root_resolved = Path(dataset_root).resolve()
        ep_manifest_file = (root_resolved / rel_path).resolve()

        if root_resolved != ep_manifest_file and root_resolved not in ep_manifest_file.parents:
            raise ValueError(f"Episode {ep_id} manifest_path traverses outside dataset_root: {manifest_rel}")

        # Check for symlinks in path components
        curr_chk = root_resolved
        for part in rel_path.parts:
            curr_chk = curr_chk / part
            if curr_chk.is_symlink():
                raise ValueError(f"Episode {ep_id} manifest_path component is a symlink: {curr_chk}")

        if not ep_manifest_file.is_file():
            raise FileNotFoundError(f"Episode {ep_id} manifest file does not exist: {ep_manifest_file}")

        # Verify SHA256 of episode manifest
        expected_sha = ep_meta.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"Episode {ep_id} missing valid 64-char sha256 checksum in dataset manifest")

        hasher = hashlib.sha256()
        with open(ep_manifest_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        computed_sha = hasher.hexdigest()
        if computed_sha != expected_sha:
            raise ValueError(f"Episode {ep_id} manifest sha256 mismatch: {computed_sha} != {expected_sha}")

        with open(ep_manifest_file, "r", encoding="utf-8") as f:
            ep_data = json.load(f)
        prov = ep_data.get("provenance", {})
        ep_split = prov.get("split")
        source_lineage = prov.get("source_lineage_id")
        asset_fam = prov.get("asset_family_id")
        program_id = prov.get("program_id")

        # Validate against lineage
        if source_lineage not in lineage_nodes:
            raise ValueError(f"Episode {ep_id} source_lineage_id {source_lineage!r} not in dataset lineage")
        if lineage_nodes[source_lineage]["split"] != ep_split:
            raise ValueError(f"Episode {ep_id} split {ep_split} disagrees with lineage split {lineage_nodes[source_lineage]['split']}")

        # Validate against asset families
        if asset_fam not in asset_family_splits:
            raise ValueError(f"Episode {ep_id} asset_family_id {asset_fam!r} not in dataset asset_families")
        if asset_family_splits[asset_fam] != ep_split:
            raise ValueError(f"Episode {ep_id} split {ep_split} disagrees with asset family split {asset_family_splits[asset_fam]}")

        # Validate against program catalog
        if program_id not in catalog:
            raise ValueError(f"Episode {ep_id} program_id {program_id!r} not in program catalog")
        cat_prog = catalog[program_id]
        expected_split = "dev" if cat_prog.split == "development" else cat_prog.split
        if ep_split != expected_split:
            raise ValueError(
                f"Episode {ep_id} program {program_id} catalog split {expected_split} != episode split {ep_split}"
            )
        if approved_manifest is not None:
            from icgs.data.collection.approved_manifest import validate_episode_provenance

            approved_errors = validate_episode_provenance(approved_manifest, prov)
            if approved_errors:
                raise ValueError(f"Episode {ep_id} is not authorized by approved manifest: {'; '.join(approved_errors)}")

    return data


def dataset_manifest_hash(manifest_data: Mapping[str, Any] | str | Path) -> str:
    """Compute a deterministic content hash for a dataset manifest."""
    if isinstance(manifest_data, (str, Path)):
        with open(manifest_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict(manifest_data)
    canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


class GeomView(Sequence[dict[str, Any]]):
    """Initial geometry view exposing observed boundaries and validity masks once."""

    def __init__(self, samples: Sequence[dict[str, Any]], *, split: str) -> None:
        self._samples = tuple(samples)
        self._split = split

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: Any) -> Any:
        return self._samples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._samples)


class DynView(Sequence[dict[str, Any]]):
    """Initial dynamics view exposing causal history and rollout targets from reset."""

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        *,
        split: str,
        excluded_windows_count: int,
    ) -> None:
        self._samples = tuple(samples)
        self._split = split
        self._excluded_windows_count = excluded_windows_count

    @property
    def split(self) -> str:
        return self._split

    @property
    def excluded_windows_count(self) -> int:
        return self._excluded_windows_count

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: Any) -> Any:
        return self._samples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._samples)


def build_view(
    dataset_manifest_path: str | Path,
    view: str,
    *,
    split: str,
    config: MethodConfig | None = None,
) -> GeomView | DynView:
    """Build a lazy indexable view over published episodes for a specific split.

    Parameters
    ----------
    dataset_manifest_path : str | Path
        Path to the validated dataset manifest.
    view : str
        Name of the view: 'geom' or 'dyn'. Other names are rejected.
    split : str
        Data partition: 'train', 'dev', or 'test'.
    config : MethodConfig | None
        Configuration used to resolve unrolling horizons and burnin.
    """
    if view not in ("geom", "dyn"):
        raise ValueError(f"Unsupported view: {view!r}. Only 'geom' and 'dyn' are supported.")
    if split not in _SPLITS:
        raise ValueError(f"Unsupported split: {split!r}")

    manifest_file = Path(dataset_manifest_path).resolve()
    manifest_data = validate_dataset_manifest(manifest_file, dataset_root=manifest_file.parent)
    if manifest_data.get("dataset_track", "primary") != "primary":
        raise ValueError("exploratory dataset_track is not permitted by training views")

    episodes_meta = manifest_data.get("episodes", ())
    dataset_root = manifest_file.parent

    # Filter and load episodes belonging to split
    loaded_episodes: list[dict[str, Any]] = []
    for ep in episodes_meta:
        m_rel = ep["manifest_path"]
        ep_m_path = Path(m_rel)
        if not ep_m_path.is_absolute():
            ep_m_path = dataset_root / ep_m_path
        ep_record = read_episode_archive(ep_m_path)
        if ep_record["provenance"]["split"] == split:
            loaded_episodes.append(ep_record)

    if view == "geom":
        samples: list[dict[str, Any]] = []
        for record in loaded_episodes:
            ep_id = record["provenance"]["episode_id"]
            prov = dict(record["provenance"])
            if "lineage_id" not in prov and "source_lineage_id" in prov:
                prov["lineage_id"] = prov["source_lineage_id"]
            online_obs = record["online_observations"]
            for b_idx, obs in enumerate(online_obs):
                sample = {
                    "sample_id": f"{ep_id}:b{b_idx}",
                    "episode_id": ep_id,
                    "boundary": b_idx,
                    "points": obs["points"],
                    "valid": obs["point_valid"],
                    "T_w_e": obs["T_w_e"],
                    "grip": obs["grip"],
                    "provenance": prov,
                }
                samples.append(sample)
        return GeomView(samples, split=split)

    elif view == "dyn":
        supervised_intervals = 16
        burnin = 8
        if config is not None and hasattr(config, "stages") and hasattr(config.stages, "A1"):
            supervised_intervals = config.stages.A1.supervised_intervals
            burnin = config.stages.A1.burnin_intervals

        samples = []
        excluded_count = 0
        for record in loaded_episodes:
            ep_id = record["provenance"]["episode_id"]
            prov = dict(record["provenance"])
            if "lineage_id" not in prov and "source_lineage_id" in prov:
                prov["lineage_id"] = prov["source_lineage_id"]
            transitions = tuple(record["transitions"])
            n_trans = len(transitions)

            if n_trans < supervised_intervals:
                # Episode too short for a full supervised rollout window
                excluded_count += 1
                continue

            # Sliding windows from boundary 0 up to n_trans - supervised_intervals
            max_start_b = n_trans - supervised_intervals
            for start_b in range(max_start_b + 1):
                target_end_b = start_b + supervised_intervals
                history_trans = transitions[:start_b]
                target_trans = transitions[start_b:target_end_b]
                permitted_trans = transitions[:target_end_b]

                sample = {
                    "sample_id": f"{ep_id}:t{start_b}",
                    "episode_id": ep_id,
                    "boundary": start_b,
                    "history_transitions": history_trans,
                    "target_transitions": target_trans,
                    "transitions": permitted_trans,
                    "supervised_intervals": supervised_intervals,
                    "burnin": burnin,
                    "provenance": prov,
                }
                samples.append(sample)

        return DynView(samples, split=split, excluded_windows_count=excluded_count)

    raise ValueError(f"Unsupported view: {view!r}")


def adapter_a0(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Adapter for Stage A0 training consumer API."""
    if "points" not in sample or "valid" not in sample:
        raise KeyError("A0 sample requires 'points' and 'valid'")
    res: dict[str, Any] = {
        "points": sample["points"],
        "valid": sample["valid"],
    }
    if "provenance" in sample:
        res["provenance"] = dict(sample["provenance"])
    return res


def adapter_a1(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Adapter for Stage A1 training consumer API."""
    if "boundary" not in sample:
        raise KeyError("A1 sample requires 'boundary'")
    if "transitions" in sample:
        transitions = sample["transitions"]
    elif "history_transitions" in sample and "target_transitions" in sample:
        transitions = tuple(sample["history_transitions"]) + tuple(sample["target_transitions"])
    else:
        raise KeyError("A1 sample requires 'transitions' or ('history_transitions', 'target_transitions')")

    res: dict[str, Any] = {
        "transitions": transitions,
        "history_transitions": sample.get("history_transitions", transitions[: sample["boundary"]]),
        "target_transitions": sample.get("target_transitions", transitions[sample["boundary"] :]),
        "boundary": sample["boundary"],
        "supervised_intervals": sample.get("supervised_intervals", 16),
        "burnin": sample.get("burnin", 8),
        "episode_id": sample.get("episode_id"),
    }
    if "provenance" in sample:
        res["provenance"] = dict(sample["provenance"])
    return res


__all__ = [
    "DynView",
    "GeomView",
    "adapter_a0",
    "adapter_a1",
    "build_view",
    "causal_prefix",
    "dataset_manifest_hash",
    "validate_dataset_manifest",
    "validate_split_lineage",
]
