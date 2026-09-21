"""Versioned, simulator-supervision-rich training layout.

The lossless archive writer remains the canonical transition store.  This module
materializes a portable sidecar layout for policy/world-model/debug consumers and
keeps unavailable simulator modalities explicit rather than inventing values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zlib
from typing import Any, Mapping

import numpy as np


LAYOUT_VERSION = 2


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write uint8 HxWx{1,3,4} PNG without adding an image dependency."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim not in (2, 3) or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("RGB/mask frame must be a non-empty uint8 image")
    if arr.ndim == 2:
        color_type = 0
        channels = 1
    else:
        channels = arr.shape[2]
        if channels not in (1, 3, 4):
            raise ValueError("PNG frame must have 1, 3, or 4 channels")
        color_type = {1: 0, 3: 2, 4: 6}[channels]
    raw = b"".join(b"\x00" + row.tobytes() for row in arr.reshape(arr.shape[0], -1, channels))
    header = struct.pack(">IIBBBBB", arr.shape[1], arr.shape[0], 8, color_type, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_dense_frames(path: Path, frames: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=np.asarray(frames))


def _write_ragged_frames(path: Path, frames: Any) -> None:
    arrays = [np.asarray(frame) for frame in frames]
    offsets = [0]
    for array in arrays:
        offsets.append(offsets[-1] + len(array))
    points = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 3), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, offsets=np.asarray(offsets, dtype=np.int64))


def initialize_dataset_layout(root: str | Path, approved_manifest: Mapping[str, Any]) -> Path:
    root = Path(root).resolve()
    catalog = list(approved_manifest.get("catalog", []))
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        (root / "programs" / split).mkdir(parents=True, exist_ok=True)
    programs, assets, splits, controllers = {}, {}, {}, {}
    for index, row in enumerate(catalog, 1):
        pid = str(row["program_id"])
        split = "dev" if row.get("split") == "development" else str(row.get("split", "train"))
        programs[pid] = row
        splits[pid] = split
        asset = str(row.get("asset_family_id", "unknown"))
        assets[asset] = {"asset_family_id": asset, "asset_version": row.get("asset_version")}
        controllers[pid] = {
            "controller": row.get("controller", {}),
            "predicates": row.get("predicates", {}),
        }
        _write_json(root / "programs" / split / f"program_{index:03d}.json", row)
    _write_json(root / "metadata" / "dataset.yaml", {
        "layout_version": LAYOUT_VERSION,
        "dataset_track": "primary",
        "modalities": {
            "rgb": "optional", "depth": "optional", "masks": "optional",
            "object_state": "optional", "contacts": "optional", "snapshots": "optional",
            "embeddings": "offline-derived",
        },
    })
    _write_json(root / "metadata" / "programs.json", programs)
    _write_json(root / "metadata" / "assets.json", assets)
    _write_json(root / "metadata" / "splits.json", splits)
    _write_json(root / "metadata" / "controller_versions.json", controllers)
    (root / "metadata" / "README.md").write_text(
        "ICGS training layout v2. Missing simulator modalities are omitted, never synthesized.\n",
        encoding="utf-8",
    )
    return root


def _require_boundaries(payload: Mapping[str, Any]) -> int:
    obs = payload.get("observations", {})
    robot = payload.get("robot", {})
    if "ee_pose" not in robot:
        raise ValueError("robot.ee_pose is required")
    boundaries = int(np.asarray(robot["ee_pose"]).shape[0])
    if boundaries < 2:
        raise ValueError("episode requires at least two observation boundaries")
    actions = np.asarray(payload.get("actions"))
    if actions.ndim < 1 or actions.shape[0] != boundaries - 1:
        raise ValueError("actions must have boundaries - 1 rows")
    for name, value in robot.items():
        if np.asarray(value).shape[0] != boundaries:
            raise ValueError(f"robot.{name} boundary count mismatch")
    for name, value in obs.items():
        if isinstance(value, (list, tuple)):
            count = len(value)
        else:
            count = np.asarray(value).shape[0]
        if count != boundaries:
            raise ValueError(f"observations.{name} boundary count mismatch")
    task = payload.get("task", {})
    for name in ("rho", "nu", "epsilon", "rho_valid", "nu_valid", "epsilon_valid"):
        if name in task and np.asarray(task[name]).shape[0] != boundaries:
            raise ValueError(f"task.{name} boundary count mismatch")
    return boundaries


def write_training_episode_layout(episode_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    episode_dir = Path(episode_dir).resolve()
    episode = dict(payload.get("episode", {}))
    if not episode.get("episode_id") or episode.get("layout_version") != LAYOUT_VERSION:
        raise ValueError("episode requires episode_id and layout_version=2")
    result = dict(payload.get("result", {}))
    if not isinstance(result.get("success"), bool):
        raise ValueError("result.success must be boolean")
    snapshot = payload.get("snapshot")
    if snapshot and snapshot.get("quality") == "exact" and not snapshot.get("fidelity_evidence"):
        raise ValueError("exact snapshot requires fidelity evidence")
    boundaries = _require_boundaries(payload)
    episode_dir.mkdir(parents=True, exist_ok=True)
    _write_json(episode_dir / "episode.json", episode)
    _write_json(episode_dir / "result.json", result)

    observations = payload.get("observations", {})
    if "rgb" in observations:
        _write_dense_frames(episode_dir / "observations" / "rgb" / "frames.npz", observations["rgb"])
    if "masks" in observations:
        _write_dense_frames(episode_dir / "observations" / "masks" / "frames.npz", observations["masks"])
    for name in ("depth", "pointcloud"):
        if name in observations:
            target = episode_dir / "observations" / name / "frames.npz"
            if name == "pointcloud":
                _write_ragged_frames(target, observations[name])
            else:
                _write_dense_frames(target, observations[name])

    robot = payload["robot"]
    (episode_dir / "robot").mkdir(parents=True, exist_ok=True)
    for name, value in robot.items():
        np.save(episode_dir / "robot" / f"{name}.npy", np.asarray(value))
    (episode_dir / "actions").mkdir(parents=True, exist_ok=True)
    np.save(episode_dir / "actions" / "actions.npy", np.asarray(payload["actions"]))

    scene = payload.get("scene")
    if scene:
        object_ids = list(scene.get("object_ids", [])) if isinstance(scene, dict) else []
        scene_arrays = {k: np.asarray(v) for k, v in scene.items() if k != "object_ids"}
        scene_dir = episode_dir / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scene_dir / "object_state.npz", **scene_arrays)
        _write_json(scene_dir / "object_roles.json", {str(i): x for i, x in enumerate(object_ids)})

    task = dict(payload.get("task", {}))
    task_dir = episode_dir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task_dir / "events.json", task.pop("events", []))
    collisions = task.pop("collisions", [])
    (task_dir / "collisions.jsonl").write_text("".join(json.dumps(_json_safe(x)) + "\n" for x in collisions), encoding="utf-8")
    for name, value in task.items():
        if isinstance(value, np.ndarray):
            np.save(task_dir / f"{name}.npy", value)

    if snapshot:
        snap_dir = episode_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        _write_json(snap_dir / "metadata.json", {k: v for k, v in snapshot.items() if k not in ("anchor", "successor")})
        for label in ("anchor", "successor"):
            state = snapshot.get(label)
            if state is not None:
                with (snap_dir / f"{label}.bin").open("wb") as fh:
                    np.savez_compressed(fh, **{k: np.asarray(v) for k, v in state.items() if isinstance(v, (np.ndarray, list, tuple, int, float, bool))})

    files = {}
    for path in sorted(p for p in episode_dir.rglob("*") if p.is_file()):
        files[str(path.relative_to(episode_dir))] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    inventory = {"layout_version": LAYOUT_VERSION, "episode_id": episode["episode_id"], "success": result["success"], "boundaries": boundaries, "files": files}
    _write_json(episode_dir / "layout_manifest.json", inventory)
    return inventory


def validate_training_episode_layout(episode_dir: str | Path) -> dict[str, Any]:
    episode_dir = Path(episode_dir).resolve()
    episode = json.loads((episode_dir / "episode.json").read_text())
    result = json.loads((episode_dir / "result.json").read_text())
    if episode.get("layout_version") != LAYOUT_VERSION or not isinstance(result.get("success"), bool):
        raise ValueError("invalid episode layout version or result")
    inventory = json.loads((episode_dir / "layout_manifest.json").read_text())
    for rel, info in inventory["files"].items():
        path = episode_dir / rel
        if not path.is_file() or _sha256(path) != info["sha256"]:
            raise ValueError(f"integrity mismatch: {rel}")
    return {"episode_id": episode["episode_id"], "success": result["success"], "boundaries": inventory["boundaries"]}


def write_cache_manifest(cache_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    required = ("model_id", "checkpoint_sha256", "preprocessing_sha256", "source_archive_sha256")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"cache manifest missing {missing[0]}")
    path = Path(cache_dir) / "cache_manifest.json"
    _write_json(path, dict(metadata))
    return path


__all__ = ["LAYOUT_VERSION", "initialize_dataset_layout", "write_training_episode_layout", "validate_training_episode_layout", "write_cache_manifest"]
