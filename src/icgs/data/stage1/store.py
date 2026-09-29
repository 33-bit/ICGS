"""One directory per episode: ``manifest.json`` + ``arrays.npz``.

Writes go to a temporary sibling directory that is renamed into place only
after the array hash is recorded and the manifest validates, so a crashed
writer never leaves a directory that looks complete. Reads never unpickle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.schema import validate_arrays, validate_manifest


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}; store null plus a reason")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, indent=1, default=_json_default, allow_nan=False)
    _reject_nonfinite(json.loads(text))
    return text


def write_episode(root: str | Path, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomically write one validated episode and return its directory."""
    root = Path(root)
    episode_id = str(manifest.get("episode_id", ""))
    if not _SAFE_ID.match(episode_id):
        raise ValueError(f"unsafe episode_id {episode_id!r}")
    destination = root / "episodes" / episode_id
    if destination.exists():
        raise FileExistsError(f"episode already stored: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    scratch = Path(tempfile.mkdtemp(prefix=f".{episode_id}.", dir=destination.parent))
    try:
        array_path = scratch / "arrays.npz"
        np.savez_compressed(array_path, **arrays)
        manifest = json.loads(canonical_json(dict(manifest)))
        manifest["arrays"] = {
            "file": "arrays.npz",
            "sha256": _sha256_file(array_path),
            "bytes": array_path.stat().st_size,
            "members": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in sorted(arrays.items())},
        }
        validate_manifest(manifest)
        validate_arrays(manifest, arrays)
        (scratch / "manifest.json").write_text(canonical_json(manifest))
        os.replace(scratch, destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return destination


def read_manifest(episode_dir: str | Path) -> dict[str, Any]:
    manifest = json.loads((Path(episode_dir) / "manifest.json").read_text())
    validate_manifest(manifest)
    return manifest


def read_episode(episode_dir: str | Path, *, verify: bool = True,
                 keys: tuple[str, ...] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    episode_dir = Path(episode_dir)
    manifest = read_manifest(episode_dir)
    array_path = episode_dir / manifest["arrays"]["file"]
    if verify and _sha256_file(array_path) != manifest["arrays"]["sha256"]:
        raise ValueError(f"array hash mismatch for {episode_dir}")
    with np.load(array_path, allow_pickle=False) as handle:
        names = handle.files if keys is None else [key for key in keys if key in handle.files]
        arrays = {name: handle[name] for name in names}
    if keys is None and verify:
        validate_arrays(manifest, arrays)
    return manifest, arrays


def iter_episode_dirs(root: str | Path) -> Iterator[Path]:
    base = Path(root) / "episodes"
    if not base.is_dir():
        return
    for path in sorted(base.iterdir()):
        if path.is_dir() and not path.name.startswith(".") and (path / "manifest.json").is_file():
            yield path


__all__ = ["canonical_json", "iter_episode_dirs", "read_episode", "read_manifest", "write_episode"]
