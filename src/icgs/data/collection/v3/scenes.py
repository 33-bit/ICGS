"""Scene layouts for v3 compilation. Geometry only; events come from structured steps."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_LAYOUT_PATH = Path(__file__).with_name("scene_layouts.json")
_SUPPORTS = ("pad", "tray", "holder", "drawer")


def apply_rest_heights(layout: dict[str, Any]) -> dict[str, Any]:
    """Put place targets at object-rest height on the supporting surface."""
    raw = layout.get("objects") or {}
    objects: dict[str, Any] = {}
    for name, spec in raw.items():
        if isinstance(spec, dict) and "pos" in spec:
            objects[name] = {
                "pos": list(spec["pos"]),
                "size": list(spec["size"]),
                "color": list(spec.get("color") or [0.5, 0.5, 0.5]),
            }
        else:
            pos, size, color = spec
            objects[name] = {"pos": list(pos), "size": list(size), "color": list(color)}
    support = next((objects[name] for name in _SUPPORTS if name in objects), None)
    if support is not None:
        top = float(support["pos"][2]) + float(support["size"][2]) / 2.0
        for name, spec in objects.items():
            if not name.startswith("target"):
                continue
            suffix = name[len("target_"):]
            obj_name = "spacer" if suffix == "spacer" else f"object_{suffix}"
            if obj_name not in objects:
                obj_name = "object_a" if "object_a" in objects else None
            half = float(objects[obj_name]["size"][2]) / 2.0 if obj_name else 0.0225
            spec["pos"][2] = top + half
    out = dict(layout)
    out["objects"] = objects
    return out


@lru_cache(maxsize=1)
def load_scene_layouts() -> dict[str, Any]:
    if not _LAYOUT_PATH.is_file():
        raise FileNotFoundError(f"v3 scene layouts missing: {_LAYOUT_PATH}")
    data = json.loads(_LAYOUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("v3 scene layouts must be a non-empty object")
    return data


def get_scene_layout(program_id: str) -> dict[str, Any]:
    layouts = load_scene_layouts()
    if program_id not in layouts:
        raise KeyError(f"no v3 scene layout for {program_id}")
    return apply_rest_heights(layouts[program_id])
