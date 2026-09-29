"""Indexed training views over one Stage-1 store (references, never copies).

``D_geom``  one row per rendered frame with at least one calibrated camera.
``D_dyn``   one row per physics step whose command was recorded; the target
            is the *achieved* next boundary, the input is state_t + command_t.
            Steps with an external intervention or a kinematic arm excursion
            (upstream ``path.visualize``) are kept but flagged.
``D_task``  one row per boundary with event labels and the raw predicate row.

Views that need a frozen reference policy (``D_value``, ``D_pair``,
``D_terminal``) are refused: Stage 1 must not fabricate continuation labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.store import canonical_json, iter_episode_dirs, read_episode


VIEW_PROTOCOL_ID = "icgs-stage1-views-v1"
STAGE1_VIEWS = ("D_geom", "D_dyn", "D_task")
REFERENCE_POLICY_VIEWS = ("D_value", "D_pair", "D_terminal")
_VIEW_KEYS = ("frame_step", "cmd_arm_joint_target_valid", "cmd_arm_joint_velocity_valid",
              "cmd_gripper_joint_velocity_valid", "cmd_arm_teleport_calls", "cmd_phase",
              "step_sim_time", "label_alpha")


class ReferencePolicyRequired(RuntimeError):
    pass


def _intervention_steps(manifest: dict[str, Any]) -> set[int]:
    steps = set()
    for item in manifest["perturbation"]["execution"]:
        if item.get("external_intervention") and item.get("applied_at_step") is not None:
            steps.add(int(item["applied_at_step"]))
    return steps


def build_views(store_root: str | Path, *, view_names: tuple[str, ...] = STAGE1_VIEWS,
                reference_policy_id: str | None = None) -> dict[str, Any]:
    for name in view_names:
        if name in REFERENCE_POLICY_VIEWS and reference_policy_id is None:
            raise ReferencePolicyRequired(f"{name} needs a frozen pi_ref identity; refused in Stage 1")
        if name not in STAGE1_VIEWS:
            raise ValueError(f"unknown or unimplemented view {name}")
    store_root = Path(store_root)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in view_names}
    sources = []
    for episode_dir in iter_episode_dirs(store_root):
        manifest, arrays = read_episode(episode_dir, keys=_VIEW_KEYS)
        episode_id = manifest["episode_id"]
        manifest_digest = hashlib.sha256((episode_dir / "manifest.json").read_bytes()).hexdigest()
        sources.append({"episode_id": episode_id, "manifest_sha256": manifest_digest,
                        "arrays_sha256": manifest["arrays"]["sha256"],
                        "outcome": manifest["outcome"]["status"],
                        "perturbation_families": manifest["perturbation"]["families"],
                        "level": manifest["source"]["level"], "task": manifest["source"]["task"]})
        common = {"episode_id": episode_id, "outcome": manifest["outcome"]["status"],
                  "task": manifest["source"]["task"], "level": manifest["source"]["level"]}
        if "D_geom" in rows and "frame_step" in arrays:
            for frame, boundary in enumerate(arrays["frame_step"]):
                rows["D_geom"].append({**common, "frame": frame, "boundary": int(boundary),
                                       "cameras": manifest["cameras"]["names"]})
        if "D_dyn" in rows:
            interventions = _intervention_steps(manifest)
            written = (arrays["cmd_arm_joint_target_valid"] | arrays["cmd_arm_joint_velocity_valid"]
                       | arrays["cmd_gripper_joint_velocity_valid"].any(axis=1))
            times = arrays["step_sim_time"]
            for step in np.flatnonzero(written):
                step = int(step)
                rows["D_dyn"].append({
                    **common, "step": step, "state": step, "achieved": step + 1,
                    "duration_s": float(times[step + 1] - times[step]),
                    "phase": int(arrays["cmd_phase"][step]),
                    "kinematic_arm_excursion": bool(arrays["cmd_arm_teleport_calls"][step] > 0),
                    "external_intervention": step in interventions,
                })
        if "D_task" in rows and "label_alpha" in arrays:
            for boundary in range(arrays["label_alpha"].shape[0]):
                rows["D_task"].append({**common, "boundary": boundary,
                                       "events": [spec["event_id"] for spec in manifest["events"]["specs"]]})
    views_dir = store_root / "views"
    views_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"view_protocol": VIEW_PROTOCOL_ID, "sources": sources, "views": {}}
    for name, items in rows.items():
        path = views_dir / f"{name}.jsonl"
        payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
        path.write_text(payload)
        summary["views"][name] = {"rows": len(items), "file": path.name,
                                  "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (views_dir / "manifest.json").write_text(canonical_json(summary))
    return summary


__all__ = ["REFERENCE_POLICY_VIEWS", "ReferencePolicyRequired", "STAGE1_VIEWS", "VIEW_PROTOCOL_ID", "build_views"]
