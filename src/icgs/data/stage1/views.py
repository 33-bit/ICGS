"""Indexed training views over one split directory (references, never array copies).

``D_geom``  one row per rendered frame. A row is *primary* only when its
            episode persisted foreground masks with a complete
            handle → object → role/category legend; the row names the depth,
            calibration and mask arrays and the frame → physics-boundary map.
``D_dyn``   one row per **physics-step transition**
            ``(step_*[s], cmd_*[s]) -> step_*[s + 1]`` of ``timing.physics_dt_s``
            seconds. Model-cadence transitions (every ``frame_stride`` steps) are
            obtained by grouping consecutive rows; the row gives the frames at
            both ends when they exist. ``prof_step_wall_s`` is never a duration.
``D_task``  one row per query prefix at model cadence (rendered boundaries),
            paired with an **independent** successful demonstration *context*
            of the same task: context event tokens (ordered first occurrences),
            the alignment target (or null), rho/nu/epsilon re-indexed to the
            tokens, and validity masks. Query and context are distinct
            episodes and never share lineage; in TEST, contexts come only from
            the context seed range and queries never do.

Views needing a frozen reference policy (``D_value``, ``D_pair``,
``D_terminal``) are refused in Stage 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.labels import context_alignment, context_event_tokens
from icgs.data.stage1.store import canonical_json, iter_episode_dirs, manifest_sha256, read_episode


VIEW_PROTOCOL_ID = "icgs-stage1-views-v2"
STAGE1_VIEWS = ("D_geom", "D_dyn", "D_task")
REFERENCE_POLICY_VIEWS = ("D_value", "D_pair", "D_terminal")
_KEYS = ("frame_step", "cmd_arm_joint_target_valid", "cmd_arm_joint_velocity_valid",
         "cmd_gripper_joint_velocity_valid", "cmd_grasp_event", "cmd_arm_teleport_calls", "cmd_phase",
         "step_sim_time", "monitor_rho", "monitor_rho_valid", "monitor_nu", "monitor_nu_valid",
         "monitor_epsilon", "monitor_epsilon_valid", "monitor_event_id", "monitor_first_occurrence")


class ReferencePolicyRequired(RuntimeError):
    pass


class LineageViolation(RuntimeError):
    pass


def _lineage_root(manifest: dict[str, Any], manifests: dict[str, dict[str, Any]]) -> str:
    episode = manifest["episode_id"]
    seen = set()
    while True:
        parent = manifests.get(episode, {}).get("lineage", {}).get("parent_episode_id")
        if parent is None or parent in seen:
            return episode
        seen.add(episode)
        episode = parent


def _frame_at_or_after(frame_step: np.ndarray, boundary: int) -> int | None:
    index = int(np.searchsorted(frame_step, boundary))
    return index if index < frame_step.shape[0] else None


def _pair_rng(split_id: str, split: str, query: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{split_id}|{split}|{query}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def build_views(split_dir: str | Path, *, view_names: tuple[str, ...] = STAGE1_VIEWS,
                reference_policy_id: str | None = None, contexts_per_query: int = 1,
                pairing_seed: int = 0) -> dict[str, Any]:
    for name in view_names:
        if name in REFERENCE_POLICY_VIEWS and reference_policy_id is None:
            raise ReferencePolicyRequired(f"{name} needs a frozen pi_ref identity; refused in Stage 1")
        if name not in STAGE1_VIEWS:
            raise ValueError(f"unknown or unimplemented view {name}")
    split_dir = Path(split_dir)
    split = split_dir.name
    manifests: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for episode_dir in iter_episode_dirs(split_dir):
        manifest, data = read_episode(episode_dir, keys=_KEYS)
        episode = manifest["episode_id"]
        manifests[episode], arrays[episode] = manifest, data
        split_info = manifest["lineage"].get("split") or {}
        sources[episode] = {"manifest_sha256": manifest_sha256(episode_dir),
                            "arrays_sha256": manifest["arrays"]["sha256"], "task": manifest["source"]["task"],
                            "outcome": manifest["outcome"]["status"], "split": split_info.get("split", split),
                            "test_role": split_info.get("test_role")}
        if split_info and split_info.get("split") != split:
            raise LineageViolation(f"{episode} declares split {split_info.get('split')} but sits in {split}")
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in view_names}
    for episode, manifest in manifests.items():
        data = arrays[episode]
        common = {"episode_id": episode, "task": manifest["source"]["task"], "outcome": manifest["outcome"]["status"]}
        cameras = manifest["cameras"]
        legend_ok = bool(cameras.get("masks_recorded") and cameras.get("mask_legend"))
        if "D_geom" in rows and "frame_step" in data:
            for frame, boundary in enumerate(data["frame_step"]):
                rows["D_geom"].append({
                    **common, "frame": frame, "boundary": int(boundary), "cameras": cameras["names"],
                    "arrays": {"depth": "cam_<cam>_depth", "intrinsics": "cam_<cam>_intrinsics",
                               "extrinsics": "cam_<cam>_extrinsics", "near_far": "cam_<cam>_near/far",
                               "mask": "cam_<cam>_mask_handles" if legend_ok else None},
                    "primary": legend_ok,
                    "non_primary_reason": None if legend_ok else "no persisted foreground mask legend",
                })
        if "D_dyn" in rows:
            written = (data["cmd_arm_joint_target_valid"] | data["cmd_arm_joint_velocity_valid"]
                       | data["cmd_gripper_joint_velocity_valid"].any(axis=1) | (data["cmd_grasp_event"] != 0))
            times = data["step_sim_time"]
            frames = {int(b): i for i, b in enumerate(data.get("frame_step", np.zeros(0, int)))}
            interventions = {int(item["applied_at_step"]) for item in manifest["perturbation"]["execution"]
                             if item.get("external_intervention") and item.get("applied_at_step") is not None}
            for step in np.flatnonzero(written):
                step = int(step)
                rows["D_dyn"].append({
                    **common, "transition": "physics_step", "state_boundary": step, "command_row": step,
                    "achieved_boundary": step + 1, "dt_s": float(times[step + 1] - times[step]),
                    "frame_before": frames.get(step), "frame_after": frames.get(step + 1),
                    "phase": int(data["cmd_phase"][step]),
                    "kinematic_arm_excursion": bool(data["cmd_arm_teleport_calls"][step] > 0),
                    "external_intervention": step in interventions,
                })
    leakage: dict[str, Any] = {"split": split, "pairs": 0, "unpaired_queries": [], "violations": []}
    if "D_task" in rows:
        roots = {e: _lineage_root(m, manifests) for e, m in manifests.items()}
        for query, manifest in sorted(manifests.items()):
            if split == "test" and sources[query]["test_role"] != "query":
                continue
            candidates = [
                c for c, cm in sorted(manifests.items())
                if c != query and cm["source"]["task"] == manifest["source"]["task"]
                and cm["outcome"]["status"] == "success" and roots[c] != roots[query]
                and (split != "test" or sources[c]["test_role"] == "context")
            ]
            if not candidates:
                leakage["unpaired_queries"].append(query)
                continue
            rng = _pair_rng(manifest["lineage"].get("split", {}).get("split_id", ""), split, query, pairing_seed)
            chosen = [candidates[i] for i in sorted(rng.choice(len(candidates), size=min(contexts_per_query,
                                                                                        len(candidates)),
                                                               replace=False))]
            query_events = [spec["event_id"] for spec in manifest["events"]["specs"]]
            q = arrays[query]
            state = {k: q[f"monitor_{k}"] for k in ("rho", "rho_valid", "nu", "nu_valid", "epsilon", "epsilon_valid")}
            boundaries = [int(b) for b in q["frame_step"]] if "frame_step" in q else list(range(q["monitor_rho"].shape[0]))
            for context in chosen:
                if context == query or roots[context] == roots[query]:
                    raise LineageViolation("query and context must be distinct episodes with distinct lineage")
                cm, cd = manifests[context], arrays[context]
                first = {spec["event_id"]: int(cd["monitor_first_occurrence"][j])
                         for j, spec in enumerate(cm["events"]["specs"])}
                tokens = context_event_tokens(first)
                if not tokens:
                    continue
                labels = context_alignment(state, query_events, tokens, boundaries)
                token_rows = [{"token": k, "event_id": event, "context_boundary": boundary,
                               "context_frame": _frame_at_or_after(cd["frame_step"], boundary)
                               if "frame_step" in cd else None} for k, (event, boundary) in enumerate(tokens)]
                leakage["pairs"] += 1
                for r, boundary in enumerate(boundaries):
                    target = int(labels["alignment_target"][r])
                    rows["D_task"].append({
                        "query_episode": query, "query_boundary": boundary,
                        "query_frame": r if "frame_step" in q else None, "task": manifest["source"]["task"],
                        "context_episodes": [context], "context_tokens": token_rows,
                        "alignment_target": target, "alignment_is_null": target == len(tokens),
                        "alignment_valid": bool(labels["alignment_valid"][r]),
                        **{name: labels[name][r].astype(int).tolist()
                           for name in ("rho", "rho_valid", "nu", "nu_valid", "epsilon", "epsilon_valid")},
                    })
    views_dir = split_dir / "views"
    views_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {
        "view_protocol": VIEW_PROTOCOL_ID, "split": split, "sources": sources, "views": {},
        "semantics": {
            "D_geom": "one rendered frame; primary rows have persisted masks + complete legend",
            "D_dyn": "one physics step: (step[s], cmd[s]) -> step[s+1]; dt_s from the simulator clock",
            "D_task": "query prefix at a rendered boundary aligned to an independent success context of the "
                      "same task; alignment null index = number of context tokens",
        },
        "pairing": {"contexts_per_query": contexts_per_query, "pairing_seed": pairing_seed,
                    "context_requirements": ["same task", "outcome success", "distinct episode",
                                             "distinct lineage root",
                                             "TEST: context seed range only; queries never from it"]},
        "leakage": leakage,
    }
    for name, items in rows.items():
        path = views_dir / f"{name}.jsonl"
        payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
        temporary = views_dir / f".{name}.jsonl.tmp"
        temporary.write_text(payload)
        temporary.replace(path)
        summary["views"][name] = {"rows": len(items), "file": path.name,
                                  "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (views_dir / "manifest.json").write_text(canonical_json(summary))
    return summary


__all__ = ["LineageViolation", "REFERENCE_POLICY_VIEWS", "ReferencePolicyRequired", "STAGE1_VIEWS",
           "VIEW_PROTOCOL_ID", "build_views"]
