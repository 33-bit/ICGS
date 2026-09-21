"""Phase-1 acceptance reports: parity, integrity, calibration, diversity, perturbation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from icgs.data.collection.v3.compiler import CompiledV3Task
from icgs.data.collection.v3.diversity import nearest_signature_distance, scene_signature
from icgs.data.collection.v3.perturbations import INVALID_OBSERVATION, SIMULATOR_CRASH
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.quota import valid_attempt_mixture, QuotaCounts, record_outcome
from icgs.data.collection.v3.steps import structured_steps
from icgs.data.schemas.episodes_v2 import validate_episode_v2


def generation_parity_gate(
    compiled: Mapping[str, CompiledV3Task],
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Block scale/authorize when catalog, compiled events, observed events, or postconditions disagree."""
    parity = program_parity_report(compiled)
    blocked = [
        pid for pid, row in parity["programs"].items()
        if compiled[pid].split == "train" and compiled[pid].training_eligible and not row["match"]
    ]
    observed = None
    if records:
        observed = observed_semantics_report(compiled, records)
        for pid, row in observed["programs"].items():
            spec = compiled.get(pid)
            if spec is not None and spec.split == "train" and spec.training_eligible and not row["match"]:
                if pid not in blocked:
                    blocked.append(pid)
    deferred = [
        pid for pid, spec in compiled.items()
        if spec.split == "train" and (spec.semantic_status == "deferred" or not spec.training_eligible)
    ]
    return {
        "ok": not blocked,
        "blocked": blocked,
        "deferred": deferred,
        "parity": parity,
        "observed": observed,
    }


def split_role(provenance: Mapping[str, Any]) -> str:
    split = provenance.get("split")
    subset = provenance.get("subset") or provenance.get("train_subset")
    if split == "train":
        return subset or "train_core"
    if split in {"dev", "development"}:
        return "development"
    if split == "test":
        return "test"
    return str(split or "unknown")


def split_disjointness_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, set[str]]] = {
        "train_core": {"scene_signature": set(), "asset_instance_id": set(), "episode_id": set(), "asset_family_id": set()},
        "train_val": {"scene_signature": set(), "asset_instance_id": set(), "episode_id": set(), "asset_family_id": set()},
        "development": {"scene_signature": set(), "asset_instance_id": set(), "episode_id": set(), "asset_family_id": set()},
        "test": {"scene_signature": set(), "asset_instance_id": set(), "episode_id": set(), "asset_family_id": set()},
    }
    sources: list[tuple[str, str | None]] = []
    for record in records:
        provenance = record.get("provenance") or record
        role = split_role(provenance)
        if role not in roles:
            continue
        signature = str(provenance.get("scene_signature") or scene_signature(record.get("randomization") or provenance))
        episode_id = provenance.get("episode_id")
        roles[role]["scene_signature"].add(signature)
        if provenance.get("asset_instance_id"):
            roles[role]["asset_instance_id"].add(str(provenance["asset_instance_id"]))
        if episode_id:
            roles[role]["episode_id"].add(str(episode_id))
        if provenance.get("asset_family_id"):
            roles[role]["asset_family_id"].add(str(provenance["asset_family_id"]))
        sources.append((role, provenance.get("source_episode_id") or provenance.get("base_episode_id")))
    names = list(roles)
    overlaps: dict[str, list[str]] = {}
    disjoint = True
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            for key in ("scene_signature", "asset_instance_id"):
                shared = sorted(roles[left][key] & roles[right][key] - {None, ""})
                if shared:
                    disjoint = False
                    overlaps[f"{left}|{right}|{key}"] = shared[:20]
    source_leaks = []
    episode_owner = {epid: role for role, payload in roles.items() for epid in payload["episode_id"]}
    for role, source in sources:
        if source and episode_owner.get(str(source), role) != role:
            disjoint = False
            source_leaks.append({"role": role, "source_episode_id": source, "source_role": episode_owner.get(str(source))})
    train_families = roles["train_core"]["asset_family_id"] | roles["train_val"]["asset_family_id"]
    test_families = roles["test"]["asset_family_id"]
    return {
        "disjoint": disjoint and not source_leaks,
        "overlaps": overlaps,
        "source_episode_leaks": source_leaks,
        "train_test_asset_family_overlap": sorted(train_families & test_families),
    }


def training_eligible_ids(compiled: Mapping[str, CompiledV3Task]) -> tuple[str, ...]:
    return tuple(pid for pid, spec in compiled.items() if spec.training_eligible)


def program_parity_report(compiled: Mapping[str, CompiledV3Task]) -> dict[str, Any]:
    programs: dict[str, Any] = {}
    all_match = True
    for pid, spec in compiled.items():
        steps = structured_steps(pid)
        texts = tuple(step.text for step in steps)
        event_prims = tuple(event["primitive"] for event in spec.events)
        step_prims = tuple(step.primitive for step in steps)
        post = tuple(step.postcondition for step in steps)
        event_post = tuple(event.get("postcondition") for event in spec.events)
        routine_types = tuple(step.get("type") for step in spec.routine)
        push_events = [
            event for event in spec.events
            if event["primitive"] in {"push", "push_through_aperture"}
        ]
        push_steps = [step for step in spec.routine if step.get("type") == "push"]
        push_ok = len(push_events) == len(push_steps)
        match = (
            texts == spec.ordered_steps
            and event_prims == step_prims
            and post == event_post
            and spec.semantic_status != "deferred"
            and push_ok
        )
        programs[pid] = {
            "match": match,
            "ordered_steps": list(spec.ordered_steps),
            "event_primitives": list(event_prims),
            "postconditions": list(post),
            "routine_types": list(routine_types),
            "semantic_status": spec.semantic_status,
            "training_eligible": spec.training_eligible,
            "deferred_reason": spec.deferred_reason,
        }
        all_match = all_match and match
    return {
        "all_primary_match": all_match,
        "programs": programs,
        "training_eligible": list(training_eligible_ids(compiled)),
    }


def observed_semantics_report(
    compiled: Mapping[str, CompiledV3Task],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """catalog steps = compiled events = observed events = postcondition labels."""
    programs: dict[str, Any] = {}
    all_match = True
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["provenance"]["program_id"], []).append(record)
    for pid, spec in compiled.items():
        catalog = tuple(step.primitive for step in structured_steps(pid))
        compiled_prims = tuple(event["primitive"] for event in spec.events)
        compiled_post = tuple(event.get("postcondition") for event in spec.events)
        observed_rows = []
        program_ok = catalog == compiled_prims == tuple(step.primitive for step in structured_steps(pid))
        for record in grouped.get(pid, ()):
            events = record.get("events") or ()
            observed = tuple(event.get("primitive") for event in events)
            observed_post = tuple(event.get("postcondition") for event in events)
            row_ok = bool(observed) and observed == catalog == compiled_prims
            if observed_post and any(observed_post):
                row_ok = row_ok and observed_post == compiled_post
            observed_rows.append({"episode_id": record["provenance"]["episode_id"], "match": row_ok, "observed": list(observed)})
            program_ok = program_ok and row_ok
        if pid in grouped:
            all_match = all_match and program_ok
        programs[pid] = {
            "catalog": list(catalog),
            "compiled": list(compiled_prims),
            "postconditions": list(compiled_post),
            "observed_episodes": observed_rows,
            "match": program_ok if pid in grouped else catalog == compiled_prims,
        }
    return {"all_observed_match": all_match, "programs": programs}


def episode_integrity_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = 0
    errors: list[str] = []
    for record in records:
        try:
            validate_episode_v2(record)
            ok += 1
        except ValueError as exc:
            errors.append(str(exc))
    return {"valid": ok, "invalid": len(errors), "errors": errors[:20]}


def calibration_integrity_report(profile: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "intrinsics", "distortion", "extrinsics", "depth_unit", "pointcloud_frame",
        "camera_intrinsics_id", "camera_extrinsics_id", "action_space_id",
        "orientation_convention", "gripper_unit",
    )
    missing = [key for key in required if key not in profile]
    return {
        "ok": not missing and profile.get("depth_unit") == "meter" and bool(profile.get("pointcloud_frame")),
        "missing": missing,
        "rgb_depth_mask_synchronized": bool(profile.get("rgb_depth_mask_synchronized")),
        "mask_id_to_object_role": profile.get("mask_id_to_object_role"),
        "action_space_id": profile.get("action_space_id"),
        "orientation_convention": profile.get("orientation_convention"),
        "gripper_unit": profile.get("gripper_unit"),
        "wrist_pose_from": (profile.get("extrinsics") or {}).get("world_from"),
    }


def diversity_report(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures = [scene_signature(sample) for sample in samples]
    assets = {sample.get("asset_instance_id") or sample.get("asset_family_id") for sample in samples}
    nearest = []
    for index, sample in enumerate(samples):
        distance = nearest_signature_distance(sample, list(samples[:index]))
        if distance is not None:
            nearest.append(distance)
    yaw = [float(sample.get("object_yaw_deg", 0.0)) for sample in samples]
    scale = [float(sample.get("scale") or sample.get("object_scale_applied") or 1.0) for sample in samples]
    return {
        "unique_scene_signatures": len(set(signatures)),
        "unique_asset_instances": len(assets - {None}),
        "episodes": len(samples),
        "nearest_signature_distance_min": min(nearest) if nearest else None,
        "pose_yaw_scale_coverage": {
            "yaw_min": min(yaw) if yaw else None,
            "yaw_max": max(yaw) if yaw else None,
            "scale_min": min(scale) if scale else None,
            "scale_max": max(scale) if scale else None,
        },
    }


def perturbation_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kinds = Counter()
    outcomes = Counter()
    episode_kinds = Counter()
    missing_intervention = 0
    missing_frame = 0
    core_sigs: set[str] = set()
    val_sigs: set[str] = set()
    counts = QuotaCounts()
    for record in records:
        provenance = record.get("provenance", {})
        kind = provenance.get("episode_kind", "unknown")
        result = provenance.get("outcome") or provenance.get("result_class") or "success"
        episode_kinds[kind] += 1
        outcomes[result] += 1
        intervention = record.get("intervention") or provenance
        intervention_kind = None
        if kind == "perturbed":
            if not intervention.get("intervention_id"):
                missing_intervention += 1
            if "application_scope" not in intervention and "intervention_frame" not in intervention:
                missing_frame += 1
            kinds[intervention.get("intervention_id", "unknown")] += 1
            raw = intervention.get("intervention_id") or ""
            intervention_kind = raw[: -len("_v1")] if raw.endswith("_v1") else raw
        if result not in {SIMULATOR_CRASH, INVALID_OBSERVATION}:
            record_outcome(counts, kind if kind in {"nominal", "perturbed"} else "nominal", result, intervention_kind)
        subset = provenance.get("subset") or provenance.get("train_subset")
        signature = provenance.get("scene_signature") or scene_signature(
            record.get("randomization") or provenance
        )
        if subset == "train_core":
            core_sigs.add(signature)
        elif subset == "train_val":
            val_sigs.add(signature)
    mixture = valid_attempt_mixture(counts)
    return {
        "episode_kind": dict(episode_kinds),
        "outcomes": dict(outcomes),
        "perturbation_kinds": dict(kinds),
        "missing_intervention_metadata": missing_intervention,
        "missing_intervention_frame": missing_frame,
        "nominal_perturbed_ratio": mixture["nominal"] / max(mixture["perturbed"], 1),
        "valid_attempt_mixture": mixture,
        "training_transition_mixture": {
            "target": list(V3_PROTOCOL.warmup_mixture),
            "measured_on": V3_PROTOCOL.mixture_measured_on,
            "views": list(V3_PROTOCOL.mixture_views),
        },
        "train_core_train_val_disjoint": core_sigs.isdisjoint(val_sigs),
        "train_core_val_overlap": sorted(core_sigs & val_sigs),
        "split_disjointness": split_disjointness_report(records),
        "allowed_kinds": list(V3_PROTOCOL.perturbation_kinds),
    }
