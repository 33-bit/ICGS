"""Plan phase-1 collection attempts. No simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from icgs.data.collection.v3.diversity import (
    episode_seed,
    is_duplicate_episode,
    nearest_signature_distance,
    register_episode,
    sample_randomization,
    scene_signature,
    train_subset_for_sample,
)
from icgs.data.collection.v3.perturbations import (
    applicable_perturbations,
    intervention_application_scope,
    perturbation_quota,
)
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.steps import get_v3_program


@dataclass(frozen=True)
class AttemptPlan:
    program_id: str
    split: str
    episode_index: int
    episode_id: str
    episode_kind: str
    scene_seed: int
    collection_seed: int
    randomization: dict[str, Any]
    intervention: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bounds_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    randomization = row["randomization"]
    return {
        "translation_m": randomization["translation_m"],
        "yaw_deg": tuple(randomization["yaw_deg"]),
        "scale": tuple(randomization["scale"]),
        "camera_profile_id": randomization.get("camera_profile_id", V3_PROTOCOL.camera_profile_id),
        "lighting_profile_id": randomization.get("lighting_profile_id", V3_PROTOCOL.lighting_profile_id),
    }


def sample_intervention(
    program_id: str,
    collection_seed: int,
    episode_index: int,
    kind: str,
    *,
    held_out: bool = False,
) -> dict[str, Any]:
    seed = episode_seed(program_id, collection_seed, episode_index)
    unit = (seed % 10000) / 10000.0
    signed = unit * 2.0 - 1.0
    signed_y = ((seed // 17) % 10000) / 10000.0 * 2.0 - 1.0
    bucket = "outer" if held_out else "inner"
    if kind == "action_pose_offset":
        params = {
            "dx_m": round(_bucket_magnitude(signed, V3_PROTOCOL.target_offset_m, bucket), 4),
            "dy_m": round(_bucket_magnitude(signed_y, V3_PROTOCOL.target_offset_m, bucket), 4),
            "d_yaw_deg": round(_bucket_magnitude(signed, V3_PROTOCOL.target_rotation_offset_deg, bucket), 3),
        }
    elif kind == "gripper_timing":
        params = {"delta_intervals": -1 if seed % 2 == 0 else 1}
    elif kind == "object_displacement":
        params = {
            "dx_m": round(_bucket_magnitude(signed, V3_PROTOCOL.object_shift_m, bucket), 4),
            "dy_m": 0.0,
            "object_role": "object_a",
        }
    elif kind == "blocker_insertion":
        params = {
            "pos": [0.30, round(_bucket_magnitude(signed, 0.05, bucket), 4), 0.775],
            "blocker_role": "inserted_blocker",
        }
    elif kind == "pause_hold":
        lo, hi = V3_PROTOCOL.pause_intervals
        params = {"intervals": lo + (seed % (hi - lo + 1))}
    else:
        raise ValueError(f"unsupported perturbation kind: {kind}")
    scope = intervention_application_scope(kind)
    external = kind in {"object_displacement", "blocker_insertion"}
    return {
        "episode_kind": "perturbed",
        "intervention_type": kind,
        "intervention_id": f"{kind}_v1",
        "intervention_seed": seed,
        "intervention_params": params,
        "application_scope": scope,
        "application_t": None,
        "intervention_frame": None,
        "source_episode_id": None,
        "base_episode_id": None,
        "source_frame": None,
        "pre_state_hash": None,
        "post_state_hash": None,
        "external_intervention": external,
        "episode_has_external_intervention": external,
        "magnitude_bucket": bucket,
        "held_out": held_out,
        "kind": kind,
    }


def _bucket_magnitude(signed: float, bound: float, bucket: str) -> float:
    """Train inner is 0 <= r < 0.70; eval outer is 0.70 <= r <= 1.00."""
    inner = V3_PROTOCOL.perturbation_inner_fraction
    sign = 1.0 if signed >= 0 else -1.0
    if bucket == "outer":
        ratio = inner + (1.0 - inner) * abs(signed)
    else:
        ratio = abs(signed) * inner
        if ratio >= inner:
            ratio = inner - 1e-6
    value = round(sign * ratio * bound, 4)
    limit = round(inner * bound, 4)
    if bucket == "inner" and abs(value) >= limit:
        value = round(sign * (limit - 10 ** (-4)), 4)
    if bucket == "outer" and abs(value) < limit:
        value = sign * limit
    return value


def _perturbed_kind_list(program_id: str, n_perturbed: int) -> list[str]:
    quota = perturbation_quota(program_id, total=n_perturbed)
    kinds: list[str] = []
    for kind, count in quota.items():
        if isinstance(count, int):
            kinds.extend([kind] * count)
    if len(kinds) < n_perturbed:
        applicable = list(applicable_perturbations(program_id))
        while len(kinds) < n_perturbed:
            kinds.append(applicable[len(kinds) % len(applicable)])
    return kinds[:n_perturbed]


class AttemptPlanner:
    """Stream unique AttemptPlans. Collection stop is quota, not this pool size."""

    def __init__(
        self,
        program_id: str,
        *,
        collection_seed: int | None = None,
        bounds: Mapping[str, Any],
        asset_family_id: str | None = None,
        n_perturbed: int | None = None,
    ) -> None:
        self.program_id = program_id
        self.spec = get_v3_program(program_id)
        self.collection_seed = V3_PROTOCOL.collection_seed if collection_seed is None else collection_seed
        self.bounds = bounds
        self.asset_family_id = asset_family_id or f"{program_id}-family"
        if n_perturbed is None:
            n_perturbed = (
                V3_PROTOCOL.train_perturbed_attempts_per_program
                if self.spec.split == "train"
                else V3_PROTOCOL.eval_perturbed_attempts_per_program
            )
        self._perturbed_kinds = _perturbed_kind_list(program_id, n_perturbed) if n_perturbed else []
        self._kind_cursor = 0
        self.index = 0
        self.seen: dict[str, set[int]] = {}
        self.previous: list[dict[str, Any]] = []
        self._nominal: list[AttemptPlan] = []
        self._scene_bucket = "eval" if self.spec.split != "train" else "train"

    def next_plan(self, episode_kind: str) -> AttemptPlan:
        if episode_kind not in {"nominal", "perturbed"}:
            raise ValueError(f"unsupported episode_kind: {episode_kind}")
        if episode_kind == "perturbed" and not self._perturbed_kinds:
            raise ValueError(f"no perturbed kinds for {self.program_id}")
        if episode_kind == "perturbed" and self._nominal:
            base = self._nominal[self._kind_cursor % len(self._nominal)]
            if base.randomization.get("object_translation_m"):
                return self._perturbed_from_nominal()
        return self._unique_plan(episode_kind)

    def ingest_plans(self, plans: list[AttemptPlan]) -> None:
        """Load a unique-scene pool so streamed retries cannot repeat those scenes."""
        for plan in plans:
            self.remember_plan(plan)

    def remember_plan(self, plan: AttemptPlan) -> None:
        sample = {
            **plan.randomization,
            "asset_family_id": plan.randomization.get("asset_family_id") or self.asset_family_id,
            "camera_profile_id": plan.randomization.get("camera_profile_id"),
            "program_id": plan.program_id,
            "split": plan.split,
        }
        register_episode(sample, self.seen)
        self.previous.append(sample)
        if plan.episode_kind == "nominal" and plan.episode_id not in {item.episode_id for item in self._nominal}:
            self._nominal.append(plan)
        self.index = max(self.index, int(plan.episode_index) + 1)

    def remember_row(self, row: Mapping[str, Any]) -> None:
        """Resume from a dataset manifest row that may only have a signature."""
        signature = row.get("scene_signature")
        seed = int(row.get("scene_seed") or 0)
        if signature:
            self.seen.setdefault(str(signature), set()).add(seed)
        self.index = max(self.index, int(row.get("episode_index") or self.index) + 1)

    def _unique_plan(self, episode_kind: str) -> AttemptPlan:
        attempts = 0
        while True:
            randomization = sample_randomization(
                self.program_id,
                self.collection_seed,
                self.index,
                self.bounds,
                scene_bucket=self._scene_bucket,
            )
            randomization["program_id"] = self.program_id
            randomization["split"] = self.spec.split
            randomization["asset_family_id"] = self.asset_family_id
            sample = {
                **randomization,
                "asset_family_id": randomization["asset_family_id"],
                "camera_profile_id": randomization["camera_profile_id"],
            }
            if is_duplicate_episode(sample, self.seen):
                self.index += 1
                attempts += 1
                if attempts > 400:
                    raise RuntimeError(f"could not find unique scenes for {self.program_id}")
                continue
            register_episode(sample, self.seen)
            randomization["nearest_scene_distance_m"] = nearest_signature_distance(sample, self.previous)
            randomization["scene_signature"] = scene_signature(sample)
            randomization["train_subset"] = train_subset_for_sample(sample)
            self.previous.append(sample)
            intervention = None
            if episode_kind == "perturbed":
                intervention = self._make_intervention(None)
            plan = self._build_plan(episode_kind, randomization, intervention)
            if episode_kind == "nominal":
                self._nominal.append(plan)
            return plan

    def _perturbed_from_nominal(self) -> AttemptPlan:
        base = self._nominal[self._kind_cursor % len(self._nominal)]
        randomization = deepcopy(base.randomization)
        intervention = self._make_intervention(base.episode_id)
        return self._build_plan("perturbed", randomization, intervention)

    def _make_intervention(self, source_episode_id: str | None) -> dict[str, Any]:
        kind = self._perturbed_kinds[self._kind_cursor % len(self._perturbed_kinds)]
        self._kind_cursor += 1
        intervention = sample_intervention(
            self.program_id,
            self.collection_seed,
            self.index,
            kind,
            held_out=self.spec.split != "train",
        )
        intervention["source_episode_id"] = source_episode_id
        intervention["base_episode_id"] = source_episode_id
        return intervention

    def _build_plan(
        self,
        episode_kind: str,
        randomization: dict[str, Any],
        intervention: dict[str, Any] | None,
    ) -> AttemptPlan:
        plan = AttemptPlan(
            program_id=self.program_id,
            split=self.spec.split,
            episode_index=self.index,
            episode_id=f"v3-{self.program_id.lower()}-{self.index:05d}",
            episode_kind=episode_kind,
            scene_seed=int(randomization["scene_seed"]),
            collection_seed=self.collection_seed,
            randomization=randomization,
            intervention=intervention,
        )
        self.index += 1
        return plan


def plan_program_attempts(
    program_id: str,
    *,
    collection_seed: int | None = None,
    n_nominal: int | None = None,
    n_perturbed: int | None = None,
    bounds: Mapping[str, Any],
    asset_family_id: str | None = None,
    planner: AttemptPlanner | None = None,
) -> list[AttemptPlan]:
    spec = get_v3_program(program_id)
    if spec.split != "train" and n_nominal is None:
        raise ValueError("phase-1 training attempts are only planned for train programs")
    n_nominal = V3_PROTOCOL.train_nominal_successes_per_program if n_nominal is None else n_nominal
    n_perturbed = V3_PROTOCOL.train_perturbed_attempts_per_program if n_perturbed is None else n_perturbed
    if planner is None:
        planner = AttemptPlanner(
            program_id,
            collection_seed=collection_seed,
            bounds=bounds,
            asset_family_id=asset_family_id,
            n_perturbed=n_perturbed,
        )
    plans = [planner.next_plan("nominal") for _ in range(n_nominal)]
    plans.extend(planner.next_plan("perturbed") for _ in range(n_perturbed))
    return plans


def plan_pilot_batch(rows_by_id: Mapping[str, Mapping[str, Any]]) -> list[AttemptPlan]:
    plans: list[AttemptPlan] = []
    for program_id in V3_PROTOCOL.pilot_program_ids:
        row = rows_by_id[program_id]
        plans.extend(
            plan_program_attempts(
                program_id,
                n_nominal=V3_PROTOCOL.pilot_nominal_per_program,
                n_perturbed=V3_PROTOCOL.pilot_perturbed_per_program,
                bounds=bounds_from_row(row),
                asset_family_id=row.get("asset_family_id"),
            )
        )
    return plans


def attempt_from_dict(payload: Mapping[str, Any]) -> AttemptPlan:
    return AttemptPlan(
        program_id=str(payload["program_id"]),
        split=str(payload["split"]),
        episode_index=int(payload["episode_index"]),
        episode_id=str(payload["episode_id"]),
        episode_kind=str(payload["episode_kind"]),
        scene_seed=int(payload["scene_seed"]),
        collection_seed=int(payload["collection_seed"]),
        randomization=dict(payload["randomization"]),
        intervention=dict(payload["intervention"]) if payload.get("intervention") else None,
    )


def provenance_from_plan(plan: AttemptPlan, *, binding: Mapping[str, Any]) -> dict[str, Any]:
    split = plan.split
    if split == "development":
        split = "dev"
    provenance = {
        "attempt_id": f"att-{plan.episode_id}",
        "episode_id": plan.episode_id,
        "program_id": plan.program_id,
        "program_semantics_version": V3_PROTOCOL.program_semantics_version,
        "split": split,
        "scene_seed": plan.scene_seed,
        "asset_instance_id": plan.randomization["asset_instance_id"],
        "execution_mode": V3_PROTOCOL.execution_mode,
        "execution_source": V3_PROTOCOL.execution_mode,
        "calibration_id": V3_PROTOCOL.camera_profile_id,
        "camera_intrinsics_id": f"{V3_PROTOCOL.camera_profile_id}-intrinsics",
        "camera_extrinsics_id": f"{V3_PROTOCOL.camera_profile_id}-extrinsics",
        "action_space_id": V3_PROTOCOL.action_space_id,
        "orientation_convention": V3_PROTOCOL.orientation_convention,
        "gripper_unit": V3_PROTOCOL.gripper_unit,
        "observation_origin": "measured",
        "raw_commands_id": f"raw-{plan.episode_id}",
        "materialized_commands_id": f"mat-{plan.episode_id}",
        "episode_kind": plan.episode_kind,
        "source_lineage_id": binding.get("source_lineage_id", f"{plan.program_id}-lineage"),
        "asset_family_id": binding.get("asset_family_id", plan.randomization.get("asset_family_id")),
        "dataset_version": V3_PROTOCOL.dataset_version,
        "collection_seed": plan.collection_seed,
        "episode_index": plan.episode_index,
        "camera_profile_id": plan.randomization.get("camera_profile_id"),
        "lighting_profile_id": plan.randomization.get("lighting_profile_id"),
        "controller_version": V3_PROTOCOL.controller_protocol_id,
        "simulator_version": V3_PROTOCOL.simulator_version,
        "physics_engine_version": V3_PROTOCOL.physics_engine_version,
        "predicate_protocol_id": V3_PROTOCOL.predicate_protocol_id,
        "program_manifest_version": V3_PROTOCOL.program_manifest_version,
        "scene_signature": plan.randomization.get("scene_signature") or scene_signature(plan.randomization),
    }
    for field in V3_PROTOCOL.phase2_reserved_fields:
        provenance[field] = None
    if plan.split == "train":
        subset = plan.randomization.get("train_subset") or train_subset_for_sample(plan.randomization)
        provenance["subset"] = subset
        provenance["train_subset"] = subset
    else:
        provenance["subset"] = None
    if plan.intervention:
        provenance["intervention_id"] = plan.intervention["intervention_id"]
        provenance["intervention_params"] = plan.intervention["intervention_params"]
        provenance["intervention_seed"] = plan.intervention.get("intervention_seed")
        provenance["intervention_type"] = plan.intervention.get("intervention_type") or plan.intervention.get("kind")
        provenance["application_scope"] = plan.intervention.get("application_scope")
        provenance["application_t"] = plan.intervention.get("application_t")
        provenance["intervention_frame"] = plan.intervention.get("application_t")
        episode_external = bool(
            plan.intervention.get("episode_has_external_intervention", plan.intervention.get("external_intervention"))
        )
        provenance["episode_has_external_intervention"] = episode_external
        provenance["external_intervention"] = episode_external
        provenance["source_episode_id"] = plan.intervention.get("source_episode_id")
        provenance["base_episode_id"] = plan.intervention.get("base_episode_id")
        provenance["magnitude_bucket"] = plan.intervention.get("magnitude_bucket")
        provenance["held_out"] = bool(plan.intervention.get("held_out", False))
        provenance["initial_scene_intervention_id"] = (
            plan.intervention.get("intervention_id")
            if plan.intervention.get("application_scope") == "initial_scene"
            else None
        )
    else:
        provenance["episode_has_external_intervention"] = False
        provenance["initial_scene_intervention_id"] = None
    return provenance


_MOVABLE_NAMES = frozenset({"blocker", "spacer", "inserted_blocker"})
_FIXTURE_TOKENS = (
    "target", "pad", "tray", "holder", "drawer", "gate", "handle",
    "park", "restore", "aperture", "wp", "open_", "close_",
)


def _is_movable(name: str) -> bool:
    lower = name.lower()
    if lower.startswith("object_") or lower in _MOVABLE_NAMES:
        return True
    return not any(token in lower for token in _FIXTURE_TOKENS)


def apply_layout_randomization(objects: Mapping[str, Any], randomization: Mapping[str, Any]) -> dict[str, Any]:
    """Apply workspace translation, relative object gap, yaw and scale. Simulator-free."""
    dx = float(randomization["object_translation_m"]["x"])
    dy = float(randomization["object_translation_m"]["y"])
    dz = float(randomization["object_translation_m"]["z"])
    gap = randomization.get("inter_object_gap_m") or {}
    gx = float(gap.get("dx_m", 0.0))
    gy = float(gap.get("dy_m", 0.0))
    yaw = randomization.get("object_yaw_deg")
    scale = randomization.get("object_scale_applied")
    if scale is None:
        scale = randomization.get("scale")
    shifted: dict[str, Any] = {}
    for name, spec in objects.items():
        extra_x = gx if _is_movable(name) else 0.0
        extra_y = gy if _is_movable(name) else 0.0
        if isinstance(spec, dict) and "pos" in spec:
            pos = list(spec["pos"])
            next_spec = dict(spec)
            next_spec["pos"] = [pos[0] + dx + extra_x, pos[1] + dy + extra_y, pos[2] + dz]
            if _is_movable(name) and scale is not None:
                size = list(next_spec.get("size") or [0.04, 0.04, 0.04])
                next_spec["size"] = [float(v) * float(scale) for v in size]
                next_spec["scale_applied"] = float(scale)
            if _is_movable(name) and yaw is not None:
                next_spec["yaw_deg"] = float(yaw)
            shifted[name] = next_spec
        else:
            pos, size, color = spec
            next_size = list(size)
            extras: dict[str, Any] = {}
            if _is_movable(name) and scale is not None:
                next_size = [float(v) * float(scale) for v in size]
                extras["scale_applied"] = float(scale)
            if _is_movable(name) and yaw is not None:
                extras["yaw_deg"] = float(yaw)
            if extras:
                shifted[name] = {
                    "pos": [pos[0] + dx + extra_x, pos[1] + dy + extra_y, pos[2] + dz],
                    "size": next_size,
                    "color": list(color),
                    **extras,
                }
            else:
                shifted[name] = ([pos[0] + dx + extra_x, pos[1] + dy + extra_y, pos[2] + dz], size, color)
    _refresh_rest_heights(shifted)
    return shifted


def _refresh_rest_heights(objects: dict[str, Any]) -> None:
    support = None
    for name in ("pad", "tray", "holder", "drawer"):
        spec = objects.get(name)
        if isinstance(spec, dict) and "pos" in spec and "size" in spec:
            support = spec
            break
    if support is None:
        return
    top = float(support["pos"][2]) + float(support["size"][2]) / 2.0
    for name, spec in objects.items():
        if not name.startswith("target") or not isinstance(spec, dict) or "pos" not in spec:
            continue
        suffix = name[len("target_"):]
        obj_name = "spacer" if suffix == "spacer" else f"object_{suffix}"
        obj = objects.get(obj_name) or objects.get("object_a")
        half = 0.0225
        if isinstance(obj, dict) and obj.get("size"):
            half = float(obj["size"][2]) / 2.0
        elif isinstance(obj, (list, tuple)) and len(obj) > 1:
            half = float(obj[1][2]) / 2.0
        spec["pos"][2] = top + half
