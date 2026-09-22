"""Bind an AttemptPlan onto compiled objects/routine. Simulator-free."""

from __future__ import annotations

from typing import Any, Mapping

from icgs.data.collection.generation.batch import (
    AttemptPlan,
    AttemptPlanner,
    apply_layout_randomization,
    plan_program_attempts,
)
from icgs.data.collection.generation.perturbations import apply_perturbation
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.collection.generation.steps import get_generation_program


def objects_as_dicts(objects: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, spec in objects.items():
        if isinstance(spec, dict) and "pos" in spec:
            out[name] = {
                "pos": list(spec["pos"]),
                "size": list(spec.get("size") or [0.04, 0.04, 0.04]),
                "color": list(spec.get("color") or [0.2, 0.2, 0.2]),
                **{k: v for k, v in spec.items() if k not in {"pos", "size", "color"}},
            }
        else:
            pos, size, color = spec
            out[name] = {"pos": list(pos), "size": list(size), "color": list(color)}
    return out


def prepare_attempt(
    plan: AttemptPlan,
    objects: Mapping[str, Any],
    routine: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply scene-seed randomization, then at most one perturbation."""
    layout = apply_layout_randomization(objects_as_dicts(objects), plan.randomization)
    next_routine = [dict(step) for step in routine]
    approach = plan.randomization.get("approach_waypoint") or {}
    release = plan.randomization.get("release_waypoint") or {}
    for step in next_routine:
        if approach:
            step["approach_waypoint"] = dict(approach)
        if release:
            step["release_waypoint"] = dict(release)
    intervention = None
    if plan.episode_kind == "perturbed" and plan.intervention:
        kind = plan.intervention["kind"]
        params = plan.intervention["intervention_params"]
        result = apply_perturbation(kind, params, layout, next_routine)
        layout = result["objects"]
        next_routine = result["routine"]
        intervention = {**plan.intervention, **result["intervention"]}
    return {
        "objects": layout,
        "routine": next_routine,
        "intervention": intervention,
        "plan": plan,
        "episode_kind": plan.episode_kind,
    }


def plan_full_generation(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    collection_seed: int | None = None,
    return_planners: bool = False,
) -> list[AttemptPlan] | tuple[list[AttemptPlan], dict[str, AttemptPlanner]]:
    """Unique-scene pool: train 200 nominal + 80 perturbed; eval 100 + 20 held-out.

    Collection stop is attempt-quota, not this pool length. Extra streamed
    scenes must reuse the returned planners so uniqueness stays global.
    """
    plans: list[AttemptPlan] = []
    planners: dict[str, AttemptPlanner] = {}
    seed = GENERATION_PROTOCOL.collection_seed if collection_seed is None else collection_seed
    for program_id, row in rows_by_id.items():
        spec = get_generation_program(program_id)
        from icgs.data.collection.generation.batch import bounds_from_row

        bounds = bounds_from_row(row)
        family = row.get("asset_family_id")
        n_nominal = (
            GENERATION_PROTOCOL.train_nominal_successes_per_program
            if spec.split == "train"
            else GENERATION_PROTOCOL.eval_contexts_per_composition
        )
        n_perturbed = (
            GENERATION_PROTOCOL.train_perturbed_attempts_per_program
            if spec.split == "train"
            else GENERATION_PROTOCOL.eval_perturbed_attempts_per_program
        )
        planner = AttemptPlanner(
            program_id,
            collection_seed=seed if spec.split == "train" else seed + GENERATION_PROTOCOL.eval_perturbation_seed_offset,
            bounds=bounds,
            asset_family_id=family,
            n_perturbed=n_perturbed,
        )
        plans.extend(
            plan_program_attempts(
                program_id,
                collection_seed=planner.collection_seed,
                n_nominal=n_nominal,
                n_perturbed=n_perturbed,
                bounds=bounds,
                asset_family_id=family,
                planner=planner,
            )
        )
        planners[program_id] = planner
    if return_planners:
        return plans, planners
    return plans
