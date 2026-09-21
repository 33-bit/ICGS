"""Build and validate the additive v3 composition manifest. Does not mutate v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.steps import V3_PROGRAMS


def build_v3_manifest(v2_manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    if not isinstance(v2_manifest, dict):
        v2_manifest = json.loads(Path(v2_manifest).read_text(encoding="utf-8"))
    rows = []
    by_id = {row["program_id"]: row for row in v2_manifest["catalog"]}
    for program_id, spec in V3_PROGRAMS.items():
        base = dict(by_id[program_id])
        base["family"] = spec.family
        base["ordered_steps"] = [step.text for step in spec.steps]
        base["structured_steps"] = [
            {
                "step_id": step.step_id,
                "primitive": step.primitive,
                "object_role": step.object_role,
                "target_role": step.target_role,
                "precondition": step.precondition,
                "success_predicate": step.success_predicate,
                "postcondition": step.postcondition,
                "articulation": step.articulation,
                "kind": step.kind,
            }
            for step in spec.steps
        ]
        base["semantic_status"] = spec.semantic_status
        base["compiler_routine_id"] = spec.compiler_routine_id
        base["future_dependency"] = spec.future_dependency
        base["training_eligible"] = spec.training_eligible
        base["deferred_reason"] = spec.deferred_reason
        base["controller"] = {
            **dict(base.get("controller") or {}),
            "protocol_id": V3_PROTOCOL.controller_protocol_id,
        }
        base["execution_modes"] = list(V3_PROTOCOL.execution_modes)
        randomization = dict(base.get("randomization") or {})
        randomization["camera_profile_id"] = V3_PROTOCOL.camera_profile_id
        base["randomization"] = randomization
        rows.append(base)
    return {
        "manifest_version": V3_PROTOCOL.program_manifest_version,
        "composition_protocol_id": V3_PROTOCOL.composition_protocol_id,
        "dataset_version": V3_PROTOCOL.dataset_version,
        "episode_schema_version": V3_PROTOCOL.episode_schema_version,
        "controller_protocol_id": V3_PROTOCOL.controller_protocol_id,
        "phase": V3_PROTOCOL.phase,
        "preserves": V3_PROTOCOL.preserves,
        "generation_targets": {
            "train_successes_per_program": V3_PROTOCOL.train_nominal_successes_per_program,
            "train_perturbed_attempts_per_program": V3_PROTOCOL.train_perturbed_attempts_per_program,
            "test_contexts_per_composition": V3_PROTOCOL.eval_contexts_per_composition,
            "quota_unit": "attempt",
            "nominal_stop": "200_successful_contexts",
            "perturbed_stop": "80_valid_attempts",
            "mixture_measured_on": V3_PROTOCOL.mixture_measured_on,
            "mixture_views": list(V3_PROTOCOL.mixture_views),
            "eval_perturbed_attempts_per_program": V3_PROTOCOL.eval_perturbed_attempts_per_program,
            "train_on": V3_PROTOCOL.train_on,
            "validation": V3_PROTOCOL.validation,
            "test_is_evaluation_only": True,
            "retain_valid_failures": True,
            "eval_nominal_successes_per_program": V3_PROTOCOL.eval_contexts_per_composition,
            "perturbation_bounds": V3_PROTOCOL.perturbation_bounds(),
            "position_bins": V3_PROTOCOL.position_bin_rule(),
        },
        "catalog": rows,
    }


def write_v3_manifest(path: str | Path, v2_manifest_path: str | Path) -> Path:
    path = Path(path)
    payload = build_v3_manifest(v2_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
