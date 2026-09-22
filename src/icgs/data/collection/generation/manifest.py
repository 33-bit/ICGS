"""Build and validate the canonical generation composition manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.collection.generation.steps import GENERATION_PROGRAMS


def build_manifest(source_manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    if not isinstance(source_manifest, dict):
        source_manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    rows = []
    by_id = {row["program_id"]: row for row in source_manifest["catalog"]}
    for program_id, spec in GENERATION_PROGRAMS.items():
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
            "protocol_id": GENERATION_PROTOCOL.controller_protocol_id,
        }
        base["execution_modes"] = list(GENERATION_PROTOCOL.execution_modes)
        randomization = dict(base.get("randomization") or {})
        randomization["camera_profile_id"] = GENERATION_PROTOCOL.camera_profile_id
        base["randomization"] = randomization
        rows.append(base)
    return {
        "manifest_version": GENERATION_PROTOCOL.program_manifest_version,
        "composition_protocol_id": GENERATION_PROTOCOL.composition_protocol_id,
        "dataset_version": GENERATION_PROTOCOL.dataset_version,
        "episode_schema_version": GENERATION_PROTOCOL.episode_schema_version,
        "controller_protocol_id": GENERATION_PROTOCOL.controller_protocol_id,
        "phase": GENERATION_PROTOCOL.phase,
        "preserves": GENERATION_PROTOCOL.preserves,
        "generation_targets": {
            "train_successes_per_program": GENERATION_PROTOCOL.train_nominal_successes_per_program,
            "train_perturbed_attempts_per_program": GENERATION_PROTOCOL.train_perturbed_attempts_per_program,
            "test_contexts_per_composition": GENERATION_PROTOCOL.eval_contexts_per_composition,
            "quota_unit": "attempt",
            "nominal_stop": "200_successful_contexts",
            "perturbed_stop": "80_valid_attempts",
            "mixture_measured_on": GENERATION_PROTOCOL.mixture_measured_on,
            "mixture_views": list(GENERATION_PROTOCOL.mixture_views),
            "eval_perturbed_attempts_per_program": GENERATION_PROTOCOL.eval_perturbed_attempts_per_program,
            "train_on": GENERATION_PROTOCOL.train_on,
            "validation": GENERATION_PROTOCOL.validation,
            "test_is_evaluation_only": True,
            "retain_valid_failures": True,
            "eval_nominal_successes_per_program": GENERATION_PROTOCOL.eval_contexts_per_composition,
            "perturbation_bounds": GENERATION_PROTOCOL.perturbation_bounds(),
            "position_bins": GENERATION_PROTOCOL.position_bin_rule(),
        },
        "catalog": rows,
    }


def write_manifest(path: str | Path, source_manifest_path: str | Path) -> Path:
    path = Path(path)
    payload = build_manifest(source_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
