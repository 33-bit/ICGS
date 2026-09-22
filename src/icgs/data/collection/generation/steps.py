"""Structured generation program steps. Catalog text in programs.py remains the string form."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from icgs.data.collection.programs import get_program, program_catalog


@dataclass(frozen=True)
class ProgramStep:
    step_id: str
    primitive: str
    text: str
    object_role: str | None = None
    target_role: str | None = None
    aperture_role: str | None = None
    precondition: str | None = None
    success_predicate: str | None = None
    postcondition: str | None = None
    articulation: str | None = None
    kind: str = "nominal"


@dataclass(frozen=True)
class ProgramSpec:
    program_id: str
    split: str
    family: str
    steps: tuple[ProgramStep, ...]
    semantic_status: str
    compiler_routine_id: str
    future_dependency: dict[str, Any] | None
    training_eligible: bool
    deferred_reason: str | None = None


def _s(
    index: int,
    primitive: str,
    text: str,
    obj: str | None = None,
    tgt: str | None = None,
    *,
    pre: str | None = None,
    succ: str | None = None,
    post: str | None = None,
    aperture: str | None = None,
    articulation: str | None = None,
    kind: str = "nominal",
) -> ProgramStep:
    return ProgramStep(
        step_id=f"s{index}",
        primitive=primitive,
        text=text,
        object_role=obj,
        target_role=tgt,
        aperture_role=aperture,
        precondition=pre,
        success_predicate=succ,
        postcondition=post,
        articulation=articulation,
        kind=kind,
    )


def _prog(
    program_id: str,
    family: str,
    steps: tuple[ProgramStep, ...],
    future_dependency: dict[str, Any] | None = None,
    semantic_status: str = "authorized",
    deferred_reason: str | None = None,
) -> ProgramSpec:
    catalog = get_program(program_id)
    texts = tuple(step.text for step in steps)
    if texts != catalog.steps:
        raise RuntimeError(f"{program_id} structured texts {texts} != catalog {catalog.steps}")
    from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL

    physics_ok = program_id in GENERATION_PROTOCOL.physics_passed_program_ids
    training_eligible = semantic_status == "authorized" and catalog.split == "train" and physics_ok
    if catalog.split == "train" and semantic_status == "authorized" and not physics_ok:
        deferred_reason = deferred_reason or "physics predicate not yet passed"
    return ProgramSpec(
        program_id=program_id,
        split=catalog.split,
        family=family,
        steps=steps,
        semantic_status=semantic_status,
        compiler_routine_id=f"stepparse-v3/{program_id}",
        future_dependency=future_dependency,
        training_eligible=training_eligible,
        deferred_reason=deferred_reason,
    )


_T18_DEP = {
    "blocker_role": "object_a",
    "blocked_role": "object_b",
    "enabling_step_id": "s2",
    "dependent_step_id": "s3",
    "reason": "object_a occupies the shared retrieval corridor in front of object_b",
}
_P4_DEP = {
    "blocker_role": "object_c",
    "blocked_role": "object_b",
    "enabling_step_id": "s3",
    "dependent_step_id": "s4",
    "reason": "object_c occupies target_b until retrieved",
}

_PROGRAMS: tuple[ProgramSpec, ...] = (
    _prog("T01", "basic-manipulation", (
        _s(1, "grasp", "grasp A", "object_a", pre="object_a_free", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "lift", "lift A", "object_a", "target_a", pre="object_a_grasped", succ="object_a_lifted", post="object_a_lifted"),
    )),
    _prog("T02", "basic-manipulation", (
        _s(1, "grasp", "grasp A", "object_a", pre="object_a_free", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "place", "place A on pad", "object_a", "target_a", pre="object_a_grasped", succ="object_a_in_target", post="object_a_placed"),
    )),
    _prog("T03", "obstacle-access", (
        _s(1, "push", "push blocker", "blocker", "push_target", pre="blocker_blocking", succ="blocker_at_push_target", post="blocker_cleared"),
        _s(2, "reach", "reach target", tgt="target_a", pre="blocker_cleared", succ="ee_at_target", post="target_reached"),
    )),
    _prog("T04", "container-access", (
        _s(1, "open", "open drawer", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "close", "close drawer", "drawer_handle", "close_target", pre="drawer_open", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("T05", "container-access", (
        _s(1, "open", "open drawer", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "retrieve", "retrieve A", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_retrieved"),
    )),
    _prog("T06", "container-access", (
        _s(1, "place", "place A into open drawer", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_placed"),
        _s(2, "close", "close", "drawer_handle", "close_target", pre="object_a_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("T07", "packing", (
        _s(1, "place", "place A", "object_a", "target_a", succ="object_a_in_target", post="object_a_placed"),
        _s(2, "place", "place B into open tray", "object_b", "target_b", pre="object_a_placed", succ="object_b_in_target", post="object_b_placed"),
    )),
    _prog("T08", "packing", (
        _s(1, "place", "place B", "object_b", "target_b", succ="object_b_in_target", post="object_b_placed"),
        _s(2, "place", "place A", "object_a", "target_a", pre="object_b_placed", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "close", "close open drawer", "drawer_handle", "close_target", pre="object_a_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("T09", "grasp-fit", (
        _s(1, "grasp", "grasp A", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "place", "place A into open holder", "object_a", "target_a", pre="object_a_grasped", succ="object_a_in_target", post="object_a_fitted"),
    )),
    _prog("T10", "grasp-fit", (
        _s(1, "push_through_aperture", "push A through aperture", "object_a", "aperture_wp", succ="object_a_past_aperture", post="object_a_past_aperture", aperture="aperture_wp"),
        _s(2, "place", "place A", "object_a", "target_a", pre="object_a_past_aperture", succ="object_a_in_target", post="object_a_placed"),
    )),
    _prog("T11", "orientation", (
        _s(1, "grasp", "grasp A", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "rotate", "rotate A", "object_a", pre="object_a_grasped", succ="object_a_rotated", post="object_a_rotated"),
        _s(3, "place", "place A on pad", "object_a", "target_a", pre="object_a_rotated", succ="object_a_in_target", post="object_a_placed"),
    )),
    _prog("T12", "gated-access", (
        _s(1, "open", "open gate", "gate_handle", "open_target", succ="gate_open", post="gate_open", articulation="open"),
        _s(2, "push_through_aperture", "push A through gate", "object_a", "target_a", pre="gate_open", succ="object_a_past_gate", post="object_a_past_gate", aperture="gate_wp"),
        _s(3, "close", "close gate", "gate_handle", "close_target", pre="object_a_past_gate", succ="gate_closed", post="gate_closed", articulation="closed"),
    )),
    _prog("T13", "park-retrieve", (
        _s(1, "park", "park blocker", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "retrieve", "retrieve A", "object_a", "target_a", pre="blocker_parked", succ="object_a_in_target", post="object_a_retrieved"),
    )),
    _prog("T14", "park-restore", (
        _s(1, "park", "park blocker", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "restore", "restore blocker", "blocker", "restore_target", pre="blocker_parked", succ="blocker_restored", post="blocker_restored"),
    )),
    _prog("T15", "park-retrieve", (
        _s(1, "park", "park A", "blocker_a", "park_target_a", succ="blocker_a_parked", post="blocker_a_parked"),
        _s(2, "park", "park B", "blocker_b", "park_target_b", pre="blocker_a_parked", succ="blocker_b_parked", post="blocker_b_parked"),
        _s(3, "retrieve", "retrieve C", "object_c", "target_c", pre="blocker_b_parked", succ="object_c_in_target", post="object_c_retrieved"),
    )),
    _prog("T16", "container-access", (
        _s(1, "open", "open drawer", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "park", "park blocker", "blocker", "park_target", pre="drawer_open", succ="blocker_parked", post="blocker_parked"),
        _s(3, "retrieve", "retrieve A", "object_a", "target_a", pre="blocker_parked", succ="object_a_in_target", post="object_a_retrieved"),
    )),
    _prog("T17", "regrasp", (
        _s(1, "grasp", "grasp A", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "temporary_place", "temporary place", "object_a", "temp_target", pre="object_a_grasped", succ="object_a_in_target", post="object_a_temp_placed"),
        _s(3, "regrasp", "regrasp A", "object_a", "target_a", pre="object_a_temp_placed", succ="object_a_in_target", post="object_a_regrasped", kind="nominal"),
    )),
    _prog("T18", "shared-corridor", (
        _s(1, "retrieve", "retrieve A", "object_a", "target_a", succ="object_a_in_target", post="object_a_retrieved"),
        _s(2, "place", "place A", "object_a", "target_a", pre="object_a_retrieved", succ="object_a_in_target", post="corridor_clear"),
        _s(3, "retrieve", "retrieve B", "object_b", "target_b", pre="corridor_clear", succ="object_b_in_target", post="object_b_retrieved"),
    ), future_dependency=_T18_DEP),
    _prog("T19", "packing", (
        _s(1, "place", "place spacer", "spacer", "target_spacer", succ="spacer_in_target", post="spacer_placed"),
        _s(2, "place", "place A", "object_a", "target_a", pre="spacer_placed", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "close", "close open drawer", "drawer_handle", "close_target", pre="object_a_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("T20", "park-retrieve", (
        _s(1, "park", "park blocker", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "retrieve", "retrieve A", "object_a", "target_a", pre="blocker_parked", succ="object_a_grasped", post="object_a_retrieved"),
        _s(3, "place", "place A on pad", "object_a", "target_a", pre="object_a_retrieved", succ="object_a_in_target", post="object_a_placed"),
    )),
    _prog("V01", "pack-add-close", (
        _s(1, "open", "open", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "place", "place A", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "close", "close", "drawer_handle", "close_target", pre="object_a_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("V02", "grasp-transport-fit", (
        _s(1, "grasp", "grasp", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "rotate", "rotate", "object_a", pre="object_a_grasped", succ="object_a_rotated", post="object_a_rotated"),
        _s(3, "place", "place into holder", "object_a", "target_a", pre="object_a_rotated", succ="object_a_in_target", post="object_a_fitted"),
    )),
    _prog("V03", "park-retrieve-restore", (
        _s(1, "park", "park", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "retrieve", "retrieve", "object_a", "target_a", pre="blocker_parked", succ="object_a_grasped", post="object_a_retrieved"),
        _s(3, "place", "place", "object_a", "target_a", pre="object_a_retrieved", succ="object_a_in_target", post="object_a_placed"),
        _s(4, "restore", "restore", "blocker", "restore_target", pre="object_a_placed", succ="blocker_restored", post="blocker_restored"),
    )),
    _prog("V04", "gated-access", (
        _s(1, "open", "open gate", "gate_handle", "open_target", succ="gate_open", post="gate_open", articulation="open"),
        _s(2, "retrieve", "retrieve", "object_a", "target_a", pre="gate_open", succ="object_a_in_target", post="object_a_retrieved"),
        _s(3, "close", "close gate", "gate_handle", "close_target", pre="object_a_retrieved", succ="gate_closed", post="gate_closed", articulation="closed"),
    )),
    _prog("P1", "pack-add-close", (
        _s(1, "open", "open", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "place", "place A", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "place", "place B", "object_b", "target_b", pre="object_a_placed", succ="object_b_in_target", post="object_b_placed"),
        _s(4, "close", "close", "drawer_handle", "close_target", pre="object_b_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("P2", "pack-add-close", (
        _s(1, "open", "open", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "place", "place B", "object_b", "target_b", pre="drawer_open", succ="object_b_in_target", post="object_b_placed"),
        _s(3, "place", "place A", "object_a", "target_a", pre="object_b_placed", succ="object_a_in_target", post="object_a_placed"),
        _s(4, "close", "close", "drawer_handle", "close_target", pre="object_a_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("P3", "pack-add-close", (
        _s(1, "open", "open", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "place", "place A", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "place", "place spacer", "spacer", "target_spacer", pre="object_a_placed", succ="spacer_in_target", post="spacer_placed"),
        _s(4, "place", "place B", "object_b", "target_b", pre="spacer_placed", succ="object_b_in_target", post="object_b_placed"),
        _s(5, "close", "close", "drawer_handle", "close_target", pre="object_b_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("P4", "pack-add-close", (
        _s(1, "open", "open", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "place", "place A", "object_a", "target_a", pre="drawer_open", succ="object_a_in_target", post="object_a_placed"),
        _s(3, "retrieve", "retrieve C", "object_c", "target_c", pre="object_a_placed", succ="object_c_in_target", post="slot_b_clear"),
        _s(4, "place", "place B", "object_b", "target_b", pre="slot_b_clear", succ="object_b_in_target", post="object_b_placed"),
        _s(5, "close", "close", "drawer_handle", "close_target", pre="object_b_placed", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    ), future_dependency=_P4_DEP),
    _prog("G1", "grasp-transport-fit", (
        _s(1, "grasp", "grasp", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "transport_through_aperture", "transport through aperture", "object_a", "target_a", pre="object_a_grasped", succ="object_a_past_aperture", post="object_a_past_aperture", aperture="aperture_wp"),
        _s(3, "place", "place into holder", "object_a", "target_a", pre="object_a_past_aperture", succ="object_a_in_target", post="object_a_fitted"),
    )),
    _prog("G2", "grasp-transport-fit", (
        _s(1, "grasp", "grasp", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "rotate", "rotate in free space", "object_a", pre="object_a_grasped", succ="object_a_rotated", post="object_a_rotated"),
        _s(3, "transport_through_aperture", "transport through aperture", "object_a", "target_a", pre="object_a_rotated", succ="object_a_past_aperture", post="object_a_past_aperture", aperture="aperture_wp"),
        _s(4, "place", "place", "object_a", "target_a", pre="object_a_past_aperture", succ="object_a_in_target", post="object_a_placed"),
    )),
    _prog("G3", "grasp-transport-fit", (
        _s(1, "open", "open gate", "gate_handle", "open_target", succ="gate_open", post="gate_open", articulation="open"),
        _s(2, "grasp", "grasp", "object_a", pre="gate_open", succ="object_a_grasped", post="object_a_grasped"),
        _s(3, "transport_through_aperture", "transport", "object_a", "target_a", pre="object_a_grasped", succ="object_a_past_gate", post="object_a_past_gate", aperture="gate_wp"),
        _s(4, "place", "place", "object_a", "target_a", pre="object_a_past_gate", succ="object_a_in_target", post="object_a_placed"),
        _s(5, "close", "close gate", "gate_handle", "close_target", pre="object_a_placed", succ="gate_closed", post="gate_closed", articulation="closed"),
    )),
    _prog("G4", "grasp-transport-fit", (
        _s(1, "grasp", "grasp", "object_a", succ="object_a_grasped", post="object_a_grasped"),
        _s(2, "temporary_place", "temporary place", "object_a", "temp_target", pre="object_a_grasped", succ="object_a_in_target", post="object_a_temp_placed"),
        _s(3, "regrasp", "regrasp", "object_a", pre="object_a_temp_placed", succ="object_a_grasped", post="object_a_regrasped", kind="nominal"),
        _s(4, "transport_through_aperture", "transport", "object_a", "target_a", pre="object_a_regrasped", succ="object_a_transported", post="object_a_transported"),
        _s(5, "place", "fit", "object_a", "target_a", pre="object_a_transported", succ="object_a_in_target", post="object_a_fitted"),
    )),
    _prog("R1", "park-retrieve-restore", (
        _s(1, "park", "park blocker", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "retrieve", "retrieve target", "object_a", "target_a", pre="blocker_parked", succ="object_a_in_target", post="object_a_retrieved"),
        _s(3, "restore", "restore blocker", "blocker", "restore_target", pre="object_a_retrieved", succ="blocker_restored", post="blocker_restored"),
    )),
    _prog("R2", "park-retrieve-restore", (
        _s(1, "park", "park A", "blocker_a", "park_target_a", succ="blocker_a_parked", post="blocker_a_parked"),
        _s(2, "park", "park B", "blocker_b", "park_target_b", pre="blocker_a_parked", succ="blocker_b_parked", post="blocker_b_parked"),
        _s(3, "retrieve", "retrieve", "object_a", "target_a", pre="blocker_b_parked", succ="object_a_in_target", post="object_a_retrieved"),
        _s(4, "restore", "restore B", "blocker_b", "restore_target_b", pre="object_a_retrieved", succ="blocker_b_restored", post="blocker_b_restored"),
        _s(5, "restore", "restore A", "blocker_a", "restore_target_a", pre="blocker_b_restored", succ="blocker_a_restored", post="blocker_a_restored"),
    )),
    _prog("R3", "park-retrieve-restore", (
        _s(1, "open", "open drawer", "drawer_handle", "open_target", succ="drawer_open", post="drawer_open", articulation="open"),
        _s(2, "park", "park blocker", "blocker", "park_target", pre="drawer_open", succ="blocker_parked", post="blocker_parked"),
        _s(3, "retrieve", "retrieve", "object_a", "target_a", pre="blocker_parked", succ="object_a_in_target", post="object_a_retrieved"),
        _s(4, "restore", "restore", "blocker", "restore_target", pre="object_a_retrieved", succ="blocker_restored", post="blocker_restored"),
        _s(5, "close", "close", "drawer_handle", "close_target", pre="blocker_restored", succ="drawer_closed", post="drawer_closed", articulation="closed"),
    )),
    _prog("R4", "park-retrieve-restore", (
        _s(1, "park", "park blocker", "blocker", "park_target", succ="blocker_parked", post="blocker_parked"),
        _s(2, "retrieve", "retrieve target A", "object_a", "target_a", pre="blocker_parked", succ="object_a_in_target", post="object_a_retrieved"),
        _s(3, "retrieve", "retrieve target B", "object_b", "target_b", pre="object_a_retrieved", succ="object_b_in_target", post="object_b_retrieved"),
        _s(4, "restore", "restore", "blocker", "restore_target", pre="object_b_retrieved", succ="blocker_restored", post="blocker_restored"),
    )),
)

GENERATION_PROGRAMS: Mapping[str, ProgramSpec] = MappingProxyType({spec.program_id: spec for spec in _PROGRAMS})
PROGRAM_FAMILIES: Mapping[str, str] = MappingProxyType({spec.program_id: spec.family for spec in _PROGRAMS})


def structured_steps(program_id: str) -> tuple[ProgramStep, ...]:
    if program_id not in GENERATION_PROGRAMS:
        raise ValueError(f"unknown generation program_id: {program_id!r}")
    return GENERATION_PROGRAMS[program_id].steps


def get_generation_program(program_id: str) -> ProgramSpec:
    if program_id not in GENERATION_PROGRAMS:
        raise ValueError(f"unknown generation program_id: {program_id!r}")
    return GENERATION_PROGRAMS[program_id]


def assert_catalog_coverage() -> None:
    expected = {pid for pid, spec in program_catalog().items() if spec.split in {"train", "development", "test"}}
    actual = set(GENERATION_PROGRAMS)
    if expected != actual:
        raise RuntimeError(f"generation catalog coverage mismatch missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


assert_catalog_coverage()
