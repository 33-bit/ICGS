"""Explicit, immutable program skeletons for the P02 split protocol.

The catalog preserves proposal IDs and task skeletons only.  It intentionally
does not bind assets, ranges, seeds, controller settings, or predicate
tolerances; those remain G2-dependent protocol inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ProgramSpec:
    """A generator/annotation-only task skeleton, never an online input."""

    program_id: str
    split: str
    steps: tuple[str, ...]
    constraint: str
    family: str | None = None


def _spec(
    program_id: str,
    split: str,
    steps: tuple[str, ...],
    constraint: str,
    family: str | None = None,
) -> ProgramSpec:
    return ProgramSpec(program_id, split, steps, constraint, family)


_SPECS = (
    _spec("T01", "train", ("grasp A", "lift A"), "grasp position, stable hold"),
    _spec("T02", "train", ("grasp A", "place A on pad"), "target relation, pad size"),
    _spec("T03", "train", ("push blocker", "reach target"), "access clearance"),
    _spec("T04", "train", ("open drawer", "close drawer"), "joint limits and final closure"),
    _spec("T05", "train", ("open drawer", "retrieve A"), "reachability after opening"),
    _spec("T06", "train", ("place A into open drawer", "close"), "closing clearance"),
    _spec("T07", "train", ("place A", "place B into open tray"), "shared free space"),
    _spec("T08", "train", ("place B", "place A", "close open drawer"), "ordering and packing"),
    _spec("T09", "train", ("grasp A", "place A into open holder"), "grasp-to-final-pose compatibility"),
    _spec("T10", "train", ("push A through aperture", "place A"), "aperture geometry, no grasp transfer"),
    _spec("T11", "train", ("grasp A", "rotate A", "place A on pad"), "orientation requirement"),
    _spec("T12", "train", ("open gate", "push A through gate", "close gate"), "prerequisite and gate closure"),
    _spec("T13", "train", ("park blocker", "retrieve A"), "parking and access"),
    _spec("T14", "train", ("park blocker", "restore blocker"), "temporary relation and restoration"),
    _spec("T15", "train", ("park A", "park B", "retrieve C"), "two objects share parking space"),
    _spec("T16", "train", ("open drawer", "park blocker", "retrieve A"), "container access"),
    _spec("T17", "train", ("grasp A", "temporary place", "regrasp A"), "regrasp and retry history"),
    _spec("T18", "train", ("retrieve A", "place A", "retrieve B"), "shared retrieval corridor"),
    _spec("T19", "train", ("place spacer", "place A", "close open drawer"), "spacer-dependent packing constraint"),
    _spec("T20", "train", ("park blocker", "retrieve A", "place A on pad"), "parking and transport path"),
    _spec("V01", "development", ("open", "place A", "close"), "selection/calibration composition"),
    _spec("V02", "development", ("grasp", "rotate", "place into holder"), "selection/calibration composition"),
    _spec("V03", "development", ("park", "retrieve", "place", "restore"), "selection/calibration composition"),
    _spec("V04", "development", ("open gate", "retrieve", "close gate"), "selection/calibration composition"),
    _spec("P1", "test", ("open", "place A", "place B", "close"), "pack-add-close", "pack-add-close"),
    _spec("P2", "test", ("open", "place B", "place A", "close"), "order and object-size variation", "pack-add-close"),
    _spec("P3", "test", ("open", "place A", "place spacer", "place B", "close"), "spacer-dependent packing", "pack-add-close"),
    _spec("P4", "test", ("open", "place A", "retrieve C", "place B", "close"), "retrieval before later placement", "pack-add-close"),
    _spec("G1", "test", ("grasp", "transport through aperture", "place into holder"), "aperture and final pose", "grasp-transport-fit"),
    _spec("G2", "test", ("grasp", "rotate in free space", "transport through aperture", "place"), "orientation through aperture", "grasp-transport-fit"),
    _spec("G3", "test", ("open gate", "grasp", "transport", "place", "close gate"), "gate prerequisite and closure", "grasp-transport-fit"),
    _spec("G4", "test", ("grasp", "temporary place", "regrasp", "transport", "fit"), "recovery remains solvable", "grasp-transport-fit"),
    _spec("R1", "test", ("park blocker", "retrieve target", "restore blocker"), "parking preserves later access", "park-retrieve-restore"),
    _spec("R2", "test", ("park A", "park B", "retrieve", "restore B", "restore A"), "restoration order", "park-retrieve-restore"),
    _spec("R3", "test", ("open drawer", "park blocker", "retrieve", "restore", "close"), "container access and closure", "park-retrieve-restore"),
    _spec("R4", "test", ("park blocker", "retrieve target A", "retrieve target B", "restore"), "two target retrievals", "park-retrieve-restore"),
)

if len({spec.program_id for spec in _SPECS}) != len(_SPECS):
    raise RuntimeError("program catalog contains duplicate IDs")

PROGRAM_CATALOG: Mapping[str, ProgramSpec] = MappingProxyType({spec.program_id: spec for spec in _SPECS})


def program_catalog() -> Mapping[str, ProgramSpec]:
    """Return the immutable, insertion-ordered catalog of explicit skeletons."""

    return PROGRAM_CATALOG


def get_program(program_id: str) -> ProgramSpec:
    """Return one explicit program skeleton or reject unknown/noncanonical IDs."""

    if not isinstance(program_id, str) or not program_id.strip() or program_id not in PROGRAM_CATALOG:
        raise ValueError(f"unknown program_id: {program_id!r}")
    return PROGRAM_CATALOG[program_id]


__all__ = ["PROGRAM_CATALOG", "ProgramSpec", "get_program", "program_catalog"]

