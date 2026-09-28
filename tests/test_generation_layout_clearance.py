"""Simulator-free physical clearance of every planned generation attempt."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from icgs.data.collection.generation.attempt_prep import plan_full_generation, prepare_attempt
from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.layout_clearance import audit_prepared_attempt
from icgs.data.collection.generation.perturbations import free_blocker_position

MANIFEST = Path("artifacts/composition/approved_composition_manifest.json")


@lru_cache(maxsize=1)
def _pool_violations():
    rows = {row["program_id"]: row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["catalog"]}
    compiled = compile_generation_catalog(MANIFEST)
    results = []
    for plan in plan_full_generation(rows):
        spec = compiled[plan.program_id]
        prepared = prepare_attempt(plan, spec.objects, spec.routine)
        kind = plan.episode_kind if plan.episode_kind == "nominal" else plan.intervention["kind"]
        results.append((plan.program_id, kind, audit_prepared_attempt(prepared["objects"], prepared["routine"])))
    return results


def test_every_nominal_attempt_is_physically_clear_for_the_panda_fingers():
    offenders = Counter()
    for program_id, kind, violations in _pool_violations():
        if kind == "nominal":
            for item in violations:
                offenders[(program_id, item.phase, item.body, item.other)] += 1
    assert not offenders, sorted(offenders.items())[:20]


def test_no_planned_attempt_spawns_interpenetrating_bodies():
    spawns = Counter()
    for program_id, kind, violations in _pool_violations():
        for item in violations:
            if item.phase == "spawn":
                spawns[(program_id, kind, item.body, item.other)] += 1
    assert not spawns, sorted(spawns.items())[:20]


def test_perturbed_contacts_come_only_from_declared_blocker_insertion():
    unexpected = Counter()
    for program_id, kind, violations in _pool_violations():
        if kind in {"nominal", "blocker_insertion"}:
            continue
        for item in violations:
            unexpected[(program_id, kind, item.phase, item.body, item.other)] += 1
    assert not unexpected, sorted(unexpected.items())[:20]


def test_clearance_audit_flags_a_neighbour_inside_the_open_finger_span():
    objects = {
        "object_a": {"pos": [0.25, 0.0, 0.775], "size": [0.045, 0.045, 0.045]},
        "blocker": {"pos": [0.25, 0.07, 0.775], "size": [0.05, 0.05, 0.05]},
        "target_a": {"pos": [0.25, 0.3, 0.775], "size": [0.06, 0.06, 0.03]},
    }
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    phases = {(item.phase, item.body, item.other) for item in audit_prepared_attempt(objects, routine)}
    assert ("grasp:pick_place", "object_a", "blocker") in phases
    # The same neighbour offset along x clears the thin finger side.
    objects["blocker"]["pos"] = [0.34, 0.0, 0.775]
    assert audit_prepared_attempt(objects, routine) == []


def test_clearance_audit_flags_spawn_interpenetration():
    objects = {
        "object_a": {"pos": [0.25, 0.0, 0.775], "size": [0.045, 0.045, 0.045], "yaw_deg": 30.0},
        "drawer_handle": {"pos": [0.25, 0.04, 0.775], "size": [0.08, 0.03, 0.03]},
    }
    assert any(item.phase == "spawn" for item in audit_prepared_attempt(objects, []))


def test_inserted_blocker_keeps_a_free_declared_position_and_slides_when_occupied():
    free = {"object_a": {"pos": [0.25, 0.2, 0.775], "size": [0.045, 0.045, 0.045]}}
    assert free_blocker_position([0.30, 0.0, 0.775], free, size=0.04) == [0.30, 0.0, 0.775]
    occupied = {"object_a": {"pos": [0.30, 0.0, 0.775], "size": [0.045, 0.045, 0.045]}}
    moved = free_blocker_position([0.30, 0.0, 0.775], occupied, size=0.04)
    assert moved[0] == 0.30 and abs(moved[1]) == pytest.approx(0.05)
    # Markers never block the insertion point.
    marker = {"target_a": {"pos": [0.30, 0.0, 0.775], "size": [0.06, 0.06, 0.03]}}
    assert free_blocker_position([0.30, 0.0, 0.775], marker, size=0.04) == [0.30, 0.0, 0.775]


def test_grasp_yaw_aligns_fingers_with_body_faces():
    from icgs.data.collection.generation.layout_clearance import grasp_yaw_delta_deg

    # Square body: nearest face, never more than 45 degrees of wrist turn.
    assert grasp_yaw_delta_deg([0.05, 0.05], 20.0, 90.0) == pytest.approx(20.0)
    assert grasp_yaw_delta_deg([0.05, 0.05], -30.0, 90.0) == pytest.approx(-30.0)
    assert grasp_yaw_delta_deg([0.05, 0.05], 0.0, 90.0) == pytest.approx(0.0)
    # Elongated body (long local y): fingers close across its short local x side.
    delta = grasp_yaw_delta_deg([0.035, 0.075], -15.3, 90.0)
    assert -90.0 <= delta < 90.0
    assert ((90.0 + delta) - (-15.3)) % 180.0 == pytest.approx(0.0, abs=1e-9)
    # Handles (long local x, no yaw) keep the default world-y finger axis.
    assert grasp_yaw_delta_deg([0.08, 0.03], 0.0, 90.0) == pytest.approx(0.0)
    assert grasp_yaw_delta_deg([0.08, 0.03], 0.0, 70.0) == pytest.approx(20.0)


def test_aligned_fingers_fit_every_planned_grasp_within_the_discrete_hold_limit():
    """Face-aligned grasp widths stay below RLBench Discrete's 0.9 open threshold."""
    import math

    from icgs.data.collection.generation.layout_clearance import ELONGATED_ASPECT_RATIO

    rows = {row["program_id"]: row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["catalog"]}
    compiled = compile_generation_catalog(MANIFEST)
    widest = 0.0
    for plan in plan_full_generation(rows):
        spec = compiled[plan.program_id]
        prepared = prepare_attempt(plan, spec.objects, spec.routine)
        for step in prepared["routine"]:
            if step["type"] not in {"grasp", "pick_place"}:
                continue
            size = prepared["objects"][step["obj"]]["size"]
            short = min(size[0], size[1])
            width = short if max(size[0], size[1]) >= ELONGATED_ASPECT_RATIO * short else max(size[0], size[1])
            widest = max(widest, width)
    # Panda stroke 0.08 m; Discrete treats >0.9 of the stroke as open.
    assert widest < 0.9 * 0.08 - 0.002, widest
    assert not math.isnan(widest)
