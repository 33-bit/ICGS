# Physical Dependency Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate-ready collector in which every program's declared constraint is enforced by simulated physics and the recorded data follows the proposal method's semantics.

**Architecture:** Simulator-free foundations first (fixture geometry, prism asset generators, one relation library shared by success, step goals and labels, randomization fields, a clearance model that knows fixtures and the Panda palm, dependency audits). Then simulator integration in the task builder, scripted expert and episode worker. Then new layouts for the 36 programs. Then the simulator gates on `vps-a`.

**Tech Stack:** Python ≥3.10,<3.13, NumPy, pytest; PyRep 4.1 / CoppeliaSim 4.1 / RLBench `02720bba` only on `vps-a` (`~/ICGS/.venv`).

**Spec:** `docs/superpowers/specs/2026-09-29-physical-dependency-generation-design.md` (approved 2026-09-29). Executors read both documents.

Status: active. Owner: repository owner (Nguyễn Quang Huy); implementer: Claude.
Motivation and observable acceptance: spec section 13 gates G1–G4 pass, then the owner decides on full generation.
Workflow class: C. Decision record: `docs/decisions/0016-physical-dependency-generation.md` (Task 1).

## Global Constraints

- Controller protocol `rlbench-timed-ik-v2` and action space `ee_pose_xyzw_grip_v1` are unchanged; timing stays measured per transition.
- Success needs every final and mandatory-history relation for 5 boundaries; at most 2 executed retries per step; valid failures are retained.
- Relation tolerances: position 0.01 m; yaw 10° modulo the footprint symmetry; inside margin 0.002 m; joint closed within 0.005 m; resting = speed < 0.01 m/s on each of the last 5 boundaries.
- Push re-push trigger 0.005 m; push success tolerance stays 0.01 m.
- Object scale 0.8–1.2; aperture gaps and pocket interiors resize with the attempt scale; drawers and gates keep fixed size.
- Measured Panda geometry about the tool tip at the default tool yaw (finger closing axis = world y): palm x [−0.064, 0.020], y [−0.100, 0.104], z [+0.037, +0.142]; open fingers y ±[0.040, 0.061], x [−0.0196, 0.0095]; closed fingers y ±0.021; fingertips 0.0016 m below the tip; finger top +0.042. Grasp tip = object centre; release = object 0.004 m above rest.
- Release rule: walls within 0.064 m of a release point along the finger axis stay ≥ 3 mm below the release fingertip height; tight walls run parallel to the finger axis.
- Asset split by generator family: train {box_cube, box_long, box_flat, box_tall, stadium}; development {hexagonal}; test {octagonal}.
- Provisional interval cap 2048 for gate G2; final cap and horizon per spec 10.1.
- Work on `main`; stage only this plan's files (another agent's uncommitted training-export files stay untouched); push to GitHub; validate on `vps-a` in `~/ICGS` after `git pull --ff-only` only.
- No training, downloads, HF publication or full generation without an explicit owner go-ahead. SKIPPED never counts as PASS.

## Review Focus

- Recorded states missing velocity, orientation or articulation (older archives, crash prefixes): relations return `None` (masked), never raise or guess — pinned in Task 4.
- Mirrored scenes: joint axis, fixture boxes and relations behave identically with y negated (drawer opening toward +y) — pinned in Tasks 2 and 6.
- Scale extremes 0.8 and 1.2 with resized fixtures: the right orientation passes an aperture or pocket and the wrong one collides at both extremes — pinned in Task 7.
- Short histories (< 5 boundaries) with data present: `resting`/`held` are `False` (not yet established), not `None` — pinned in Task 4.
- Archived scene geometry must reload identically for offline label recomputation — pinned in Task 4 (round trip).

## Current and target states

Current (`bc0b37e` collector, audit `docs/experiments/generation-validation/collector-semantics-20260929.md`): slab containers, handle-block drawers and gates, marker apertures, 4 cm label tolerance with historical eligibility, perturbations applied before the episode, one execution mode, identical box assets in every split, fixed-side layouts, no interval cap.

Target: spec sections 3–12. Unchanged: controller, action space, archive format (new data are additive fields), lineage/split rules, quota rules, launcher.

## Invariants and scope

- Baseline Instant Policy code, checkpoints, preprocessing and native formats are untouched.
- The archive schema only gains fields (`articulations` in scene states, wrist masks, scene geometry, interventions); readers of existing fields keep working.
- Out of scope: phase-2 anchors/branches/snapshots, force/penetration thresholds, training, the training-export reader (another agent's work).

## File structure

| File | Responsibility |
| --- | --- |
| `docs/decisions/0016-physical-dependency-generation.md` (create) | Accepted decision: fixtures, relation semantics, perturbation timing, asset split, limits, identifiers |
| `src/icgs/data/collection/generation/protocol.py` (modify) | New protocol constants |
| `src/icgs/data/collection/generation/fixtures.py` (create) | Fixture specs, boxes, joints, inner regions, passages, mirror/scale, serialization |
| `src/icgs/data/collection/generation/asset_families.py` (create) | Prism generators, footprints, meshes, split |
| `src/icgs/data/collection/generation/relations.py` (create) | Scene geometry and physical relations; condition evaluation |
| `src/icgs/data/collection/generation/task_labels.py` (rewrite) | ν/ρ/ε from relations |
| `src/icgs/data/collection/generation/diversity.py`, `batch.py` (modify) | Mirror flag, execution mode, offset side; mirroring in layout randomization |
| `src/icgs/data/collection/generation/layout_clearance.py` (modify) | Fixtures, palm, release rule, articulation and passage sweeps |
| `src/icgs/data/collection/generation/dependency_audit.py` (create) | Skip and consequence audits |
| Phase 2+ files | Listed per phase below |

## Phases and progress

- [ ] Phase 1 — simulator-free foundations (Tasks 1–8, detailed below). Validation: `pytest` on the new and touched test files, `python3 -B scripts/validate_fast.py`.
- [ ] Phase 2 — simulator integration (Tasks 9–15). Detailed steps are written into this plan in a commit at phase start, from the Phase 1 interfaces.
- [ ] Phase 3 — layouts and manifest (Tasks 16–20). Same rule.
- [ ] Phase 4 — gates G2–G4 and documentation (Tasks 21–23). Same rule.

Decision log (task-local):
- 2026-09-29: transit height is a per-program layout value (default 0.90 m, validated by the clearance audit) instead of a per-connector computation — simpler and deterministic; satisfies spec 6.1's clearance requirement.
- 2026-09-29: spec corrected before planning — grasp tip is the object centre, so walls along the finger axis must stay low (release rule).
- 2026-09-29: drawer stroke is set by the palm (≥ housing front + 0.104 + 0.008 m, about 0.18 m for a 0.11 m interior) so releases into the open tray keep the palm clear of the housing roof.
- 2026-09-29: adjacency blocking needs an elongated target (or blockers on both axes): a square body is simply grasped along the clear axis; audit and expert both choose the clear face pair.
- 2026-09-29: the drawer fin sits beside the tray on world +x (0.10 m) in every mirror state; a front-centre fin is hit by the 20 cm palm during releases (found by the plan dry run).
- 2026-09-29: plan dry run (every Phase 1 code block applied to a scratch copy of `6086a38`+spec, local Python 3.14): 58 new tests pass; `tests/test_generation*.py tests/test_capacity_probe.py tests/test_program_bindings.py` → 751 passed, 2 strict xfail (Task 7 pool tests). Simulator: not run.
- 2026-09-29: targets inside drawers are tray-relative (markers parented to the tray in the builder; shifted with the joint in the audit).

---

## Phase 1 — simulator-free foundations

### Task 1: Decision record and protocol constants

**Files:**
- Create: `docs/decisions/0016-physical-dependency-generation.md`
- Modify: `src/icgs/data/collection/generation/protocol.py` (fields after `predicate_success_m`)
- Modify: `docs/README.md` (decision table)
- Test: `tests/test_generation_physical_protocol.py`

**Interfaces:**
- Produces: `GENERATION_PROTOCOL.provisional_interval_cap: int`, `.push_retry_tolerance_m: float`, `.rest_speed_mps: float`, `.settle_boundaries: int`, `.settle_max_boundaries: int`, `.joint_closed_tolerance_m: float`, `.inside_margin_m: float`, `.fit_yaw_tolerance_deg: float`, `.near_threshold_fraction: float`, `.execution_mode_ids: tuple[str, ...]`, `.default_transit_z_m: float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generation_physical_protocol.py
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


def test_physical_dependency_protocol_values():
    p = GENERATION_PROTOCOL
    assert p.provisional_interval_cap == 2048
    assert p.push_retry_tolerance_m == 0.005
    assert p.push_retry_tolerance_m < p.predicate_success_m == 0.01
    assert p.rest_speed_mps == 0.01
    assert (p.settle_boundaries, p.settle_max_boundaries) == (5, 10)
    assert p.joint_closed_tolerance_m == 0.005
    assert p.inside_margin_m == 0.002
    assert p.fit_yaw_tolerance_deg == 10.0
    assert p.near_threshold_fraction == 0.8
    assert p.execution_mode_ids == ("scripted_direct", "scripted_offset")
    assert p.default_transit_z_m == 0.90
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_generation_physical_protocol.py -q`
Expected: FAIL with `AttributeError: 'GenerationProtocol' object has no attribute 'provisional_interval_cap'`

- [ ] **Step 3: Add the constants**

In `protocol.py`, directly after `predicate_success_m: float = 0.01`:

```python
    # Physical dependency generation (decision 0016).
    provisional_interval_cap: int = 2048
    push_retry_tolerance_m: float = 0.005
    rest_speed_mps: float = 0.01
    settle_boundaries: int = 5
    settle_max_boundaries: int = 10
    joint_closed_tolerance_m: float = 0.005
    inside_margin_m: float = 0.002
    fit_yaw_tolerance_deg: float = 10.0
    near_threshold_fraction: float = 0.8
    execution_mode_ids: tuple[str, ...] = ("scripted_direct", "scripted_offset")
    default_transit_z_m: float = 0.90
```

- [ ] **Step 4: Write the decision record**

Create `docs/decisions/0016-physical-dependency-generation.md`:

```markdown
# 0016: Physical dependency scenes and method-aligned generation

Date: 2026-09-29. Status: accepted.
Decision owner and approval evidence: repository owner approved the design
direction, the limit changes and prism asset families in conversation on
2026-09-29 and approved the written spec
`docs/superpowers/specs/2026-09-29-physical-dependency-generation-design.md`.

## Context

The 2026-09-29 collector audit found slab containers, handle-block drawers and
gates, marker apertures, and only 2 of 57 skippable steps physically required.
Labels used a 4 cm tolerance and historical eligibility, perturbations were
applied before the episode, every split used the same box assets, and episode
length exceeded the proposal's 512-interval figures. The proposal's primary
claims (H1–H3) need executed delayed consequences.

## Decision

- Scenes use physical fixtures: jointed drawers and sliding gates (contact
  articulation, no attachment), aperture walls, slot holders and trays, and
  adjacency blocking. Every declared enabling step must be physically required
  (static skip audit, simulator spot checks), except a recorded ordering-only list.
- One relation library defines success, step goals and task labels: position
  0.01 m, yaw 10° modulo symmetry, inside margin 0.002 m, joint closed 0.005 m,
  resting < 0.01 m/s for 5 boundaries. Eligibility uses current state.
- Perturbations follow the proposal: commanded target offset ≤ 2 cm/10° with the
  scene target unchanged; gripper close one interval early/late; object shift
  ≤ 3 cm and blocker insertion applied mid-episode at a stationary boundary with
  pre/post state recorded as an external intervention.
- Two scripted execution modes (`scripted_direct`, `scripted_offset`) and a
  balanced scene mirror.
- Asset families are prism generators split by family: train box_cube, box_long,
  box_flat, box_tall, stadium; development hexagonal; test octagonal.
- One interval is one recorded controller transition. Generation cap and
  benchmark horizon are set from measured expert lengths (spec 10.1) and
  recorded here before full generation. Drawer interior about 15 × 11 cm and
  aperture clearance 5–10 mm per side replace the proposal's example ranges.
- Identifiers (dataset, manifest, layout, semantics, predicate protocol,
  execution modes) are bumped when the manifest is regenerated.

## Alternatives considered

- Keep symbolic dependencies and relabel the claim: rejected; H1–H3 need physical
  consequences and the test contexts come from this run.
- Jointless sliding bodies in rails: rejected; drift and no clean closed state.
- Import RLBench drawer and door models: rejected; side grasps, fixed size, space.
- Count-threshold simulator gate: rejected; statistically meaningless at 3 seeds.

## Consequences

Collector, builder, layouts, labels and manifest change together; the previous
smoke receipts and unused HF prefix are retired. Validation: spec section 13.
Caps are appended here after gate G2.

## Compatibility implications

New dataset identity; archive format gains fields only; baseline Instant Policy
code, checkpoints and native formats are unaffected; the uncommitted
training-export reader must adopt the new fields separately.
```

In `docs/README.md`, add after the row for decision 0006:

```markdown
| [Physical dependency generation decision](decisions/0016-physical-dependency-generation.md) | Physical fixtures, relation semantics, perturbation timing, asset split and interval caps for generation |
```

- [ ] **Step 5: Run tests and fast validation**

Run: `python3 -m pytest tests/test_generation_physical_protocol.py tests/test_generation.py -q && python3 -B scripts/validate_fast.py`
Expected: PASS (existing protocol freeze tests unchanged because no identifier changed).

- [ ] **Step 6: Commit**

```bash
git add docs/decisions/0016-physical-dependency-generation.md docs/README.md \
  src/icgs/data/collection/generation/protocol.py tests/test_generation_physical_protocol.py
git commit -m "Record physical dependency generation decision and protocol values"
```

### Task 2: Fixture geometry

**Files:**
- Create: `src/icgs/data/collection/generation/fixtures.py`
- Test: `tests/test_generation_fixtures.py`

**Interfaces:**
- Produces: dataclasses `Box(name, center, size, yaw_deg=0.0, moving=False)` with `.z_range`, `.footprint()`; `JointSpec(name, origin, axis, stroke_m, open_threshold_m, brake_force_n)` (drawer stroke = max(interior-exposure threshold + 0.01, housing front + palm half span 0.104 + 0.008)); `Region(polygon, z_low, z_high)`; `Passage(point, normal, along, half_gap_m, top_z)`; fixtures `DrawerSpec`, `GateSpec`, `ApertureSpec`, `PocketSpec`; `Fixture = DrawerSpec | GateSpec | ApertureSpec | PocketSpec`; functions `fixture_boxes(fx, opening=0.0) -> list[Box]`, `fixture_joint(fx) -> JointSpec | None`, `initial_opening(fx) -> float`, `grip_point(fx, opening) -> tuple[float, float, float] | None`, `inner_region(fx, opening=0.0) -> Region | None`, `passage_of(fx) -> Passage | None`, `mirror_fixture(fx) -> Fixture`, `scale_fixture(fx, scale) -> Fixture`, `fixture_to_dict(fx) -> dict`, `fixture_from_dict(d) -> Fixture`, `fixtures_from_layout(layout) -> tuple[Fixture, ...]`. Constants `TABLE_TOP_Z_M=0.752`, `WALL_T_M=0.006`, `FINGERTIP_BELOW_TIP_M=0.0016`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_fixtures.py
import math

import pytest

from icgs.data.collection.generation.fixtures import (
    FINGERTIP_BELOW_TIP_M, PALM_HALF_SPAN_M, ApertureSpec, DrawerSpec, GateSpec, PocketSpec,
    fixture_boxes, fixture_from_dict, fixture_joint, fixture_to_dict, fixtures_from_layout,
    grip_point, initial_opening, inner_region, mirror_fixture, passage_of, scale_fixture,
)

PALM_BOTTOM_ABOVE_TIP_M = 0.037


def _drawer(**overrides):
    values = dict(name="drawer", center=(0.30, 0.20), axis=(0.0, -1.0), inner=(0.15, 0.11),
                  object_clearance_m=0.062)
    values.update(overrides)
    return DrawerSpec(**values)


def _box_named(boxes, suffix):
    return next(box for box in boxes if box.name.endswith(suffix))


def test_drawer_grip_clears_roof_and_fin_leaves_palm_clear():
    drawer = _drawer()
    tip_z = grip_point(drawer, 0.0)[2]
    assert tip_z - FINGERTIP_BELOW_TIP_M >= drawer.roof_top_z + 0.010 - 1e-9
    assert tip_z + 0.018 <= drawer.fin_top_z <= tip_z + 0.032
    assert tip_z + PALM_BOTTOM_ABOVE_TIP_M > drawer.fin_top_z


def test_drawer_open_threshold_exposes_whole_interior():
    drawer = _drawer()
    joint = fixture_joint(drawer)
    assert joint.stroke_m > joint.open_threshold_m
    region = inner_region(drawer, joint.open_threshold_m)
    # Every interior corner lies at or beyond the housing front edge along the axis.
    frame_origin_y = drawer.center[1]
    along = [(frame_origin_y - y) for _x, y in region.polygon]  # axis is -y
    assert min(along) >= drawer.front_v - 1e-9
    closed = inner_region(drawer, 0.0)
    assert max(frame_origin_y - y for _x, y in closed.polygon) < drawer.front_v
    assert closed.z_high == pytest.approx(drawer.roof_bottom_z)


def test_open_interior_centre_is_beyond_palm_reach_from_housing():
    drawer = _drawer()
    opening = fixture_joint(drawer).stroke_m - 0.005
    assert opening - PALM_HALF_SPAN_M >= drawer.front_v


def test_drawer_end_walls_obey_release_rule_and_moving_parts_follow_opening():
    drawer = _drawer()
    closed, opened = fixture_boxes(drawer, 0.0), fixture_boxes(drawer, 0.05)
    end = _box_named(closed, "_tray_end_front")
    assert end.z_range[1] - drawer.floor_top_z <= 0.012 + 1e-9
    for before, after in zip(closed, opened):
        shift = math.hypot(after.center[0] - before.center[0], after.center[1] - before.center[1])
        assert shift == pytest.approx(0.05 if before.moving else 0.0, abs=1e-12)


def test_gate_panel_blocks_when_closed_and_clears_gap_at_threshold():
    gate = GateSpec(name="gate", center=(0.30, 0.0), axis=(0.0, 1.0), gap_m=0.08)
    joint = fixture_joint(gate)
    passage = passage_of(gate)
    assert passage.half_gap_m == pytest.approx(0.04)
    panel_closed = _box_named(fixture_boxes(gate, 0.0), "_panel")
    panel_open = _box_named(fixture_boxes(gate, joint.open_threshold_m), "_panel")
    assert panel_closed.center[1] - panel_closed.size[1] / 2 <= -0.04
    assert panel_closed.center[1] + panel_closed.size[1] / 2 >= 0.04
    assert panel_open.center[1] - panel_open.size[1] / 2 >= 0.04 - 1e-9
    assert passage.top_z == pytest.approx(0.752 + gate.wall_h_m)


def test_aperture_and_pocket_scale_with_objects():
    aperture = ApertureSpec(name="aperture", center=(0.25, 0.02), axis=(1.0, 0.0), gap_m=0.05)
    assert passage_of(scale_fixture(aperture, 1.2)).half_gap_m == pytest.approx(0.03)
    pocket = PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.053, 0.08))
    scaled = inner_region(scale_fixture(pocket, 0.8))
    xs = [x for x, _y in scaled.polygon]
    assert max(xs) - min(xs) == pytest.approx(0.053 * 0.8)
    walls = fixture_boxes(pocket)
    side, end = _box_named(walls, "_side_left"), _box_named(walls, "_end_front")
    assert side.size[2] > end.size[2]


def test_regions_are_counterclockwise():
    for fixture in (_drawer(), PocketSpec(name="holder", center=(0.2, 0.1), inner=(0.05, 0.08), yaw_deg=30.0)):
        polygon = inner_region(fixture, 0.0).polygon
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))
        assert area2 > 0


def test_mirror_negates_y_and_keeps_joint_semantics():
    drawer = _drawer()
    mirrored = mirror_fixture(drawer)
    assert mirrored.axis == (0.0, 1.0)
    assert fixture_joint(mirrored).open_threshold_m == pytest.approx(fixture_joint(drawer).open_threshold_m)
    symmetric = lambda boxes: [b for b in boxes if not b.name.endswith(("_fin", "_tray_bar"))]
    original = sorted((round(b.center[0], 9), round(-b.center[1], 9), round(b.center[2], 9)) for b in symmetric(fixture_boxes(drawer)))
    reflected = sorted((round(b.center[0], 9), round(b.center[1], 9), round(b.center[2], 9)) for b in symmetric(fixture_boxes(mirrored)))
    assert original == reflected
    # The fin stays on world +x in both mirror states (the palm is not mirror-symmetric).
    assert grip_point(drawer, 0.0)[0] == pytest.approx(drawer.center[0] + drawer.fin_offset_x_m)
    assert grip_point(mirrored, 0.0)[0] == pytest.approx(drawer.center[0] + drawer.fin_offset_x_m)


def test_serialization_round_trip_and_layout_parsing():
    fixtures = (
        _drawer(initial_open=True),
        GateSpec(name="gate", center=(0.3, 0.0), axis=(0.0, 1.0), gap_m=0.08),
        ApertureSpec(name="aperture", center=(0.25, 0.02), axis=(1.0, 0.0), gap_m=0.05),
        PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.053, 0.08), yaw_deg=15.0),
    )
    for fixture in fixtures:
        assert fixture_from_dict(fixture_to_dict(fixture)) == fixture
    layout = {"fixtures": [fixture_to_dict(fixture) for fixture in fixtures]}
    assert fixtures_from_layout(layout) == fixtures
    assert initial_opening(fixtures[0]) == pytest.approx(fixture_joint(fixtures[0]).stroke_m)
    assert initial_opening(fixtures[1]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_fixtures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'icgs.data.collection.generation.fixtures'`

- [ ] **Step 3: Implement `fixtures.py`**

```python
"""Declarative fixture geometry for generation scenes. Simulator-free.

A fixture is a static structure, optionally carrying one body on a prismatic
joint. The task builder instantiates its boxes, the clearance audit treats
them as solids, and the relation library reads joint openings, inner regions
and passages from it (decision 0016). Lengths are metres in world axes; the
table top is at z = 0.752 m. The measured Panda hand sets the rules: palm
bottom 0.037 m above the tool tip, fingertips 0.0016 m below it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Any, Mapping, Sequence, Union

TABLE_TOP_Z_M = 0.752
WALL_T_M = 0.006
TRAY_FLOAT_M = 0.002
FINGERTIP_BELOW_TIP_M = 0.0016
GRIP_FINGERTIP_CLEARANCE_M = 0.010
FIN_TOP_ABOVE_TIP_M = 0.022
BRAKE_FORCE_N = 3.0
# Measured palm reach along the finger axis from the tool tip; an open drawer
# must carry its interior centre this far (plus margin) beyond the housing.
PALM_HALF_SPAN_M = 0.104


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float = 0.0
    moving: bool = False

    @property
    def z_range(self) -> tuple[float, float]:
        return (self.center[2] - self.size[2] / 2.0, self.center[2] + self.size[2] / 2.0)

    def footprint(self) -> list[tuple[float, float]]:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0
        return [
            (self.center[0] + c * x - s * y, self.center[1] + s * x + c * y)
            for x, y in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
        ]


@dataclass(frozen=True)
class JointSpec:
    name: str
    origin: tuple[float, float, float]
    axis: tuple[float, float]
    stroke_m: float
    open_threshold_m: float
    brake_force_n: float = BRAKE_FORCE_N


@dataclass(frozen=True)
class Region:
    """Inner volume of a container: convex counter-clockwise polygon and z interval."""

    polygon: tuple[tuple[float, float], ...]
    z_low: float
    z_high: float


@dataclass(frozen=True)
class Passage:
    """Opening in a wall line: signed plane, gap interval and wall top."""

    point: tuple[float, float]
    normal: tuple[float, float]
    along: tuple[float, float]
    half_gap_m: float
    top_z: float


def _unit(axis: Sequence[float]) -> tuple[float, float]:
    x, y = float(axis[0]), float(axis[1])
    norm = math.hypot(x, y)
    if norm <= 1e-12:
        raise ValueError("fixture axis must be nonzero")
    return (x / norm, y / norm)


class _Frame:
    """Fixture frame: ``v`` along ``axis`` and ``u`` across it (axis turned -90 deg)."""

    def __init__(self, center: Sequence[float], axis: Sequence[float]):
        self.cx, self.cy = float(center[0]), float(center[1])
        self.ax, self.ay = _unit(axis)
        self.ux, self.uy = self.ay, -self.ax

    def point(self, u: float, v: float) -> tuple[float, float]:
        return (self.cx + u * self.ux + v * self.ax, self.cy + u * self.uy + v * self.ay)

    @property
    def yaw_deg(self) -> float:
        return math.degrees(math.atan2(self.uy, self.ux))

    def box(self, name: str, u: float, v: float, z_low: float, z_high: float,
            size_u: float, size_v: float, *, moving: bool = False) -> Box:
        x, y = self.point(u, v)
        return Box(name, (x, y, (z_low + z_high) / 2.0), (size_u, size_v, z_high - z_low), self.yaw_deg, moving)

    def rectangle(self, u0: float, u1: float, v0: float, v1: float) -> tuple[tuple[float, float], ...]:
        return tuple(self.point(u, v) for u, v in ((u1, v0), (u1, v1), (u0, v1), (u0, v0)))


@dataclass(frozen=True)
class DrawerSpec:
    name: str
    center: tuple[float, float]
    axis: tuple[float, float]
    inner: tuple[float, float]
    object_clearance_m: float
    side_wall_h_m: float = 0.025
    end_wall_h_m: float = 0.012
    fin_width_m: float = 0.03
    fin_t_m: float = 0.012
    # The fin sits beside the tray on world +x in every mirror state, clear of
    # the palm (x -0.064..+0.020 about the tip) during releases into the tray.
    fin_offset_x_m: float = 0.10
    initial_open: bool = False
    kind: str = "drawer"

    @property
    def floor_top_z(self) -> float:
        return TABLE_TOP_Z_M + TRAY_FLOAT_M + WALL_T_M

    @property
    def outer(self) -> tuple[float, float]:
        return (self.inner[0] + 2.0 * WALL_T_M, self.inner[1] + 2.0 * WALL_T_M)

    @property
    def roof_bottom_z(self) -> float:
        return self.floor_top_z + self.object_clearance_m

    @property
    def roof_top_z(self) -> float:
        return self.roof_bottom_z + WALL_T_M

    @property
    def front_v(self) -> float:
        return self.outer[1] / 2.0 + 0.005

    @property
    def fin_v(self) -> float:
        return self.front_v + 0.008 + self.fin_t_m / 2.0

    @property
    def grip_tip_z(self) -> float:
        return self.roof_top_z + GRIP_FINGERTIP_CLEARANCE_M + FINGERTIP_BELOW_TIP_M

    @property
    def fin_top_z(self) -> float:
        return self.grip_tip_z + FIN_TOP_ABOVE_TIP_M

    @property
    def open_threshold_m(self) -> float:
        return self.front_v + self.inner[1] / 2.0

    @property
    def stroke_m(self) -> float:
        return max(self.open_threshold_m + 0.01, self.front_v + PALM_HALF_SPAN_M + 0.008)


@dataclass(frozen=True)
class GateSpec:
    name: str
    center: tuple[float, float]
    axis: tuple[float, float]
    gap_m: float
    wall_len_m: float = 0.12
    wall_h_m: float = 0.05
    fin_width_m: float = 0.06
    fin_t_m: float = 0.012
    initial_open: bool = False
    kind: str = "gate"

    @property
    def wall_top_z(self) -> float:
        return TABLE_TOP_Z_M + self.wall_h_m

    @property
    def grip_tip_z(self) -> float:
        return self.wall_top_z + GRIP_FINGERTIP_CLEARANCE_M + FINGERTIP_BELOW_TIP_M

    @property
    def fin_top_z(self) -> float:
        return self.grip_tip_z + FIN_TOP_ABOVE_TIP_M

    @property
    def panel_u(self) -> float:
        return -(WALL_T_M + 0.004)


@dataclass(frozen=True)
class ApertureSpec:
    name: str
    center: tuple[float, float]
    axis: tuple[float, float]
    gap_m: float
    wall_len_m: float = 0.12
    wall_h_m: float = 0.04
    kind: str = "aperture"


@dataclass(frozen=True)
class PocketSpec:
    """Slot container; local x across the finger axis (tight), local y along it."""

    name: str
    center: tuple[float, float]
    inner: tuple[float, float]
    yaw_deg: float = 0.0
    side_wall_h_m: float = 0.025
    end_wall_h_m: float = 0.012
    kind: str = "pocket"

    @property
    def floor_top_z(self) -> float:
        return TABLE_TOP_Z_M + WALL_T_M

    def frame(self) -> _Frame:
        yaw = math.radians(self.yaw_deg)
        return _Frame(self.center, (-math.sin(yaw), math.cos(yaw)))


Fixture = Union[DrawerSpec, GateSpec, ApertureSpec, PocketSpec]
_KINDS = {"drawer": DrawerSpec, "gate": GateSpec, "aperture": ApertureSpec, "pocket": PocketSpec}


def fixture_joint(fixture: Fixture) -> JointSpec | None:
    if isinstance(fixture, DrawerSpec):
        frame = _Frame(fixture.center, fixture.axis)
        x, y = frame.point(0.0, 0.0)
        return JointSpec(f"{fixture.name}_joint", (x, y, TABLE_TOP_Z_M + TRAY_FLOAT_M + WALL_T_M / 2.0),
                         (frame.ax, frame.ay), fixture.stroke_m, fixture.open_threshold_m)
    if isinstance(fixture, GateSpec):
        frame = _Frame(fixture.center, fixture.axis)
        x, y = frame.point(fixture.panel_u, 0.0)
        return JointSpec(f"{fixture.name}_joint", (x, y, TABLE_TOP_Z_M + fixture.wall_h_m / 2.0),
                         (frame.ax, frame.ay), fixture.gap_m + 0.02, fixture.gap_m + 0.01)
    return None


def initial_opening(fixture: Fixture) -> float:
    joint = fixture_joint(fixture)
    if joint is None or not getattr(fixture, "initial_open", False):
        return 0.0
    return joint.stroke_m


def fixture_boxes(fixture: Fixture, opening: float = 0.0) -> list[Box]:
    if isinstance(fixture, DrawerSpec):
        return _drawer_boxes(fixture, float(opening))
    if isinstance(fixture, GateSpec):
        return _gate_boxes(fixture, float(opening))
    if isinstance(fixture, ApertureSpec):
        return _wall_segments(fixture, _Frame(fixture.center, fixture.axis))
    if isinstance(fixture, PocketSpec):
        return _pocket_boxes(fixture)
    raise TypeError(f"unsupported fixture: {fixture!r}")


def _drawer_boxes(d: DrawerSpec, opening: float) -> list[Box]:
    f = _Frame(d.center, d.axis)
    t, table = WALL_T_M, TABLE_TOP_Z_M
    out_u, out_v = d.outer
    across_in = out_u + 0.010
    back_v = -out_v / 2.0 - 0.005
    v0, v1 = back_v - t, d.front_v
    mid, length = (v0 + v1) / 2.0, v1 - v0
    floor_low, floor_top = table + TRAY_FLOAT_M, d.floor_top_z
    o = opening
    fin_u = d.fin_offset_x_m * f.ux
    return [
        f.box(f"{d.name}_roof", 0.0, mid, d.roof_bottom_z, d.roof_top_z, across_in + 2 * t, length),
        f.box(f"{d.name}_wall_left", -(across_in + t) / 2.0, mid, table, d.roof_bottom_z, t, length),
        f.box(f"{d.name}_wall_right", (across_in + t) / 2.0, mid, table, d.roof_bottom_z, t, length),
        f.box(f"{d.name}_wall_back", 0.0, back_v - t / 2.0, table, d.roof_bottom_z, across_in, t),
        f.box(f"{d.name}_tray_floor", 0.0, o, floor_low, floor_top, out_u, out_v, moving=True),
        f.box(f"{d.name}_tray_side_left", -(out_u - t) / 2.0, o, floor_top, floor_top + d.side_wall_h_m, t, out_v, moving=True),
        f.box(f"{d.name}_tray_side_right", (out_u - t) / 2.0, o, floor_top, floor_top + d.side_wall_h_m, t, out_v, moving=True),
        f.box(f"{d.name}_tray_end_back", 0.0, o - (out_v - t) / 2.0, floor_top, floor_top + d.end_wall_h_m, out_u - 2 * t, t, moving=True),
        f.box(f"{d.name}_tray_end_front", 0.0, o + (out_v - t) / 2.0, floor_top, floor_top + d.end_wall_h_m, out_u - 2 * t, t, moving=True),
        f.box(f"{d.name}_tray_tongue", 0.0, o + (out_v / 2.0 + d.fin_v - d.fin_t_m / 2.0) / 2.0, floor_low,
              floor_top + d.end_wall_h_m, d.fin_width_m, d.fin_v - d.fin_t_m / 2.0 - out_v / 2.0, moving=True),
        f.box(f"{d.name}_tray_bar", fin_u / 2.0, o + d.fin_v, floor_low, floor_top + d.end_wall_h_m,
              abs(fin_u) + d.fin_width_m, d.fin_t_m, moving=True),
        f.box(f"{d.name}_fin", fin_u, o + d.fin_v, floor_low, d.fin_top_z, d.fin_width_m, d.fin_t_m, moving=True),
    ]


def _wall_segments(fixture: GateSpec | ApertureSpec, f: _Frame) -> list[Box]:
    half, length, top = fixture.gap_m / 2.0, fixture.wall_len_m, TABLE_TOP_Z_M + fixture.wall_h_m
    return [
        f.box(f"{fixture.name}_wall_minus", 0.0, -(half + length / 2.0), TABLE_TOP_Z_M, top, WALL_T_M, length),
        f.box(f"{fixture.name}_wall_plus", 0.0, half + length / 2.0, TABLE_TOP_Z_M, top, WALL_T_M, length),
    ]


def _gate_boxes(g: GateSpec, opening: float) -> list[Box]:
    f = _Frame(g.center, g.axis)
    top = g.wall_top_z
    return _wall_segments(g, f) + [
        f.box(f"{g.name}_panel", g.panel_u, opening, TABLE_TOP_Z_M, top, WALL_T_M, g.gap_m + 0.02, moving=True),
        f.box(f"{g.name}_fin", g.panel_u, opening, top + 0.002, g.fin_top_z, g.fin_width_m, g.fin_t_m, moving=True),
    ]


def _pocket_boxes(p: PocketSpec) -> list[Box]:
    f = p.frame()
    t, table, floor_top = WALL_T_M, TABLE_TOP_Z_M, p.floor_top_z
    ix, iy = p.inner
    return [
        f.box(f"{p.name}_floor", 0.0, 0.0, table, floor_top, ix + 2 * t, iy + 2 * t),
        f.box(f"{p.name}_side_left", -(ix + t) / 2.0, 0.0, floor_top, floor_top + p.side_wall_h_m, t, iy + 2 * t),
        f.box(f"{p.name}_side_right", (ix + t) / 2.0, 0.0, floor_top, floor_top + p.side_wall_h_m, t, iy + 2 * t),
        f.box(f"{p.name}_end_back", 0.0, -(iy + t) / 2.0, floor_top, floor_top + p.end_wall_h_m, ix, t),
        f.box(f"{p.name}_end_front", 0.0, (iy + t) / 2.0, floor_top, floor_top + p.end_wall_h_m, ix, t),
    ]


def grip_point(fixture: Fixture, opening: float) -> tuple[float, float, float] | None:
    if isinstance(fixture, DrawerSpec):
        frame = _Frame(fixture.center, fixture.axis)
        x, y = frame.point(fixture.fin_offset_x_m * frame.ux, float(opening) + fixture.fin_v)
        return (x, y, fixture.grip_tip_z)
    if isinstance(fixture, GateSpec):
        x, y = _Frame(fixture.center, fixture.axis).point(fixture.panel_u, float(opening))
        return (x, y, fixture.grip_tip_z)
    return None


def inner_region(fixture: Fixture, opening: float = 0.0) -> Region | None:
    if isinstance(fixture, DrawerSpec):
        f = _Frame(fixture.center, fixture.axis)
        iu, iv = fixture.inner
        o = float(opening)
        return Region(f.rectangle(-iu / 2.0, iu / 2.0, o - iv / 2.0, o + iv / 2.0),
                      fixture.floor_top_z, fixture.roof_bottom_z)
    if isinstance(fixture, PocketSpec):
        ix, iy = fixture.inner
        return Region(fixture.frame().rectangle(-ix / 2.0, ix / 2.0, -iy / 2.0, iy / 2.0),
                      fixture.floor_top_z, fixture.floor_top_z + 0.2)
    return None


def passage_of(fixture: Fixture) -> Passage | None:
    if isinstance(fixture, (GateSpec, ApertureSpec)):
        f = _Frame(fixture.center, fixture.axis)
        return Passage((f.cx, f.cy), (f.ux, f.uy), (f.ax, f.ay), fixture.gap_m / 2.0,
                       TABLE_TOP_Z_M + fixture.wall_h_m)
    return None


def mirror_fixture(fixture: Fixture) -> Fixture:
    center = (fixture.center[0], -fixture.center[1])
    if isinstance(fixture, PocketSpec):
        return replace(fixture, center=center, yaw_deg=-fixture.yaw_deg)
    return replace(fixture, center=center, axis=(fixture.axis[0], -fixture.axis[1]))


def scale_fixture(fixture: Fixture, scale: float) -> Fixture:
    s = float(scale)
    if isinstance(fixture, ApertureSpec):
        return replace(fixture, gap_m=fixture.gap_m * s)
    if isinstance(fixture, PocketSpec):
        return replace(fixture, inner=(fixture.inner[0] * s, fixture.inner[1] * s))
    return fixture


def fixture_to_dict(fixture: Fixture) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": fixture.kind}
    for item in fields(fixture):
        if item.name == "kind":
            continue
        value = getattr(fixture, item.name)
        payload[item.name] = list(value) if isinstance(value, tuple) else value
    return payload


def fixture_from_dict(payload: Mapping[str, Any]) -> Fixture:
    kind = str(payload["type"])
    if kind not in _KINDS:
        raise ValueError(f"unknown fixture type: {kind}")
    values = {
        key: tuple(float(v) for v in value) if isinstance(value, list) else value
        for key, value in payload.items() if key != "type"
    }
    return _KINDS[kind](**values)


def fixtures_from_layout(layout: Mapping[str, Any]) -> tuple[Fixture, ...]:
    return tuple(fixture_from_dict(item) for item in layout.get("fixtures", ()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_fixtures.py -q`
Expected: PASS (10 tests). If `test_mirror_negates_y_and_keeps_joint_semantics` fails on box ordering, keep the comparison sorted as written; do not special-case names.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/fixtures.py tests/test_generation_fixtures.py
git commit -m "Add declarative fixture geometry for physical generation scenes"
```

### Task 3: Prism asset families

**Files:**
- Create: `src/icgs/data/collection/generation/asset_families.py`
- Test: `tests/test_generation_asset_families.py`

**Interfaces:**
- Produces: `Generator(generator_id, family, split, sides, elongation, height_ratio, symmetry_deg)`; `GENERATORS: dict[str, Generator]`; `generator(generator_id) -> Generator`; `families_for_split(split) -> tuple[str, ...]`; `generators_for_split(split) -> tuple[str, ...]`; `footprint(generator_id, width_m) -> tuple[tuple[float, float], ...]` (CCW, flat pair across local y, width = flat-to-flat along y); `height_m(generator_id, width_m) -> float`; `extents(polygon) -> tuple[float, float]`; `prism_mesh(polygon, height) -> tuple[list[float], list[int]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_asset_families.py
from collections import Counter

import pytest

from icgs.data.collection.generation.asset_families import (
    GENERATORS, extents, families_for_split, footprint, generator, generators_for_split,
    height_m, prism_mesh,
)


def test_split_is_by_family_and_close_to_70_15_15():
    counts = {split: len(families_for_split(split)) for split in ("train", "development", "test")}
    assert counts == {"train": 5, "development": 1, "test": 1}
    families = [families_for_split(split) for split in ("train", "development", "test")]
    assert not set(families[0]) & set(families[1]) and not set(families[0]) & set(families[2])


def test_every_split_offers_compact_and_elongated_members():
    for split in ("train", "development", "test"):
        members = [generator(item) for item in generators_for_split(split)]
        assert any(item.symmetry_deg == 180.0 for item in members)
        assert any(item.symmetry_deg < 180.0 for item in members)


@pytest.mark.parametrize("generator_id", sorted(GENERATORS))
def test_footprint_is_convex_ccw_with_width_across_y(generator_id):
    polygon = footprint(generator_id, 0.045)
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))
    assert area2 > 0
    for (x0, y0), (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], polygon[2:] + polygon[:2]):
        assert (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) >= -1e-12
    assert extents(polygon)[1] == pytest.approx(0.045)
    item = generator(generator_id)
    assert height_m(generator_id, 0.045) == pytest.approx(0.045 * item.height_ratio)


@pytest.mark.parametrize("generator_id", sorted(GENERATORS))
def test_prism_mesh_is_closed_and_outward(generator_id):
    polygon = footprint(generator_id, 0.04)
    vertices, indices = prism_mesh(polygon, 0.05)
    assert len(vertices) == 3 * 2 * len(polygon)
    edges = Counter()
    for a, b, c in zip(indices[0::3], indices[1::3], indices[2::3]):
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(u, v)] += 1
    for (u, v), count in edges.items():
        assert count == 1 and edges[(v, u)] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_asset_families.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `asset_families.py`**

```python
"""Parametric prism asset generators and their split (decision 0016). Simulator-free.

The proposal splits asset families 70/15/15 by mesh family before
augmentation and tests on new mesh families, not scaled copies. Every
generator is a right prism over a convex footprint with a flat face pair
across its local y axis, so the top-down gripper closes on flats.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Generator:
    generator_id: str
    family: str
    split: str
    sides: int
    elongation: float
    height_ratio: float
    symmetry_deg: float


GENERATORS: dict[str, Generator] = {item.generator_id: item for item in (
    Generator("box_cube", "box_cube", "train", 4, 1.0, 1.0, 90.0),
    Generator("box_long", "box_long", "train", 4, 2.1, 1.0, 180.0),
    Generator("box_flat", "box_flat", "train", 4, 1.0, 0.6, 90.0),
    Generator("box_tall", "box_tall", "train", 4, 1.0, 1.4, 90.0),
    Generator("stadium", "stadium", "train", 0, 1.8, 1.0, 180.0),
    Generator("hex_prism", "hexagonal", "development", 6, 1.0, 1.0, 60.0),
    Generator("hex_long", "hexagonal", "development", 6, 1.8, 1.0, 180.0),
    Generator("oct_prism", "octagonal", "test", 8, 1.0, 1.0, 45.0),
    Generator("oct_long", "octagonal", "test", 8, 1.8, 1.0, 180.0),
)}
_ARC_SEGMENTS = 8


def generator(generator_id: str) -> Generator:
    try:
        return GENERATORS[str(generator_id)]
    except KeyError as exc:
        raise ValueError(f"unknown asset generator: {generator_id}") from exc


def generators_for_split(split: str) -> tuple[str, ...]:
    return tuple(item.generator_id for item in GENERATORS.values() if item.split == split)


def families_for_split(split: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.family for item in GENERATORS.values() if item.split == split))


def footprint(generator_id: str, width_m: float) -> tuple[tuple[float, float], ...]:
    item = generator(generator_id)
    half = float(width_m) / 2.0
    if item.sides == 0:
        straight = half * (item.elongation - 1.0)
        points = []
        for center_x, start in ((straight, -90.0), (-straight, 90.0)):
            for index in range(_ARC_SEGMENTS + 1):
                angle = math.radians(start + 180.0 * index / _ARC_SEGMENTS)
                points.append((center_x + half * math.cos(angle), half * math.sin(angle)))
        return tuple(points)
    n = item.sides
    radius = half / math.cos(math.pi / n)
    points = []
    for index in range(n):
        angle = math.radians(90.0 + 180.0 / n + 360.0 * index / n)
        points.append((radius * math.cos(angle) * item.elongation, radius * math.sin(angle)))
    return tuple(points)


def height_m(generator_id: str, width_m: float) -> float:
    return float(width_m) * generator(generator_id).height_ratio


def extents(polygon) -> tuple[float, float]:
    xs = [float(x) for x, _y in polygon]
    ys = [float(y) for _x, y in polygon]
    return (max(xs) - min(xs), max(ys) - min(ys))


def prism_mesh(polygon, height: float) -> tuple[list[float], list[int]]:
    n = len(polygon)
    h = float(height) / 2.0
    vertices: list[float] = []
    for z in (-h, h):
        for x, y in polygon:
            vertices.extend((float(x), float(y), z))
    indices: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        indices.extend((i, j, n + j, i, n + j, n + i))
    for k in range(1, n - 1):
        indices.extend((n, n + k, n + k + 1))
        indices.extend((0, k + 1, k))
    return vertices, indices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_asset_families.py -q`
Expected: PASS (4 test functions, 22 parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/asset_families.py tests/test_generation_asset_families.py
git commit -m "Add prism asset generators split by family"
```

### Task 4: Relation library

**Files:**
- Create: `src/icgs/data/collection/generation/relations.py`
- Test: `tests/test_generation_relations.py`

**Interfaces:**
- Consumes: Task 2 `fixture_joint`, `inner_region`, `passage_of`, `fixture_to_dict`, `fixture_from_dict`, `Fixture`; Task 1 protocol constants.
- Produces: `BodyGeometry(footprint, height_m, symmetry_deg=90.0)` with `to_dict`/`from_dict`; `SceneGeometry(bodies, fixtures, containers, oriented_targets, reach_offsets_m)` with `to_dict`/`from_dict`; `position(state, name)`, `yaw_deg(state, name)`, `speed(state, name)`, `joint_position(state, fixture_name)`, `yaw_error_deg(a, b, symmetry_deg)`, `at_pose(state, body, target, geometry=None, *, position_m=None, yaw_tolerance_deg=None, planar=False)`, `inside(state, body, fixture_name, geometry, *, margin_m=None)`, `joint_closed(state, fixture_name, *, tolerance_m=None)`, `joint_open(state, fixture_name, geometry)`, `access_clear(state, body, geometry, *, reach_m=0.061, exclude=())`, `slot_clear(state, target, geometry, *, exclude=())`, `resting(states, body, *, speed_mps=None, boundaries=None)`, `held(states, robots, body, *, boundaries=None, drift_m=0.01)`, `lifted(state, initial_state, body, *, lift_m=0.03)`, `passed(states, body, fixture_name, geometry)`, `reached(robots, point, *, tolerance_m=None)`, `evaluate_condition(name, step, states, robots, geometry=None) -> bool | None`. Scene state format: `{"objects": [{"name", "position", "orientation_xyzw", "linear_velocity", "valid"}], "articulations": [{"name", "position", "velocity"}]}`; robot state `{"T_w_e": 4x4, "grip": 0|1}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_relations.py
import math

import numpy as np
import pytest

from icgs.data.collection.generation.fixtures import (
    ApertureSpec, DrawerSpec, PocketSpec, fixture_joint, inner_region,
)
from icgs.data.collection.generation.relations import (
    BodyGeometry, SceneGeometry, access_clear, at_pose, evaluate_condition, held, inside,
    joint_closed, joint_open, passed, reached, resting, slot_clear, yaw_error_deg,
)

SQUARE = ((0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02))


def _quat(yaw_deg):
    half = math.radians(yaw_deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _obj(name, xyz, yaw=0.0, velocity=(0.0, 0.0, 0.0)):
    item = {"name": name, "position": list(xyz), "orientation_xyzw": _quat(yaw), "valid": True}
    if velocity is not None:
        item["linear_velocity"] = list(velocity)
    return item


def _state(*objects, articulations=()):
    return {"objects": list(objects), "articulations": list(articulations)}


def _robot(xyz, grip):
    transform = np.eye(4)
    transform[:3, 3] = xyz
    return {"T_w_e": transform, "grip": grip}


def _geometry(**extra):
    bodies = {"object_a": BodyGeometry(SQUARE, 0.04, 90.0), "blocker": BodyGeometry(SQUARE, 0.04, 90.0)}
    return SceneGeometry(bodies=bodies, **extra)


def test_yaw_error_respects_symmetry():
    assert yaw_error_deg(95.0, 5.0, 90.0) == pytest.approx(0.0)
    assert yaw_error_deg(95.0, 5.0, 180.0) == pytest.approx(90.0)


def test_at_pose_position_planar_and_oriented_targets():
    state = _state(_obj("object_a", (0.2, 0.0, 0.77), yaw=12.0), _obj("target_a", (0.205, 0.0, 0.8)))
    assert at_pose(state, "object_a", "target_a") is False
    assert at_pose(state, "object_a", "target_a", planar=True) is True
    geometry = _geometry(oriented_targets={"target_a": 90.0})
    assert at_pose(state, "object_a", "target_a", geometry, planar=True) is False
    assert at_pose(state, "missing", "target_a") is None


def test_inside_drawer_follows_joint_and_requires_floor_contact():
    drawer = DrawerSpec(name="drawer", center=(0.3, 0.2), axis=(0.0, -1.0), inner=(0.15, 0.11), object_clearance_m=0.062)
    geometry = _geometry(fixtures={"drawer": drawer})
    opening = fixture_joint(drawer).open_threshold_m
    cx, cy = np.mean(inner_region(drawer, opening).polygon, axis=0)
    z = drawer.floor_top_z + 0.02
    joint = {"name": "drawer_joint", "position": opening, "velocity": 0.0}
    assert inside(_state(_obj("object_a", (cx, cy, z)), articulations=[joint]), "object_a", "drawer", geometry) is True
    lifted_state = _state(_obj("object_a", (cx, cy, z + 0.03)), articulations=[joint])
    assert inside(lifted_state, "object_a", "drawer", geometry) is False
    assert inside(_state(_obj("object_a", (cx, cy, z))), "object_a", "drawer", geometry) is None


def test_inside_pocket_rejects_misaligned_yaw():
    pocket = PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.048, 0.08))
    geometry = _geometry(fixtures={"holder": pocket})
    z = pocket.floor_top_z + 0.02
    assert inside(_state(_obj("object_a", (0.25, 0.16, z), yaw=0.0)), "object_a", "holder", geometry) is True
    assert inside(_state(_obj("object_a", (0.25, 0.16, z), yaw=20.0)), "object_a", "holder", geometry) is False


def test_joint_relations_and_missing_articulation():
    drawer = DrawerSpec(name="drawer", center=(0.3, 0.2), axis=(0.0, -1.0), inner=(0.15, 0.11), object_clearance_m=0.062)
    geometry = _geometry(fixtures={"drawer": drawer})
    closed = _state(articulations=[{"name": "drawer_joint", "position": 0.004}])
    opened = _state(articulations=[{"name": "drawer_joint", "position": fixture_joint(drawer).open_threshold_m}])
    assert joint_closed(closed, "drawer") is True and joint_closed(opened, "drawer") is False
    assert joint_open(opened, "drawer", geometry) is True and joint_open(closed, "drawer", geometry) is False
    assert joint_closed(_state(), "drawer") is None


def test_resting_short_history_is_false_and_missing_velocity_is_none():
    still = [_state(_obj("object_a", (0.2, 0.0, 0.77))) for _ in range(3)]
    assert resting(still, "object_a") is False
    assert resting(still * 2, "object_a") is True
    moving = still * 2 + [_state(_obj("object_a", (0.2, 0.0, 0.77), velocity=(0.02, 0.0, 0.0)))]
    assert resting(moving, "object_a") is False
    assert resting([_state(_obj("object_a", (0.2, 0.0, 0.77), velocity=None))] * 5, "object_a") is None


def test_held_requires_closed_grip_and_stable_offset():
    states = [_state(_obj("object_a", (0.2, 0.0, 0.85 + 0.001 * k))) for k in range(5)]
    robots = [_robot((0.2, 0.0, 0.85 + 0.001 * k), 0) for k in range(5)]
    assert held(states, robots, "object_a") is True
    robots[-1] = _robot((0.2, 0.0, 0.86), 1)
    assert held(states, robots, "object_a") is False
    assert held(states[:3], robots[:3], "object_a") is False


def test_passed_counts_crossings_inside_gap_below_top():
    aperture = ApertureSpec(name="aperture", center=(0.25, 0.0), axis=(1.0, 0.0), gap_m=0.06)
    geometry = _geometry(fixtures={"aperture": aperture})
    through = [_state(_obj("object_a", (0.25, y, 0.775))) for y in (-0.05, -0.01, 0.02, 0.06)]
    beside = [_state(_obj("object_a", (0.40, y, 0.775))) for y in (-0.05, 0.06)]
    above = [_state(_obj("object_a", (0.25, y, 0.90))) for y in (-0.05, 0.06)]
    assert passed(through, "object_a", "aperture", geometry) is True
    assert passed(beside, "object_a", "aperture", geometry) is False
    assert passed(above, "object_a", "aperture", geometry) is False


def test_access_and_slot_clearance():
    geometry = _geometry()
    near = _state(_obj("object_a", (0.25, 0.0, 0.77)), _obj("blocker", (0.25, 0.07, 0.77)))
    far = _state(_obj("object_a", (0.25, 0.0, 0.77)), _obj("blocker", (0.25, 0.20, 0.77)))
    assert access_clear(near, "object_a", geometry) is False
    assert access_clear(far, "object_a", geometry) is True
    slot = _state(_obj("blocker", (0.25, 0.20, 0.77)), _obj("target_b", (0.25, 0.20, 0.77)))
    assert slot_clear(slot, "target_b", geometry) is False
    assert slot_clear(slot, "target_b", geometry, exclude=("blocker",)) is True


def test_reached_and_condition_mapping():
    robots = [_robot((0.25, -0.05, 0.80), 1)]
    assert reached(robots, (0.25, -0.05, 0.805)) is True
    geometry = _geometry()
    step = {"object_role": "object_a", "target_role": "target_a", "postcondition": "object_a_placed"}
    placed = [_state(_obj("object_a", (0.2, 0.0, 0.77)), _obj("target_a", (0.2, 0.0, 0.77)))] * 5
    assert evaluate_condition("object_a_placed", step, placed, [_robot((0.2, 0.0, 0.9), 1)] * 5, geometry) is True
    assert evaluate_condition("object_a_placed", step, placed[:2], [_robot((0.2, 0.0, 0.9), 1)] * 2, geometry) is False
    assert evaluate_condition("unknown_relation", step, placed, [_robot((0, 0, 1), 1)] * 5, geometry) is None
    assert evaluate_condition("drawer_open", {}, placed, [], geometry) is None


def test_scene_geometry_round_trip():
    geometry = _geometry(
        fixtures={"holder": PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.048, 0.08))},
        containers={"target_a": "holder"}, oriented_targets={"target_a": 90.0}, reach_offsets_m={"target_a": 0.01},
    )
    assert SceneGeometry.from_dict(geometry.to_dict()) == geometry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_relations.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `relations.py`**

```python
"""Physical relations over recorded scene states (decision 0016). Simulator-free.

One definition serves success checks, step goals and task-memory labels:
held (stable grasp), resting (stable placement), inside a container, joint
closed/open, pose within 1 cm and 10 degrees, passage through an opening,
access clearance and reach. A relation returns ``None`` when the recorded
state cannot decide it; callers mask it instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from icgs.data.collection.generation.fixtures import (
    Fixture,
    fixture_from_dict,
    fixture_joint,
    fixture_to_dict,
    inner_region,
    passage_of,
)
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL

OPEN_FINGER_REACH_M = 0.061
HELD_DISTANCE_M = 0.10
LIFT_MIN_M = 0.03
HELD_DRIFT_M = 0.01
FLOOR_CONTACT_M = 0.005
DEFAULT_REACH_OFFSET_M = 0.03


@dataclass(frozen=True)
class BodyGeometry:
    footprint: tuple[tuple[float, float], ...]
    height_m: float
    symmetry_deg: float = 90.0

    def to_dict(self) -> dict[str, Any]:
        return {"footprint": [list(point) for point in self.footprint], "height_m": float(self.height_m),
                "symmetry_deg": float(self.symmetry_deg)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BodyGeometry":
        return cls(tuple((float(x), float(y)) for x, y in payload["footprint"]),
                   float(payload["height_m"]), float(payload.get("symmetry_deg", 90.0)))


@dataclass(frozen=True)
class SceneGeometry:
    bodies: Mapping[str, BodyGeometry] = field(default_factory=dict)
    fixtures: Mapping[str, Fixture] = field(default_factory=dict)
    containers: Mapping[str, str] = field(default_factory=dict)
    oriented_targets: Mapping[str, float] = field(default_factory=dict)
    reach_offsets_m: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodies": {name: body.to_dict() for name, body in self.bodies.items()},
            "fixtures": {name: fixture_to_dict(fixture) for name, fixture in self.fixtures.items()},
            "containers": dict(self.containers),
            "oriented_targets": {name: float(value) for name, value in self.oriented_targets.items()},
            "reach_offsets_m": {name: float(value) for name, value in self.reach_offsets_m.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneGeometry":
        return cls(
            bodies={name: BodyGeometry.from_dict(item) for name, item in (payload.get("bodies") or {}).items()},
            fixtures={name: fixture_from_dict(item) for name, item in (payload.get("fixtures") or {}).items()},
            containers={str(k): str(v) for k, v in (payload.get("containers") or {}).items()},
            oriented_targets={str(k): float(v) for k, v in (payload.get("oriented_targets") or {}).items()},
            reach_offsets_m={str(k): float(v) for k, v in (payload.get("reach_offsets_m") or {}).items()},
        )


def _objects(state: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("name")): item
        for item in (state or {}).get("objects", ())
        if isinstance(item, Mapping) and item.get("name") and item.get("valid", True)
    }


def position(state: Mapping[str, Any] | None, name: str | None) -> np.ndarray | None:
    item = _objects(state).get(str(name)) if name else None
    if not item or item.get("position") is None:
        return None
    value = np.asarray(item["position"], dtype=np.float64)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def yaw_deg(state: Mapping[str, Any] | None, name: str | None) -> float | None:
    item = _objects(state).get(str(name)) if name else None
    quaternion = None if not item else item.get("orientation_xyzw")
    if quaternion is None:
        return None
    x, y, z, w = (float(value) for value in quaternion)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def speed(state: Mapping[str, Any] | None, name: str | None) -> float | None:
    item = _objects(state).get(str(name)) if name else None
    velocity = None if not item else item.get("linear_velocity")
    if velocity is None:
        return None
    return float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))


def joint_position(state: Mapping[str, Any] | None, fixture_name: str) -> float | None:
    names = {str(fixture_name), f"{fixture_name}_joint"}
    for item in (state or {}).get("articulations", ()):
        if isinstance(item, Mapping) and str(item.get("name")) in names and item.get("position") is not None:
            return float(item["position"])
    return None


def yaw_error_deg(a: float, b: float, symmetry_deg: float) -> float:
    period = float(symmetry_deg) if float(symmetry_deg) > 0.0 else 360.0
    delta = (float(a) - float(b)) % period
    return min(delta, period - delta)


def _tip(robot: Mapping[str, Any] | None) -> np.ndarray | None:
    transform = np.asarray((robot or {}).get("T_w_e"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    return transform[:3, 3]


def _body_polygon(state, name: str, geometry: SceneGeometry | None) -> np.ndarray | None:
    if geometry is None or name not in geometry.bodies:
        return None
    center, heading = position(state, name), yaw_deg(state, name)
    if center is None or heading is None:
        return None
    c, s = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    points = np.asarray(geometry.bodies[name].footprint, dtype=np.float64)
    return points @ np.asarray([[c, s], [-s, c]]) + center[:2]


def _signed_edge_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance of each point inside each edge of a CCW polygon (positive = inside)."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return cross / lengths[None, :]


def _point_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if (_signed_edge_distances(point[None, :], polygon)[0] >= 0.0).all():
        return 0.0
    best = math.inf
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        t = float(np.clip(np.dot(point - start, edge) / max(float(np.dot(edge, edge)), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (start + t * edge))))
    return best


def at_pose(state, body, target, geometry: SceneGeometry | None = None, *, position_m: float | None = None,
            yaw_tolerance_deg: float | None = None, planar: bool = False) -> bool | None:
    tolerance = GENERATION_PROTOCOL.predicate_success_m if position_m is None else float(position_m)
    here, there = position(state, body), position(state, target)
    if here is None or there is None:
        return None
    delta = (here - there)[:2] if planar else here - there
    if float(np.linalg.norm(delta)) > tolerance:
        return False
    if geometry is None or target not in geometry.oriented_targets:
        return True
    heading, wanted = yaw_deg(state, body), yaw_deg(state, target)
    if heading is None or wanted is None:
        return None
    limit = GENERATION_PROTOCOL.fit_yaw_tolerance_deg if yaw_tolerance_deg is None else float(yaw_tolerance_deg)
    return yaw_error_deg(heading, wanted, geometry.oriented_targets[target]) <= limit


def _region(state, fixture_name: str, geometry: SceneGeometry | None):
    fixture = None if geometry is None else geometry.fixtures.get(fixture_name)
    if fixture is None:
        return None
    opening = 0.0
    if fixture_joint(fixture) is not None:
        measured = joint_position(state, fixture_name)
        if measured is None:
            return None
        opening = measured
    return inner_region(fixture, opening)


def inside(state, body, fixture_name, geometry: SceneGeometry | None, *, margin_m: float | None = None) -> bool | None:
    region = _region(state, fixture_name, geometry)
    polygon = _body_polygon(state, body, geometry)
    center = position(state, body)
    if region is None or polygon is None or center is None:
        return None
    margin = GENERATION_PROTOCOL.inside_margin_m if margin_m is None else float(margin_m)
    half = geometry.bodies[body].height_m / 2.0
    bottom, top = center[2] - half, center[2] + half
    if abs(bottom - region.z_low) > FLOOR_CONTACT_M or top > region.z_high:
        return False
    distances = _signed_edge_distances(polygon, np.asarray(region.polygon, dtype=np.float64))
    return bool((distances >= margin).all())


def joint_closed(state, fixture_name, *, tolerance_m: float | None = None) -> bool | None:
    measured = joint_position(state, fixture_name)
    if measured is None:
        return None
    tolerance = GENERATION_PROTOCOL.joint_closed_tolerance_m if tolerance_m is None else float(tolerance_m)
    return abs(measured) <= tolerance


def joint_open(state, fixture_name, geometry: SceneGeometry | None) -> bool | None:
    fixture = None if geometry is None else geometry.fixtures.get(fixture_name)
    joint = None if fixture is None else fixture_joint(fixture)
    measured = joint_position(state, fixture_name)
    if joint is None or measured is None:
        return None
    return measured >= joint.open_threshold_m


def access_clear(state, body, geometry: SceneGeometry | None, *, reach_m: float = OPEN_FINGER_REACH_M,
                 exclude: Sequence[str] = ()) -> bool | None:
    center = position(state, body)
    if center is None or geometry is None:
        return None
    for other in geometry.bodies:
        if other == body or other in exclude or position(state, other) is None:
            continue
        polygon = _body_polygon(state, other, geometry)
        if polygon is None:
            return None
        if _point_polygon_distance(center[:2], polygon) < reach_m:
            return False
    return True


def slot_clear(state, target, geometry: SceneGeometry | None, *, exclude: Sequence[str] = ()) -> bool | None:
    point = position(state, target)
    if point is None or geometry is None:
        return None
    for other in geometry.bodies:
        if other in exclude or position(state, other) is None:
            continue
        polygon = _body_polygon(state, other, geometry)
        if polygon is None:
            return None
        if _point_polygon_distance(point[:2], polygon) <= 0.0:
            return False
    return True


def resting(states, body, *, speed_mps: float | None = None, boundaries: int | None = None) -> bool | None:
    count = GENERATION_PROTOCOL.settle_boundaries if boundaries is None else int(boundaries)
    limit = GENERATION_PROTOCOL.rest_speed_mps if speed_mps is None else float(speed_mps)
    window = list(states)[-count:]
    speeds = [speed(state, body) for state in window]
    if not window or any(value is None for value in speeds):
        return None
    return len(window) == count and all(value < limit for value in speeds)


def held(states, robots, body, *, boundaries: int | None = None, drift_m: float = HELD_DRIFT_M) -> bool | None:
    count = GENERATION_PROTOCOL.settle_boundaries if boundaries is None else int(boundaries)
    window = list(zip(list(states)[-count:], list(robots)[-count:]))
    offsets = []
    for state, robot in window:
        tip, here = _tip(robot), position(state, body)
        grip = (robot or {}).get("grip")
        if tip is None or here is None or grip is None:
            return None
        if float(grip) > 0.5 or float(np.linalg.norm(here - tip)) > HELD_DISTANCE_M:
            return False
        offsets.append(here - tip)
    if len(offsets) < count:
        return False
    return max(float(np.linalg.norm(item - offsets[0])) for item in offsets) <= drift_m


def lifted(state, initial_state, body, *, lift_m: float = LIFT_MIN_M) -> bool | None:
    here, start = position(state, body), position(initial_state, body)
    if here is None or start is None:
        return None
    return float(here[2] - start[2]) >= lift_m


def passed(states, body, fixture_name, geometry: SceneGeometry | None) -> bool | None:
    fixture = None if geometry is None else geometry.fixtures.get(fixture_name)
    opening = None if fixture is None else passage_of(fixture)
    if opening is None or body not in geometry.bodies:
        return None
    point = np.asarray(opening.point, dtype=np.float64)
    normal = np.asarray(opening.normal, dtype=np.float64)
    along = np.asarray(opening.along, dtype=np.float64)
    half_height = geometry.bodies[body].height_m / 2.0
    previous = None
    for state in states:
        here = position(state, body)
        if here is None:
            previous = None
            continue
        side = float(np.dot(here[:2] - point, normal))
        if previous is not None and previous[0] * side < 0.0:
            fraction = previous[0] / (previous[0] - side)
            crossing = previous[1] + fraction * (here - previous[1])
            within = abs(float(np.dot(crossing[:2] - point, along))) <= opening.half_gap_m
            if within and crossing[2] - half_height < opening.top_z:
                return True
        previous = (side, here)
    return False


def reached(robots, point, *, tolerance_m: float | None = None) -> bool | None:
    tolerance = GENERATION_PROTOCOL.predicate_success_m if tolerance_m is None else float(tolerance_m)
    tips = [tip for tip in (_tip(robot) for robot in robots) if tip is not None]
    if not tips or point is None:
        return None
    goal = np.asarray(point, dtype=np.float64)
    return min(float(np.linalg.norm(tip - goal)) for tip in tips) <= tolerance


_PLACEMENT_SUFFIXES = ("_placed", "_parked", "_restored", "_temp_placed", "_fitted")
_HELD_SUFFIXES = ("_grasped", "_regrasped", "_transported")
_SLOT_CLEAR = {"slot_b_clear": ("target_b", ("object_b",))}
_ACCESS_CLEAR = {"corridor_clear": "object_b"}


def _fixture_for(token: str, step: Mapping[str, Any], geometry: SceneGeometry | None) -> str | None:
    if geometry is None:
        return None
    for candidate in (step.get("object_role"), step.get("aperture_role"), step.get("target_role")):
        if candidate and candidate in geometry.fixtures:
            return str(candidate)
    matches = [name for name, fixture in geometry.fixtures.items() if getattr(fixture, "kind", None) == token]
    return matches[0] if len(matches) == 1 else None


def _all_true(parts) -> bool | None:
    if any(part is False for part in parts):
        return False
    if any(part is None for part in parts):
        return None
    return True


def _placed(states, body, target, geometry: SceneGeometry | None) -> bool | None:
    state = states[-1]
    parts = [at_pose(state, body, target, geometry), resting(states, body)]
    container = None if geometry is None else geometry.containers.get(str(target))
    if container is not None:
        parts.append(inside(state, body, container, geometry))
    return _all_true(parts)


def evaluate_condition(name, step: Mapping[str, Any], states, robots, geometry: SceneGeometry | None = None) -> bool | None:
    """Evaluate a catalog pre/postcondition at the last boundary of ``states``."""
    name = str(name or "")
    states, robots = list(states), list(robots)
    if not name or not states:
        return None
    state = states[-1]
    body, target = step.get("object_role"), step.get("target_role")
    for token in ("drawer", "gate"):
        if name in {f"{token}_open", f"{token}_closed"}:
            fixture = _fixture_for(token, step, geometry)
            if fixture is None:
                return None
            return joint_open(state, fixture, geometry) if name.endswith("_open") else joint_closed(state, fixture)
    if name in _SLOT_CLEAR:
        slot, exclude = _SLOT_CLEAR[name]
        return slot_clear(state, slot, geometry, exclude=exclude)
    if name in _ACCESS_CLEAR:
        return access_clear(state, _ACCESS_CLEAR[name], geometry)
    if name in {"blocker_cleared", "blocker_blocking"}:
        cleared = at_pose(state, body, target, geometry, planar=True)
        if cleared is None:
            return None
        return cleared if name == "blocker_cleared" else not cleared
    if name == "target_reached":
        point = position(state, target)
        if point is None:
            return None
        offset = None if geometry is None else geometry.reach_offsets_m.get(str(target))
        return reached(robots, point + np.asarray([0.0, 0.0, DEFAULT_REACH_OFFSET_M if offset is None else offset]))
    if name.endswith("_free"):
        tip = _tip(robots[-1]) if robots else None
        here = position(state, body)
        grip = (robots[-1] or {}).get("grip") if robots else None
        if tip is None or here is None or grip is None:
            return None
        return float(grip) > 0.5 or float(np.linalg.norm(here - tip)) > HELD_DISTANCE_M
    if name.endswith(("_past_aperture", "_past_gate")):
        fixture = _fixture_for("aperture" if name.endswith("_past_aperture") else "gate", step, geometry)
        return None if fixture is None else passed(states, body, fixture, geometry)
    if name.endswith("_rotated"):
        grip = held(states, robots, body)
        if grip is not True:
            return grip
        start, now = yaw_deg(states[0], body), yaw_deg(state, body)
        if start is None or now is None or geometry is None or body not in geometry.bodies:
            return None
        return yaw_error_deg(now, start + 90.0, geometry.bodies[body].symmetry_deg) <= GENERATION_PROTOCOL.fit_yaw_tolerance_deg
    if name.endswith("_lifted"):
        if target and position(state, target) is not None:
            return _all_true([at_pose(state, body, target, geometry), held(states, robots, body)])
        return _all_true([lifted(state, states[0], body), held(states, robots, body)])
    if name.endswith(_HELD_SUFFIXES):
        return held(states, robots, body)
    if name.endswith("_retrieved"):
        if target:
            return _placed(states, body, target, geometry)
        return _all_true([lifted(state, states[0], body), held(states, robots, body)])
    if name.endswith(_PLACEMENT_SUFFIXES):
        return None if not target else _placed(states, body, target, geometry)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_relations.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/relations.py tests/test_generation_relations.py
git commit -m "Add shared physical relation library for success, goals and labels"
```

### Task 5: Task labels from relations

**Files:**
- Modify (rewrite body): `src/icgs/data/collection/generation/task_labels.py`
- Modify: `tests/test_generation.py:325-376` (two label tests)
- Modify: `tests/test_generation_attempt.py:495-519` (`test_materializer_writes_measured_task_labels_when_steps_are_bound`)
- Test: `tests/test_generation_task_labels.py` (new)

**Interfaces:**
- Consumes: Task 4 `evaluate_condition`, `SceneGeometry`.
- Produces: `materialize_task_labels(structured_steps, object_states, robot_states=None, geometry=None) -> dict[str, np.ndarray]` with keys `rho, rho_valid, nu, nu_valid, epsilon, epsilon_valid`, each shaped `(boundaries, events)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_task_labels.py
import numpy as np

from icgs.data.collection.generation.fixtures import DrawerSpec, fixture_joint
from icgs.data.collection.generation.relations import BodyGeometry, SceneGeometry
from icgs.data.collection.generation.task_labels import materialize_task_labels

SQUARE = ((0.02, -0.02), (0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02))


def _state(object_x, joint=None):
    objects = [
        {"name": "object_a", "position": [object_x, 0.0, 0.77], "orientation_xyzw": [0, 0, 0, 1],
         "linear_velocity": [0.0, 0.0, 0.0]},
        {"name": "target_a", "position": [0.2, 0.0, 0.77], "orientation_xyzw": [0, 0, 0, 1]},
    ]
    articulations = [] if joint is None else [{"name": "drawer_joint", "position": joint}]
    return {"objects": objects, "articulations": articulations}


def _robots(count):
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.3, 0.9]
    return [{"T_w_e": transform, "grip": 1} for _ in range(count)]


def test_displacement_after_placement_gives_rho_one_nu_zero():
    steps = [{"object_role": "object_a", "target_role": "target_a", "postcondition": "object_a_placed"}]
    xs = [0.0] + [0.2] * 6 + [0.23] * 6
    labels = materialize_task_labels(steps, [_state(x) for x in xs], _robots(len(xs)))
    assert labels["nu_valid"][6, 0] and labels["nu"][6, 0] == 1.0
    assert labels["nu_valid"][-1, 0] and labels["nu"][-1, 0] == 0.0
    assert labels["rho"][-1, 0] == 1.0


def test_eligibility_uses_current_prerequisite_state():
    drawer = DrawerSpec(name="drawer", center=(0.3, 0.2), axis=(0.0, -1.0), inner=(0.15, 0.11), object_clearance_m=0.062)
    geometry = SceneGeometry(bodies={"object_a": BodyGeometry(SQUARE, 0.04)}, fixtures={"drawer": drawer})
    steps = [
        {"object_role": "drawer", "postcondition": "drawer_open"},
        {"object_role": "object_a", "target_role": "target_a", "precondition": "drawer_open",
         "postcondition": "object_a_placed"},
    ]
    opened = fixture_joint(drawer).open_threshold_m
    joints = [0.0, opened, opened, 0.0]
    labels = materialize_task_labels(steps, [_state(0.0, joint) for joint in joints], _robots(4), geometry)
    assert labels["epsilon_valid"][:, 1].all()
    assert list(labels["epsilon"][:, 1]) == [0.0, 1.0, 1.0, 0.0]
    assert list(labels["rho"][:, 0]) == [0.0, 1.0, 1.0, 1.0]
    assert labels["nu"][3, 0] == 0.0


def test_undecidable_relations_stay_masked():
    steps = [{"object_role": "drawer", "postcondition": "drawer_open"}]
    labels = materialize_task_labels(steps, [_state(0.0)] * 3, _robots(3))
    assert not labels["nu_valid"].any() and not labels["rho_valid"].any()
    assert labels["epsilon_valid"].all()
```

Replace the two tests in `tests/test_generation.py` (lines 325–376) with states that carry velocities and at least five boundaries at rest:

```python
    def test_task_labels_use_measured_states_and_explicit_masks(self):
        import numpy as np
        steps = [{
            "object_role": "object_a", "target_role": "target_a",
            "postcondition": "object_a_placed", "precondition": None,
        }]

        def state(object_x):
            return {"objects": [
                {"name": "object_a", "position": [object_x, 0.0, 0.0], "linear_velocity": [0.0, 0.0, 0.0]},
                {"name": "target_a", "position": [0.2, 0.0, 0.0]},
            ]}

        states = [state(0.0)] + [state(0.2)] * 5
        robots = [{"T_w_e": np.eye(4), "grip": 1} for _ in states]
        labels = materialize_task_labels(steps, states, robots)
        self.assertEqual(labels["nu"].shape, (6, 1))
        self.assertTrue(labels["nu_valid"][0, 0])
        self.assertFalse(labels["nu"][0, 0])
        self.assertTrue(labels["nu_valid"][5, 0])
        self.assertTrue(labels["rho"][5, 0])
        self.assertTrue(labels["epsilon_valid"].all())

    def test_task_labels_measure_grasp_history_and_eligibility(self):
        import numpy as np
        steps = [
            {"object_role": "object_a", "postcondition": "object_a_grasped", "precondition": "object_a_free"},
            {"object_role": "object_a", "target_role": "target_a", "postcondition": "object_a_placed", "precondition": "object_a_grasped"},
        ]
        states, robots = [], []
        track = [(0.0, 0.4, 1)] + [(0.0, 0.0, 0)] * 5 + [(0.2, 0.2, 1)] * 5
        for object_x, ee_x, grip in track:
            states.append({"objects": [
                {"name": "object_a", "position": [object_x, 0.0, 0.0], "linear_velocity": [0.0, 0.0, 0.0]},
                {"name": "target_a", "position": [0.2, 0.0, 0.0]},
            ]})
            transform = np.eye(4)
            transform[0, 3] = ee_x
            robots.append({"T_w_e": transform, "grip": grip})
        labels = materialize_task_labels(steps, states, robots)
        self.assertTrue(labels["epsilon_valid"][0, 0])
        self.assertTrue(labels["epsilon"][0, 0])
        self.assertTrue(labels["rho"][5, 0])
        self.assertTrue(labels["epsilon"][5, 1])
        self.assertTrue(labels["rho"][10, 0])
        self.assertFalse(labels["nu"][10, 0])
        self.assertTrue(labels["nu"][10, 1])
```

Replace `test_materializer_writes_measured_task_labels_when_steps_are_bound` in `tests/test_generation_attempt.py`: the scene states now carry velocities (resting is otherwise undecidable) and placement is established only after 5 resting boundaries:

```python
def test_materializer_writes_measured_task_labels_when_steps_are_bound(tmp_path: Path):
    raw = _raw(predicates_ok=True)
    states = tuple(
        {"objects": [
            {"name": "object_a", "position": [0.2 if index else 0.0, 0.0, 0.0], "linear_velocity": [0.0, 0.0, 0.0]},
            {"name": "target_a", "position": [0.2, 0.0, 0.0]},
        ]}
        for index in range(5)
    )
    raw = RawAttempt(
        observations=raw.observations, actions=raw.actions, scene_states=states,
        collision_events=raw.collision_events, sim_time_s=raw.sim_time_s,
        predicates_ok=True, terminal_reason=raw.terminal_reason,
    )
    binding = {
        **_binding(),
        "structured_steps": [{
            "object_role": "object_a", "target_role": "target_a",
            "postcondition": "object_a_placed", "precondition": None,
        }],
    }
    materialized = materialize_raw_attempt(raw, _job(), binding)
    assert materialized.episode_record["nu_valid"].all()
    assert materialized.episode_record["rho"][1][0] == 0.0
    assert materialized.episode_record["rho"][4][0] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_task_labels.py -q`
Expected: FAIL (`materialize_task_labels() takes from 2 to 3 positional arguments` or wrong ε values).

- [ ] **Step 3: Rewrite `task_labels.py`**

```python
"""Task-memory labels for phase-1 event views (decision 0016).

``nu`` is the step's postcondition relation recomputed at each boundary,
``rho`` its historical occurrence, and ``epsilon`` the conjunction of the
step's prerequisite relations at that boundary (current state, not history).
All come from :mod:`relations`; undecidable relations stay masked.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from icgs.data.collection.generation.relations import SceneGeometry, evaluate_condition


def _source_step(steps: Sequence[Mapping[str, Any]], index: int, name: str) -> Mapping[str, Any] | None:
    for prior in range(index - 1, -1, -1):
        if name in {steps[prior].get("postcondition"), steps[prior].get("success_predicate")}:
            return steps[prior]
    return None


def materialize_task_labels(
    structured_steps: Sequence[Mapping[str, Any]],
    object_states: Sequence[Mapping[str, Any]],
    robot_states: Sequence[Mapping[str, Any]] | None = None,
    geometry: SceneGeometry | None = None,
) -> dict[str, np.ndarray]:
    """Return ``rho/nu/epsilon`` and validity masks aligned to T+1 boundaries."""
    boundaries, events = len(object_states), len(structured_steps)
    shape = (boundaries, events)
    rho, nu, epsilon = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    rho_valid, nu_valid, epsilon_valid = (np.zeros(shape, dtype=bool) for _ in range(3))
    robots = list(robot_states or ({} for _ in range(boundaries)))
    if len(robots) != boundaries:
        raise ValueError("robot_states must align with object_states")
    states = list(object_states)
    for t in range(boundaries):
        states_t, robots_t = states[: t + 1], robots[: t + 1]
        for index, step in enumerate(structured_steps):
            observed = evaluate_condition(step.get("postcondition"), step, states_t, robots_t, geometry)
            if observed is not None:
                nu[t, index] = float(observed)
                nu_valid[t, index] = True
                rho[t, index] = float(observed or (t > 0 and rho[t - 1, index] > 0.5))
                rho_valid[t, index] = True
            prerequisite = step.get("precondition")
            if not prerequisite:
                epsilon[t, index] = 1.0
                epsilon_valid[t, index] = True
                continue
            source = _source_step(structured_steps, index, str(prerequisite)) or step
            current = evaluate_condition(prerequisite, source, states_t, robots_t, geometry)
            if current is not None:
                epsilon[t, index] = float(current)
                epsilon_valid[t, index] = True
    return {
        "rho": rho, "rho_valid": rho_valid,
        "nu": nu, "nu_valid": nu_valid,
        "epsilon": epsilon, "epsilon_valid": epsilon_valid,
    }


__all__ = ["materialize_task_labels"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_task_labels.py tests/test_generation.py tests/test_generation_attempt.py tests/test_generation_worker.py -q`
Expected: PASS. If a worker test asserts old 4 cm label values, update its fixture to carry velocities and ≥ 5 resting boundaries as above; never loosen the relation.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/task_labels.py tests/test_generation_task_labels.py tests/test_generation.py tests/test_generation_attempt.py
git commit -m "Derive task-memory labels from physical relations with current eligibility"
```

### Task 6: Randomization fields and scene mirroring

**Files:**
- Modify: `src/icgs/data/collection/generation/batch.py` (`bounds_from_row`, `apply_layout_randomization`)
- Modify: `src/icgs/data/collection/generation/diversity.py` (`sample_randomization`, `scene_signature`)
- Test: `tests/test_generation_mirroring.py`

**Interfaces:**
- Produces: randomization keys `mirror: bool`, `execution_mode: str`, `offset_side: ±1`; `bounds_from_row(row)` reads `row["randomization"]["mirror"]` (default False) and `row["execution_modes"]`; `apply_layout_randomization` negates y of every position when `randomization["mirror"]` is true (before translation).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_mirroring.py
import pytest

from icgs.data.collection.generation.batch import apply_layout_randomization, bounds_from_row
from icgs.data.collection.generation.diversity import sample_randomization

ROW = {
    "execution_modes": ["scripted_direct", "scripted_offset"],
    "randomization": {
        "translation_m": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
        "yaw_deg": [0.0, 0.0], "scale": [1.0, 1.0], "mirror": True,
    },
}


def test_mirror_and_modes_are_balanced_by_index():
    bounds = bounds_from_row(ROW)
    samples = [sample_randomization("T04", 1, index, bounds) for index in range(8)]
    assert [item["mirror"] for item in samples] == [False, True] * 4
    assert [item["execution_mode"] for item in samples] == ["scripted_direct"] * 2 + ["scripted_offset"] * 2 + ["scripted_direct"] * 2 + ["scripted_offset"] * 2
    assert {item["offset_side"] for item in samples} <= {-1, 1}


def test_mirror_disabled_by_default():
    row = {"randomization": {**ROW["randomization"], "mirror": False}}
    bounds = bounds_from_row(row)
    assert not any(sample_randomization("T04", 1, index, bounds)["mirror"] for index in range(4))
    assert sample_randomization("T04", 1, 0, bounds)["execution_mode"] == "scripted_waypoint_v1"


def test_layout_mirroring_negates_y_before_translation():
    objects = {"object_a": ([0.25, 0.10, 0.775], [0.04, 0.04, 0.04], [1, 0, 0]),
               "target_a": ([0.30, -0.12, 0.775], [0.05, 0.05, 0.02], [0, 1, 0])}
    randomization = {"object_translation_m": {"x": 0.01, "y": 0.02, "z": 0.0}, "mirror": True}
    result = apply_layout_randomization(objects, randomization)
    pos_a = result["object_a"]["pos"] if isinstance(result["object_a"], dict) else result["object_a"][0]
    pos_t = result["target_a"]["pos"] if isinstance(result["target_a"], dict) else result["target_a"][0]
    assert pos_a[:2] == pytest.approx([0.26, -0.08])
    assert pos_t[:2] == pytest.approx([0.31, 0.14])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_mirroring.py -q`
Expected: FAIL with `KeyError: 'mirror'`.

- [ ] **Step 3: Implement**

In `batch.py` `bounds_from_row`, extend the returned dict:

```python
        "mirror": bool(randomization.get("mirror", False)),
        "execution_modes": tuple(row.get("execution_modes") or (GENERATION_PROTOCOL.execution_mode,)),
```

In `diversity.py` `sample_randomization`, directly before `sample["train_subset"] = train_subset_for_sample(sample)`:

```python
    modes = tuple(bounds.get("execution_modes") or (GENERATION_PROTOCOL.execution_mode,))
    sample["execution_mode"] = str(modes[(episode_index // 2) % len(modes)])
    sample["mirror"] = bool(bounds.get("mirror")) and episode_index % 2 == 1
    sample["offset_side"] = 1 if _unit_interval(seed, 20) >= 0.5 else -1
```

In `diversity.py` `scene_signature`, after building `payload` and before hashing:

```python
    for key in ("mirror", "execution_mode"):
        if sample.get(key) is not None:
            payload[key] = sample.get(key)
```

In `batch.py` `apply_layout_randomization`, read `mirror = bool(randomization.get("mirror"))` at the top, and in both branches replace the first use of the designed position with a mirrored copy:

```python
            pos = list(spec["pos"])
            if mirror:
                pos[1] = -pos[1]
```

and in the tuple branch:

```python
            pos, size, color = spec
            pos = [pos[0], -pos[1], pos[2]] if mirror else list(pos)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_mirroring.py tests/test_generation_batch.py tests/test_generation.py tests/test_generation_layout_clearance.py -q`
Expected: PASS (manifest rows have no `mirror`, so existing plans are unchanged apart from the new keys).

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/batch.py src/icgs/data/collection/generation/diversity.py tests/test_generation_mirroring.py
git commit -m "Add balanced scene mirroring and execution mode fields to attempt plans"
```

### Task 7: Clearance model with fixtures, palm and release rule

**Files:**
- Modify: `src/icgs/data/collection/generation/layout_clearance.py`
- Test: `tests/test_generation_fixture_clearance.py`

**Interfaces:**
- Consumes: Task 2 (`fixture_boxes`, `fixture_joint`, `initial_opening`, `grip_point`, `inner_region`, `passage_of`, `Fixture`).
- Produces: `audit_prepared_attempt(objects, routine, *, fixtures=(), containers=None, margin_m=0.003) -> list[ClearanceViolation]`. Object specs may carry `symmetry_deg` (generator bodies, flats across local y). Routine semantics: `open_articulation`/`close_articulation` with `obj` = fixture name (tool returns to the default yaw; the fin must slide along world y); `transport_through_aperture` with `aperture_wp` = fixture name (low passage); `push` with `via` = fixture name (gap centre); `reach` checks fingers and palm at the reach point. Targets listed in `containers` under a jointed fixture are tray-relative (they move with the joint). Before a release into a container the tool turns so the finger axis is world y (pocket: the pocket frame's y) and the held body turns with it. Grasps try every face pair allowed by symmetry, nearest first, and keep the first clear one. Constants `PALM_BOX_M`, `PALM_Z_ABOVE_TIP_M`, `FINGER_TOP_ABOVE_TIP_M`, `PLACE_RELEASE_CLEARANCE_M`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_fixture_clearance.py
from icgs.data.collection.generation.fixtures import (
    ApertureSpec, DrawerSpec, PocketSpec, fixture_joint, inner_region, scale_fixture,
)
from icgs.data.collection.generation.layout_clearance import audit_prepared_attempt

LONG = [0.0735, 0.035, 0.035]  # elongated box: closes across local y only
OPEN_PLACE_CLOSE = [
    {"type": "open_articulation", "obj": "drawer"},
    {"type": "pick_place", "obj": "object_a", "target": "target_a"},
    {"type": "close_articulation", "obj": "drawer"},
]


def _drawer():
    return DrawerSpec(name="drawer", center=(0.30, 0.20), axis=(0.0, -1.0), inner=(0.15, 0.11),
                      object_clearance_m=0.062)


def _drawer_objects(drawer, height=0.045):
    xs, ys = zip(*inner_region(drawer, 0.0).polygon)
    return {
        "object_a": {"pos": [0.10, -0.25, 0.752 + height / 2], "size": [0.045, 0.045, height]},
        # Tray-relative target, declared in the closed drawer.
        "target_a": {"pos": [sum(xs) / 4, sum(ys) / 4, drawer.floor_top_z + height / 2], "size": [0.05, 0.05, 0.02]},
    }


def test_placing_into_a_closed_drawer_hits_the_housing():
    drawer = _drawer()
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    violations = audit_prepared_attempt(_drawer_objects(drawer), routine, fixtures=(drawer,),
                                        containers={"target_a": "drawer"})
    assert any(item.other.startswith(("drawer_roof", "drawer_wall")) for item in violations)


def test_open_place_close_is_clear():
    drawer = _drawer()
    assert audit_prepared_attempt(_drawer_objects(drawer), OPEN_PLACE_CLOSE, fixtures=(drawer,),
                                  containers={"target_a": "drawer"}) == []


def test_content_taller_than_roof_clearance_blocks_closing():
    drawer = _drawer()
    violations = audit_prepared_attempt(_drawer_objects(drawer, height=0.08), OPEN_PLACE_CLOSE, fixtures=(drawer,),
                                        containers={"target_a": "drawer"})
    assert any(item.phase.startswith("ride") for item in violations)


def test_tall_end_walls_violate_the_release_rule():
    objects = {"object_a": {"pos": [0.25, -0.10, 0.7745], "size": [0.045, 0.045, 0.045]},
               "target_a": {"pos": [0.25, 0.16, 0.758 + 0.0225], "size": [0.05, 0.05, 0.02]}}
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    tall = PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.053, 0.08), end_wall_h_m=0.03)
    assert any(item.phase.startswith("release") and "_end_" in item.other
               for item in audit_prepared_attempt(objects, routine, fixtures=(tall,)))
    low = PocketSpec(name="holder", center=(0.25, 0.16), inner=(0.053, 0.08))
    assert audit_prepared_attempt(objects, routine, fixtures=(low,)) == []


def test_aperture_passes_aligned_long_body_and_blocks_turned_one_at_both_scale_extremes():
    aperture = ApertureSpec(name="aperture", center=(0.25, 0.02), axis=(1.0, 0.0), gap_m=0.051)
    routine = [
        {"type": "grasp", "obj": "object_a"},
        {"type": "transport_through_aperture", "obj": "object_a", "target": "target_a", "aperture_wp": "aperture"},
    ]
    for scale in (0.8, 1.2):
        size = [value * scale for value in LONG]
        aligned = {"object_a": {"pos": [0.25, -0.12, 0.752 + size[2] / 2], "size": size, "yaw_deg": 90.0},
                   "target_a": {"pos": [0.25, 0.16, 0.752 + size[2] / 2], "size": [0.06, 0.06, 0.02]}}
        scaled = scale_fixture(aperture, scale)
        assert not [v for v in audit_prepared_attempt(aligned, routine, fixtures=(scaled,)) if v.phase.startswith("passage")]
        turned = {**aligned, "object_a": {**aligned["object_a"], "yaw_deg": 0.0}}
        assert [v for v in audit_prepared_attempt(turned, routine, fixtures=(scaled,)) if v.phase.startswith("passage")]


def test_square_body_is_grasped_along_the_clear_axis_but_long_body_is_blocked():
    objects = {"object_a": {"pos": [0.25, 0.0, 0.7745], "size": [0.045, 0.045, 0.045]},
               "blocker": {"pos": [0.25, 0.07, 0.7795], "size": [0.055, 0.055, 0.055]},
               "target_a": {"pos": [0.25, -0.2, 0.7745], "size": [0.05, 0.05, 0.02]}}
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    assert audit_prepared_attempt(objects, routine) == []
    objects["object_a"] = {"pos": [0.25, 0.0, 0.7695], "size": LONG}
    assert any(item.phase.startswith("grasp") and item.other == "blocker"
               for item in audit_prepared_attempt(objects, routine))


def test_palm_collides_with_tall_neighbour_along_the_only_grasp_axis():
    objects = {"object_a": {"pos": [0.25, 0.0, 0.7695], "size": LONG},
               "blocker_tall": {"pos": [0.25, 0.09, 0.80], "size": [0.03, 0.03, 0.10]},
               "target_a": {"pos": [0.25, -0.2, 0.7695], "size": [0.05, 0.05, 0.02]}}
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    assert any(item.phase.startswith("palm:grasp") for item in audit_prepared_attempt(objects, routine))
```

In `tests/test_generation_layout_clearance.py`, the existing unit test assumed a square body can only be grasped on one axis. Replace it (face-pair choice now grasps a square body along the clear axis):

```python
def test_clearance_audit_flags_a_neighbour_inside_the_open_finger_span():
    # An elongated body closes across its short side only, so a neighbour on that axis blocks it.
    objects = {
        "object_a": {"pos": [0.25, 0.0, 0.7695], "size": [0.0735, 0.035, 0.035]},
        "blocker": {"pos": [0.25, 0.07, 0.775], "size": [0.05, 0.05, 0.05]},
        "target_a": {"pos": [0.25, 0.3, 0.775], "size": [0.06, 0.06, 0.03]},
    }
    routine = [{"type": "pick_place", "obj": "object_a", "target": "target_a"}]
    phases = {(item.phase, item.body, item.other) for item in audit_prepared_attempt(objects, routine)}
    assert ("grasp:pick_place", "object_a", "blocker") in phases
    # The same neighbour offset along x clears the thin finger side.
    objects["blocker"]["pos"] = [0.34, 0.0, 0.775]
    assert audit_prepared_attempt(objects, routine) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_fixture_clearance.py -q`
Expected: FAIL with `TypeError: audit_prepared_attempt() got an unexpected keyword argument 'fixtures'`.

- [ ] **Step 3: Implement the extension**

Add after the existing imports and constants of `layout_clearance.py`:

```python
from icgs.data.collection.generation.fixtures import (
    Fixture, PocketSpec, fixture_boxes, fixture_joint, grip_point, initial_opening, inner_region, passage_of,
)

PALM_BOX_M = ((-0.064, 0.020), (-0.100, 0.104))
PALM_Z_ABOVE_TIP_M = (0.037, 0.142)
FINGER_TOP_ABOVE_TIP_M = 0.042
PLACE_RELEASE_CLEARANCE_M = 0.004
```

Add helpers after `_sweep`:

```python
def _hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(set((round(float(x), 12), round(float(y), 12)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _translate(polygon, dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in polygon]


def _inside_polygon(point, polygon) -> bool:
    n = len(polygon)
    for index in range(n):
        (x0, y0), (x1, y1) = polygon[index], polygon[(index + 1) % n]
        if (x1 - x0) * (point[1] - y0) - (y1 - y0) * (point[0] - x0) < -1e-12:
            return False
    return True


def _grasp_turn_options(size_xy, yaw_deg: float, finger_axis_deg: float,
                        symmetry_deg: float | None = None) -> list[float]:
    """Tool-yaw changes aligning the fingers with a face pair, nearest first.

    A square footprint offers both face pairs, so a neighbour on one axis does
    not block a grasp along the other; elongated footprints close across the
    short side only. Generator bodies (``symmetry_deg`` set) have flats across
    their local y axis.
    """
    width, depth = float(size_xy[0]), float(size_xy[1])
    if symmetry_deg is None:
        delta = grasp_yaw_delta_deg(size_xy, yaw_deg, finger_axis_deg)
        period = 180.0 if max(width, depth) >= ELONGATED_ASPECT_RATIO * min(width, depth) else 90.0
    else:
        period = float(symmetry_deg)
        delta = (float(yaw_deg) + 90.0 - float(finger_axis_deg) + period / 2.0) % period - period / 2.0
    if period >= 180.0:
        return [delta]
    options = {delta + k * period for k in (-1, 0, 1)}
    return sorted((value for value in options if abs(value) <= 135.0), key=lambda value: (abs(value), value))
```

Replace `_rest_center`:

```python
def _rest_center(xy, size, supports, regions=()) -> list[float]:
    z = TABLE_TOP_Z_M + size[2] / 2.0
    for position, support_size in supports:
        if abs(xy[0] - position[0]) <= support_size[0] / 2.0 and abs(xy[1] - position[1]) <= support_size[1] / 2.0:
            z = max(z, position[2] + support_size[2] / 2.0 + size[2] / 2.0)
    for region in regions:
        if region is not None and _inside_polygon(xy, region.polygon):
            z = max(z, region.z_low + size[2] / 2.0)
    return [float(xy[0]), float(xy[1]), z]
```

Replace `audit_prepared_attempt` entirely:

```python
def audit_prepared_attempt(
    objects: Mapping[str, Any],
    routine: Iterable[Mapping[str, Any]],
    *,
    fixtures: Sequence[Fixture] = (),
    containers: Mapping[str, str] | None = None,
    margin_m: float = 0.003,
) -> list[ClearanceViolation]:
    """Return every clearance violation of one prepared attempt (empty when clean)."""
    specs = {name: _spec(item) for name, item in objects.items()}
    symmetry = {name: item.get("symmetry_deg") for name, item in objects.items() if isinstance(item, Mapping)}
    supports = [(position, size) for name, (position, size, _yaw) in specs.items() if name in SUPPORT_NAMES]
    by_name = {fixture.name: fixture for fixture in fixtures}
    openings = {fixture.name: initial_opening(fixture) for fixture in fixtures}
    containers = dict(containers or {})

    def regions():
        return [inner_region(fixture, openings[fixture.name]) for fixture in fixtures]

    def target_xyz(name: str) -> list[float]:
        position = list(specs[name][0])
        fixture = by_name.get(containers.get(name, ""))
        joint = None if fixture is None else fixture_joint(fixture)
        if joint is not None:
            position[0] += joint.axis[0] * openings[fixture.name]
            position[1] += joint.axis[1] * openings[fixture.name]
        return position

    state: dict[str, tuple[list[float], list[float], float]] = {}
    for name, (position, size, yaw) in specs.items():
        if is_solid(name):
            rest = position if name in SUPPORT_NAMES else _rest_center(position, size, supports, regions())
            state[name] = (rest, size, yaw)
    violations: list[ClearanceViolation] = []

    def footprint(name: str) -> list[tuple[float, float]]:
        position, size, yaw = state[name]
        return _rectangle(position, size[0] / 2.0, size[1] / 2.0, yaw)

    def z_range(name: str) -> tuple[float, float]:
        position, size, _yaw = state[name]
        return position[2] - size[2] / 2.0, position[2] + size[2] / 2.0

    def check(phase: str, body: str, polygon, bottom_z: float, top_z: float = float("inf"),
              skip=frozenset(), out: list | None = None) -> None:
        sink = violations if out is None else out
        for other in state:
            if other == body or other in skip:
                continue
            low, high = z_range(other)
            if high <= bottom_z + 1e-9 or low >= top_z - 1e-9:
                continue
            if other in SUPPORT_NAMES and phase.startswith(("grasp", "release", "push", "slide")):
                if bottom_z >= high - 1e-9:
                    continue
            depth = _penetration(polygon, footprint(other))
            if depth > 0.0:
                sink.append(ClearanceViolation(phase, body, other, depth))
        for fixture in fixtures:
            for box in fixture_boxes(fixture, openings[fixture.name]):
                if box.name in skip:
                    continue
                low, high = box.z_range
                if high <= bottom_z + 1e-9 or low >= top_z - 1e-9:
                    continue
                depth = _penetration(polygon, box.footprint())
                if depth > 0.0:
                    sink.append(ClearanceViolation(phase, body, box.name, depth))

    def check_hand(phase: str, body: str, center_xy, tip_z: float, turn: float, *,
                   open_fingers: bool = True, skip=frozenset(), out: list | None = None) -> None:
        fingers = OPEN_FINGERS_M if open_fingers else CLOSED_FINGERS_M
        check(phase, body, _box(center_xy, fingers, margin_m, turn), tip_z - FINGERTIP_BELOW_TIP_M,
              tip_z + FINGER_TOP_ABOVE_TIP_M, skip, out)
        check(f"palm:{phase}", body, _box(center_xy, PALM_BOX_M, 0.0, turn),
              tip_z + PALM_Z_ABOVE_TIP_M[0], tip_z + PALM_Z_ABOVE_TIP_M[1], skip, out)

    for name in list(state):
        if name in SUPPORT_NAMES:
            continue
        low, high = z_range(name)
        check("spawn", name, footprint(name), low, high)

    finger_axis = DEFAULT_FINGER_AXIS_DEG
    for step in routine:
        kind = step.get("type")
        body, target = step.get("obj"), step.get("target")
        held = bool(step.get("held"))
        if kind in {"open_articulation", "close_articulation"} and body in by_name:
            fixture = by_name[body]
            joint = fixture_joint(fixture)
            finger_axis = DEFAULT_FINGER_AXIS_DEG
            if abs(joint.axis[0]) > 1e-6:
                violations.append(ClearanceViolation(f"grip:{kind}", body, "fin_axis_not_world_y", abs(joint.axis[0])))
            start = openings[fixture.name]
            goal = joint.stroke_m - 0.005 if kind == "open_articulation" else 0.0
            gx, gy, gz = grip_point(fixture, start)
            check_hand(f"grip:{kind}", body, (gx, gy), gz, 0.0, skip=frozenset({f"{fixture.name}_fin"}))
            region = inner_region(fixture, start)
            riders = [name for name in state
                      if region is not None and _inside_polygon(state[name][0][:2], region.polygon)]
            dx, dy = joint.axis[0] * (goal - start), joint.axis[1] * (goal - start)
            moving_before = [box for box in fixture_boxes(fixture, start) if box.moving]
            moving_after = [box for box in fixture_boxes(fixture, goal) if box.moving]
            for before, after in zip(moving_before, moving_after):
                swept = _hull(before.footprint() + after.footprint())
                low, high = before.z_range
                for other in state:
                    if other in riders:
                        continue
                    o_low, o_high = z_range(other)
                    if o_high <= low + 1e-9 or o_low >= high - 1e-9:
                        continue
                    depth = _penetration(swept, footprint(other))
                    if depth > 0.0:
                        violations.append(ClearanceViolation(f"slide:{kind}", before.name, other, depth))
            static_boxes = [box for box in fixture_boxes(fixture, start) if not box.moving]
            for name in riders:
                position, size, yaw = state[name]
                local = _rectangle((0.0, 0.0), size[0] / 2.0, size[1] / 2.0, yaw)
                swept = _hull(_translate(local, position[0], position[1])
                              + _translate(local, position[0] + dx, position[1] + dy))
                low, high = z_range(name)
                for box in static_boxes:
                    b_low, b_high = box.z_range
                    if b_high <= low + 1e-9 or b_low >= high - 1e-9:
                        continue
                    depth = _penetration(swept, box.footprint())
                    if depth > 0.0:
                        violations.append(ClearanceViolation(f"ride:{kind}", name, box.name, depth))
                state[name] = ([position[0] + dx, position[1] + dy, position[2]], size, yaw)
            openings[fixture.name] = goal
            continue
        if body in state and (kind in _GRASP_TYPES or (kind in _HELD_CONTINUATION_TYPES and not held)):
            position, size, yaw = state[body]
            options = _grasp_turn_options(size[:2], yaw, finger_axis, symmetry.get(body))
            chosen, first = options[0], None
            for delta in options:
                trial: list[ClearanceViolation] = []
                check_hand(f"grasp:{kind}", body, position, position[2],
                           finger_axis + delta - DEFAULT_FINGER_AXIS_DEG, out=trial)
                if first is None:
                    first = trial
                if not trial:
                    chosen, first = delta, []
                    break
            violations.extend(first or [])
            finger_axis += chosen
        if kind == "grasp_rotate":
            finger_axis += float(step.get("yaw_deg", 90.0))
            if body in state:
                position, size, yaw = state[body]
                state[body] = (position, size, yaw + float(step.get("yaw_deg", 90.0)))
        if kind == "reach" and target in specs:
            point = target_xyz(target)
            check_hand("reach", "tool", point, point[2] + float(step.get("reach_z", 0.03)),
                       finger_axis - DEFAULT_FINGER_AXIS_DEG)
        if kind == "transport_through_aperture" and body in state and step.get("aperture_wp") in by_name:
            aperture = by_name[step["aperture_wp"]]
            opening = passage_of(aperture)
            position, size, yaw = state[body]
            half_h = size[2] / 2.0
            tip_z = opening.top_z + 0.005 + FINGERTIP_BELOW_TIP_M
            if tip_z - half_h < TABLE_TOP_Z_M + 0.005:
                violations.append(ClearanceViolation("passage:low", body, aperture.name,
                                                     TABLE_TOP_Z_M + 0.005 - (tip_z - half_h)))
            reach = max(size[0], size[1]) + 0.02
            nx, ny = opening.normal
            px, py = opening.point
            local = _rectangle((0.0, 0.0), size[0] / 2.0, size[1] / 2.0, yaw)
            swept = _hull(_translate(local, px - nx * reach, py - ny * reach)
                          + _translate(local, px + nx * reach, py + ny * reach))
            check("passage", body, swept, tip_z - half_h, tip_z + half_h, frozenset({body}))
        if kind == "push" and body in state and target in specs:
            position, size, yaw = state[body]
            goal = target_xyz(target)
            via_name = step.get("via")
            if via_name in by_name:
                gap = passage_of(by_name[via_name])
                path = [[gap.point[0], gap.point[1], position[2]], goal]
            elif via_name in specs and via_name != target:
                path = [target_xyz(via_name), goal]
            else:
                path = [goal]
            contact = CLOSED_FINGERS_M[1][1] + size[1] / 2.0
            dx, dy = path[0][0] - position[0], path[0][1] - position[1]
            norm = math.hypot(dx, dy) or 1.0
            start = (position[0] - dx / norm * (contact + 0.03), position[1] - dy / norm * (contact + 0.03))
            previous = position
            local = _rectangle((0.0, 0.0), size[0] / 2.0, size[1] / 2.0, yaw)
            for point in path:
                ux, uy = point[0] - previous[0], point[1] - previous[1]
                length = math.hypot(ux, uy) or 1.0
                end = (point[0] - ux / length * contact, point[1] - uy / length * contact)
                check("push", body, _sweep(start, end, CLOSED_FINGERS_M, margin_m), position[2] - FINGERTIP_BELOW_TIP_M,
                      position[2] + FINGER_TOP_ABOVE_TIP_M, frozenset({body}))
                body_sweep = _hull(_translate(local, previous[0], previous[1]) + _translate(local, point[0], point[1]))
                check("push:body", body, body_sweep, position[2] - size[2] / 2.0, position[2] + size[2] / 2.0,
                      frozenset({body}))
                start, previous = end, point
            state[body] = ([goal[0], goal[1], position[2]], size, yaw)
        if kind in {"open_articulation", "close_articulation"} and body in state and target in specs:
            # Legacy handle-block articulation (pre-fixture layouts).
            position, size, yaw = state[body]
            goal = specs[target][0]
            handle_extent = ((-size[0] / 2.0, size[0] / 2.0), (-size[1] / 2.0, size[1] / 2.0))
            bottom = position[2] - FINGERTIP_BELOW_TIP_M
            check("slide:fingers", body, _sweep(position, goal, CLOSED_FINGERS_M, margin_m), bottom)
            check("slide:handle", body, _sweep(position, goal, handle_extent, 0.0), position[2] - size[2] / 2.0)
        releases = kind in _RELEASE_TYPES or (kind == "transport_through_aperture" and step.get("release", True))
        if releases and body in state and target in specs:
            goal = target_xyz(target)
            position, size, yaw = state[body]
            container = by_name.get(containers.get(target, ""))
            if container is not None:
                # The expert turns the held body so the fingers align with the container
                # (pocket frame y, otherwise world y) before descending into it.
                release_axis = DEFAULT_FINGER_AXIS_DEG + (container.yaw_deg if isinstance(container, PocketSpec) else 0.0)
                yaw += release_axis - finger_axis
                finger_axis = release_axis
            placed = [goal[0], goal[1], position[2]] if "handle" in body else _rest_center(goal, size, supports, regions())
            state[body] = (placed, size, yaw)
            low, high = z_range(body)
            check(f"place:{kind}", body, footprint(body), low, high)
            check_hand(f"release:{kind}", body, goal, placed[2] + PLACE_RELEASE_CLEARANCE_M,
                       finger_axis - DEFAULT_FINGER_AXIS_DEG, skip=frozenset({body}))
    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_fixture_clearance.py tests/test_generation_layout_clearance.py -q`
Expected: new tests PASS. The two pool tests now also check palms against real heights and fail on the current handle-block layouts (dry run 2026-09-29: palm contacts in G3, R3, T05, T12, T16, V04 nominal; the same programs in perturbed plans). These are Phase 3 layout inputs; keep the palm geometry and mark exactly these two tests:

```python
@pytest.mark.xfail(strict=True, reason="Phase 3 relayout: palm contacts with handle blocks in G3, R3, T05, T12, T16, V04")
def test_every_nominal_attempt_is_physically_clear_for_the_panda_fingers():
```

```python
@pytest.mark.xfail(strict=True, reason="Phase 3 relayout: palm contacts with handle blocks in G3, R3, T05, T12, T16, V04")
def test_perturbed_contacts_come_only_from_declared_blocker_insertion():
```

Phase 3 Task 19 removes both markers.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/layout_clearance.py tests/test_generation_fixture_clearance.py tests/test_generation_layout_clearance.py docs/plans/active/physical-dependency-generation.md
git commit -m "Teach the clearance audit fixtures, palm, face-pair choice, articulation and passage"
```

### Task 8: Dependency audit harness

**Files:**
- Create: `src/icgs/data/collection/generation/dependency_audit.py`
- Test: `tests/test_generation_dependency_audit.py`

**Interfaces:**
- Consumes: Task 7 `audit_prepared_attempt(objects, routine, *, fixtures)`.
- Produces: `routine_units(routine) -> list[tuple[int, ...]]`; `DependencyFinding(program_id, unit, label, required)`; `skip_audit(program_id, objects, routine, fixtures=()) -> list[DependencyFinding]`; `ORDERING_ONLY: dict[str, frozenset[str]]` (program → unit labels allowed to be ordering-only); `unexplained_skips(findings) -> list[DependencyFinding]`; `consequence_conflicts(objects, routine, fixtures, variant) -> bool` for variants `packing_shift`, `park_shift`, `turn_held`, `too_tall` taking `(objects, routine, fixtures, **params)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_dependency_audit.py
from icgs.data.collection.generation.dependency_audit import (
    DependencyFinding, consequence_conflicts, routine_units, skip_audit, unexplained_skips,
)


def test_units_group_held_chains():
    routine = [
        {"type": "grasp", "obj": "object_a"},
        {"type": "lift", "obj": "object_a", "held": True},
        {"type": "place", "obj": "object_a", "target": "target_a"},
        {"type": "pick_place", "obj": "object_b", "target": "target_b"},
    ]
    assert routine_units(routine) == [(0, 1, 2), (3,)]


def _blocked_scene():
    # An elongated target closes across local y only, so a blocker on that axis blocks it;
    # a square target would simply be grasped along the clear axis.
    objects = {
        "blocker": {"pos": [0.25, 0.07, 0.7795], "size": [0.055, 0.055, 0.055]},
        "object_a": {"pos": [0.25, 0.0, 0.7695], "size": [0.0735, 0.035, 0.035]},
        "park_target": {"pos": [0.40, 0.07, 0.7795], "size": [0.08, 0.08, 0.02]},
        "target_a": {"pos": [0.25, -0.20, 0.7745], "size": [0.06, 0.06, 0.02]},
    }
    routine = [
        {"type": "pick_place", "obj": "blocker", "target": "park_target"},
        {"type": "pick_place", "obj": "object_a", "target": "target_a"},
    ]
    return objects, routine


def test_adjacent_blocker_is_physically_required():
    objects, routine = _blocked_scene()
    findings = skip_audit("T13", objects, routine)
    assert [item.required for item in findings] == [True]
    assert unexplained_skips(findings) == []


def test_distant_blocker_is_reported_as_symbolic():
    objects, routine = _blocked_scene()
    objects["blocker"] = {**objects["blocker"], "pos": [0.25, 0.20, 0.7795]}
    findings = skip_audit("T13", objects, routine)
    assert unexplained_skips(findings) == [DependencyFinding("T13", (0,), "pick_place(blocker->park_target)", False)]


def test_park_shift_toward_the_retrieved_body_conflicts():
    objects, routine = _blocked_scene()
    objects["park_target"] = {**objects["park_target"], "pos": [0.36, 0.0, 0.7795]}
    assert consequence_conflicts(objects, routine, (), "park_shift", shift_m=0.03) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_generation_dependency_audit.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `dependency_audit.py`**

```python
"""Static dependency audits (decision 0016). Simulator-free.

A declared enabling step is physically required when removing it makes a
later step collide in the clearance model. A representative wrong decision of
a delayed-consequence family must collide downstream. Ordering-only steps are
listed explicitly with the program they belong to.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from icgs.data.collection.generation.layout_clearance import audit_prepared_attempt

_RELEASES = frozenset({"place", "pick_place"})
ORDERING_ONLY: dict[str, frozenset[str]] = {
    "T14": frozenset({"pick_place(blocker->park_target)"}),
}


@dataclass(frozen=True)
class DependencyFinding:
    program_id: str
    unit: tuple[int, ...]
    label: str
    required: bool


def routine_units(routine: Sequence[Mapping[str, Any]]) -> list[tuple[int, ...]]:
    units: list[tuple[int, ...]] = []
    index = 0
    while index < len(routine):
        step = routine[index]
        if step.get("type") != "grasp":
            units.append((index,))
            index += 1
            continue
        chain = [index]
        cursor = index + 1
        while cursor < len(routine) and routine[cursor].get("obj") == step.get("obj"):
            kind = routine[cursor].get("type")
            if kind == "grasp":
                break
            chain.append(cursor)
            if kind in _RELEASES or (kind == "transport_through_aperture" and routine[cursor].get("release", True)):
                break
            cursor += 1
        units.append(tuple(chain))
        index = chain[-1] + 1
    return units


def _label(routine, unit) -> str:
    first, last = routine[unit[0]], routine[unit[-1]]
    kinds = "+".join(str(routine[index].get("type")) for index in unit)
    return f"{kinds}({first.get('obj') or ''}->{last.get('target') or ''})"


def _keys(objects, routine, fixtures):
    return {(item.phase, item.body, item.other) for item in audit_prepared_attempt(objects, routine, fixtures=fixtures)}


def skip_audit(program_id: str, objects: Mapping[str, Any], routine: Sequence[Mapping[str, Any]],
               fixtures: Sequence[Any] = ()) -> list[DependencyFinding]:
    base = _keys(objects, routine, fixtures)
    findings = []
    units = routine_units(routine)
    for unit in units[:-1]:
        trial = [step for index, step in enumerate(routine) if index not in unit]
        new = _keys(objects, trial, fixtures) - base
        findings.append(DependencyFinding(program_id, unit, _label(routine, unit), bool(new)))
    return findings


def unexplained_skips(findings: Sequence[DependencyFinding]) -> list[DependencyFinding]:
    return [
        item for item in findings
        if not item.required and item.label not in ORDERING_ONLY.get(item.program_id, frozenset())
    ]


def _shift_toward(objects, marker: str, toward: str, shift_m: float) -> dict[str, Any]:
    out = deepcopy(dict(objects))
    here, there = list(out[marker]["pos"]), out[toward]["pos"]
    dx, dy = there[0] - here[0], there[1] - here[1]
    norm = math.hypot(dx, dy) or 1.0
    here[0] += dx / norm * shift_m
    here[1] += dy / norm * shift_m
    out[marker] = {**out[marker], "pos": here}
    return out


def consequence_conflicts(objects: Mapping[str, Any], routine: Sequence[Mapping[str, Any]], fixtures: Sequence[Any],
                          variant: str, **params: Any) -> bool:
    """Whether the named wrong decision produces a new clearance violation downstream."""
    base = _keys(objects, routine, fixtures)
    if variant == "park_shift":
        park = next(step for step in routine if str(step.get("target", "")).startswith("park"))
        later = routine[routine.index(park) + 1]
        changed = _shift_toward(objects, park["target"], later["obj"], float(params.get("shift_m", 0.03)))
        return bool(_keys(changed, routine, fixtures) - base)
    if variant == "packing_shift":
        first = next(step for step in routine if step.get("type") in _RELEASES)
        changed = _shift_toward(objects, first["target"], str(params["toward"]), float(params.get("shift_m", 0.03)))
        return bool(_keys(changed, routine, fixtures) - base)
    if variant == "turn_held":
        body = str(params["body"])
        changed = deepcopy(dict(objects))
        changed[body] = {**changed[body], "yaw_deg": float(changed[body].get("yaw_deg") or 0.0) + 90.0}
        return bool(_keys(changed, routine, fixtures) - base)
    if variant == "too_tall":
        body = str(params["body"])
        changed = deepcopy(dict(objects))
        size = list(changed[body]["size"])
        size[2] = float(params["height_m"])
        changed[body] = {**changed[body], "size": size}
        return bool(_keys(changed, routine, fixtures) - base)
    raise ValueError(f"unknown consequence variant: {variant}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation_dependency_audit.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Phase 1 validation and commit**

Run: `python3 -m pytest tests/test_generation*.py tests/test_capacity_probe.py tests/test_program_bindings.py -q && python3 -B scripts/validate_fast.py`
Expected: PASS (xfail markers from Task 7 are allowed only with their recorded reason).

```bash
git add src/icgs/data/collection/generation/dependency_audit.py tests/test_generation_dependency_audit.py docs/plans/active/physical-dependency-generation.md
git commit -m "Add static skip and consequence audits for physical dependencies"
git push origin main
```

Update this plan's progress list (Phase 1 checked, evidence: command outputs and counts) in the same commit.

---

## Phase 2 — simulator integration (detailed at phase start)

Each task below lists its deliverable and acceptance test; the TDD steps with code are written into this plan in one commit before the phase starts.

- **Task 9: Builder fixtures and prism meshes.** `scripts/generation_build_tasks.py` builds fixture boxes (static respondable shapes; moving parts grouped into one dynamic compound), creates prismatic joints via `pyrep.backend._sim_cffi.lib.simCreateJoint(sim_joint_prismatic_subtype, sim_jointmode_force, 0, NULL, NULL, NULL)` with limits `[0, stroke]`, motor on, target velocity 0, force 3 N, sets `initial_opening`, adds a `<fixture>_grip` dummy on the fin, and creates movable bodies from `asset_families.prism_mesh`. Acceptance: pure unit test of a `build_plan(spec) -> list[dict]` instruction list; on `vps-a`, build all programs and a model-load check that every joint exists with the planned limits.
- **Task 10: Recorded state, cap and settling.** Worker: `articulations` in every scene state (joint position/velocity), `SceneGeometry` in the record, declared foreground list and wrist masks (`obs_config.wrist_camera.mask = True`, archived as `wrist_mask_frames`/`wrist_mask_frame_boundaries`), provisional cap 2048 (`interval_limit` valid failure), settle-until-resting (≤ 10 boundaries) replacing fixed settle holds and the 12 post-retreat holds. Acceptance: stubbed worker tests (existing loader pattern in `tests/test_generation_worker.py`); archive validator accepts masks.
- **Task 11: Goals, success and labels via relations.** Pure `step_goals.py` maps each routine step to its relation(s); success = RLBench conditions and all final/history relations for 5 boundaries; `near_threshold` flag; labels with `SceneGeometry`. Acceptance: unit tests over recorded-state fixtures.
- **Task 12: Articulation by contact.** Expert emits approach-above-fin, descend to `grip_point`, close without attach, joint-axis moves in ≤ 12 mm steps with closed loop on the measured joint (≤ 3 corrections), open, retreat; step goal `joint_open`/`joint_closed`. Acceptance: plan-level tests; `vps-a` spike reuse (open/close a built drawer and gate).
- **Task 13: Passage, holder yaw, grasp symmetry, modes, transit.** Low passage before wall planes; rotate to pocket yaw (≤ 2° final error) before release; grasp alignment by generator symmetry choosing the face pair whose open fingers and palm are clear of neighbours (the same rule as the clearance audit); drawer-content markers parented to the tray; `scripted_offset` via point (40 mm lateral, +30 mm transit, 16 mm steps); per-program `transit_z_m`. Acceptance: plan-level tests; clearance audit of both modes.
- **Task 14: Push flush-face and 5 mm re-push.** Tool yaw flush with the nearest cube face (≤ 45°); retry when error > 0.005 m; lateral check before gap entry. Acceptance: unit tests; push spot check on `vps-a`.
- **Task 15: Perturbations with method semantics.** Plan level (`perturbations.py`, `batch.py`): offset only the first step targeting `target_role`, marker never moved, yaw offset applied; grip timing early/late; displacement and insertion scheduled `after_step` at a stationary boundary (displacement prefers an already placed body). Runtime: apply after the step, record intervention boundary and pre/post state, flag the transition external. Acceptance: plan tests; stubbed runtime tests; displacement episode yields ρ=1, ν=0.

## Phase 3 — layouts and manifest (detailed at phase start)

- **Task 16: Layout schema and compiler.** `scene_layouts.json` gains `fixtures`, per-object `generator`/`width_m`, `containers`, `oriented_targets`, `transit_z_m`; compiler emits fixture-based routine steps; `prepare_attempt` mirrors and scales fixtures. Acceptance: compiler and prepare tests.
- **Task 17: Drawer programs** (T04, T05, T06, T08, T16, T19, V01, P1–P4, R3). Acceptance: pool clearance, skip and consequence audits green for these programs.
- **Task 18: Gate, aperture, holder and tray programs** (T07, T09, T10, T11, T12, V02, V04, G1–G4). Same acceptance.
- **Task 19: Blocker, corridor and remaining programs** (T01–T03, T13–T15, T17, T18, T20, V03, R1, R2, R4). Same acceptance; `ORDERING_ONLY` finalized with reasons; the two strict-xfail pool tests from Task 7 pass and lose their markers.
- **Task 20: Manifest regeneration.** New identifiers (dataset, manifest, composition, layout, semantics, predicate protocol, execution modes), per-row generators by split, `mirror: true`, execution modes; validator rejects cross-split generators; protocol freeze tests updated. Acceptance: manifest and binding tests; L0.

## Phase 4 — gates and documentation (detailed at phase start)

- **Task 21: Audit tool and gate G2.** `tests/regression/generation_program_audit.py` gains `--modes`, `--mirror`, `--skip-step` and per-family perturbed selection; run 144 coverage attempts plus 5 skipped-step attempts on `vps-a`; diagnose every failure; fix and re-run.
- **Task 22: Gate G3.** One attempt per perturbation kind per mechanism family; intervention records and ρ=1/ν=0 rows verified from archives.
- **Task 23: Caps, gate G4 and documentation.** Compute cap and horizon from G2 lengths, record in protocol and decision 0016; 36-program smoke through the canonical worker with a launcher-accepted receipt; experiment record, `docs/components/generation.md`, memory. Then stop for the owner's go-ahead.

## Validation and experiment strategy

- Tier L0 (`scripts/validate_fast.py`) and focused pytest after every task; simulator runs only on `vps-a` at a pushed commit.
- Gates G1–G4 and the launch-time yield check follow spec section 13; no count thresholds.
- Evidence (commands, commit, counts, outputs paths) is appended to this plan per phase.

## Compatibility

Dataset/cache: new identity; archives gain fields only. Config: manifest regenerated and re-approved in Task 20. Checkpoints and baseline: unaffected. Evaluation: benchmark horizon set in Task 23.

## Risks, open questions and recovery

- Palm checks may reveal contacts in current layouts during Phase 1 (tracked as Phase 3 inputs, not hidden).
- Bullet jitter on jointed trays; mitigated by brake force, limits and closed loop; measured in Tasks 12 and 21.
- Prism flats narrower than box faces; grasp stability checked per family in G2.
- Recovery: each phase is a separate set of commits on `main`; reverting a phase restores the previous collector without touching user files or archives.

## Final evidence

Filled in at completion.
