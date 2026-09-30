# Pre-flight and split lock for Stage-1 (2026-09-30)

Environment: unchanged (`vps-a`; RoboHiMan `33f71d30`, PyRep `231a1ac6`,
RLBench `587a6a0e`, CoppeliaSim 4.1.0, Python 3.11). Code: audit run at
`06efd14`; the timing-check fix was re-run at `d3c8db3` for 5 tasks (see
"Compatibility"). Split locked by `scripts/robohiman_split_lock.py`, which is
deterministic: a re-run reproduces the same lock and overlap file byte for byte.

No Stage-1 dataset was generated, no model was trained, and pi_ref and Stage 2
were not started. Pre-flight used at most 3 episodes per task (44 in total).

## 1. Compatibility — 21 PASS, 1 PARTIAL, 0 FAIL, 0 UNSUPPORTED

**Command.** `scripts/robohiman_compat.py` per task, 7 in parallel
(`compat-reports/`, `compat-summary.json`).

**Per-task checks:**
- reset;
- expert success (a second seed only if the first fails);
- action logging (arm joint targets, finger velocities, grasp events);
- achieved-state logging (finite, command ≠ achieved, step interval = 0.05 s
  within float32 clock spacing);
- four calibrated cameras;
- numpy depth→world vs PyRep;
- graspable mask-box containment;
- success-condition conjunction vs `task.success()`;
- every monitor event observed in the successful episode, in spec order;
- one early-grasp-close episode stored (failure retention);
- provenance (reset RNG, factor RNG states, code/RoboHiMan revisions, TTM
  hash, quirks).

| Task | Level | Verdict | Steps | Failure-retention episode |
| --- | --- | --- | --- | --- |
| open_drawer | A | PASS | 199 | failure |
| close_drawer | A | PASS | 195 | success (early close did not break the push; noted) |
| put_in_opened_drawer | A | PASS | 142 | failure |
| take_out_of_opened_drawer | A | PASS | 210 | failure |
| box_out_of_opened_drawer | A | PASS | 211 | invalid_execution |
| box_in_cupboard | A | PASS | 366 | failure |
| box_out_of_cupboard | A | PASS | 273 | failure |
| broom_out_of_cupboard | A | PASS | 334 | failure |
| sweep_to_dustpan | A | PASS | 262 | failure |
| rubbish_in_dustpan | AP only | **PARTIAL** | 130 | failure |
| put_in_without_close | C | PASS | 420 | failure |
| sweep_and_drop | C | PASS | 578 | failure |
| take_out_without_close | C | PASS | 438 | invalid_execution |
| transfer_box | C | PASS | 514 | invalid_execution |
| put_in_and_close | C | PASS | 604 | failure |
| take_out_and_close | C | PASS | 628 | invalid_execution |
| put_two_in_same | C | PASS | 579 | failure |
| take_two_out_of_same | C | PASS | 636 | invalid_execution |
| put_two_in_different | C | PASS | 924 | failure |
| take_two_out_of_different | C | PASS | 975 | invalid_execution |
| box_exchange | C | PASS | 616 | failure |
| retrieve_and_sweep | C | PASS | 388 | failure |

**Geometry and predicates on all 22 tasks:**
- numpy vs PyRep back-projection: 0.0 m max difference;
- success-condition conjunction equals `task.success()` at every boundary;
- step interval equals 0.05 s within the float32 clock spacing.

**Mask containment** is 1.0 on the 11 tasks where a graspable is visible in
the final frame; elsewhere the object ends inside the cupboard or dustpan.

**The one PARTIAL:** `rubbish_in_dustpan` is PARTIAL only because upstream
disables its strategy 0; it was audited on AP strategy 2.

**Timing tolerance.** Five long tasks were first flagged by a fixed 1e-6 s
timing tolerance. The deviation (3.05e-6 s) is below CoppeliaSim's float32
clock spacing at t≈48 s (3.8e-6 s). The check was made float32-aware and
those five tasks were re-run.

## 2. Monitor coverage — all 22 tasks, 3 semantic families

**Generic layer** (`monitors.py`, `labels.py`):
- predicate kinds `detected`, `grasped`, `grasped_any`, `detected_count_ge`,
  `released_count_ge`, `drawer_open`, `drawer_closed`, all built from upstream
  condition classes;
- probes of each task's registered success conditions;
- name binding (drawer option, drawer pair from `index_comb`, grocery);
- the reusable `_pick_place` event pattern ("placed" = at goal *and* released);
- conjunctive relations and alpha/rho/nu/epsilon derivation.

**Task-specific layer.** `TASK_SPECS` entries: which objects, sensors and
joints a task uses, plus its event list.

**Validation.** `label-timelines.json` covers 44 episodes.
- In every successful episode all events occur, in spec order.
- Failure episodes leave alpha on the first missed event.
- Two-drawer tasks: the expert closes drawer 0 before opening drawer 1, which
  confirms the `drawer0_closed` event (`put_two_in_different`: 180 → 420 →
  599 → 697 → 940).
- Manual frame checks: `preflight-take_two_out_of_different-…-sheet.png`,
  `preflight-transfer_box-…-sheet.png`, plus the round-2 sheets.

**Unsupported or limited semantics:**
- alpha is ambiguous for mutually independent subtasks (it points at the first
  pending event in spec order);
- a "handle grasped" physical-grip predicate does not exist (there is no
  attachment; gripper closure alone is not a relation);
- sweeping is represented only by the dirt-in-dustpan count;
- `rubbish_in_dustpan` has no A-level episodes.

## 3. Known upstream quirks (recorded per episode and per task in the manifest)

- Kinematic grasp; the attach is applied after the last physics step of a
  waypoint.
- `path.visualize()` arm excursions.
- Reset residuals and RRTConnect sensitivity; reset can leave engine-internal
  contact state.
- Colosseum factor RNGs advance every reset.
- Native workers call `np.random.seed(None)`.
- Success sensors fire while the object is still held: success precedes
  release by 4–14 steps in 9 tasks.
- The float32 simulation clock.
- The `camera_pose` factor resolves 2 of its 3 targets (the front camera
  still moves up to ~10 cm).
- A misnamed `object_size` factor is always enabled in 13 tasks.
- `rubbish_in_dustpan` strategy 0 is disabled, so native `train_A` collects
  nothing for it.
- `sweep_and_drop`'s language order differs from its execution order.
- Two-drawer tasks do not require closing the first drawer, but the expert
  closes it.

## 4. Overlap matrix (`split-overlap.json`)

Each cell is the fraction of the evaluation group's items that also appear in
TRAIN. Assets are compared by geometry signature (`asset-inventory/`): mesh
vertex count plus scale-invariant extent ratios. `item`, `item0` and `item1`
are one block asset.

| Factor | TRAIN vs DEV-composition | TRAIN vs TEST-held-out-composition | TRAIN vs TEST-dependency-stress |
| --- | --- | --- | --- |
| exact task | 0.00 | 0.00 | 0.00 |
| primitive skill (action + place) | 1.00 | 1.00 | 1.00 |
| step type | 1.00 | 1.00 | 1.00 |
| step transition (bigram) | 0.40 | 0.44 | 0.50 |
| full structure sequence | 0.00 | 0.00 | 0.00 |
| scene file (TTM hash) | 0.00 | 0.00 | 0.00 |
| manipulated-object geometry | 1.00 | 1.00 | 1.00 |

- DEV-composition vs TEST-held-out-composition: 0% shared tasks and
  sequences, 11% shared transitions (`CLOSE>OPEN`).
- Every held-out task introduces at least one transition unseen in TRAIN:
  `PLACE:drawer_in>CLOSE`, `CLOSE>OPEN`, `PLACE:drawer_in>PICK:surface`,
  `PLACE:table>PICK:table`, `PICK_TOOL:cupboard>SWEEP:dustpan`.
- **Perturbation families.** TRAIN uses every enabled Colosseum strategy
  except `distractor`, `background_texture` and `all_mixed`, which are
  reserved for `TEST-unseen-perturbation`.
- **Lineage.** Disjoint numpy seed ranges and distinct factor seeds per split
  (TRAIN 42, DEV 4242, TEST 244); TEST inference contexts are drawn only from
  seeds 390,000,000–399,999,999.

**Claims each track supports:**
- **Known-task configuration robustness:** `TEST-config`.
- **Unseen perturbation family:** `TEST-unseen-perturbation`.
- **Held-out composition** (new task structure and transitions from seen
  primitives and assets): `TEST-held-out-composition`.
- **Dependency-related stress test:** `TEST-dependency-stress` (prerequisite
  chains).
- **Not supported:** asset generalization, unseen primitives, and
  dependency-mechanism generalization.

## 5. Frozen split

- `artifacts/robohiman/icgs_robohiman_stage1_split_v1.json`
- SHA256 `2d1e7dcfd300480854d9d35e1c7abbf196863790be7ff6fcaa65317a73b032ef`
- Lock: `…split_v1.lock`
- Decision: [ADR 0017](../../../decisions/0017-icgs-stage1-split.md)
