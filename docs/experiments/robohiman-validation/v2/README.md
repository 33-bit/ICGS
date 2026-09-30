# RoboHiMan gates, round 2 (2026-09-30): monitors, replay fidelity, dependency

Follows [round 1](../README.md). Host and simulator environment unchanged
(`vps-a`, RoboHiMan `33f71d30`, PyRep `231a1ac6`, RLBench `587a6a0e`,
CoppeliaSim 4.1.0, Python 3.11). No large data was collected; no D_value or
D_pair view exists (the view builder refuses them without a pi_ref identity).

Code revisions: monitor episodes `a3098f1` (labels re-derived offline with
monitor v3 at `f288bdf`); tuning fidelity `e25bb81` (open_drawer `26aa521`);
held-out fidelity `3c63eae`; drawer dependency `26aa521`; place dependency
`17fba12`.

## Gate 1 — task-monitor coverage: PASS (10 tasks, 3 semantic families)

| Family | Tasks with reviewed monitors | Validated episodes |
| --- | --- | --- |
| articulated_drawer | open_drawer, close_drawer, put_in_without_close, put_in_and_close, put_two_in_same | round 1 (9) + put_two_in_same |
| container_pick_place (new) | box_in_cupboard, box_exchange | 1 + 2 (one induced failure) |
| dustpan_tool_use (new) | sweep_to_dustpan, sweep_and_drop, rubbish_in_dustpan* | 1 + 2 (one induced failure) |

\*`rubbish_in_dustpan` has a monitor but strategy 0 is disabled upstream, so no
A-level episode exists (see quirks).

**Generic layer.** Predicate kinds built only from upstream condition classes
on live handles: `detected`, `grasped`, `grasped_any`, `detected_count_ge`,
`drawer_open/closed`. On top of that: probes of the task's own registered
success conditions, name resolution from the task instance (drawer option,
grocery), conjunctive event relations (`in_region&!grasped`), and
alpha/rho/nu/epsilon derivation (`icgs.data.stage1.labels`).

**Task-specific layer** (`monitors.TASK_SPECS`). Which objects, sensors and
joints each task uses; the source of each threshold; and the event list
(relation, prerequisites, current requirements, count-only-when-eligible).

**Runtime traces** (`stage1-report-monitor.json`, labels re-derived offline
from the stored raw traces with monitor v3):

- box_exchange success: sugar grasped 141 → placed (detected & released) 300
  → spam grasped 432 → placed 562. Upstream success already fires at 544,
  while the spam is still held.
- box_exchange failure (early close at the sugar grasp): sugar is never grasped.
  alpha stays on `sugar_grasped` for the whole episode, while the spam events
  still occur.
- box_in_cupboard: grasped 81 → placed 208 (the upstream sensor fires at 186,
  while the object is still held).
- sweep_to_dustpan: broom 134 → all 5 dirt particles in the dustpan at 244.
- sweep_and_drop success: broom 128 → dirt 232 → rubbish grasped 456 → in
  dustpan 535. The expert does the broom first; the oracle language lists the
  rubbish first. Event order follows execution.
- sweep_and_drop failure (early close on the broom): the broom never leaves its
  holder and the dirt is never swept; the rubbish events still occur
  (494 / 590).
- put_two_in_same: drawer open 165 → first block 409 → second block 606.
  Events are order-invariant counts.

**Manual check.** Frame sheets `mon-*-sheet.png` agree with every event
boundary listed above.

**Two corrections came out of this check:**
- "Placed" events now require release. Upstream sensors detect objects that
  are still being carried.
- alpha (first pending eligible event in spec order) is ambiguous for
  independent subtasks. In the sweep_and_drop failure, alpha points at the
  broom while the rubbish is handled. This is recorded as a limitation, not
  papered over.

## Gate 2 — snapshot/replay fidelity: PARTIAL

**Diagnosis.**

1. Upstream resets can leave engine-internal contact state. With one reset
   lineage, a non-target top drawer creeps from 0.5 mm to 15.9 mm in 3 s.
2. CoppeliaSim 4.1 cannot save that state. Configuration-tree set, dynamics
   reset and initial-velocity parameters all yield a non-creeping drawer
   (every variant: max diff 15.3–15.4 mm).
3. Round 1's long-horizon drift was mostly this hidden start state.
4. At a physical-grip anchor (gripper closed on a drawer handle, no attachment)
   a restore loses the grip. The state itself is restored exactly (≤4e-6 m),
   but the outcome flips.

**Change: canonical start.** Restore the post-reset snapshot before execution.
This is an opt-in deviation recorded in `environment.canonical_start`.

Before/after, full recorded history replayed from S0 (put_in_and_close):

| | Drawer drift |
| --- | --- |
| Without canonical start | 61 mm, predicates 3–5% differ |
| With canonical start | ≤1.8 mm, predicate disagreement ≤0.2%, outcome equal |

**Experiment** (`robohiman_fidelity.py`).
- One reference expert run; an anchor after every waypoint (up to 8–11 per task).
- 4 reproduction strategies × 3 repeats; each repeat replays the recorded
  commands open loop to the end.
- Metrics: EE, joints, objects, articulations, raw predicates, grasp set, depth
  at the anchor and at the end, final outcome, and each strategy's own
  repeat-to-repeat floor.

**Proposed acceptance rule for branch anchors** (`robohiman_fidelity_summary.py`).
Proposed *after* the tuning runs, then checked on held-out tasks and seeds.

- **At the anchor:** EE ≤1.5 mm, arm joints ≤2e-3 rad, objects and
  articulations ≤2 mm, depth mean ≤2 mm, raw predicates and grasp set identical.
- **Over the replayed continuation:** all outcomes equal, predicate
  disagreement ≤1%, final object/articulation drift ≤ max(10 mm, 2× the
  repeat floor).
- Sweeping dirt particles are excluded from the object terms (repeats already
  differ by 1–3 cm); their effect enters through the count predicates.

**Accepted anchors / tested anchors** under the recommended strategy per anchor class:

| Anchor class | Recommended strategy | Tuning (5 tasks) | Held-out (4 runs, new seeds) | Direct restore instead |
| --- | --- | --- | --- | --- |
| free (gripper open) | direct anchor restore | 20/21 | 13/16 | — |
| kinematic grasp (object attached) | direct anchor restore | 13/13 | 13/14 | — |
| physical grip (handle) | nearest contact-free snapshot + replay | 2/2 | 1/2 | 0/4 (outcome flips every time) |
| **all** | | **35/36** | **27/32** | |

- The held-out physical-grip failure (put_two_in_same): the drawer is 3.7 mm off
  at the anchor, the final block 12.7 mm off. Predicates are identical and the
  outcome is equal.
- The remaining rejections are dropped objects' final resting pose (repeat
  floors themselves reach 13–26 mm) and 2–9 mm anchor offsets after long
  full-history replays.
- Files: `fidelity-*.json`, `heldout-fidelity-*.json`, `*-summary.json`.

**Why PARTIAL:**
- the rule was set after seeing the data;
- held-out acceptance is 84%, not ~100%;
- there are only 4 physical-grip anchors;
- no strategy is exact.

Branch collection therefore requires, per anchor, a 3-repeat acceptance check
before use and "approximate" labelling. Exact interventions are never claimed.

## Gate 3 — dependency audit: PARTIAL (consequential decisions exist but are sparse)

**Setup.**
- Canonical start → prefix replayed to the anchor → candidate applied at the
  decision waypoint → the scripted-expert mirror continues with 5 mm Gaussian
  jitter on every later waypoint. This is **not pi_ref**.
- 20 trials per candidate.
- Pre-declared rule: both candidates locally successful in ≥90% of trials,
  downstream gap ≥0.30, Fisher exact p<0.05.

| Decision | Task | Local outcome | Downstream \| local (Wilson 95%) | Consequential pairs |
| --- | --- | --- | --- | --- |
| drawer pull extent (0.151 / 0.152–0.158 / 0.155–0.161 / 0.161 / 0.17 m) | put_in_without_close | drawer_open 20/20 each | 10/20 [0.30,0.70], 18/20, 19/20, 20/20, 20/20 | 4 pairs vs 0.151 m (gap 0.40–0.50, p 0.0004–0.014) |
| same | put_in_and_close | 20/20 each | 19/20, 19/20, 19/20, 20/20, 20/20 | none |
| place pose ±3 cm (waypoint frame) | put_in_and_close | item_in_drawer 20/20 | 20/20 all | none |
| place pose ±3 cm | put_two_in_same | first block in drawer 20/20 | 18–20/20 | none |
| place pose ±5 cm | box_exchange | sugar_on_table 20/20 | 17–20/20 | none |

**Interpretation.**
- One real `local_success(a1)=local_success(a2)`, `downstream(a1)≠downstream(a2)`
  case exists. It sits in a ~1 cm band just above the benchmark's own 0.15 m
  drawer-open threshold, and it depends on the sampled scene: the structurally
  identical put_in_and_close setting does not reproduce it.
- Mechanism: the fixed, mark-relative place waypoint becomes unreachable (or
  the item misses) when the drawer is barely open. That is a property of the
  scripted expert's planner; a learned pi_ref may differ.
- Place-pose decisions of ±3–5 cm produced no significant consequence in any
  of the three compositional tasks.
- RoboHiMan alone does not currently supply enough clean consequential
  decisions for the main ICGS evidence.

## Storage cadence

Measured compressed bytes (128² × 4 cameras):
- ≈0.3 KB per physics step for all low-dimensional state and commands;
- ≈178 KB per rendered frame (depth float32 ≈95 KB, RGB ≈82 KB).

Depth is not exactly 24-bit quantized, and a 24-bit integer encoding saves only
3%, so float32 stays.

| Store at every physics step (0.05 s) | Store at model cadence (every 2nd step = 0.1 s) + every event boundary |
| --- | --- |
| commands (arm targets/velocities, finger velocities, grasp/release, phase, waypoint, teleport count); sim/wall time; arm joints, velocities, forces (+validity), targets; tip pose; finger joints and open amount; grasp set; object poses and velocities; task-joint positions; raw predicates; task success | RGB, depth, per-frame intrinsics/extrinsics/near/far, frame gripper pose/open, task low-dim state |

**Why per step:** open-loop replay needs every command row; A1 needs command
vs achieved at physics resolution; and event boundaries and grasp attaches
happen between frames.

**Measured effect:** stride 2 + event frames = 0.52 frames per boundary →
≈95 KB/step (58 MB for a 619-step episode vs 108 MB at stride 1).
`robohiman_collect.py` now defaults to `--frame-stride 2`.

## Upstream quirks preserved and recorded per episode

`environment.known_upstream_quirks_preserved` (from `pins.quirks_for`):
- kinematic grasp, with the attach applied after the last step of a waypoint;
- `path.visualize` arm excursions;
- reset residuals and RRTConnect sensitivity;
- factor RNGs;
- engine-internal state left by reset;
- `np.random.seed(None)` in native workers;
- the misnamed `object_size` factor enabled in `no_variations` (13 tasks);
- `rubbish_in_dustpan` strategy 0 disabled (native `train_A` collects nothing
  for it);
- sweep_and_drop execution order vs its language;
- success sensors that fire while the object is still held (box_exchange,
  box_in_cupboard).
