# Physical dependency scenes and method-aligned generation — design

Date: 2026-09-29. Status: design approved in conversation (direction, limit
changes, prism asset families); written spec awaiting owner review.
Workflow class: C (research/data migration). Follow-up artifacts: decision
record `docs/decisions/0016-physical-dependency-generation.md` and plan
`docs/plans/active/physical-dependency-generation.md`.

## 1. Goal

Regenerate the 36-program phase-1 dataset so that it serves the proposal
method, not only the collector's own success checks:

1. every program's declared constraint variable (proposal program catalog) is
   enforced by simulated physics, so skipping or mis-ordering an enabling step
   produces a physical conflict;
2. labels, perturbations, assets, randomization and recorded state follow the
   proposal's data semantics (views `geom`, `dyn`, `task`; later anchors and
   branches);
3. proposal numbers that do not fit a top-down Panda expert are replaced by
   measured, recorded values.

Full generation starts only after the gates in section 13 pass and the owner
gives an explicit go-ahead.

## 2. Evidence this design rests on

FACT (audit 2026-09-29, `docs/experiments/generation-validation/collector-semantics-20260929.md`):

- Containers (drawer, tray, holder, pad) are static slabs 2.5–4 cm tall; 8 of 12
  drawer programs have only a loose handle block; apertures are markers; gates
  are loose blocks; success is "within 1 cm of a marker".
- Static skip audit: removing any one earlier routine step breaks geometry in
  only 2 of 57 cases (T12 open gate, P4 retrieve C).
- Longest nominal episodes: P3 812, R2 790, P4 760, R3 736 recorded actions
  (64/60/57/56 s simulated); 7 of 12 test compositions exceed 512 actions.
- Push final errors at `9ef4140`: median 4.7 mm, 5 of 15 at 8 mm or more, maximum
  9.8 mm against the 10 mm tolerance.

FACT (throwaway spike on `vps-a`, `outputs/spike-articulation/`, not product code):

- `lib.simCreateJoint` (PyRep cffi) creates a prismatic joint; the task builder
  can save it inside a `.ttm` model.
- A tray on a brake-controlled prismatic joint (3 N), driven by the canonical
  action mode (`EndEffectorPoseViaIK` + `Discrete(attach_grasped_objects=False)`)
  with the default tool orientation, opened 0.1193 m of a 0.12 m command and
  closed back to 0.000 m. The fingers closed on a 12 mm fin gripped above the
  roof; nothing was attached to the gripper. A cube in the tray travelled with
  it, carried by the tray wall. With the drawer closed, a top-down reach to the
  cube stopped at tip z 0.835 m on the roof.
- Panda geometry relative to the tool tip, default orientation (finger closing
  axis = world y): palm x [-0.064, 0.020], y [-0.100, 0.104], z [+0.037, +0.142];
  open fingers y ±[0.040, 0.061]; closed fingers y ±0.021; fingertips 1.6 mm
  below the tip; finger top +0.042. The 20.5 cm palm width along the finger
  axis and the 3.7 cm palm height above the fingertips drive every clearance rule
  below.

## 3. Method alignment matrix

Every row is a requirement of the proposal or `docs/method/data-training.md`,
the design element that meets it, and the check that proves it.

| # | Method requirement | Design element | Verification |
| --- | --- | --- | --- |
| M1 | Cross-stage dependencies are physical; test families define delayed consequences (pack–add–close, grasp–transport–fit, park–retrieve–restore) | Fixture library and layout rules (sections 4–5) | Skip and consequence audits (G1); skipped-step simulator runs (G2) |
| M2 | Custom predicates: stable grasp/lift, stable placement, inside container (OBB, 2 mm), drawer closed (joint within 5 mm, 5 intervals), fit ≤1 cm/10°, restored blocker, full success = final and mandatory-history predicates for 5 intervals | Shared relation library (section 7) used by success, step goals and labels | Unit tests per relation; G2 |
| M3 | Task labels: occurrence ρ historical; current relation ν recomputed from state at t; eligibility ε = conjunction of prerequisites at t; free-space postconditions masked; ρ=1, ν=0 cases after displacement | Label materialisation from the relation library (section 7.3) | Unit tests; displacement episodes contain ρ=1, ν=0 rows |
| M4 | Perturbations: target offset ≤2 cm/10°, gripper closing one interval early/late, object shift ≤3 cm after the robot reaches a stationary state, blocker inserted into a free-space path, physics executed after the intervention, pre/post states and intervention ID stored | Perturbation semantics (section 8) | Unit tests; G3 |
| M5 | At least two distinct valid execution modes when the task permits | Two scripted modes differing in route and speed (section 6.6) | G2 covers both modes |
| M6 | Asset families split 70/15/15 by mesh family or generator before augmentation; test uses new mesh families | Parametric prism families (section 9) | Manifest validator rejects cross-split generators; pool audit |
| M7 | Randomization: translation in task regions, yaw, size 0.8–1.2, nuisance lighting/camera; balanced target position, favourable side and grasp accessibility | Keep 0.8–1.2; fit fixtures resize with the object; balanced scene mirroring (section 5.3) | Pool audit (G1); plan unit test for 50/50 mirror balance |
| M8 | Contact segments re-executed; free-space connectors from a planner; no teleportation | Collision-checked lift–transit–descend connectors (section 6.1) | Connector audit (G1); teleport detector (G2) |
| M9 | Generation stops at a fixed interval cap; timing measured | Interval = one recorded controller transition; cap from measured lengths, enforced by the worker (section 10) | G2 episode lengths; worker cap test |
| M10 | Snapshots and labels see articulation state | Joint position/velocity recorded every boundary (section 12) | Archive validator; label tests |
| M11 | Foreground segmentation is a perception assumption; policy `declared_objects_and_blockers` | Declared-foreground list including fixtures, and wrist masks recorded (section 12) | Archive validator |

Unchanged: controller protocol `rlbench-timed-ik-v2`, action space
`ee_pose_xyzw_grip_v1`, measured timing, lineage and split rules, 5-boundary
final hold, retries at most 2 per step, retained valid failures, quota rules.

## 4. Fixture library

Fixtures are built by `scripts/generation_build_tasks.py` from declarative
layout entries. Movable bodies stay primitive or mesh shapes as today. Numbers
are nominal; each is a design rule checked by the static audit, not a hand-tuned
constant.

### 4.1 Drawer (T04, T05, T06, T08, T16, T19, V01, P1–P4, R3)

- Housing (static, respondable): roof, two side walls, back wall; open front.
- Tray (dynamic compound: floor, four walls 25 mm, fin) on a prismatic joint
  parented to the housing; joint limits [0, stroke]; motor on, target velocity
  0, brake force 3 N (holds position against contact, yields to the arm).
- Fin: 12 mm thick along the pull axis, 60 mm wide, in front of the housing.
  At grip height the fingertips (tip − 1.6 mm) are at least 10 mm above the roof
  top, the fin top is between tip + 18 mm and tip + 32 mm (at least 20 mm of
  finger contact, palm bottom at tip + 37 mm clear of the fin). The default
  finger axis is the pull axis, so the fingers act on fin faces by contact,
  never by attachment.
- Interior: about 0.15 × 0.11 m (proposal 18–30 × 15–25 cm is replaced; section
  10.2); roof clearance = tallest drawer-program object at scale 1.2 + 8 mm;
  stroke = interior depth + 15 mm, so an open drawer exposes the whole interior.
- Physical meaning: closed roof blocks top-down placement or retrieval; contents
  travel with the tray; an object whose top exceeds the roof clearance
  (stacked, tilted) prevents closing.

### 4.2 Sliding gate (T12, V04, G3)

- Static wall 40–50 mm tall along the passage boundary with a gap.
- Gate panel of wall height on a prismatic joint sliding along the wall (world y
  so the default finger axis applies), offset 10 mm from the wall plane; fin on
  the panel top gripped above the wall top.
- Gap width: wide enough for the programmed passage in the default grasp
  orientation (gates test prerequisite and closure, not orientation).
- Physical meaning: a closed panel blocks the passage route at passage height.

### 4.3 Aperture (T10, G1, G2)

- Two static wall segments with a gap; wall height is chosen per program from
  its object height range (about 40 mm).
- Passage height: carried object bottom at least 5 mm above the table and at
  least 15 mm below the wall top; fingertips at least 5 mm above the wall top.
  Only the object's lower part crosses the wall plane, so the gap constrains
  the object cross-section, not the fingers or palm. Both conditions must hold
  at scales 0.8 and 1.2 (static audit).
- Gap = object cross-section across the travel direction in the correct
  orientation × attempt scale + 2 × clearance, clearance 5–10 mm per side; the
  wrong orientation must interfere by at least 5 mm.
- T10 pushes through the gap on the table; lateral push error must stay below
  the clearance.
- Physical meaning: grasp or push orientation and offset decide passage.

### 4.4 Holder and tray

- Holder (T09, V02, G1, G2, G4): floor plate and 25 mm walls forming a pocket of
  the object's footprint family × attempt scale + 4 mm per side. A square box
  fits only within about ±8–12° of the aligned yaw across scales 1.2–0.8
  (elongated boxes about ±6°), in line with the 10° fit predicate.
- Tray (T07): same construction; interior holds A and B side by side with
  about 15 mm total slack, so A's placement decides whether B still fits.
- Pad (T02, T11): flat support, unchanged.
- Walls stay below the open fingertips at release (fingertips ≥ half object
  height + 22 mm above the floor), so releases inside pockets are clear.

### 4.5 Adjacency blocking (T03, T13, T15, T16, T18, T20, V03, R1–R4, P4)

- A blocker (or blocking object) sits on the target object's grasp side along
  the finger axis at a centre distance that never interpenetrates at scale 1.2
  and always intersects the open finger span at scale 0.8 (for the current sizes
  about 70–75 mm).
- T03: the reach point lies on the table inside the blocker footprint; reaching
  it is impossible until the blocker is pushed away.
- T18: object A occupies object B's finger envelope (the shared corridor).
- P4: object C occupies target B (existing).
- Parking bays are placed so that the nominal park pose is clear, while a
  pose 30 mm towards the later retrieval or restore envelope obstructs it
  (delayed consequence for park–retrieve–restore anchors).

### 4.6 Regrasp necessity (T17, G4)

The object starts against a static corner so that the only collision-free
initial grasp closes across the axis that is incompatible with the final fit or
aperture. The temporary place is in the open; the regrasp uses the other axis.

### 4.7 Ordering-only steps

Steps whose necessity is definitional rather than physical are listed in the
audit allow-list with the reason: T14 restore after park (restoration task),
T01/T02/T09/T11 single-object sequences, and final close steps whose
consequence is the closed-state predicate.

## 5. Layouts

### 5.1 Rules

Per program, layouts satisfy all fixture rules above, keep every nominal expert
path collision-free for both execution modes, and satisfy the delayed-consequence
rule of their family (section 11.2). Layout coordinates stay in
`scene_layouts.json`; fixtures are layout entries with a `fixture` type and
parameters.

### 5.2 Scaling

Movable bodies keep size randomization 0.8–1.2. Holder pockets, tray
interiors and aperture gaps resize in x/y with the attempt scale so clearances
stay in band. Drawers and gates keep fixed dimensions sized for scale 1.2.

### 5.3 Balanced mirroring

Each attempt draws a scene mirror flag (y → −y for every body, target and
fixture; fixtures are rotated 180° about z so pull and slide directions flip),
stratified 50/50 per program and split. Target position, favourable side and
initial grasp accessibility therefore alternate across attempts (proposal
"balanced so that one task family is not always placed on the same side").

## 6. Scripted expert

### 6.1 Connectors

Free-space motion is lift → transit → descend. Transit height is computed per
connector: the highest fixture or body under the swept path plus the carried
extent below the tip plus 15 mm, never below 0.90 m. The static audit checks
every connector against fixtures, bodies and the palm box.

### 6.2 Articulation by contact

Approach above the fin at transit height, descend to grip height, close
(`Discrete`, no attach), move along the joint axis in at most 12 mm steps,
closed loop on the measured joint position until within 5 mm of the goal (at
most 3 corrective moves), open, retreat. The step goal is the joint relation.

### 6.3 Grasps

Face-aligned by family symmetry (box 90°, hexagonal 60°, octagonal 45°,
stadium 180°); elongated members close across the short axis. Where a later
aperture or holder constrains the held object's orientation, the expert plans
the grasp face pair and any in-transit rotation so the orientation is compatible
when it arrives (G1, T09): the orientation choice has a delayed consequence.
Programs with explicit rotate or regrasp steps (T11, V02, G2, T17, G4) are laid
out so that the orientation reachable without that step is incompatible,
making the step physically necessary.

### 6.4 Placement into holders and low passages

Before descending into a holder, rotate the held object to the pocket yaw
(symmetry-aware, 5° increments, final error at most 2°). Before crossing an
aperture or gate, descend to passage height outside the wall plane, cross, then
rise.

### 6.5 Pushes

Rotate the tool so the finger face is flush with the nearest cube face (at most
45°). Push along the goal direction with the existing closed-loop completion.
Re-push when the step error exceeds 5 mm (at most 2 retries). The success
tolerance stays 10 mm. For pushes through a gap, lateral error must be below the
clearance before the object reaches the wall plane, otherwise the push re-aims.

### 6.6 Execution modes

- `scripted_direct`: approach from above, 12 mm steps.
- `scripted_offset`: approach through a lateral via point 40 mm to the side
  (side alternates per attempt), transit 30 mm higher, 16 mm steps.

Both modes are valid for every program and balanced per program. The mode is
recorded per episode.

### 6.7 Holds

Replace fixed settle holds with settle-until-stable: after release, hold until
the placed body moves slower than 1 cm/s for 5 boundaries (at most 10). Remove
the 12 post-retreat holds. Grip actuation holds stay as measured.

## 7. Relations, success and labels

### 7.1 Relation library

One simulator-free module evaluates relations on recorded state (bodies,
articulations, robot):

| Relation | Definition |
| --- | --- |
| `grasped_stable(o)` | fingers closed on o (not all open amounts > 0.9), o lifted ≥ 3 cm above its support, o-to-gripper drift < 1 cm and 10° over 5 boundaries |
| `placed_stable(o)` | released, support contact within 3 mm, speed < 1 cm/s over 5 boundaries |
| `inside(o, c)` | oriented bounding box of o inside c's inner volume with a 2 mm margin |
| `at_pose(o, t)` | position ≤ 1 cm and yaw ≤ 10° modulo the family symmetry |
| `joint_closed(j)` / `joint_open(j)` | within 5 mm of the closed limit / at least the required opening |
| `passage(o, a)` | history: o crossed the aperture or gate plane inside the gap below the wall top |
| `access_clear(o)` | no declared body inside o's open-finger envelope |
| `restored(b)` | b satisfies its restore relation under `at_pose` |

### 7.2 Success and step goals

Each routine step's goal is its relation (for example `open` → `joint_open`,
`place into drawer` → `inside` and `placed_stable`). Success = RLBench
conditions and every final and mandatory-history relation holding for 5
boundaries. Unsafe terminal (object leaves the workspace, controller failure)
fails the episode. A final error above 0.8 × tolerance is flagged
`near_threshold` and reported separately.

### 7.3 Task labels

- ν(t) = relation at t from recorded state, using the same library and
  tolerances as success (the current 4 cm label tolerance is removed).
- ρ(t) = ρ(t−1) OR ν(t).
- ε(t) = conjunction of the step's prerequisite relations evaluated at t
  (current, not historical): for example "place A into drawer" requires
  `joint_open(drawer)` now, plus `grasped_stable(A)` for a held continuation or
  `access_clear(A)` for a pick-and-place.
- Free-space postconditions and relations without the required state stay
  masked with a reason.

## 8. Perturbations

| Kind | Semantics |
| --- | --- |
| `action_pose_offset` | offset the commanded target pose of one step by up to 2 cm and 10° yaw; the scene target never moves; retries may correct |
| `gripper_timing` | the close (or open) command is issued one interval before the final approach waypoint completes (−1) or one interval after arrival (+1) |
| `object_displacement` | at a stationary boundary (robot at rest after a completed step), shift a stationary body by up to 3 cm, including an already placed body (creates ρ=1, ν=0) |
| `blocker_insertion` | at a stationary boundary, insert a declared blocker on the upcoming free-space or contact path at a non-interpenetrating spot |
| `pause_hold` | unchanged (executed hold of 2–10 intervals) |

Displacement and insertion are external interventions: the archive records
the intervention ID, boundary, and pre/post state, and flags the transition as
external so that the `dyn` view never treats it as the effect of a command. The
expert re-plans each step from measured poses, so recovery is executed physics.

## 9. Asset families

Parametric prism generators with flat parallel faces, split before augmentation:

| Split | Generators |
| --- | --- |
| train (5) | cube box, long box, flat box, tall box, stadium prism (two flats, rounded ends) |
| development (1) | hexagonal prism (regular and elongated members) |
| test (1) | octagonal prism (regular and elongated members) |

The achieved ratio is 5/1/1 generators (71/14/14 %). Blockers and spacers use
their split's generators. Each episode records generator ID and parameters. The
manifest lists allowed generators per row, and the validator rejects
cross-split use. Meshes are convex prisms (`Shape.create_mesh`), respondable,
mass scaled with volume. Holders and apertures follow the generator footprint.

## 10. Limits and protocol values

### 10.1 Interval and caps

- One interval = one recorded controller transition with measured duration.
- Generation cap = max(512, round up to a multiple of 128 of 1.5 × the longest
  nominal expert episode in gate G2). The worker enforces it
  (`interval_limit` valid failure).
- Benchmark horizon H = max(512, round up to a multiple of 128 of 2 × the
  longest nominal test-composition expert episode).
- Both values are recorded in the protocol and the decision record before
  full generation.

### 10.2 Changed or kept limits

| Proposal limit | Decision |
| --- | --- |
| Drawer inner size 18–30 × 15–25 cm | about 15 × 11 cm (reach, table space, program layouts) |
| Aperture clearance 5–30 mm over the object cross-section | 5–10 mm per side over the passing cross-section; wrong orientation interferes ≥ 5 mm |
| Generation stop at 512 intervals; horizon H512 | section 10.1 |
| Size 0.8–1.2 | kept; fit fixtures resize with it |
| Drawer closed, inside, fit, stable grasp/placement, restored, 5-interval hold, near-threshold reporting | kept, implemented as section 7 |
| Force/penetration unsafe limits (40 N, 2 mm) | unchanged and out of scope (feasibility gate) |

### 10.3 Identifiers

New dataset identity and protocol bumps: dataset version, program manifest
version, composition protocol, layout version, program semantics version,
predicate protocol (`physical-relations-history-hold-v3`) and execution mode
IDs. The approved manifest is regenerated and re-approved. The earlier smoke
receipts and the unused HF prefix are retired.

## 11. Static verification (no simulator)

### 11.1 Clearance model

Extend `layout_clearance` with fixture solids (roof, walls, panels, pockets),
the palm box and finger boxes at the measured offsets, connectors at transit
height, passage heights, articulation sweeps, and releases inside pockets. It
runs over every planned attempt, each with its planned mode and mirror side.

### 11.2 Dependency audits (regression tests)

- Skip audit: removing any declared enabling step produces a clearance
  violation in a later step, except the section 4.7 allow-list.
- Consequence audit, on each program's nominal layout at scales 0.8 and 1.2:
  a representative wrong decision produces a downstream conflict. Cases: A placed 30 mm towards the tray centre (P1–P3, T07, T08,
  T19), park pose shifted 30 mm (R1–R4, T13, T15, T16, T20, V03), held-object
  yaw rotated 90° at the aperture or holder (G1, G2, G4, T09, T11, T17, V02),
  drawer object top above the roof clearance (drawer programs).

## 12. Recorded data additions

- `articulations` in every scene state: joint name, position, velocity, limits.
- Declared foreground: every body (objects, blockers, spacers, fixture parts)
  with simulator handles; wrist masks captured so the
  `declared_objects_and_blockers` foreground policy can be applied exactly.
- Intervention records with boundary and pre/post state; external-transition
  flag.
- Execution mode, mirror flag, asset generator ID and parameters, fixture
  parameters.
- Episode schema version bump for these fields; archive writer and validator
  updated together.

## 13. Acceptance gates before full generation

Each gate protects one thing nothing else checks. Gates find defects; they do
not estimate yield (section 13.1).

| Gate | Runs | Pass |
| --- | --- | --- |
| G1 static | unit tests, L0, section 11 audits | all pass |
| G2 simulator coverage | one nominal attempt per program × execution mode × mirror side (144), plus one attempt per mechanism family (drawer, gate, aperture, holder, adjacency blocker) with its enabling step skipped | 0 crashes, invalid observations, teleports or runaway episodes (provisional cap 2048); every failure diagnosed from its trace: expert or scene defects are fixed and the program re-run, genuine physical variation is accepted; every skipped-step attempt fails physically (the static model cannot tell whether fingers shove a movable blocker aside) |
| G3 perturbations | one attempt per perturbation kind on one program per mechanism family | 0 crashes or invalid observations; intervention boundary and pre/post state recorded; the displacement run yields ρ=1, ν=0 rows |
| G4 archive smoke | canonical worker and archive writer, one nominal attempt per program | launcher accepts the receipt; archive validator accepts the new fields |

The final generation cap and horizon (section 10.1) are computed from the G2
episode lengths.

### 13.1 Launch-time yield check

After roughly the first 10 attempts per program, the operator reads the run
report. Any program below about 70% success is paused and diagnosed before it
consumes its attempt cap. The run is resumable, so this needs no new tooling.

## 14. Phases

1. Decision record, protocol values, relation library, label semantics,
   perturbation semantics, push and hold changes (usable on current scenes).
2. Fixture builder (drawer, gate, aperture, holder, tray), articulation and
   passage primitives, recorded articulations and masks.
3. Prism asset families, manifest schema, validator.
4. Layouts for 36 programs with mirroring; static audits green.
5. Simulator gates G2–G4; documentation; owner go-ahead for full generation.

Each phase is committed on `main` with its validation evidence. Simulator runs
use `~/ICGS` on `vps-a` at the pushed commit.

## 15. Risks

- Jointed trays and panels under Bullet: jitter or drift. Mitigation: brake
  force, joint limits, closed-loop articulation, audit.
- Longer episodes (transit heights, modes). Mitigation: holds trimmed; cap from
  measurement.
- Hexagonal and octagonal flats are narrower than box faces; grasp stability is
  verified per family in G2.
- The uncommitted training-export reader in the working tree (another agent's)
  assumes the current schema; it will need updating and is not touched here.
- Compute: two modes and longer episodes raise generation time; measured in G2
  before launch.

## 16. Out of scope

Phase-2 anchors, branches and snapshots (this design only provides the physical
consequences and recorded state they need), force/penetration thresholds, real
robots, external datasets and any training run.
