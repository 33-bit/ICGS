# Generation collector physical-semantics audit and repair

Status: complete — collector gates PASS at `9ef4140`; awaiting owner acceptance.
Owner and date: Claude agent `48ed9964` for the repository owner, 2026-09-28/29.
Category B/C collector correction. Full generation remains stopped until the
owner accepts this record and the limits listed at the end.

| Research contract | Definition or authoritative reference |
| --- | --- |
| Baseline | Collector at `923ab486ce5a3cdc0635f346a42d7632859d57af` (placement/release fixes of 2026-09-28) |
| Hypothesis | The 36 programs can be demonstrated without teleports, frozen bodies, dropped carries or inferred timing, with success labels that require every program step |
| Changed components | Worker execution/labels, compiler routines, expert push/held continuations, 14 scene layouts, blocker insertion, task-model marker rule |
| Unchanged components | Approved manifest, program/step catalog, seeds and planner order, randomization bounds, 0.01 m predicate tolerance, archive format, controller action mode, observation config |
| Dataset | None published; diagnostic runs only |
| Evaluation tasks | All 36 programs (T01–T20, V01–V04, P1–P4, G1–G4, R1–R4) |
| Metrics | Outcome, per-step goals/retries, teleports, measured physics steps, final predicate distances |
| Seeds | Production planner nominal indices 0–2 per program (collection seed 20260920) |
| Configuration | `tests/regression/generation_program_audit.py`; approved manifest SHA256 `d50b737c…` |

## Findings at the baseline (923ab48)

Static, simulator-free audit of the full 7,520-attempt plan pool plus a
108-attempt simulator audit on `vps-a` (36 programs × nominal indices 0–2,
instrumented `run_program`):

- **Timing was inferred, not measured.** Every transition was written as
  `duration_s=0.05`, `physics_substeps=1`. RLBench's IK and gripper actions
  step until convergence: 1–18 physics steps per action (T12 index 0: 436
  actions, 784 physics steps). This contradicted ADR 0007.
- **Fabricated success.** Handles were teleported onto their target marker and
  made static at release (snaps up to 23.7 mm; every T19 close ended ~15 mm
  short). Every placed object was made non-dynamic, non-respondable and
  non-collidable, so later fingers and bodies passed through it.
- **Dropped carries.** `lift`, `grasp_rotate` and `transport_through_aperture`
  re-approached with an open gripper after a `grasp`: T01/T11/V02/G1–G4
  released and regrasped the held object; G2 dropped it from 15 cm.
  Retrieve→place and transport→place compiled into two full pick-and-places.
- **Pushes grazed.** `push_z = obj_z + 0.02` put the fingertips (1.6 mm below
  the tool tip) above the top of 0.8-scale cubes, and lateral waypoint noise
  started pushes off-centre (T03/T12 failures, objects driven sideways).
- **Out-of-order recovery.** A final pass re-ran earlier steps after later
  ones (T12 pushed A after closing the gate) and success ignored mandatory
  history: T04 (open→close) and T14 (park→restore) are satisfied by the
  initial scene.
- **Ill-posed layouts.** With the approved randomization (±2 cm relative gap,
  0.8–1.2 scale, ±30° yaw) bodies spawned inside each other (T06 in 54% of
  attempts, up to 26 mm) and the measured open-finger span (±61 mm along world
  y) swept through neighbours in T04/T06/T08/T12/T16/T19/V01/P1–P4/R2/R3,
  e.g. T16 index 0 launched its blocker 1.67 m. Inserted blockers could spawn
  inside scene bodies in 12 programs. `park_target_a`-style targets were built
  as visible slabs while every other target is an invisible marker.

Measured Panda geometry (tool tip frame, world axes, default yaw): open fingers
x ∈ [−0.0196, 0.0095] m, y ∈ ±0.0608 m; closed fingers y ∈ ±0.021 m; fingertips
0.0016 m below the tip; default tool quaternion tilted 14° about y.

## Changes

Commits `9bb9e38`, `6e310f3` and `dda860a` on `main`:

- Measured per-transition simulator time and physics substeps
  (`achieved_duration_s`, `physics_substeps`); the command keeps the declared
  0.05 s nominal interval separately.
- Compiler held continuations: a step that continues with the held object
  never opens the gripper; retrieve→place becomes grasp, lift, place;
  transport→place keeps the object until the placement.
- Grasps: the tool turns above each body so the fingers close on faces (the
  nearest face, or across an elongated body's short side, choosing the
  wrist-safe equivalent). A grasp counts only if the body is the sole
  attachment and the fingers closed below RLBench `Discrete`'s 0.9 open
  threshold (otherwise its next close command releases the body).
- Push: contact at object centre height, finger-face-centred push line,
  closed-finger + object half-extent contact offset from live geometry, no
  lateral noise behind the object, and a forward-only closed-loop completion
  on the measured object position.
- Release: carried bodies are released 4 mm above rest height and settle under
  physics. No snaps, no dynamics toggling, no frozen bodies. Every routine
  target marker follows its body's settled height.
- Step-local goals (grasp held, placement within 0.01 m, planar push/slide
  goal, reach, rotation check) with at most two executed retries aimed at the
  true target; an unrecoverable step stops the routine.
- Success (`predicate_protocol_id = rlbench-nearcondition-history-hold-v2`):
  RLBench conditions, every step goal in program order, and every final
  predicate on the last 5 boundaries. `step_outcomes` and `success_criteria`
  are recorded in debug metadata.
- Layout revision for T04, T05, T06, T08, T12, T13, T14, T16, T19, V01,
  P1–P4, R1–R4 (positions; 0.06 m blockers become 0.055 m); inserted blockers
  slide along their declared row to the nearest collision-free position
  (declared and applied positions recorded); suffixed park/restore targets are
  invisible markers; four task descriptions no longer name absent elements.
- `layout_clearance.audit_prepared_attempt` encodes the measured finger
  geometry and grasp-yaw rule; tests require zero nominal violations, zero
  spawn overlaps and face-aligned grasp widths below the `Discrete` limit over
  the full 7,520-attempt plan pool.

## Reproduction

Host `vps-a`, `/home/huy2325/ICGS`, CPython 3.10.21 `.venv`, CoppeliaSim 4.1,
PyRep `8f420be8`, RLBench `02720bba`, CPU/Xvfb, task models rebuilt with
`scripts/generation_build_tasks.py --all-programs` at each revision.

```bash
.venv/bin/python -B tests/regression/generation_program_audit.py run \
  --out outputs/audit36/<name> --programs all --nominal 0,1,2 --parallel 6
.venv/bin/python -B tests/regression/generation_program_audit.py summary outputs/audit36/<name>
```

Baseline and the first repair iteration used the equivalent diagnostic driver
`outputs/audit36/sim_audit.py` (same instrumentation, retained on `vps-a`).

## Results

Nominal indices 0–2 of every program (108 attempts per revision):

| Revision | Success | Non-marker teleports | Retries | Episodes > 512 actions |
| --- | --- | --- | --- | --- |
| `923ab48` baseline | 101 labelled (7 failures) | 45 attempts, up to 23.7 mm | n/a | 34 |
| `9bb9e38` | 104 | 0 | 6 | 24 |
| `6e310f3` | **108** | 0 | 0 | 26 |

Baseline failures: G2-2, R3-2, T03-2, T06-1, T12-2, T16-0, T16-2; the true
simulated time was 1.79× the recorded time and a single action took up to 24
physics steps. `9bb9e38` failures were V02-2 (grasp across the 75 mm side left
the fingers "open") and G4-0/1/2 (temporary pad marker 22 mm below rest
height); both were fixed in `6e310f3`.

`6e310f3`, all 36 programs × 3 seeds: every attempt met the RLBench
conditions, every step goal in order and the 5-boundary final hold; maximum
final predicate distance 9.0 mm (T03, lateral push drift, addressed by the
finger-face-centred push line in `dda860a`); median 344.5 actions; up to 18
physics steps per action (mean 1.66).

| Program | Success | Retries | Max final error (mm) | Actions |
| --- | --- | --- | --- | --- |
| T01 | 3/3 | 0 | 2.0 | 108–109 |
| T02 | 3/3 | 0 | 1.2 | 185–189 |
| T03 | 3/3 | 0 | 9.0 | 146–150 |
| T04 | 3/3 | 0 | 2.0 | 229–230 |
| T05 | 3/3 | 0 | 2.6 | 307–318 |
| T06 | 3/3 | 0 | 2.5 | 328–339 |
| T07 | 3/3 | 0 | 2.0 | 350–366 |
| T08 | 3/3 | 0 | 2.6 | 527–542 |
| T09 | 3/3 | 0 | 1.9 | 181–189 |
| T10 | 3/3 | 0 | 3.3 | 236–249 |
| T11 | 3/3 | 0 | 2.2 | 230–243 |
| T12 | 3/3 | 0 | 2.2 | 334–338 |
| T13 | 3/3 | 0 | 2.3 | 316–319 |
| T14 | 3/3 | 0 | 3.6 | 291–298 |
| T15 | 3/3 | 0 | 4.2 | 474–492 |
| T16 | 3/3 | 0 | 2.5 | 460–469 |
| T17 | 3/3 | 0 | 3.2 | 289–295 |
| T18 | 3/3 | 0 | 2.4 | 362–370 |
| T19 | 3/3 | 0 | 2.0 | 509–521 |
| T20 | 3/3 | 0 | 2.1 | 326–337 |
| V01 | 3/3 | 0 | 2.4 | 415–418 |
| V02 | 3/3 | 0 | 2.0 | 232–237 |
| V03 | 3/3 | 0 | 2.8 | 467–486 |
| V04 | 3/3 | 0 | 3.7 | 410–420 |
| P1 | 3/3 | 0 | 2.4 | 605–621 |
| P2 | 3/3 | 0 | 3.0 | 606–619 |
| P3 | 3/3 | 0 | 3.4 | 790–812 |
| P4 | 3/3 | 0 | 2.6 | 749–760 |
| G1 | 3/3 | 0 | 3.5 | 202–203 |
| G2 | 3/3 | 0 | 2.4 | 238–248 |
| G3 | 3/3 | 0 | 5.9 | 414–427 |
| G4 | 3/3 | 0 | 2.4 | 291–299 |
| R1 | 3/3 | 0 | 2.7 | 462–480 |
| R2 | 3/3 | 0 | 3.6 | 781–790 |
| R3 | 3/3 | 0 | 3.5 | 718–736 |
| R4 | 3/3 | 0 | 2.5 | 628–633 |

### Gates at `dda860a`

- Push programs, nominal 0–2 (`outputs/audit36/push-dda860a`): **PASS 9/9**.
  Final errors T03 2.0/4.7/1.2 mm, T10 3.0/2.2/2.0 mm, T12 5.3/2.3/8.1 mm,
  all lateral. Two of nine pushes (T03-0, T10-1) needed one executed
  corrective re-push: a yawed cube touched at a corner rotates and drifts
  sideways; the step goal detects >10 mm and the retry re-pushes. Centring the
  push line on the finger face helped T03, was neutral for T10 and worse for
  T12 in these seeds; errors stay inside the 10 mm predicate.
- 36-program pipeline smoke through the canonical worker, archive writer and
  `validate_closed_result` (`tests/regression/generation_placement_probe.py`,
  `outputs/readiness-runtime.json`, keep retention, validation mode, no HF,
  parallelism 6, nominal index 0 of every program): **PASS, 36/36 success**,
  every timeline consistent, every archive valid, clean tree, 1,754 s,
  2.6 GiB retained. The proof `outputs/smoke-dda860a/proof.json` (SHA256
  `4ea07e89753dc98022f1721409b1650f28400d8d43b48da14f944c89025f7c71`)
  passes the launcher's `validate_smoke_receipt`. Reading all 36 archives
  back: per-transition `dt` is measured (0.05–0.80 s), `debug.json` records
  `predicate_protocol_id = rlbench-nearcondition-history-hold-v2`, mandatory
  history and final hold true, and one `step_outcomes` row per routine step.
- First production-order plan of every applicable perturbation kind for all
  36 programs (158 attempts, `outputs/audit36/perturbed-dda860a`): 153
  success, 5 `valid_failure`, 0 crashes/invalid observations, 0 teleports,
  65 executed retries (mostly corrections of `action_pose_offset`
  placements). Valid failures: T01 `action_pose_offset` (the perturbation
  moves the lift target *and* offsets the lift command, so the lifted body
  ends 12 mm from the moved marker; held lifts have no scripted retry),
  R1/T14 `blocker_insertion` (the inserted blocker is struck by opening
  fingers or occupies the restore spot), and T12 `object_displacement`/
  `blocker_insertion` push failures. The T12 traces exposed two push-recovery
  defects fixed in `9ef4140`: completion kept advancing after the body slid
  off the finger path, and retries re-used an aperture waypoint the body had
  already passed.

### Final gates at `9ef4140`

- Push programs T03/T10/T12, nominal 0–2 plus perturbed 32
  (`object_displacement`) and 48 (`blocker_insertion`)
  (`outputs/audit36/push-9ef4140`): **PASS 15/15** — including both T12
  perturbed attempts that failed at `dda860a`, now recovered by one or two
  executed corrective pushes. Largest final error 9.8 mm (T12-48).
- 36-program pipeline smoke (same command and runtime as above, nominal
  index 0): **PASS, 36/36 success**, every archive valid, every timeline
  consistent, clean tree, 1,852 s, 2.6 GiB retained, median 342.5 and maximum
  812 actions. The launcher's `validate_smoke_receipt` accepts
  `outputs/smoke-9ef4140/proof.json` (SHA256
  `7123a61d930ff3e99fc431b1108c010217172b8b25a26a9760b26ccf41ae2d2f`).

Collector validation at `9ef4140` combines this smoke, the push re-check and
the unchanged-code evidence of `6e310f3` (108/108 nominal) and `dda860a`
(158 perturbed); `9ef4140` changes only push completion/retry behaviour.

## Launch handoff (NOT RUN)

Full generation was **not** started. After the owner accepts this record, an
operator on `vps-a` (`~/ICGS` at the accepted commit, task models rebuilt) can
launch the prepared production runtime with the new receipt:

```bash
TMPDIR="$PWD/outputs/tmp" XDG_CACHE_HOME="$PWD/outputs/cache" \
HF_HOME="$PWD/outputs/cache/hf" .venv/bin/python -B scripts/generation_launch.py \
  --runtime-config outputs/generation-vps-a-runtime.json \
  --approved-manifest artifacts/composition/approved_composition_manifest.json \
  --code-revision "$(git rev-parse HEAD)" \
  --smoke-receipt outputs/smoke-9ef4140/proof.json \
  --hf-token-path outputs/readiness-hf/credential --detach
```

Use a code revision whose collector files equal those of the smoke (only
documentation may differ). The runtime's run root and HF prefix
(`generation/archive-v1-vps-a-20260927`) are still unused.

## Owner decisions and remaining limits

These are deliberate abstractions or open items, not defects hidden by the
gates above:

- **Articulated containers are free handle blocks.** Drawers and gates are a
  dynamic handle slid along the table between markers; the container body is
  static. The physics is honest (no snaps) but there is no joint.
- **Most dependencies are symbolic.** Bodies travel at 0.90 m, so parking a
  blocker, opening a drawer or passing an aperture waypoint is rarely
  physically required; T12's gate (the push path crosses the closed gate) and
  P4's occupied slot are the physically enforced dependencies.
- **Episode length.** Nominal P1–P4, R2–R4, T08 and T19 episodes take 510–812
  RLBench actions, above `collection.max_episode_intervals = 512`. The
  generation worker does not enforce that cap and the units differ (variable
  RLBench actions versus fixed control intervals); training ingestion must
  decide truncation or resampling.
- **Pushes are corner contacts.** Yawed cubes drift sideways; step goals and
  bounded corrective re-pushes keep outcomes inside 10 mm, and final errors of
  8–10 mm occur.
- **Perturbation semantics are unchanged.** `action_pose_offset` still moves
  the target marker and offsets the commands; `gripper_timing` changes hold
  steps after a Discrete actuation that always runs to completion.
- **Inserted blockers** keep their declared row but slide to a collision-free
  spot; they can still obstruct paths or targets (valid failures).
- Measured `achieved_duration_s` is a float32 simulator-clock difference
  (resolution ≈1e-5 s); `physics_substeps` is the exact count.
- Not run: full 7,520-attempt collection, HF publication of these archives,
  training ingestion of archive views, L2/C1–C5 checkpoint gates.
