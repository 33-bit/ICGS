# Placement controller root-cause experiment

Status: superseded by the [collector-semantics audit](collector-semantics-20260929.md),
which removes the release guard/freeze and handle snap introduced here and
fixes the T12 push failure (push height, contact and recovery). Retained as
historical evidence.

Category B, CPU Colab (VPS unavailable), 2026-09-28. Baseline: local main
`55585ea0af6e27ab9a478a41db24791646ca0723`; Colab cloned `0f92037` and
received the exact local worker plus opt-in diagnostic instrumentation.
No HF credentials, publication, training, or full quota run.

## Contract

Compare the first two canonical nominal attempts of T06 and V02. Keep approved
manifest, scene/collection seeds, tolerance (0.01 m), layouts, camera settings,
and pinned simulator stack unchanged. Test one controller/layout-adapter change
at a time before combining. No tolerance relaxation or new object-position snaps.
Native Instant Policy inference/training is outside this experiment.

Requested/used seeds: T06 13996986, 132996328; V02 376137200, 1882450994;
collection seed 20260920. These reproduce the earlier focused pilot, not the
full evaluation planner's separate seed schedule.

## Environment and reproduction

OAuth2 session `icgs-cpu-rootcause-20260928`, `/content/ICGS`, Python 3.10.21,
CPU torch 2.2.0, NumPy 1.26.4. Setup receipt PASS using:

```sh
python scripts/setup_environment.py --profile generation --python-version 3.10 \
  --venv-root /content/ICGS/.venv --provision-simulator --build-generation-tasks \
  --runtime-config /content/ICGS/outputs/rootcause-runtime.json \
  --receipt /content/ICGS/outputs/rootcause-setup-receipt.json
```

PyRep `8f420be8064b1970aae18a9cfbc978dfb15747ef`, RLBench
`02720bba4c73fe02eb75df946b8791b806028a9d`, simulator SHA256
`512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8`.
One worker per bounded sequential script; temporary independent hypothesis probes
may overlap. Source-only diagnostics use canonical full wrist capture, but do not
write training episodes. Archive validation is a separate later gate.

## Evidence so far

- Baseline four episodes: 0 successes, 4 valid failures (worker exit 0 is not
  success). T06 distances 0.180836 m and 1.847083 m; V02 0.101584 m and
  14.753653 m. All recorded action/observation counts were aligned. This does not
  certify true internal physics timestep metadata.
- T06 trace: native `Discrete()` attaches both `object_a` and `drawer_handle`
  when closing on A. A remains at table height as the tip lifts and carries the
  handle; the controller incorrectly recomputes a carrying offset from A.
- Changing only `Discrete(attach_grasped_objects=False)` retains the worker's
  explicit named-object grasp. T06 seed 13996986 then carries only A and ends at
  0.009101 m from target. Other baseline mechanism snapping remains present;
  this is a scoped attachment proof, not physical-articulation acceptance.
- V02 repeated baseline matches 0.101584 m. The object follows the retreat after
  the attachment list becomes empty and parent is None. Both finger open amounts
  remain 1.0. `apply_prepared_layout` silently ignores sizes because the pinned
  Shape has `scale_object`/`get_bounding_box`, not `set_size`; target heights use
  scaled dimensions despite the physical shape staying unscaled.
- Scaling-only hypothesis: both former V02 failures succeed, at 0.000862 m and
  0.000385 m. After release the object remains at the holder instead of following
  the retreat. Actual finger apertures now fall below 0.9 during the grasp and
  reopen to about 0.99. This isolates the missing physical scale application;
  no new gripper state machine or enlarged tolerance was needed.
- Three new focused CPU tests failed before their fixes and passed after:
  absolute/idempotent physical scaling, actual singleton attachment, and matching
  recovery step. First expanded fixture run: 127 passed / 8 failed because the
  existing test stub did not accept native Discrete's configuration argument.
  The stub now requires auto-grasp disabled. Fresh expanded run: **189 passed**
  (worker, archive, generation, expert, batch, attempt), 0 skipped, 7.41 s.
- L0 on CPU Colab: **PASS**, 22 harness tests, exit 0. No native IP/model or
  checkpoint claim; L2/C1–C5, L4, training and HF publication NOT RUN.
- Pilot startup attempts rejected before any simulator job: nonexistent base
  run root, then insufficient verification scratch reserve. The corrected probe
  creates a fresh run root and reserves two writer caps (2 GiB); no safety check
  was weakened. The no-trace bounded run completed all 12 jobs in 946.0 s and
  closed/validated every archive with no simulator crash or invalid observation.
  Outcomes were 9 `success` and 3 `valid_failure`: T06 index 1 and V02 indices
  12/24. Their final errors were 0.085788 m, 0.206930 m and 0.815166 m respectively.
  Archive validity does not explain or excuse these remaining failures.
  The proof is retained at
  [the committed proof](placement-rootcause-cpu-20260928-proof.json); publication and
  training were not run. A one-job no-trace T06 replay passed in 70.9 s.

The first traced bounded run exceeded the 600 s worker bound, after which the
queue correctly stopped on preserved archive scratch. The initial diagnosis of
trace snapshot overhead was wrong: the parent polls for exit without draining
stdout/stderr pipes, while the child prints the trace before archive closure.
Large output can fill the pipe and block the healthy child. Two real-child CPU
regressions (1 MiB on stdout or stderr, then write a completion marker) each
failed by timeout before repair. The worker now drains both streams using
bounded, retryable `communicate` calls while renewing heartbeats, checking stop
markers, and preserving TERM/KILL deadlines. It also drains during TERM grace.
Tracing remains opt-in; neither timeout nor safety-stop policy is relaxed.

The [post-pipe-fix traced replay](placement-rootcause-cpu-20260928-traced-proof.json)
(T06 indices 0/1 and V02 12/24) completed in
301.24 s. All four complete archives validated; T06-0 succeeded, the other
three reproduced the same numerical failures. This contradicts the earlier
trace-overhead diagnosis and verifies that output draining fixes the timeout.
The final focused suite on CPU Colab reports **192 passed, 0 skipped, 8.72 s**,
including stdout/stderr pressure and verbose graceful-shutdown regressions.

### Remaining controller failures (not fixed)

- T06 seed 132996328: A is within 1 mm of its target before release. During
  opening it moves from x=0.24947 to 0.18788 m; recovery repeats the displacement
  to x=0.16459 m. This localizes the problem to release/contact behavior, not
  carrying offsets, archive corruption, or a need for looser predicates.
- V02 seed 1068535031: both fingers remain above 0.98 after close and explicit
  attachment. The attachment list becomes empty during the lift and A stays on
  the table. Pinned RLBench `Discrete.action` treats both fingers >0.9 as open;
  with native auto-grasp disabled its repeated close enters the release branch.
  Thus singleton attachment at one instant is not proof of a maintained grasp.
  A pose-aware grasp/hold correction needs separate validation across scale/yaw.
- V02 seed 697076834: explicit grasp is never confirmed; A is displaced beyond
  reach (x=1.01517 m), yet the controller continues carrying motions with stale
  object-tip assumptions. The current repeated-close recovery is insufficient.

These failures remain blockers. No new teleport, frozen object, changed seed,
relaxed predicate, or fabricated successful outcome was introduced to hide them.

Final focused command, cwd `/content/ICGS`, Python 3.10.21:

```sh
.venv/bin/python -B -m pytest -q tests/test_generation_worker.py \
  tests/test_generation_archive.py tests/test_generation.py \
  tests/test_generation_expert.py tests/test_generation_batch.py \
  tests/test_generation_attempt.py
.venv/bin/python -B scripts/validate_fast.py
.venv/bin/python -B -m pytest -q tests/test_generation_queue.py \
  tests/test_generation_storage.py tests/test_generation_control.py \
  tests/test_generation_planner.py
```

Queue/storage/control/planner CPU fixtures: **181 passed, 0 skipped, 25.84 s**.
L0: PASS (22 harness tests); selected CPU fixtures: PASS. Full L1, native
checkpoint C1–C5, L4, training/export acceptance, HF publication, throttling and
full-generation readiness: NOT RUN / not certified by this experiment.
An additional local Python 3.14 test invocation initially failed collection
because ICGS was not installed/on the path; the explicit `PYTHONPATH=src` rerun
passed 189 tests. CPU Colab results above are the requested acceptance environment.

Tested runtime source hashes (local files uploaded to Colab; the dirty checkout's
Git HEAD alone is not the tested code identity):

- `scripts/generation_episode_worker.py`: `0a22e86a88900637a78e9e30dcf89b651eddd7318c0f96940243cef398a3321c`
- `scripts/generation_worker.py` after pipe repair: `6cdc733275511ce1d75f790c54259cf7b83e81aaab802915e0cb67a67d90159a`

Diagnostic/queue metadata and trace logs were copied locally to
`outputs/placement-evidence-final-metadata-20260928.tar.gz` (4,614,367 bytes,
SHA256 `5f392d67aa188831a25fa8be134d6b930aee2d5396eb1a7ad1a1ab49db902d03`).
This bundle excludes numeric `.npz` chunks; it must not be described as a full
archive backup. Separate full-archive transfer is tracked below when verified.
The CPU session was stopped prematurely during that transfer (operator error).
The local `outputs/placement-pilot-fixed5-raw.tar` is incomplete: 705,807,360 of
745,779,200 expected bytes, and is **not** a verified backup. Remote expected
SHA256 was `6b0de49a8db2d2f4d6eb8b12c29cce31507509c890087cd355f81160440755d2`.
The raw numeric collection was not fully preserved before session termination;
do not use the partial tar as evidence of complete local archive availability.
The verified metadata bundle, committed proof files and final test logs survive.
`colab --auth oauth2 sessions` confirms no active sessions after termination.
The early probe also failed after one completed job when reusing an active worker
lease with a new instance, then when using a worker ID beyond the configured
scope. The final probe uses one configured ID and a stable instance identity in
strictly sequential `--once` invocations; no lease is deleted or stolen.
These startup/probe failures and the traced safety-stopped run are preserved,
not deleted or resumed around the queue's guard. Their HF publication status is
NOT RUN: publication was disabled, no coordinator launched, and workers had no
HF credentials.

Bounded regression command (12 jobs, one simulator slot; no coordinator/refill):

```sh
/content/ICGS/.venv/bin/python -B -m tests.regression.generation_placement_probe \
  --runtime-config outputs/rootcause-runtime.json \
  --output-root outputs/placement-pilot-fixed5
```

Selects indices 0, 1, 12, 24 of T06/V02, spanning all three scale strata, and
indices 0/1 of T01/T08 controls. Set `ICGS_PLACEMENT_TRACE=0` for this bounded
run; tracing is for focused reproductions only. The proof records the actual
planned seeds, source-file hashes, dirty status, all outcomes, and complete
closed-result validation. Numerical task outcomes and archive validity are
separate claims.

## Other discovered risks (not acceptance evidence)

The stale retry-step selection and proximity-only attachment helper were repaired.
Attachment lifetime and release-contact behavior remain unresolved as listed above.
Rotation motions currently command no rotation. Existing success-snapping and
high-level action timing need explicit accounting before training-ready claims.
These must not be hidden by archive structural validation.

### Release-contact follow-up

The patched worker's 36-program rerun reduced the residual physical failures to
R2, T08 and T12. Traces showed R2 blocker B and T08 object A launching when a
dynamic placed body was re-enabled immediately after gripper release; T20 and
R3 exhibited the same release-contact mechanism. A bounded
`release_collision_guard` now covers release and retreat, and successfully
placed non-articulated bodies remain settled/non-respondable until an explicit
later grasp re-enables physics. The focused five-program replay passed R2,
T08, R3 and T20 with complete archive validation; the seven-program control
replay (G2, P3, R3, T11, T17, T19, T20) was 7/7 success with complete archive
validation.

T12 remains a valid failure. Its gate opens, but the push-through primitive
leaves object A on the near side before the close action, about 0.73 m from the
target. Lowering the push height and moving the catalog aperture waypoint to
the open marker were tested independently and reverted because neither changed
the outcome. No tolerance, snap, seed or archive result was changed. Full
generation remains stopped pending a trace-backed T12 aperture fix and a fresh
36-program rerun.
