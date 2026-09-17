# Plan: isolated RLBench E01 G1 feasibility collection

Status: active -- Phase 0-4 code-ready conditional on independent reviewer audit clearance; live physical proof remains pending Phase 6 smoke execution on named T4 Colab session.
Owner: repository owner; execution coordinator.

Motivation and observable acceptance criteria: establish whether the pinned
RLBench/PyRep/CoppeliaSim 4.1 stack can produce a small, correctly timed,
losslessly archived, Drive-first/HF-second exploratory physical trace. A PASS
requires a live pinned-API receipt, one closed archive with verified dual-store
publication, and no claim beyond exploratory G1 feasibility.

Research motivation, baseline and hypothesis: P02's archive/view runner is the
unchanged baseline. The hypothesis is that the exact upstream simulator can be
driven through a low-level target-hold controller for two real 0.05-second steps
per 0.1-second interval, while keeping full measured clouds separate from a
derived online representation.

## Current and target states

Current source contains an unreviewed E01 controller/oracle/uploader path. Sol and
Gemini investigations found non-existent RLBench API calls, a synthetic fallback
clock and trajectory, lossy canonical cloud conversion, shared dev-track leakage,
and batch rather than per-episode publication. The current path is blocked.

Target source follows [ADR0012](../../decisions/0012-exploratory-rlbench-g1-track.md),
[ADR0007](../../decisions/0007-icgs-timed-data-lineage.md), and
[ADR0011](../../decisions/0011-episode-archive-and-initial-views.md). It uses the
pinned upstream revision only through the outer RLBench environment boundary,
records actual measurement and execution results, and leaves primary ICGS tracks
untouched.

## Invariants and scope

- E01 is feasibility-only and cannot be used as P11 training, calibration, or
  benchmark data.
- No T01--T20/V01--V04/P/G/R program is substituted, relabelled, or changed.
- A live G1 API probe precedes any E01 archive or upload.
- No synthetic timestamp, command, fixed-length pad, cloud replacement, or
  publication success inference is permitted.
- Canonical raw clouds and poses may differ from online arrays; neither can carry
  task/oracle/predicate fields. Discrete measured and online gripper state remains
  equal unless a later approved decision changes that semantic.
- Each closed episode is immutable and copied Drive-first, HF-second, then indexed
  last; failure stops before another physical attempt begins.
- No training, C1--C5, G2, primary collection, or benchmark workload is in scope.

## Phases and progress

- [x] **0. Reproduce and encode release blockers.** The G1 entrypoint now has a
  bounded, Drive-only v2 receipt guard: it requires the pinned reset tuple,
  source-bound RLBench/PyRep imports, live Jacobian-IK target hold through exactly
  two `Scene.step()` calls, queried 0.05-second simulator timing, finite raw cloud,
  clean shutdown, pinned source/archive evidence, and an atomic FAIL-before-PASS
  receipt lifecycle. The Phase-0 collection entrypoint contains no controller,
  archive, mock-success, or upload path and refuses every real or mock collection
  call. Local RED→GREEN coverage includes malformed/forged receipts, ambient module
  caches, dirty source trees, non-Drive paths, target-hold/clock failures, and stale
  PASS replacement. Independent re-audit: PASS. This is not a live G1 result.
- [x] **1. Live pinned-API G1 probe.** On 2026-09-17 local time, the named T4
  session `icgs-e01-g1-20260916` mounted Drive, provisioned the exact pinned
  Python 3.11/CoppeliaSim/PyRep/RLBench environment, and ran the bounded
  non-demo `PickAndLift` probe. The final Drive receipt is `PASS` and passes the
  strict Phase-0 verifier: two real 0.05-second scene steps advanced the backend
  clock by 0.10000002384185791 seconds; a finite `float64 [128,128,3]` wrist
  cloud was observed; target-hold joint/gripper errors were within tolerance; and
  the outer environment shut down cleanly. The receipt records the clean pinned
  source revisions and CoppeliaSim archive hash. An initial fail-closed probe
  found the pinned scene requires wrist depth capture as well as point-cloud
  output; the source-backed root cause was corrected with a focused regression
  test before the single successful retry. No live demonstration was executed,
  and no episode archive, shard, manifest, upload, or collection call occurred.
- [x] **2. Correct archive/view boundaries.** Measured-vs-online cloud/pose
  validation now follows ADR0011 with independent stream continuity and equal
  binary grip. Manifests carry an explicit `dataset_track`; exploratory data is
  rejected by primary training views. Archive schema/version and primary fixtures
  remain compatible.
- [x] **3. Implement a fail-closed controller and expert materializer.** The
  controller requires private pinned RLBench handles, queried PyRep timestep,
  backend simulator time, Jacobian IK, joint targets, gripper actuation, live
  clock advancement, raw-observation access, and one-time outer shutdown. The
  materializer requires concrete demo waypoints plus content hash/protocol
  identity; missing pose/grip, nominal fallback, truncation, and padding fail
  closed. No live expert demonstration was run.
- [x] **4. Implement continuous dual publication.** The uploader requires Drive
  and HF targets, verifies every immutable episode artifact Drive-first, submits
  one HF commit, reads back exact bytes at its revision, and publishes an
  exploratory index last. Single-file uploads, credential/source files, path
  traversal, and mismatched immutable artifacts fail closed. The runner invokes a
  required publisher hook before index persistence and stops on transfer failure.
  No Drive/HF episode publication was run.
- [ ] **5. Local verification and independent review.** Local test-driven hardening
  complete across all Pro audit invariants: (a) target preflight before simulator
  construction with reversible Drive probe and non-destructive HF capability check;
  (b) separate-process CLI supervisor with hard wall timeout, process-group termination
  (SIGTERM -> bounded SIGKILL), durable JSON outcome persistence via atomic fsync, and honest UNKNOWN
  cleanup reporting; (c) parameter validation failing BEFORE preflight/worker in both API and CLI
  (`num_attempts == 1`, `1 <= max_intervals <= 16`, positive finite wall/timeout caps, positive disk limit);
  (d) file-backed bounded logs preventing subprocess pipe-buffer deadlocks while bounding supervisor memory;
  (e) structured worker-result JSON (`icgs_worker_result_v1`) written atomically with fsync and strictly verified
  by supervisor via per-run hex nonce, schema containment, exactly 1 published episode, dual Drive+HF verified index,
  and completed cleanup before declaring `SUCCESS`;
  (f) secure explicit `--hf-token-file` (default `/content/.icgs_hf_token`) with regular non-symlink verification,
  token never passed in CLI argv/env/logs/outcome and cleared immediately after client init;
  (g) syntactic Drive directory containment rejecting symlinks, `/`, and non-Drive roots;
  (h) CAS parent_commit optimistic concurrency control; (i) strict publication allowlist rejecting symlinks
  and nested structures; (j) fail-closed oracle tail waypoint cap handling; and (k) outer environment cleanup
  on launch failure. Audit status: PENDING external Pro reviewer audit clearance.
- [ ] **6. Bounded live smoke.** Create a new named T4 Colab session, remount
  Drive, verify stored HF credential without exposing it, run only one E01 attempt
  with no more than 16 intervals and strict wall/disk caps, read back Drive/HF
  bytes, and stop the named VM. Record actual result as PASS/FAIL/NOT RUN without
  reclassifying it as G2/training evidence.

## Validation and experiment strategy

L0: `python3 -B scripts/validate_fast.py` and `python3 -B -S
scripts/validate_fast.py`.

L1: `.venv/bin/python -B -m unittest tests/test_rlbench_exploratory.py
tests/test_episode_archives.py tests/test_collection_runner.py
tests/test_episode_views.py tests/test_episode_data.py tests/test_architecture.py -v`.

L3 G1: a named Colab T4 session, exact Python/upstream revisions, and a bounded
API probe. A single 16-interval E01 attempt is Phase 6 work only after Phases
2--4 have been implemented and reviewed. Drive and HF readback checks are part
of Phase-6 evidence. A missing Drive mount, expert demo, clock/IK capability, or
transfer verification is FAIL/BLOCKED and stops the job.

## Compatibility

Native policy/data/checkpoint formats, P11 training inputs, and primary evaluation
are unchanged. P02 archive files retain their version; the corrected validator
preserves both canonical and online representations rather than requiring their
numeric equality. Existing manifests remain valid under the explicit primary-track
default; exploratory manifests are intentionally not interchangeable.

## Risks, open questions and recovery

The concrete pinned API, actual expert trajectory structure, stable target hold,
and Drive remount all require the live probe. If any fails, retain the safe receipt
and diagnostics, stop the named VM, and leave E01 collection blocked. Do not retry
with higher-level `task.step`, fabricated coordinates, a different timestep, or a
different task identity. Drive-only episodes that fail HF publication remain
unindexed and resume through checksum-verified publication; they are never
silently deleted or regenerated under the same ID.

## Final evidence

Phase-0 local evidence, 2026-09-16, cwd
`/Users/33bit/AI/Research/VLA/ICGS`:

- PASS — `.venv/bin/python -B -m unittest tests/test_rlbench_exploratory.py
  tests/test_episode_archives.py tests/test_collection_runner.py
  tests/test_episode_views.py tests/test_episode_data.py tests/test_architecture.py -v`:
  149 executed, 0 failures, 0 skips.
- PASS — `python3 -B scripts/validate_fast.py`: L0 19/19.
- PASS — `python3 -B -S scripts/validate_fast.py`: L0 19/19.
- PASS — targeted `py_compile` and `git diff --check`.
- PASS — independent narrow Phase-0 re-audit after the source-binding correction.
- PASS — live G1 API probe, 2026-09-17 local time: named session
  `icgs-e01-g1-20260916`, mounted Drive output
  `/content/drive/MyDrive/ICGS-data-20260916/g1-probe/g1-receipt.json`, and
  `PickAndLift` without `--run-live-demo`. Its strict in-session verifier
  accepted the protocol, clock, raw cloud, target-hold, cleanup, and pinned
  provenance evidence. A direct post-stop Drive readback verified final receipt
  SHA256 `ff8fdfb68a609f553c7a936b070ad4443c48929e28948524df378dca53080a25`.
  The output directory contained only `g1-receipt.json`.
- PASS — focused TDD correction after the first live fail-closed receipt:
  `.venv/bin/python -B -m unittest tests/test_rlbench_exploratory.py` ran 44
  tests, 0 failures. The new regression locks the required wrist depth capture.
- PASS — post-G1 affected L1 regression:
  `.venv/bin/python -B -m unittest tests/test_rlbench_exploratory.py
  tests/test_episode_archives.py tests/test_collection_runner.py
  tests/test_episode_views.py tests/test_episode_data.py tests/test_architecture.py -v`
  ran 150 tests, 0 failures, 0 skips.
- PASS — post-G1 L0: `python3 -B scripts/validate_fast.py` and
  `python3 -B -S scripts/validate_fast.py`, 19/19 each; targeted `py_compile`
  and `git diff --check` also passed. The two host-Python `SyntaxWarning`s in
  `tests/test_task_router.py` were pre-existing and unrelated.
- PASS — Phase 2--4 focused implementation regression:
  `.venv/bin/python -B -m unittest tests/test_rlbench_exploratory.py
  tests/test_rlbench_phase34.py tests/test_episode_archives.py
  tests/test_collection_runner.py tests/test_episode_views.py
  tests/test_episode_data.py tests/test_architecture.py -v`: 165 tests, 0
  failures, 0 skips. This is local fake/unit evidence only; it does not certify
  a live simulator, expert demonstration, Drive transfer, HF commit, or training
  dataset.
- PASS — independent re-audit of the evidence correction: a fresh Drive
  readback matched the documented `ff8fdfb68a609f553c7a936b070ad4443c48929e28948524df378dca53080a25`
  receipt SHA256 and its PASS clock/cloud/target-hold/cleanup/pin evidence; no
  stale receipt digest or collection-scope claim remained in the records.
- PASS — `colab stop -s icgs-e01-g1-20260916`; `colab sessions` then reported no
  active sessions.
- NOT RUN — E01 archive generation, Drive/HF publication, live expert
  demonstration, G2, training, C1--C5, and benchmark workloads. A bounded live
  archive smoke and independent review remain prerequisites before any physical
  collection; no training-data claim is made.
