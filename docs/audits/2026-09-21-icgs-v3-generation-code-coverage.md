# ICGS v3 generation-plan/code coverage audit

Date: 2026-09-21  
Audit revision: working tree after coordinator/watchdog and raw-executor repair
Scope: `docs/superpowers/plans/2026-09-21-primary-v3-distributed-generation.md`,
the v3 generation spec, distributed runtime, pilot runner, materializer,
training layout, views, queue, planner, validation and HF publisher.

## Module and execution map

| Responsibility | Runnable owner | Evidence / status |
|---|---|---|
| Program semantics, layouts, predicates | `src/icgs/data/collection/v3/{steps,compiler,scene_layouts}.py` | Used by build and pilot; parity/build smoke exercised |
| Attempt planning and quotas | `v3/{batch,quota,diversity,perturbations}.py` | Unit-tested; centralized planner used by coordinator |
| Simulator execution | `scripts/colab_v3_pilot_episodes_worker.py` | RLBench smoke runner; now captures measured simulator observations rather than a repeated tip point |
| Closed episode/attempt archive | `v3/rlbench_attempt.py` and pilot writer | Full layout + artifact manifest required by validator |
| Queue and worker | `v3/distributed_queue.py`, `scripts/colab_v3_distributed_worker.py` | Atomic claim/ready states; worker is credential-free |
| Validation/ingestion | `v3/distributed_validation.py`, coordinator | Identity, checksums, timeline, schema, layout and artifact manifest checks |
| Resume/planning | `v3/distributed_planner.py`, coordinator resume helpers | Reconstructs pending/claimed/ready/ingested/published state; remote manifest fetched on startup |
| HF publication | `v3/distributed_publication.py` | Semantic paths, compact manifest, receipts/views, bounded batches and remote hash verification hook |
| Launch/keepalive | `scripts/colab_v3_distributed_launch.py`, watchdog | Exact 200 displays, detached output, atomic PID receipt, bounded replacement and coordinator lock |
| Training views | `src/icgs/data/datasets/v3_views.py` | Pointer generation is now emitted per episode; sampler remains downstream |

## Plan coverage matrix

| Plan task | Result | Remaining gap / proof |
|---|---|---|
| T1 contracts + queue | PASS | 24+ focused tests; no distributed lock owner yet |
| T2 full-fidelity materialization | PASS (local) | `execute_raw_attempt(task, env, spec) -> RawAttempt` is the shared compatibility boundary; canonical materializer writes layout/artifact manifest. |
| T3 planner/quota/resume | PASS with recovery repair | Planner counters and attempt caps are tested; coordinator reconstructs local closed state and fetches remote manifest. Remote conflict tests remain pure-local. |
| T4 closed-result validation | PASS after repair | Episodes without `layout/` or `artifact_manifest.json` are rejected; crash/invalid attempts require `attempt.json`. |
| T5 five-minute HF publisher | PASS after repair | Uses semantic paths, compact manifests, `resume_receipt.json`, `publication_receipt.json`, four view pointers, bounded commits and a remote hash verifier. |
| T6 worker | PASS (local seam) | Binding is loaded from the approved manifest; native worker seam consumes shared raw executor/materializer/writer; CLI fallback remains pilot-compatible. |
| T7 coordinator/watchdog/launcher | PASS (local) | Non-blocking single-owner lock, PID command-line identity checks, bounded worker/coordinator replacement, atomic launch receipt and fixed displays are tested. |
| T8 docs/validation | IN PROGRESS | This audit and launch audit are updated; full plan checklist is not retroactively marked complete. |
| T9 full launch | PAUSED | Session was explicitly stopped for schema repair. No new session may launch until fresh smoke plus schema-aware bounded worker/publication receipts pass. |

## FACT / ASSUMPTION / UNKNOWN

FACTS:

- The old full-run batch contained only `episode.json` and `physics.json` under
  internal job paths; it did not satisfy the frozen training-layout contract.
- The v3 schema requires `T` actions, `T+1` observations, `T` durations,
  measured origin, and retained `valid_failure` episodes.
- Local focused generation/queue/validation/layout tests pass after the repair;
  the paused Colab run is not evidence for the repaired writer.
- The previous canonical HF commit `09368256…` contains 200 compact-manifest
  episodes from the pre-repair writer and must not be treated as repaired data.

ASSUMPTIONS:

- Existing HF paths under `primary_v3/` can be removed in a dedicated cleanup
  commit without touching `primary_v2/`; this must be verified against the repo
  tree before deletion.
- The available RLBench observation profile supplies wrist point clouds and
  gripper poses; RGB/depth/masks/joints are emitted only when actually present.

UNKNOWNS:

- Whether every historical legacy batch path is still present at the same HF
  revision after subsequent commits.
- Whether Colab can keep 200 simulator/Xvfb processes alive long enough for the
  repaired run without reclaim; this requires a fresh bounded smoke.
- Whether the watchdog replacement semantics satisfy the plan under a real
  coordinator death; no live replacement proof exists yet.

## Required next proof

1. Run local focused tests and `validate_fast.py` on the repaired tree.
2. Start a fresh CPU/TPU VM only after the source is committed.
3. Run one repaired T03 and one repaired valid-failure attempt; inspect the
   closed result file inventory and `layout/layout_manifest.json`.
4. Run 2-worker and 200-worker bounded smokes; reject any result with fewer than
   the required layout/artifact files or any synthetic observation arrays.
5. Publish one repaired preflight batch under a new `primary_v3/preflight/...`
   prefix and verify every remote hash.
6. Clean or quarantine the legacy `primary_v3/results/...` batch only after a
   remote file inventory confirms its exact paths; never mutate `primary_v2/`.
7. Resume full generation only after these receipts pass.

## Repair evidence (2026-09-21)

The local repair is bounded to orchestration/reuse and does not launch a
simulator or Colab session. `CoordinatorLock` holds an `fcntl.LOCK_EX |
fcntl.LOCK_NB` descriptor for the full coordinator lifetime; the watchdog
proves the lock is free before replacing a coordinator and verifies every PID
against the expected run root/script/worker ID. Launch receipts now contain all
200 worker PIDs and per-slot restart counters. The worker's injectable native
seam calls the shared `execute_raw_attempt`, `materialize_raw_attempt`, and
`write_closed_attempt_result` path; the existing CLI pilot fallback remains for
the already-provisioned RLBench environment.

Local evidence: focused control/raw tests 18/18 PASS; complete generation and
distributed suite 107/107 PASS; `scripts/validate_fast.py` 19/19 PASS;
Python compilation and `git diff --check` PASS. L3 simulator, HF publication,
and full quota launch remain NOT RUN in this repair turn.

## Second detailed field audit (2026-09-21)

The provenance/schema comparison against section 10 found and repaired two
additional omissions: held-out status is now copied into
`episode.provenance.held_out`, and pilot online observations are projected to
the four-field whitelist before validation. Crash/invalid pilot attempts now
retain numeric prefix arrays (`points` + offsets, poses, grip, actions), plan
identity and a non-outcome failure cause. Persisted execution sidecars use the
single canonical `outcome` field; `result_class` remains an internal runner
classification only.

The following are explicit NOT-PASS items for a future simulator gate, rather
than silently treated as covered:

1. `lighting_applied` and `camera_viewpoint_applied` now go through the shared
   simulator adapter, which fails closed when the expected VisionSensor/Light
   objects or setters are unavailable. A fresh L3 smoke must still verify the
   named RLBench objects and record the returned application receipt; local
   tests only cover the injected-object behavior.
2. `rho`, `nu`, and `epsilon` task-label arrays are now materialized from
   measured object states with explicit boolean masks. Geometric postconditions
   are labeled when observable; grasp/articulation predicates remain
   `valid=false` rather than being inferred from final success. A fresh L3
   receipt must confirm the measured-state coverage is sufficient for each
   authorized program.
3. `V3_PROTOCOL.layout_version == 3` identifies the v3 dataset tree, while the
   approved portable training sidecar remains `training_layout.LAYOUT_VERSION ==
   2` under the 2026-09-19 design. These are currently separate namespaces, but
   the operator documentation must keep that distinction explicit.

No full-generation launch is authorized by this audit alone. The above items
remain tracked until a fresh bounded simulator/publication receipt closes them.
