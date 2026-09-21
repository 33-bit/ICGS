# ICGS primary v3 distributed full-launch record

Status: **BLOCKED BEFORE FULL LAUNCH**

This record is updated only from fresh receipts produced by the committed
distributed coordinator. It does not treat a local unit test, an old simulator
receipt, or a skipped gate as evidence of a full collection run.

## Fixed launch contract

- Dataset: `33bit/icgs`, prefix `primary_v3/`
- Protocol: `icgs-primary-v3`, schema `icgs_episode_v2`
- Session: `icgs-primary-v3-full` (Colab TPU `v6e1`)
- Workers: exactly 200, displays `:200` through `:399`
- Coordinator: exactly 1, token isolated at `/content/.icgs_hf_token`
- Publication cadence: 300 seconds, final flush forced at quota completion
- Full quotas: T01–T20 `200 nominal successes + 80 valid perturbed`; V/P/G/R `100 nominal successes + 20 held-out valid perturbed`

## Fresh preflight evidence

| Gate | Command / receipt | Result |
|---|---|---|
| Clean committed source | `git status --porcelain=v1 -uall` | NOT RUN |
| v3 parity | `colab_v3_phase1_parity.py` | PASS: all_primary_match=true |
| 36 task build | `colab_v3_build_tasks.py --programs ...` | PASS: 36/36 built |
| 36-program strict smoke | `colab_v3_pilot_episodes_worker.py` | FAIL: 34 success, 2 valid_failure (`T03`, `T12`), 0 simulator crashes |
| 2-worker queue smoke | distributed launcher, publication disabled | NOT RUN |
| 200-worker bounded smoke | one job per worker, publication disabled | NOT RUN |
| Small real publication | verified HF commit/hash | NOT RUN |

## Live receipt

| Field | Value |
|---|---|
| Run ID | `icgs-primary-v3-run` |
| Colab session ID | `icgs-primary-v3-full` / `tpu-v6e1-s-kkb-euw4a0-3fnsv28mgt7tq` |
| code revision | worker/run identity `7891199`; coordinator publication hotfixes through `29157c1` |
| approved manifest SHA256 | `cf08ba141df8c6d131e88362ed6aba389af5bcfa64822956f54f30b33fa614b0` |
| PyRep revision | `8f420be8064b1970aae18a9cfbc978dfb15747ef` |
| RLBench revision | `02720bba4c73fe02eb75df946b8791b806028a9d` |
| CoppeliaSim SHA256 | `512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8` |
| coordinator PID | `511307` |
| watchdog PID | `48724` |
| active worker slots | 200 (`:200` through `:399`) |
| first HF revision | `09368256ff84c034063669bee900afb172dfe068` |
| last HF revision | `09368256ff84c034063669bee900afb172dfe068` at this checkpoint |
| run status | `RUNNING` |

## 2026-09-21 fresh VM gate result

The fresh v6e1 VM passed v3 parity and built all 36 task models. The strict
36-program smoke produced valid `T+1` timelines for every program, but did not
close the scale gate: T03 (contact push blocker) remained a `valid_failure` at
4.52 cm from its push target after three physical contact-push retries. T12
passed after the task models were rebuilt with child dynamics enabled.

The attempted hypotheses were recorded and rejected without changing the data
contract: gripper-open contact, target overshoot, and repeated contact pushes.
No tolerance increase, object snap, pick-place remap or failure relabeling was
accepted. Therefore the run launched zero full-generation workers, zero
coordinator publication commits and zero Hugging Face data commits.

The Colab session remains allocated for owner inspection and further bounded
push-controller debugging. Full launch remains blocked until a fresh 36/36
smoke has zero FAIL/SKIPPED and the T03 predicate is physically satisfied under
the frozen 1 cm predicate protocol.

## 2026-09-21 bounded CPU T03 debug

The failure was reproduced on the temporary CPU session `icgs-debug-cpu` with
the same v3 task model. A motion trace showed that commanding the tool tip to
the marker centre (`y=0.080 m`) placed the blocker centre at `y≈0.126 m`: the
open Panda contact face is approximately `0.045 m` ahead of the commanded tip.
The existing three retries then alternated the push direction and ended at
`4.52 cm` from the target.

The planner now exposes this measured geometry as
`V3_PROTOCOL.push_contact_offset_m = 0.045` and commands the contact point at
`target - contact_offset` while preserving physical open-contact push semantics.
The regression test asserts this offset without changing the 1 cm predicate or
remapping `push` to `pick_place`.

Bounded CPU verification: T03 returned `success`, `n_actions=137`,
`n_obs=138`, `timeline_ok=true`, and blocker-to-target distance `0.00224 m`.
The full v6e1 36-program gate has not yet been rerun; the scale gate therefore
remains blocked pending that fresh receipt.

## 2026-09-21 fresh v6e1 rerun

Session `icgs-primary-v3-full` was recreated as TPU `V6E1` from committed
revision `1ff4985`. Parity passed and all 36 procedural task models built. The
fresh one-attempt-per-program smoke completed with 32 `success`, 4
`valid_failure` (`T01`, `T06`, `V02`, `R3`), 0 simulator crashes, and valid
`T+1` timelines for all 36 programs. T03 now passes at `0.00224 m` from its
push target. The owner subsequently confirmed that a `valid_failure` is an
accepted retained episode outcome rather than a launch blocker. This receipt
therefore passes the revised gate because all 36 attempts produced retained
episodes with valid timelines and there were zero crashes, invalid observations,
or skips. Nominal `valid_failure` still does not count toward the nominal-success
quota.

## 2026-09-21 full launch and assignment recovery

The launch passed a two-worker smoke (2/2 retained successes) and a bounded
200-worker smoke (200/200 exit zero; 160 successes and 40 retained valid
failures). The full run then launched exactly 200 fixed-display worker slots,
one coordinator and one watchdog. A coordinator restart gap was repaired by
reconstructing planner state from pending/claimed/ready jobs and compact manifest
state from ingested/published closed results; publication batches are bounded so
heartbeats and commits remain observable.

The Colab CLI temporarily pruned the local name while the server assignment
remained alive. Server assignments were enumerated, the accidentally created
replacement endpoint was unassigned, and the original endpoint
`tpu-v6e1-s-kkb-euw4a0-3fnsv28mgt7tq` was rebound to
`icgs-primary-v3-full`. No full-run result was deleted. The first canonical HF
commit `09368256ff84c034063669bee900afb172dfe068` was verified at its revision:
200 compact manifest episodes, comprising 177 `success` and 23
`valid_failure`, with zero failure-attempt rows. Generation and scheduled
publication remain live.

## Required publication contents

Each verified commit must include closed `success` and `valid_failure` episode
directories, crash/invalid attempt records, `dataset_manifest.json`,
`resume_receipt.json`, view pointer manifests and a publication receipt. A
`valid_failure` must never be moved to crash quarantine merely because its final
predicate is false.

## Operator stop/resume policy

The coordinator and watchdog are detached with keepalive. They do not call
`colab stop`. A remote reclaim is recovered by starting the coordinator with the
same run root after validating the remote immutable manifest and planner state.
No worker receives the Hugging Face token.

## Remaining risk

The full run is paused/stopped for schema repair. No workers or coordinator are
currently active. A repaired run must start from a fresh VM and fresh bounded
receipts; the old ephemeral queue is not reusable as repaired data.

## 2026-09-21 schema/publishing pause

The first full-run publication exposed a schema defect: the worker's pilot
writer emitted only `episode.json` and `physics.json`, and the publisher placed
them below the internal `results/{job_id}/` path. The run was paused before any
further workers were launched. The repair now materializes the measured v2
training layout (`layout/observations`, `layout/robot`, `layout/actions`,
`layout/task`, `result.json`, `layout_manifest.json`), root
`artifact_manifest.json`, `telemetry.npz`, and `execution.json`; unavailable
RGB/depth/joint/object modalities remain omitted rather than synthesized.
Publisher paths are semantic `episodes/{PROGRAM_ID}/{EPISODE_ID}` or
`attempts/{PROGRAM_ID}/{ATTEMPT_ID}`. Closed-result validation now rejects an
episode without a valid layout or artifact manifest. The already published
legacy batch at `09368256...` was removed from the current HF tree in cleanup
commits `aa25a783...` and `7c96162e...`; the pre-repair bytes remain recoverable
only through HF history and are documented in `primary_v3/legacy_schema_cleanup.json`.
No new session or worker has been started after the pause.
