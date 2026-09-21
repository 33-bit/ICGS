# ICGS primary v3 distributed full-launch record

Status: **NOT LAUNCHED**

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
| v3 parity | `colab_v3_phase1_parity.py` | NOT RUN |
| 36 task build | `colab_v3_build_tasks.py --programs ...` | NOT RUN |
| 36-program strict smoke | `colab_v3_pilot_episodes_worker.py` | NOT RUN |
| 2-worker queue smoke | distributed launcher, publication disabled | NOT RUN |
| 200-worker bounded smoke | one job per worker, publication disabled | NOT RUN |
| Small real publication | verified HF commit/hash | NOT RUN |

## Live receipt

| Field | Value |
|---|---|
| Run ID | NOT RUN |
| Colab session ID | NOT RUN |
| code revision | `097bb5c` plus later launch commits, if any |
| approved manifest SHA256 | NOT RUN |
| PyRep revision | NOT RUN |
| RLBench revision | NOT RUN |
| CoppeliaSim SHA256 | NOT RUN |
| coordinator PID | NOT RUN |
| watchdog PID | NOT RUN |
| active worker slots | 0 |
| first HF revision | NOT RUN |
| last HF revision | NOT RUN |
| run status | `NOT_LAUNCHED` |

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

The current implementation still requires the fresh VM gates above. Full quota
generation must not start until all mandatory gates are recorded PASS.
