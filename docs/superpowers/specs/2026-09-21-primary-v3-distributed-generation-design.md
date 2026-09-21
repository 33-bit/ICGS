# Primary v3 Distributed Generation Design

**Date:** 2026-09-21

**Status:** Approved in conversation; pending written-spec review

**Owner:** Repository owner / assigned implementer

## Goal

Generate the complete `icgs-primary-v3` phase-1 dataset on one Colab TPU
`v6e-1` VM using exactly 200 simulator workers and one commit coordinator. The
coordinator is the only process that plans attempts, owns quota state, updates
the canonical manifest, and publishes to the Hugging Face dataset
`33bit/icgs`. It publishes closed batches at most once every 300 seconds while
work is in progress and performs a final flush when all quotas are complete.

The launch must preserve every phase-1 invariant in
`docs/audits/2026-09-20-icgs-v3-generation.md`. In particular, a physical
`valid_failure` is a retained episode, not a skipped or quarantined crash.

## Scope

In scope:

- one persistent Colab `v6e-1` session;
- pinned CoppeliaSim 4.1, PyRep and RLBench setup;
- all 36 authorized primary-v3 programs;
- 200 concurrent simulator worker processes;
- one filesystem-backed coordinator and queue;
- exact nominal-success and perturbed-valid-attempt quotas;
- complete episode, attempt, layout, manifest, receipt and validation files;
- resumable, atomic Hugging Face publication every five minutes;
- watchdogs that restart failed worker processes without reusing an attempt;
- progress and terminal receipts sufficient to audit or resume the run.

Out of scope:

- phase-2 reference policy, recovery, branch, continuation and value data;
- training, preprocessing or model execution;
- changes to `primary_v2`;
- Google Drive as a required correctness or resume dependency;
- counting `simulator_crash` or `invalid_observation` toward a valid quota.

## Mandatory pre-launch gates

The coordinator must not create the full queue until all gates pass on the same
VM that will perform collection:

1. The checked-out repository revision is the requested committed revision and
   the source tree is clean.
2. `generation_parity_gate` passes for all 36 programs.
3. The v3 approved manifest has 36 `generation_authorized` rows and the frozen
   v3 protocol identities.
4. All 36 procedural task models build under the pinned simulator stack.
5. A fresh strict one-attempt-per-program simulator smoke has 36 retained
   episode outcomes (`success` or `valid_failure`), zero `simulator_crash`, zero
   `invalid_observation`, and zero SKIPPED, with valid `T` actions, `T+1`
   observations and `T` achieved durations. A `valid_failure` remains a real
   predicate failure and is never relabeled as success.
6. The known place-height gate is measured on the fresh smoke. Predicate
   failures outside `predicate_success_m = 0.01` must be classified and retained
   as `valid_failure`; a historical receipt cannot relabel a fresh outcome.
7. Hugging Face authentication succeeds and the existing `primary_v3`
   manifest, if any, validates before being used as resume input.
8. The remote `primary_v2` subtree is unchanged by a dry-run commit plan.

Failure of any gate leaves the session alive for diagnosis but launches zero
full-generation workers.

## Architecture

### Coordinator

The coordinator is the single authority for:

- one `AttemptPlanner` per program;
- all `QuotaCounts` and attempt caps;
- the global scene-signature and episode-ID uniqueness sets;
- the pending, claimed, ready, ingested and published job states;
- the canonical dataset manifest and resume receipt;
- Hugging Face credentials and API calls;
- worker heartbeat supervision and replacement;
- the 300-second publication clock.

The coordinator never runs a simulator. It maintains a bounded queue containing
at most 400 pending jobs so plans are created close to execution while all
in-flight plans remain registered in the uniqueness index.

### Workers

Exactly 200 persistent workers are launched with IDs `000` through `199`.
Each worker receives a deterministic X display number `200 + worker_id`; it
must not use `xvfb-run -a`, whose non-atomic display search failed under high
concurrency testing.

Workers do not receive the Hugging Face token and do not update shared quota or
manifest files. A worker atomically claims one job, launches an isolated
RLBench environment, executes exactly that precomputed attempt, writes a closed
result directory, shuts down the environment and claims another job. A worker
crash leaves its claimed job recoverable by the coordinator.

### Filesystem queue

The ephemeral run root is `/content/icgs-primary-v3-run`:

```text
control/
  run.json
  coordinator-heartbeat.json
  worker-heartbeats/worker-NNN.json
queue/
  pending/JOB.json
  claimed/worker-NNN/JOB.json
  ready/JOB/result.json
  ingested/JOB.json
  published/JOB.json
staging/
  episodes/EPISODE_ID/
  attempts/ATTEMPT_ID/
  views/
manifests/
  dataset_manifest.json
  resume_receipt.json
  publication_receipt.json
logs/
  coordinator.jsonl
  workers/worker-NNN.jsonl
```

All state transitions use same-filesystem atomic rename. A result becomes
eligible for ingestion only after the worker writes every artifact, fsyncs the
closed directory, and atomically publishes `result.json` into `queue/ready`.

## Job contract

Every job JSON contains:

- run ID and immutable code/manifest/protocol digests;
- `attempt_id`, planned `episode_id` when the attempt can become an episode,
  program, split and subset role;
- serialized `AttemptPlan`, including episode index, seed, randomization,
  scene signature, episode kind and optional single intervention;
- asset family/instance, source lineage and execution-mode identities;
- worker timeout and retry-generation number;
- expected output root and required artifact list.

The coordinator assigns globally unique attempt IDs and episode indices before
queue publication. A retried crash keeps the attempt identity for audit but is
not silently reused as a successful episode. A new physical attempt receives a
new attempt ID and the planner advances without forgetting the failed plan.

## Result classification and required files

### `success` and `valid_failure`

Both are valid `icgs_episode_v2` episodes and must contain:

- `episode.json` with full frozen provenance and all phase-2 keys set to null;
- safe numeric observation/action archive with no pickle requirement;
- online observations (`points`, `point_valid`, `T_w_e`, `grip`);
- boundary-aligned robot and object states;
- `T` executed transitions and achieved `dt`, with `T+1` observations;
- wrist RGB/depth/mask and calibration/extrinsic/intrinsic identities when
  measured by the configured observation profile;
- joint positions/velocities, end-effector poses and gripper states;
- event/task labels and pointer-layout sidecars for the four phase-1 views;
- result, checksums and per-file manifest.

A nominal `valid_failure` is retained but does not increment the nominal-success
quota. A perturbed `valid_failure` is retained and does increment the valid
perturbed-attempt quota. Neither may be discarded because its final predicate
is false.

### `simulator_crash` and `invalid_observation`

These are attempt records, not episodes. They must contain an `attempt_id`, null
`episode_id`, terminal classification, traceback/error details, valid prefix
metadata and any safely retained observation/action prefix. They never enter
training views and never count toward nominal or perturbed valid quotas.

## Work allocation and quota policy

The minimum valid workload is fixed by protocol:

- each T01–T20: 200 nominal successes plus 80 valid perturbed attempts;
- each V01–V04: 100 nominal successes plus 20 held-out perturbed attempts;
- each P/G/R program: 100 nominal successes plus 20 held-out perturbed
  attempts.

The expected minimum is 7,520 valid attempts; crashes, invalid observations and
nominal physical failures can increase the executed-attempt count.

Allocation is dynamic rather than a static 7,520/200 split. The coordinator
selects the next program by normalized quota deficit, with round-robin tie
breaking across train, development and test. For each program it follows
`next_episode_kind`: nominal attempts until the success target is met, then
perturbed attempts. It stops planning a program only when `quota_met` is true or
the declared attempt cap is reached. Hitting a cap is terminal `INCOMPLETE`, not
success.

All 200 workers draw from the same queue. This avoids per-worker planner resets,
duplicate scene signatures and quota drift.

## Coordinator ingestion

For every ready result, the coordinator:

1. verifies job/result identity and immutable digests;
2. validates outcome vocabulary and episode-versus-attempt rules;
3. validates timeline, schema, calibration and required-file checksums;
4. rejects duplicate scene signatures, episode IDs and asset-instance leakage;
5. records the outcome in the correct quota counter;
6. atomically updates the local manifest and resume receipt;
7. moves the job to `ingested` without deleting its staging files.

Validation failure creates a coordinator rejection record and leaves the
original worker output intact. It does not relabel an invalid result as a valid
failure.

## Hugging Face publication

The target repository is the dataset `33bit/icgs`; all new data lives below
`primary_v3/`. The coordinator reads `/content/.icgs_hf_token`; workers never
read or receive this path.

Every 300 seconds, when at least one ingested job is unpublished, the
coordinator creates one atomic commit containing:

- all closed episode and attempt directories since the previous successful
  commit;
- the canonical `dataset_manifest.json`;
- `resume_receipt.json` with per-program quota counts;
- the latest closed publication/run metadata;
- view pointer manifests that reference only already included episodes.

Before committing, the coordinator refetches the remote manifest and performs a
three-way identity merge. Conflicting bytes for an existing immutable path stop
publication. Rate-limit or transient network failures retain the pending batch
and retry on the next bounded backoff; they do not regenerate or drop data.

After each commit, the coordinator verifies the remote revision and hashes,
writes a local publication receipt, then marks jobs `published`. The final
quota-complete flush is immediate rather than waiting for the next five-minute
tick.

## Resume and recovery

On startup or coordinator restart:

- fetch and validate the remote `primary_v3` manifest;
- ingest local published and unpublished receipts;
- reconstruct quota counts and every planner's seen signature/index set;
- recover unexpired claimed jobs or requeue stale claimed jobs;
- preserve closed ready results without rerunning physics;
- refuse to continue if local and remote immutable identities conflict.

Worker watchdog replacement is bounded: the coordinator maintains exactly 200
worker slots while the run is active, but repeated failures of the same job are
limited by the protocol attempt cap and recorded explicitly.

## Keepalive and shutdown policy

The Colab CLI session keepalive remains active. Remote coordinator and worker
manager processes use `start_new_session=True`, heartbeat files and a lightweight
watchdog. The watchdog restarts a dead coordinator only after verifying that no
live coordinator lock owner exists.

The automation does not call `colab stop` after launch. Completion changes run
status to `COMPLETE`, performs the final commit and leaves the session allocated
and idle for owner inspection. A manual stop command is separate and explicit.

## Observability and receipts

The run must expose:

- active/target worker count;
- pending, claimed, ready, ingested and published counts;
- per-program nominal-success, nominal-failure, perturbed-valid, crash and
  invalid-observation counts;
- queue latency and episode wall time;
- coordinator and worker restart counts;
- last attempted and last successful HF commit timestamps/revisions;
- disk usage and remaining capacity;
- run status: `PREFLIGHT`, `RUNNING`, `INCOMPLETE`, `COMPLETE` or `FAILED`.

Receipts must never include credentials or environment-variable values.

## Validation strategy

Local tests must cover:

- atomic job claiming and stale-claim recovery;
- centralized uniqueness with multiple simulated workers;
- quota accounting for all four outcomes, including retained valid failures;
- deterministic dynamic workload allocation;
- coordinator restart and remote-manifest resume;
- five-minute commit scheduling and final flush;
- commit failure/retry without data loss;
- manifest conflict refusal;
- episode/attempt required-file and timeline validation;
- no credential access from the worker entry point.

VM validation proceeds in order:

1. compile/parity preflight;
2. fresh 36-program strict smoke;
3. 2-worker queue/coordinator smoke with publication disabled;
4. 200-worker bounded one-job smoke with publication disabled;
5. one small real publication batch and remote hash verification;
6. full quota launch with the five-minute coordinator enabled.

No SKIPPED gate counts as PASS. Full generation starts only after steps 1–5
produce closed PASS receipts.

## Remaining risks

- Colab can reclaim the VM despite keepalive; correctness therefore depends on
  frequent remote commits and deterministic resume, not on indefinite uptime.
- The token file is process-isolated by convention, not by a separate Unix user
  security boundary on Colab.
- 200 simultaneous simulators can create transient CPU and X-server pressure;
  deterministic displays remove the observed allocation race but do not remove
  VM resource limits.
- Historical 36/36 smoke evidence and the older place-height failure conflict;
  the mandatory fresh smoke is the authority for this launch.
