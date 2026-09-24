# ICGS generation schema and artifact inventory

Status: **documentation of the current implementation** (2026-09-24).

This document describes what the canonical generation runtime currently writes,
why each item exists, which consumer reads it, and whether it is part of the
published Hugging Face artifact. It is an inventory, not permission to delete or
change files. Any storage reduction that changes the serialized episode or the
required training layout needs a separate migration decision and fresh validation.

## Scope and source of truth

The phase-1 generation contract is defined by:

- protocol and quotas: `src/icgs/data/collection/generation/protocol.py`,
  `quota.py`, `batch.py`;
- episode schema: `src/icgs/data/schemas/episode_records.py` and
  `src/icgs/data/collection/generation/episode_record.py`;
- simulator materialization: `scripts/generation_episode_worker.py` and
  `src/icgs/data/collection/generation/rlbench_attempt.py`;
- training sidecar layout: `src/icgs/data/training_layout.py`;
- distributed queue/result validation: `scripts/generation_worker.py` and
  `src/icgs/data/collection/generation/distributed_validation.py`;
- Hugging Face publication: `src/icgs/data/collection/generation/distributed_publication.py`.

The distributed production path currently launches `scripts/generation_worker.py`;
that worker launches `scripts/generation_episode_worker.py` for each claimed job.
The `rlbench_attempt.py` writer is a separate reusable archive/materialization path
and should not be confused with the distributed subprocess writer.

## Frozen protocol identity

These values identify stored artifacts and must not be renamed as part of a storage
cleanup:

| Field | Current value | Meaning |
|---|---|---|
| `dataset_version` | `icgs-primary-v3` | Dataset identity |
| `episode_schema_version` | `icgs_episode_v2` | Semantic episode record schema |
| `program_manifest_version` | `3` | Approved program catalog identity |
| `composition_protocol_id` | `icgs-composition-primary-v2` | Composition identity |
| `controller_protocol_id` | `rlbench-timed-ik-v2` | Timed controller identity |
| `execution_mode` | `scripted_waypoint_v1` | Exactly one phase-1 execution mode |
| `camera_profile_id` | `rlbench-wrist-depth-v1` | Measured observation profile |
| `collection_seed` | `20260920` | Collection seed identity |

The protocol also declares `GenerationProtocol.layout_version = 3`. The current
training-layout module declares `LAYOUT_VERSION = 2` and writes that value into
`layout/layout_manifest.json`. This is an observed implementation discrepancy,
not silently resolved here; it requires an explicit compatibility decision before
changing either serialized value.

## End-to-end lifecycle

```text
AttemptPlan
  -> GenerationJob (queue/pending)
  -> worker claim (queue/claimed)
  -> simulator subprocess
  -> local result directory (ready)
  -> closed-result validation
       |-- invalid -> queue/quarantined + validation_failure.json
       `-- valid   -> queue/ingested
  -> HF publisher
       |-- data/receipt reconciliation
       `-- queue/published
```

The coordinator is the only validator, quota owner and HF publisher. Workers do
not receive HF credentials. `success` and `valid_failure` are episode records;
`simulator_crash` and `invalid_observation` are attempt records with
`episode_id = null` and never count toward the success quota.

## Episode artifact: successful or valid-failure result

The following tree describes the distributed writer after a valid physical run.
Some files are optional when a modality is unavailable; unavailable data must be
omitted, not filled with synthetic values.

```text
<result_dir>/
├── episode.json
├── execution.json
├── layout/
│   ├── episode.json
│   ├── result.json
│   ├── observations/
│   │   └── pointcloud/frames.npz
│   ├── robot/
│   │   ├── ee_pose.npy
│   │   ├── gripper.npy
│   │   └── [joint_state.npy when present]
│   ├── actions/actions.npy
│   ├── task/
│   │   ├── events.json
│   │   ├── collisions.jsonl
│   │   └── [rho/nu/epsilon*.npy when materialized]
│   └── layout_manifest.json
├── telemetry.npz
├── views/
│   ├── D_geom.json
│   ├── D_temporal.json
│   ├── D_dyn.json
│   └── D_task.json
└── artifact_manifest.json
```

### `episode.json` — canonical semantic record

Written by `_write_episode()` in `scripts/generation_episode_worker.py` using
`assemble_episode()`.

Top-level fields currently include:

| Field | Meaning |
|---|---|
| `schema_version` | Must be `icgs_episode_v2` |
| `provenance` | Immutable identity, split/subset, seeds, program/controller/camera IDs, outcome and intervention lineage |
| `online_observations` | Measured `T+1` boundary observations |
| `transitions` | Measured `T` commands and achieved durations |
| `dt` | `T` achieved durations |
| `robot_states` | `T+1` robot state snapshots when captured |
| `object_states` | `T+1` scene/object snapshots when captured |
| `snapshot` | Phase-2 reserved snapshot metadata; currently mostly null |
| `events`, task labels | Optional event/task annotations |
| `intervention`, `randomization` | Perturbation and scene-randomization provenance when applicable |

Each online observation contains the measured contract fields:

```text
points       [N, 3] float32 world-frame XYZ
point_valid  [N] boolean validity mask
T_w_e        [4, 4] end-effector-to-world transform
grip         0 closed / 1 open
```

The current distributed writer serializes these arrays into JSON lists. That is a
major storage cost and is one reason an episode can approach 1 GB in the capacity
probe. The JSON representation is semantically canonical today; replacing it with
an NPZ-backed representation would be a schema/archive migration, not a cleanup.

`episode.json` is consumed by `validate_episode()`, generation views, episode
archives and downstream dataset readers. It is therefore **training-relevant and
not removable without a migration**.

### `layout/` — versioned training/supervision sidecar

Written by `write_training_episode_layout()` in `src/icgs/data/training_layout.py`.
It materializes a portable file-oriented view:

| Path | Contents | Current role |
|---|---|---|
| `layout/episode.json` | Small identity/layout metadata | Training/debug metadata |
| `layout/result.json` | `success`, outcome and terminal reason | Training/result metadata |
| `layout/observations/pointcloud/frames.npz` | Ragged point clouds plus offsets | Layout consumer input; duplicates point clouds from root `episode.json` |
| `layout/robot/ee_pose.npy` | `T+1` end-effector poses | Layout consumer input; overlaps root observations/robot states |
| `layout/robot/gripper.npy` | `T+1` grip values | Layout consumer input; overlaps root observations |
| `layout/robot/joint_state.npy` | Joint positions/velocities when available | Optional simulator/world-model input |
| `layout/actions/actions.npy` | Pose/grip action vectors | Layout consumer input; overlaps root transitions |
| `layout/task/events.json` | Structured task events | Task-view input |
| `layout/task/collisions.jsonl` | Collision annotations | Debug/world-model/task input |
| `layout/task/*.npy` | Numeric task labels (`rho`, `nu`, `epsilon`, validity masks) | Task/dynamics input when present |
| `layout/layout_manifest.json` | Per-file SHA256/byte inventory and boundaries | Required integrity validation |

The current closed-result validator requires `layout/layout_manifest.json` and
validates the layout before ingestion. The historical schema-repair decision also
requires the repaired training layout. Therefore the layout is **currently a
required published contract**, even though several arrays duplicate root semantic
fields.

### `telemetry.npz` — auxiliary simulator telemetry

The distributed writer stores pose, grip and action arrays, plus auxiliary fields
depending on the writer path. The `rlbench_attempt.py` path additionally stores
joint positions/velocities, depth frames and masks. These data are useful for
debugging, replay diagnosis and future world-model work, but the current
`generation_views.py` pointer views do not read `telemetry.npz` directly.

Classification: **debug/future-consumer sidecar; not required by the current phase-1
pointer-view code**, but it is included in the current artifact inventory and must
not be removed from an active run without changing the artifact contract.

### `execution.json` — execution/debug receipt

Written by the distributed episode worker from the simulator row after removing
`_timed_obs` and `result_class`. It can retain internal actions, robot/object state,
plan, task labels, sensor-randomization and binding fields. It is not referenced by
the current training view readers or publisher logic except through generic file
inventory/publication.

Classification: **strongest current candidate for a debug-only/local file**. It can
be excluded from published training artifacts only after an explicit publication
profile change and validator/test update.

### `views/*.json` — pointer views

These contain pointer records for `D_geom`, `D_temporal`, `D_dyn` and `D_task`; they
do not copy observations. The HF publisher also writes compact prefix-level view
pointers. They are small but useful for dataset discovery and split/task sampling.

### `artifact_manifest.json` — immutable result inventory

Contains attempt/episode identity, outcome and a map of every other file to
`sha256` and byte count. `distributed_validation.py` requires it and checks that
the actual file set and hashes match exactly. **Required for ingestion, resume and
publication integrity; not redundant metadata in the current contract.**

## Attempt artifact: simulator crash or invalid observation

```text
<result_dir>/
├── attempt.json
├── valid_prefix.npz          # when any measured prefix is available
└── artifact_manifest.json
```

`attempt.json` contains the attempt identity, `outcome`, terminal/error fields,
scene/program provenance and null episode identity. `valid_prefix.npz` preserves
measured actions/poses (and, in the distributed pilot writer, point offsets and
grips) for diagnosis without mislabeling the result as a valid episode.
The artifact manifest is still required so quarantine/publication can verify the
attempt inventory. These records are **not training episodes** and do not count
toward quota success.

## Queue and run-control files

These files normally live under `<run_root>/queue` and `<run_root>/control`; they
are not episode data:

| Path | Meaning | Retention |
|---|---|---|
| `queue/pending/*.json` | Planned immutable jobs | Until claimed |
| `queue/claimed/<worker>/*` | Claimed job and lease metadata | Until ready/recovery |
| `queue/ready/<job>/` | Closed worker result awaiting validation | Until ingested/quarantined |
| `queue/ingested/<job>/` | Validated result awaiting HF publication | Until published |
| `queue/published/<job>/` | Published immutable local receipt/result | Resume/audit state |
| `queue/quarantined/<job>/validation_failure.json` | Malformed/invalid result diagnostic | Required failure evidence |
| `queue/heartbeats/*` | Worker/coordinator lease and heartbeat | Liveness/recovery |
| `run.json` | Run identity, manifest digest, runtime snapshot path and quota bound | Resume contract |
| `runtime_config.json` | Host/run configuration snapshot | Reproducibility |
| `launch.json` | Worker/coordinator/watchdog PIDs and restart counters | Operations/debug |
| `coordinator-heartbeat.json` | Phase, queue counts, planner progress and publication state | Operations/debug |
| `publication_receipt.json` | HF data/receipt commit state and hashes | Resume/reconciliation |
| `publication_manifest.json` | Local manifest sent to HF | Publication input |
| `resume_receipt.json` | Remote-resume/run progress summary | HF resume |

These control files are small relative to sensor artifacts. They should not be
counted as the cause of the multi-terabyte dataset estimate.

## Hugging Face published prefix

For a valid episode, `distributed_publication.py` uploads every file in the
validated `file_sha256` inventory below:

```text
<hf_subfolder>/episodes/<PROGRAM_ID>/<EPISODE_ID>/<all validated result files>
```

For a crash/invalid attempt it uses:

```text
<hf_subfolder>/attempts/<PROGRAM_ID>/<ATTEMPT_ID>/<all validated result files>
```

It also uploads prefix-level:

```text
dataset_manifest.json
resume_receipt.json
views/D_geom.json
views/D_temporal.json
views/D_dyn.json
views/D_task.json
publication_receipt.json
```

The publisher does not currently filter out `execution.json`, `telemetry.npz`, or
duplicate layout arrays; if they are in the validated inventory, they are uploaded.

## Training versus debug classification

| Component | Current status | Why |
|---|---|---|
| Root `episode.json` | Required semantic/training source | Read by validators, archives and episode views |
| Root `attempt.json` | Required only for crash/invalid audit | Excluded from training views |
| `layout/` arrays | Required by current repaired-result validator | Current layout contract; partially duplicates root record |
| `layout_manifest.json` | Required integrity receipt | Prevents incomplete/mutated layout ingestion |
| `artifact_manifest.json` | Required integrity receipt | Exact file inventory/hash fence |
| `views/*.json` | Discovery/pointer metadata | Small; no raw data duplication |
| `telemetry.npz` | Debug/future modality sidecar | Not read by current pointer views |
| `execution.json` | Debug/forensics sidecar | No current training consumer found |
| `valid_prefix.npz` | Failure diagnosis | Only for crash/invalid attempts |
| queue/control files | Operations/resume | Never training samples |

## Duplication and size findings

The largest likely duplication is point cloud storage:

1. root `episode.json` serializes every point as JSON;
2. `layout/observations/pointcloud/frames.npz` stores the same frames as compressed
   binary;
3. `execution.json` may retain internal state but excludes `_timed_obs` itself;
4. `telemetry.npz` adds overlapping pose/action/grip arrays and optional modalities.

The capacity probe measured roughly 0.92–0.93 GB per retained attempt, but did not
retain a component-level size breakdown. Therefore this report does **not** claim a
percentage saving for any proposed cleanup. The next safe measurement is a single
episode inventory that records `bytes` by top-level file before changing the writer.

## Safe reduction path (proposal, not applied)

1. Measure one complete success and one valid failure with a per-file size table.
2. Confirm actual training readers for root JSON versus `layout/` arrays.
3. Introduce an explicit publication profile with `debug_sidecars = false` for
   `execution.json` and `telemetry.npz`, while retaining them in local run storage.
4. Add a versioned layout/archive migration before removing duplicate point-cloud
   storage. Do not change `icgs_episode_v2` or silently replace JSON arrays.
5. Rerun closed-result validation, resume and one isolated HF publication batch.
6. Only then recalculate full storage/ETA and authorize production generation.

## Known limitations and open questions

- No retained episode from the TPU probe is available for exact per-file bytes.
- `execution.json` has no current consumer found by repository search, but its
  absence is not yet an accepted publication-profile rule.
- `telemetry.npz` may be needed by future world-model/debug workflows even though
  phase-1 pointer views do not read it.
- The root semantic JSON and training sidecar layout intentionally serve different
  contracts today; removing one is a migration, not ordinary file cleanup.
- The `GenerationProtocol.layout_version` versus `training_layout.LAYOUT_VERSION`
  discrepancy must be resolved by an explicit compatibility decision.

