# 0018: RoboHiMan dataset layout and Stage-1 episode schema v2

Date: 2026-09-30
Status: accepted
Decision owner and approval evidence: project owner, storage/schema brief of
2026-09-30 ("Before full dataset generation, make the following storage/schema
fixes"); no episodes had been generated under the previous layout.

## Context

The round-3 layout wrote episodes to ad-hoc stores. Its IDs encoded run,
variation and strategy. The collector enforced the split only when asked to.
Several stored fields were ambiguous:
- `label_alpha` was a per-episode monitor index, not the context-conditioned
  ICGS alignment;
- `cmd_wall_s` looked like an action duration;
- canonical EE targets were written without derivation metadata;
- masks had no handle-to-object legend;
- labels had no uncertainty masks.

The frozen split ([ADR 0017](0017-icgs-stage1-split.md)) requires lineage to be
checkable per episode.

## Decision

**Layout.** One dataset root, `datasets/robohiman/` (git-ignored), with no
project, stage or version names in its paths:
- `manifest.json` — layout and schema versions, split id and hash, policies;
- `splits.json` — a byte copy of the locked split manifest;
- `preprocessing/` — TRAIN-only statistics with an episode-id hash;
- `reports/audit.json`, `reports/leakage.json`, `reports/reproducibility.json`
  and `reports/generation_log.jsonl`;
- `train/`, `dev/` and `test/`, each with `episodes/<id>/{manifest.json,arrays.npz}`,
  `attempts/`, `derived/<id>/<name>_v<k>.{npz,json}` and
  `views/{D_geom,D_dyn,D_task}.jsonl` + `manifest.json`.

**Episode IDs.** IDs are `ep-<task>-<run8>-<index>`, where `run8` is 8 random
hex characters per collector invocation. Variation, strategy, split, seed and
schema live only in the manifest. IDs are unique across the whole dataset.

**Immutability.** Raw episodes are written into a temporary sibling directory.
Files are made read-only and the directory is renamed into place. Later
computations go to `derived/` with these fields:
- `is_derived`;
- the derivation method and version;
- the source manifest and array hashes;
- conventions;
- the reconstruction error.

Derived products are never overwritten. Canonical EE targets are
`canonical_ee_actions_v1`. They use the world frame, metres and quaternion
xyzw. They are derived from native `cmd_arm_joint_target`, which is kept
unchanged.

**Split enforcement.** `Dataset.commit_episode` refuses an episode unless all
of the following agree with the frozen split and the dataset:
- the declared split and the destination directory;
- the split id and hash;
- the task, strategy and variation;
- the per-episode seed, which must be in the split range, and the factor seed;
- the reset RNG state, which must equal the state produced by the declared seed;
- the gradient flag, tracks and TEST role;
- withheld strategies, which are allowed only in TEST;
- the parent, which must be committed in the same split;
- the ID, which must be unique;
- every mask handle, which must appear in the legend.

The collector checks every episode seed before it starts the simulator.

**Schema `icgs_stage1_episode_v2`.**
- `timing`:
  - `physics_dt_s`;
  - `transition = "one physics step: (step[s], cmd[s]) -> step[s+1]"`;
  - `frame_stride`;
  - `model_dt_s = frame_stride * physics_dt_s`.
- Every `step_sim_time` interval must equal one physics step, within the
  float32 clock spacing.
- The name `prof_step_wall_s` replaces `cmd_wall_s` and is profiling only.
- `label_*` and `cmd_wall_s` names are rejected.
- `monitor_event_id`, `monitor_rho/nu/epsilon` and `monitor_first_occurrence`
  are required. Each monitor state has a `*_valid` mask. A mask is false within
  2 boundaries of a relation change. For `observable: false` events it is
  false for nu/rho, so no postcondition is fabricated. For the event id it is
  false when pending events are unordered.
- `cam_*`/`frame_*` arrays have one row per `frame_step` entry.
- When masks are recorded, `cameras.mask_legend` maps handle → object →
  instance → role/category.

**Views.**
- `D_dyn` rows are physics-step transitions. Model-cadence transitions group
  `frame_stride` rows.
- `D_geom` rows are primary only with persisted masks and a legend.
- Each `D_task` row gives:
  - the query episode, boundary and frame;
  - the context episode(s) and ordered context event tokens (event id and
    context boundary/frame);
  - `alignment_target`, which is the first token not yet achieved and equals
    `len(tokens)` for null;
  - `alignment_valid`;
  - rho/nu/epsilon re-indexed to the tokens, with their valid masks.
- The context is a successful episode of the same task, distinct from the
  query and with a different lineage root.
- In TEST, queries come only from outside the context seed range and contexts
  only from inside it.
- Pairing is deterministic given split id, query and pairing seed.

## Alternatives considered

- Keep per-run stores and enforce the split in a later audit. This was rejected
  because leakage would be detected only after it was written.
- Store alpha per episode. This was rejected because alignment depends on the
  context that the model sees.
- Replace `arrays.npz` with a chunked format. This was unnecessary at the
  planned scale, and the owner asked not to replace it.

## Consequences

- Scratch stores remain for gates and smoke checks (`--store`). Only
  `--dataset` output is Stage-1 data.
- Round-1–3 evidence episodes use schema v1 and are not readable by the v2
  validator. They stay as historical evidence and are not migrated.
- Views and reports are rebuilt per split with `scripts/robohiman_views.py`.

## Compatibility implications

- Dataset format: new layout and schema v2; there is no v1 data to migrate.
- Checkpoints, baseline and evaluation protocol: not affected (no training has
  consumed Stage-1 data).
- Split: unchanged (split-v1 hash is verified on every open).
