# Instant Policy baseline identities

Current runtime is src/icgs. The historical source identity below is distinct from
published vv19 checkpoint profile. Old ip paths/commands in historical sections are
provenance, not current execution instructions. Use the current route at the end.

Identifier: `instant-policy-original-65dc94e`.
Source: [matdmiller/instant_policy at 65dc94e347df5bca4e390a6f959cd3308f8ae2bc](https://github.com/matdmiller/instant_policy/tree/65dc94e347df5bca4e390a6f959cd3308f8ae2bc).
Onboarding date: 2026-09-06. All 29 original local files matched that source before
harness edits. [upstream-files.sha256](upstream-files.sha256) identifies those bytes,
including the original README before its additive navigation section.

This identifies the original implementation, not a successfully reproduced paper
result. The paper-checkpoint config, weights, original data and measured results
were unavailable at onboarding. Historical constants below are deliberately
frozen documentation; live source may change later through the
[baseline-change policy](../RESEARCH.md). Do not update this identity silently.

## Complete source configuration

After modularization, the executable canonical composition is
`icgs.configuration.defaults.instant_policy_original()` with frozen sections;
old `ip.configs.base_config.config` imports were removed; explicit `to_legacy` supports native artifact conversion. The table
below records original values, not new profile defaults. An independent differential
test compares every field with the hash-verified original configuration.

Owner at baseline: [base_config.py](../../src/icgs/configuration/defaults.py).
The dictionary is Python with tensor values, not an executable YAML config.

| Key | Original value |
| --- | --- |
| record | False |
| save_dir | None |
| scene_encoder_path | ./checkpoints/scene_encoder.pt |
| pre_trained_encoder | True |
| freeze_encoder | True |
| save_every | 100000 |
| compile_models | False |
| local_num_freq | 10 |
| local_nn_dim | 512 |
| hidden_dim | 1024 |
| num_demos | 2 |
| randomise_num_demos | False |
| num_demos_test | 2 |
| traj_horizon | 10 |
| device | cuda |
| batch_size | 16 |
| batch_size_val | 1 |
| num_scenes_nodes | 16 |
| pre_horizon | 8 |
| pos_in_nodes | True |
| num_layers | 2 |
| lr | 1e-5 |
| weight_decay | 1e-2 |
| use_lr_scheduler | False |
| num_warmup_steps | 1000 |
| num_diffusion_iters_train | 100 |
| num_diffusion_iters_test | 8 |
| num_iters | 50000000001 |
| test_every | 50000 |
| randomize_g_prob | 0.1 |
| min_actions | float32 tensor: three -0.01 translations and three -π/60 axis-angle components |
| max_actions | float32 tensor: three +0.01 translations and three +π/60 axis-angle components |

Presence in config does not prove consumption: training hardcodes validation batch
one and validation interval 20000; `batch_size_val` and `test_every` do not
control those values in train.py. This enormous max-step default is not a smoke run.

## Entry-point configuration and artifacts

All relative runtime paths below are resolved from the working directory. Use
`ip/` for the existing documented quickstart after package/environment setup.

| Entry | Original behavior |
| --- | --- |
| Train from scratch | Base config; overrides record/save_dir; default record=0, use_wandb=0, run_name=test, save_path=./runs; data paths ./data/train and ./data/val |
| Fine-tune | ./checkpoints/config.pkl and model.pt by default; overrides compile_models=False, batch size (CLI default 16), save_dir and record; strict=True load mapped to config device; optional compilation afterwards |
| Evaluate | ./checkpoints/config.pkl and model.pt; overrides batch size 1, num_demos (CLI default 2), compile_models=False, inference steps 4; strict=True; graph reinit; optional compilation |
| Deploy example | Same checkpoint directory; overrides num_layers=2, batch size 1, demos=2, inference steps 4, compile_models=False; strict=False; graph reinit with at least one demo |

Scratch-mode CLI `--batch_size` and `--compile_models` do not override the base
configuration; those flags are applied in the fine-tuning branch. Fine-tuning is
weight initialization, not a verified optimizer/scheduler-state resume.

Trainer uses one device, 16-mixed precision, gradient clipping norm 1, validation
every 20000 batches, two sanity validation steps and logging every 500 steps.
When recording, config is saved to `<save_path>/<run_name>/config.pkl`; model
saves include `best.pt`, `last.pt`, periodic step files and optionally compiled
last checkpoints. Compiled save repair strips `_orig_mod.` from keys.

AGI ordinarily loads the separate scene encoder file during construction even
when a policy checkpoint is loaded. Preserve that artifact requirement; don't
assume model.pt alone is sufficient. Checkpoint files are Lightning-format saves,
not merely a raw AGI state dict. Strict=False in deployment is observed historical
behavior, not a compatibility guarantee.

## Model and data identity

GraphDiffusion wraps AGI; AGI composes the two-stage scene encoder, GraphRep, local/
conditioning/action heterogeneous transformers and translation/rotation/gripper
heads. Scene encoding is pretrained/frozen by default. Graph construction depends
on batch, demonstration, waypoint and prediction dimensions.

The exact original [policy data contract](../components/policy-data-contract.md)
documents point-cloud filtering, 2048-point sampling, six gripper geometry nodes,
edge direction/masks, frame transforms, action targets, normalization and serialized
attributes. Those semantics and source files at the pinned revision are part of
this baseline. Cached features are coupled to their encoder and preprocessing.

Preprocessing example defaults: two conditioning demos, ten waypoints, horizon
eight, translation spacing 0.01, rotation spacing three degrees, and cached
embeddings enabled on CUDA. However it passes `subsample=False` to sample_to_live,
so those spacing thresholds are not enforced by that call. The example contains
no actual demonstrations; it cannot reproduce the original training corpus.

## Evaluation identity

Default eval CLI: task `plate_out`, two demos, five rollouts, rotation restriction
enabled, compilation disabled. README explicitly demonstrates ten rollouts.
Neither specifies a complete paper benchmark or seed set.

rollout_model defaults to 30 replanning iterations; eval passes execution horizon
eight and checkpoint trajectory horizon. It launches non-headless RLBench with all
observations enabled, EndEffectorPoseViaIK plus Discrete gripper, custom linear arm
paths and a fixed initial arm joint vector. Rotation restriction clips the task's
z range around its midpoint ±π/3. The task mapping has 17 examples, not an asserted
paper benchmark task list.

Point clouds combine front/left-shoulder/right-shoulder cameras, select mask IDs
greater than 60, voxel-downsample at 0.01, then use shared filtering/subsampling.
Demonstrations are live simulator demos; the code does not set an explicit task
variation in the rollout loop. Success fraction is successful rollouts divided by
requested completed rollout count. Failure/retry and cleanup limitations are in
the [architecture](../ARCHITECTURE.md) and [audit](../audits/harness-onboarding.md).

## Environment and reproducibility status

Declared stack: Linux-oriented Python 3.10.15, PyTorch 2.2.0/CUDA 11.8, PyG 2.5.0,
Lightning 2.4.0, Diffusers 0.31.0, NumPy 1.26.4, SciPy 1.14.1, Open3D 0.18.0,
WandB 0.18.7. The environment also includes other CUDA packages/build-specific
pins; it is not proof of a freshly solvable installation.

[README](../../README.md) supplies setup and Google Drive weight-download guidance.
No download checksums, original dataset fingerprint, checkpoint config dump,
RLBench/PyRep/CoppeliaSim revisions, original seeds, expected scores or tolerances
were established. The seed helper is not called by train/eval/deployment.

ASSUMPTION: distributed weights may correspond to source defaults; unverified.
UNKNOWN: exact paper-training recipe and numerical reproduction.
Current validation and bounded opt-in commands are owned by
[tests/README](../../tests/README.md).

## Provenance check

From repository root, this read-only command prints each historical mismatch:

```bash
shasum -a 256 -c docs/baselines/upstream-files.sha256
```

The additive README guidance intentionally produces one mismatch after onboarding;
all other original files should match at onboarding completion. Later authorized
runtime changes also produce differences: inspect them against baseline policy,
do not regenerate the historical inventory to conceal drift. No checksum check
alone proves runtime behavior or checkpoint compatibility.

## Current published-native reproduction

Canonical source default is `icgs.configuration.defaults.instant_policy_original()`.
Published profile is `icgs.artifacts.published.published_config()`, packaged and
bound to the verified model hash. It changes only reference-evidenced settings:
pretrained auxiliary encoder IO off, batch1, steps4, live voxel .01 and sampling
before scene encoding. See [ADR0005](../decisions/0005-published-profile-fidelity.md).

The target artifact is downloaded from vv19's official script: file ID
1TM_zU1pVOqPuWZL3E9knNp4w-p7EBwwt, 471777552 bytes, SHA256
119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5.
No auxiliary scene_encoder.pt or config.pkl is needed. Native strict loading
maps all 674 alias entries to all 337 model-owned tensors; no missing/unexpected/
shape/conflict tolerance or random fill is used.

[README setup](../../README.md) provides exact commands. CLI is `icgs infer`
with absolute checkpoint/input/output paths, installed wheel and no old source
or reference binary in the import path. [CLI mapping](../components/cli-and-data.md)
covers train/evaluate/prepare-data. Old namespace compatibility is intentionally
removed under [ADR0004](../decisions/0004-unified-icgs-v5.md).

C1–C5 passed on Colab T4/CPython3.10.21/torch2.2.0+cu118/PyG2.5 for the supplied
two-demo synthetic fixture and seeds17/29/41. This proves scoped native execution/
fidelity, not task success or paper benchmark reproduction.
[Actual metrics/logs](../experiments/vv19-validation/README.md) record tolerances,
artifact/config identity, repeated contexts and installed execution.

Historical source defaults remain regression-tested, but they are not blindly
used as an oracle for published preprocessing/RNG behavior. Original old retry,
training and unsupported occupancy limitations remain documented. No retraining,
new neural v5 model or planner was implemented to pass these gates.
