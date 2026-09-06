# Original Instant Policy baseline

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
`ip.configs.original.instant_policy_original()` with frozen sections;
`ip.configs.base_config.config` remains a legacy fresh-dictionary export. The table
below records original values, not new profile defaults. An independent differential
test compares every field with the hash-verified original configuration.

Owner at baseline: [base_config.py](../../ip/configs/base_config.py).
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

## Modular runtime reproduction route

The modularization intentionally changes implementation file bytes. Historical
checksums stay frozen and now report those migrations; they are not expected to
all pass on the refactored tree. The named source baseline remains
instant-policy-original-65dc94e and its canonical composition is
instant_policy_original. Numerical parity is bounded to the tests described in
[validation](../../tests/README.md), not a completed paper benchmark reproduction.

Setup is unchanged: use the original Linux-oriented environment.yml, matching
PyG/torch extensions and a separately provisioned RLBench/PyRep/CoppeliaSim stack;
install ip editable as in README. No dependency upgrades were made.

Dataset preparation still requires real demonstrations in prepare_data.py;
it is not a complete data-generation recipe. Implementation now lives in
ip.data.preprocessing and ip.data.dataset with old utils exports retained.
Use existing PyG data_N.pt files with original schema and trusted provenance.
Newer PyTorch's weights_only default is not silently patched in RunningDataset:
it can reject these files and trigger retries; use the declared stack until a
separate compatibility fix is approved.

From repository root after provisioning/installing, existing CLI paths remain:

```bash
cd ip
python train.py --run_name=baseline --record=1 --use_wandb=1 --fine_tune=0 --data_path_train=./data/train --data_path_val=./data/val
python eval.py --task_name=plate_out --num_demos=2 --num_rollouts=10 --restrict_rot=1 --compile_models=0
```

These are long-running research commands, NOT validation defaults. Supply actual
data and explicit compute scope; source max steps remain 50000000001. Fine-tuning
uses --fine_tune=1 and --model_path/--model_name as before; optimizer-resume
equivalence is not claimed.

Shared loading is ip.composition.load_policy. Eval preserves strict loading; deploy
preserves explicit non-strict mode with complete mismatch warnings. Loading prefers
saved resolved_config.json when present, otherwise trusted config.pkl. Registered
original leaf/alias names are preserved; explicit compiled/alias translations are
diagnosed. Actual published checkpoints were unavailable and compatibility remains
unverified. Generated fixtures do not replace that evidence.

Original encoder loading, evaluation rotation/mask/camera settings, sample order,
sampler timestep arithmetic and action-frame quirks are retained. See
[architecture](../ARCHITECTURE.md) and
[composition examples](../components/composition-examples.md) for current APIs.
