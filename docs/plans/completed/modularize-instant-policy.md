# Modularize Instant Policy — audit, design and migration plan

Status: completed (structural implementation and available validation); original-stack integration certification remains unclaimed.
Design approved by repository owner on 2026-09-06; implementation completed 2026-09-06.
Date: 2026-09-06. Owner: repository maintainer; implementation agent: Codex.
Task: Prompt 2, Refactor Instant Policy into a Modular Research Architecture.
This is category C under [WORKFLOW](../../WORKFLOW.md).

## Outcome and approval boundary

Preserve original Instant Policy as a named composition while making encoder,
graph builder, graph processing, action codec, objective and sampler replaceable.
Training, evaluation and deployment share construction and use public behavior.
No algorithm improvement is authorized. This document supplies the pre-runtime-edit
deliverables requested in the task; it is not an accepted architecture ADR.

The owner approved this design after the re-audit. Execute the phases below and
update this record. Current [ARCHITECTURE](../../ARCHITECTURE.md) will be updated
as implementation lands; this plan retains the pre-migration evidence.

## 1. Re-audit boundary and current tree

FACT: repository remains unborn main, without a commit or remote; all files are
untracked, including the completed harness. Existing files and .DS_Store metadata
are user-owned. Do not infer a clean tree from empty git diff.

FACT: all 28 intentionally unchanged originals still match the historical
[checksum inventory](../../baselines/upstream-files.sha256); root README has only
the earlier harness navigation addition. Runtime source identity remains
65dc94e347df5bca4e390a6f959cd3308f8ae2bc. Direct imports and relevant method bodies
were re-read, not inferred from the previous architecture document.

```text
ip/
  __init__.py
  configs/{__init__.py,base_config.py}
  models/
    __init__.py
    model.py                 AGI
    scene_encoder.py
    graph_rep.py
    graph_transformer.py
    diffusion.py             GraphDiffusion + training/inference infrastructure
    occupancy_net.py          stale occupancy-training path
  utils/
    __init__.py
    common_utils.py
    data_proc.py
    running_dataset.py
    normalizer.py
    repairs.py
    rl_bench_utils.py
    rl_bench_tasks.py
  train.py
  eval.py
  deployment.py
  prepare_data.py
  sandbox.py
  scripts/download_weights.sh
setup.py
environment.yml
README.md
AGENTS.md
.gitignore
media/rollout_roll.gif
scripts/validate_fast.py
tests/{README.md,test_harness.py,test_cpu_smoke.py}
.agents/skills/{onboard-repository,encode-invariant}/SKILL.md
docs/
  README.md, ARCHITECTURE.md, WORKFLOW.md, RESEARCH.md
  audits/harness-onboarding.md
  baselines/{instant_policy.md,upstream-files.sha256}
  components/policy-data-contract.md
  decisions/{0001-harness-boundary.md,TEMPLATE.md}
  experiments/TEMPLATE.md
  plans/TEMPLATE.md
  plans/completed/research-harness.md
  plans/active/modularize-instant-policy.md   this document
  third-party-notices.md
```

Applicable owners read: AGENTS, documentation map, workflow, architecture, research
policy, original baseline, semantic contract, validation guide, decision 0001 and
completed harness plan. Local onboard-repository guides this audit; encode-invariant
will guide accepted-rule tests. GitNexus is unavailable; AST import inventory and
direct source tracing were used instead. No download, install or simulator launch.

## 2. Current dependency and state map

Arrows mean depends on or calls; dotted conceptual ownership below is explained
in prose rather than inferred from folder names.

```text
train.py ──→ GraphDiffusion ──→ AGI
   │              │             ├─→ SceneEncoder ────→ common_utils
   │              │             ├─→ GraphRep ─────────→ common_utils
   │              │             ├─→ GraphTransformer ─→ PyG
   │              │             └─→ MLP heads / graph state / feature caching
   │              ├─→ Lightning / DDIMScheduler / Normalizer
   │              ├─→ common_utils ─→ torch + NumPy + SciPy + Open3D + PyG
   │              └─→ repairs / trainer checkpoint lifecycle
   ├─→ RunningDataset ─→ torch.load + NumPy/SciPy augmentation
   └─→ WandbLogger / argparse / config.pkl

eval.py ──→ GraphDiffusion + rollout_model ──→ RLBench/tasks + data_proc
deployment.py ──→ GraphDiffusion + data_proc + model.model feature helpers
prepare_data.py ──→ SceneEncoder + data_proc ──→ Open3D/PyG/torch.save
occupancy_net.py ──→ Lightning/WandB/dataset + missing visualiser
```

Direct-import FACT: AGI, SceneEncoder, GraphRep and GraphTransformer do not import
RLBench. Do not describe this refactor as removing an existing direct RLBench
import from those modules. The leakage is evaluator knowledge of model internals,
environment-specific processing mixed with rollout control, and costly indirect
utility imports. models is not uniformly reusable core: GraphDiffusion and
occupancy_net contain training infrastructure.

### Shared state and implicit coupling

- AGI constructs and owns every encoder, graph, transformer and head directly.
  Model init creates/loads encoder weights and may compile individual internals.
- AGI and GraphRep store batch/demo/horizon/device state; reinit_graphs mutates it.
  GraphDiffusion validation changes it and later restores training dimensions.
- GraphDiffusion stores the same modules under self.model.* and top-level aliases
  graph_rep, scene_encoder, local_encoder, cond_encoder, action_encoder,
  action_head_trans, action_head_rot and action_head_grip. Those are registrations,
  not mere descriptions. A local torch alias probe emits duplicate key prefixes;
  actual checkpoint contents are still unavailable.
- Training/test_step overwrite data.actions, actions_grip and diff_time.
  AGI caches attributes in the input, temporarily replaces pos_obs/batch_pos_obs,
  and writes action-conditioned scene features. A reusable caller cannot safely
  infer immutability or concurrent inference.
- Model and graph consume the same large mutable config alongside trainer options.
  CLI behavior differs between scratch/fine-tune and eval/deploy.
- Data shapes implicitly couple 2048 cloud points, 16 scene nodes, 512 local width,
  10 demo waypoints, six gripper points and horizon eight. Layer/head dimensions
  and graph metadata must agree. Cached features carry no encoder fingerprint.
- Evaluation computes demo embeddings on the first replanning step of each rollout.
  Moving caching earlier or retaining it across all rollouts can change FPS RNG
  consumption and therefore is not automatically behavior-preserving.
- CWD-relative ./checkpoints, config.pkl and separate encoder weights are implicit
  initialization inputs. There is no general resolved-config manifest.
- RLBench owns mask>60, camera set, arm-path monkeypatch, start joints, restricted
  rotation range and terminate/reward success semantics; none is generic policy math.

Sources: [AGI](https://github.com/33-bit/ICGS/blob/32e177f92b2c5611d30a62b4c9de65e1ba04f965/ip/models/model.py),
[GraphDiffusion](https://github.com/33-bit/ICGS/blob/32e177f92b2c5611d30a62b4c9de65e1ba04f965/ip/models/diffusion.py),
[GraphRep](https://github.com/33-bit/ICGS/blob/32e177f92b2c5611d30a62b4c9de65e1ba04f965/ip/models/graph_rep.py),
[preprocessing](https://github.com/33-bit/ICGS/blob/32e177f92b2c5611d30a62b4c9de65e1ba04f965/ip/utils/data_proc.py),
[rollout](https://github.com/33-bit/ICGS/blob/32e177f92b2c5611d30a62b4c9de65e1ba04f965/ip/utils/rl_bench_utils.py).

## 3. Existing execution flows

### Training

base_config dictionary OR trusted checkpoint config.pkl → branch-specific overrides
→ GraphDiffusion construction/weight load → AGI components/encoder checkpoint/graph
→ RunningDataset for preprocessed data_N.pt → PyG loaders → Lightning Trainer.fit
→ timestep/noise generation → noisy graph input + relative point labels → L1 loss
→ AdamW → validation inference/pose metrics → periodic/best/last checkpoints.

Offline raw-demo conversion happens BEFORE train.py, not inside its loader.
Validation batch size is hardcoded one. Training uses eight loader workers,
drop_last, shuffle, mixed precision and gradient clipping. fine_tune loads model
weights; it is not a verified trainer/optimizer resume.

### Evaluation

trusted config.pkl → eval overrides (batch1, demos CLI, inference steps4)
→ strict GraphDiffusion checkpoint load → graph reinit/eval/optional compilation
→ RLBench launch/patches/live demo collection → reset/observation
→ segmented world cloud → local cloud/PyG sample → demo/live scene embeddings
→ model.test_step → relative transforms/grips → world poses/xyzw/RLBench action
→ task.step → termination/reward success fraction → normal-path shutdown.

### Inference/deployment

User-supplied world cloud, T_w_e and grip + demos → shared point-cloud/waypoint
processing → PyG data → cached/demo/live embedding calls → GraphDiffusion.test_step
→ [B,P,4,4] relative transforms and [B,P,1] grips → controller-specific execution.
deployment.py loads strict=False and forces num_layers=2; it is an incomplete
example, not a tested robot application.

## 4. Ranked architectural problems

| Severity | Concrete coupling | Prevented research mistake / value of change |
| --- | --- | --- |
| HIGH | AGI constructs encoder, graph, stages and heads | Encoder/backbone ablations currently require editing model wiring |
| HIGH | GraphDiffusion combines algorithm, inference and Lightning | Sampler/objective replacement unnecessarily touches training infrastructure |
| HIGH | Evaluation/deploy call model.model helpers and manage caches | Alternative policies must reproduce AGI internals |
| HIGH | Mutable shared config and branch-specific overrides | Unclear experiment identity and accidental baseline mutation |
| HIGH | Module aliases and implicit checkpoint config | Seemingly structural changes may break pretrained loading |
| HIGH | Hidden data/graph/frame contracts | Shape-compatible replacement may still change scientific semantics |
| MEDIUM | common_utils eagerly imports Open3D/PyG for geometry | Cheap action/core tests inherit unrelated dependencies |
| MEDIUM | Environment conversion, patching and metric loop share one function | Second environment integration requires copying policy plumbing |
| MEDIUM | Model-owned graph/cache/scheduler state | Reentrancy and cache validity are implicit |
| MEDIUM | Generic utils owns preprocessing/storage/action semantics | Ownership and replacement impact are hard to find |
| LOW | Flat ip namespace and historical names | Compatibility cost exceeds benefit of wholesale rename |
| LOW | Unused imports, naming, formatting | Address only when needed for dependency isolation |

Existing defects are recorded, not algorithm work: logger unassigned in one
recording branch; infinite dataset/reset retries; missing occupancy visualiser and
unsupported SceneEncoder argument; example-only preprocessing/deployment; unusual
frame/time-embedding/sampler behavior. Do not bundle their correction into this
architecture change. If an infrastructure extraction cannot avoid changing a
failing branch, isolate and report the intended behavior decision before editing it.

## 5. Baseline invariants

Authority: the owner's Prompt 2 plus existing
[research policy](../../RESEARCH.md) and
[original baseline](../../baselines/instant_policy.md).

1. Named original composition and complete source configuration remain available.
2. Preserve original math and operation ordering: graph nodes/edges/masks and node
   order, point sampling and waypoint rules, action targets/padding, grip mapping,
   normalization, labels, losses, DDIM schedule/loop, SVD and composition/clipping.
3. Preserve world/end-effector frame conventions, quaternion ordering and
   current-observation-relative action chunks; do not chain actions.
4. Preserve observed R.T@points-t action-scene transform, integer sampler timestep
   indexing and chained time-embedding assignment until separately approved.
5. Preserve model initialization and RNG call order where feasible; do not move
   extra random initializers/cached FPS calls across baseline boundaries.
6. Preserve old PyG sample field names (including graps_demos), serialization
   schema, cached/raw behavior, augmentation and train/val selection semantics.
7. Preserve eval protocol: camera/mask/voxel processing, live demos/reset/variation
   handling, arm patches/joints, rotation restriction, action chunk and metric.
8. Preserve existing checkpoint loading routes where possible; never silently
   discard, overwrite or ambiguously remap parameters.
9. Preserve original CLI working-directory assumptions and default profiles.
   New explicit config overrides are additional behavior, not changed defaults.
10. Runtime remains independent of harness files under ADR0001.
11. Freeze original provenance hashes; record changed runtime files, never regenerate
    historical hashes to conceal migration.

If exact preservation cannot be established, record UNKNOWN or a blocked
acceptance criterion before proceeding across that boundary. A CPU mock-pipeline
pass cannot certify original GPU policy parity or paper reproduction.

## 6. Alternatives and package decision

Recommended: incremental extraction inside ip with explicit typed composition and
legacy import adapters. Keep coherent existing implementations, remove constructor/
infrastructure coupling and add narrow public contracts.

Alternative: rename into root instant_policy/. More descriptive import namespace,
but creates broad import and serialized-path migration unrelated to replaceability.
Alternative: src/instant_policy/. Better installed-package-only test isolation,
but same namespace migration plus build/test changes and CWD/compatibility work.

Choose existing root ip for this migration: already a valid setuptools namespace,
least checkpoint/import disruption, editable workflow unchanged. Validate installed
imports in a provisioned environment and avoid from-src imports. This does not rule
out a separately justified package rename later.

## 7. Proposed boundaries

B=batch, D=demos, T=waypoints, P=prediction horizon, S=scene nodes, E=feature width.
Names below are proposed APIs, not existing symbols.

| Boundary / location | Responsibility and input → output | Why / replacement contract | Construction and allowed dependencies |
| --- | --- | --- | --- |
| Observation, ActionTrajectory, PreparedContext in ip/types.py | World XYZ/T_w_e/grip; relative trajectory/grips; episode-local demo cache | Public semantic types, not wrappers for every tensor dict | stdlib typing/dataclasses; concrete tensor/array values; no environment classes |
| Structured configuration in ip/configs/structured.py | Validated sections/selectors → independent resolved experiment config | Explicit identities and overrides; no shared mutable tensor defaults | stdlib; conversion from legacy tensor config at compatibility edge |
| SceneEncoder (existing module) | (features or None, points[N,3], membership[N]) → (features[M,E], positions[M,3], membership[M]) | Replaceable nn.Module; preserve geometry/feature semantics for baseline graph | composition creates from SceneConfig; torch/PyG/positional helpers only |
| Conditioning functions in ip/policy.py | Demos + observation + encoder → prepared context/model-ready sample | Own cache lifecycle and hide AGI helpers; no speculative DemoEncoder class | policy session owns cache; data preprocessing and core components |
| GraphRep (existing module) | Dimension/context/action features → HeteroData + action-node selection metadata | Replaceable graph builder; learned feature parameters remain here | GraphConfig; torch/PyG/positional helpers; no trainer/environment |
| Graph stage bundle in ip/models/backbone.py | Graph dictionaries → latent gripper features | Replace all three stages or one stage if metadata compatible | Explicit factory builds modules; small execution helper, not a universal graph API |
| GraphDenoiser in ip/models/denoiser.py | Model-ready noisy batch → [B,P,6,7] predictions | Composition of supplied encoder/graph/stages/heads; no constructor checkpoint IO | Narrow model dimensions + supplied components; torch/PyG/geometry |
| OriginalActionCodec in ip/actions.py | Relative transforms ↔ normalized model state; node labels/recovery and decoded action trajectory | Own action representation and label geometry; replaceable within compatible algorithm family | ActionConfig + gripper geometry; torch and pure geometry |
| OriginalDiffusionObjective in ip/algorithms/diffusion.py | Batch/network/schedule/codec → loss and diagnostics | Separates objective from Lightning; future algorithms may replace this too | Noise/training sections; torch/Diffusers; no logger |
| OriginalDiffusionSampler in same module | Denoiser + condition + codec/schedule → ActionTrajectory | Replace sampling independently within declared state/denoiser contract | SamplingConfig; explicit instance state, torch/Diffusers |
| InstantPolicy in ip/policy.py | prepare_context; predict(observation, context) → ActionTrajectory | Stable callable low-level policy, reusable by evaluator or later planner | Plain facade around network/codec/sampler; no Lightning/RLBench/WandB |
| Data functions / RunningDataset in ip/data/ | Raw samples ↔ existing PyG storage/model fields, preprocessing and augmentation | Separate IO from generic observation/action contracts; retain serialized format | torch/PyG/Open3D/SciPy only where actually needed |
| RLBenchAdapter in ip/environments/rlbench.py | RLBench demos/observations/step → core Observation + outcome; ActionTrajectory → existing command | Isolate simulator classes, masks, patches and environment actions | Outer integration imports RLBench, data/geometry/types; never constructs model |
| evaluate_policy in ip/evaluation.py | Policy + narrow environment behavior → success metric | No model.model/test_step internals; supports test environment/policy | Structural callable protocol; no direct RLBench or concrete policy construction |
| Lightning training adapter in ip/training.py | Built policy/objective + batches → optimizer/logging/checkpoint lifecycle | Infrastructure depends on algorithm, not reverse | Lightning/WandB at run setup, dataset and composition results |
| Build/load functions in ip/composition.py and ip/checkpoints.py | Resolved config + explicit component overrides/artifacts → network/policy/training bundle | Single obvious wiring path and diagnostic checkpoint boundary | Outer composition may import runtime components; lazy trainer construction |

No abstract base class for every boundary. Use structural protocols only at public
replacement seams (scene encoder, compatible graph builder, sampler and evaluated
policy/environment), concrete dataclasses for actual semantic state, functions
elsewhere. Keep graph parameter owners distinct from stateless execution helpers.

Original scene representation is local geometry, not arbitrary semantics. A semantic
encoder may reuse the old graph only if it supplies compatible positions, cardinality
and features. Incompatible representation changes require a matched graph/denoiser
composition and a declared research experiment.

## 8. Target dependency direction and state ownership

```text
CLI train / eval / deployment / prepare_data
       ↓
composition + checkpoint IO + training/evaluation orchestration
       ↓                         ↓
policy facade                 environment adapter
       ↓                         ↓
denoiser + objective/sampler   Observation / ActionTrajectory
       ↓
scene encoder + graph builder + graph stages + action codec
       ↓
core geometry / positional helpers / semantic types / narrow config
```

Proposed core set: ip/types.py, geometry.py, actions.py, policy.py,
algorithms/, configs/structured.py, configs/original.py, and actual reusable model
modules denoiser/scene_encoder/graph_rep/graph_transformer/backbone/embeddings.
No core imports RLBench, PyRep, WandB, Lightning, argparse, entry scripts, training,
evaluation, checkpoint IO or composition. Data preprocessing imported by policy
may use Open3D, but pure model/action geometry must not transitively import it.

Compatibility modules model.py/diffusion.py and stale occupancy_net.py are explicitly
outside the reusable core set. They cannot be imported from the new core. All old
paths remain adapters/re-exports, not a duplicate implementation.

Network owns learned modules once for ordinary policy use. A Lightning adapter
registers that network under self.model and retains historical aliases if needed
to preserve its state dict. The policy facade is not another nn.Module parent
that introduces policy.model.* keys. Stage execution helpers do not re-register
the three modules under a new backbone.* prefix.

Each policy session owns its scheduler and graph/cache state; no process-global
registry or mutable baseline object. PreparedContext is explicitly episode-local.
Reset/recompute its cache at original rollout boundaries and reject accidental
cross-policy cache use. Thread-safe concurrent sharing is not promised in this task.

## 9. Proposed target tree and migration map

Only directories containing real implementations are created. Existing files not
shown as moved remain, including setup/environment/assets and historical docs.

```text
ip/
  configs/{base_config.py,structured.py,original.py}
  types.py
  geometry.py
  actions.py
  composition.py
  checkpoints.py
  policy.py
  training.py
  evaluation.py
  algorithms/{__init__.py,diffusion.py}
  models/
    denoiser.py, backbone.py, embeddings.py
    scene_encoder.py, graph_rep.py, graph_transformer.py
    model.py, diffusion.py              legacy entry adapters
    occupancy_net.py                    untouched, documented stale route
  data/{__init__.py,preprocessing.py,dataset.py}
  environments/{__init__.py,rlbench.py}
  utils/                               compatibility exports + remaining diagnostics
  train.py, eval.py, deployment.py, prepare_data.py
  sandbox.py, scripts/download_weights.sh
tests/
  existing tests
  test_config.py, test_architecture.py, test_geometry.py
  test_composition.py, test_policy.py, test_checkpoints.py
  test_model_integration.py
docs/
  updated architecture/research/baseline/semantic/validation owners
  decisions/0002-runtime-composition.md
  decisions/0003-checkpoint-compatibility.md
  plans/active/modularize-instant-policy.md
```

| Old file / behavior | New owner | Reason |
| --- | --- | --- |
| configs/base_config.py mutable global | structured.py + original.py; base_config compatibility export | Immutable config and explicit legacy conversion |
| models/model.py construction | composition.py | Central wiring and component factories |
| models/model.py forward/feature work | models/denoiser.py; context orchestration in policy.py | Inject components; preserve forward math |
| models/model.py stage execution | models/backbone.py | Replaceable stages without renaming leaf parameters |
| models/model.py label/action helpers | actions.py | Consolidate action semantic ownership |
| models/scene_encoder.py | Keep | Already coherent; change dependency/config wiring only |
| models/graph_rep.py | Keep | Preserve graph math; narrow config and explicit build contract |
| models/graph_transformer.py | Keep | Already separates GNN processing from graph construction |
| models/diffusion.py objective/sampling | algorithms/diffusion.py | Remove Lightning coupling from generative algorithm |
| models/diffusion.py callbacks/optimizer/logging | training.py; legacy import adapter retained | Preserve Lightning checkpoint route without policy→trainer dependency |
| utils/common_utils.py math/positional encoding | geometry.py / models/embeddings.py | Pure tensor/geometry imports; preserve formulas |
| utils/common_utils.py point-cloud downsampling | data/preprocessing.py | Open3D stays at processing boundary |
| utils/common_utils.py diagnostics/legacy names | Keep functions/re-exports | Avoid unnecessary churn and preserve callers |
| utils/normalizer.py | actions.py; legacy export | One action representation owner |
| utils/data_proc.py | data/preprocessing.py; legacy exports | Explicit domain ownership, same serialized data |
| utils/running_dataset.py | data/dataset.py; legacy export | Explicit IO/augmentation owner, no retry-policy change |
| utils/rl_bench_utils.py | environments/rlbench.py + evaluation.py; legacy rollout adapter | Environment conversion separated from generic policy execution |
| utils/rl_bench_tasks.py | environments/rlbench.py mapping; legacy export | Task classes remain in integration layer |
| utils/repairs.py | checkpoints.py; legacy exports | Explicit key translation/diagnostics and no silent loss |
| train.py / eval.py / deployment.py | Keep entry paths, delegate construction/run setup | Preserve researcher commands, remove duplicated internal calls |
| prepare_data.py encoder construction | composition encoder factory | Consistent encoder settings without inventing data collection |
| occupancy_net.py | Keep excluded stale legacy route | Unrelated pretraining repair is not this migration |

## 10. Configuration and component construction

Choose frozen dataclasses plus plain Python baseline/derived config. This uses no
Hydra/OmegaConf dependency, file auto-discovery, import-by-string plugin loader or
new YAML language. Dataclasses make runtime type/shape validation explicit; YAML
can be reconsidered only when cross-language/static-file configuration is needed.

Sections: scene, graph, backbone, action, diffusion/objective, sampling, runtime,
training and evaluation. Tensor-valued action limits become immutable numeric
tuples in config and fresh tensors at construction. Do not freeze tensors in a
mutable nested config and call that immutable. Each component receives only its
section plus explicitly needed dimension contracts, not the entire experiment.

Canonical function: instant_policy_original() → fresh ExperimentConfig.
Default values reproduce the full source config. Evaluation/deployment are
explicit profiles preserving their existing overrides rather than overwriting the
canonical source baseline. A legacy config conversion reads each known value,
retains recognized unused historical metadata, and diagnoses unknown keys; it must
not silently fall back to defaults for a malformed checkpoint config.

Override order: canonical source OR trusted checkpoint config → explicit entry
profile → explicitly supplied overrides. Log/serialize resolved sections and
component identities; distinguish unspecified flags from default-valued flags.
Old CLI behavior is preserved when no new overrides are supplied.

Construction functions:
- build_components(config, factories=None) constructs in the original order;
- build_policy(config, factories=None) constructs the core policy once;
- build_training_module(config, factories=None) wraps the same network/objective;
- checkpoint load feeds the same config and component construction path.

Factories are an explicit immutable mapping/record passed by the caller, seeded
with named original implementations. No mutable module-global plugin registration.
Component selectors must resolve or fail with an actionable error. Injected test
components are legitimate interface implementations, never sys.modules fakes.

AGI(config) at its historical path remains a compatibility adapter that converts
legacy config and requests the shared components. New GraphDenoiser never imports
that adapter or the composition layer. Baseline parameter initialization order,
including original graph parameters and heads, is tested before claiming parity.
Scene checkpoint IO/freeze/compilation happens in composition after/between the
same original initialization steps; no leaf constructor downloads/loads weights.

## 11. Checkpoint compatibility strategy

Keep old ip import paths available. Preserve AGI leaf attribute names and
Lightning self.model.* plus original alias registration where feasible.
Do not drop apparently unused learned graph parameters: they may be in artifacts.

A pure key-normalization/diagnostic function accepts a state mapping and expected
model state. It reports:
- original and translated key names;
- missing/unexpected keys;
- tensor-shape mismatch;
- colliding aliases with unequal values;
- unsupported schema/config/component identity.

Treat _orig_mod as a literal path segment, not a wildcard regex. Recognize only
explicit mappings with fixtures. If aliases collapse to one destination, verify
equality before merging; fail on conflict instead of last-key-wins. A strict load
is default. Legacy deployment's non-strict mode remains explicit and emits the
full mismatch report; it must never silently discard weights. Shape mismatches
remain failures unless a separately approved migration defines a transformation.

For baseline training saves, retain compatible state registration and legacy
config.pkl where required; include additive versioned resolved-composition metadata.
Do not imply arbitrary whole-object pickle or optimizer-resume compatibility from
a successful weights-only load. New load helpers do not rewrite input checkpoints.

Cheap tests use generated real torch modules/tensors with independent expected key
maps and representative legacy prefix fixtures. Those tests prove mapping behavior,
NOT compatibility with an unseen published checkpoint. L2 is needed for that claim.

## 12. Validation strategy and acceptance gates

Existing L0/L1 tests remain. Add tests before each extraction; use a bounded
pre-refactor source snapshot outside runtime as a differential reference when
dependencies are available. No permanent duplicate baseline code in the package.
Use focused numeric fixtures/golden metadata where their provenance can be recorded.

| Proof | Tier / environment | Acceptance |
| --- | --- | --- |
| Config is fresh, immutable, deterministic; legacy/entry-profile precedence | L1, stdlib/torch for conversion | Required locally |
| Internal dependency graph and forbidden imports, including package initializers | L0, AST | Required locally; allowed and direct/transitive forbidden fixtures |
| Pure geometry/action/normalizer outputs and padding/labels | L1, CPU torch/NumPy/SciPy as needed | Required for moved math |
| Policy/scene/graph/sampler replacement using small real contract implementations | L1, CPU; PyG only in graph-specific tests | Required runnable test proves meaningful replacement |
| Two policies do not share config/scheduler/cache; context reset follows episode semantics | L1 | Required locally |
| Train/eval/deploy route to shared construction; evaluator accepts synthetic environment/policy | L1; trainer integration conditional on Lightning | Required public path tests, not source-line greps |
| Legacy prefixes, compiled segments, conflicts, missing/unexpected/shape mismatches | L1, torch | Required generated-checkpoint proof |
| Real graph metadata/edges/features, denoiser forward and backward, original-vs-new RNG-controlled outputs | L2, actual PyG/extensions and research stack | Required before claiming original-model numerical equivalence |
| Real original encoder/checkpoint load and policy inference | L2, trusted assets/research stack | Required before claiming published-checkpoint compatibility |
| RLBench adapter and rollout behavior | L3, simulator | Required before claiming environment parity |
| Multi-seed research result | L4, matched protocol | Not an acceptance claim of this refactor |

Never monkeypatch missing dependencies into fake “passing” imports. Small injected
components test composition only, and are labeled as such. Configuration builds
without weights is not real model construction. Selecting pre_trained_encoder=False
or reduced dimensions in a test is an explicit test configuration, not baseline.

Current host lacks torch_cluster, Lightning, Diffusers, Open3D, RLBench and PyRep;
CUDA and model/data assets are absent. Torch/PyG package presence is not extension
compatibility. No dependency upgrade or download is implicit. Implementation can
establish local composition/architecture tests, but unavailable integration checks
must remain SKIPPED and the corresponding claims withheld. A failed available
baseline-parity test stops that phase; it cannot be relabeled SKIPPED.

Use [encode-invariant](../../../.agents/skills/encode-invariant/SKILL.md) for accepted
new core-dependency rules after design approval. Keep local command, optional hook,
CI invocation and branch protection reporting separate; do not add CI or hooks.

## 13. Future extension thought experiments

| Scenario | Minimal extension / touched owner | What stays intact / limitation |
| --- | --- | --- |
| A replace scene encoder | New encoder implementation + explicit factory/config in experiment | Graph/stages/objective/sampler unchanged only if node/frame/feature contract matches; invalidate cached features |
| B semantic/object encoder | Scene factory and possibly graph-representation adapter | Original graph usable only with compatible local positions/cardinality/features; no promise that arbitrary object tokens fit |
| C temporal/history context | Higher-level context producer around policy or a declared conditioning implementation | Environment can keep delivering the same observation; dataset/protocol changes must be explicit if history is trained |
| D world model later | Outer agent composes representation, world model and existing low-level predict API | No world-model dependency in InstantPolicy; no empty world_models package now |
| E MCTS/MCGS later | Outer planner owns state/world-model/value/search and delegates selected low-level goals/context | Low-level policy remains reusable; symbolic/latent goal grounding is future explicit work, not invented API |
| F second benchmark | One adapter implementing observation/demo/step/outcome contract and benchmark config | Same policy composition; evaluate_policy reusable when success protocol fits, otherwise separate benchmark evaluator |
| G sampler replacement | New compatible sampler + factory selector | Scene/graph/denoiser unchanged; flow matching with a different training target also needs a new objective and training experiment |
| H action representation replacement | New codec plus compatible denoiser/objective/graph as required | Environment still consumes decoded pose trajectories if physical command semantics match; not a guaranteed one-file change |

These are design stress tests, not implemented research methods. A changed dimension,
frame, node meaning or trajectory command contract is a semantic migration, not
merely an interchangeable class name.

## 14. Incremental implementation phases

- [x] Phase 0: re-read authority/source, map imports/state/flows, verify unchanged
  original files and run existing L0/L1. Write this pre-edit proposal.
- [x] Phase 1: owner approves design; record accepted composition and checkpoint
  ADRs. Capture a recoverable before-image with hash inventory; resolve Git
  snapshot approach without staging user-owned files implicitly.
- [x] Phase 2: add characterization/config tests; extract pure geometry/action and
  positional helpers with exact functions. Keep legacy exports and run L0/L1.
- [x] Phase 3: structured baseline/profile/legacy config conversion, explicit
  component factory path and injected GraphDenoiser. Preserve leaf names/order.
  Test encoder/graph/stage replacement and generated key inventory.
- [x] Phase 4: extract original objective/sampler and public policy/context lifecycle.
  Test compatible sampler replacement, caching and original mathematical fixtures.
- [x] Phase 5: move checkpoint handling and Lightning adapter behind composition.
  Test diagnostics, saved metadata and original adapter routes; run available L2.
- [x] Phase 6: separate data ownership, environment conversion and evaluator.
  Update train/eval/deploy/prepare entry paths to shared public APIs in coherent
  batches, never leave half-migrated imports. Test fake environment with real policy
  facade, preserve original RLBench protocol and cache/RNG ordering.
- [x] Phase 7: encode accepted dependency closure, run default and installed-package
  checks as available, component swaps and differential comparisons.
- [x] Phase 8: update actual architecture, baseline location notes, semantic owners,
  validation guide, reproduction commands and minimal extension examples. Keep
  original identity/config/provenance frozen. Review all touched files.
- [x] Phase 9: record final results by tier, compatibility limits, missing assets,
  remaining debt and next task; move plan only when claimed acceptance is met.

Each phase records changed files, tests, commands, result/skip reasons and rollback
points here. Detailed test-first implementation steps follow design approval; no
external skill/plugin is required for a future agent to understand this record.

## 15. Explicit non-changes, recovery and open questions

Not changing: mathematical formulas, graph order/relations, sampler indexing,
normalization/action frames, augmentation/data schema, evaluation settings, model
dimensions/defaults, dependency versions, original seed defaults, simulator internals'
behavior, or paper performance claims. No new planner/world model/value model/
memory/language system, generic robotics framework, Hydra, global registry, package
namespace move, occupancy-pretraining repair, formatting sweep, new logging
framework, expensive automatic validation, physical robot actions or CI setup.

Recovery: before runtime edits, preserve the current source+harness as a task-owned
before-image or owner-authorized commit. Since Git is unborn, git reset/checkout
cannot recover these files. Never recursively delete/replace the workspace. Keep
scoped per-phase patches and restore only task-owned changes after checking for
concurrent user edits. Do not use the historical upstream checksum file as the
sole recovery record for the newer harness.

Decision pending: approve proposed component boundaries, ip layout, dataclass/
Python config strategy and checkpoint approach.
External evidence pending: trusted original checkpoints/config/encoder, datasets,
simulator revisions and a compatible Linux/CUDA test environment.
ASSUMPTION: preserving numerical bodies, initialization order and key ownership
will permit legacy loading; requires real checkpoint validation.
UNKNOWN: paper-level reproduction and numerical tolerances on the original stack.
No compatibility invariant is knowingly waived by this proposal.

## 16. Audit evidence and next action

Commands executed from repository root:
- git status --short; git log -3 --oneline (confirms unborn HEAD, not a runtime failure).
- rg --files and AST-derived complete local Python import inventory.
- direct source reads of entry points, model/graph/diffusion/encoding, data,
  normalizer, checkpoint repair, common geometry and occupancy imports.
- awk excluding README from upstream-files.sha256 | shasum -a 256 -c -:
  PASS, 28 originals. No runtime bytes changed during this audit.
- python3 -B scripts/validate_fast.py: PASS, syntax/links/boundary + 14 tests.
- python3 -B -m unittest discover -s tests -p test_cpu_smoke.py -v:
  PASS, four tests, no skips.
- read-only torch toy probes: shared registered modules emit alias key prefixes;
  chained advanced-index assignment does not change the original toy tensor.
  These illustrate risks, not proof of full original-model behavior.
- find_spec/asset existence: missing dependencies/assets as listed above.

L2 model validation: SKIPPED, required packages/extensions and assets unavailable.
L3 simulator: SKIPPED, simulator environment unavailable.
L4 benchmark, training, downloads: NOT RUN.
Changed files in this turn: this proposed active plan only. No accepted architecture
docs or runtime behavior have been rewritten.

Next action: owner review and approval of this design, followed by incremental
implementation with explicit compatibility evidence. This phase does not claim
that the requested runtime refactor has been implemented.

## 17. Implementation ledger after owner approval

- Owner approved the proposed design on 2026-09-06. Decisions 0002 and 0003 are
  accepted. Work remains in place because Git is unborn; no original files staged.
- Before-image: /tmp/ip-refactor-before.yw3tVQ/workspace.tar, SHA-256
  0a90ab3503875a81922f68c136283a0bdc4e7b4698116450991469cb021bb3d0.
  Extracted source beside the archive supports scoped recovery/differential tests.
- Config/geometry RED: missing new modules. GREEN: immutable configs, legacy/profile
  roundtrips and pure geometry tests pass. Original geometry function bodies were
  mechanically extracted and independently reviewed as AST-identical.
- Composition RED: missing builder. GREEN: real PyG graph + injected scene/stages
  forward/backward, independent instances, custom graph geometry, selector errors
  and legacy GraphRep construction. Original parameter registration names retained.
- Algorithm/policy RED: missing public API. GREEN: original objective/sampler with
  a controlled schedule, sampler replacement, input preservation and cache ownership.
- Checkpoint task: generated-module/serialized-fixture tests pass. Review exposed
  unreported equal collapsed aliases; regression added and full mappings now reported.
- Config review exposed original graph width limit on custom graphs; gated to the
  original graph kind. Selector preflight moved before model/artifact construction.
- Differential proof: three opt-in tests pass against hash-verified before-image:
  original graph edges/seeded parameters, denoiser/labels with injected components,
  and exact original sampling loop under controlled schedule. This is not full-model
  or published-checkpoint numerical certification.
- Data/evaluator moved with legacy exports. Open3D/RLBench imports isolated, public
  fake-environment tests and real PyG serialization pass. Retry and cleanup behavior
  retained, including original bare demo/reset exception handlers. No simulator ran.
- Training adapter extraction is source-reviewed but execution SKIPPED (Lightning
  unavailable). Real original checkpoint/model test added, opt-in and prerequisite
  checked; not silently substituted by synthetic components.
- Validation and final documentation are being finalized; no completion claim yet.

## 18. Final implementation report

Outcome: the approved modular architecture is implemented, with no intentional
algorithm change. Sections 1–16 preserve the pre-edit audit/design; actual current
owners/APIs are in [ARCHITECTURE](../../ARCHITECTURE.md). No actual-paper reproduction,
published-checkpoint load, original CUDA model or simulator success claim is made.

### Final runtime tree

```text
ip/
  types.py, geometry.py, actions.py
  composition.py, checkpoints.py, policy.py, training.py, evaluation.py
  configs/{base_config.py,original.py,structured.py}
  algorithms/{__init__.py,diffusion.py}
  models/
    denoiser.py, backbone.py, embeddings.py
    scene_encoder.py, graph_rep.py, graph_transformer.py
    model.py, diffusion.py                       compatibility adapters
    occupancy_net.py                            retained stale route
  data/{__init__.py,preprocessing.py,dataset.py}
  environments/{__init__.py,rlbench.py}
  utils/                                        legacy exports and diagnostics
  train.py, eval.py, deployment.py, prepare_data.py
  sandbox.py, scripts/download_weights.sh
```

Public boundaries: Observation, ActionTrajectory, PreparedContext, PredictivePolicy,
EvaluationEnvironment; frozen ExperimentConfig sections; ComponentFactories;
GraphDenoiser; original SceneEncoder/GraphRep/graph stages; OriginalActionCodec;
OriginalDiffusionObjective/OriginalDiffusionSampler; InstantPolicy; GraphDiffusion
Lightning adapter; RLBenchAdapter and evaluate_policy. The old→new migration table
in section 9 remains accurate. Additional public run_training centralizes the old
trainer procedure; test_loading.py protects resolved overrides at artifact loading.

### Files and decisions

Relative to the approved pre-runtime snapshot: 35 implementation/test/doc files
added, 24 existing files modified, and this active plan modified/moved to completed.
The 20 new runtime Python files implement the owners in the tree above; 12 new
test modules cover architecture, config, geometry, composition, policy, data,
evaluation, checkpoint mapping/loading, differential methods, model and training
integration. Three new documentation files are composition-examples and ADRs
0002/0003. No original runtime file was deleted; moved responsibilities have
compatibility exports. setup.py, environment.yml, .gitignore, media, graph_transformer
math and stale occupancy code remain unchanged. Historical hashes remain frozen.
No files were staged/committed/pushed; the user's repository is still unborn.

Accepted decisions: keep ip rather than root/src namespace migration; use immutable
dataclass/Python config and explicit immutable factory mappings rather than Hydra/
auto-discovery; isolate environment/training; preserve parameter owners and diagnose
checkpoint translations. The objective and scheduler have independent config IDs
(diffusion.kind and diffusion.scheduler_kind). Old base_config.config is now a
compatibility view of the canonical composition, not a competing mutable runtime
source. Extra standalone/multi-demo load tests were justified by review defects.

### Baseline and compatibility

Canonical config: ip.configs.original.instant_policy_original(). All legacy fields
and tensor values compare exactly with the hash-verified original config on the
tested host. Entry eval/deploy profiles retain four inference steps and batch one;
source baseline remains eight test steps and batch16. Original numerical bodies
are preserved; independently compared selected original methods pass exactly.

Checkpoint: generated raw/Lightning/compiled/alias mappings and serialized loads
pass, including unequal collision/shape rejection and full translation diagnostics.
Original leaf names and Lightning aliases retained. Loading config/weights occurs
through shared composition. Real published artifact compatibility is UNVERIFIED.
The new GraphDiffusion load adapter is weights/config reconstruction, not certified
Lightning optimizer-resume or arbitrary whole-object-pickle compatibility.

Configuration: legacy dict conversion validates original keys; versioned resolved
JSON preserves named selectors and sections. Entry profile precedes explicit load
overrides; resolved demo count is no longer overwritten by the raw load argument.
Custom factories must be supplied by the caller on load. Saved identifiers do not
serialize implementation code, so record the experiment revision/factory provenance.

Dataset: original PyG schema, target generation, augmentation, cache behavior and
retry algorithm retained. Open3D calls moved to lazy preprocessing imports.
Original bare torch.load remains; on local Torch2.11, PyG deserialization can be
rejected by newer weights_only defaults and the preserved retry loop can hang.
This inherited environment-drift issue is NOT fixed or covered up by the refactor.

Environment: original task aliases, masks/cameras, arm patches/joints, action anchor,
rotation restrictions, retries, termination/reward metric and normal-only shutdown
preserved. Public fake-environment and action-conversion tests pass. L3 simulator
parity is UNVERIFIED. TASK_NAMES is now a lazy Mapping, not a mutable dict; documented
aliases/indexing/iteration remain, arbitrary task-map mutation is not guaranteed.

### Actual validation commands and results

All commands ran from repository root. Host: macOS15.7.2 arm64, Python3.14.4,
torch2.11.0, NumPy2.4.4, SciPy1.17.1, PyG2.5.0; no CUDA. No dependencies installed.

| Command / evidence | Result |
| --- | --- |
| python3 -B scripts/validate_fast.py | PASS: syntax, docs links, harness boundary, 17 stdlib tests (14 harness + 3 architecture) |
| python3 -B -S scripts/validate_fast.py | PASS: same L0 with site packages disabled |
| IP_LEGACY_SOURCE_ROOT=/tmp/ip-refactor-before.yw3tVQ python3 -B -m unittest discover -s tests -p 'test_*.py' | 74 selected: 70 PASS, 4 SKIPPED, zero failures/errors |
| IP_LEGACY_SOURCE_ROOT=/tmp/ip-refactor-before.yw3tVQ python3 -B -m unittest discover -s tests -p test_differential.py -v | PASS4: all baseline config fields, graph structure/seeded params, denoiser/labels, controlled-schedule original sampler |
| python3 -B -m unittest discover -s tests -p test_loading.py -v | PASS2 after expected red reproduction of override/preflight defects |
| python3 -B -m ip.eval --help | PASS without simulator/Lightning/Open3D/Diffusers; no rollout |
| Core/composition/legacy utility/environment module import probe | PASS without optional environment/training libraries |
| IP_RUN_MODEL_TESTS=1 python3 -B -m unittest discover -s tests -p test_model_integration.py -v | SKIPPED: missing torch_cluster, diffusers, lightning, open3d |
| Original scene construction on host (during selector regression) | FAIL in installed PyG typing inspector: typing.Union has no _name; original-scene math not changed to bypass it |
| Before-image comparison of setup.py/environment.yml/historical hash inventory | PASS unchanged |

The four skipped tests in the full enabled-differential run: Open3D path, real
RLBench task resolution, explicit original-model L2 test, Lightning adapter.
Missing weights/encoder/data and simulator prevent full integration. No full
training, dataset preprocessing job, asset download, robot operation or L4 benchmark
was launched (NOT RUN). Installed-package build/install validation was NOT RUN;
package layout/setup remain unchanged. Source import checks are not installed-wheel
proof. Default discovery without IP_LEGACY_SOURCE_ROOT additionally skips four
differential tests.

Test limitations: injected scene/stage/schedule/policy/environment collaborators
exercise public seams and real graph/heads, not original PointNet/FPS/CUDA kernels.
Some focused data tests replace only unavailable filtering/load IO to test other
behavior; they do not assert that those missing paths passed. Optional original-
artifact test does not download data. No fake third-party modules are substituted.
PyG/Python3.14 JIT deprecation warnings are visible and not suppressed into a claim.

### Review and recovery evidence

Independent core review found equal-alias diagnostics and custom-width validation
defects; fixed with regressions. Selector preflight was improved. Independent
environment review found no migration regression. Final whole-refactor review
found a resolved-demo override/cache defect and standalone unknown encoder fallback;
both reproduced red, fixed, verified green, and re-reviewed with no blocking
findings remaining. No unresolved review issue is silently parked.

Before-image archive and reports remain in /tmp/ip-refactor-before.yw3tVQ. That is
task-owned recovery material, not a runtime dependency. The durable result/limits
are recorded here; removing the archive does not affect runtime operation. Do not
delete original untracked files or reset this unborn repository for cleanup.

### Research readiness and next task

Encoder, graph geometry/builder, graph stages, sampler, objective and schedule can
now be selected/injected without editing unrelated runtime owners. Generic evaluation
uses a public policy/adapter contract. Five minimal composition/future examples
are in [composition-examples](../../components/composition-examples.md), including
scene, sampler, second environment, outer world model and outer planner. No new
research method was implemented.

Remaining debt: old recording-without-WandB error, stale occupancy route,
unbounded data/environment retries, unprovisioned original stack/assets,
unverified published model/simulator parity, original frame/time-indexing quirks,
and original compilation internals for custom encoders. Config artifacts identify
components but do not distribute custom factory code.

Next logical research-engineering task: provision and pin the declared Linux/CUDA
environment plus trusted original encoder/policy artifacts, then run the new L2
checkpoint/inference and original-component differential checks. Establish that
reproduction evidence before implementing research methods. A loader compatibility
fix for newer PyTorch should be a separately classified behavior change.
