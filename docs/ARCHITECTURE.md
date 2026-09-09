# ICGS unified runtime

ICGS is the method-level project. Its only installed runtime is `src/icgs`.
Instant Policy is one internal policy and a collection of models/diffusion
components, not an external backend or a package that owns the method.

## Dependencies and ownership

```text
CLI → configuration + composition + artifacts
                      ↓
training/evaluation → execution → policy + environment adapter
                                  ↓
                      state + sampling/proposal algorithms
                                  ↓
                  neural encoders/graph/backbone/denoiser
                                  ↓
                         contracts + geometry
```

| Directory | Actual responsibility |
| --- | --- |
| contracts | World-frame Observation, native ActionTrajectory, narrow policy/environment capabilities |
| state | Owner-bound immutable demo content, mutable context features, scoped RNG restoration |
| geometry | Existing pose, rotation and point-set transforms, grouped in transforms.py |
| models/layers | Shared positional encodings |
| models/encoders | Native IP scene encoder; no speculative v5 encoder |
| models/graphs | Native topology, learned graph features, masks and session scratch |
| models/backbones | GraphTransformer and native three-stage processing |
| models/denoisers | Native graph denoiser with registered parameter names preserved |
| algorithms/diffusion | Action codec, original schedule/objective/sampler in coherent files |
| algorithms/planning | Sequential candidate proposer only; no fake search/value/dynamics |
| policies | InstantPolicy public observation/context → native action inference |
| execution | Rollout loop and root-relative → absolute command prefix |
| environments/rlbench | Lazy simulator setup, tasks, observation/action conversion |
| evaluation | Benchmark entry delegating the shared execution protocol |
| data | Native PyG preprocessing/datasets; safe versioned inference NPZ |
| configuration | Frozen schema/defaults, explicit JSON parser, profiles and validation |
| artifacts | Checkpoint inspection/translation, hash-bound published profile and legacy format repair |
| training | Lightning module versus runner; retained unsupported occupancy source |
| cli | infer, train, evaluate, prepare-data and lightweight help |

Models never import policy, search, training, environment, CLI or checkpoint IO.
Search/candidate algorithms use proposer capability, not concrete InstantPolicy.
Runtime never imports old ip, instant_policy.so, docs, tests or root scripts.
No sys.path/sys.modules aliases form part of the architecture.

## Configuration and construction

Authoritative native defaults live in configuration.schema/defaults. Added ICGS
method defaults live in the packaged
[primary JSON](../src/icgs/configuration/profiles/icgs_primary.json), with immutable
typed validation in configuration.method. See [parameter ownership](method/parameters.md)
for current versus planned consumers, shape locks and explicit overrides. Root JSON
presets are explicit user inputs, not files implicitly read by an installed wheel.
Precedence: runtime defaults/entry profile < explicit JSON sections < explicit CLI
overrides. Relative artifact paths in JSON resolve against that file's directory.
Unknown fields, component IDs and shape/horizon/type mismatches fail before model IO.

Composition builds explicit components from immutable factory maps. Neural
constructors do not load checkpoints. Caller-supplied factories are supported by
Python composition; CLI selects only implemented identifiers. No Hydra or discovery.

Published profile is separate from historical source defaults. Its packaged profile
is tied to the actual checkpoint SHA256 and exposed reference configuration.
It disables auxiliary encoder loading, uses live voxel size .01, and draws sampling
noise before first scene encoding. See [fidelity decision](decisions/0005-published-profile-fidelity.md).
All 337 model-owned keys are loaded strictly from 674 reference alias entries;
equal aliases are explicitly reported, conflicts/shapes/missing keys fail.

## Public semantics and state

Observation contains segmented world XYZ [N,3], T_w_e [4,4], and observed grip0/1.
Policy prepares local clouds exactly once. ActionTrajectory holds root-relative
[B,P,4,4] transforms and normalized [B,P,1] grips. Current public observation API
supports B=1; K candidates are sequential proposals, not batch B.

Prepared contexts own immutable numeric demo copies and a content hash. Their
owner/count/waypoint checks prevent cross-policy or mismatched use. Mutable
embeddings may be copied for a branch, but graph/scheduler/model are never cloned
per node. Sessions and global seeded scopes are not concurrent-call safe.
A frozen record does not imply tensor contents are immutable; demo arrays use
read-only byte-backed storage, mutable outputs are caller-owned.

Published profile does not persist scene embeddings between calls, matching observed
reference ordering. Historical source profile preserves its old cache path. Both
paths are explicit; no arbitrary goal/latent-state support is claimed.

## Execution and command paths

- `icgs infer`: safe NPZ → hash-bound strict published load → prepare context →
  full native forward/sampler → validated relative/absolute trajectories.
- `icgs train`: explicit config/data → same component construction → Lightning
  adapter/objective/optimizer → native checkpoints and resolved metadata.
- `icgs evaluate`: strict published config/weights → RLBench adapter and common
  execution loop → original termination/reward success fraction.
- `icgs prepare-data`: explicit NPZ with D conditioning demos plus a live trajectory
  → original conversion/schema → PyG files. No invented data collector.
- Method foundations: `propose_candidates` → K real native trajectories;
  `absolute_prefix` → common root-anchored targets and known/unknown timing.

The original source formulas, masks, six gripper points, graph stages and sampler
indexing remain. Recorded/simulator data, broad task success, optimizer-resume and
whole-object old-pickle compatibility are separate claims.

## Deliberate compatibility changes

Old public ip imports/CLI paths are removed by owner authorization. The checkpoint
was moved to artifacts/checkpoints/vv19 without changing bytes; no user data was
deleted. setup.py's placeholder metadata was replaced by pyproject.toml.
Historic documents link to pinned Git source rather than maintaining a second runtime.

Unsupported occupancy/pretraining code is preserved under training/stages, fails
explicitly on construction and is never part of native inference. Missing visualizer/
incompatible historical encoder calls are not disguised as a working port.
The old recording-without-WandB and unbounded loader/reset retry behavior is retained;
run the validated environment and use explicit resource limits.

[Method foundation status](components/v5-foundations.md),
[semantic contract](components/policy-data-contract.md),
[usage examples](components/composition-examples.md), and
[completed acceptance record](plans/completed/icgs-v5-unified.md) provide deeper evidence.

## Approved target extension — not implemented

The September 2026 [ICGS target](method/README.md) adds a separate physical point
encoder/decoder and history, demonstration events/task memory/router, physical
dynamics, continuation/terminal evaluators and budgeted reranking/shooting/MCTS.
Native IP remains the frozen proposal component. New timed execution, executed
episode collection and staged learning are planned owners, not current CLI claims.

The [component roadmap](plans/active/icgs-method-implementation.md) records precise
dependencies and evidence gates. The target's 5 mm physical preprocessing is not
the published IP 10 mm preprocessing; 0.1 s is a proposed control wrapper, not
native step timing. No new runtime modules were added by the documentation adoption.
