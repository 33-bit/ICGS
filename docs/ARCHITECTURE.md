# Current software architecture

Instant Policy is a composition of reusable components inside the existing `ip`
namespace. The [original baseline](baselines/instant_policy.md) preserves historical
identity and values; [ADR 0002](decisions/0002-runtime-composition.md) owns dependency
rules. This is a structural refactor, not a new algorithm.

## Runtime map and dependencies

```text
train.py / eval.py / deployment.py / prepare_data.py
             ↓
composition.py + checkpoints.py + training/evaluation orchestration
             ↓                                  ↓
policy.py                                  environments/rlbench.py
             ↓                                  ↓
models/denoiser + algorithms/diffusion       types Observation/ActionTrajectory
             ↓
SceneEncoder + GraphRep + graph stages + action codec
             ↓
geometry / positional embeddings / narrow configuration
```

| Owner | Responsibility |
| --- | --- |
| [configs/structured.py](../ip/configs/structured.py) | Frozen experiment sections and validation |
| [configs/original.py](../ip/configs/original.py) | instant_policy_original, legacy conversion, entry profiles, versioned resolved metadata |
| [composition.py](../ip/composition.py) | Explicit factories, component initialization/encoder IO, shared network/policy/training/load construction |
| [types.py](../ip/types.py) | Observation, ActionTrajectory, PreparedContext, policy/environment protocols |
| [geometry.py](../ip/geometry.py) | Original NumPy/SciPy pose and torch action/rotation/SVD math |
| [actions.py](../ip/actions.py) | Normalizer and OriginalActionCodec: normalization, labels, transforms and rigid recovery |
| [models/scene_encoder.py](../ip/models/scene_encoder.py) | Original two-stage point encoder; injected by composition |
| [models/embeddings.py](../ip/models/embeddings.py) | Original positional encodings, without Open3D imports |
| [models/graph_rep.py](../ip/models/graph_rep.py) | Original graph structure, learned embeddings, masks/features; narrow GraphConfig |
| [models/graph_transformer.py](../ip/models/graph_transformer.py) | Original TransformerConv/MLP blocks |
| [models/backbone.py](../ip/models/backbone.py) | Original stage factory and stateless three-stage execution |
| [models/denoiser.py](../ip/models/denoiser.py) | GraphDenoiser: supplied components and original forward computation |
| [algorithms/diffusion.py](../ip/algorithms/diffusion.py) | OriginalDiffusionObjective, OriginalDiffusionSampler and DDIM factory |
| [policy.py](../ip/policy.py) | Inference, model-ready batch API, demo preprocessing/cache lifecycle |
| [data/preprocessing.py](../ip/data/preprocessing.py) | Original waypoint/cloud/target/serialization code; lazy Open3D |
| [data/dataset.py](../ip/data/dataset.py) | Original indexed loading/retries and augmentation |
| [training.py](../ip/training.py) | Lightning GraphDiffusion adapter, metrics/optimizer/checkpoint lifecycle, run_training |
| [evaluation.py](../ip/evaluation.py) | evaluate_policy over public policy/environment protocols |
| [environments/rlbench.py](../ip/environments/rlbench.py) | Simulator task mapping/setup/demos/observation/action conversion and legacy rollout adapter |
| [checkpoints.py](../ip/checkpoints.py) | Explicit state translation, mismatch diagnostics and trusted loading |

No reusable core depends on RLBench, WandB, Lightning, argparse, composition or
training. Core imports numerical libraries; policy preprocessing still needs Open3D
when actual filtering runs. Pure geometry/model imports do not reach Open3D.
The AST local-dependency check includes indirect imports and package initializers;
it does not inspect arbitrary dynamic imports or third-party internals.

`models/model.py::AGI` is the legacy construction adapter;
`models/diffusion.py::GraphDiffusion` re-exports the Lightning adapter.
`utils/` retains legacy exports/diagnostics. New core does not import those adapters.
`occupancy_net.py` remains a documented stale pretraining path, outside reusable
core. No world-model/planner/history/language package has been created.

## Construction and ownership

`instant_policy_original()` returns independent frozen sections with immutable
numeric action-limit tuples. `from_legacy` requires original fields; `to_legacy`
returns fresh tensors/dictionary. The old `base_config.config` is a compatibility
export only, not a shared state source for new training/evaluation code.

`ComponentFactories` holds immutable caller-supplied mappings for scene, graph,
backbone, codec, scheduler, objective and sampler. Original names use explicit
built-ins; unknown identifiers fail before artifact/model creation. No automatic
plugin discovery or process-global registration. Components receive only relevant
sections plus explicit dimensions/device.

`build_network` creates components in original order and preserves registered
names scene_encoder, graph, local_encoder, cond_encoder, action_encoder and three
prediction heads. Stage execution does not add a backbone.* state-dict prefix.
`build_policy` composes the network, codec, schedule, objective and sampler.
The plain policy facade is not an additional nn.Module parent. A training wrapper
registers the network under model and retains original aliases for compatibility.

Graph, schedule and context state belong to one policy instance. Context records
carry an owner token; cross-policy reuse fails. Context preparation does not eagerly
encode demos; reset clears embeddings per rollout, preserving the original first-
prediction cache timing. Sampling clones the model-ready caller batch, but original
internal mutation/order is retained. Policies are not promised thread-safe for
concurrent calls. Device is selected at construction; no general post-construction
policy.to(device) contract exists for unregistered graph/codec tensors.

## Execution flows

Training:
structured baseline or saved config → entry profile → build_training_module →
shared build_policy/network → PyG dataset/loaders → Lightning training_step →
callable objective → original noise/labels/denoiser/L1 → optimizer → validation
sampler/metrics → legacy-named checkpoints and config.pkl plus resolved metadata.

Evaluation:
read saved resolved_config.json (if present) or trusted config.pkl → eval profile
(batch1, demos, four sampler steps) → shared build_policy → diagnostic strict state
load → eval/optional compilation → RLBenchAdapter → evaluate_policy →
prepare/reset context → policy.predict → anchored trajectory commands → task step →
original success fraction.

Deployment:
same load_policy, explicit legacy deployment profile/non-strict diagnostics →
prepare_context → predict(Observation, context) → controller-specific execution.
The shipped file is still an example requiring demos/observations/controller.

Data:
raw environment demo → generic point-cloud/waypoint processing → existing PyG
sample schema → save_sample or RunningDataset. RLBench-specific masks/cameras
remain in the environment adapter, not generic preprocessing. A second environment
implements the same Observation/ActionTrajectory boundary, not AGI internals.

## Public component contracts

- Encoder: (x or None, XYZ[N,3], membership[N]) → features[M,E], positions[M,3],
  membership[M]. Original graph requires compatible frame/cardinality/features.
- Graph: constructor(GraphConfig,batch_size,device), initialise_graph/update_graph,
  graph dictionaries/action node metadata, edge_dim and gripper geometry.
- Backbone factory: (BackboneConfig,input_channels,edge_dim) → three nn.Modules;
  each accepts feature/edge-index/edge-attribute dictionaries.
- Codec: normalization + encode/decode + get_labels + gripper-node positions and
  rigid recovery. Arbitrary action semantics may require a paired graph/objective.
- Objective: constructor(DiffusionConfig,codec,schedule), callable(network,data)
  returns differentiable scalar. Original exposes add_noise for legacy callers.
- Sampler: constructor(SamplingConfig,DiffusionConfig,codec,schedule);
  sample(network,data) → ActionTrajectory.
- Policy: eval, prepare_context(demos,prepared=False), reset_context,
  predict(Observation,context), predict_batch(PyG data).
- Environment: launch, collect_demos, reset, observe, encode_action, step, shutdown.

Exact tensors, masks, frames and original mathematical quirks remain in
[policy-data-contract](components/policy-data-contract.md). Original full model,
published-checkpoint and simulator parity are not certified by synthetic-component
tests. See [validation](../tests/README.md) and the
[migration record](plans/completed/modularize-instant-policy.md).

## Experiments and limits

[Composition examples](components/composition-examples.md) show actual factory/config
use, second-environment integration and hypothetical outer world-model/planner
composition. No research method is implemented by those future examples.

Compilation currently targets original scene internals and graph/head modules;
custom encoders should use compile_models=False unless compatible. Custom factories
must be supplied again on load; saved IDs are not executable code distribution.
The original recording-without-WandB bug, dataset/reset retry behavior, mathematical
quirks and stale occupancy route are not repaired. New Torch weights_only defaults
can make original dataset loading hang on retried PyG deserialization errors; use
the declared environment and address that separately.

Host compatibility observation: constructing the original scene encoder on the
development Python 3.14/PyG 2.5 stack fails in PyG's typing inspector
(`typing.Union` lacks `_name`). The encoder mathematical implementation is not
rewritten to bypass this dependency issue. Use the declared Python 3.10 stack for
original-component validation; available synthetic collaborators do not certify
that original encoder on this host.
