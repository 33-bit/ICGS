# ICGS target method

Status: **approved target, not implemented**. Authority: owner approval of the
documentation/component-plan package on 2026-09-07. The [archived proposal](../proposals/README.md)
remains the scientific source; the refinements below make implementation explicit.
[Current architecture](../ARCHITECTURE.md) and [foundation status](../components/v5-foundations.md)
describe what actually runs.

## Reading and authority

| Owner | Read for |
| --- | --- |
| [Contracts](contracts.md) | Shared types, online information, time/frames, state and interfaces |
| [Neural method](neural.md) | Exact forward passes, dimensions, masks and gradient boundaries |
| [Data and training](data-training.md) | Dataset lineage, generation, labels, losses and phase order |
| [Planning and evaluation](planning-evaluation.md) | Search semantics, B0–B7, B3 design, experiments and metrics |
| [Requirements matrix](requirements.md) | Proposal section/equation → specification → plan → test |
| [Master roadmap](../plans/active/icgs-method-implementation.md) | P00–P13 dependencies, pilot gates and progress |
| [Boundary decision](../decisions/0006-icgs-target-boundaries.md) | New model/algorithm responsibilities, native preservation |
| [Execution/data decision](../decisions/0007-icgs-timed-data-lineage.md) | Named cadence and executed-data provenance |
| [Reference decision](../decisions/0008-reference-value-and-stopping.md) | Frozen reference value versus learned stopping |

Requirement labels used throughout:

- **PR — proposal requirement:** directly specified in the immutable source.
- **AD — approved decision:** explicitly approved by the owner, including the
  separate stopping rule and separate B3 diffusion student.
- **ID — implementation default:** deterministic resolution within the approved
  method; not a measured optimum or a new scientific claim. Changes require a
  recorded configuration/protocol delta and affected tests.
- **EI — existing implementation:** backed by a named current source/test/evidence.
- **FG — feasibility gate:** evidence still required before the dependent workload.

The matrices use these labels rather than treating a planned test as current proof.
The original proposal governs scientific intent; explicit AD resolutions govern
their named ambiguities. IDs cannot relax PR/AD invariants. FG results are recorded
before dependent jobs are launched. No new runtime implementation is authorized by
merely reading these plans.

## Research question and claims

**PR:** Given one or two successful demonstrations, choose actions that preserve
the ability to finish the remaining task without test-time gradients, task reward,
language/program input or cross-test archive updates. The primary scope is rigid,
quasi-static tabletop manipulation with Panda and limited articulated containers.

- H1: outcome-trained, full-context evaluation distinguishes locally successful
  branches with different downstream continuation success, even with correct stage.
- H2: crossing relevant interaction boundaries improves over direct Q, reranking
  and shooting under matched data/prior/evaluator, horizon and compute controls.
- H3: fixing query history/scene/candidates while changing an identifiable suffix
  changes decisions in the direction supported by executed outcomes; nuisance
  demonstration route/speed changes should not destroy task success.

Execution length, causal history length and dependency span are separate axes.
Neural building blocks and MCTS are not themselves novelty claims. If the model
does not cross a dependency boundary, better results support continuation-value
selection, not demonstrated long-range world-model reasoning. If MCTS does not
beat shooting, retain that result.

## Architecture and lifecycle

```text
independent raw demonstrations → event encoder → immutable event memory M_C
                                             ↘ native full/window contexts
real observation + previous executed command → physical state b → task state q
                                               ↓                  ↓
                                   frozen native IP + router → absolute commands
                                               ↓
              branch-local b → physical dynamics → decoded world cloud
                                  ↓                     ↓
                             achieved pose         re-encode → update b
                                                              ↓
                                              update q with M_C
                                                              ↓
                                      terminal hazards + continuation value
                                                              ↓
                                 rerank / shooting / MCTS → execute r intervals
                                                              ↓
                                     measured observations → new real root
```

**AD:** models own tensors; algorithms own losses/search; state owns causal memory;
training owns optimizer/runner/artifacts; environments own simulator interfaces;
benchmark evaluation owns privileged scoring. Only composition/outer IO loads
artifacts. Runtime never reads this documentation or imports harness code.

**PR:** the world model receives no task context. The task tracker consumes all
observed history through recurrent state. Imagined memory is branch-local and
never replaces the real history. Real-root IP consumes measured clouds; only
imagined-node IP consumes model clouds. Every hypothesis receives the same
absolute physical command, not the same relative delta.

**AD:** `pi_ref` is frozen IP plus the learned router, without learned stopping.
Its continuations terminate at external physical task terminal/deadline during
label collection. The learned completion gate is a separate deployment wrapper
shared by B1–B7, with its own artifact ID and false-stop report. B0 remains native.
The deployed gated B2 is not silently called the exact value-target controller.

## Native preservation and global constraints

- Canonical runtime is `src/icgs`; no `ip` shims or wrapped legacy runtime.
- Preserve the strict hash-bound published IP artifact, registered model keys,
  preprocessing, graph/action semantics and sampling RNG order.
- Native public inference remains B=1. Candidate K, native action horizon P=8,
  native demo waypoints D_wp=10, and demonstration count D=1 or 2 are distinct.
  Proposal symbol T=8 maps to runtime P, not `traj_horizon` (demo waypoints).
- Added encoder uses 5 mm/FPS preprocessing; published IP live preprocessing uses
  10 mm/native sampling. Neither silently replaces the other.
- Existing PyG and inference NPZ formats remain native formats; ICGS episode
  records/configuration/checkpoints use separately versioned schemas.
- Preserve declared Python >=3.10,<3.13 and pinned runtime dependencies; local
  Python 3.14 L0 execution is not a supported model environment.
- No training, downloads, preprocessing, simulator workloads or robot motion as
  incidental validation. Paid/large compute and primary collection require scope.
- Numerical settings are proposed defaults; SKIPPED never counts as PASS.

## Limitations and optional tracks

**PR:** observed history need not be a sufficient physical state. XYZ cannot
identify color, unknown physical properties or full-arm collision state. Foreground
segmentation is a perception assumption, not raw-pixel evaluation. Ambiguous
contexts form a separate set. Decoded clouds need not correspond to realizable
hidden states. Three bootstrap heads are correlated deterministic predictors,
not calibrated posterior or full contact-mode samples. Open-loop belief search
does not correctly value information gathering.

ManiLong-Shot reproduction, LIBERO/CALVIN transfer, ManiSkill/MimicGen external
data, predicted segmentation, strict length/dependency-mechanism stress, independent
dynamics trunks, joint contact-mode models, and real robots are separately scoped.
See [extension gates](planning-evaluation.md#optional-extension-gates). No external
dataset download or benchmark score is required to finish this documentation.
