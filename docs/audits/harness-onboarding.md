# Harness onboarding audit and approved design

Audit date: 2026-09-06. Implementation approved by the repository owner after the
pre-edit report. This dated record is evidence and rationale; the
[authority map](../README.md) routes current policy.

## Source and inspection boundary

- Instant Policy: [65dc94e347df5bca4e390a6f959cd3308f8ae2bc](https://github.com/matdmiller/instant_policy/tree/65dc94e347df5bca4e390a6f959cd3308f8ae2bc).
- repository-harness: [e765792b635b4d5e3e5fc0578f82f9ca5dea2681](https://github.com/hoangnb24/repository-harness/tree/e765792b635b4d5e3e5fc0578f82f9ca5dea2681).
- Local checkout initially had no commits, remote or tracked files: all 29 original
  files were untracked and byte-identical to upstream (recursive diff excluding
  .git). No applicable local AGENTS existed before onboarding.
- All original Python modules, package initializers, entry scripts, configuration,
  environment manifest, setup, README and download script were inspected.
  Media was inventoried and byte-compared, not treated as executable behavior.
- Upstream reference docs, AGENTS, all five SKILL.md files and invocation metadata,
  installation payload manifests, bootstrap paths, validation entry points,
  architecture, workflow, plans, decision templates and relevant decisions were
  inspected. Upstream Rust distribution tests were not executed.
- Reference clones were created outside the workspace at
  `/tmp/instant-policy-audit.KFDXEn`. No upstream installer was run.
  This was not an execution of upstream onboard-repository's strict
  no-temporary-files/evidence-capsule protocol.
- GitNexus was unavailable; source tracing was direct. Active host: Darwin arm64,
  Python 3.14.4. No intended Linux/CUDA environment installation was attempted.

FACT means directly verified source/command evidence. ASSUMPTION means inference
not fully verified. UNKNOWN means additional evidence is needed. Neither inferred
boundaries nor test defaults create repository policy.

## Instant Policy findings

[Architecture](../ARCHITECTURE.md) contains the verified file map and train/eval/
deployment traces. [Baseline](../baselines/instant_policy.md) contains the original
configuration and entry-point overrides. [Semantic contract](../components/policy-data-contract.md)
contains research-critical data/graph/action details, avoiding duplicated rules.

FACT: runtime is the flat ip package; no src-layout migration exists. Training
uses separately preprocessed serialized PyG data, GraphDiffusion/AGI, Lightning
and manual checkpoint recording. Evaluation uses saved config, strict model load,
RLBench live demos and success fraction. Deployment and preprocessing are examples
requiring user code/data.

FACT: documentation was only the README. No tests, CI, research contract, baseline
manifest, ADRs or plans existed. Sandbox prints a subset of package versions.
WandB/config/checkpoint saves are the existing run-recording mechanisms, not a
complete reproducibility system.

FACT: environment.yml specifies Linux-specific builds and CUDA 11.8 PyTorch/PyG;
RLBench is separate. Google Drive supplies pretrained assets via gdown without
repository-published checksums. No runtime assets or dataset are present locally.
setup.py has empty install_requires, so package installation is not dependency proof.

FACT: train/eval do not invoke the existing seed helper. Evaluation uses four
inference diffusion iterations despite eight in source defaults. Deploy uses
strict=False while eval and fine-tuning use strict=True. Scene encoder weights
can be required during construction even for policy-checkpoint loading.

### Source-visible debt, intentionally not repaired

| Finding | Evidence | Risk / next proof |
| --- | --- | --- |
| Recording without WandB leaves logger unassigned | ip/train.py recording branch | Reproduce with bounded entry-point test in later fix |
| Dataset loader retries all exceptions indefinitely | RunningDataset.__getitem__ | Timeouts/data integrity and future explicit failure policy |
| Demo collection/reset retries can be unbounded; shutdown not finally-protected | rollout_model | Isolated simulator lifecycle/timeout testing |
| Missing ip.utils.visualiser and unsupported SceneEncoder num_layers argument | occupancy_net.py | Occupancy pretraining route is not verified operable |
| Empty demos / None observation-controller placeholders | prepare_data.py / deployment.py | Not runnable fixtures or drivers |
| Hardcoded horizon eight in rollout/deploy scaffolding | rl_bench_utils.py / deployment.py | Config shape changes require integration characterization |
| Action-conditioned cloud formula differs from general inverse rigid transform | AGI.forward | Characterize with nonzero rotation and translation before changing |
| Chained advanced-indexing time-embedding assignment | GraphRep.update_graph | Verify intended mutation with a focused numerical test |
| Nonstandard sampler timestep indexing | GraphDiffusion.test_step | Preserve observed behavior before testing alternatives |
| Documentation clone URL differs from requested fork; setup metadata is placeholder-like | README / setup.py | Provenance clarity; do not silently “fix” runtime packaging |
| Original ignore file lacks dataset/checkpoint-specific rules | .gitignore | Inspect research outputs before staging |

ASSUMPTION: available upstream checkpoint config may match source defaults.
UNKNOWN: actual config/weights/checksums, paper data generation and splits,
encoder-training recipe, exact simulator revisions, seeds, matched expected metrics
and hardware tolerances. Syntax and source identity do not resolve these gaps.

## Reference harness mechanisms

The current upstream is a repository protocol plus a Rust maintenance binary and
Bash/PowerShell bootstrap. Its former SQLite control plane and protocol v1 are
retired, not current dependencies. The installer stages/builds a candidate, verifies
remote checksum and version, installs/updates managed core state, and supports
backups/conflict handling. Even bootstrap dry-run can stage/build/download a
candidate; it was not used as a read-only diagnostic.

| Mechanism | Purpose | Decision | Reason |
| --- | --- | --- | --- |
| Small AGENTS entrypoint | Route authority | Adopt | Cheap startup context |
| Documentation map | Establish owners | Adopt | Avoid contradictory research rules |
| Work-shaped process | Scale ceremony | Adapt | Add engineering/experiment/migration categories |
| Markdown plans | Durable working memory | Adapt | Add baseline, dataset/checkpoint compatibility and recovery |
| ADRs | Lasting decisions | Adapt | Include research semantics and comparison consequences |
| Completion evidence | Bound claims | Adopt | Explicit PASS/FAIL/SKIPPED/NOT RUN |
| Accepted invariants | Mechanical guards | Adapt | Python-native, narrow accepted scope |
| Operational runbooks | Real prerequisite/ownership guidance | Adapt | GPU/simulator cost and lifecycle limits |
| Rust installer/updater and .harness-core | Distribution/provenance transactions | Reject | Unnecessary additional maintenance system |
| Authenticated transcript capsules | Exact patch/source verification | Reject for default workflow | Disproportionate for lightweight research onboarding |
| Client-specific shims | Route other agents | Optional | No demonstrated client need yet |
| Fresh-agent improvement experiments | Test guidance interventions | Optional | Require repeated friction and authorized scope |
| Upstream Rust/release tests | Prove upstream distribution | Reject for consumer | Not Instant Policy validation |

## All reference skills

The reference contains exactly five skills; all were explicitly inspected.
No skill is adopted verbatim.

| Skill | Purpose | Decision | Problem and stability assessment |
| --- | --- | --- | --- |
| onboard-repository | Evidence-backed unfamiliar-repo mapping | Adapt | Recurring need; stable inspect/trace/propose flow; drop compulsory capsule and service-resource ledgers |
| encode-invariant | Enforce accepted rules with positive/negative proof | Adapt | Stable useful flow; route local owners and do not invent future core boundaries |
| engineering-wisdom | Contextual engineering review | Optional, not installed | Useful later; heuristics cannot become mandatory architecture |
| audit-onboarding-proposal | Independent transcript/patch audit | Optional, not installed | Useful for high-risk evidence review; current full capsule protocol is costly |
| improve-harness | Test an intervention against observed friction | Conditional optional, not installed | Only after repeated failures, explicit scope and representative replay |

The two adapters preserve source attribution in
[third-party notices](../third-party-notices.md), but not installer coupling.
They are repository-local and do not require a particular agent plugin.

Research-specific candidates: create-experiment, add-research-component,
validate-baseline, reproduce-baseline, audit-experiment and compare-configs.
All deferred: none has a sufficiently stable repeated local workflow today.
An experiment template is sufficient initial support.

## Approved minimum design

Selected: documentation + two short skills + focused standard-library/static and
CPU tests. Documentation-only lacks mechanical feedback; full upstream
installation adds distribution machinery without research benefit.

Files and owners are in the [documentation map](../README.md). One semantic contract
covers current data/graph/action interfaces instead of class-by-class documents.
A provenance checksum inventory identifies the original source without duplicating
its implementation. One boundary decision accepts harness/runtime separation.
Templates guide plans, decisions and real experiments; no hypothetical config,
planner, world model or empty folder is created.

The current runtime imports no harness surface. Only root README gets a navigation
addition. ip, setup.py, environment.yml, .gitignore and media remain byte-identical.
The historical original README checksum intentionally differs after navigation
is added; the inventory must not be regenerated to hide that change.

### Validation design

L0 checks syntax, local links and syntactic reverse dependencies with positive/
negative fixtures. L1 characterizes independent config copies and normalization
on CPU. L2 requires real construction, trusted assets and a representative sample.
L3 needs simulator ownership and a bounded environment procedure. L4 needs a
frozen research comparison and explicit compute budget. Exact commands and limits
are owned by [tests/README](../../tests/README.md).

Accepted invariant: runtime never depends on harness. Proposed “core never depends
on simulator/WandB” rules are NOT accepted by this onboarding; occupancy training
already prevents a blanket models-package claim. Baseline config-copy isolation is
a tested technique, not an implemented general experiment-composition framework.

### Rejected complexity

No runtime refactor, package relocation, new component interface, config framework,
baseline code fork, metric database, services, custom orchestration, scheduler,
installer, updater, hooks, CI provider selection, merge policy, compulsory transcript
capsules, speculative research skills or automatic expensive validation.

## Implementation evidence

The [implementation record](../plans/completed/research-harness.md) records progress,
test results, skill evaluation, final inventory and remaining risks. The next task is to audit and redesign
runtime architecture under this harness, first characterizing compatibility and
known semantic subtleties—not immediately implementing a new research method.
