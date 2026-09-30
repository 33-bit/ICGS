# Documentation map

This repository is the system of record. Read the smallest relevant owner below.
[AGENTS](../AGENTS.md) routes agents; it is not a second specification.

| Owner | Knowledge owned |
| --- | --- |
| [Root README](../README.md) | Installation and user quickstart |
| [ARCHITECTURE](ARCHITECTURE.md) | Current modules, dependency direction, execution paths |
| [WORKFLOW](WORKFLOW.md) | Change categories, planning/decision triggers, handoff |
| [RESEARCH](RESEARCH.md) | Baseline-change policy, experiments, reproducibility expectations |
| [Original baseline](baselines/instant_policy.md) | Historical source/defaults, entry-point overrides, assets and reproduction limits |
| [Policy data contract](components/policy-data-contract.md) | Current data, graph, frame, action and normalization semantics |
| [RoboHiMan backbone](components/robohiman.md) | Primary task/environment path under migration: pinned simulator venv, Stage-1 records, instrumentation, gates |
| [RoboHiMan decision](decisions/0016-robohiman-backbone.md) | RoboHiMan as upstream backbone; ICGS-owned instrumentation; custom generator non-primary |
| [RoboHiMan migration plan](plans/active/robohiman-backbone-migration.md) | Phases, go/no-go gates and recovery for the backbone switch |
| [Migration classification](audits/2026-09-30-robohiman-migration-classification.md) | Generic vs custom-generator vs mixed components before refactor |
| [RoboHiMan validation evidence](experiments/robohiman-validation/README.md) | Parity, A0, replay, dependency, split-overlap and smoke-collection results |
| [RoboHiMan validation round 2](experiments/robohiman-validation/v2/README.md) | Monitor families, anchor replay fidelity and acceptance rule, n=20 dependency audit, storage cadence |
| [Data generation](components/generation.md) | Custom 36-program generator (legacy/reference since ADR 0016; not scaled): modules, HF archive lifecycle, record classes, resume and view finalization |
| [Generation artifacts](components/generation-artifacts.md) | Legacy v2 and lossless HF archive trees, file inventories, consumers and retention |
| [Environment setup](components/environment.md) | Portable CPU/CUDA/generation profiles, verification and credential boundary |
| [Composition examples](components/composition-examples.md) | Current component replacement APIs and future extension boundaries |
| [Boundary decision](decisions/0001-harness-boundary.md) | Runtime independence from repository harness |
| [Runtime composition decision](decisions/0002-runtime-composition.md) | Accepted core dependency direction and composition/config strategy |
| [Checkpoint decision](decisions/0003-checkpoint-compatibility.md) | Legacy translation, strictness and diagnostic policy |
| [Unified ICGS decision](decisions/0004-unified-icgs-v5.md) | Canonical src/icgs namespace and mandatory published gates |
| [Published fidelity decision](decisions/0005-published-profile-fidelity.md) | Evidence-backed live preprocessing and RNG-order profile |
| [V5 foundation status](components/v5-foundations.md) | Implemented/designed/deferred method-level ownership |
| [Target ICGS method](method/README.md) | Approved target, neural/data/search specifications; not implemented status |
| [ICGS parameters](method/parameters.md) | Central JSON defaults, units, locks, overrides and consumer readiness |
| [Observability design](components/observability.md) | Approved local logging/tracing/capture design and current implementation boundary |
| [Observability plan](plans/active/icgs-observability.md) | Recorder, debugging tools, CLI/training integration and acceptance |
| [Observability boundary decision](decisions/0010-local-observability.md) | Accepted local-first recorder, retention and optional-backend boundary |
| [Configuration decision](decisions/0009-central-method-configuration.md) | Packaged defaults and separate resolved/reference identities |
| [Proposal provenance](proposals/README.md) | Immutable uploaded source, date and SHA256 |
| [Target requirements](method/requirements.md) | Proposal-to-specification/plan/test traceability |
| [ICGS implementation roadmap](plans/active/icgs-method-implementation.md) | P00–P13 component plans, dependencies and pilot gates |
| [Target boundaries decision](decisions/0006-icgs-target-boundaries.md) | Added components and native preservation |
| [Timed data decision](decisions/0007-icgs-timed-data-lineage.md) | New control protocol, executed data and replay provenance |
| [Reference/stopping decision](decisions/0008-reference-value-and-stopping.md) | Frozen continuation target separate from learned stopping |
| [CLI and data](components/cli-and-data.md) | Explicit config paths, command migration, safe input/output schema |
| [Published acceptance evidence](experiments/vv19-validation/README.md) | Actual Colab checkpoint/inference/fidelity results |
| [Decision template](decisions/TEMPLATE.md) | Format for new durable decisions |
| [Plan template](plans/TEMPLATE.md) | Optional fields for resumable work |
| [Experiment template](experiments/TEMPLATE.md) | Lightweight research-contract record |
| [Validation guide](../tests/README.md) | Validation tiers, ownership, commands, result vocabulary |
| [Onboarding audit](audits/harness-onboarding.md) | Dated evidence, design approval, adoption choices and discovered debt |
| [Generation contract audit](audits/2026-09-20-generation-contract.md) | Historical collection quota, splits, schema, perturbations and views |
| [Generation launch record](audits/2026-09-21-generation-launch.md) | Historical distributed launch evidence and publication receipts |
| [Resumable multi-host generation](plans/active/resumable-multihost-generation.md) | Current HF archive resume, shared-filesystem worker leases and host-scoped launch contract |
| [Generation storage and view finalization](plans/active/generation-storage-and-view-finalization.md) | HF source-of-truth lossless archive, bounded local retention and frozen training-view plan |
| [Generation storage decision](decisions/0015-generation-storage-and-view-snapshots.md) | Accepted HF archive, binary deduplication and provisional/final view boundary |
| [Generation launch readiness](plans/completed/generation-launch-readiness.md) | Run-wide storage admission, fresh-host task build and selected-host acceptance |
| [Generation readiness evidence](experiments/generation-validation/readiness-20260927/README.md) | VPS-only commands, live smoke and bounded publication/backpressure evidence |
| [Colab archive capacity experiment](experiments/generation-validation/archive-v6e1-stress-20260927.md) | OAuth2 V6e-1 concurrency comparison, publication service rate and quota-aware ETA |
| [Collector semantics audit](experiments/generation-validation/collector-semantics-20260929.md) | 36-program physical/label audit: measured timing, no snaps/freezes, step goals, clearance layouts |
| [Third-party notices](third-party-notices.md) | Attribution and license for adapted guidance |

Source code owns implementation detail; tests establish only the behavior they
actually exercise. Current accepted decisions and research policy own intended
constraints. If observed code conflicts with accepted intent, report the conflict:
do not silently rewrite either into agreement. Dated audits and completed plans
are historical evidence, not new authority.

Active plans live at `docs/plans/active/<slug>.md`; completed plans at
`docs/plans/completed/<slug>.md`. Create directories only for real records.
Component documents describe research-critical boundaries, not every class.
Experiments receive their own Markdown record only when a real experiment exists.

For the approved proposal migration, read the target method and roadmap together.
Current architecture remains factual; target specifications use explicit proposal,
decision, implementation-default and feasibility labels. Future component plans
are active work, not evidence that models were trained or benchmarks passed.

The harness consists of the agent instructions, docs, skills, tests, and root
validation script. It is not an installer or an experiment-tracking platform.
