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
| [Composition examples](components/composition-examples.md) | Current component replacement APIs and future extension boundaries |
| [Boundary decision](decisions/0001-harness-boundary.md) | Runtime independence from repository harness |
| [Runtime composition decision](decisions/0002-runtime-composition.md) | Accepted core dependency direction and composition/config strategy |
| [Checkpoint decision](decisions/0003-checkpoint-compatibility.md) | Legacy translation, strictness and diagnostic policy |
| [Unified ICGS decision](decisions/0004-unified-icgs-v5.md) | Canonical src/icgs namespace and mandatory published gates |
| [Published fidelity decision](decisions/0005-published-profile-fidelity.md) | Evidence-backed live preprocessing and RNG-order profile |
| [V5 foundation status](components/v5-foundations.md) | Implemented/designed/deferred method-level ownership |
| [Target ICGS method](method/README.md) | Approved target, neural/data/search specifications; not implemented status |
| [ICGS parameters](method/parameters.md) | Central JSON defaults, units, locks, overrides and consumer readiness |
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
