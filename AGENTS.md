# Working in Instant Policy

Start with [the documentation map](docs/README.md). Before changing runtime code,
read [architecture](docs/ARCHITECTURE.md), [workflow](docs/WORKFLOW.md), and the
[original baseline](docs/baselines/instant_policy.md). Read the
[semantic contract](docs/components/policy-data-contract.md) for data, graph,
frame, action, or normalization changes.

- Classify the change using the workflow: small engineering, component experiment,
  or major migration. The workflow owns plan and decision-record triggers.
- Research changes follow [RESEARCH](docs/RESEARCH.md); baseline behavior,
  preprocessing, action/graph semantics, checkpoint loading, dataset format,
  seeds, and evaluation protocol must never change silently.
- Choose validation using [tests/README.md](tests/README.md).
  `python3 -B scripts/validate_fast.py` is cheap L0, not model execution.
- Report commands, environment, PASS/FAIL/SKIPPED/NOT RUN, skip reasons, and
  remaining risk using that validation guide.
- Preserve [harness/runtime independence](docs/decisions/0001-harness-boundary.md).
  Do not move runtime packages or introduce interfaces merely to fit this harness.
- Use [onboard-repository](.agents/skills/onboard-repository/SKILL.md) for requested
  onboarding, and [encode-invariant](.agents/skills/encode-invariant/SKILL.md) for
  accepted-rule enforcement. Neither authorizes speculative architecture changes.

Inspect Git state before edits. Existing untracked work belongs to the user.
Do not launch training, downloads, preprocessing, simulator workloads, or robot
motion as incidental validation. Advice, audits, and diagnoses remain read-only.
