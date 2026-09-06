---
name: onboard-repository
description: Use when explicitly asked to onboard, map, or audit this research repository or its agent-facing guidance. Not for ordinary code changes or automatic startup audits.
---

# Onboard a research repository

Read [AGENTS](../../../AGENTS.md), the [authority map](../../../docs/README.md),
and only relevant source/validation owners. If onboarding another repository,
find its actual owners rather than importing Instant Policy rules.

## Inspect and propose

Capture revision (or explicitly unborn Git state), dirty/untracked files, and
source provenance. Preserve existing work. Inspection alone does not authorize
edits, dependency installation, downloads, model loading or simulator startup.

Produce these concrete outputs:

1. A module map with source paths and training, evaluation and deployment traces.
   Distinguish runnable entry points from examples requiring user integration.
2. A baseline ledger separating source defaults, checkpoint configuration and
   entry-point overrides. Record missing artifact identities and environment
   evidence; do not equate source identity with reproduced performance.
3. A validation table: exact command/working directory, prerequisites, cost,
   observed result and remaining risk. Use the [validation owner](../../../tests/README.md).
4. FACT / ASSUMPTION / UNKNOWN findings. Distinguish observed implementation from
   accepted policy. Never define “core” or new dependency boundaries by inference.
5. Ranked proposals: prevented mistake, source evidence, destination owner,
   proposed change, unresolved choice, and proof that would exercise the change.

End with scoped proposals and a workspace-change report. Do not claim external
state equivalence unless it was measured. For an approved follow-up, recheck target
state and apply only the approved scope using the [workflow](../../../docs/WORKFLOW.md).
No transcript capsule, installer or agent-specific tool is required.

Example: if syntax passes but checkpoints are absent, report syntax PASS and
selected checkpoint validation SKIPPED with its asset reason—not “baseline ready.”

Adapted from repository-harness onboard-repository at revision e765792;
[attribution and license](../../../docs/third-party-notices.md).
