---
name: encode-invariant
description: Use when asked to enforce an accepted architecture, reliability, compatibility, or quality rule, or prevent a documented violation from recurring. Not for choosing new research architecture from existing code patterns.
---

# Encode an accepted invariant

Read [AGENTS](../../../AGENTS.md), [workflow](../../../docs/WORKFLOW.md), the
accepted rule's owner, and [validation ownership](../../../tests/README.md).

## Establish the rule

Before writing a guard, produce a short boundary record: authority and exact rule;
scope; allowed case; forbidden case; authorized exceptions; diagnostic and next
action. If authority or a material choice is missing, report it and request the
smallest decision. Code organization, comments and existing tests are evidence of
behavior, not acceptance of a new boundary.

For this repository, [ADR 0001](../../../docs/decisions/0001-harness-boundary.md)
accepts runtime/harness independence. It does not authorize a package-wide ban on
Lightning, WandB or RLBench, nor classify GraphDiffusion as a new adapter layer.

## Implement and exercise

Use the smallest existing validation owner that can see the accepted scope.
Explain what it cannot see. Add an actionable failure naming the violating file,
rule owner and compliant next action. Use isolated fixtures for negative proof;
never leave deliberate violations in runtime files.

Run both an allowed case and the targeted violation. Verify that the latter fails
for the intended rule, not a missing dependency or syntax error. An empty scan or
a green unchanged repository does not establish detection. Keep integration proof
separate from static pattern checks.

Example: a runtime import of a root harness script must fail the static guard;
a test importing runtime is allowed. Computed paths and alias-based imports need
additional analysis; do not describe a literal scan as complete dependency proof.

## Handoff

Report authority/scope, changed validation owner, positive and negative results,
and limitations. Separately report local execution, optional hooks, checked-in CI
invocation and externally verified branch protection. Do not install hooks or
change CI/merge settings without separate authorization. Discovery is not enforcement.

Adapted from repository-harness encode-invariant at revision e765792;
[attribution and license](../../../docs/third-party-notices.md).
