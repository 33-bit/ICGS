# 0004: One ICGS runtime and mandatory published-checkpoint gates

Date: 2026-09-06. Status: accepted by explicit user v5 migration brief.

## Context

The method is ICGS; IP is one policy plus internal neural/diffusion components.
Prior work centered the ip namespace and allowed unverified published compatibility.
The new task explicitly replaces those choices.

## Decision

Canonical runtime is src/icgs. Migrate actual owners and callers, not a wrapper
around an IP subtree. Remove old import/CLI compatibility packages once callers
migrate, preserving user artifacts/data. Use one pyproject/config/composition path.
Source layout and state-dict ownership remain separate concerns.

Supersedes ADR0002's keep-ip/layout/legacy-import clauses and ADR0003's requirement
to retain old import adapters. Does not supersede numerical preservation, explicit
diagnostics or harness independence. The target vv19 checkpoint path is strict-only;
prior deployment non-strict behavior is not permitted for its acceptance.

C1 artifact/config, C2 strict native load, C3 real inference, C4 target fidelity and
C5 standalone installed execution are mandatory before this migration is complete.
No test-double output, alternative checkpoint or skipped check substitutes for them.
Target-evidenced compatibility fixes may correct differences from the old ICGS
snapshot but require explicit evidence/tests, not silent mathematical rewriting.

## Alternatives and consequences

Keeping ip beside icgs or vendoring a monolith is explicitly rejected by the owner.
Src layout costs import/CLI/test migration but gives one method-level system.
Published binary may be used only as an external reference, never native dependency.
No retraining, new paid compute or broad benchmark is authorized.

## Compatibility and state

World/local/root action semantics remain; caches are session-owned, immutable demo
content is distinct from mutable branch history, and seeded calls restore RNG.
Whole-object old pickle compatibility is not promised. Missing reference hardware/
essential information yields INCOMPLETE, not a lowered completion threshold.
See [completed plan](../plans/completed/icgs-v5-unified.md).
