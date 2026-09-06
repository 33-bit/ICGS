# 0001: Keep the harness outside the runtime

Date: 2026-09-06
Status: accepted by the repository owner with the onboarding design.

## Context

Instant Policy will support future research changes. Repository guidance must help
agents without becoming a runtime framework or a prerequisite for using artifacts.

## Decision

Runtime code under `src/icgs/` and packaging in `pyproject.toml` must not depend on
`AGENTS.md`, `docs/`, `.agents/`, `tests/`, or root harness `scripts/`.
Deleting those harness surfaces must not break training, inference, evaluation,
checkpoint loading, data processing or simulator integration. Dependencies may
flow from tests/validation into runtime, never back.

The existing `ip/scripts/download_weights.sh` is an operational asset, not a
root harness script. Runtime imports of existing numerical, model, logging and
simulator dependencies are outside this decision; this ADR does not redesign
their boundaries.

## Alternatives considered

- Install upstream repository-harness: extra updater and managed state with no
  runtime benefit for this project.
- Introduce runtime factories/config loaders through the harness: creates coupling
  and preempts the separate architecture migration.
- Documentation only: insufficient executable feedback for common accidental
  reverse dependencies.

## Consequences

Use repository-owned Markdown, skills and local checks. The
[static validator](../../scripts/validate_fast.py) flags syntactic harness imports,
literal dynamic-import targets, and harness-shaped literal resource paths in
runtime Python. It covers common violations, not arbitrary computed paths,
indirect external reads, alias resolution or full transitive dependency semantics.
Review the full diff and use integration proof when those mechanisms change.

## Compatibility implications

No runtime bytes, package layout, config consumers, datasets or checkpoints change
during onboarding. Static checks are local tools; no hook, CI or branch protection
is installed. There are no exceptions permitting runtime dependence on the harness.
