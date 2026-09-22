# Generation Canonicalization Implementation Plan

**Category:** C — architecture/runtime migration
**Authority:** [ADR 0014](../../decisions/0014-canonical-generation.md)
**Design:** [canonical generation design](../../superpowers/specs/2026-09-22-generation-canonicalization-design.md)

## Goal

Make `icgs.data.collection.generation` and semantic `generation_*` entry points
the only supported data-generation runtime. Remove obsolete launch, recovery,
publication and debug tooling without changing serialized episode identities.

## Invariants

- Canonical runtime remains under `src/icgs` and does not import harness scripts.
- Worker, coordinator, queue, validator and publisher use only the generation
  namespace or shared runtime owners.
- Valid failures remain valid episodes; crash and invalid-observation attempts
  remain closed failure records.
- Existing payload identity strings remain readable. This migration changes
  source layout and APIs, not stored schema meaning.
- No simulator, training, publication or full generation is launched as
  incidental validation.

## Implemented migration

- [x] Renamed the collection package to
  `src/icgs/data/collection/generation/` and updated package metadata/imports.
- [x] Renamed protocol, compiler, manifest, schema and view APIs to semantic
  names without numeric generation suffixes.
- [x] Removed the worker's compatibility-collector seam; production execution
  now invokes the canonical episode worker directly.
- [x] Renamed all supported operational entry points to `generation_*` names.
- [x] Replaced the version-coded profile and composition artifact names with
  `generation.json`, `method.json` and `approved_composition_manifest.json`.
- [x] Removed obsolete sidecar, repack, backfill, exploratory, resume-receipt,
  session-debug and one-off Colab tools.
- [x] Removed tests that existed only for deleted compatibility paths and
  renamed the surviving generation tests semantically.
- [x] Added `docs/components/generation.md` as the current operational owner;
  launch audits are explicitly historical evidence.
- [x] Added an L0 naming invariant with allowed and forbidden fixtures. It
  checks Python filenames/import literals and intentionally does not reject
  serialized protocol strings or historical prose.
- [x] Deleted repository bytecode/cache debris from runtime, scripts, tests and
  generation artifacts.

## Compatibility and recovery

The old Python import and script names stop working by design. Existing episode
records retain their schema and protocol identifiers. Existing remote data is
not mutated by this migration. A later real run must use a fresh disjoint run ID
and validate any resume source against its immutable manifest.

## Verification checklist

- [x] Targeted invariant fixtures: canonical case accepted; legacy path and
  static/dynamic legacy import cases rejected with an ADR 0014 diagnostic.
- [x] Focused generation test suite passes on the current final tree.
- [x] Dependency-free L0 validator passes on the current final tree.
- [x] Full local test suite passes, with opt-in integration gates explicitly
  skipped and not counted as pass.
- [x] Every unavailable prerequisite is
  reported without counting skips as pass.
- [x] `git diff --check`, stale-reference scans and final Git review pass.
- [x] Exact commands, environment, counts and remaining risk are recorded below.

## Final evidence

Fresh evidence (2026-09-22, repository root, macOS, CPython 3.11.15 in
`.venv`, pytest 8.4.2):

- `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation.py tests/test_generation_*.py`
  — **PASS**, 118 passed.
- `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests` — **PASS**, 797 passed,
  10 skipped, 3 non-failing collection/deprecation warnings. The skipped tests
  are opt-in published/differential gates and are not counted as pass.
- `python3 -B scripts/validate_fast.py` — **PASS**: syntax, local links,
  harness boundary, canonical naming and 22/22 harness self-tests. L1/L2/L3/L4
  — **NOT RUN** by this migration check.
- `git diff --check` — **PASS**.
- Reference scan over `src`, `scripts`, `tests`, `artifacts`, `docs` found no
  stale runtime imports or operational paths; legacy spellings remain only in
  isolated negative fixtures, ADR naming tables, and serialized/historical
  provenance strings by design.

L1 simulator, L2 model, L3 simulator integration, L4 benchmark, Hugging Face
publication and full generation are **NOT RUN** for this source
canonicalization. Root-level `pytest -q` is not an acceptance command because
research outputs contain duplicate test module names; the repository-owned
`tests/` run above is the authoritative full local suite.
