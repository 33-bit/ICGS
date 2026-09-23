# Generation Acceptance Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining bounded generation acceptance gaps with reproducible VPS evidence without starting production full generation.

**Architecture:** Reuse the canonical launcher, coordinator, worker, watchdog, simulator probe, and HF publication paths. Run each failure/recovery scenario in a unique validation namespace, preserve credentials only on the VPS, and record receipts in the repository without changing serialized protocol/schema strings.

**Tech Stack:** Python 3.10/3.12, uv environment, CoppeliaSim/PyRep/RLBench, pytest/unittest, POSIX shared filesystem, Hugging Face Hub.

**Spec:** `docs/plans/active/generation-runtime-hardening.md` and `docs/plans/active/resumable-multihost-generation.md`

## Global Constraints

- Never modify production HF prefixes or launch the full production quota during acceptance.
- Never commit or copy credentials; workers must remain credential-free.
- Use a fresh disjoint run ID and validation prefix for every scenario.
- Keep serialized protocol/schema strings unchanged.
- Report `PASS`, `FAIL`, `SKIPPED`, and `NOT RUN` distinctly; skipped evidence never closes a gate.

### Task 1: Reconcile VPS environment and acceptance runner

**Files:**
- Modify: `docs/plans/active/generation-runtime-hardening.md`
- Modify: `docs/plans/active/resumable-multihost-generation.md`
- Create: `docs/experiments/generation-validation/validation-vps-20260924/README.md`
- Create: `docs/experiments/generation-validation/validation-vps-20260924/acceptance_receipt.json`

- [ ] Confirm remote checkout revision, clean source state excluding VPS-local `.icgs/`, installed interpreter, simulator probe, and focused generation tests.
- [ ] Record exact commands and prerequisites before running any fault scenario.
- [ ] Record environment limitations separately from runtime gate results.

### Task 2: Execute live bounded fault/recovery gates

**Files:**
- Create: `docs/experiments/generation-validation/validation-vps-20260924/fault-gates.json`
- Create: `docs/experiments/generation-validation/validation-vps-20260924/receipts/`

- [ ] Run malformed-result injection and verify quarantine diagnostics, accounting, and continued progress.
- [ ] Run forced infrastructure/simulator-crash materialization and verify no episode/quota corruption.
- [ ] Force one controlled coordinator restart and verify queue reconstruction, lease fencing, and continued publication.
- [ ] Exercise invalid-observation materialization and verify it remains distinct from valid failure.
- [ ] Remove only the local validation `run_root`, resume from the pinned HF manifest with a disjoint run ID, and verify no duplicate immutable identities.
- [ ] Attach a second host in workers-only mode and verify host-scoped lease/heartbeat without coordinator duplication.

### Task 3: Close gates and document readiness

**Files:**
- Modify: `docs/plans/active/generation-runtime-hardening.md`
- Modify: `docs/plans/active/resumable-multihost-generation.md`
- Modify: `docs/experiments/generation-validation/validation-vps-20260924/README.md`

- [ ] Run L0 and focused/full repository checks available in the VPS environment; disclose missing model dependencies instead of substituting fakes.
- [ ] Verify all required bounded fault/recovery gates have live receipts.
- [ ] Keep production full generation and L4 benchmark marked `NOT RUN` unless separately authorized.
- [ ] Commit only evidence/docs and implementation fixes required by failed acceptance checks.

