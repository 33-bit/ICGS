# Portable ICGS Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Provide portable CPU/CUDA/generation setup and verification from a clean clone.

**Architecture:** Keep Python dependency ownership in `pyproject.toml` and `uv.lock`; expose setup profiles through one idempotent CLI. Keep simulator provisioning separate but callable from the generation profile, and write redacted receipts consumed by a verifier and the VPS acceptance record.

**Tech Stack:** Python 3.10–3.12, uv, setuptools, pinned PyTorch/PyG wheels, apt/Xvfb/Mesa, pinned CoppeliaSim/PyRep/RLBench.

**Spec:** `docs/superpowers/specs/2026-09-24-portable-environment-design.md`

## Global Constraints

- Canonical installer is `uv` plus `pyproject.toml`/`uv.lock`.
- CPU and CUDA 11.8 are explicit profiles.
- Real HF/W&B credentials never enter Git, receipts, logs, or worker environments.
- Existing serialized protocol/schema strings remain unchanged.
- Setup verification never launches full generation or publishes to HF.

### Task 1: Dependency/profile contracts

**Files:**
- Modify: `pyproject.toml`, `.env.example`, `README.md`
- Remove: `environment.yml`
- Test: `tests/test_environment_setup.py`

- [ ] Add explicit optional dependency groups and profile metadata.
- [ ] Add failing tests for profile names, Python floor, credential-path redaction, and stale Conda exclusion.
- [ ] Implement the contract without changing runtime semantics.
- [ ] Run the focused tests and commit.

### Task 2: Portable setup and verification CLIs

**Files:**
- Create: `scripts/setup_environment.py`
- Create: `scripts/verify_environment.py`
- Modify: `scripts/generation_environment.py`
- Test: `tests/test_environment_setup.py`, `tests/test_generation_environment.py`

- [ ] Add failing tests for dry-run command planning, idempotent receipt writing, and missing-package verification.
- [ ] Implement profile-aware setup with injectable command runner; default to safe host checks and explicit simulator opt-in.
- [ ] Implement redacted receipts and verifier status output.
- [ ] Run focused tests and commit.

### Task 3: Documentation and clean-clone proof

**Files:**
- Modify: `docs/README.md`, `docs/components/generation.md`, `tests/README.md`
- Create: `docs/components/environment.md`

- [ ] Document CPU, CUDA 11.8, and generation setup commands, credential handling, and troubleshooting.
- [ ] Add a clean-clone/static proof that no canonical instruction requires `/content`, Conda, or `ip_env`.
- [ ] Run L0, focused tests, and diff checks; commit.

### Task 4: VPS installation and acceptance record

**Files:**
- Modify: `docs/experiments/generation-validation/validation-cpu-20260922/README.md`

- [ ] Pull the GitHub commit on `vps-a`.
- [ ] Install the CPU + generation profile, retain `/home/huy2325/.icgs_hf_token` mode `600`, and run verifier.
- [ ] Run repository tests available on the VPS, simulator readiness, and bounded gates without full generation.
- [ ] Record exact PASS/FAIL/SKIPPED/NOT RUN evidence and remaining limits.

## Validation commands

```bash
python -B -m pytest -q tests/test_environment_setup.py tests/test_generation_environment.py
python3 -B scripts/validate_fast.py
git diff --check
```

The VPS acceptance must additionally report the Python version, package matrix,
OS, simulator revisions/checksum, credential mode/path presence, and any missing
GPU/model dependencies.
