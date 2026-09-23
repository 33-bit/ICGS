# Portable ICGS Environment Design

Date: 2026-09-24  
Status: approved for implementation

## Goal

Make a fresh clone reproducibly installable on CPU or CUDA Linux hosts, with an
optional pinned RLBench/CoppeliaSim generation profile, and provide a verifier
that reports missing or extraneous runtime components without exposing secrets.

## Decisions

- `uv` plus `pyproject.toml` and `uv.lock` is the canonical Python installer.
- CPU and CUDA 11.8 are explicit profiles; neither silently substitutes for the
  other.
- Generation dependencies are an explicit optional profile. Simulator setup is
  idempotent, pinned to the existing CoppeliaSim checksum and PyRep/RLBench
  revisions.
- `environment.yml` is removed from the canonical setup path because it is a
  stale Conda export with unrelated packages and a historical `ip_env` name.
- HF/W&B credentials are external files or environment variables. Real values
  never enter the repository, setup receipt, command log, or worker environment.

## Required interfaces

`python scripts/setup_environment.py --profile cpu|cuda118|generation` must:

1. validate the host and Python version;
2. create or reuse a project virtual environment;
3. install the selected locked dependency profile and the editable package;
4. optionally provision pinned simulator assets for `generation`;
5. write a redacted JSON receipt with package/profile/platform identities.

`python scripts/verify_environment.py --profile ...` must emit machine-readable
JSON and a human-readable summary with PASS/FAIL/SKIPPED statuses for Python,
ICGS imports, core dependencies, PyG ABI, renderer, simulator, RLBench/PyRep,
and credential-path presence. It must never print credential contents.

## Compatibility and scope

- Existing serialized dataset/protocol/schema strings are unchanged.
- Existing CUDA inference requirements remain available as a documented profile.
- Existing generation scripts continue to consume explicit runtime configs.
- No model training, full generation, HF publication, or robot motion is part of
  setup verification.

## Acceptance

- A clean Linux clone can run setup and verification without Conda or `/content`
  assumptions.
- CPU VPS installation supplies PyTorch and all dependencies needed for the
  repository-owned test suite.
- Generation profile verifies Xvfb/Mesa, CoppeliaSim, PyRep and RLBench.
- VPS credential file remains mode `600`, is referenced by path only, and is not
  committed or included in receipts.
- Documentation gives exact clone/setup/verify commands and profile choices.
