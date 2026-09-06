# Research Harness Implementation Plan

Status: completed. Approved by the repository owner on 2026-09-06.

Goal: make Instant Policy navigable, baseline-aware, and safely verifiable without
changing research runtime behavior.

Architecture: additive Markdown authority, two adapted generic skills, a
standard-library static validator, and focused Python tests. Runtime never reads
the harness. No installer, config framework, dependency installation, or CI change.

Tech stack: Markdown, Git provenance, Python 3.10+ standard library; PyTorch and
NumPy only for the explicitly selected CPU smoke tests.

Spec: [approved audit and design](../../audits/harness-onboarding.md).
Execution: inline task-by-task with test-first validator development and review.
This plan is self-contained; no external agent plugin is needed to resume it.

## Current state and constraints

- Local root: `/Users/33bit/AI/Research/VLA/ICGS`.
- Unborn `main`, no remote, no tracked files; all original files are user-owned.
- 29 original files match upstream `65dc94e347df5bca4e390a6f959cd3308f8ae2bc`.
- Reference harness: `e765792b635b4d5e3e5fc0578f82f9ca5dea2681`.
- Preserve `ip/`, `setup.py`, `environment.yml`, `.gitignore`, and media bytes.
- Only original-file edit: add a navigation link to root README.
- No new research method, runtime interface, package move, or fabricated benchmark.
- No commit/staging of original untracked files. A worktree cannot be based on an
  absent local commit; approved edits are performed in place.

## Tasks and proof

### 1. Repository-owned documentation

- [x] Persist audit, source links, adoption/rejection rationale, unknowns.
- [x] Add authority map, concise AGENTS route, architecture, workflow, research
  policy, baseline definition, current semantic contract, one boundary ADR.
- [x] Add plan, decision, and experiment templates; no empty scaffolding.
- [x] Capture upstream SHA-256 inventory from the audited original files. It
  records provenance, not a rule that all future source changes are forbidden.

### 2. Static validator (test first)

Files: `scripts/validate_fast.py`, `tests/test_harness.py`, `tests/README.md`.
Interface: `check_syntax(root)`, `check_links(root)`, `check_boundary(root)` each
return diagnostic strings; CLI runs all three plus validator tests and returns
nonzero on failures. Accept `--root` for isolated fixture verification.

- [x] Write real temporary-tree tests for allowed files, bad Python, broken local
  links/anchors, reference links, forbidden runtime imports and literal harness
  resource paths, and CLI failure propagation. Fixtures never modify runtime.
- [x] Run `python3 -B -m unittest discover -s tests -p test_harness.py -v`; first
  failure must identify the missing validator, not a dependency problem.
- [x] Implement deterministic checks; ignore fenced examples and remote URLs for
  local link checking. Report static scope limits, not full dependency proof.
- [x] Run positive and negative tests. Run `python3 -B scripts/validate_fast.py`.

### 3. CPU characterization

File: `tests/test_cpu_smoke.py`. Test original config deep-copy isolation,
normalizer action/label round trips, and independently calculated horizon bounds.
No full model, RLBench, download, data loader, or GPU. Missing top-level packages
skip; broken installed imports fail. The original runtime is characterized, not
modified to force a test to pass.

- [x] Add smoke tests; run with `python3 -B -m unittest discover -s tests -p
  test_cpu_smoke.py -v`; distinguish skips from passes.
- [x] Exercise a temporary mutated normalizer to prove the assertions catch a
  relevant behavior change without editing `ip/`.

### 4. Adapt two generic skills

Files: `.agents/skills/{onboard-repository,encode-invariant}/SKILL.md`.

- [x] Run read-only baseline scenarios without adapted skills; retain outcomes.
- [x] Write and validate each short skill with correct relative owner links.
- [x] Test with a fresh consuming agent; report controls honestly, including when
  a control already behaves correctly. Do not claim measured improvement then.
- [x] Preserve upstream attribution and record removed capsule/installer coupling.

### 5. Verify and hand off

- [x] Run L0 and L1, skill metadata validation, documentation link checks, and
  upstream-byte comparison for all intentionally untouched files.
- [x] Review diagnostics, exact commands, authority overlap, and all added files.
- [x] Record model/environment checks as SKIPPED if prerequisites unavailable;
  expensive benchmark/training work is NOT RUN.
- [x] Record evidence here; move this plan to `completed/` with stable links.
- [x] Hand off files/tree, ownership, skills, validation, risks, and next task.

## Compatibility, recovery, and open questions

Checkpoint and dataset formats remain unchanged. Exact downloaded checkpoint
metadata, simulator revisions, original training-data provenance, and measured
performance are unknown. Do not invent them.

Recovery: remove only the additions listed in this plan and the added README
navigation section after inspecting current changes. Never reset this unborn
repository or delete its original untracked files. Runtime deletion-independence
will be established by unchanged source inspection plus bounded static guards;
L0 is not end-to-end execution proof.

## Decisions and progress

- 2026-09-06: user approved the pre-edit report and minimum implementation.
- 2026-09-06: implementation started; original source remains unchanged.

## Final evidence

Completed 2026-09-06. The approved additive harness exists, its local acceptance
checks pass, and all original runtime/packaging/environment/media bytes remain
unchanged. No research-method implementation or runtime refactor was performed.

### Final inventory (21 additions, one existing file modified)

```text
AGENTS.md
docs/
  README.md
  ARCHITECTURE.md
  WORKFLOW.md
  RESEARCH.md
  audits/harness-onboarding.md
  baselines/instant_policy.md
  baselines/upstream-files.sha256
  components/policy-data-contract.md
  decisions/0001-harness-boundary.md
  decisions/TEMPLATE.md
  plans/TEMPLATE.md
  plans/completed/research-harness.md
  experiments/TEMPLATE.md
  third-party-notices.md
.agents/skills/
  onboard-repository/SKILL.md
  encode-invariant/SKILL.md
scripts/validate_fast.py
tests/
  README.md
  test_harness.py
  test_cpu_smoke.py
```

Modified: root README, exactly six added navigation lines. Intentionally unchanged:
all ip files, setup.py, environment.yml, .gitignore and media (28 original files).
No empty active-plan scaffolding is retained. Third-party notices are the one
small addition to the proposed tree, needed to retain upstream license attribution.

Authority map: [docs/README](../../README.md). It routes architecture, workflow,
research policy, original baseline, semantic contracts, decisions and validation
to separate owners. AGENTS is 26 lines and references those owners.

### Skills and behavior checks

Adapted: onboard-repository and encode-invariant. Adopted verbatim: none.
Optional/not installed: engineering-wisdom, audit-onboarding-proposal.
Conditional/not installed: improve-harness, only after repeated observed friction.
Rejected for this installation: mandatory capsule machinery, upstream installer/
updater coupling, speculative research skills. Full classifications and reasoning
are in the [audit](../../audits/harness-onboarding.md).

The skill-authoring checks used a read-only no-guidance agent followed by fresh
consuming agents for each adaptation. The control already rejected “syntax pass
means baseline-ready,” and correctly labeled its narrower core-boundary proposal
as requiring agreement. This is not a demonstrated failure or measured skill gain.
The forward onboarding agent correctly separated source identity, checkpoint config
and overrides, and reported then-missing in-progress owner links as FAIL rather
than hiding them. Those links were subsequently created and all L0 checks rerun.
The invariant agent enforced only ADR 0001, refused to infer a simulator/logger-free
core, and distinguished local evidence from external enforcement. Its suggested
boundary-only CLI test was added and passed. These are bounded application checks,
not a repeated-trial effectiveness benchmark or proof about every coding agent.
Both skill metadata validators passed; links are included in local L0.

### Commands and results

Working directory for commands below: repository root, unless explicitly noted.
Host: macOS 15.7.2 arm64, Python 3.14.4, torch 2.11.0, NumPy 2.4.4; CUDA unavailable.
This is NOT the baseline's declared Linux/Python 3.10/PyTorch 2.2 environment.

| Command / observation | Result |
| --- | --- |
| `python3 -B scripts/validate_fast.py` | PASS: syntax compilation, local links, static boundary, 14 harness tests, no skips |
| `python3 -B -m unittest discover -s tests -p test_cpu_smoke.py -v` | PASS: four CPU characterization tests, no skips |
| `python3 -B -S -m unittest discover -s tests -p test_cpu_smoke.py -v` | Four SKIPPED with explicit torch/NumPy absence under no-site interpreter; confirms skip behavior, not an L1 PASS |
| `python3 -B /Users/33bit/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/onboard-repository` | PASS: skill metadata |
| Same quick_validate.py command with `.agents/skills/encode-invariant` | PASS: skill metadata |
| `awk '$2 != "README.md"' docs/baselines/upstream-files.sha256 \| shasum -a 256 -c -` | PASS: all 28 untouched originals |
| `shasum -a 256 -c docs/baselines/upstream-files.sha256` | Expected FAIL for original README only; 28/29 match |
| `diff -u /tmp/instant-policy-audit.KFDXEn/instant_policy/README.md README.md` | Only approved six-line addition; expected diff exit 1 |
| `diff -rq ip /tmp/instant-policy-audit.KFDXEn/instant_policy/ip` | PASS: no runtime differences |
| `bash -n ip/scripts/download_weights.sh` | PASS: syntax only, no download |
| `git status --short`, `git ls-files`, `git diff --check` | Repository still unborn/untracked; no staging. diff --check alone is vacuous here and is not used as content proof |
| Dependency find_spec and artifact-path existence probes | lightning/diffusers/open3d/rlbench/pyrep absent; ip/checkpoints and ip/data absent |

Initial validator tests failed as intended before the script existed. Additional
red tests exposed context-invalid Python (top-level return) and undefined Markdown
reference labels; both were implemented and verified green. Valid and forbidden
fixtures exercise actual checks and CLI exit behavior without importing runtime.

Normalizer mutation proof: streamed an in-memory source variant replacing horizon
linspace(1,P,P) with linspace(1,1,P), substituted it only into the test namespace,
and ran the independent horizon-bound test. It failed at the expected numeric
assertion (6/12 elements differ); the enclosing probe reported PASS: mutation
detected. No file in ip was edited, and subsequent original L1 tests passed.

Independent implementation review found no critical or important remaining issue.
Its minor clarification about fixture subprocesses was verified and corrected in
tests/README. Later root L0 reruns included all 14 tests.

### Validation limits and outstanding research evidence

- L0 and narrow L1: PASS on the host above. Static checker limitations are explicit.
- L2: SKIPPED; full research dependencies, trusted checkpoints/encoder and a
  representative model fixture are unavailable. Checkpoint compatibility and
  policy inference are not certified.
- L3: SKIPPED; RLBench/PyRep/CoppeliaSim and rendering/model assets unavailable.
- L4, full training, large preprocessing, downloads and robot operation: NOT RUN;
  outside this onboarding's local acceptance criteria and compute authorization.
- No dependency installation, CI/hook setup, branch protection, commit, push or
  model/runtime migration was performed.
- Unrelated .DS_Store metadata appeared at root, docs and docs/plans during work.
  These files were not part of the harness and were left untouched. External state
  equivalence is not claimed.

### Baseline, workflow and next task

Baseline strategy: pinned source plus historical hash inventory, complete source
configuration, explicit entry overrides, current semantic map and artifact/protocol
unknowns. No duplicate implementation or new config consumer. Category A stays
lightweight; B records a controlled research comparison; C keeps a resumable plan
and records durable decisions. Validation tiers and reporting live in tests/README.

Remaining unknowns: original checkpoint config/hashes, data/splits, encoder-training
recipe, simulator revisions, seeds, expected numerical performance and tolerances.
Remaining debt: source-visible loader/rollout retry behavior, recording branch bug,
stale occupancy code, example-only preprocessing/deployment, hardcoded horizons,
frame/time-embedding/sampler subtleties. See audit for exact source evidence.

Recommended next task: audit and redesign Instant Policy runtime architecture under
this harness, beginning with characterization fixtures and compatibility boundaries.
Do not immediately implement new research methods. Git integration remains with
the user: original files were untracked before work and have not been staged.
