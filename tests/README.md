# Validation ownership and evidence

This file owns validation tiers, commands and reporting. The repository originally
had no automated tests. Checks now cover harness integrity, modular composition,
semantic CPU contracts, checkpoint translation and opt-in model integration,
not baseline research performance.

## Tiers and selection

| Level | What to check | Prerequisites / cost | When |
| --- | --- | --- | --- |
| L0 static/harness | Syntax, local docs links, syntactic harness boundary, validator positive/negative tests | Python 3.10+ standard library; seconds | Every harness or Python change |
| L1 CPU smoke | Config construction/copy isolation, action/label normalization and horizon bounds | Importable PyTorch + NumPy; seconds | Config, normalizer or related contract changes |
| L2 model integration | Actual construction, checkpoint loading, trusted representative fixture, finite/shape-correct inference | Compatible research stack, encoder/policy assets, CUDA where necessary; minutes/resource-dependent | Model, graph, diffusion, checkpoint or data-contract changes |
| L3 environment integration | Imports, simulator startup, real observation and action conversion, bounded rollout | RLBench/PyRep/CoppeliaSim, rendering and model assets; minutes or longer | Environment/adapter/preprocessing changes affecting observations or commands |
| L4 research benchmark | Frozen protocol, multi-seed task evaluation, matched baseline/ablation comparison | Full environment, artifacts and explicitly agreed compute budget; expensive | Scientific performance claims |

L0 is not import validation. L1 is not a CPU full-model guarantee. L2–L4 have no
general automated runner or repository fixture yet: specify the required bounded
proof in the task/experiment before running. Never run “all tests” as shorthand
for full training or a benchmark.

## Canonical cheap commands

Run from the repository root (or use absolute script paths):

```bash
python3 -B scripts/validate_fast.py
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```

The first command runs L0 only (including architecture closure tests) and exits 1 on a check/test failure. It reports
L1–L4 as NOT RUN. `--root <fixture-root>` inspects an isolated tree but intentionally
does not run self-tests; it cannot certify that the installed harness is complete.

Individual validator tests:

```bash
python3 -B -m unittest discover -s tests -p test_harness.py -v
```

The legacy test_cpu_smoke.py skips when torch or numpy cannot be found;
installed-but-broken imports fail. The full CPU suite requires real torch, NumPy,
SciPy and PyG; use L0 alone on a dependency-free host.
Unittest may exit zero with skipped tests: read its counts and report those tests
as SKIPPED, never a CPU validation pass. No fake torch/PyG modules are substituted.
Python `-B` avoids bytecode files. Tests create/remove only their own temporary
fixture directories outside the repository.

## Check ownership and exact scope

- [test_harness.py](test_harness.py): exercises the real validator with bad syntax,
  missing file/heading targets, valid links, reverse dependencies, runtime resource
  literals, empty trees and CLI failure propagation.
- [test_cpu_smoke.py](test_cpu_smoke.py): characterizes original config and Normalizer
  behavior without loading the full policy. Independent numeric bounds supplement
  round trips, which alone could miss a symmetric normalization bug.
- [validate_fast.py](../scripts/validate_fast.py): parses repository-owned Python
  under ip/scripts/tests/docs/.agents plus root Python files. Syntax checking
  compiles in memory without executing scanned files or writing bytecode. It does
  not install packages. Its self-tests run bounded Python CLI subprocesses against
  owned fixtures; no research entry point is launched.
- The boundary scan covers runtime Python under ip and setup.py, not shell,
  arbitrary generated code or external libraries. It flags harness-root imports,
  literal dynamic-import targets and exact harness-shaped string literals. It does
  not resolve aliases, compute paths, prove transitive third-party isolation, or
  analyze every filesystem operation. A diagnostic is a reviewable violation
  candidate; prose literals can require inspection. Rule authority is
  [ADR 0001](../docs/decisions/0001-harness-boundary.md).
- Link checking supports simple inline Markdown links/images, reference definitions,
  ATX headings (including repeated heading suffixes), and HTML src/href. It skips
  fenced examples and remote URLs. It is not a full CommonMark renderer, does not
  verify external availability or free-form backtick references, and does not
  validate arbitrary HTML anchors. Keep maintained links in supported syntax.
- No model/runtime module imports tests or the validator. Removing harness surfaces
  must preserve runtime operation; static results are bounded evidence, not a
  replacement for appropriate integration checks.
- No optional hook, CI job or external branch-protection requirement was installed.
  A local PASS does not mean a merge-blocking check exists.

## L2 prerequisites and bounded evidence

Start with the [baseline artifact requirements](../docs/baselines/instant_policy.md).
Do not deserialize untrusted pickle/checkpoint data. Record torch/PyG/Lightning/
Diffusers/Open3D versions, device, model and encoder checksums, resolved checkpoint
config, loading strictness and any overrides.

For an authorized integration task, construct the actual GraphDiffusion with the
checkpoint config, batch one and no compilation; load the original scene encoder
and checkpoint in the same way as eval.py. Build or load one trusted PyG fixture
covering the [semantic contract](../docs/components/policy-data-contract.md).
Record construction/load exceptions; validate actual action/grip shapes, finite
values, transform properties and seed-dependent repeatability/tolerances before
claiming inference passed. The repository does not yet supply that fixture, so
this is a test design requirement, not a claimed runnable command.

Even version/import checking needs real dependencies. In a provisioned environment,
`python -B ip/sandbox.py` prints torch/CUDA/PyG/Lightning versions only; it does not
validate Open3D, Diffusers, checkpoint compatibility or inference.

## L3 controlled simulator operation

Before startup, record exact RLBench/PyRep/CoppeliaSim revisions, rendering
availability, task/demo counts, model artifacts, time budget and ownership of the
simulator process. The current environment code retries some failures indefinitely
and does not guarantee cleanup on exceptions. Use an operator-controlled deadline;
stop only processes owned by this run. Do not use broad kill/cleanup commands.

After package installation, this import-only probe does not launch a simulator:

```bash
python -B -c 'import rlbench; import pyrep; from ip.utils.rl_bench_utils import rollout_model; print("PASS: environment imports only")'
```

Only with the prerequisites and explicit simulator execution approval, an existing
minimal-rollout command is:

```bash
cd ip
python -B eval.py --task_name=plate_out --num_demos=2 --num_rollouts=1 --restrict_rot=1 --compile_models=0
```

It still collects demonstrations, can hang in retries, and may be costly. One
rollout is NOT a benchmark or a deterministic simulator smoke. The CLI does not
expose a seed, headless flag, max-step limit or timeout; do not invent those flags.
Return to repository root before running root-relative checks.

Robot deployment is never implicit validation: deployment.py contains placeholders
and no controller safety system. Physical robot actions require a separately
authorized integration procedure.

## L4 research benchmark

Create a real [research contract](../docs/experiments/TEMPLATE.md). Freeze baseline,
tasks/variations, demonstrations, seeds, data, checkpoints, rotation restriction,
execution/prediction horizon, diffusion steps, hardware and metric aggregation.
The original repository does not provide a complete benchmark matrix or seed CLI.
Do not claim paper reproduction from the example task list or a single rollout.

## Reporting vocabulary

- PASS: the named check executed and satisfied its assertions.
- FAIL: the check executed and failed, including an installed dependency that fails
  to import. Include the diagnostic.
- SKIPPED: a selected check intentionally could not execute because its prerequisite
  was unavailable; give the specific dependency/asset/environment reason.
- NOT RUN: the check was not selected or attempted.

Every handoff includes command, working directory, interpreter/environment, result
and counts, skip/failure reasons, and remaining risk. A summary cannot upgrade a
skipped required check to passed. Initial expected red tests and negative fixtures
are test-development evidence, not unresolved runtime defects.

## Modularization tests and reproducible commands

| Test owner | Evidence |
| --- | --- |
| test_architecture.py | Accepted local import closure, direct/indirect violations and package initializers; stdlib L0 |
| test_config.py | Frozen baseline, independent tensor conversion, profiles, custom selectors, resolved metadata |
| test_geometry.py / test_cpu_smoke.py | Original pose/SVD/action/normalization CPU behavior |
| test_composition.py | Real PyG graph/heads forward/backward with injected encoder/stages, graph replacement, legacy graph config, preflight |
| test_policy.py | Sampler swap, original loop/objective with controlled schedule, context isolation/cache reset, observation and evaluator API |
| test_data.py | Real PyG serialization/caches and target semantics; Open3D path conditional |
| test_evaluation.py | Public policy/fake-environment metric/horizon/action contract; RLBench task resolution conditional |
| test_checkpoints.py | Generated raw/Lightning states, equal/unequal aliases, compiled keys, strictness, shapes, source immutability |
| test_loading.py | Resolved demo override survives artifact load/context/prediction; standalone encoder selector rejects unknown IDs |
| test_training.py | Real Lightning wrapper registration; skipped without Lightning |
| test_differential.py | Opt-in verified original-source method comparison; no duplicate runtime implementation |
| test_model_integration.py | Explicit opt-in trusted original artifacts and CUDA inference |

The architecture rule is [ADR0002](../docs/decisions/0002-runtime-composition.md).
Its local graph includes syntactic imports and parent initializers; it does not
prove arbitrary dynamic imports or third-party internals. It is run by the canonical
L0 command; no CI or branch protection was added.

With a trusted original-source checkout at the pinned baseline revision:

```bash
IP_LEGACY_SOURCE_ROOT=/path/to/original-checkout python3 -B -m unittest discover -s tests -p test_differential.py -v
```

The test checks historical SHA-256 values before executing selected definitions.
It compares all original config fields, exact graph edges/seeded parameters, and
denoiser/labels/sampler under controlled components/schedule. This does not exercise
original PointNet/FPS kernels or DDIM behavior on CUDA. The refactor's before-image
can be used locally; it is not a permanent required runtime/harness resource.

Only after trusting/provisioning checkpoint assets and the original stack:

```bash
IP_RUN_MODEL_TESTS=1 IP_CHECKPOINT_DIR=/absolute/trusted/checkpoints python3 -B -m unittest discover -s tests -p test_model_integration.py -v
```

Missing prerequisites/assets explicitly skip. Available dependencies that fail
construction/import/load/assertions fail. The test uses original model composition
and synthetic clouds, with only an explicit encoder artifact-path override; it does
not generate a benchmark or download assets. No simulator test launches implicitly.

On the development host, PyTorch/PyG emit Python-3.14 JIT deprecation warnings;
these are recorded dependency warnings, not numerical parity proof or test failures.
The original declared stack is still recommended for research execution.

An attempted original-scene construction during selector regression testing failed
in the host PyG typing inspector (`typing.Union` has no `_name`). This is an observed
host dependency failure, not a successful original-component check. The selector
test now rejects unknown IDs before construction; it does not hide or fix that
separate original-encoder compatibility issue.
