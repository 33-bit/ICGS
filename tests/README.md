# Validation ownership and evidence

Canonical runtime is src/icgs. L0 must scan that root and reject empty/missing
source, forbidden dependencies and old ip/binary imports; a moved tree cannot
produce a false-green check by disappearing from discovery.

## Tiers and canonical commands

| Tier | Purpose | Prerequisites |
| --- | --- | --- |
| L0 | Syntax, doc links, src/package boundaries, local dependency closure including initializers, negative fixtures | Python standard library |
| L1 | Contracts/config/geometry, real PyG synthetic composition, data IO, candidate/context/RNG, evaluator and checkpoint translation | Installed ICGS and actual dependencies |
| L2 / C1–C5 | Actual published checkpoint, full native encoder/graph/sampling, reference fidelity, installed standalone command | Pinned Linux Python3.10/CUDA11.8 and trusted published artifact |
| L3 | RLBench integration/task smoke | Separately provisioned simulator |
| L4 | Research benchmark and matched statistical comparison | Explicit protocol and compute scope |

```bash
python3 -B scripts/validate_fast.py
python3 -B -S scripts/validate_fast.py
/content/icgs-check-env/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

The package must be installed; tests no longer inject repository roots into sys.path.
L0 uses only stdlib. Full L1 uses the validated environment; missing modules are not
substituted with fake imports. Core fixture tests inject concrete small collaborators
only at public seams, not to certify missing full-model behavior.

Unittest zero exit with skipped tests is not a full PASS. Report selected/executed/
skipped counts. C1–C5 are mandatory for this migration, not optional because a host
initially lacks dependencies. Opt-in decorators serve cheap routine runs; the
acceptance job must run actual checkpoint tests and reference comparison.

## Test owners

| File | Protected behavior |
| --- | --- |
| test_harness.py | Syntax/link/boundary CLI positive/negative fixtures |
| test_architecture.py | Direct/indirect core dependency direction, package initializers |
| test_src_package.py | Actual canonical source exists, no old ip runtime imports/tree |
| test_config.py / test_v5_config.py | Baseline values, frozen config, profiles, unknown IDs/types/paths/horizons |
| test_geometry.py / test_cpu_smoke.py | Pose/rotation/SVD and normalization |
| test_composition.py | Encoder/graph/stage replacement and real PyG forward/backward |
| test_policy.py | Public inference, owned immutable context, branch feature isolation, cache/reset |
| test_candidates.py | K/B/P provenance and independent root-anchored prefixes |
| test_inference_contract.py | Safe NPZ and exception-safe Python/NumPy/CPU/CUDA RNG scope |
| test_data.py | Native PyG serialization/augmentation and actual Open3D filtering when installed |
| test_evaluation.py | Original metric/cadence/step exception behavior via fake environment, real action conversion |
| test_checkpoints.py | Generated alias/compiled/shape/missing diagnostics and artifact immutability |
| test_loading.py | Resolved config survives load→context→prediction |
| test_training.py | Actual Lightning wrapper alias registration (no training job) |
| test_cli.py | Installed help outside checkout, rejected published-runtime overrides |
| test_differential.py | Verified original source methods, not two facades of new implementation |
| test_model_integration.py | Real strict published native inference, explicitly enabled |
| regression/published_fidelity.py | Actual external vv19 vs native outputs, repeatability-derived tolerance |

Tests remain flat for reliable unittest discovery; regression/fixtures are explicit
standalone scripts, not silently undiscovered unittest suites. L0 counts its selected
19 tests and negative fixtures; full discovery counts are in acceptance evidence.

## Published integration commands

After setup in [README](../README.md), with trusted assets:

```bash
ICGS_RUN_PUBLISHED=1 ICGS_CHECKPOINT=/content/icgs-artifacts/model.pt ICGS_FIXTURE=/content/icgs-evidence/input.npz /content/icgs-check-env/bin/python -B -m unittest discover -s tests -p test_model_integration.py -v
```

Reference comparison uses the exact command in
[published validation evidence](../docs/experiments/vv19-validation/README.md).
It runs the original binary in an external reference process/environment, not
inside native runtime. Normal native installed inference never adds reference to
PYTHONPATH or imports instant_policy/ip.

Original source differential:
```bash
IP_LEGACY_SOURCE_ROOT=/content/source-reference /content/icgs-check-env/bin/python -B -m unittest discover -s tests -p test_differential.py -v
```

Source hashes are checked before selected original definitions execute. These
source comparisons do not replace target fidelity; checkpoint/normalization/
preprocessing settings can differ from the earlier source fork.

## Guarantees and limitations

L0 is syntax/local static analysis, not arbitrary dynamic dependency proof. Markdown
checking covers simple inline/reference links, ATX headings and HTML src/href,
not external availability or a full renderer. Learned algorithms/models do not
depend on harness files. No CI provider, hook or branch protection was added.

Published fixture uses nondegenerate synthetic geometry and two demonstrations
with nontrivial transforms; it proves execution/fidelity, not task success.
Full scene/FPS/DDIM runs are real on Colab T4. CUDA/ABI pins matter; no mass dependency
upgrade or encoder-math workaround is part of setup.

RLBench launches remain explicit and normal-path cleanup preserves original
semantics; retries may be unbounded. Do not run unknown files through the legacy
retry loader as a quick smoke test. Train/large preprocessing/benchmark/robot work
requires separate scope; helper tests never authorize it.

## Reporting

PASS: named check executed and met assertions.
FAIL: check ran and failed, including installed-but-broken imports.
SKIPPED: selected check unavailable with explicit prerequisite reason.
NOT RUN: not selected/attempted.

Always give command, cwd, interpreter/versions, counts, reason and remaining claim
limits. Preserve failed intermediate evidence and explain the correction. No
skipped required gate becomes PASS. C1 load metadata, C2 strict load, C3 inference,
C4 fidelity and C5 independent installed execution are distinct claims.
