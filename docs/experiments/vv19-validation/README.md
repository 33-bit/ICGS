# Published checkpoint acceptance evidence

Validation date: 2026-09-06. Target source vv19/instant_policy revision
72281736672a555e9c9834eb15c225e2f1baaf93; downloaded model SHA256
119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5,
471777552 bytes. No retraining or alternate checkpoint.

The external reference binary strictly loaded all 674 state entries. Native ICGS
strictly maps those entries to 337 model-owned tensors, with equal alias mappings
reported and no missing/unexpected/conflicting/shape-mismatched tensors. Encoder
weights are included; packaged hash-bound configuration comes from exposed reference
config/normalizer/scheduler, not guessed tensor shapes. See
[manifest](../../baselines/vv19-artifact.json) and packaged profile.

## Evidence boundaries

- C1: artifact/hash/reference configuration and auxiliary-dependency audit PASS.
- C2: actual strict target load into native src/icgs PASS.
- C3: real native GPU forward/sampling on two-demo geometric fixture, finite/SE3/grip
  outputs, cold/warm/reset and no gradients PASS.
- C4: matched reference/native input, checkpoint/settings, seeds17/29/41 PASS.
  Tolerance is calculated from reference repeatability before native comparison;
  exact report in [final-gates.log](final-gates.log).
- C5: installed wheel executed from /tmp with no old ip/instant_policy imports,
  reference checkout absent from import path, no unused training/environment/model
  components loaded: [standalone.log](standalone.log) PASS.

These claims concern the tested synthetic fixture/settings only. No RLBench task
success, broad benchmark, optimizer resume, whole-object old pickle or complete v5
method is claimed. Initial native mismatch was 0.01696–0.04430 translation and
different grips; explicit published live voxel/RNG-order profile fixed it. No
tolerance was widened to fit that mismatch.

Final fresh-session acceptance: native matrix error up to2.2352e-7, translation
up to6.7056e-8, rotation up to2.6573e-7 radians, grips exactly equal. Full suite:
88 selected, 87 PASS, 1 SKIPPED (RLBench task-class import); zero failures/errors.
L0: 19 PASS. [acceptance.json](acceptance.json) and
[all-commands.log](all-commands.log) contain actual final command output.
Validated wheel SHA256: 7509bcdb66f7da26b3a7bfebf414b410d7e3b9a25b6b34d0dca0e8abf3625d49.

## Fixture and reproduction

[tests/fixtures/published_smoke.npz](../../../tests/fixtures/published_smoke.npz)
is geometric synthetic data, generated with NumPy seed20260906, three nondegenerate
cloud clusters, two ten-waypoint demos, and non-identity live/demo rotations and
translations. SHA256 75206e40b2475693cdfccaa2e9f6a4b2ae8b54c33917496dc3403e6b9c448d66.
Generator: [generate_smoke.py](../../../tests/fixtures/generate_smoke.py).
Frame/schema: [CLI/data contract](../../components/cli-and-data.md).

Validated stack: Colab T4, CPython3.10.21, torch2.2.0+cu118, PyG2.5.0,
torch-cluster1.6.3+pt22cu118, torch-scatter2.1.2+pt22cu118, pyg-lib0.4.0+pt22cu118,
NumPy1.26.4, SciPy1.14.1, Diffusers0.31.0, Open3D0.18.0, Lightning2.4.0.
[requirements-inference-cu118.txt](../../../requirements-inference-cu118.txt)
records tested direct dependencies/ABI-specific wheels. Full installed versions
are captured in validation output when available. Use the README setup, not host
Python3.14 or a CPU/emulated binary as a claimed GPU reference.

Reference is external only. With reference checkout /content/vv19-reference and
installed ICGS wheel, the actual comparison entry is:

```bash
PYTHONPATH=/content/vv19-reference /content/icgs-check-env/bin/python /content/icgs-validation-checkout/tests/regression/published_fidelity.py --checkpoint /content/icgs-artifacts/model.pt --fixture /content/icgs-evidence/input.npz --output-directory /content/icgs-evidence
```

The normal native command does not set PYTHONPATH or require that reference:

```bash
cd /tmp
/content/icgs-check-env/bin/icgs infer --checkpoint /content/icgs-artifacts/model.pt --input /content/icgs-evidence/input.npz --output /content/icgs-evidence/actions.npz --seed 17
```

## Operational notes

The first local x86 Docker attempt failed on Open3D SIGILL under ARM emulation;
container was stopped. Native macOS had an OpenMP failure in Open3D; no model code
was changed to mask it. Explicit user authorization then allowed Colab inference.
First Colab install selected torch cu121 against cu118 extensions; pinning the
official cu118 wheel corrected the ABI mismatch. A later idle Colab session lost
its proxy; recorded execution evidence was recovered from CLI history, and a fresh
named T4 session is used for final rerun/download and cleanup.

[unit-tests.log](unit-tests.log) records focused actual-stack tests. Simulator task
resolution may be SKIPPED because RLBench is not installed; it is not C1–C5 proof.
Runtime schema/root/RNG/context issues found in independent review were fixed with
regressions before final wheel verification.

Colab file-download API later failed even though execution had completed. Final logs
and JSON were recovered verbatim from the CLI execution-history export, SHA256
d5dff2f9c57de5660055fe2e29319e99ce082108ee20eaca55f8fd673256fa99.
Native output NPZ remained remote and was not recovered; the checked-in fixture and
exact commands reproduce it. Both task GPU assignments were explicitly released;
colab sessions returned no active sessions. No background compute remains.
