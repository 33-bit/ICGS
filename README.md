# ICGS

Unified research runtime in `src/icgs`. Instant Policy is an internal native
policy/components baseline; method-level world models, learned evaluators and
planners remain [designed/deferred](docs/components/v5-foundations.md).

Start with [AGENTS](AGENTS.md), the [documentation map](docs/README.md),
[architecture](docs/ARCHITECTURE.md), and [checkpoint evidence](docs/experiments/vv19-validation/README.md).

## Validated inference environment

Linux x86_64, Python3.10, NVIDIA CUDA11.8 wheels. These commands match the isolated
Colab environment used for acceptance; no full Conda export or host Python3.14 is
assumed compatible.

```bash
uv venv --python 3.10 /content/icgs-check-env
uv pip install --python /content/icgs-check-env/bin/python -r requirements-inference-cu118.txt
uv pip install --python /content/icgs-check-env/bin/python --no-deps .
mkdir -p /content/icgs-artifacts
/content/icgs-check-env/bin/python -m gdown --no-cookies https://drive.google.com/uc?id=1TM_zU1pVOqPuWZL3E9knNp4w-p7EBwwt -O /content/icgs-artifacts/model.pt
/content/icgs-check-env/bin/python tests/fixtures/generate_smoke.py --output /content/icgs-evidence/input.npz
cd /tmp
/content/icgs-check-env/bin/icgs infer --checkpoint /content/icgs-artifacts/model.pt --input /content/icgs-evidence/input.npz --output /content/icgs-evidence/actions.npz --device cuda --seed 17
```

Run installation commands from the checkout, then the last inference command works
from any directory. For an existing wheel, install its absolute path instead of dot.
torch-cluster/scatter/pyg-lib must match torch/CUDA ABI; they are explicitly pinned
in the requirements file, not silently replaced with Python mocks.

The published checkpoint is 471777552 bytes, SHA256
`119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5`.
Hash identifies downloaded bytes, not an author-issued checksum. Encoder weights
are included; no config.pkl or scene_encoder.pt download is required for this target.
Do not commit weights. Local cached artifact is at
`artifacts/checkpoints/vv19/model.pt`.

## Commands

```bash
icgs --help
icgs infer --help
icgs train --help
icgs evaluate --help
icgs prepare-data --help
```

Help does not load models or start environments. Train/evaluate/prepare-data require
explicit config/data/artifact arguments. See [CLI/config contracts](docs/components/cli-and-data.md).
No training, full benchmark or physical robot execution is automatic.

## Cheap validation

```bash
python3 -B scripts/validate_fast.py
/content/icgs-check-env/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

L0 includes canonical src/package/import-boundary checks and negative fixtures.
Read counts and skips; CPU/test-double proof is not published-checkpoint fidelity.
Actual C1–C5 acceptance uses real weights and the external vv19 oracle, with logs.

## Research provenance

Native numerical source originated from Instant Policy, “In-Context Imitation
Learning via Graph Diffusion” (Vosylius and Johns, ICLR2025).
[Upstream](https://github.com/vv19/instant_policy),
[project](https://www.robot-learning.uk/instant-policy).
Historical source identity and target-specific differences are recorded in
[baseline documentation](docs/baselines/instant_policy.md). Reference distribution
notices are in [third-party notices](docs/third-party-notices.md).
No full benchmark success or complete v5 method is claimed.
