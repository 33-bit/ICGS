# ICGS

Unified research runtime in `src/icgs`. Instant Policy is an internal native
policy/components baseline; method-level world models, learned evaluators and
planners remain [designed/deferred](docs/components/v5-foundations.md).

Start with [AGENTS](AGENTS.md), the [documentation map](docs/README.md),
[architecture](docs/ARCHITECTURE.md), and [checkpoint evidence](docs/experiments/vv19-validation/README.md).

## Portable environments

The canonical installer is `uv` with the locked project metadata. It works from a
fresh clone without Conda or a fixed filesystem prefix. Python 3.10, 3.11 and 3.12
are supported.

```bash
python3 scripts/setup_environment.py --profile cpu
python3 scripts/verify_environment.py --profile cpu --json
```

Use `--profile cuda118` on a Linux host with the NVIDIA CUDA 11.8 runtime. Use
`--profile generation` for the pinned Xvfb/Mesa/RLBench/CoppeliaSim setup, then add
`--provision-simulator --runtime-config <absolute-runtime-config>` when simulator
assets are explicitly wanted. Setup is idempotent and writes a redacted receipt to
`.icgs/setup_receipt.json`.

The historical published inference stack is still available in
`requirements-inference-cu118.txt`; the exact C1–C5 command and artifact checksums
remain documented in [published validation evidence](docs/experiments/vv19-validation/README.md).
The PyTorch/PyG ABI must match the selected profile; setup never substitutes fake
modules.

Credentials stay outside Git. Set a coordinator-only token path or an environment
variable when a command explicitly needs publication. Setup and worker processes
remove credential variables from child environments, and receipts contain only
redacted command metadata.

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
python3 -B -m pytest -q tests/test_environment_setup.py tests/test_generation_environment.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
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
