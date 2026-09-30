# Environment setup

ICGS uses `pyproject.toml` and `uv.lock` as the single dependency source. A new
machine needs Python 3.10, 3.11 or 3.12 and `uv`; the setup command creates or
reuses `.venv`, installs the selected locked profile and installs ICGS editable.

## Profiles

| Profile | Use | Host requirements |
| --- | --- | --- |
| `cpu` | Core tests and CPU inference | Python 3.10–3.12 |
| `cuda118` | Published CUDA inference | Linux x86_64, NVIDIA driver/runtime compatible with CUDA 11.8 |

From a clean clone:

```bash
python3 scripts/setup_environment.py --profile cpu
python3 scripts/verify_environment.py --profile cpu
```

For a CUDA host:

```bash
python3 scripts/setup_environment.py --profile cuda118
python3 scripts/verify_environment.py --profile cuda118 --json
```

The RoboHiMan simulator uses its own isolated venv; see
[RoboHiMan backbone](robohiman.md#simulator-environment-isolated-from-the-icgs-venv).

## Credentials

HF and W&B credentials stay outside Git. A coordinator may receive an explicit
credential path such as `/secure/credentials/hf-token`; workers must not receive
that path or its value. The setup receipt at `.icgs/setup_receipt.json` records the
profile, platform and redacted command plan only. The verifier checks only whether
supplied credential paths exist and have restrictive permissions; it never opens
or prints their contents.

## Verification and troubleshooting

The verifier emits one JSON object with `PASS`, `FAIL`, `SKIPPED` or `NOT_RUN` for
Python, ICGS imports, core dependencies, PyG ABI, renderer, simulator, RLBench/
PyRep and credentials. A missing optional CUDA component is reported
explicitly. It does not publish to Hugging Face or run a simulator.

If `uv sync --locked` reports a stale lock file, run `uv lock` from the repository
root and commit the resulting lock update with the dependency change. CPU and
CUDA profiles are mutually exclusive because their PyTorch indexes differ. If
PyG reports an ABI mismatch, verify that the host selected the matching profile
and Python version before reinstalling the environment.

No setup path depends on Conda, `environment.yml`, `ip_env` or a `/content`
directory. All roots are configurable absolute paths in the runtime profile.
