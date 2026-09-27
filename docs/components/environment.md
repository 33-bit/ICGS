# Environment setup

ICGS uses `pyproject.toml` and `uv.lock` as the single dependency source. A new
machine needs Python 3.10, 3.11 or 3.12 and `uv`; the setup command creates or
reuses `.venv`, installs the selected locked profile and installs ICGS editable.

## Profiles

| Profile | Use | Host requirements |
| --- | --- | --- |
| `cpu` | Core tests and CPU inference | Python 3.10–3.12 |
| `cuda118` | Published CUDA inference | Linux x86_64, NVIDIA driver/runtime compatible with CUDA 11.8 |
| `generation` | CPU rendered RLBench collection | Linux, apt privileges, Xvfb/Mesa; simulator provisioning is opt-in |

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

### Rebuild a generation host from a clone

The supported generation target is a compatible Linux host with `apt` access,
Xvfb/Mesa, enough disk for the simulator and staging, and network access for the
explicit setup downloads. It is not a claim that RLBench generation runs on any
operating system. Install `git`, `curl`, `tar`, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) first. A host
without a supported Python can use `uv python install 3.10`; confirm that
`uv python find 3.10` resolves an interpreter. Do not use a system Python 3.13+
as the generation environment. In a non-login shell, ensure the installed
`uv` directory is on `PATH` (often `$HOME/.local/bin` for a standalone
installation) before running the commands below.

From a fresh clone, keep rebuildable assets under the clone so that no separate
`icgs-vps-env` directory is required. `outputs/` and `.venv/` are Git-ignored:

```bash
git clone https://github.com/33-bit/ICGS.git
cd ICGS
git rev-parse HEAD  # record this exact code revision with the run receipts
uv python install 3.10
mkdir -p outputs
cp src/icgs/configuration/profiles/generation_runtime.json outputs/generation-runtime.json
```

Edit `outputs/generation-runtime.json` before provisioning. Set
`machine.repo_root` to the absolute clone path, `machine.python_executable` to
`<clone>/.venv/bin/python`, `machine.simulator_root` to
`<clone>/outputs/CoppeliaSim`, and `machine.rlbench_root` to
`<clone>/outputs/RLBench`. Set `run.run_root` to a fresh absolute path under
`<clone>/outputs/runs/`, and give `run.run_id` a distinct value. The copied file
is a **validation example**, not a production launch profile: leave publication
disabled during setup. Do not put an HF token or its path in this runtime JSON.

Provision the locked generation profile and pinned simulator sources together:

```bash
"$(uv python find 3.10)" -B scripts/setup_environment.py \
  --profile generation --python-version 3.10 \
  --venv-root "$PWD/.venv" \
  --provision-simulator --build-generation-tasks \
  --runtime-config "$PWD/outputs/generation-runtime.json" \
  --receipt "$PWD/outputs/setup_receipt.json"
```

This opt-in command runs `uv sync --locked`, host package installation, the
checksum-checked CoppeliaSim download/extraction, and pinned PyRep/RLBench
checkout/installation. The explicit `--build-generation-tasks` flag then launches
a bounded headless simulator build of all 36 generated task modules and scene
assets; provisioning the upstream RLBench checkout alone does not supply them.
It does not collect episodes. On a rerun, keep `--provision-simulator` and the same
runtime config: `uv sync` can remove unmanaged simulator packages, so the
provisioner reapplies them. Without `--build-generation-tasks`, setup does not
start a simulator. Neither mode starts workers, a coordinator, or collection.

Verify the installed profile without launching an episode. Supply a
coordinator-only HF credential path outside Git if publication will be used;
the command checks its permissions but never prints its value:

```bash
export ICGS_HF_TOKEN_PATH=/secure/credentials/hf-token
./.venv/bin/python -B scripts/verify_environment.py --profile generation \
  --repo-root "$PWD" --python "$PWD/.venv/bin/python" \
  --simulator-root "$PWD/outputs/CoppeliaSim" \
  --rlbench-root "$PWD/outputs/RLBench" \
  --credential-path "$ICGS_HF_TOKEN_PATH" \
  --receipt "$PWD/outputs/setup_receipt.json" --json
LD_LIBRARY_PATH="$PWD/outputs/CoppeliaSim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH=src ./.venv/bin/python -B -m pytest -q -rs \
  tests/test_generation*.py tests/test_capacity_probe.py
./.venv/bin/python -B scripts/validate_fast.py
```

The verifier checks that every generated task module matches the compiler and
has a nonempty scene asset. Its `PASS` proves environment imports, paths and
task availability, not a live episode or full-generation readiness. A fresh
Python 3.10 environment and all-task build were exercised on `vps-a` under the
clone in September 2026. Continue with the
[data-generation launch gate](generation.md#full-generation-readiness), not
the validation example's launch flags.

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
PyRep and credentials. A missing optional CUDA or generation component is reported
explicitly. It does not publish to Hugging Face or run a simulator.

If `uv sync --locked` reports a stale lock file, run `uv lock` from the repository
root and commit the resulting lock update with the dependency change. CPU and
CUDA profiles are mutually exclusive because their PyTorch indexes differ. If
PyG reports an ABI mismatch, verify that the host selected the matching profile
and Python version before reinstalling the environment.

No setup path depends on Conda, `environment.yml`, `ip_env` or a `/content`
directory. All roots are configurable absolute paths in the runtime profile.
