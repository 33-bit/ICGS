# Central ICGS configuration implementation

Status: completed — configuration infrastructure, not missing model consumers.
Category C shared configuration migration.
Owner: repository owner / implementation agent. Approved: 2026-09-09.
Goal: one packaged JSON default source for added ICGS, with immutable typed
validation, explicit override files and unchanged native IP behavior.

## Scope and decisions

Primary source: `src/icgs/configuration/profiles/icgs_primary.json`.
Keep MethodConfig constructor/from_dict/from_file/to_dict and existing effective
values/shape locks. Extend missing model/training/data/search/evaluation settings.
Do not unlock architecture, migrate native IP, implement missing consumers or run
training/collection/simulator jobs. Unknown physical setup remains null and gated.
Derived dimensions are computed, not independent tuneable values. Existing width/
head/H fields remain compatible aliases with their prior cross-section checks.

## Work and acceptance

- [x] Add failing configuration tests: old values, missing/new source behavior,
  strict types/errors, overlays, frozen nested values and resolved identity.
- [x] Implement typed resource-loaded defaults and package data; preserve reference
  identity rules and distinguish complete config hash from reference-policy hash.
- [x] Inventory proposal settings in JSON, add explicit override example, document
  key/unit/consumer/restriction mapping and revise P00–P13 consumption instructions.
- [x] Run focused tests, native-config regressions, L0 and wheel loading outside
  checkout. Record counts, environment, failures, limitations and review fixes.

## User work and recovery

Initially dirty P01 plan plus four untracked timed-execution/runtime-test files
belong to the user's ongoing work. Do not touch their runtime/test content. Add
only a scoped configuration-consumption note to P01, preserving all its evidence.
No commit/push/reset/checkout. If validation fails, retain failures and fix scoped
configuration; do not modify baseline or silently relax architecture checks.

## Evidence

Date: 2026-09-09. CWD: `/Users/33bit/AI/Research/VLA/ICGS` unless stated.
Existing editable installation: `.venv/bin/python`, Python3.11.15, Darwin arm64,
NumPy1.26.4, torch2.2.0. L0 uses host Python3.14.4 (stdlib only). No dependency
download/upgrade or external checkpoint/simulator workload occurred.

| Command / check | Result | Counts / limits |
| --- | --- | --- |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_method*.py'` | PASS |39 executed:24 new config +15 existing method contracts;0 skips |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_config.py'` | PASS |5/5;0 skips; native defaults/roundtrip |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_v5_config.py'` | PASS |4/4;0 skips; native file/profile behavior |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_architecture.py'` | PASS |3/3;0 skips |
| `.venv/bin/python -B -m unittest discover -s tests -p 'test_src_package.py'` | PASS |2/2;0 skips |
| `python3 -B scripts/validate_fast.py` | PASS |syntax/links/boundaries and19/19 self-tests |
| `python3 -B -S scripts/validate_fast.py` | PASS |same19/19 self-tests under isolated stdlib |
| Installed-wheel config tests, command below | PASS |24/24;0 skips; Python3.11.15, no source-path injection |
| `git diff --check` | PASS |no whitespace errors |
| Parameter inventory audit | PASS |268 JSON config leaves documented, including units/consumer/restriction mapping |
| Native source/profile and archive diff checks | PASS |native config/defaults/loader and reference fingerprint code unchanged; proposal SHA256 preserved |
| User's timed-execution runtime checksums | PASS |three runtime files byte-identical; P01 progress/evidence preserved with additive note only |
| User's untracked timed-execution test checksum | CHANGED EXTERNALLY |initial `6d4e8c69...`, final `cd5b1ce0...`; no task tool/agent edited this file, concurrent change retained |
| Full neural L1, C1–C5, L3/L4, training/collection | NOT RUN |configuration-only scope; no new model/physics/fidelity claim |

### Red/green and review

New tests first exposed missing identity/readiness APIs, duplicate-key acceptance,
numeric type validation and resource-source behavior. Later RED tests caught invalid
numeric bounds, relative resolved-path identity drift and student warmup exceeding
training length. Each was corrected before the final39-test GREEN run. The old
effective-default regression compares every prior field with a literal baseline.

Independent review identified resolved-path identity drift, missing reactive warmup
validation and three incorrect documentation units. All were fixed; scoped rereview
reported no remaining blockers. Rereviewer used host source-root diagnostics only;
the supported editable/wheel results above are the installed verification evidence.

Resolution: `resolved_config()` requires absolute paths or null and exports every
field; signed files require complete absolute metadata and verify identity before
intentional overrides. Ordinary experiment files may still use file-relative paths.
Absolute path spelling is preserved rather than silently changing recorded identity.
Shared optimizer warmup checks also cover the reactive student's maximum updates.
Full config identity never substitutes for frozen reference identity.

### Standalone package verification

Temporary artifacts live under `/tmp/icgs-config-wheel.BhTbvP`. The normal
`python -m build --wheel --no-isolation` attempt failed because the local `build`
module has no executable `__main__`; no dependency installation was attempted to
conceal that failure. The installed setuptools84 backend built the wheel directly:

```sh
.venv/bin/python -c 'from setuptools.build_meta import build_wheel; print(build_wheel("/tmp/icgs-config-wheel.BhTbvP"))'
.venv/bin/python -m venv --without-pip /tmp/icgs-config-wheel.BhTbvP/env
uv pip install --python /tmp/icgs-config-wheel.BhTbvP/env/bin/python --no-deps --no-index /tmp/icgs-config-wheel.BhTbvP/icgs-0.2.0-py3-none-any.whl
```

The final rebuilt wheel was installed with the same command plus `--reinstall`.
From `/tmp/icgs-config-wheel.BhTbvP`:

```sh
/tmp/icgs-config-wheel.BhTbvP/env/bin/python -I -B -m unittest discover -s /Users/33bit/AI/Research/VLA/ICGS/tests -p 'test_method_config.py' -v
```

Only test source is supplied from the repository; imports resolve to the new
environment's site-packages. Independent checks confirmed module origin, packaged
JSON presence, old geometry/optimizer defaults, derived terminal-input dimension,
per-demo event capacity and JSON identity serialization. Build outputs are local
ignored artifacts, not a published release.

### Remaining limitations

The inventory includes planned consumers; already implemented physical constructors
are not automatically rewired by loading this metadata. Per-plan addenda require
passing typed sections and testing supported nondefault propagation before claiming
consumer support. Architecture/preprocessing locks remain enforced. Null physical
setup/seed/resource fields block operation readiness but no metadata check proves
artifact trust, file existence, simulator calibration or authorization.

No model training/collection/runtime fidelity claim follows from config tests.
No commit, staging, branch change, push, reset or worktree operation was performed.
The unrelated test-file checksum change was observed in the final audit; it was
not overwritten or used to claim extra configuration acceptance.
