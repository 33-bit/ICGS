# Data generation

ICGS has one supported data-generation system. Runtime owners live under
`src/icgs/data/collection/generation`; operational entry points use the
`scripts/generation_*` prefix.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `protocol.py` | Frozen dataset, schema, controller, quota and split identities |
| `steps.py`, `scenes.py`, `compiler.py` | Program semantics, scene layouts and executable task compilation |
| `batch.py`, `attempt_prep.py`, `quota.py`, `diversity.py` | Attempts, perturbations, seeds, uniqueness and stop conditions |
| `episode_record.py`, `rlbench_attempt.py`, `task_labels.py` | Measured episode/attempt materialization and labels |
| `distributed_contracts.py`, `distributed_queue.py` | Immutable jobs/results and atomic filesystem queue |
| `distributed_planner.py`, `distributed_validation.py` | Central quota authority and closed-result validation |
| `distributed_publication.py` | Conflict-safe Hugging Face commits and resumable receipts |
| `generation_worker.py` | Credential-free persistent worker |
| `generation_coordinator.py` | Single-owner ingestion, planning and publication control plane |
| `generation_watchdog.py` | Bounded replacement of missing worker/control processes |
| `generation_launch.py` | Preflight and launch of the configured distributed run |

`artifacts/composition/approved_composition_manifest.json` is the only approved
program manifest. `src/icgs/configuration/profiles/generation.json` owns
generation-specific method settings. There is no legacy protocol selection or
fallback collector.

## Record classes

`success` and `valid_failure` are valid episodes. Simulator crashes and invalid
observations are closed attempt records and must never be mislabeled as valid
failures. A valid episode has `T` actions, `T+1` causal online observations and
`T` durations. Every closed result carries an immutable file inventory and an
artifact manifest; publication starts only after validation.

Serialized records retain their existing `schema_version` and protocol identity
strings. Those values identify stored artifacts and are not source-module names.
Changing them requires a separate data migration.

## Distributed lifecycle

```text
planner → pending → claimed → ready → ingested → published
             worker ────────┘       coordinator ─────────┘
```

Workers have no Hugging Face token. The coordinator alone validates results,
updates quota state and publishes. Temporary directories containing `.partial-`
are not queue entries. Remote conflicts fail closed; matching immutable hashes
may be reconciled during an explicit resume.

## Operational rules

- Use a new disjoint run ID after a dead Colab assignment.
- Resume from the remote manifest, never from remembered chat state.
- Preserve valid failures and closed infrastructure attempts.
- Do not count skipped validation as pass.
- Do not launch generation as incidental validation.
- Publication rate limits and transient upload timeouts defer work; they do not
  authorize dropping queue entries.

Historical generation audits under `docs/audits/` record what happened during
earlier launches. They are evidence, not current commands.

## Validation

Cheap validation:

```bash
python3 -B scripts/validate_fast.py
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation*.py
```

Simulator execution, full collection and large Hugging Face publication require
an explicitly provisioned environment and separate authorization.
