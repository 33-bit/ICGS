# Canonical Generation Design

## Goal

Make one semantic, resumable data-generation system the only supported
generation path, while preserving serialized artifact compatibility and
historical provenance.

## Scope

In scope are the primary collection package, Colab generation entry points,
task/materialization execution helpers, generation tests, generation artifacts,
and operational documentation. Out of scope are the Instant Policy model
baseline, published-checkpoint inference, model architecture, and already
published data deletion.

## Architecture

`icgs.data.collection.generation` owns protocol, planning, task compilation,
execution materialization, queueing, validation and publication. `scripts/` holds
only thin semantic operational entry points. No primary worker may import a
legacy collector or a script whose name encodes an obsolete generation version.

The data payload retains its current schema value for compatibility. Source
module names and Python symbols lose numeric generation suffixes; documentation
uses “primary generation” as the operational name.

## Cleanup policy

Delete generated one-off scripts and receipts after confirming they are not
tracked runtime owners. Delete tracked v1/v2/g2 generation branches only after
their tests and live call sites have migrated. Keep historical baseline and ADR
documents, but mark them as provenance. Do not delete checkpoints, manifests,
episodes, or large research artifacts.

## Validation

- static reference scan finds no imports or operational docs using legacy
  generation module names;
- `python3 -B scripts/validate_fast.py` passes;
- focused primary generation, queue, validation, publication and worker tests
  pass;
- full simulator/data generation is not run as incidental validation.
