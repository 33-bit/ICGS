# Decision: local-first observability at outer runtime boundaries

Date: 2026-09-09. Status: accepted by the approved observability design and implementation request.
Decision owner and approval evidence: repository owner approval recorded in
[the active observability plan](../plans/active/icgs-observability.md).

## Context

ICGS had native metric output, optional Lightning/W&B logging and CLI status
messages, but no correlated local evidence store. A diagnostic facility must
help reconstruct failures without changing native action, graph, preprocessing,
checkpoint, reference or RNG behavior. It must also remain usable in an installed
wheel without importing models, simulators or remote services.

## Decision

Use a small stdlib recorder under `src/icgs/observability/` with immutable typed
configuration in the packaged primary JSON. The recorder owns contextvars,
correlated event/span records, bounded per-process JSONL queues, atomic startup
and summary manifests, explicit NumPy-only capture, and stdlib-only offline
inspection. Library capabilities are no-op by default; CLI and other outer
boundaries start a run explicitly and attach only owned handlers to the `icgs`
logger namespace.

Recorders may be passed into candidate, execution, physical-rollout and training
seams. Models, geometry, state, checkpoints and graphs do not own logger objects
or file/network dependencies. W&B is an optional lazy outer adapter with a
separate bounded queue; local records remain authoritative, online mode requires
an explicit configuration choice, and no automatic login or upload is performed.
Capture accepts only explicitly supplied copied CPU NumPy numeric arrays and
rejects unsafe or over-quota inputs. Missing/dropped/partial evidence remains
incomplete and is never converted into a successful research gate.

## Alternatives considered

- A global logger or import-time handler would leak state across runs and could
  alter library behavior; it is rejected.
- A generic telemetry/plugin framework would expand scope and create speculative
  producers; it is rejected.
- W&B-first or network-only logging would make local evidence unavailable and
  would add unauthorized transport to validation; it is rejected.
- Capturing arbitrary tensors, pickles or simulator snapshots through the common
  recorder would risk device transfers and unsafe replay; it is rejected.

## Consequences

Every explicit run gets an exclusive directory with startup/resolved identity,
per-process event/metric streams and a final completeness summary. Queue,
chunk, metric and capture quotas bound overhead and preserve control progress at
the cost of possible explicitly reported evidence loss. Offline inspection can
report unfinished spans, drops and missing summaries without executing captures.
Task plans P02/P05/P06/P08/P09/P10/P12 retain ownership of future producers; this
implementation emits no phantom search/router/evaluator records.

## Compatibility implications

The native `ExperimentConfig`, published profile, checkpoint aliases, policy data
contract, action/graph math, evaluation semantics and reference fingerprint are
unchanged. The full MethodConfig fingerprint includes observability settings, but
the explicit reference projection excludes them. No training, download, network,
simulator or robot workload is required to construct or inspect a run. C1–C5
remain separate mandatory acceptance gates for affected native paths.
