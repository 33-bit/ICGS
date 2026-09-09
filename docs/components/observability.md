# Local logging, correlated tracing and bounded failure capture

Status: approved design direction; Tasks 1–5 runtime implementation present and
Task 6 acceptance/documentation in progress. C1–C5 remain required and are not
claimed by local observability tests.
Owner approval: 2026-09-09, local logging with optional W&B, execution tracing,
and bounded diagnostic capture. This document does not claim the system exists.
Current inspection revision: `main` at `cacef87` with an uncommitted working tree.
See the [implementation plan](../plans/active/icgs-observability.md).

## Current state and goal

Native Lightning training logs Train_Loss and validation grip/translation/rotation
losses, with optional WandbLogger. CLI inference prints JSON and saves metadata;
evaluation uses progress/final success output. The native training runner documents
an undefined logger when record=True and use_wandb=False. The implemented outer
recorder now supplies a local structured event/metric sink, correlated span store,
bounded NumPy capture and offline inspection; future method components still do
not emit events until their own runtime plans are implemented.

Goal: follow a failure from input through context/physical state, proposal,
prediction, evaluation, search and execution; distinguish missing evidence from
negative results. Keep human logs, research metrics, and detailed diagnostic data
separate. Local records remain authoritative even when W&B is enabled.

## Ownership and compatibility

Create `src/icgs/observability/` for context, recording, safe serialization,
inspection and optional backend adapters. Outer CLI/execution/training boundaries
own recorders. Models and geometry do not import logging, W&B, Lightning or file IO.
Algorithms may receive an optional small recorder capability/callback; do not add
mutable logger objects to PhysicalState, TaskState, checkpoints or model graphs.
Existing collaborators can be wrapped in traced callables at composition time.

No import-time handler installation, files, threads, network or global logger
changes. Explicit run initialization attaches only owned handlers to the `icgs`
logger namespace. Reinitialization removes/closes only this run's handlers.
Console logs go to stderr; preserve inference JSON and evaluation results on stdout.
No checkpoint/data/action/preprocessing/reference-policy semantic change.

The recorder is optional for library calls (no-op by default). CLI commands create
a local run unless observability.mode is off. New configuration stays in the
packaged primary JSON and immutable typed configuration. Native ExperimentConfig
and strict published-profile comparison remain unchanged: use a separate logging
config adapter, never inject logging sections into a legacy model configuration.

Implemented owners are `src/icgs/observability/`, `src/icgs/cli/logs.py`, the
outer native CLI wrappers, candidate/evaluation seams, the physical rollout and
timed execution wrappers, and the local training adapter. Future P02/P05/P06/P08/P09/P10/P12 producers
remain planned unless a current boundary explicitly emits the limited records
listed below; no search ranking, task-router, simulator snapshot or model-forward
producer is fabricated.

Fix the native record-without-W&B logger initialization in a named, tested change
while integrating the native training adapter. Preserve metrics, cadence,
optimizer, model initialization and checkpoint behavior. Do not repair unrelated
native runner defects under this work.

## Run files and identity

An explicit run creates `outputs/runs/<UTC-start>-<uuid4>/` by default. Resolve the
output root once; create exclusively and never overwrite another run. UUID uses
OS randomness, never Python/NumPy/torch policy RNG. Each resumed invocation creates
a new attempt directory with parent_run_id/resumed_from metadata; no append to a
different process's old stream. No simulator snapshot restoration is implied.

```text
run.json                 immutable startup identity and declared capabilities
resolved_config.json     full sanitized/resolved config and its identity
events/p<PID>-r<RANK>-000001.jsonl
metrics/p<PID>-r<RANK>-000001.jsonl
captures/<capture_id>/manifest.json + bounded numeric .npz
summary.json             final status, stream/capture totals and completeness
```

run.json records schema_version, run_id, parent_run_id, command name, code revision
and dirty flag when provided/discoverable, Python/package/platform/device metadata,
config SHA256, native/reference/evaluator/encoder/dataset IDs and actual seeds as
available. Mark unavailable provenance null with a reason; never invent values.
Gather at the outer boundary once, with no repository/harness requirement in an
installed package. Git discovery uses a bounded subprocess only inside a checkout;
no environment-variable dump, remote URLs or automatic dirty-patch capture.

Large checkpoint/dataset hashes use existing trusted metadata; never re-read huge
artifacts in the logging hot path. An artifact path/hash identifies evidence,
not its trustworthiness or full physical reproducibility.

An outer boundary calls `write_resolved_config()` once after its already-resolved
runtime configuration and available identities are known. The startup manifest is
not mutated. A failed explicit write is not replaced by a logging-only fallback and
leaves the run incomplete; a library close without an outer runtime record carries
an explicit unavailable reason rather than claiming a full runtime identity.

Startup manifests write atomically. summary.json is atomic at graceful close and
records succeeded/failed/interrupted plus logs_complete and counters. A run without
a summary is incomplete, not automatically failed or successful. Dataset/benchmark
acceptance checks must reject incomplete audit evidence where it is required.

## Correlation and event schema

Every event has schema_version1, UTC timestamp, process-monotonic timestamp_ns,
process_id, rank, per-process sequence number, severity, component, event name,
run_id, optional span_id/parent_span_id and bounded fields. Domain context supports
episode_id, decision_id, candidate_id, branch_id, transition_id, boundary_index,
head_id and origin real/imagined. These are structured fields, not parsed from text.

IDs scope correctly: candidate index0 in another decision is a different candidate.
Use producer's stable data IDs where available; create local counters otherwise.
Missing context stays null. Context is immutable and inherited with contextvars;
explicitly pass IDs across threads/processes. Per-process streams avoid concurrent
file appends. Monotonic clocks order within a process only; multi-process inspection
uses wall time for display, never claims a universal causal order from timestamps.

Spans record start/end, parent, operation, duration_ns, outcome and exception
reference. A missing end is displayed as unfinished. Host spans measure enqueue/
host elapsed time, not completed GPU compute. Existing explicitly synchronized
latency measurements retain their meaning and are recorded with timing_kind.
Logging must not add torch.cuda.synchronize(), .item(), .cpu() or tensor reductions
merely to generate a log. Expensive numeric inspection is a separately enabled,
bounded diagnostic operation with its overhead recorded.

Public interfaces (planned):

```python
RunRecorder.start(config, *, command, metadata) -> RunRecorder
recorder.bind(**ids) -> BoundRecorder
recorder.event(name, *, level="INFO", component, fields=None) -> None
recorder.span(name, *, component, fields=None) -> ContextManager[BoundRecorder]
recorder.metrics(values, *, axis, step, component, units=None) -> None
recorder.exception(error, *, component, fields=None) -> None
recorder.capture(name, arrays, *, context, metadata) -> CaptureResult
recorder.write_resolved_config(runtime_config, *, metadata, unavailable) -> None
recorder.close(*, status, error=None) -> RunSummary
```

No-op implementation accepts the same calls and performs no work. Context managers
record exception chains then re-raise the original error; logging/cleanup failure
must not replace the operational exception. Logging disabled never skips validation
or changes numerical/control behavior.

## Event vocabulary and integration

| Boundary | Required diagnostic records |
| --- | --- |
| CLI/artifacts | run start/end, load/validate spans, resolved identities, explicit config/profile rejection |
| P01 execution | commanded root/world targets and grip, requested/achieved duration, substeps, before/after pose/grip when already host-side, safe-hold/cleanup failures |
| P02/P08 data | lineage/split IDs, attempt/branch/trial start/end, rejection/censor/timeout reason, replay mode/discrepancy references, observed-through horizon coverage |
| P03/P04/P07 physical path | encode/decode/memory/dynamics spans, tensor metadata, physical lineage, boundary/origin, fixed head, validation error; no task labels in physics inputs |
| P05/P06 context | event/window IDs, invalid windows, selected route, probabilities when already available, context hash, task reset/replay events |
| P09 evaluation | active horizon, V/S/progress, calibrated terminal probabilities/temperature ID, output validation; original finite/nonfinite status |
| P10 search | candidate/parent IDs, U/F/active masses, leaf return, visits, budget/call/cache counters, selection/tie/fallback reason, incomplete evaluation |
| P11 training | phase/update, loss components, optimizer LR/gradient summary already computed, sampling/seed/config IDs, validation metrics, checkpoint/resume events |
| P12/P13 evaluation | requested/completed/invalid episodes, outcome/false-stop/timeout, per-task metrics, support/regret and trace links; no unobserved oracle cause asserted |

Implement only existing boundaries initially; update plans for components not yet
present. Wrapping PhysicalRollout collaborators supplies encode/decode/memory spans
without putting an observability dependency in model classes. Native candidate
proposals preserve existing measured seconds and seeded call ordering. Future
search must emit decisions itself; never fabricate a candidate-ranking record from
the chosen action alone.

Failure attribution is evidence, not automatic diagnosis. An exception identifies
where execution failed; low value or ensemble disagreement does not prove causal
fault. Cross-link causal identifiers and preserve unknown cause explicitly.

## Configuration and modes

Add an `observability` section to the existing primary JSON. All values below become
central defaults, not duplicated Python literals. Native CLIs accept optional
`--logging-config`, `--log-dir`, `--log-level` and `--trace-mode`; precedence is
packaged logging defaults, user-local CLI `.env`, explicit file, explicit CLI. The
project-root `.env` is read only by CLI startup (never library calls), accepts the
`ICGS_WANDB_*` variables shown in `.env.example`, and may set `WANDB_API_KEY` for
the W&B SDK without invoking `wandb login`; exported environment values take
precedence over `.env`. It is intentionally ignored by Git. Logging-only file envelope
is `{schema_version:1, observability:{...}}`; reuse the exact typed section rather
than a second schema. No implicit configuration file search occurs outside this
user-authorized project-local `.env`.

| Setting | Default / behavior |
| --- | --- |
| mode | normal; alternatives off/debug/capture |
| output_dir | outputs/runs; relative to explicit config file, otherwise invocation cwd |
| console_level | INFO |
| file_level | INFO; debug/capture effective threshold DEBUG |
| trace_components / trace_episodes | empty tuple = all available components/episodes at selected mode |
| metric_every_steps |100; validation/end/exception records always emitted |
| queue_capacity / reserved_critical_slots |8192 /256 |
| flush_interval_s / shutdown_timeout_s |1 /2 |
| event_chunk_bytes / max_event_chunks_per_process |10485760 /10 |
| metrics_total_bytes_per_process |67108864 |
| max_record_bytes / max_field_string_chars |65536 /2048 |
| recent_event_capacity |200 in-memory compact events |
| capture_max_count / capture_max_bytes_each / capture_total_bytes |3 /16777216 /50331648 |
| capture_arrays |false, including capture mode until explicitly enabled |
| wandb.enabled / wandb.mode |false /offline; online requires explicit choice |
| wandb.project / entity / tags |icgs /null /empty tuple |
| wandb.metric_prefixes |train., val., evaluation., planning., runtime. |
| wandb.send_config / upload_artifacts |false /false |

Normal mode captures lifecycle, failures, metric samples and compact decision
summaries. Debug adds selected component spans and rejected-candidate summaries.
Capture adds bounded failure bundles with recent trace/context; raw arrays require
the additional explicit opt-in. Off creates no recorder files/threads/backend.
Library defaults remain no-op; CLI chooses mode centrally. Mode affects evidence
volume and overhead, never the algorithm's work budget or RNG.

Sampling is deterministic by update/decision index modulo configured cadence,
not random sampling. WARNING/ERROR/final results bypass sampling. Mark skipped
detail counts and selected filters. Do not claim debug traces cover all branches
when filtering/sampling was applied. Changed logging config affects full run ID/
config identity but not frozen reference fingerprint; enabled overhead still
counts in actual wall-time evaluation.

## Serialization, capture and sensitive data

Allow small JSON primitives/mappings/sequences with bounded depth and record size.
Never fall back to arbitrary repr(), pickle, tensor.tolist() or object introspection.
Shape/dtype/device/requires_grad metadata is safe without values; available host
scalars and numeric arrays can be summarized only at declared boundaries. Nonfinite
metric becomes value:null plus nonfinite:true and an error event, not invalid JSON
or a zero that changes statistics. Metric names preserve original native names;
new dotted names use explicit axes optimizer_step/episode/decision/transition.
Do not mix axes into one artificial global step or average absent metrics as zeros.

Redact case-insensitive sensitive keys password/token/secret/api_key/authorization/
credential and their descendants; do not capture environment, exception locals,
complete argv, signed URLs or credentials. Sanitize URL query strings and bounded
exception text; retain traceback frames and causal chains without local variables.
This reduces risk, not a guarantee that arbitrary user free text contains no secret.
Keep local files user-readable/writable only where supported. W&B mirrors only
allowlisted numeric metrics and run/trace IDs by default, not raw event text.

Capture accepts only explicitly supplied numeric CPU NumPy arrays, validated dtype,
bounded entry/key inventory and source byte size before copying; no CUDA transfer is
initiated by recorder. Store NPZ with allow_pickle=False reader and SHA256/shape/dtype
manifest. Deep-copy admitted arrays at capture submission to prevent later mutation.
The NPZ writer is bounded before disk output, and quotas aggregate the complete
archive-plus-manifest bundle per run; `CaptureResult` and summary bytes use that same
stored-byte definition. Reject oversized captures rather than truncate tensors into
misleading reproduction fixtures. The rank0 process owns captures in v1. Other
processes record references/requests but do not race on a shared capture quota. Emit
capture-skipped reason.

Snapshots of simulator internals remain P08 replay artifacts, not generic logger
captures. A diagnostic bundle supports an explicitly designed reproducer, not
automatic safe replay or arbitrary deserialization/execution.

## Backpressure, retention and failure behavior

Use one bounded producer queue/writer thread per process. Producers serialize
bounded host metadata and enqueue without blocking control; detail fills only the
nonreserved queue capacity. Critical events use reserved slots; if full, write a
bounded stderr warning and increment loss counters rather than block motion or
throw over the original error. Emit accumulated drop totals when a writer recovers.
The queue contains copied immutable primitive records, never references to tensors.

At event chunk limit, stop accepting further debug/info events, preserve existing
chunks, and reserve remaining emergency/summary capacity for failure/completeness
records; never delete old evidence automatically. A critical record that cannot fit
the configured event chunks may use only the finite emergency bound
`(max_record_bytes + 1) * reserved_critical_slots`; there is no unbounded bypass.
Metrics stop at their cap and mark incomplete. No implicit retention deletion or
cross-run cleanup. Short writes, flush/close failures and unknown disk-full/write
errors disable or account for the failing sink, warn once plus final summary
counters, and leave the computation's original outcome intact. If file output cannot be
initialized, fail CLI startup before any model/control work rather than silently
claim a local recorded run. Mid-run logging failure never independently selects
an action, cancels controller motion or changes task outcome.

Flush periodically and on graceful failure/end, with bounded close timeout. A failed
outer resolved-config write is not replaced by a logging-only fallback. No
claim of power-loss/fsync durability; abrupt process death may lose buffered events
or leave one partial final line. Reader ignores only an incomplete final JSONL
line with explicit warning; interior malformed records are corruption errors.
Partial evidence cannot satisfy required research-audit gates, even if numerical
task execution succeeded. Report task outcome and observability completeness separately.

## Optional W&B integration

Local recording is always primary when enabled. Import W&B only at explicit
backend startup; off/local mode does not import or initialize it. Reuse the existing
optional training dependency, no new remote service or agent telemetry framework.
No automatic login or upload in setup/tests. Requested-but-missing W&B is a startup
configuration error; a later service failure disables only that mirror and records
it locally. Network work uses a separate bounded queue from local writes. The
existing daemon worker serializes all `run.log` calls before `run.finish`; close only
signals and joins it for the central shutdown deadline, recording local mirror loss
if that deadline expires.

Rank0 publishes numeric metrics using explicit step axes. Store W&B run ID locally;
native --use-wandb is an explicit online opt-in mapped through the same adapter.
Do not initialize both the old WandbLogger and a second W&B run. Bridge Lightning's
logger API to the local recorder and optional single remote backend; maintain
Train_Loss/Val_* names, existing epoch/step aggregation and checkpoint switches.
record=False no longer implies that library code may secretly initialize a remote
logger; remote use is controlled explicitly. Changes in this orchestration must
be documented separately from unchanged native training mathematics.

## Offline inspection

Add lightweight `icgs logs inspect <run-dir>` for run status, config/provenance,
error/drop/capture totals and missing evidence; `icgs logs trace <run-dir>` accepts
episode/decision/candidate filters and prints parent/child spans and linked records.
`--json` emits machine-readable results; human tables use stdout and warnings stderr.
Commands are stdlib-only, handle multiple process streams without assuming total
clock ordering, and never import model/simulator/W&B packages or execute captures.
They refuse path traversal and symlink references in capture paths, validate capture
identity/manifest size/SHA256 without deserializing NPZ data, and enforce one
aggregate byte budget across bounded JSON/JSONL/manifest/archive reads. Directory
enumeration is bounded before sorting, including empty JSONL files.

## Acceptance

Logging tests cover context nesting/exception restoration, independent threads and
streams, finite serialization/redaction, queue saturation, disk failure/partial
records, exclusive run directory creation, bounded captures, optional-backend
failure and offline inspection. Default behavior comparisons verify RNG state,
model keys/outputs and command ordering unaffected; no extra GPU synchronization.
Tests may simulate writer faults and backend SDK boundaries, never use a fake
logger to assert that missing neural or simulator behavior is correct.

Run focused installed-environment tests, existing CLI/training/geometry/memory/
candidate regressions, both L0 commands and isolated installed CLI help/inspection.
Native touched inference/training boundaries retain required published compatibility
gates; any unavailable C1–C5 is reported NOT RUN/SKIPPED, never replaced by a trace
unit test. No actual training, network upload, simulator or robot workload is
authorized by writing or reading this design.
