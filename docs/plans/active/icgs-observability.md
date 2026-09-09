# ICGS local logging and correlated tracing implementation plan

Status: active — Tasks 1–5 runtime implemented; Task 6 acceptance evidence and
documentation recorded below. C1–C5 remain mandatory and explicitly NOT RUN.
Category C cross-component infrastructure. Owner: repository owner / assigned worker.
Approved direction: local logging + optional W&B + correlated traces + bounded capture.
Spec: [observability](../../components/observability.md).

## Goal and boundaries

Enable offline tracing of a failure across ICGS components with honest evidence
completeness, preserved native behavior and centralized settings. Current native
Lightning/W&B metrics and CLI output are partial logging, not a common trace store.

Use stdlib logging, JSONL, contextvars, queues and filesystem primitives; NumPy only
for explicit capture, Lightning/W&B only in optional outer adapters. No runtime
imports of docs/tests/scripts, no model dependencies on observability, no callbacks
stored in numeric state. Existing model initialization/RNG/numerical semantics stay.
No training/download/network/simulator/robot workload as implementation validation.

## Ownership and dependency sequence

1. **Configuration and records:** `src/icgs/configuration/method.py`,
   `method_schema.py` only if necessary, primary JSON; create
   `src/icgs/observability/{__init__,context,records,serialization}.py`.
2. **Local lifecycle/writer:** create `observability/{recorder,local}.py`.
3. **Capture/inspection:** create `observability/{capture,inspect}.py` and CLI
   `src/icgs/cli/logs.py`.
4. **Existing execution seams:** CLI/composition/execution/candidate and physical
   rollout wrappers, with no edits to mathematical neural forwards.
5. **Training/remote adapter:** create `observability/{lightning,wandb}.py`; update
   native training logger construction and test it without running training.
6. **Integration documentation and gates:** update P00/P01/P02/P03/P04/P07/P08/
   P09/P10/P11/P12/P13 with actual versus future event producers. No phantom
   producer for an unimplemented search/router/evaluator.

Do not have multiple workers edit central configuration, CLI dispatch or the same
execution owner concurrently. Foundation interfaces merge before dependent tasks.
Models remain unaware of logging; traced callable wrappers and outer event
producers own failures/timing and optional metadata summaries.

## Task 1 — Central configuration, strict event types and correlation

Tests: `tests/test_observability.py`; method-config tests.

- [x] RED/GREEN: defaults enable normal CLI local logging with W&B disabled; library
  no-op starts no thread/file. Reject unknown keys, invalid levels/modes, zero caps,
  reserve>=queue capacity and inconsistent capture quotas. Test old method-file
  overlays still load, full config hash includes logging but reference hash does not.
- [x] RED/GREEN: nested bind/span contexts propagate run/episode/decision/candidate IDs;
  exceptions restore parent context, two threads do not inherit accidental IDs.
- [x] Implement frozen records and recorder capability from the spec. IDs use OS
  UUID/counters, never policy RNG. Distinguish process-local sequence/clock from
  cross-process causal IDs; preserve before/after domain boundary identifiers.
- [x] Implement bounded JSON serialization: no repr/pickle/implicit tensor values,
  strict finite output with explicit invalid metric records; sensitive-key and
  URL-query redaction; traceback frames without locals.
- [x] GREEN: run `.venv/bin/python -B -m unittest discover -s tests -p
  'test_observability.py' -v` and `test_method*.py`; record selected/executed/skips.

Core regression fixture (turn into TestCase methods using actual recorder types):

```python
with recorder.bind(episode_id="e1", decision_id="d2").span(
        "policy.propose", component="proposal") as span:
    span.event("candidate.created", component="proposal", fields={"index": 0})
# Parsed records: same episode/decision; candidate event parent is the proposal
# span; end duration >= 0; parent context after exit is unchanged.
```

Numeric example fixtures are test expectations, not additional default sources.
Do not use blanket source-string bans on numbers to test this design.

## Task 2 — Local writer, run manifest and completeness

Tests: `tests/test_observability.py`.

- [x] RED/GREEN: two simultaneous run starts cannot overwrite files. Missing/invalid
  output directory fails before mocked model-entry callable is invoked. Each
  process uses a distinct JSONL stream; repeated initialization has no duplicate
  console/file records and leaves unrelated handlers intact.
- [x] Implement exclusive run directory, immutable startup/resolved metadata and
  atomic final summary. Metadata includes source/config/artifact identities only
  when available. Resume gets a fresh run ID plus explicit parent link.
- [x] Implement bounded nonblocking queue, critical reserve, periodic flush,
  chunk/metric caps, dropped counters and bounded graceful close. Preserve old
  evidence, no automated deletion/retention cleanup.
- [x] RED/GREEN fault cases: saturate a small queue, inject short write/disk full,
  fail summary write, flush/close during an operational exception. Assert surfaced
  stderr/completeness counters and original error preserved; no fabricated SUCCESS.
  Periodic flush and bounded unstarted-writer close are covered. Power-loss/fsync
  durability is explicitly out of scope, not a missing acceptance gate.
- [x] Prove mode off does not inspect lazy event fields, spawn threads or touch
  filesystem. No global root logger or environment mutation.

Separate task outcome from logs_complete in summaries. After abrupt interruption,
missing summary is incomplete. Queue drops cannot silently disappear from evidence.

## Task 3 — Bounded capture and offline trace inspection

Tests: `tests/test_observability_capture.py`, `tests/test_observability_cli.py`.

- [x] RED/GREEN: captures disabled by default; capture mode without array opt-in records
  recent events/context only. CPU numeric arrays within quota roundtrip identically;
  object arrays/oversized/GPU tensors are rejected without repr or device transfer.
- [x] Implement explicit copied numeric capture, safe NPZ, checksum/shape/dtype
  manifest, per-run rank0 quota and capture-skipped events. Never deserialize or
  execute a simulator snapshot through this mechanism.
- [x] RED/GREEN: mutate caller array after capture submission; saved artifact must retain
  submitted values. Repeated captures stop at count/bytes caps without deleting
  previous artifacts. Failure to capture cannot replace the original exception.
- [x] Implement `icgs logs inspect` and `icgs logs trace` filters/JSON output.
  Read records incrementally with limits, mark unfinished spans/drops and tolerate
  only a partial final line. Interior corruption and malicious capture paths fail.
- [x] Verify installed commands from outside checkout with no torch/RLBench/W&B
  imports. Human logs stderr, command data stdout. No browser/dashboard required.

## Task 4 — Existing CLI, proposal, physical and execution integration

Tests: `tests/test_observability_integration.py`; current CLI/candidate/timed tests.

- [x] Add CLI logging flags and logging-only configuration adapter that uses the
  central typed section. Leave native model JSON/strict published checks untouched.
- [x] Initialize/close run in a try/except/finally outer scope. Log input validation,
  artifact loading, context preparation, proposal and output saving spans without
  changing existing stdout results, saved inference fields or invocation order.
- [x] Wrap existing PhysicalRollout encoder/decoder/dynamics/memory collaborators
  for host spans. Observe only metadata or already-computed host values; no new
  tensor conversion/reduction or GPU synchronize for ordinary logs.
- [x] Add optional recorder seams to existing candidate and execution orchestration;
  log candidate seeds/source identity and existing timing, selected absolute command
  and measured result, safe-hold/cleanup exception chain. Library defaults no-op.
- [x] Compare enabled/disabled under fixed seeds: same next Python/NumPy/torch RNG,
  same call ordering and a real `PhysicalMemory` component's outputs/state-dict
  keys. This is a bounded host comparison, not neural fidelity evidence and does
  not substitute for published acceptance. No new CUDA synchronization was added;
  CUDA-spy evidence remains **NOT RUN** because CUDA is unavailable.
- [x] Log ownership/context failures with observed/imagination provenance; do not
  infer an unavailable task/branch/event ID merely to fill a schema field.

Logging overhead is real wall time; do not subtract it from benchmark budgets.
Capture runs are diagnostic tracks and cannot be mislabeled as normal latency runs.

## Task 5 — Local training metrics and optional W&B

Tests: `tests/test_observability_training.py`; current native training regression.

- [x] RED/GREEN: adapter-level native metric-name/record/use-wandb behavior is
  covered with actual logger construction and patched SDK transport seams. The
  `record=True/use_wandb=False` undefined-logger defect is fixed in the named
  outer logger construction path.
- [x] Conditional native Trainer construction coverage enumerates all four
  record/use-wandb combinations without fitting. It is **SKIPPED** in the
  installed environment because Lightning is unavailable; no fake import or
  training run substitutes for that evidence.
- [x] Implement Lightning logger bridge to local recorder, preserving native metric
  names and epoch/step aggregation. No extra model aliases, optimizer work or
  checkpoint-format changes. Configure LR monitoring consistently with logger use.
- [x] Implement lazy optional W&B adapter with rank0 ownership, separate bounded
  remote queue and one run ID. Online is explicit; default backend disabled/offline.
  No implicit login, credentials persistence, full-config or capture uploads.
- [x] RED/GREEN: disabled backend never imports W&B; missing explicitly requested
  dependency rejects startup; a patched SDK transport failure disables only the
  mirror while local metrics remain authoritative and the failure is reported.
  Tests patch the SDK transport seam only and never contact the real service.
  Patched SDK transport, failure/disable, and shutdown seams are **PASS**.
  Actual no-training Lightning Trainer construction is **SKIPPED** because
  Lightning is unavailable. Live W&B service/network transport is **NOT RUN**
  and **OUT OF SCOPE**, regardless of installed dependency; generic remote/live
  checks are not required acceptance gates.
- [x] Explicit step axes prevent train/episode/decision counters from colliding.
  NaN remains invalid in JSONL and is not sent as a normal numeric W&B metric.

## Task 6 — Cross-component documentation and acceptance

- [x] Update architecture/map/parameter reference and declare implemented recorder
  versus planned event producers. Add logging sections and events to each affected
  component plan; preserve their prior progress/validation evidence.
- [x] Record durable dependency direction/config/retention decision through the
  repository decision template; do not introduce observability into model ownership.
- [x] Run the six new test owners and focused current regressions; build/install
  package and exercise installed parser/import/inspection probes without checkout.
  Cheap successful command-body lifecycle seams are tested separately. Both L0
  commands must pass. Keep exact command/interpreter/count/error/skip evidence in
  this plan.
- [x] Validate file size/queue/capture bounds using tiny configured limits and
  deterministic fixtures, not huge stress jobs. The finite critical emergency
  bound, pre-copy capture rejection, concurrent quota admission, permissions and
  bounded close tests pass. A separate host-overhead benchmark is **NOT RUN**.
- [x] Review run reconstruction: one known synthetic failure can be traced from
  decision/candidate through nested physical spans to error and capture, while a
  dropped/missing segment is explicitly identified as unknown.

Native affected paths require existing fidelity gates before renewed equivalence
claims; local tests cannot certify full published inference. Model/controller
workloads remain explicitly authorized separately. Requested remote logging does
not authorize a network test or upload during implementation.

## Recovery and completion

Disable the new recorder through mode off to recover original computation; retain
failed trace evidence. Revert only scoped changes after inspecting user work; no
deletion of run directories or baseline artifacts. Do not change completed-run
statistics based solely on whether a log writer succeeded.

## Implementation progress and evidence — 2026-09-09

The dependency sequence is implemented in one worker on `main`; no commit, push,
or worktree was used. The manager briefly created and safely deleted
`codex/icgs-observability` before implementation, after verifying it matched
`main`; all implementation edits were made directly on `main`. No training job,
download, preprocessing job, network transport, simulator workload or robot motion
was used.

| Task | Current implementation/evidence |
| --- | --- |
| 1 | **Implemented.** `ObservabilityConfig`/`WandbConfig` are typed from the packaged JSON; strict overlays, contextvars, no-op capability, bounded serialization/redaction and exception records are covered by `tests/test_observability.py` plus the method-config suite. |
| 2 | **Implemented.** Exclusive run identity, atomic startup/resolved/summary files, per-process JSONL writer, bounded queue/chunks/metrics, owned handlers and completeness counters are covered by `tests/test_observability_local.py`. Short writes, disk-full writes, periodic flush/close faults, summary-write failure, metric-cap loss and bounded shutdown are green; power-loss/fsync durability is explicitly out of scope. |
| 3 | **Implemented.** Explicit copied CPU NumPy capture, bounded entry/key admission, whole-bundle NPZ+manifest quota accounting, manifest/archive integrity checks, partial-line reader, traversal refusal and stdlib-only `icgs logs inspect/trace` are covered by `tests/test_observability_capture.py` and `tests/test_observability_cli.py`. |
| 4 | **Implemented.** CLI flags/lifecycle, candidate/evaluation/physical/timed-execution recorder seams and fixed-seed candidate RNG preservation are implemented. A real `PhysicalMemory` component comparison covers enabled/disabled Python/NumPy/torch RNG probes, call order, state-dict keys and exact output; focused evidence is in `tests/test_observability_integration.py` and existing candidate/evaluation/timed tests. Published/native inference equivalence and CUDA-spy evidence are not re-claimed. |
| 5 | **Implemented with bounded validation.** Local Lightning bridge, rank-0 lazy W&B adapter, native metric-name allowlist, disable-on-first-transport-failure behavior, serialized worker-side finish with central shutdown timeout, and the `record=True/use_wandb=False` logger initialization fix are implemented. Patched SDK transport/failure/disable/shutdown seams are executed **PASS**; actual no-training Lightning Trainer construction is **SKIPPED** because Lightning is unavailable, and live W&B service/network transport is **NOT RUN** and **OUT OF SCOPE**, regardless of installed dependency. No tensor normalization or CUDA transfer/sync was added. |
| 6 | **Partially complete.** Documentation/ADR and producer ownership addenda are recorded. Focused tests, both L0 commands, installed parser/import/inspection probes, command-body seam tests, and the bounded reconstruction audit are green. Required C1–C5 and actual no-training Lightning Trainer construction remain missing. Patched SDK transport/failure/disable/shutdown seams are **PASS**; live W&B service/network transport is **NOT RUN** and **OUT OF SCOPE**, regardless of installed dependency. Generic remote/live checks are not required acceptance gates. |

### Sol audit round 1 disposition — 2026-09-09

The first read-only Sol audit rejected acceptance. Its failed evidence is preserved
below; the malformed fabricated link in finding 2 was ignored, and the real span
exit location is `src/icgs/observability/recorder.py`.

| Finding | Disposition and evidence |
| --- | --- |
| 1 writer faults/bounds | **Fixed and tested.** Short writes, disk-full writes, periodic flush/close failures, summary-write failure, metric-cap loss, finite critical emergency capacity and bounded close are covered by `test_observability_local.py`; failures set counters/incompleteness without replacing the operational error. |
| 2 fault isolation | **Fixed and tested.** Shared recorder span enter/exit, capture, exception formatting and close boundaries account diagnostic failures, restore context and preserve the original exception; real span-entry UUID failure and `UnprintableError`/failing `_emit` regressions pass. The timed outer adapter also contains a failing optional `span()` seam. |
| 3 identity | **Fixed and tested.** Outer boundaries write the sanitized resolved runtime config, available identities and explicit unavailable reasons once; startup `run.json` remains immutable, a failed explicit write is not replaced by a logging-only fallback, and the CLI uses the loader's already-computed checkpoint identity without a second large hash. |
| 4 P01 evidence | **Fixed and tested.** The timed transition record contains bounded host-side commanded target/grip, before/after pose/grip, both boundary IDs, requested/achieved timing, substeps and available sensor provenance; ordering and controller semantics remain covered by the timed suite. |
| 5 training/W&B | **Fixed within available scope.** Native `Train_Loss`/`Val_*` names survive the allowlist and a first remote transport failure disables only the mirror while local logging remains. No tensor normalization, CUDA transfer or sync was added; actual Lightning callback/Trainer validation is **SKIPPED** because Lightning is unavailable, and no live W&B transport ran. |
| 6 thresholds/no-op | **Fixed and tested.** Effective file thresholds are admitted before field serialization; disabled spans create no UUID, context mutation or lazy-field access. |
| 7 capture concurrency | **Fixed and tested.** Metadata-only capture works with raw arrays disabled; manager initialization, lazy recent-event capture and quota admission are bounded and locked, and oversized arrays are rejected before copying. |
| 8 permissions | **Fixed and tested.** Runtime-owned directories, files and temporary files request explicit `0700`/`0600` modes and runtime code never mutates the process umask. |
| 9 offline inspection | **Fixed and tested.** JSON manifests and JSONL lines are read under limits, traversal/symlink capture references are refused, and missing/corrupt/partial evidence forces `logs_complete=false` rather than crashing or remaining falsely complete. |
| 10 evidence/reporting | **In progress and recorded below.** Main-only implementation history, the six test owners, installed parser/import/inspection probes, successful command-body seam tests, real component comparison, exact validation results, C1–C5 `NOT RUN`, and out-of-scope power-loss/fsync durability are documented without converting skips into passes or claiming plan completion. |

### Sol audit round 2 disposition — ready for re-audit — 2026-09-09

The second read-only Sol audit rejected five concrete gaps. The following narrow
fixes were implemented on `main`; this is a disposition for the requested same-Sol
re-audit, not an acceptance or plan-completion claim.

| Finding | Disposition and regression evidence |
| --- | --- |
| 1 capture quota | **Fixed and tested.** Raw-array entry count and key length are bounded before copying; an explicitly bounded non-seekable NPZ writer prevents oversized temporary archives; actual archive plus manifest bytes enforce per-capture and aggregate quotas; `CaptureResult` and summary totals report the complete bundle; exact unpublished rejected directories are cleaned without touching accepted captures. `tests/test_observability_capture.py` covers empty-array overhead, pre-copy entry/key rejection, preservation of prior bundles and complete-byte accounting. |
| 2 remote close | **Fixed and tested.** The existing daemon worker drains `run.log` calls and invokes `run.finish` only after them. `close()` only signals and joins for the configured recorder shutdown deadline; timeout records a local mirror error and returns. Blocked-log, blocked-finish, ordering, timeout and original-operational-error seams pass in `tests/test_observability_training.py`; no live SDK transport ran. |
| 3 capture integrity | **Fixed and tested.** Inspection validates schema/identity/name/array metadata and archive size/SHA256, hashes archives incrementally without NPZ deserialization, and marks corrupted or malformed captures incomplete even when summary count is zero. Corrupt archive, corrupt manifest and malformed-zero-summary regressions pass in `tests/test_observability_cli.py`; metadata-only capture remains valid. |
| 4 inspection bounds | **Fixed and tested.** One aggregate read budget covers run/summary JSON, JSONL, capture manifests and archive hashes. Directory enumeration admits at most `limit + 1` entries before sorting, including empty JSONL files; overflow and multi-bundle budget regressions pass in `tests/test_observability_cli.py`. |
| 5 CLI/evidence/docs | **Fixed within available scope and tested.** Infer/train/prepare-data/evaluate command bodies have bounded collaborator seams asserting startup-before-work, exactly-once resolved identity, success close and preserved output/order. Installed `--help` remains a parser/import probe only. Patched SDK failure/disable checks are executed **PASS**; live W&B and native Lightning Trainer validation remain **NOT RUN/SKIPPED**, and C1–C5 remain mandatory **NOT RUN**. Historical failed evidence below is preserved. |

### Preserved failed evidence

These failures were observed before the corresponding implementation and remain
recorded rather than being rewritten as passes:

- `.venv/bin/python -B -m unittest tests/test_observability.py -v` — **FAIL**,
  5 errors from the absent observability section/package (the invalid-key case
  passed against the old schema).
- `.venv/bin/python -B -m unittest tests/test_observability_local.py -v` — first
  **FAIL**, one missing span-end record under an intentionally tiny 512-byte test
  chunk; the test fixture was corrected to isolate correlation behavior from the
  chunk-cap case, and the saturation case remains separately covered.
- `.venv/bin/python -B -m unittest tests/test_observability_capture.py tests/test_observability_cli.py -v`
  — first **FAIL**, one intentionally inconsistent test quota and the expected
  missing `logs` command; both were corrected before the green rerun.
- `.venv/bin/python -B -m unittest tests/test_observability_integration.py -v`
  — initial **FAIL**, both seam tests rejected the new `recorder` keyword before
  the optional seams were added.
- `.venv/bin/python -B -m unittest tests/test_observability_training.py -v`
  — initial **FAIL**, the optional adapter modules did not yet exist.
- `.venv/bin/python -B -m unittest tests.test_timed_execution.TimedExecutionTests.test_timed_adapter_records_measured_transition_and_cleanup_chain -v`
  — initial **ERROR** because the timed adapter did not yet accept the optional
  recorder seam; the focused test then passed after the boundary-only addition.
- `.venv/bin/python -B -m unittest tests.test_observability_training.TrainingObservabilityTests.test_non_rank0_wandb_skips_sdk_and_preserves_local_recorder -v`
  — initial **FAIL** because the enabled adapter imported W&B on rank 1; the
  rank-gate test then passed without importing the SDK.

The first package-probe commands were also retained as environment evidence:

- `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <tmp>` — **FAIL**;
  this installed test interpreter has no `pip` module.
- `python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir <tmp>` — **FAIL**;
  host CPython 3.14.4 is outside the declared project range `<3.13`.
- Direct setuptools backend wheel creation followed by `uv pip install --python
  .venv/bin/python --no-deps --target <tmp> <wheel>` — **PASS**; this was the
  compatible no-download installed-package path used for the final probe.

### Focused green evidence

- `.venv/bin/python -B -m unittest tests/test_observability.py tests/test_observability_local.py tests/test_observability_capture.py tests/test_observability_cli.py tests/test_observability_integration.py tests/test_observability_training.py -v` — **PASS** before the timed/rank-gate additions, 22 selected, 22 passed, 0 failures/errors/skips.
- `.venv/bin/python -B -m unittest tests/test_observability.py tests/test_observability_local.py tests/test_observability_capture.py tests/test_observability_cli.py tests/test_observability_integration.py tests/test_observability_training.py tests/test_timed_execution.py -v` — **PASS**, 47 selected, 47 passed, 0 failures/errors/skips.
- `.venv/bin/python -B -m unittest tests/test_observability.py tests/test_observability_local.py tests/test_observability_capture.py tests/test_observability_cli.py tests/test_observability_integration.py tests/test_observability_training.py tests/test_timed_execution.py -v` — **PASS with one expected skip; not a full PASS**, 79 selected, 78 passed, 0 failures/errors, 1 skipped (`Lightning` unavailable for native Trainer construction).
- `.venv/bin/python -B -m unittest tests/test_method_config.py tests/test_candidates.py tests/test_evaluation.py tests/test_cli.py tests/test_timed_execution.py -v` — **executed subset passed; not a full PASS**, 59 selected, 58 passed, 1 skipped (RLBench unavailable), 0 failures/errors.
- `.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v` — **PASS with skips; not a full PASS**, 273 selected, 264 passed, 0 failures/errors, 9 skipped. Skips were four differential tests without `IP_LEGACY_SOURCE_ROOT`, one RLBench test, one CUDA test, one opt-in published-checkpoint test, one native Trainer-construction test because Lightning is unavailable, and one existing Lightning training test.
- `python3 -B scripts/validate_fast.py` — **PASS**, L0 selected 19, 19 passed; L1/L2/L3/L4 **NOT RUN**.
- `python3 -B -S scripts/validate_fast.py` — **PASS**, L0 selected 19, 19 passed; L1/L2/L3/L4 **NOT RUN**.
- `git diff --check` — **PASS**; `.venv/bin/python -B -m compileall -q src tests` — **PASS**.

### Task 6 bounded external evidence

The latest package probe used cwd `/tmp/icgs-observability-final.xdyKJ4/outside-installed`,
the wheel built from the compatible `/Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python`
3.11.15 setuptools backend, and the installed target
`/tmp/icgs-observability-final.xdyKJ4/site`. With `PYTHONPATH` set only to that
target and `-S`, the following all **PASS** without network, model, simulator,
Lightning or W&B imports:

- `.venv/bin/python -B -c 'from setuptools.build_meta import build_wheel; import sys; print(build_wheel(sys.argv[1]))' /tmp/icgs-observability-final.xdyKJ4/wheels` — built `icgs-0.2.0-py3-none-any.whl`.
- `uv pip install --python /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python --no-deps --target /tmp/icgs-observability-final.xdyKJ4/site /tmp/icgs-observability-final.xdyKJ4/wheels/icgs-0.2.0-py3-none-any.whl` — installed successfully.
- `PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main --help` — **PASS**, installed parser/import probe only; not a successful command-body lifecycle.
- `PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main logs --help` — PASS.
- `PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main logs inspect /tmp/icgs-observability-final.xdyKJ4/synthetic/20260909T082115.080023Z-bcae873b605143b8a6f03f57bd154e0c --json` — PASS, `logs_complete=true`, 4 events, 0 metrics.
- `PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main logs trace /tmp/icgs-observability-final.xdyKJ4/synthetic/20260909T082115.080023Z-bcae873b605143b8a6f03f57bd154e0c --episode episode-1 --decision decision-1 --candidate candidate-1 --json` — PASS, 3 filtered records, 1 completed span, 0 unfinished spans.
- `RANK=1 PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -c 'from icgs.observability.config import from_mapping; from icgs.observability.wandb import start; assert start(from_mapping({"wandb": {"enabled": True}}).wandb, run_id="rank1") is None; print("RANK1_SKIP_PASS")'` — PASS, rank-1 SDK skip.
- `PYTHONPATH=/tmp/icgs-observability-final.xdyKJ4/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -c 'import sys, icgs.cli.main; bad=[name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "wandb" or name.startswith("wandb.")]; assert not bad, bad; print("NO_MODEL_OR_WANDB_IMPORTS")'` — PASS.

The bounded synthetic fixture produced a failed run with one copied CPU NumPy
capture, candidate and decision IDs, nested `policy.propose` → `physical.rollout`
spans, a linked exception/failure event, a complete summary, and no unfinished
spans. A first fixture assertion intentionally emitted its failure after leaving
the explicit bind, so the filtered trace correctly omitted that event; the final
fixture kept the failure inside the bind and passed without changing runtime code.
Tiny configured quotas and deterministic arrays were used; no stress workload or
retention cleanup was run.

Environment for focused installed tests: `/Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python`, CPython 3.11.15, NumPy/Torch/PyG installed, Lightning and W&B unavailable in that environment. The host `python3` used for L0 is CPython 3.14.4; it was used only for stdlib/static validation.

Remaining evidence: required native C1 artifact/config, C2 strict load, C3
inference, C4 fidelity and C5 standalone execution. The deterministic
short-write, disk-full, periodic flush, close, metric-cap and summary-write
fault cases are covered by the focused suite. Actual no-training Lightning
Trainer construction is **SKIPPED** because Lightning is unavailable. Patched
SDK transport/failure/disable/shutdown seams are **PASS**. Live W&B
service/network transport is **NOT RUN** and **OUT OF SCOPE**, regardless of
installed dependency; generic remote/live checks are not acceptance gates.
Power-loss/fsync durability is explicitly out of scope, not a missing
acceptance test. RLBench execution, differential-source checks, CUDA checks
and published-checkpoint integration remain unavailable or explicitly
unexecuted.
Unavailable or unexecuted gates remain **NOT RUN**, never PASS. This plan stays
active.

### Prior post-documentation verification (before Sol audit round 2) — 2026-09-09

The final post-documentation rerun was performed from
`/Users/33bit/AI/Research/VLA/ICGS` on `main`, using the installed supported
interpreter `/Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python` (CPython
3.11.15) for runtime tests and host `python3` (CPython 3.14.4) only for L0.

- `.venv/bin/python -B -m unittest tests/test_observability.py tests/test_observability_local.py tests/test_observability_capture.py tests/test_observability_cli.py tests/test_observability_integration.py tests/test_observability_training.py tests/test_timed_execution.py` — **PASS with one expected skip; not a full PASS**, 79 selected, 78 passed, 0 failures/errors, 1 skipped (Lightning unavailable for native Trainer construction).
- `.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'` — **PASS with skips; not a full PASS**, 273 selected, 264 passed, 0 failures/errors, 9 skipped (four differential-source checks, RLBench, CUDA, opt-in published checkpoint, native Trainer construction, and existing Lightning training).
- `python3 -B scripts/validate_fast.py` — **PASS**, 19/19 L0 harness tests; L1/L2/L3/L4 **NOT RUN**.
- `python3 -B -S scripts/validate_fast.py` — **PASS**, 19/19 L0 harness tests; L1/L2/L3/L4 **NOT RUN**.
- `.venv/bin/python -B -m compileall -q src tests` — **PASS**.
- `git diff --check` — **PASS**.
- From cwd `/tmp/icgs-observability-final.xdyKJ4/outside-installed`, with only the compatible installed target on `PYTHONPATH` and `-S`: `python -S -m icgs --help`, `python -S -m icgs logs --help`, synthetic `logs inspect --json`, filtered `logs trace --episode episode-1 --json`, disabled-W&B no-import probe, and no-heavy-import probe — **PASS**. Inspection reported `logs_complete=true`, 4 events and 0 metrics; trace reported 3 filtered records, 1 completed span and 0 unfinished spans.
- `git status --short --branch` remains `## main...origin/main [behind 2]`; no commit, push, branch creation or destructive cleanup occurred. C1–C5 and actual no-training Lightning Trainer construction remain **NOT RUN/SKIPPED**, not passes. Patched SDK transport/failure/disable/shutdown seams are **PASS**; live W&B service/network transport is **NOT RUN** and **OUT OF SCOPE**, regardless of installed dependency, and generic remote/live checks are not acceptance gates. CUDA, RLBench, differential-source and published-checkpoint evidence remain **NOT RUN/SKIPPED**. The plan remains active.

### Round 2 implementation verification — 2026-09-09

The round-2 fixes were rerun from cwd
`/Users/33bit/AI/Research/VLA/ICGS` on `main`. Runtime tests used
`/Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python` (CPython 3.11.15); the
two L0 commands used host `python3` (CPython 3.14.4). No training, download,
preprocessing job, network/W&B transport, simulator workload or robot motion
was run.

- `.venv/bin/python -B -m unittest tests/test_observability.py tests/test_observability_local.py tests/test_observability_capture.py tests/test_observability_cli.py tests/test_observability_integration.py tests/test_observability_training.py tests/test_cli.py tests/test_timed_execution.py -v` — **PASS with one expected skip; not a full PASS**, 95 selected, 94 passed, 0 failures/errors, 1 skipped (native Lightning Trainer construction unavailable because Lightning is not installed). This includes the round-2 capture quota/integrity, bounded inspection, W&B close, and four command-body lifecycle seam regressions.
- `.venv/bin/python -B -m unittest tests/test_method_config.py tests/test_candidates.py tests/test_evaluation.py tests/test_cli.py tests/test_timed_execution.py -v` — **executed subset passed; not a full PASS**, 65 selected, 64 passed, 0 failures/errors, 1 skipped (RLBench unavailable).
- `.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v` — **PASS with skips; not a full PASS**, 287 selected, 278 passed, 0 failures/errors, 9 skipped (four differential-source checks without `IP_LEGACY_SOURCE_ROOT`, RLBench, CUDA, opt-in published checkpoint, native Lightning Trainer construction, and existing Lightning training).
- `python3 -B scripts/validate_fast.py` — **PASS**, L0 19/19; L1/L2/L3/L4 **NOT RUN**.
- `python3 -B -S scripts/validate_fast.py` — **PASS**, L0 19/19; L1/L2/L3/L4 **NOT RUN**.
- `.venv/bin/python -B -m compileall -q src tests` — **PASS**.
- `git diff --check` — **PASS**.

Bounded installed-package checks used the no-download local wheel path in
`/tmp/icgs-observability-round2.QEOvch`, with outside cwd
`/tmp/icgs-observability-round2.QEOvch/outside` and the same CPython 3.11.15
interpreter:

- `.venv/bin/python -B -c 'from setuptools.build_meta import build_wheel; import sys; print(build_wheel(sys.argv[1]))' /tmp/icgs-observability-round2.QEOvch/wheels` — **PASS**, built `icgs-0.2.0-py3-none-any.whl`.
- `uv pip install --python /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python --no-deps --target /tmp/icgs-observability-round2.QEOvch/site /tmp/icgs-observability-round2.QEOvch/wheels/icgs-0.2.0-py3-none-any.whl` — **PASS**, no dependency download.
- `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs --help` and `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs logs --help` — **PASS**, installed parser/import probes only; neither is a successful command-body lifecycle.
- `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -c 'from pathlib import Path; from icgs.observability.config import from_mapping; from icgs.observability.recorder import RunRecorder; root=Path("/tmp/icgs-observability-round2.QEOvch/synthetic"); r=RunRecorder.start(from_mapping({"output_dir":str(root),"file_level":"DEBUG","flush_interval_s":0.01,"shutdown_timeout_s":1.0}),command="installed-probe"); span=r.bind(episode_id="episode-1",decision_id="decision-1",candidate_id="candidate-1").span("probe",component="probe"); entered=span.__enter__(); entered.event("probe.event",component="probe"); span.__exit__(None,None,None); r.close(); print(r.path)'` — **PASS**, bounded installed synthetic recorder fixture at `/tmp/icgs-observability-round2.QEOvch/synthetic/20260909T093929.168501Z-a0bdf9f53a70433b9c5aa6520b386293`.
- `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main logs inspect /tmp/icgs-observability-round2.QEOvch/synthetic/20260909T093929.168501Z-a0bdf9f53a70433b9c5aa6520b386293 --json` — **PASS**, 4 events, 0 metrics, `logs_complete=true`.
- `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -m icgs.cli.main logs trace /tmp/icgs-observability-round2.QEOvch/synthetic/20260909T093929.168501Z-a0bdf9f53a70433b9c5aa6520b386293 --episode episode-1 --decision decision-1 --candidate candidate-1 --json` — **PASS**, 3 filtered events, 1 completed span, 0 unfinished spans.
- `RANK=1 PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -c 'from icgs.observability.config import from_mapping; from icgs.observability.wandb import start; assert start(from_mapping({"wandb":{"enabled":True}}).wandb,run_id="rank1") is None; print("RANK1_SKIP_PASS")'` — **PASS**, rank-1 SDK skip without import/transport; the patched SDK transport/failure/disable/shutdown seams are executed **PASS**, while live W&B service/network transport is **NOT RUN** and **OUT OF SCOPE**, regardless of installed dependency.
- `PYTHONPATH=/tmp/icgs-observability-round2.QEOvch/site /Users/33bit/AI/Research/VLA/ICGS/.venv/bin/python -S -c 'import sys, icgs.cli.main; bad=[name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "wandb" or name.startswith("wandb.")]; assert not bad, bad; print("NO_MODEL_OR_WANDB_IMPORTS")'` — **PASS**; no model or W&B imports.

The plan remains active. C1–C5 and actual no-training Lightning Trainer
construction remain **NOT RUN/SKIPPED**, never passes. Patched SDK
transport/failure/disable/shutdown seams are **PASS**. Live W&B service/network
transport is **NOT RUN** and **OUT OF SCOPE**, regardless of installed
dependency; generic remote/live checks are not required acceptance gates. CUDA,
RLBench, differential-source and published-checkpoint evidence remain
**NOT RUN/SKIPPED**. Power-loss/fsync durability remains explicitly out of
scope rather than a missing acceptance gate.

### Latest Sol audit disposition — 2026-09-09

Sol accepted the cheap, scoped runtime after two repair rounds; overall
acceptance remains incomplete because C1–C5 and actual no-training Lightning
Trainer construction are still missing. No over-engineering is requested.
This is a dated clarification of evidence labels, not a rewrite of the
preserved failed evidence; the plan remains active.
