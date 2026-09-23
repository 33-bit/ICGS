# CPU generation validation — 2026-09-22

Status: **FAIL — bounded acceptance completed with simulator failures**

Two remote attempts were made. The first exposed an unbounded refill and was
stopped safely. The corrected third run used commit `ba88ecd` and proved the
validation bound, quarantine and coordinator resume behavior, but the CPU
simulator produced only infrastructure crashes, so the overall receipt remains
FAIL. No production dataset prefix was modified and no HF publication was
performed.

## Environment and commands

- Session: `generation-validation-cpu` (CPU, endpoint `m-s-kkb-usw3b1-2y4u3f90bozrv`)
- Runtime config: `/content/generation-runtime.json`
- Runtime config SHA256: `efc92c7f29824b6dc5c0f81146ccf8c89d2b146b2b335bac1ee5b438c7b40329`
- Workers: 2; configured simulator slots: 2; display base: 20; screen: 1024×768
- Provisioning: PASS — pinned CoppeliaSim, PyRep and RLBench setup completed
- Contract preflight: PASS — 92 focused remote tests after setting the repository cwd
- Launch used canonical `scripts/generation_launch.py` and the JSON validation plan.

## Gate results

| Gate | Status | Evidence / reason |
| --- | --- | --- |
| success | FAIL | No success episode before safe stop |
| valid_failure | NOT RUN | Not reached |
| invalid_observation | NOT RUN | Not reached |
| infrastructure_failure | PASS | 27 `simulator_crash` attempts were materialized |
| malformed_result | NOT RUN | Not reached |
| coordinator_restart | FAIL | Watchdog restarted coordinator on `pid_invalid`; controlled restart gate was not completed |
| publication | NOT RUN | Zero published jobs; no HF commit or remote hash verification |
| resource bound | FAIL | Coordinator refilled to 400 jobs despite validation plan `max_jobs=7` |

The corrected r3 run changed the resource-bound result to PASS: `run.json`
carried `validation_max_jobs=7` and the queue ended at exactly seven ingested
jobs plus one quarantined malformed result. It also recorded PASS for
`infrastructure_failure`, `malformed_result`, and manual coordinator restart
recovery. The required success and valid-failure episodes were not produced;
publication was therefore NOT_RUN.

After r3, commit `231d8f3` hardened the headless renderer contract with
`QT_QPA_PLATFORM=xcb`, software GL/Mesa DRI, GLX-enabled Xvfb and portable
episode-worker paths. Local generation tests (183) and L0 pass. A final CPU
provision attempt was stopped because the canonical environment provisioning
command produced no completion log for an extended period; no workers or
generation were started in that attempt.

The queue reached 380 pending jobs after the safe stop. Workers and the
coordinator/watchdog were terminated before unbounded generation could
continue. The validation receipt and launch/heartbeat summaries are stored
alongside this record; they contain no credentials or episode binaries.

## Follow-up required

The coordinator must derive its refill target from the active validation plan
and never enqueue more than `max_jobs`. A new CPU acceptance run is required
after that fix; this record remains FAIL and the active plan must not be moved
to completed until all required gates pass.

The session was explicitly stopped and `colab sessions` returned no active
assignments.

## 2026-09-23 VPS follow-up (bounded, not full acceptance)

The simulator stack was retested on `vps-a` instead of Colab using the GitHub
checkout at commit `0b29eb724e6759bb207d4c05070359d4b1a7a664`. The dependency
checkout was made writable before task-model generation; the portable builder
then produced all 36 RLBench task Python/TTM pairs. A strict 36-program smoke
receipt contained 34 `success` and 2 retained `valid_failure` results (`T01`,
`T11`), zero simulator crashes, and valid `T+1` timelines for every program.

The canonical distributed launcher was run with two workers, two simulator
slots, `validation_max_jobs=7`, and a fresh run ID. The first VPS attempt
exposed two runtime defects and was stopped without reusing its queue:

- the launcher omitted `--runtime-config` for the coordinator, causing an
  immediate watchdog `pid_invalid` duplicate attempt;
- a `COMPLETE` publication receipt prevented later ingested jobs from opening a
  new HF batch.

Those defects are covered by commits `32471a0`, `a970ca2`, and `0b29eb7`. The
corrected run `generation-validation-vps-b104-20260923` kept a stable
coordinator/watchdog (`restart_history=[]`), never exceeded seven in-flight or
closed jobs, and published three success episodes (`G1`, `G2`, `G3`) to the
isolated prefix
`validation/validation-cpu-20260922/vps-b104/`. The remote prefix contained 76
files and a three-episode dataset manifest; data/final-receipt revisions were:
`b80253d`/`c4a4e96` (G2), `a152045`/`1a525c5` (G1), and
`3374bc1`/`7ca1c96` (G3). No production prefix was modified and the token was
never committed.

The bounded run was stopped intentionally after the third verified publication
to avoid starting full collection. It is therefore evidence for renderer/task
startup, bounded queueing, watchdog startup, multi-batch publication and remote
artifact listing—not a PASS for the full validation plan. Distributed
`valid_failure`, malformed-result, infrastructure-failure and controlled
coordinator-restart gates remain NOT RUN in this VPS follow-up.

## 2026-09-24 VPS acceptance for resumable/multi-host hardening

The GitHub checkout was fast-forwarded to `e31df1f` (the resumable/multi-host
implementation at `6341777` plus the direct launcher-entrypoint fix). The VPS
environment used Python 3.12.14, the pinned CoppeliaSim checkout, PyRep and
RLBench. No credential was committed; the HF token was installed temporarily
as a mode-600 file and removed after the run.

### Gate evidence

| Gate | Status | Evidence / reason |
| --- | --- | --- |
| generation contract suite on VPS | PASS | `PYTHONPATH=src /home/huy2325/icgs-vps-env/bin/python -B -m pytest -q tests/test_generation*.py` — 218 passed |
| simulator readiness | PASS | `generation_simulator_probe.py --worker` under Xvfb; 2 physics steps, advancing clock, finite 128×128×3 cloud |
| current-commit success episode | PASS | G1, 273 actions / 274 observations, valid timeline |
| current-commit valid failures | PASS | T01 and T11, valid `T+1` timelines, no simulator crash |
| current-commit additional smoke | PASS | T02–T09 completed; 8 success and T01 valid-failure in the bounded smoke before safe stop |
| bounded distributed queue | PASS | 2 workers + coordinator + watchdog; queue target remained 7; no restart history |
| HF publication and remote verification | PASS | 4 verified data/final commits under `validation/validation-cpu-20260922/vps-6341777`; public manifest contains 4 episodes and source run `generation-validation-vps-6341777-20260923` |
| HF resume preflight | PASS | New disjoint run downloaded the remote manifest, pinned revision `5dd492641f32e2822755a70690860d60bc5e5140`, and recorded manifest SHA in `resume_bootstrap.json` |
| shared-filesystem worker-only attach | PASS | Host `host-b`, worker scope `001`, host-local runtime snapshot and lease heartbeat verified; worker/watchdog stopped immediately after attach check |
| malformed-result quarantine | NOT RUN live | Unit coverage PASS; no malformed artifact injected into this VPS run |
| infrastructure-failure materialization | NOT RUN live | Unit coverage PASS; no forced simulator crash injected |
| controlled coordinator restart | NOT RUN live | Watchdog remained stable (`restart_history=[]`); no forced restart was injected |
| full generation | NOT RUN | Bounded run was stopped with two slow claimed jobs remaining; no production prefix was touched |

The bounded run was stopped safely after four verified publications. Its queue
and receipts remain under `/home/huy2325/icgs-vps-run-6341777` for inspection;
the temporary credential file was removed. The acceptance result is evidence
for the hardening paths and isolated publication prefix, not a PASS for the
complete production generation plan.

### VPS environment limits

The focused generation suite passed, but the full repository suite was **FAIL
at collection** because the VPS environment has no `torch` (20 test modules
could not import it). `python3 -B scripts/validate_fast.py` ran the 22 harness
tests but was **FAIL** on five historical `vv19-validation` log links absent
from the clean checkout; these are documentation evidence files, not runtime
code failures. L1/L2 model execution and L4 benchmark remain NOT RUN.
