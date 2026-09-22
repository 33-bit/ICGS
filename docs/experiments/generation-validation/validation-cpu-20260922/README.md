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
