# RoboHiMan backbone validation (2026-09-30)

Smoke-scale evidence for the [migration plan](../../plans/active/robohiman-backbone-migration.md)
under [ADR 0016](../../decisions/0016-robohiman-backbone.md). Not a benchmark.

## Environment

Host `vps-a` (Intel i5-8250U, 8 threads, 15 GB, no GPU, Xvfb). Simulator venv
`~/ICGS/outputs/robohiman/.venv`: Python 3.11.16, numpy 1.26.4, RoboHiMan
`33f71d30`, PyRep `231a1ac6`, RLBench `587a6a0e`, CoppeliaSim 4.1.0 (the
generation host's install). RoboHiMan tree clean except pip-generated
`colosseum.egg-info`. Offline report ran in the ICGS `.venv` (torch CPU).
Gate code revisions: `gates` at `38bf567`–`9e3b1a6`, `gates-v2` at
`a9e0535`–`6f17925`, `gates-v3` at `c508b7c`/`d6b3817`; smoke store at `7266633`.

## Commands

```bash
# on vps-a, cd ~/ICGS/outputs/robohiman && source env.sh ; X='xvfb-run -a -s "-screen 0 1024x768x24"'
$X .venv/bin/python -B ~/ICGS/scripts/robohiman_gates.py a0 --task put_in_without_close --seed 4 --out a0.json
$X ... robohiman_gates.py parity --task open_drawer --seed 1 --out parity-open_drawer.json
$X ... robohiman_gates.py parity --task put_in_without_close --seed 2 --out parity-put_in_without_close.json
$X ... robohiman_gates.py replay --task open_drawer --seed 5 --anchor-waypoint 2 --out replay-open_drawer.json
$X ... robohiman_gates.py replay --task put_in_without_close --seed 3 --anchor-waypoint 3 --out replay-put_in_without_close.json
$X ... robohiman_gates.py replay --task put_in_without_close --seed 4 --anchor-waypoint 2 --out replay-put_in_without_close-wp2.json
$X ... robohiman_gates.py replay --task put_in_and_close --seed 6 --anchor-waypoint 10 --out replay-put_in_and_close.json
$X ... robohiman_gates.py dependency --task put_in_without_close --seed 7 --anchor-waypoint 3 --trials 3 --deltas 0.0 -0.05 -0.08 -0.11
$X ... robohiman_gates.py dependency ... --deltas -0.085 -0.09   # dependency-fine-a
$X ... robohiman_gates.py dependency ... --deltas -0.095 -0.1    # dependency-fine-b
$X ... robohiman_gates.py dependency --task put_in_and_close --seed 8 --anchor-waypoint 3 --trials 2 --deltas 0.0 -0.06 -0.09
$X ... robohiman_collect.py ...    # 9 episodes, see collection-summary.jsonl
$X ... robohiman_gates.py canonical --task open_drawer --store store-smoke-7266633 --seed 1
PYTHONPATH=src .venv/bin/python -B scripts/robohiman_stage1_report.py --store outputs/robohiman/store-smoke-7266633 \
    --out stage1-report-smoke.json --frames-dir review-frames      # ICGS venv
python -B scripts/robohiman_split_audit.py --robohiman <RoboHiMan@33f71d3> --out split-audit.json
```

## Results

| Gate | Result | Key numbers (files in this directory) |
| --- | --- | --- |
| A0 geometry | PASS | numpy back-projection = PyRep (max 0.0 m), vs live cloud ≤3.4e-7 m; 100% of task-object mask pixels inside the simulator world box ±1 cm on 4 cameras (Wall3 98.9%); masks exact integer handles; reused 5 mm physical preprocessing produced 2048/128/32 tensors on every episode (`a0-*`, `stage1-report-smoke`) |
| A1 command/achieved/timing | PASS (with noted uncertainty) | sim dt = 0.05 s on every step; no row where command equals achieved next joints; tracking lag mean ≈0.007 rad, p95 ≈0.03 rad; canonical EE commands via simulator chain with 0.9–3.7 mm chain residual; commanded-vs-achieved-next EE gap mean 2.4–4.2 mm, p95 ≤1 cm (`canonical-smoke`) |
| B labels | PASS on 2 tasks (manual check on 2 episodes) | open_drawer + put_in_without_close + put_in_and_close ordered events; frame sheets show open@167, grasp@340, placed@393, closed@578; failure episode keeps rho(grasped)=1 with nu=0 (`*-sheet.png`) |
| Failure retention | PASS | early-grasp timing → `failure`; short pull → `invalid_execution` (no path); both stored with labels and perturbation provenance; object displacement → success |
| Mirror parity | PASS (distributional, not bitwise) | upstream-vs-upstream final object 0.67–0.78 mm, upstream-vs-mirror 0.87–1.7 mm, all 8 open_drawer runs succeed; upstream itself diverges at boundary 0–1 |
| Snapshot / replay | PARTIAL | see below |
| Dependency | PARTIAL | narrow consequential band near the drawer-open threshold, n≤3 |
| Split overlap | Diagnostic delivered | see below |

### Snapshot/replay (declared threshold 5 mm tip/object/articulation, identical predicates/outcome)

| Strategy | open_drawer (anchor 157/197) | put_in_without_close (204/429) | put_in_and_close (417/603) |
| --- | --- | --- | --- |
| fresh reset + open-loop command replay | ≤0.12 mm, predicates identical ✔ | reset placement failed for same lineage | drawer 61 mm, item 17 mm, predicates 3–5% differ, outcome equal ✘ |
| post-reset snapshot + open-loop replay | ≤0.74 mm tip, ~0 objects ✔ | tip 0.84 mm, drawer 14 mm, item 20 mm, predicates identical ✘ | drawer 61 mm ✘ |
| anchor config-tree restore + open-loop replay | grip lost: drawer 0.22 m, outcome flips ✘ | tip 0.9 mm, item 12–27 mm, predicates identical ✘ | tip 0.8 mm, objects ≤3.2 mm, drawer ≤4.7 mm, predicates 0.18% ✘ (near) |
| restore repeatability (restore twice) | 1.1 mm | 15 mm | 1.3 mm |

No strategy is mm-exact for long contact-rich histories; configuration-tree
restore is unusable at anchors with physical grip (handle), mm-level at
non-contact anchors. Outcome agreement holds for replay-based strategies.

### Dependency audit (put_in_* tasks, decision = pull extent at waypoint 3)

| Drawer opening after decision | Local `drawer_open` (0.15 m) | Downstream success |
| --- | --- | --- |
| 0.24 / 0.19 / 0.16 m | yes | 3/3 each (put_in_without_close); 2/2 (put_in_and_close 0.24/0.18) |
| 0.157 m | yes | 3/3 |
| 0.151–0.155 m | yes | 2/3 (put_in_without_close), 1/2 (put_in_and_close) |
| 0.146 m | no | 2/3 |
| 0.13–0.14 m | no | 0/3 (`invalid_execution`: no path to place waypoint) |

Continuation policy is the scripted-expert mirror with 5 mm waypoint jitter,
not pi_ref. The consequence mechanism is place-waypoint reachability.

### Split overlap (`split-audit.json`)

- L4 train (A∪AP∪C∪CP) → atomic test: task, primitive, scene, asset and
  perturbation-family overlap all 100%.
- L4 train → compositional test: 4/12 tasks, 15/19 primitives, 4/12 scenes,
  8/11 named shapes, 12/12 perturbation families seen; 8 test tasks are unseen
  as whole tasks.
- 13 task configs leave an `object_size` factor (misnamed `recv_obj_color`)
  enabled in the `no_variations` strategy, so native A/C data for drawer and
  transfer tasks randomizes drawer/cupboard scale 0.9–1.15.
- Native workers call `np.random.seed(None)`; episode placement lineage is not
  recoverable from native data.

## Not run

Bulk collection, pi_ref, Stage-2 continuations, training, public RoboHiMan
archive download, native-vs-instrumented byte comparison of saved demos.
