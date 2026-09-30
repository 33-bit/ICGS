# RoboHiMan backbone: instrumentation, Stage-1 records and gates

Status: migration in progress under [ADR 0016](../decisions/0016-robohiman-backbone.md)
and the [migration plan](../plans/active/robohiman-backbone-migration.md). Gate
evidence lives in [robohiman-validation](../experiments/robohiman-validation/README.md).
The custom generator in [generation](generation.md) is legacy/reference and is
not scaled further.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `icgs.data.stage1.schema` | `icgs_stage1_episode_v1` manifest/array contract; perturbation vs outcome vocabularies |
| `icgs.data.stage1.store` | Atomic episode directory (`manifest.json` + `arrays.npz`), hash-verified reads, no pickle |
| `icgs.data.stage1.labels` | alpha/rho/nu/epsilon from raw predicate traces |
| `icgs.data.stage1.views` | D_geom / D_dyn / D_task indexes; refuses D_value/D_pair/D_terminal without pi_ref |
| `icgs.environments.robohiman.pins` | Upstream pins and native A/AP/C/CP/test split table |
| `…robohiman.session` | Builds the environment with the upstream generator's own calls |
| `…robohiman.recorder` | Per-physics-step command/state capture through instance wrappers |
| `…robohiman.expert` | Mirror of `Scene.get_demo` with anchors, perturbations, failure retention |
| `…robohiman.monitors` | Upstream-condition predicates and reviewed event specs per task |
| `…robohiman.snapshot` | Config-tree snapshot/restore candidate and open-loop command replay |
| `…robohiman.camera` / `kinematics` | Offline depth→world XYZ; Panda FK for canonical EE commands |
| `scripts/robohiman_collect.py` | Bounded smoke collection (refuses >20 episodes per call) |
| `scripts/robohiman_gates.py` | parity / replay / a0 / dependency simulator gates |
| `scripts/robohiman_stage1_report.py` | Offline views + A0/A1/B report over a store |
| `scripts/robohiman_split_audit.py` | Static train/test factor-overlap audit |

## Simulator environment (isolated from the ICGS venv)

RoboHiMan's checkout vendors only overlay files; they equal RVT's submodule pins
except one `print`. Rebuild on a Linux host that already has CoppeliaSim 4.1.0
(for example the generation host's `outputs/CoppeliaSim`):

```bash
mkdir -p outputs/robohiman && cd outputs/robohiman
git clone https://github.com/chenyt31/RoboHiMan.git && git -C RoboHiMan checkout 33f71d30d9a100a83725ff00dfcc27374d5ce810
git clone https://github.com/stepjam/PyRep.git && git -C PyRep checkout 231a1ac6b0a179cff53c1d403d379260b9f05f2f
git clone https://github.com/buttomnutstoast/RLBench.git && git -C RLBench checkout 587a6a0e6dc8cd36612a208724eb275fe8cb4470
export COPPELIASIM_ROOT=$PWD/../CoppeliaSim LD_LIBRARY_PATH=$PWD/../CoppeliaSim QT_QPA_PLATFORM_PLUGIN_PATH=$PWD/../CoppeliaSim
uv venv --python 3.11 .venv && P=$PWD/.venv/bin/python
uv pip install --python $P wheel setuptools==69.0.3 numpy==1.26.4 cffi pillow pyquaternion natsort \
  html-testRunner "hydra-core>=1.3.2" omegaconf scipy
uv pip install --python $P --no-build-isolation -e PyRep
uv pip install --python $P -e RLBench -e RoboHiMan/HiMan-Bench/robot-colosseum
uv pip install --python $P --no-deps -e ../..   # ICGS itself, numpy-only code paths
export PYTHONPATH=$PWD/../../src
```

`cffi` is unpinned (upstream `cffi==1.14.2` does not build on 3.11); it is a
build dependency of the simulator binding only. Every episode manifest records
the resolved checkouts, dirty paths, task TTM/py hashes, Python and platform.

## Commands (smoke scale)

```bash
X='xvfb-run -a -s "-screen 0 1024x768x24"'
$X .venv/bin/python -B scripts/robohiman_collect.py --store STORE --run-id RUN \
    --task open_drawer --variation 0 --episodes 1 --seed 1
$X .venv/bin/python -B scripts/robohiman_gates.py parity --task open_drawer --seed 1 --out parity.json
$X .venv/bin/python -B scripts/robohiman_gates.py replay --task put_in_without_close --anchor-waypoint 3 --seed 3 --out replay.json
$X .venv/bin/python -B scripts/robohiman_gates.py a0 --task put_in_without_close --seed 4 --out a0.json
$X .venv/bin/python -B scripts/robohiman_gates.py dependency --task put_in_without_close --anchor-waypoint 3 --out dep.json
python -B scripts/robohiman_stage1_report.py --store STORE --out report.json --frames-dir frames/
python -B scripts/robohiman_split_audit.py --robohiman outputs/robohiman/RoboHiMan --out split.json
```

Execution perturbations are JSON objects passed with `--perturbation`:
`{"family":"execution_pose_offset","waypoint":3,"delta_m":[0,-0.1,0]}`,
`{"family":"gripper_timing","waypoint":2,"close_early_at_distance_m":0.03}`
(or `"delay_steps":k`), `{"family":"object_displacement","object":"item","waypoint":6,"delta_m":[0.03,0,0]}`.
Benchmark perturbations are selected with `--strategy` (Colosseum strategy index;
0 = A/C, others = AP/CP factors).

## Record semantics

- `step_*` rows are measured boundaries (S+1); `cmd_*` rows (S) are what was
  written to the controller during that step: arm joint targets (Reflexxes path
  point, PD loop), arm joint velocities (policy action mode), finger joint
  target velocities, and kinematic grasp/release events. Canonical EE targets
  are derived offline by fitted Panda FK only when its residual passes.
- `frame_*`/`cam_*` rows are rendered at every boundary by default (upstream
  skips gripper-actuation steps). Depth is stored normalized with per-frame
  near/far, K and camera-to-world T; `camera.depth_to_world` reproduces PyRep.
- `perturbation` lists enabled Colosseum factors (with factor RNG states in
  `lineage`) and applied execution perturbations; `outcome.status` is measured:
  `success | failure | timeout | invalid_execution`; simulator errors become
  `attempts/*.json`. Recoverability stays `unknown` without executed evidence.
- `model_boundary` lists online fields; predicates, object poses, labels and
  task identity are offline-only.

## Known upstream behaviour recorded, not hidden

- Grasping is a kinematic attach (`Gripper.grasp` re-parents the object).
- `path.visualize()` teleports the arm along the planned path and back before
  each waypoint; those rows carry `cmd_arm_teleport_calls > 0`.
- Robot reset leaves up to ~3e-4 rad joint residuals; RRTConnect plans are then
  different, so re-executing an episode from the same RNG state is not bitwise
  reproducible, and placement can even fail for a previously valid state.
- Colosseum factors own RNGs that advance every reset; reset lineage is numpy
  state plus every factor bit-generator state (`session.lineage_state`).
- 13 task configs leave an `object_size` factor (named `recv_obj_color`)
  enabled in the `no_variations` strategy; those A/C episodes correctly carry
  `perturbation.families = ["benchmark_factor"]`.
- Configuration-tree restore loses physical grip at contact anchors. Branch
  trials start from a post-reset snapshot (contact-free) and replay the
  recorded prefix open loop; long-horizon replay drifts by centimetres in
  articulated/free objects while predicates and outcome agree.

## Round-2 additions

- Monitors: generic predicate kinds + `TASK_SPECS` for 10 tasks in three
  families; event relations may conjoin raw predicates; labels can be
  re-derived offline (`robohiman_stage1_report.py --relabel`).
- `--canonical-start` (collection) restores the post-reset snapshot before
  execution; required for episodes that will be branched or replayed.
- Anchor reproduction: direct restore for free and kinematic-grasp anchors,
  nearest contact-free snapshot + open-loop replay for physical-grip anchors;
  accept an anchor only after `robohiman_fidelity.py` repeats pass
  `robohiman_fidelity_summary.py`.
- Storage: low-dim state and commands every physics step; RGB-D every second
  step plus every event boundary (default `--frame-stride 2`).
- Dependency audit: `robohiman_gates.py dependency --offset=dx,dy,dz` (waypoint
  frame) or `--deltas` (approach axis), 20 trials, Wilson + Fisher summary.
