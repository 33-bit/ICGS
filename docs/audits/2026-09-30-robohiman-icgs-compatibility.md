# RoboHiMan / HiMan-Bench → ICGS compatibility audit

Audit target: `chenyt31/RoboHiMan`, checkout `33f71d30d9a100a83725ff00dfcc27374d5ce810` (`Initial clean commit`, 2025-10-16). The repository was inspected from source, not only README. A bounded simulator smoke test was run on `vps-a`; no RoboHiMan archive was downloaded and no source in the cloned repository was edited.

## Executive verdict

| Phase | Verdict | Short reason |
|---|---|---|
| A0 geometry | **PASS (with adapter)** | Live and saved observations provide metric-recoverable depth, intrinsics, extrinsics, and world-frame XYZ. The published collection scripts disable point-cloud and masks, but this is reversible for recollection and depth reconstruction is supported. |
| A1 temporal physical representation + initial dynamics | **PARTIAL** | Ordered achieved state is rich, but saved demos contain no commanded action, timestamps, cadence, action duration, or failed trajectories. Consecutive achieved poses are a useful surrogate target, not commanded dynamics. |
| B task/event tracker | **PARTIAL** | Task code contains machine-readable handles and predicates, but episodes save neither predicate traces nor event IDs. A task-specific monitor can label fresh execution; automatic offline labeling from the saved pickle alone is not reliable. |

Branch/continuation collection feasibility: **PARTIAL**. The environment can collect fresh branches, but no exact intermediate snapshot is saved. `Task.get_state()` exposes a configuration-tree byte blob, not proven complete physics/controller/RNG state.

Recommendation: **USE WITH ADAPTER**. More specifically, use existing episodes for A0 and A1 warm-up/surrogate behavior; use fresh RoboHiMan collection with an ICGS recorder for commanded dynamics, event labels, failures, and branch anchors.

Final choice: **B. RoboHiMan dùng tốt cho A0/A1 nhưng B cần custom annotation/data**, with the qualification that high-confidence commanded A1 dynamics also needs a collection adapter.

## Evidence and commands

Primary source checkout: `/tmp/RoboHiMan` at the commit above. Important commands:

```text
git -C /tmp/RoboHiMan rev-parse HEAD
nl -ba HiMan-Bench/robot-colosseum/collect_dataset_train_*.sh
nl -ba HiMan-Bench/robot-colosseum/colosseum/rlbench/utils.py
nl -ba baselines/RVT-Baseline/rvt/libs/RLBench/rlbench/backend/scene.py
nl -ba baselines/RVT-Baseline/rvt/libs/PyRep/pyrep/objects/vision_sensor.py
```

The collection scripts explicitly set RGB/depth true, masks and point clouds false, 256×256, and use A/AP/C/CP episode counts (below). `utils.py:109-212` maps those flags to `ObservationConfigExt`; `utils.py:325-364` writes RGB-encoded depth and `utils.py:411-434` nulls image/point-cloud fields before pickling. `scene.py:198-222` converts normalized depth with `near + depth*(far-near)` before point-cloud generation. `vision_sensor.py:128-175` captures depth and produces world-frame XYZ; `vision_sensor.py:177-191` derives intrinsics. `scene.py:533-549` stores camera intrinsics, extrinsics, near and far clipping values in `Observation.misc`.

Bounded smoke test on `vps-a` (CoppeliaSim/PyRep/RLBench installed; RoboHiMan task TTM files and textures provisioned for the test):

| task | result | observed evidence |
|---|---|---|
| `open_drawer` | PASS | 182 ordered frames; front depth `(128,128)`, normalized range `0.2206846..0.9242227`; point cloud `(128,128,3)` float64, range approximately `[-2.51,1.49]`; EE pose `(7,)`; joints and velocities present; `misc` contained front K/T/near/far; descriptions included `open drawer` and `grasp/pull` decomposition. |
| `put_in_and_close` | PASS | 518 ordered frames; same calibration/depth/XYZ availability; `task_low_dim_state` `(213,)`; descriptions included `open → put → close`. |

The smoke test exercised live observations rather than a downloaded public episode. A follow-up serializer prototype reached the save/reload stage but failed only while JSON-serializing a NumPy boolean; therefore saved-file numeric discrepancy output is **NOT RUN**, not silently counted as PASS. The earlier live task smoke is the evidence for simulator observation availability.

## Compatibility matrix

| Requirement | Required by ICGS | RoboHiMan support | Evidence | Adapter | Severity |
|---|---|---|---|---|---|
| Metric depth | A0 | Yes, normalized depth plus clipping planes; live smoke confirms numeric depth | `scene.py:198-220`; `const.py:29`; `utils.py:325-364` | RGB24 decode, then near/far conversion | MEDIUM |
| Camera K/T | A0 | Yes, per-frame `misc` | `scene.py:533-549` | copy and validate convention | NONE |
| World XYZ | A0 | Live field exists; collection disables serialized field | `vision_sensor.py:137-175`; train scripts lines 33-36 | reconstruct from depth or enable point cloud | LOW |
| Foreground mask | A0 | Optional but disabled; saved mask encoding is lossy/uint8 | `utils.py:123-131`, `366-409`; `scene.py:194-196` | prefer task/object handle monitor or recollect masks | HIGH |
| Invalid-depth mask | A0 | No explicit valid mask | depth API and clipping range only | finite/range checks; retain validity provenance | MEDIUM |
| EE pose/gripper | A1/IP | Yes, achieved pose XYZW and binary gripper | `scene.py:292-297`; `observation.py:8-70` | pose transform and grip convention | NONE |
| Joints/velocities | A1 | Yes | `observation.py:8-70` | normalize and align | LOW |
| Commanded action | A1 | **No saved field** | generator uses `MoveArmThenGripper(JointVelocity(), Discrete())` at `dataset_generator_atomic.py:140-144` and saves only `Demo` | instrument `task.step`/action mode | BLOCKER for commanded dynamics |
| Timestamp/cadence/duration | A1 | **No** | saved schema in `utils.py:433-440`; `_DT=0.05` is not persisted | recorder with sim step and command duration | HIGH |
| Success/failure | A1/B | successful demos only; final task success available in runtime | `scene.py:443-447`; generator `dataset_generator_atomic.py:203-228` | collect failures explicitly | HIGH |
| Machine predicates | B | Yes in task source | task files and `extensions/conditions.py:30-242` | task monitor | MEDIUM |
| Predicate traces/event IDs | B | No | no trace field in `Demo`/`save_demo` | write per-frame monitor output | HIGH |
| Exact snapshot | branches | No episode snapshot; runtime helper only | `task.py:338-349`; `demo.py:17-18` | test configuration-tree restore or recollect anchors | HIGH |

## Dataset organization, perturbations, and leakage

The shell scripts define:

| split | tasks | episodes | variation selector | image |
|---|---|---:|---:|---|
| A | 10 atomic | 20/task | index 0 | 256² |
| AP | same 10 atomic | 1/task | -1, enabled perturbation configurations | 256² |
| C | 4 compositional (`put_in_without_close`, `sweep_and_drop`, `take_out_without_close`, `transfer_box`) | 5/task | index 0 | 256² |
| CP | same 4 C tasks | 1/task | -1 | 256² |
| atomic test | same 10 atomic names | 1/task | -1 | 128² |
| compositional test | the 4 C/CP names plus 8 additional names | 1/task | -1 | 128² |

Evidence: `collect_dataset_train_A.sh:6-41`, `train_AP.sh:6-41`, `train_C.sh:6-35`, `train_CP.sh:6-35`, and the two test scripts. `multi_level_data_rearrange.py:18-42` creates symlinked merged task directories; it does not create new execution lineage.

Leakage risk is **HIGH**: exact atomic task names occur in train and test; code, TTM scene, primitives and assets are reused. C/CP reuse the same four task names. AP/CP are enabled variation/perturbation configurations, not evidence of failed attempts. Collection workers call `np.random.seed(None)` (`dataset_generator_atomic.py:138` and `252`), so `env.seed=42` is not sufficient to identify episode lineage; `variation_number.pkl` records task variation, not complete simulator RNG/physics state. Same asset instance and task dependency structure can therefore cross splits. A proper split manifest must key task, primitive, scene/asset, variation configuration, worker/episode seed, and lineage.

## Saved episode format

Expected tree from `save_demo` (`utils.py:220-444`):

```text
task/variation*/episodes/episodeN/
  left_shoulder_rgb/*.png, left_shoulder_depth/*.png
  right_shoulder_rgb/*.png, right_shoulder_depth/*.png
  overhead_rgb/*.png, overhead_depth/*.png
  wrist_rgb/*.png, wrist_depth/*.png
  front_rgb/*.png, front_depth/*.png
  (mask folders only when enabled)
  low_dim_obs.pkl
  variation_number.pkl                 # when generator passes variation
  variation_descriptions.pkl
```

`low_dim_obs.pkl` is a pickled ordered `Demo` of `Observation` objects. Fields from `Observation`/live smoke are camera slots, `joint_positions`, `joint_velocities`, `joint_forces`, `gripper_open`, `gripper_pose`, `gripper_matrix`, `gripper_joint_positions`, touch forces, `task_low_dim_state`, ignore-collision state, and `misc` calibration. There is no action array, timestamp, command duration, success file, or per-frame simulator snapshot. Point-cloud fields are set to `None` by `save_demo` even when generated in memory. No actual RoboHiMan public episode was downloaded during this audit; this tree is source-derived and must be checked against a downloaded archive before bulk conversion.

## A0 findings

**Can geometry be trained without recollecting all observations? Yes, conditionally.** Use front/left/right shoulder depth, decode RGB24 with `DEPTH_SCALE=2**24-1`, convert normalized depth to meters with saved near/far, back-project with saved K, apply saved camera-to-world T, concatenate cameras, finite/range filter, workspace/foreground crop, voxelize and FPS/sample to 2048. This is sufficient for ICGS world-frame XYZ and transform into EE frame. The live smoke directly verified world-frame point-cloud scale and calibration fields.

The default public collection does not save point clouds or masks. Reconstructing XYZ is credible because the exact source path used for point clouds is present. Masks are not equivalent: `rgb_handles_to_mask` multiplies RGB by 255 and converts to a scalar handle, while `save_demo` stores `(mask*255).astype(uint8)`; instance IDs therefore require a numeric validation and should not be assumed lossless. With no foreground mask, use calibrated workspace/task-specific cropping or recollect masks. A0 is PASS for geometry, not a claim that object segmentation is solved.

## A1 findings

**Can a causal transition dataset be made? Partially.** Ordered observations are present because `Demo` is a list (`rlbench/demo.py:4-18`) and scene collection appends after path execution (`scene.py:323-447`). The exact runtime action mode is joint velocity arm plus discrete gripper (`dataset_generator_atomic.py:140-144`, compositional counterpart lines 140-144). However, the saved object only records achieved observations. Existing downstream conversion constructs behavior “actions” from consecutive/absolute gripper poses (`src/utils/env_engine.py:62-89`; `baselines/3d-Diffuser-Actor-Baseline/utils/utils_with_rlbench.py:591-615`), which is a useful surrogate but loses commanded-vs-achieved residuals. Do not label those commands as actual commands.

The ICGS native interface expects world `T_w_e`, XYZW quaternion externally, binary grip (0 closed/1 open), and root-anchored absolute targets derived from consecutive poses (`src/icgs/data/preprocessing/native.py:102-125`; `src/icgs/execution/commands.py:7-29`). Therefore RoboHiMan is conventionally adaptable, but not exactly timed/action-compatible. No simulator tick, action cadence, duration or controller state is persisted. Existing demos are success-only: `scene.py:443-447` rejects final unsuccessful execution and the generator saves after successful return. Natural failed attempts: **NO**. Structured perturbations: **YES**. Failed trajectory outcome: **NO**.

Use existing A/AP/C/CP for state/trajectory warm-up and behavior targets. Recollect (or instrument fresh collection) for commanded `u_t`, achieved `x_{t+1}`, duration, substeps, contact/attachment state, and failed branches.

## B findings and concrete task mappings

The task source is enough to build a deterministic monitor during fresh simulator execution, but not enough to label every saved frame without task-specific code/state access.

1. `open_drawer.py:24-40` creates a named drawer joint and registers `DrawerCondition(joint, 0.15, "open")`; oracle text is `grasp handle → pull drawer open`. A monitor can define `ν=open(drawer)` from joint displacement, `ρ=open` on the first debounced false→true crossing, and `ε=eligible(next)` from the prerequisite that the handle is reachable/graspable. Alignment `α` can be tied to the transition boundary where the drawer condition first becomes true. The exact joint threshold is machine-readable, but event boundary debounce/eligibility policy is an ICGS adapter decision.
2. `put_in_and_close.py:26-44` creates `DetectedCondition(item, success_sensor)` AND `DrawerCondition(joint, 0.03, "close")`; oracle text is `open → put → close`. Per-frame `ν` can contain `{inside_drawer, drawer_closed}`; `ρ_put` is the first proximity true crossing, `ρ_close` the first joint-close crossing; `ε_close` requires `inside_drawer`, while `ε_put` requires an open drawer and object grasp/reach. The final success condition is conjunctive, so a final success scalar alone cannot distinguish order/occurrence.
3. `transfer_box.py:20-52` exposes drawer options, object handle, `DetectedCondition(item, success_sensor)`, and language `open → take out → put in cupboard`; it is suitable for a similar relation monitor.

Generic predicate implementations are in `colosseum/rlbench/extensions/conditions.py:30-242` (`JointCondition`, `DetectedCondition`, `GraspedCondition`, `DetectedSeveralCondition`, and `DrawerCondition`). `Task.success()` only aggregates final conditions (`rlbench/backend/task.py:180-213`, `287-300`); there is no generic event sequence object. `task_low_dim_state` is a flat numeric vector, not a self-describing semantic predicate trace; object ordering must be recovered from the task code.

## Snapshot/replay and continuation

`Task.get_state()` (`baselines/RVT-Baseline/rvt/libs/RLBench/rlbench/backend/task.py:338-349`) returns configuration-tree bytes and object count. `Object.get_configuration_tree()` (`baselines/RVT-Baseline/rvt/libs/PyRep/pyrep/objects/object.py:463-474`) documents object-relative poses/joints/path values. This is useful infrastructure, but no episode saves it; `Demo.restore_state()` only restores NumPy RNG (`rlbench/demo.py:17-18`). Configuration trees do not prove restoration of controller integrators, contacts/attachments, physics solver state, camera randomization or simulator RNG. Therefore branch/continuation collection is PARTIAL: implement and numerically validate an anchor recorder, then compare two restored executions on EE pose, object pose, depth/XYZ and predicates. If exact restore fails, deterministic reset-and-replay can be a fallback only after measured drift bounds.

## Instant Policy compatibility

| Property | RoboHiMan | Instant Policy expected | Compatible? | Adapter |
|---|---|---|---|---|
| Robot/control | Panda, RLBench action mode uses joint-velocity arm + discrete grip | policy input is robot-agnostic; deployment controller consumes EE targets | Adapter required | execute predicted absolute EE pose through a measured IK/controller bridge |
| EE pose | PyRep position + quaternion XYZW | `T_w_e` 4×4; `pose_to_transform` uses `Rot.from_quat(pose[3:])` | Yes | preserve XYZW and meters |
| Grip | scalar 0/1 | 0 closed / 1 open | Yes | no inversion |
| Point cloud | world XYZ available/reconstructable; downstream baselines transform to EE frame | segmented world-frame cloud, then EE-frame preprocessing | Yes with segmentation/depth adapter | use `sample_to_cond_demo`/`sample_to_live` semantics |
| Camera | five fixed named cameras, K/T saved | policy only requires calibrated world XYZ | Yes | choose front + shoulders and validate frame |
| Action | saved dataset has no action; runtime collection arm command is joint velocity | expected deployment output is absolute EE targets + grip | No exact | collect commanded targets or treat pose deltas as surrogate |
| Timing | not saved | ICGS timed contract requires measured duration | No | recorder/controller wrapper |

The official Instant Policy repository’s deployment README defines `demo={'pcds': [], 'T_w_es': [], 'grips': []}`, world-frame segmented clouds, `T_w_e`, and 0/1 grips; `sim_utils.py` uses XYZW conversion and transforms predicted actions to absolute EE poses. This confirms the geometry/gripper convention but not commanded-action or timing equivalence.

## Missing data and minimal adapter

Can derive offline: metric XYZ from saved depth/K/T; EE transforms; grip; finite-depth validity; task/variation IDs; surrogate absolute targets; source-level predicate definitions.

Must re-render/replay or recollect: trustworthy foreground masks; commanded joint/EE action; action duration/cadence/substeps; achieved-vs-commanded pairing; contact/attachment/controller state; failed attempts; predicate traces; exact branch anchors.

Suggested adapter:

```text
icgs_adapters/robohiman/
  dataset_reader.py       # tree + Demo/PNG loader, provenance checks
  camera_adapter.py       # RGB24 decode, near/far, K/T, XYZ validity
  pointcloud.py           # crop/voxel/FPS/2048 and mask policy
  action_adapter.py       # achieved-pose surrogate; explicit commanded schema
  task_monitor.py         # task-specific alpha/rho/nu/epsilon predicates
  replay.py               # snapshot/restore drift tests
  manifest.py              # task/asset/variation/seed/lineage split
```

## Risk

Geometry: **LOW–MEDIUM** (calibration/depth path is evidenced; masks are not). Action conversion: **HIGH** (command absent). Annotation: **HIGH** (task-specific monitor and no trace). Replay: **HIGH** (no saved exact snapshot). Benchmark protocol: **HIGH** (task-name and primitive/asset overlap, uncontrolled worker RNG, successful-only collection).

## Five smallest next steps before any large dataset

1. Download exactly one public A episode and run a reader that verifies PNG decode, K/T, near/far, XYZ scale, and saved-field schema; fix NumPy scalar serialization in the bounded audit script.
2. Enable masks and validate handle-ID preservation against simulator handles; otherwise define a documented workspace/object crop.
3. Instrument one atomic and one compositional collection to save commanded action, achieved state, sim timestep, duration, substeps, controller mode and final outcome.
4. Implement monitors for `open_drawer` and `put_in_and_close`; compare predicate crossings with runtime success and manually inspect a few frames.
5. Prototype `get_state()` restore twice from one anchor and publish numerical drift tolerances before collecting branches/continuations; at the same time create a split manifest keyed by asset, scene, variation, seed and lineage.

## Validation status and provenance

- `python3 -B scripts/validate_fast.py`: **FAIL** at local documentation-link check (`docs/plans/active/physical-dependency-generation.md` has undefined reference `"mirror"`); harness self-tests 22/22 passed; L1/L2/L3/L4 not run by this command. This is an existing unrelated ICGS worktree issue, not changed by this audit.
- RoboHiMan source inspection: **PASS** for the cited files at commit `33f71d3`.
- VPS live smoke (`open_drawer`, `put_in_and_close`): **PASS** for bounded live observation collection (182 and 518 frames respectively); no large dataset generated.
- Public saved-episode download: **NOT RUN** (intentionally avoided large external archive).
- Exact snapshot/replay numerical comparison: **NOT RUN**.
- Full RoboHiMan train/test generation: **NOT RUN**.
- Instant Policy model inference integration: **NOT RUN**.

License/provenance: RoboHiMan `setup.cfg` identifies Colosseum metadata and MIT license for that package; the repository acknowledges RLBench, PyRep, RVT, 3D-Diffuser-Actor, OpenPi, LLaMA-Factory, Colosseum and DeCo. Vendored RLBench/PyRep carry their own license files. Check the asset/dataset-host license before redistributing derived data; the commit and simulator versions (CoppeliaSim 4.1.0/RLBench fork) must be pinned in any release manifest.
