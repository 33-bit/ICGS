# Current policy data and geometry contract

Observed at original source `65dc94e` and structurally preserved by modularization.
Source owns detailed implementation; a later semantic change
must update this current contract and identify its effect on the
[original baseline](../baselines/instant_policy.md).

## Raw observations and frames

[Data processing](../../src/icgs/data/preprocessing/native.py) expects demos with lists
`pcds`, `T_w_es`, `grips`. Each cloud is segmented world-frame XYZ;
`T_w_e` is a 4×4 end-effector-to-world transform. Gripper inputs are 0 closed /
1 open. Clouds are transformed by `inverse(T_w_e)` into the corresponding
end-effector frame. Translation units are consistent with meter-scale source
limits (0.01 described as 1 cm); rotations used in six-dimensional actions are
axis-angle radians. Trajectory spacing APIs take rotation thresholds in degrees.

Outlier filtering uses Open3D statistical filtering (20 neighbors, std ratio 2).
Random sampling selects 2048 points, with replacement only when fewer survive.
Empty clouds are not a supported fixture. Conditioning waypoint selection retains
endpoints/state changes and distributes extra waypoints heuristically; callers
cannot assume every arbitrary trajectory yields exactly the requested count.

## Serialized PyG sample

B=batch, D=conditioning demos, T=demo waypoints, P=prediction horizon, S=scene
nodes, E=scene embedding width. Original defaults: D=2, T=10, P=8, S=16, E=512.

| Attribute | Semantics / shape |
| --- | --- |
| pos_demos | Concatenated demo cloud points, [D*T*2048, 3] per uncached sample |
| batch_demos | Cloud membership per point; PyG batching shifts batch-like indices |
| graps_demos | Existing spelling; normalized gripper states [B,D,T,1] |
| demo_T_w_es | Original demo world poses [B,D,T,4,4] |
| pos_obs | Current local cloud [2048,3] per sample; retained with cached features |
| batch_pos_obs | Current-cloud membership per point |
| current_grip | Normalized current gripper [B] |
| actions | Relative targets [B,P,4,4] |
| actions_grip | Normalized target gripper [B,P] |
| T_w_e | Current world pose [B,4,4] |
| demo_scene_node_embds / pos | Optional [B,D,T,S,E] / [B,D,T,S,3] |
| live_scene_node_embds / pos | Optional [B,S,E] / [B,S,3] |
| action_scene_node_embds / pos | Computed by AGI [B,P,S,E] / [B,P,S,3], not saved by preparation |
| diff_time | Added during training/inference [B,1] |

save_sample saves `data_<offset+i>.pt` or returns the first live sample when no
save directory is supplied. Cached demo features replace pos_demos/batch_demos
with None; live points remain because action-conditioned scene encoding needs them.
The encoder identity and preprocessing are therefore cache compatibility inputs.

RunningDataset's default check requires only actions and actions_grip, not every
model-required field. It retries another random index on any load/validation error.
Directory entries must correspond to the contiguous indexed sample convention used
by training; non-data entries can inflate sample count. L1 must not exercise this
unbounded retry loop against unknown data.

## Action targets and normalization

For current timestep i and available future i+j:

```text
action[j-1] = inverse(T_w_e[i]) @ T_w_e[i+j]
world_target[j-1] = T_w_e[i] @ action[j-1]
```

Targets beyond the trajectory end are identity transforms and the last gripper
state; there is no explicit padding mask. A chunk's actions share the observation
reference pose; do not chain them as incremental transforms.

[Normalizer and action codec](../../src/icgs/algorithms/diffusion/codec.py) scale each six-dimensional action bound
by horizon index 1..P. Translation base bounds are ±0.01; rotation components
±π/60. Linear mappings send these bounds to [-1,1]. Labels are not six-dimensional
pose vectors: translation correction is repeated across gripper nodes; rotation
is a gripper-node displacement. Label translation bounds are twice action bounds;
rotation displacement bounds use gripper length 0.06 and
`2 * sqrt(2*l*l*(1-cos(max_angle)))` in each coordinate.

AGI's label transform is `inverse(noisy_action) @ target_action`. Rotation
displacement is computed after zeroing its translation. Labels concatenate
translation, rotational displacement and target grip, giving [B,P,6,7].
GraphDiffusion normalizes the first six label channels and uses L1 loss.

Current implementation owners are GraphDenoiser, OriginalActionCodec and
OriginalDiffusionObjective under src/icgs; historical ip imports are intentionally
removed. Policy returns ActionTrajectory and operates on cloned sampling
batches, preserving internal numerical operations without mutating caller data.

Predicted gripper states use sign; exactly zero is possible and the rollout's
threshold maps it to closed. Pose utilities use XYZ + quaternion xyzw externally;
an internal quaternion-to-axis-angle utility uses scalar-first quaternions.
Do not conflate those representations.

## Scene and graph semantics

SceneEncoder samples at ratios 0.125 then 0.0625 with FPS and nearest assignment.
For the original 2048-point clouds this targets 16 nodes with width 512.
Its two positional scales are 1/0.05 and 1/0.2; model construction hardcodes ten
scene-encoder frequencies even though GraphRep reads local_num_freq.

GraphRep has scene and gripper node types. Six gripper geometry positions are:
(0,0,0), (0,0,-0.06), (0,0.06,0), (0,-0.06,0), (0,0.06,0.06),
(0,-0.06,0.06). Nodes cover demo waypoints, one current observation, and P noisy
action slots. All generated edge masks require equal sample batch.

| Edge group | Actual direction/membership |
| --- | --- |
| rel | Dense scene→scene, scene→gripper, gripper→gripper inside matching time/demo |
| cond | All demo gripper times → current gripper |
| demo | Same demonstration, later waypoint → immediately previous waypoint |
| time_action | Action gripper slots → different action slots |
| rel_cond | Current gripper → action gripper slots |
| rel_demo / rel_action | Split local scene relations into non-action / action subsets |

Read initialise_graph assignments, not just the stale edge_types list: it lists
demo_action but does not construct that relation. There is no observation-history
encoder. Demonstration temporal edges are not live temporal context.

Local relations use position differences. Temporal gripper relations use relative
transforms between participating frames; cond uses a learned embedding. Position
encoding has 63 channels at ten frequencies; edge attributes concatenate two such
encodings (126). With pos_in_nodes, positions are appended to node features.
AGI consumes the graph in local → conditioning → action transformer stages.

## Preserve implementation distinctions

AGI constructs action-conditioned clouds with `R.T @ points - t`, as written,
not the general inverse rigid transform `R.T @ (points - t)`. Do not “correct”
the formula during harness work or describe it as already mathematically equivalent.

GraphRep's diffusion-time assignment uses chained advanced indexing. Whether the
intended update reaches the original embedding needs a focused behavioral test;
the code/comment alone is not proof. The harness does not repair it.

GraphDiffusion calls scheduler.set_timesteps but loops over integer k in reverse,
uses k in scheduler.step, and gives the first model call num_diffusion_iters_train.
Its node-displacement/SVD composition and clipping are the observed sampler; do not
replace this with a generic DDIM description or conventional indexing silently.
These subtleties are targets for the later characterization/refactoring task.

## Published profile and method boundary

Published vv19 live preprocessing includes voxel size0.01 and initial noise precedes
scene encoding. The historical source cache path is not equivalent under the same
seed. Published profile selects those observed behaviors, with strict hash-bound
embedded encoder weights and no auxiliary checkpoint (ADR0005).

Contexts own read-only byte-backed demo arrays and immutable mappings; mutable
features are isolated via branch_copy. Candidate K differs from batch B/horizon P;
prefix targets use one root pose, with unknown timing left explicitly unknown.

## Separate target ICGS contracts

The [planned method contracts](../method/contracts.md) add timed commands, physical/
task memory and executed episode records without changing this native contract.
Proposal T=8 maps to native prediction horizon P, not native demo-waypoint count10.
Added physical preprocessing is separate; planned neural/action/evaluator examples
are not installed APIs. See [target decision](../decisions/0006-icgs-target-boundaries.md).
