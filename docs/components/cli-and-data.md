# CLI, configuration and inference data

Only runtime package icgs is installed. CLI helpers import heavy libraries only
after parsing; --help performs no model/simulator/training work.

| Old path | New command / mapping |
| --- | --- |
| ip/eval.py | icgs evaluate --config configs/baselines/instant_policy_published_vv19.json --checkpoint /absolute/model.pt |
| ip/train.py | icgs train --config /absolute/experiment.json --data-train /absolute/train --data-val /absolute/val |
| fine_tune + model_path/model_name | train --checkpoint /absolute/model.pt; resolved config explicitly supplies architecture |
| task_name / restrict_rot / num_rollouts | config.evaluation fields; --num-rollouts explicit override |
| num_demos | config.graph.num_demos or --num-demos |
| record/save_path/run_name | config.training.record/save_dir and --run-name; no hidden output directory |
| ip/prepare_data.py example | icgs prepare-data --config /absolute/config.json --input /absolute/raw.npz --output /absolute/data |
| deployment placeholder | icgs infer for recorded input, or public policy API for a separately implemented robot adapter |

JSON fields: schema_version=1, optional profile=train/fine_tune/eval/deploy/published,
config with known section overrides. Runtime defaults/profile < explicit file <
explicit CLI overrides. Relative scene checkpoint/save_dir values resolve against
the config file directory, not process CWD. CLI checkpoints/input/output are explicit
paths; absolute paths are recommended and used by acceptance. Root presets contain
no duplicate defaults. Installed package never searches for root configs.

Published evaluation rejects configuration differing from the hash-bound model
profile, including numerical runtime flags; device/demo count/steps are supported
inference choices. Unsupported architecture overrides are not ignored. Historical
source training profile is distinct from published deployment.

## Safe NPZ schema v1

No pickle/object arrays; exact fields:

| Field | Shape / meaning |
| --- | --- |
| schema_version | scalar integer1 |
| demo_points | [D,T,N,3] segmented world XYZ |
| demo_poses | [D,T,4,4] end-effector-to-world transforms |
| demo_grips | [D,T], observed0/1 |
| points | [N_live,3] segmented live world XYZ |
| root_pose | exactly[4,4], live end-effector-to-world |
| grip | scalar observed0/1 |

Dense point count is an interchange format, not a forced model feature dimension.
Native preprocessing selects 2048 local points per waypoint. Demo count/waypoints,
finite values and valid homogeneous/rotation matrices are checked. Published target
uses two demos by default, ten selected waypoints and prediction horizon8.

infer output: actions[B,P,4,4], grips[B,P,1], root_pose, absolute_targets=root_pose
@actions, schema_version and JSON metadata scalar. Grips are normalized commands,
not achieved states. Metadata names artifact hash, seed/settings/time/frame semantics.
Input fixture is reproducible geometric synthetic data with two demos and nontrivial
poses; it is not an RLBench success example.

prepare-data uses the same safe container with D conditioning demos followed by one
live trajectory, preserving the old native PyG fields/target padding. Cache embeddings
is opt-in and requires a real configured encoder artifact. Existing dataset loader
retry and augmentation behavior is unchanged; newer Torch weights_only defaults are
a separately documented compatibility risk. No fake collector is supplied.
