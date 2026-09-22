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

## Python Data Collection and Episode Archive API

The [ADR0011](../decisions/0011-episode-archive-and-initial-views.md) initial-data
slice exposes a bounded Python API (`icgs.data.collection`), not an `icgs collect`
CLI. It stores supplied, physically executed attempts; it does not supply an
RLBench scene generator, scripted expert, calibrated controller, or seed demos.
Synthetic integration tests do not establish physical reproducibility.

### Bounded Collection Runner

The job owner supplies the names used below from a provisioned simulator binding
and explicit run configuration. This is an API wiring example, not a standalone
physical generation command; its example caps do not authorize a workload.

```python
from icgs.data.collection import CollectionLimits, run_collection

# Explicit finite resource limits
limits = CollectionLimits(
    max_attempts=10,
    max_intervals=256,
    max_wall_time_s=60.0,
    max_disk_bytes=100 * 1024 * 1024,
)

# Run bounded collection composing injected environment factory
report = run_collection(
    specs=attempt_specs,
    environment_factory=my_env_factory,
    dataset_root="/absolute/path/to/dataset",
    dataset_manifest_path="/absolute/path/to/dataset/dataset_manifest.json",
    limits=limits,
    config=resolved_method_config,
    protocol_manifest=protocol_manifest,
    code_revision=actual_code_revision,
    is_dirty=actual_is_dirty,
    generator_version=generator_version,
    binding_manifest_path="/absolute/path/to/artifacts/generation/manifests/program_bindings.json",
    required_program_ids=("T06", "T08", "T09", "T11", "T13", "T14"),
    dirty_patch_digest=actual_dirty_patch_sha256,
    online_provider=online_provider,
)
```

Each `AttemptSpec` needs explicit `episode_id`, `program_id`, `source_lineage_id`,
`asset_family_id`, `split`, `generator_seed`, `reset_seed`, `action_seed`,
`calibration_id`, `observation_origin="measured"`, `raw_commands_id`, and
`materialized_commands_id`. Supply its `commands` sequence or pass a
`command_factory(spec, environment)` to `run_collection`. Seeds may be equal only
by explicit choice; the command/environment providers must actually use them.
The runner passes `reset_seed` to reset; recording a generator/action seed does
not prove that an external factory honored it.

`protocol_manifest` contains nonempty `environment_protocol_id`,
`controller_protocol_id`, `sensor_protocol_id`, `asset_protocol_id`,
`split_protocol_id`, `collection_protocol_id`, and a nonempty finite-JSON
`metadata` mapping. These identifiers are caller-supplied provenance, not measured
G1/G2 certifications. Dirty source requires a SHA256 `dirty_patch_digest`; use
`None` for a clean source tree. Do not substitute a literal `HEAD` for code identity.

`binding_manifest_path` validates scene/asset/version, workspace and randomization
ranges, seed IDs, expert/waypoint/controller/predicate protocols, calibration and
split/lineage closure before an environment is constructed. `required_program_ids`
limits preflight to the bounded pilot or full train set; missing or incomplete
bindings fail closed.

The environment must implement the P01 timed reset/advance/close capability.
`online_provider` receives the initial `TimedObservation`, then each
`ExecutedTransition`, and returns a mapping of `points`, `point_valid`, `T_w_e`,
and `grip` for that boundary. Alternatively the environment supplies
`get_online_observation()`. Collected online arrays are validated and detached;
measured timed observations remain separately archived. Missing online data is
not reconstructed silently during persistence.

Operational properties and limitations:

- Run one writer per dataset root. Limits are checked between operations;
  nonpreemptible backend calls can overrun the wall deadline.
- Programs, source ancestry and asset-family splits are validated before running.
  Physical task failures are retained as data, not confused with corrupted records.
- Infrastructure/factory/cleanup errors stop the run with accounted work and
  exactly-once environment close. Inspect the returned status and attempt records.
- Disk allowances cover actual serialized shard/JSON bytes and temporary index
  coexistence. They measure file bytes, not filesystem allocation blocks or memory.
- Operational diagnostics use one `quarantine/<attempt_id>/error_report.json`.
  If this cannot fit, `quarantine_unwritten` preserves unsaved diagnostics/evidence
  in the returned report; the caller must retain that report. No on-disk quarantine
  is claimed in that case.
- `read_episode_archive` validates checksums, keys, schema, shapes/dtypes and
  boundary continuity. `read_archive_metadata` exposes status/annotations separately
  without loading the heavy arrays; it is not a substitute for full validation.

### Episode Views and Consumer Adapters

The explicit dataset manifest has `manifest_version: 1`, an `episodes` list
(`episode_id`, root-relative `manifest_path`, `sha256`), a `lineage` list
(`lineage_id`, `parent_ids`, `split`), and an `asset_families` list
(`asset_family_id`, `split`). Start with an empty episode list and real approved
lineage/family assignments. Published episode entries are added by the runner.
No filename inference or late random split is performed.

The validated manifest provides `geom` and `dyn` views; later task/outcome views
are not implemented by this slice. Current view construction loads episode data
into memory; do not describe it as a streaming large-dataset loader.

```python
from icgs.data.datasets.episodes import adapter_a0, adapter_a1, build_view

# Initial geometry view (each observed boundary once)
geom_view = build_view(manifest_path, "geom", split="train", config=config)
a0_batch = adapter_a0(geom_view[0])

# Initial dynamics view (separates causal history transitions from successor target window; no future leakage)
dyn_view = build_view(manifest_path, "dyn", split="train", config=config)
sample = dyn_view[0]
history = sample["history_transitions"]  # transitions[:boundary]
target = sample["target_transitions"]    # transitions[boundary:boundary + supervised_intervals]
a1_batch = adapter_a1(sample)
```

Views/adapters preserve P11-required episode/split/lineage provenance. Short
episodes remain available to geometry; dynamics excludes windows without the
configured supervised coverage. This does not authorize training.

For an interrupted index update, the explicit Python `reconcile_episode` operation
accepts a known episode manifest; it is not a command to rescan directories or
rerun physics. A conflicting or incompatible episode must not replace the index.

### Offline validation

With ICGS installed in a supported Python environment, these checks read data or
exercise small fixtures without constructing a simulator:

```bash
python -B -c 'import sys; from icgs.data.datasets.episodes import validate_dataset_manifest; validate_dataset_manifest(sys.argv[1]); print("manifest valid")' /absolute/path/to/dataset/dataset_manifest.json
python -B -c 'import sys; from icgs.data.archives import read_episode_archive; read_episode_archive(sys.argv[1]); print("archive valid")' /absolute/path/to/dataset/episodes/episode-id/manifest.json
python -B -m unittest discover -s tests -p 'test_episode*.py' -v
python -B -m unittest discover -s tests -p 'test_collection_runner.py' -v
```

Physical simulation bindings, scene/expert generation, asset and sensor protocol
artifacts, and measured G1/G2 reports remain prerequisites outside this software
slice. Published C1–C5 and real simulator/training workloads are separate evidence.
