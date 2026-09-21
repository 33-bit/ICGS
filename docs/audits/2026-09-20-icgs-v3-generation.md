# Báo cáo generation ICGS primary v3 — phase 1

| | |
|---|---|
| Ngày | 2026-09-20 |
| Dataset | `icgs-primary-v3` |
| Schema episode | `icgs_episode_v2` |
| Phase | 1 |
| Trạng thái collection | **Chưa launch quota đầy đủ.** Pipeline, quota, schema và test đã khóa trong code. |
| `primary_v2` | Không ghi đè. Manifest `approved_composition_manifest.json` và profile `icgs_primary.json` giữ nguyên. |

Tài liệu này là mô tả generation hiện có trong repo. Reviewer không cần suy từ tên field hay từ bản review trước. Nếu một câu dưới đây không khớp code, đó là lỗi của báo cáo.

Nguồn khóa:

- Protocol: `src/icgs/data/collection/v3/protocol.py`
- Profile: `src/icgs/configuration/profiles/icgs_primary_v3.json`
- Manifest v3 (additive): `artifacts/composition/approved_composition_manifest_v3.json` (sinh từ v2, không sửa v2)
- Plan: `docs/plans/active/icgs-primary-v3-phase1.md`

---

## 1. Phạm vi phase 1

Phase 1 tạo dữ liệu để train và kiểm tra:

- geometry encoder (`D_geom`)
- temporal physical representation (`D_temporal`)
- initial dynamics (`D_dyn`)
- event/task tracker cơ bản (`D_task`)

Phase 1 **không** tạo:

- `pi_ref` / reference-policy execution
- recovery bank, `D_recovery`
- anchor / branch / continuation / `D_value`
- `D_audit`

Các field phase 2 vẫn có trên episode, giá trị **bắt buộc `null`**. Không được ghi `recovery_outcome = success` nếu recovery chưa chạy trong simulator.

---

## 2. Định danh đóng băng

| Field | Giá trị |
|---|---|
| `dataset_version` | `icgs-primary-v3` |
| `program_manifest_version` | `3` |
| `episode_schema_version` | `icgs_episode_v2` |
| `composition_protocol_id` | `icgs-composition-primary-v2` |
| `controller_protocol_id` | `rlbench-timed-ik-v2` |
| `execution_source` / `execution_mode` | `scripted_waypoint_v1` (đúng một mode) |
| `program_semantics_version` | `v3` |
| `camera_profile_id` | `rlbench-wrist-depth-v1` |
| `collection_seed` | `20260920` |
| `layout_version` | `3` |

---

## 3. Program, split, ai được train

36 program primary. E01 không thuộc primary.

| Split catalog | ID | Số program | Vai trò dữ liệu |
|---|---|---|---|
| `train` | T01–T20 | 20 | Collection train. Bên trong tách `subset=train_core` (học) và `subset=train_val` (validation nội bộ). |
| `development` | V01–V04 | 4 | Evaluation only. Không vào training view. |
| `test` | P1–P4, G1–G4, R1–R4 | 12 | Evaluation only. Không vào training view. |

Ánh xạ dùng khi train/eval model:

| Tên | Giá trị khóa | Nghĩa |
|---|---|---|
| `train_on` | `train_core` | Sampler training chỉ lấy episode `split=train` và `subset=train_core`. |
| `validation` | `train_val` | Validation nội bộ: `split=train` và `subset=train_val`. |
| `evaluation` | `development` + `test` | Dev và test, gồm cả perturbed held-out. |

`subset` được gán theo hash của `scene_signature` (`train_core_fraction = 0.8`), **không** gán theo frame. Một episode thuộc đúng một subset.

`training_eligible = true` chỉ khi: `semantic_status=authorized` **và** `split=train` **và** physics đã pass. V/P/G/R luôn `training_eligible=false`. Program lệch catalog/compiler phải `deferred` hoặc `training_eligible=false`.

Physics predicate đã pass 36/36 trên Colab với env tách từng program.

Family bắt buộc có trong train: `basic-manipulation`, `container-access`, `packing`, `grasp-fit`, `orientation`, `park-retrieve`, `gated-access`.

---

## 4. Quota collection — khác training mixture

Hai con số này **không cùng một thứ**.

### 4.1 Collection (khi chạy generator)

Đơn vị là **attempt**, không phải số episode success đã publish.

| Program | Nominal | Perturbed |
|---|---|---|
| Mỗi T01–T20 | Chạy đến khi có **200 successful contexts** | **80 valid attempts** (mọi `success` hoặc `valid_failure`) |
| Mỗi V01–V04 | **100 successful contexts** | **20 held-out valid attempts** |
| Mỗi P/G/R | **100 successful contexts** | **20 held-out valid attempts** |

Quy tắc đếm:

- Nominal `valid_failure` được giữ nếu observation hợp lệ; **không** đếm vào 200/100 success.
- Perturbed `success` đếm vào 80 (train) hoặc 20 (eval) attempts; **không** đếm vào 200 success.
- `simulator_crash` và `invalid_observation` **không** đếm vào quota, không vào dynamics.
- Stop condition: `quota_met` trong `src/icgs/data/collection/v3/quota.py`. Không stop vì `len(published_successes) == 280`.
- Cap an toàn: `quota_attempt_cap_multiplier = 5` lần target (tránh vòng lặp vô hạn).

Thứ tự planner: điền nominal đến đủ success, rồi mới điền perturbed attempts.

### 4.2 Training sampler (sau khi đã có episode)

Không phải tỷ lệ attempt lúc gen.

| View | Mixture |
|---|---|
| `D_geom` | Mọi frame hợp lệ của `train_core`. Không 70/30. |
| `D_temporal` | 70% transition nominal / 30% transition perturbed, trên `train_core`. |
| `D_dyn` | Cùng 70/30 trên `train_core`. |
| `D_task` | Pointer event trên `train_core`. |

`warmup_mixture = [0.7, 0.3]`, `mixture_measured_on = training_transitions`, `mixture_views = [D_temporal, D_dyn]`.

### 4.3 Khối lượng tối thiểu nếu mọi nominal success ngay lần đầu

| Hạng | Công thức | Số attempt hợp lệ tối thiểu |
|---|---|---|
| Train | 20 × (200 + 80) | 5600 |
| Dev | 4 × (100 + 20) | 480 |
| Test | 12 × (100 + 20) | 1440 |
| **Tổng** | 20×280 + 16×120 | **7520** |

Crash/invalid làm tăng số lần chạy, không tăng quota. CPU Colab ~10–20 s/episode. **Chưa launch.**

---

## 5. Hai loại record: episode và attempt

| `outcome` | Record | `episode_id` | Vào `D_dyn` / training view |
|---|---|---|---|
| `success` | `episode.json` (`icgs_episode_v2`) | có | có, nếu đúng split/subset |
| `valid_failure` | `episode.json` | có | có (là failure vật lý hợp lệ) |
| `simulator_crash` | `attempt.json` | **null** | không |
| `invalid_observation` | `attempt.json` | **null** | không |

Chỉ dùng **một** field kết quả: `outcome`. Không dùng `result_class` cho cùng nghĩa.

Field mô tả nguyên nhân, khác `outcome`:

| Field | Vai trò |
|---|---|
| `failure_type` | Loại lỗi vật lý nếu biết (`object_slip`, …). `null` khi `outcome=success`. Không copy giá trị `outcome`. |
| `terminal_reason` | Lý do dừng (`predicate_satisfied`, `predicate_failed`, `simulator_exception`, `observation_incomplete`). |
| `terminal_t` | Boundary/action index lúc kết thúc. |
| `valid_observation_until` | Index observation hợp lệ cuối. |
| `terminated` | `true` nếu attempt kết thúc có chủ đích. |
| `truncated` | `true` nếu dừng sớm hơn timeline đầy đủ (`terminal_t` < số action đã lên kế hoạch). |

Invariant `T` actions, `T+1` observations, `T` `dt` áp dụng cho **phần đã ghi** của episode hợp lệ (`success` / `valid_failure`). `valid_failure` được phép ngắn hơn demo đầy đủ; `T+1` vẫn đúng trên prefix đã lưu. Crash/invalid không phải episode nên không áp invariant đó.

`dt_t = achieved_duration_s`. Không sửa timestamp tay.

---

## 6. Timeline và observation

Mỗi episode hợp lệ:

- `T` transition/action
- `T+1` observation, robot state, object state
- `T` giá trị `dt`

Observation online (whitelist): `points`, `T_w_e`, `grip`, `point_valid`.

State giàu hơn nằm ở `robot_states` / `object_states` / layout sidecar: RGB, depth, mask, joint, object pose/vel, ρ/ν/ε khi generator ghi được. Không bịa modality thiếu.

---

## 7. Scene randomization — giá trị đã áp dụng

Seed episode:

```text
episode_seed = hash(dataset_version, program_id, collection_seed, episode_index)
collection_seed = 20260920
```

Eval dùng thêm `eval_perturbation_seed_offset = 1000003` trên collection seed của planner.

Randomization áp lên geometry (không chỉ metadata):

| Trục | Giá trị |
|---|---|
| Translation workspace | ±1.2 cm trên x,y (từ binding), stratified |
| Yaw | ±30°, 3 bin |
| Scale | 0.8–1.2, 3 bin |
| Position strata | 4 bin: train dùng `{0,1,2}`, eval dùng `{3}` |
| `inter_object_gap_m` | ±2 cm, chỉ object động (không dịch pad/target cùng lúc) |
| Approach / release waypoint | ±2 cm |
| Lighting | 3 profile + jitter azimuth; lưu `lighting_applied` |
| Camera viewpoint | offset yaw/pitch nhỏ; profile wrist cố định |

Thứ tự apply: **layout theo seed trước**, rồi tối đa **một** perturbation.

Duplicate: cùng `scene_signature` đã thấy thì loại, **kể cả seed khác**.

---

## 8. Held-out evaluation — không chỉ đổi seed

Train và eval **không** dùng cùng phân phối scene/perturbation.

| | Train (T01–T20) | Dev/test |
|---|---|---|
| Position bin | `{0, 1, 2}` | `{3}` (`eval_held_out_position_bin`) |
| Magnitude `r` | inner: `0 ≤ r < 0.70` | outer: `0.70 ≤ r ≤ 1.00` |
| `held_out` | `false` | `true` |
| Vào training view | `train_core` có; `train_val` không | không |

`r` là tỷ lệ so với biên tuyệt đối của loại perturbation đó. Hai bucket không chồng tại 0.70: inner exclusive, outer inclusive.

Khóa trong manifest `generation_targets.position_bins` và `generation_targets.perturbation_bounds.magnitude_buckets`.

Bốn role phải disjoint theo cả ba:

1. `scene_signature`
2. `asset_instance_id`
3. `source_episode_id` / `base_episode_id` (không trỏ sang role khác)

`asset_family_id` trùng giữa train và test được **báo** riêng (để đo OOD hình dạng nếu cần), không dùng làm điều kiện held-out thay cho ba mục trên.

Hàm kiểm tra: `split_disjointness_report` trong `src/icgs/data/collection/v3/report.py`.

---

## 9. Perturbation

Mỗi perturbed attempt đúng **một** can thiệp. Perturbed train clone scene của một nominal cùng program (`source_episode_id` = `base_episode_id` = episode nominal). Đó là cặp counterfactual.

### 9.1 Biên tuyệt đối (không thay bằng inner/outer)

| Loại | Biên vật lý | `application_scope` | `episode_has_external_intervention` | `application_t` |
|---|---|---|---|---|
| `action_pose_offset` | ≤ 2 cm, ≤ 10° | `event` | false | frame place, điền lúc chạy |
| `gripper_timing` | ±1 interval | `timestep` | false | frame grip |
| `object_displacement` | ≤ 3 cm | `initial_scene` | true | `null` (đã nằm trong `o_0`) |
| `blocker_insertion` | blocker khai báo, không chồng object | `initial_scene` | true | `null` |
| `pause_hold` | 2–10 interval, sim chạy thật | `event` | false | đầu routine |

`blocker_insertion` chỉ family `obstacle-access`, `gated-access`, `park-retrieve`, `park-restore`, `park-retrieve-restore`, `grasp-fit`. Program khác: `not_applicable`, chia 80 (train) hoặc 20 (eval) cho loại còn lại. T01 không có blocker; T03 có.

Mục tiêu train khi đủ 5 loại: 16 attempt/loại (`perturbed_per_kind=16`).

### 9.2 D_dyn — cờ can thiệp

Ba field, nghĩa khác nhau:

| Field | Khi nào `true` / có giá trị |
|---|---|
| `episode_has_external_intervention` | Episode có displacement/blocker lúc reset. |
| `transition_has_external_intervention` | Chỉ khi `application_scope` ∈ {`timestep`,`event`} **và** `application_t = t` của sample D_dyn. Displacement lúc reset → **false** mọi `t`. |
| `initial_scene_intervention_id` | ID can thiệp nếu scope là `initial_scene`; ngược lại `null`. |

Không tạo transition giả tại `t=0` cho can thiệp đã nằm trong `o_0`.

---

## 10. Provenance episode — field nào có, field nào null

Luôn ghi (phase 1):

```text
attempt_id
episode_id
program_id
program_semantics_version
dataset_version
program_manifest_version
split
subset                  # train_core | train_val | null trên dev/test
scene_signature
scene_seed
asset_family_id
asset_instance_id
episode_kind            # nominal | perturbed
execution_mode
execution_source
controller_version
simulator_version
physics_engine_version
predicate_protocol_id
calibration_id
camera_profile_id
camera_intrinsics_id
camera_extrinsics_id
action_space_id
orientation_convention  # xyzw
gripper_unit            # open_1_closed_0
outcome
failure_type
terminal_reason
terminal_t
valid_observation_until
terminated
truncated
observation_origin      # measured
```

Wrist camera: `T_world_camera_t = T_world_ee @ T_ee_cam`. Depth unit `meter`. Point cloud frame `world`.

Khi perturbed, thêm:

```text
intervention_id
intervention_type
intervention_params
intervention_seed
application_scope
application_t
source_episode_id
base_episode_id
episode_has_external_intervention
initial_scene_intervention_id
magnitude_bucket          # inner | outer
held_out
```

Phase 2 — **có key, giá trị null**:

```text
policy_id
policy_version
parent_episode_id
anchor_id
snapshot_id
branch_id
candidate_id
candidate_index
continuation_group_id
trial_id
trial_seed
recovery_id
recovery_budget
recovery_outcome
snapshot_fidelity
restore_validation_result
```

`snapshot` object cũng có mặt với mọi key null: `sim_time`, `rng_state`, robot/object/articulation/gripper/controller/physics state, `scene_asset_digest`, `snapshot_fidelity`, `restore_validation_result`.

---

## 11. Semantics — không chỉ predicate

Trước khi authorize/scale một train program:

```text
catalog ordered_steps
= compiled events
= observed task events
= postcondition labels
```

Gate: `generation_parity_gate` (`src/icgs/data/collection/v3/report.py`). Generator `--protocol v3` gọi gate này. Program lệch → `blocked`, không scale.

Ràng buộc primitive:

- `push` / `push_through_aperture` compile và execute là `type=push` (gripper mở, không grasp). Worker **không** đổi thành `pick_place`.
- `open` / `close` là `open_articulation` / `close_articulation` (slide theo trục, freeze handle sau close).
- `regrasp` nominal không gắn nhãn recovery.
- `pause_hold` là hold vật lý, không sửa timestamp.

---

## 12. View pointer

Không copy observation. Crash/invalid không vào view.

| View | Role mặc định | Nội dung |
|---|---|---|
| `D_geom` | `train` → `train_core` | Mọi boundary hợp lệ; RGB/depth/pointcloud/mask + `calibration_id`. |
| `D_temporal` | `train_core` | Cửa sổ quanh contact/event, rồi mix 70/30. |
| `D_dyn` | `train_core` | `(history, a_t, o_t, o_{t+1}, dt)` + cờ intervention ở mục 9.2; mix 70/30. |
| `D_task` | `train_core` | Pointer event, ρ, ν, ε theo timestep. Không suy event chỉ từ success cuối. |

`build_v3_view(..., role="validation"|"evaluation")` cho `train_val` và dev/test.

---

## 13. Uniqueness index (pool + retry)

`plan_full_generation` tạo pool unique-scene (train 200 nominal + 80 perturbed; eval 100+20). Đó **không** phải điều kiện dừng collection.

Nếu nominal fail, generator stream thêm scene. Toàn bộ pool + scene mới dùng **một** `AttemptPlanner.seen`:

- `plan_full_generation(..., return_planners=True)` trả planner.
- Stream/resume: `planner.next_plan` trên planner đó, hoặc `ingest_plans` / `remember_row` rồi mới `next_plan`.
- **Cấm** tạo `AttemptPlanner` mới từ `index=0` sau khi đã có pool — sẽ trùng scene.

Generator v3 ingest `scene_signature` từ manifest khi resume.

---

## 14. Pipeline

```text
quota.next_episode_kind
  → AttemptPlanner.next_plan(kind)     # uniqueness index
  → prepare_attempt                    # layout + đúng một perturbation
  → worker / expert scripted_waypoint_v1
  → assemble_episode_v2  hoặc  assemble_attempt_record
```

File:

| File | Việc |
|---|---|
| `src/icgs/data/collection/v3/quota.py` | Stop condition attempt |
| `src/icgs/data/collection/v3/batch.py` | AttemptPlanner, magnitude bucket, provenance |
| `src/icgs/data/collection/v3/diversity.py` | Seed, strata, signature, duplicate |
| `src/icgs/data/collection/v3/attempt_prep.py` | `prepare_attempt`, `plan_full_generation` |
| `src/icgs/data/collection/v3/perturbations.py` | 5 kind, scope |
| `src/icgs/data/collection/v3/report.py` | Parity, leakage, mixture |
| `src/icgs/data/datasets/v3_views.py` | Pointer views + 70/30 |
| `scripts/colab_v3_pilot_episodes_worker.py` | Apply layout; không remap push |
| `scripts/colab_g2_dataset_generator.py --protocol v3` | Vòng collection |

`--allow-planned` **cấm** trên v3 production. Manifest phải authorize.

---

## 15. Lệnh (chưa chạy full)

Một attempt:

```bash
ICGS_V3_ATTEMPT_JSON=/path/plan.json \
  python scripts/colab_v3_pilot_episodes_worker.py T01
```

Generator (không `--allow-planned`):

```bash
python scripts/colab_g2_dataset_generator.py \
  --protocol v3 \
  --approved-manifest artifacts/composition/approved_composition_manifest_v3.json \
  --use-manifest-targets \
  --tasks T01
```

---

## 16. Test đã khóa hành vi trên

Chạy:

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_primary_v3.py' -q
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_v3_batch.py' -q
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_v3_expert.py' -q
PYTHONPATH=src python3 -B -m pytest tests/test_colab_generator_manifest_gate.py -q
```

Lần chạy gần nhất: `test_primary_v3` 25, `test_v3_batch` 18, `test_v3_expert` 8, generator gate 6 — pass.

Hành vi test bao gồm (không phải danh sách hết file):

- Train quota 200 success rồi 80 perturbed attempts; crash không đếm.
- V01/P1: **100 nominal + 20 held-out perturbed** (không còn 0 perturbed).
- Inner `r < 0.70`, outer `r ≥ 0.70`.
- Bốn split disjoint trên signature / asset instance / source episode.
- Stream retry không trùng signature pool.
- Push không remap pick-place.
- D_dyn: initial-scene không gắn transition tại t=0; có `initial_scene_intervention_id`.
- Provenance có đủ identity + field phase 2 null.
- Manifest ghi biên tuyệt đối và position bins.
- v3 từ chối `--allow-planned`.
- Parity gate pass trên catalog authorized.

---

## 17. Checklist reviewer (câu trả lời trực tiếp)

| Câu hỏi review | Trả lời trong generation này |
|---|---|
| 70/30 là quota collection? | Không. Collection = 200 success + 80 valid perturbed. 70/30 chỉ sampler `D_temporal`/`D_dyn`. |
| Train trên toàn bộ `split=train`? | Không. `train_on=train_core`. `train_val` là validation. |
| Dev/test có perturbation? | Có. 100 success nominal + **20** held-out perturbed / program. |
| Held-out chỉ đổi seed? | Không. Position bin 3 + magnitude outer; train bin 0–2 + inner. |
| `outcome` và `result_class` cùng nghĩa? | Không. Chỉ `outcome`. |
| Crash có `episode_id`? | Không. Chỉ `attempt_id`. |
| `intervention_frame=0` cho displacement? | Không. `application_scope=initial_scene`, `application_t=null`. |
| D_dyn gắn external tại t=0 cho displacement? | Không. `transition_has_external_intervention=false`. |
| Push thành pick-place? | Không. |
| `--allow-planned` trên v3? | Cấm. |
| Pool 200+80 là stop? | Không. Stop là quota success/attempt. Pool + retry chung uniqueness index. |
| Phase 2 field có trên schema? | Có, giá trị null. |
| Đã launch 7520 attempt? | **Chưa.** |

---

## 18. Việc còn lại trước full scale

1. Pilot nhỏ trên vài family + vài perturbation kind.
2. Báo cáo ngắn: parity, schema/timeline, split leakage, diversity, perturbation quota.
3. Chỉ khi các mục đó pass mới chạy 200+80 / 100+20.
