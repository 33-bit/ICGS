# Báo cáo: định nghĩa task program và trạng thái perturbation/recovery

Ngày: 2026-09-20.
Phạm vi: catalog P02, approved composition manifest, primitive compiler, generator Colab, config method, và dữ liệu đã gen (local + Hugging Face).
Kết luận ngắn: **task program đã freeze 20/4/12; data đã gen là expert `scripted_waypoint_v1`; perturbation/recovery mix chưa có.**

---

## 1. Kết luận

1. Program primary là 36 skeleton (`T01`–`T20`, `V01`–`V04`, `P1`–`P4`, `G1`–`G4`, `R1`–`R4`). `E01` thuộc exploratory, bị loại khỏi primary.
2. Lớp executable copy `ordered_steps` từ catalog, nhưng **compiler không parse steps**: nó lookup `raw_specs[program_id]` hardcode. Một số routine thiếu bước catalog (đặc biệt `close` ở T06/T08/T19).
3. Mọi program trong approved manifest hiện `execution_status = generation_authorized` và chỉ có một mode: `scripted_waypoint_v1`.
4. Method spec yêu cầu mix 50/30/20 (demo / reference / perturbation-recovery) sau freeze, warm-up 70/30. Config JSON đã ghi các số này. **Generator không consume.**
5. Data đã gen (HF `33bit/icgs`) là episode success scripted + quarantine crash. Không có intervention ID, không có recovery label, không có mix class.

---

## 2. Ba lớp định nghĩa program

### 2.1 Catalog skeleton (canonical)

File: `src/icgs/data/collection/programs.py`

`ProgramSpec` chỉ chứa intent:

| Field | Ý nghĩa |
|---|---|
| `program_id` | T01–T20, V01–V04, P1–P4, G1–G4, R1–R4, E01 |
| `split` | `train` / `development` / `test` / `dev` (E01) |
| `steps` | ordered semantic primitives |
| `constraint` | ràng buộc vật lý/ngữ nghĩa |
| `family` | chỉ set trên E/P/G/R trong catalog; T/V = `None` |

Catalog **không** bind asset, seed, controller, predicate, workspace.

Cardinality:

| Tập | Số | Split |
|---|---|---|
| Catalog `_SPECS` | 37 | 20 train + 4 development + 12 test + 1 E01 |
| Approved primary | 36 | 20/4/12 (E01 excluded) |
| Compiler `raw_specs` | 36 | cùng 20/4/12 |
| G2-pilot historical | 6 | T06, T08, T09, T11, T13, T14 |

### 2.2 Approved composition manifest (binding để gen)

File: `artifacts/composition/approved_composition_manifest.json`

Protocol: `icgs-composition-primary-v1`, `manifest_version: 1`.

Mỗi row bắt buộc có:

`program_id`, `split`, `family`, `ordered_steps`, `scene_id`, `asset_family_id`, `asset_version`, `source_lineage_id`, `workspace{x_m,y_m,z_m}`, `randomization{translation_m, yaw_deg, scale, camera_profile_id, lighting_profile_id, waypoint_delta_m}`, `seed_demos` (đúng 5 ID), `execution_modes`, `controller`, `predicates`, `execution_status`.

Trạng thái hiện tại (cả 36 row):

- `execution_status`: `generation_authorized`
- `execution_modes`: `["scripted_waypoint_v1"]`
- predicates: `position_m = 0.01`, `rotation_deg = 10`, `hold_intervals = 5`, `unsafe_penetration_m = 0.002`
- controller: `rlbench-timed-ik-v1`, `step_s = 0.05`
- randomization: translation ±0.012 m (xy), yaw ±30°, scale 0.8–1.2, `waypoint_delta_m` ±0.01 m

`generation_targets`:

| Key | Giá trị |
|---|---|
| `train_successes_per_program` | 200 |
| V01 / V02 / V03 / V04 | 63 / 63 / 62 / 62 |
| `test_contexts_per_composition` | 100 |
| `retain_valid_failures` | true |
| `test_is_evaluation_only` | true |

Asset split khai báo 70/15/15, closed by `asset_family_id`. Validator (`src/icgs/data/collection/approved_manifest.py`) bắt `ordered_steps` khớp catalog và cardinality 20/4/12.

Lưu ý: `docs/audits` README composition cũ nói nhiều entry còn `planned`; **manifest hiện tại đã nâng cả 36 lên `generation_authorized`**. Config `dataset.minimum_execution_modes = 2` **mâu thuẫn** với manifest (chỉ 1 mode).

### 2.3 Primitive compiler (task chạy được)

File: `src/icgs/data/collection/primitive_compiler.py`

Luồng:

1. Load approved JSON.
2. Lookup `raw_specs[pid]` (không parse `ordered_steps`).
3. Ghi đè tolerance điều kiện bằng `predicates.position_m` (0.01 m).
4. Copy `ordered_steps` / split / family / scene từ manifest.
5. Emit `ProceduralTask` RLBench: cuboid objects, `NearCondition`, waypoint dummy, expert `routine`.

Primitive thực sự dùng: `lift`, `pick_place`, `reach`, `grasp_rotate`, `place`, `transport_through_aperture`, `touch_retreat`. Test cho phép `push`; **không routine nào đang dùng `push`**.

`init_episode` jitter ≈ `((index*17)%7-3)*0.004` m trên xy (tối đa ±0.012 m). Đây là scene variation, không phải perturbation protocol.

Generator: `scripts/colab_g2_dataset_generator.py` — default `--execution-mode scripted_waypoint_v1`, compile catalog nếu có, fallback `TASK_SPECS` chỉ 6 pilot. Success mới publish; fail → quarantine.

---

## 3. Catalog đầy đủ

Family trên bảng lấy từ **approved manifest** (T/V không có family trong `ProgramSpec`).

### 3.1 Train — T01–T20

| ID | Family | Ordered steps (catalog) | Routine compiler | Constraint |
|---|---|---|---|---|
| T01 | basic-manipulation | grasp A → lift A | `lift(object_a)` | grasp position, stable hold |
| T02 | basic-manipulation | grasp A → place A on pad | `pick_place(object_a)` | target relation, pad size |
| T03 | obstacle-access | push blocker → reach target | `pick_place(blocker)`, `reach(target_a)` | access clearance |
| T04 | container-access | open drawer → close drawer | 2× `pick_place` handle | joint limits and final closure |
| T05 | container-access | open drawer → retrieve A | handle + `pick_place(object_a)` | reachability after opening |
| T06 | container-access | place A into open drawer → **close** | `pick_place(object_a)` **(thiếu close)** | closing clearance |
| T07 | packing | place A → place B into open tray | 2× `pick_place` | shared free space |
| T08 | packing | place B → place A → **close open drawer** | `pick_place(B)`, `pick_place(A)` **(thiếu close)** | ordering and packing |
| T09 | grasp-fit | grasp A → place A into open holder | `pick_place(object_a)` | grasp-to-final-pose compatibility |
| T10 | grasp-fit | push A through aperture → place A | `transport_through_aperture` | aperture geometry, no grasp transfer |
| T11 | orientation | grasp A → rotate A → place A on pad | `grasp_rotate(90°)`, `place` | orientation requirement |
| T12 | gated-access | open gate → push A through gate → close gate | open, transport, close | prerequisite and gate closure |
| T13 | park-retrieve-restore | park blocker → retrieve A | 2× `pick_place` | parking and access |
| T14 | park-retrieve-restore | park blocker → restore blocker | park, `touch_retreat(object_a)`, restore | temporary relation and restoration |
| T15 | park-retrieve-restore | park A → park B → retrieve C | 3× `pick_place` | two objects share parking space |
| T16 | container-access | open drawer → park blocker → retrieve A | 3× `pick_place` | container access |
| T17 | regrasp | grasp A → temporary place → regrasp A | 2× `pick_place` | regrasp and retry history |
| T18 | shared-corridor | retrieve A → place A → retrieve B | 2× `pick_place` | shared retrieval corridor |
| T19 | packing | place spacer → place A → **close open drawer** | spacer + A **(thiếu close)** | spacer-dependent packing |
| T20 | park-retrieve-restore | park blocker → retrieve A → place A on pad | park + retrieve/place | parking and transport path |

### 3.2 Development — V01–V04

| ID | Family | Ordered steps | Routine |
|---|---|---|---|
| V01 | pack-add-close | open → place A → close | 3× `pick_place` |
| V02 | grasp-transport-fit | grasp → rotate → place into holder | `grasp_rotate`, `place` |
| V03 | park-retrieve-restore | park → retrieve → place → restore | park, retrieve, restore |
| V04 | gated-access | open gate → retrieve → close gate | open, retrieve, close |

### 3.3 Test — P / G / R (locked)

| ID | Family | Ordered steps | Ghi chú |
|---|---|---|---|
| P1 | pack-add-close | open → place A → place B → close | pack-add-close |
| P2 | pack-add-close | open → place B → place A → close | đảo thứ tự / size |
| P3 | pack-add-close | open → place A → place spacer → place B → close | spacer-dependent packing |
| P4 | pack-add-close | open → place A → retrieve C → place B → close | retrieve trước place sau |
| G1 | grasp-transport-fit | grasp → transport through aperture → place into holder | 1× transport |
| G2 | grasp-transport-fit | grasp → rotate in free space → transport through aperture → place | rotate + transport |
| G3 | grasp-transport-fit | open gate → grasp → transport → place → close gate | gate prerequisite |
| G4 | grasp-transport-fit | grasp → temporary place → regrasp → transport → fit | constraint: **recovery remains solvable**; routine = 2× `pick_place` |
| R1 | park-retrieve-restore | park blocker → retrieve target → restore blocker | |
| R2 | park-retrieve-restore | park A → park B → retrieve → restore B → restore A | restoration order |
| R3 | park-retrieve-restore | open drawer → park blocker → retrieve → restore → close | |
| R4 | park-retrieve-restore | park blocker → retrieve A → retrieve B → restore | two targets |

### 3.4 Exploratory (không primary)

| ID | Split | Steps | Family |
|---|---|---|---|
| E01 | `dev` | grasp target → lift target | `rlbench-exploratory` |

Manifest ghi `excluded_tracks`: E01 “not part of primary 20/4/12 cardinality”. Runner cấm E-series trên primary track.

### 3.5 Lệch catalog vs compiler (quan trọng)

Compiler **không** derive routine từ `ordered_steps`. Lệch đã thấy:

- T03 catalog `push blocker` → compile `pick_place(blocker)` (không dùng primitive `push`).
- T06 / T08 / T19 catalog có `close` → routine không close.
- T10 catalog `push A through aperture` → `transport_through_aperture` (grasp-carry, không push).
- G4 catalog 5 bước → 2 `pick_place`.
- T17/G4 “regrasp” là **bước trong skeleton**, không phải recovery data sau perturbation.

---

## 4. Perturbation / recovery: spec vs code vs data

### 4.1 Spec (chưa phải evidence đã collect)

Nguồn: `docs/method/data-training.md` (status: **target, not collected data**).

Sau freeze reference:

| Mix | Tỷ lệ | Ý nghĩa |
|---|---|---|
| Warm-up | 70 / 30 | demo attempts / perturbed |
| Frozen | 50 / 30 / 20 | executed demo / reference / **bounded perturbation+recovery** |

Perturbation được phép (phải chạy physics, ghi **intervention ID**, pre/post obs):

- target offset ≤ 2 cm / 10°
- grip timing ±1 interval
- stationary object displacement ≤ 3 cm
- declared blocker insertion
- pause 2–10 intervals, retry executed (không sửa timestamp)

Recovery label: cần **ít nhất một executed recovery thành công** trong budget. Khoảng cách tới demo **không** phải recovery label. Unrepresented recovery → null.

P08 (anchor/pair recovery) trong plan: runtime unimplemented; `pairs.py` planned, **không có file**.

### 4.2 Config đã khai (không được generator đọc)

`src/icgs/configuration/profiles/icgs_primary.json` → `MethodConfig`:

```text
dataset.episode_mixture              = [0.5, 0.3, 0.2]
dataset.warmup_episode_mixture       = [0.7, 0.3]
dataset.target_offset_m              = 0.02
dataset.target_rotation_offset_deg   = 10.0
dataset.grip_timing_offset_intervals = 1
dataset.object_shift_m               = 0.03
dataset.pause_intervals              = [2, 10]
dataset.minimum_execution_modes      = 2
losses.recovery_pair_weight          = (loss weight only)
collection.pilot.program_ids         = T06, T08, T09, T11, T13, T14
```

`episode_mixture` chỉ được validate tổng ratio. Không consumer trong `runner.py`, `attempts.py`, `training_layout.py`, hay `colab_g2_dataset_generator.py`.

### 4.3 Code generation thực tế

| Thành phần | Hành vi |
|---|---|
| Generator `--execution-mode` | default `scripted_waypoint_v1` |
| `collect_single_episode` | một expert scripted; không perturb waypoint |
| Success gate | predicate fail → exception |
| Loop | đếm success tới quota; không sample mix |
| Episode record | `execution_mode=scripted_waypoint_v1` |
| `randomization_declared` | copy range từ binding, không apply protocol perturbation |
| Fail path | `quarantine/` + `failure_attempts`, `status=failed_attempt` |
| `run_collection` | provenance `execution_mode`; không mix sampler |
| `training_layout` v2 | không có mix/kind/intervention |
| `annotations.py` | event overlap rho/nu/eligibility; không recovery pool |
| G4 text | family/constraint, không episode type |

Manifest approved cũng chỉ target **success counts**, không quota perturbation/recovery.

### 4.4 Cái dễ nhầm với recovery

| Có trong repo | Thực chất | Không phải |
|---|---|---|
| T17 / G4 regrasp | bước nominal của program | executed recovery sau perturbation |
| `waypoint_delta_m` ±1 cm | range khai báo | mix perturbation |
| `init_episode` ±1.2 cm xy | scene variation | intervention ID |
| failure quarantine | crash / predicate fail | recovery label |
| `retain_valid_failures: true` | giữ attempt hỏng | 20% recovery pool |
| `recovery_pair_weight` | hệ số loss | data pair |

---

## 5. Data đã gen

### 5.1 Local artifacts

| Artifact | Nội dung |
|---|---|
| `artifacts/composition/suite-smoke-receipt.json` | 36/36 PASS, một scripted success / program (smoke, unpublished) |
| `artifacts/composition/remote_dataset_manifest.json` | `"episodes": []` (snapshot local rỗng) |
| `artifacts/g2-pilot/` | 6 task `.py`/`.ttm` + bindings; README: chưa accept training episode |
| G2-pilot `predicate_tolerances.position_m` | 0.08 m (lỏng hơn approved 0.01 m) |

### 5.2 Hugging Face `33bit/icgs`

Kiểm tra prefix (thời điểm báo cáo):

| Prefix | Episodes | Loại |
|---|---|---|
| `primary_v2/` | 2885 success | `scripted_waypoint_v1` (2882/2885) |
| `primary/` | 497 success | cùng mode scripted |
| `exploratory/` | 15 | E01 / G2-pilot, không mix |

Trong `primary_v2/dataset_manifest.json`: **0** match keyword `perturb` / `recover` / `interven`.

Ví dụ success `primary_v2/episodes/g2-ep-t01-10000/`:

- sidecar: `episode_id`, `program_id`, `split`, `seed` only
- `result.json`: `success`, `terminal_reason = predicate_satisfied`
- `manifest.json`: `randomization_declared` = copy range, không phải perturbation đã apply

Ví dụ fail `primary_v2/failure_attempts/g2-ep-t01-10034/`: `failed_attempt`, nguyên nhân kiểu V-REP crash — **không** phải recovery trial.

### 5.3 Schema episode

Provenance bắt buộc (`src/icgs/data/schemas/episodes.py`): `episode_id`, `program_id`, `source_lineage_id`, `asset_family_id`, `split`, `calibration_id`, `observation_origin=measured`, `raw_commands_id`, `materialized_commands_id`.

Không có field mix class, intervention ID, recovery outcome, hay perturbation seed trong schema bắt buộc.

---

## 6. File nguồn chính

| Vai trò | Path |
|---|---|
| Catalog | `src/icgs/data/collection/programs.py` |
| Approved validator | `src/icgs/data/collection/approved_manifest.py` |
| Bindings (G2-pilot) | `src/icgs/data/collection/bindings.py` |
| Compiler | `src/icgs/data/collection/primitive_compiler.py` |
| Collection runner | `src/icgs/data/collection/runner.py` |
| Annotations | `src/icgs/data/collection/annotations.py` |
| Training sidecar | `src/icgs/data/training_layout.py` |
| Generator | `scripts/colab_g2_dataset_generator.py` |
| Method spec | `docs/method/data-training.md` |
| Config mix | `src/icgs/configuration/profiles/icgs_primary.json` |
| Approved catalog | `artifacts/composition/approved_composition_manifest.json` |
| Composition protocol | `docs/superpowers/specs/2026-09-18-composition-data-protocol.md` |
| P02 plan | `docs/plans/active/icgs-p02-episode-data.md` |

Tests khóa cardinality: `tests/test_episode_data.py`, `tests/test_36_composition_compiler.py`, `tests/test_approved_composition_manifest.py`, `tests/test_program_bindings.py`.

---

## 7. Khoảng trống còn lại

1. **Perturbation/recovery generator chưa implement**: không sampler mix, không apply `target_offset_m` / `object_shift_m` / pause / blocker insertion, không intervention ID, không recovery continuation.
2. **Mode thứ hai thiếu**: spec/config đòi ≥2 execution modes; manifest/gen chỉ `scripted_waypoint_v1`.
3. **Routine lệch catalog** ở T06/T08/T19 (`close`) và T03 (`push` → `pick_place`).
4. **Failure ≠ recovery**: quarantine crash không thỏa “executed recovery under allowed budget”.
5. **P08 pairs/recovery grouping**: file `pairs.py` không tồn tại.
6. **README composition** (nếu còn nói nhiều program `planned`) **stale** so với manifest `generation_authorized`.

---

## 8. Việc tiếp theo (gợi ý, chưa làm trong báo cáo này)

Nếu mục tiêu là đủ data cho `L_recovery` / pair recovery:

1. Freeze protocol perturbation (seed, object, magnitude, intervention ID, pre/post obs).
2. Generator: perturb → physics → expert/reference recover → chỉ label recovery khi recover thành công trong budget.
3. Thêm `execution_mode` thứ hai nếu vẫn giữ `minimum_execution_modes = 2`.
4. Ghi mix class trên episode provenance (demo / reference / perturbation / recovery) trước khi scale thêm success scripted.
5. Quyết định T06/T08/T19: sửa compiler cho khớp `close`, hoặc sửa catalog steps cho khớp routine.

Báo cáo này là snapshot kiểm tra, không chứng nhận G2 physical gate hay primary collection đã đủ mix.
