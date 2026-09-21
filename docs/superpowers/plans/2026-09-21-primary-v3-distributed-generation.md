# Primary v3 Distributed Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate and launch a resumable 200-worker/one-coordinator Colab pipeline that generates the complete `icgs-primary-v3` phase-1 dataset and publishes closed batches to `33bit/icgs` every 300 seconds.

**Architecture:** Pure Python contracts, queue, planner, validator and publisher modules own deterministic state and are unit-tested without a simulator. Thin Colab entry points run one exact preplanned RLBench attempt per worker, while one coordinator owns all planners, quota counts, manifest mutation and Hugging Face commits. A launcher performs mandatory preflight/smoke gates, starts exactly 200 deterministic-display workers plus one coordinator, and leaves the session alive.

**Tech Stack:** Python 3.11, NumPy 1.26.4, CoppeliaSim 4.1, pinned PyRep/RLBench, `huggingface_hub==0.26.2`, Colab CLI, filesystem atomic rename, unittest/pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-primary-v3-distributed-generation-design.md`

## Global Constraints

- Dataset identity is `icgs-primary-v3`; episode schema is `icgs_episode_v2`; composition protocol is `icgs-composition-primary-v2`; controller protocol is `rlbench-timed-ik-v2`.
- Preserve `primary_v2`, `artifacts/composition/approved_composition_manifest.json` and `src/icgs/configuration/profiles/icgs_primary.json` byte-for-byte.
- Use exactly 200 worker slots and one coordinator; workers never read the Hugging Face token or call the Hugging Face API.
- Publish to dataset `33bit/icgs` below `primary_v3/`; normal cadence is 300 seconds and quota completion forces a final flush.
- Retain both `success` and `valid_failure` as episodes. `simulator_crash` and `invalid_observation` are attempt records with null `episode_id` and never count toward a valid quota.
- Train quotas are 200 nominal successes plus 80 valid perturbed attempts per T program. Development/test quotas are 100 nominal successes plus 20 held-out valid perturbed attempts per program.
- Use one centralized `AttemptPlanner` per program. Never reconstruct a planner at index zero after plans or manifest rows exist.
- Every episode has `T` transitions, `T+1` observations/states and `T` achieved durations. Do not fabricate missing modalities.
- Use fixed X display `200 + worker_id`; never use `xvfb-run -a` for the 200-worker launch.
- No training, preprocessing, model execution, robot motion or benchmark workload is part of this plan.
- A fresh 36-program smoke, a 2-worker queue smoke, a 200-worker bounded smoke and a verified small publication must pass before full quota launch. The 36-program smoke accepts retained `success` and `valid_failure` episodes with valid timelines; crash, invalid observation and SKIPPED remain blocking.
- Do not stop the Colab session after launch or completion; leave it allocated for owner inspection.

## File structure

- `src/icgs/data/collection/v3/distributed_contracts.py`: immutable run/job/result contracts and strict JSON serialization.
- `src/icgs/data/collection/v3/distributed_queue.py`: atomic filesystem queue, claims, heartbeats and stale-claim recovery.
- `src/icgs/data/collection/v3/distributed_planner.py`: centralized 36-program planners, dynamic quota allocation and resumable state.
- `src/icgs/data/collection/v3/distributed_validation.py`: closed-result validation and manifest/quota ingestion.
- `src/icgs/data/collection/v3/distributed_publication.py`: five-minute batch selection, remote merge and verified Hugging Face commits.
- `src/icgs/data/collection/v3/rlbench_attempt.py`: full-fidelity materialization of successful and valid-failure raw attempts.
- `scripts/colab_v3_distributed_worker.py`: one persistent worker loop; no credentials or publication code.
- `scripts/colab_v3_distributed_coordinator.py`: coordinator event loop and progress receipts.
- `scripts/colab_v3_distributed_watchdog.py`: coordinator/worker liveness and bounded replacement.
- `scripts/colab_v3_distributed_launch.py`: VM gates, deterministic X displays and detached launch.
- `tests/test_v3_distributed_*.py`: pure local tests for every state transition and failure mode.
- `docs/audits/2026-09-21-icgs-v3-full-launch.md`: exact preflight, launch and live-run receipt.

---

### Task 1: Immutable distributed contracts and atomic queue

**Files:**
- Create: `src/icgs/data/collection/v3/distributed_contracts.py`
- Create: `src/icgs/data/collection/v3/distributed_queue.py`
- Create: `tests/test_v3_distributed_queue.py`

**Interfaces:**
- Consumes: `AttemptPlan`, `attempt_from_dict`, frozen protocol IDs.
- Produces: `RunConfig`, `GenerationJob`, `WorkerResult`, `QueueCounts`, `FilesystemJobQueue`.

- [ ] **Step 1: Write failing contract round-trip and invariant tests**

```python
def test_job_roundtrip_preserves_exact_attempt_plan():
    plan = make_plan("T01", index=17, kind="perturbed")
    job = GenerationJob.create(
        run_id="run-1", attempt_id="attempt-1", episode_id="episode-1",
        program_id="T01", plan=plan, code_revision="a" * 40,
        manifest_sha256="b" * 64, output_root="/content/run/staging",
    )
    assert GenerationJob.from_dict(job.as_dict()) == job
    assert GenerationJob.from_dict(job.as_dict()).plan == plan

def test_attempt_outcome_requires_null_episode_id():
    with pytest.raises(ValueError, match="episode_id must be null"):
        WorkerResult(
            job_id="job-1", attempt_id="attempt-1", episode_id="episode-1",
            program_id="T01", outcome="simulator_crash", result_dir="x",
            file_sha256={}, timeline=None,
        )
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_queue.py -q`

Expected: collection error because `distributed_contracts` does not exist.

- [ ] **Step 3: Implement strict dataclasses and JSON serializers**

```python
@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_root: str
    code_revision: str
    approved_manifest_sha256: str
    worker_count: int = 200
    publish_interval_s: int = 300
    hf_repo: str = "33bit/icgs"
    hf_subfolder: str = "primary_v3"

@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    run_id: str
    attempt_id: str
    episode_id: str
    program_id: str
    plan: AttemptPlan
    code_revision: str
    manifest_sha256: str
    output_root: str
    retry_generation: int = 0

@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    attempt_id: str
    episode_id: str | None
    program_id: str
    outcome: str
    result_dir: str
    file_sha256: dict[str, str]
    timeline: dict[str, int] | None
```

Validate exact keys, nonblank IDs, SHA lengths, `worker_count == 200`,
`publish_interval_s == 300`, outcome vocabulary and episode-ID rules. Serialize
plans through `AttemptPlan.as_dict()` and `attempt_from_dict()`.

- [ ] **Step 4: Write failing atomic queue tests**

```python
def test_two_workers_cannot_claim_same_job(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(make_job("job-1"))
    first = queue.claim("000")
    second = queue.claim("001")
    assert first.job_id == "job-1"
    assert second is None

def test_stale_claim_requeues_without_changing_job_identity(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    queue.enqueue(make_job("job-1"))
    claimed = queue.claim("000", now_s=10.0)
    recovered = queue.recover_stale(now_s=71.0, stale_after_s=60.0)
    assert recovered == ["job-1"]
    assert queue.claim("001").attempt_id == claimed.attempt_id
```

- [ ] **Step 5: Implement `FilesystemJobQueue`**

Implement these exact methods: `__init__(root: str | Path) -> None`,
`enqueue(job: GenerationJob) -> Path`,
`claim(worker_id: str, *, now_s: float | None = None) -> GenerationJob | None`,
`write_heartbeat(worker_id: str, job_id: str | None, *, now_s: float | None = None) -> Path`,
`publish_ready(worker_id: str, result: WorkerResult) -> Path`,
`iter_ready() -> Sequence[WorkerResult]`,
`mark_ingested(result: WorkerResult) -> Path`,
`mark_published(job_id: str) -> Path`,
`recover_stale(*, now_s: float, stale_after_s: float) -> list[str]`, and
`counts() -> QueueCounts`.

Write JSON to a sibling temporary file, flush and `os.fsync`, then `os.replace`.
Claim with `os.replace(pending_path, claimed_path)` and treat
`FileNotFoundError` as a lost race. Reject duplicate immutable job bytes.

- [ ] **Step 6: Run queue tests**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_queue.py -q`

Expected: PASS, zero skipped.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/icgs/data/collection/v3/distributed_contracts.py \
  src/icgs/data/collection/v3/distributed_queue.py \
  tests/test_v3_distributed_queue.py
git commit -m "feat(data): add atomic v3 generation queue"
```

### Task 2: Full-fidelity raw-attempt materialization

**Files:**
- Create: `src/icgs/data/collection/v3/rlbench_attempt.py`
- Modify: `scripts/colab_g2_dataset_generator.py`
- Create: `tests/test_v3_rlbench_attempt.py`

**Interfaces:**
- Consumes: raw RLBench observations/actions, `GenerationJob`, approved binding, `assemble_episode_v2`, `assemble_attempt_record`, `write_episode_archive`, `write_training_episode_layout`.
- Produces: `RawAttempt`, `MaterializedAttempt`, `materialize_raw_attempt`, `write_closed_attempt_result`.

- [ ] **Step 1: Write failing timeline and valid-failure tests with fake observations**

```python
def test_valid_failure_materializes_episode_and_keeps_t_plus_one(tmp_path):
    raw = fake_raw_attempt(success=False, observations=5, actions=4)
    materialized = materialize_raw_attempt(raw, make_job(), approved_binding())
    assert materialized.outcome == "valid_failure"
    assert materialized.episode_record["provenance"]["episode_id"] == "episode-1"
    assert len(materialized.episode_record["observations"]) == 5
    assert len(materialized.episode_record["transitions"]) == 4

def test_invalid_observation_is_attempt_with_null_episode_id():
    raw = fake_raw_attempt(success=False, observations=1, actions=0)
    materialized = materialize_raw_attempt(raw, make_job(), approved_binding())
    assert materialized.outcome == "invalid_observation"
    assert materialized.attempt_record["episode_id"] is None
```

- [ ] **Step 2: Run the materialization tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_rlbench_attempt.py -q`

Expected: import failure for `rlbench_attempt`.

- [ ] **Step 3: Implement reusable raw/materialized records**

```python
@dataclass(frozen=True)
class RawAttempt:
    observations: Sequence[Any]
    actions: Sequence[np.ndarray]
    scene_states: Sequence[dict[str, Any]]
    collision_events: Sequence[dict[str, Any]]
    sim_time_s: float
    predicates_ok: bool
    terminal_reason: str
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None

@dataclass(frozen=True)
class MaterializedAttempt:
    outcome: str
    episode_record: dict[str, Any] | None
    attempt_record: dict[str, Any] | None
    auxiliary: dict[str, Any]
```

`materialize_raw_attempt` downsamples measured wrist point clouds, constructs
boundary-aligned `ExecutedTransition` records from achieved simulator timing,
preserves RGB/depth/masks/joints/end-effector/gripper/object states, and returns
`valid_failure` whenever a valid prefix has at least one transition. It uses
`invalid_observation` only when the required measured prefix cannot be formed.

- [ ] **Step 4: Extract raw execution from the legacy generator without changing v1 behavior**

Move the raw-observation/action capture loop from `collect_single_episode` into
the exact public signature
`execute_raw_attempt(task: Any, env: Any, spec: Mapping[str, Any]) -> RawAttempt`.

Keep `collect_single_episode -> tuple[dict[str, Any], dict[str, Any]]` as a
compatibility wrapper: call `execute_raw_attempt`, raise on a false predicate as
before for v1 callers, and materialize successful output through the shared
helper. Correct its stale return annotation from a three-tuple to a two-tuple.

- [ ] **Step 5: Implement closed result writing**

Implement the exact signature
`write_closed_attempt_result(materialized: MaterializedAttempt, job: GenerationJob, result_dir: str | Path) -> WorkerResult`.

Write to `result_dir.with_name(result_dir.name + ".partial")`; episode outcomes
use `write_episode_archive` and `write_training_episode_layout`, attempt outcomes
use `assemble_attempt_record` plus retained safe prefix arrays. Generate SHA256
for every file, fsync, atomically rename to `result_dir`, then return a
`WorkerResult`.

- [ ] **Step 6: Run compatibility and new tests**

Run:

```bash
.venv/bin/python -B -m pytest \
  tests/test_v3_rlbench_attempt.py \
  tests/test_colab_generator_manifest_gate.py \
  tests/test_v3_batch.py tests/test_v3_expert.py -q
```

Expected: PASS, zero skipped.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/icgs/data/collection/v3/rlbench_attempt.py \
  scripts/colab_g2_dataset_generator.py tests/test_v3_rlbench_attempt.py
git commit -m "feat(data): materialize full-fidelity v3 attempts"
```

### Task 3: Central planners, dynamic allocation and resume

**Files:**
- Create: `src/icgs/data/collection/v3/distributed_planner.py`
- Create: `tests/test_v3_distributed_planner.py`

**Interfaces:**
- Consumes: approved rows, remote/local manifest, `AttemptPlanner`, `QuotaCounts`, `quota_for_program`, `next_episode_kind`, `quota_met`.
- Produces: `DistributedPlanner`, `PlannerSnapshot`, deterministic `next_job` and `record_result`.

- [ ] **Step 1: Write failing centralized uniqueness and quota-allocation tests**

```python
def test_two_hundred_job_fill_has_unique_signatures_and_ids():
    planner = make_distributed_planner(empty_manifest())
    jobs = [planner.next_job() for _ in range(200)]
    assert len({job.job_id for job in jobs}) == 200
    assert len({job.plan.randomization["scene_signature"] for job in jobs}) == 200

def test_valid_failure_counts_only_for_perturbed_quota():
    planner = make_distributed_planner(empty_manifest(), programs=("T01",))
    nominal = planner.next_job()
    planner.record_result(make_result(nominal, "valid_failure"))
    assert planner.counts("T01").nominal_successes == 0
    resumed = make_distributed_planner(
        manifest_with_nominal_successes("T01", count=200), programs=("T01",)
    )
    perturbed = resumed.next_job()
    resumed.record_result(make_result(perturbed, "valid_failure"))
    assert resumed.counts("T01").perturbed_valid_attempts == 1
```

- [ ] **Step 2: Run planner tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_planner.py -q`

Expected: import failure for `distributed_planner`.

- [ ] **Step 3: Implement planner construction and snapshot**

Implement these exact public signatures:
`DistributedPlanner.from_manifest(run: RunConfig, approved_rows: Mapping[str, Mapping[str, Any]], manifest: Mapping[str, Any], inflight_jobs: Sequence[GenerationJob] = ()) -> DistributedPlanner`,
`next_job() -> GenerationJob | None`,
`record_result(result: WorkerResult, provenance: Mapping[str, Any]) -> None`,
`counts(program_id: str) -> QuotaCounts`,
`quota_complete() -> bool`, and `snapshot() -> PlannerSnapshot`.

`from_manifest` calls `counts_from_manifest`, remembers every manifest row and
in-flight plan in each program planner, and advances episode IDs beyond all
episode/attempt IDs. `next_job` selects the greatest normalized deficit and uses
split round-robin to break ties.

- [ ] **Step 4: Add resume and attempt-cap tests**

```python
def test_resume_never_reuses_remote_or_inflight_signature():
    resumed = DistributedPlanner.from_manifest(
        run_config(), rows(), manifest_with_one_episode(),
        inflight_jobs=(make_inflight_job(),),
    )
    new = resumed.next_job()
    assert new.plan.randomization["scene_signature"] not in known_signatures()

def test_attempt_cap_yields_incomplete_not_complete():
    planner = planner_at_cap_without_quota()
    assert planner.next_job() is None
    assert planner.quota_complete() is False
    assert planner.terminal_status() == "INCOMPLETE"
```

- [ ] **Step 5: Run planner tests**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_planner.py -q`

Expected: PASS, zero skipped.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/icgs/data/collection/v3/distributed_planner.py \
  tests/test_v3_distributed_planner.py
git commit -m "feat(data): centralize v3 quota planning"
```

### Task 4: Closed-result validation and manifest ingestion

**Files:**
- Create: `src/icgs/data/collection/v3/distributed_validation.py`
- Create: `tests/test_v3_distributed_validation.py`

**Interfaces:**
- Consumes: `GenerationJob`, `WorkerResult`, episode/attempt JSON, file hashes, `validate_episode_v2`, split reports.
- Produces: `ValidatedResult`, `validate_closed_result`, `ingest_validated_result`.

- [ ] **Step 1: Write failing success, valid-failure and crash validation tests**

```python
@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_episode_outcomes_require_complete_timeline_and_hashes(tmp_path, outcome):
    job, result = write_fixture(tmp_path, outcome=outcome, transitions=3, observations=4)
    validated = validate_closed_result(job, result)
    assert validated.outcome == outcome
    assert validated.episode_entry["outcome"] == outcome

def test_crash_rejects_non_null_episode_id(tmp_path):
    job, result = write_fixture(tmp_path, outcome="simulator_crash", episode_id="episode-1")
    with pytest.raises(ValueError, match="episode_id must be null"):
        validate_closed_result(job, result)
```

- [ ] **Step 2: Run validation tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_validation.py -q`

Expected: import failure for `distributed_validation`.

- [ ] **Step 3: Implement exact validation and immutable ingestion**

```python
@dataclass(frozen=True)
class ValidatedResult:
    job: GenerationJob
    result: WorkerResult
    provenance: dict[str, Any]
    episode_entry: dict[str, Any] | None
    attempt_entry: dict[str, Any] | None
```

Implement exact functions
`validate_closed_result(job: GenerationJob, result: WorkerResult) -> ValidatedResult`
and
`ingest_validated_result(manifest: Mapping[str, Any], validated: ValidatedResult) -> dict[str, Any]`.

Recompute every file hash, validate exact timeline lengths, outcome/ID rules,
protocol IDs, calibration, split/subset, scene signature and asset instance.
Return a new manifest object; refuse different bytes for an existing immutable
ID. Use `split_disjointness_report` after insertion and reject leakage.

- [ ] **Step 4: Add duplicate and leakage refusal tests**

Test duplicate episode IDs, duplicate scene signatures, cross-role asset
instances, missing modality declarations and mismatched checksums. Assert the
input manifest bytes remain unchanged after every rejection.

- [ ] **Step 5: Run validation tests**

Run:

```bash
.venv/bin/python -B -m pytest tests/test_v3_distributed_validation.py \
  tests/test_primary_v3.py tests/test_v3_batch.py -q
```

Expected: PASS, zero skipped.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/icgs/data/collection/v3/distributed_validation.py \
  tests/test_v3_distributed_validation.py
git commit -m "feat(data): validate distributed v3 results"
```

### Task 5: Five-minute verified Hugging Face publisher

**Files:**
- Create: `src/icgs/data/collection/v3/distributed_publication.py`
- Create: `tests/test_v3_distributed_publication.py`

**Interfaces:**
- Consumes: ingested jobs, local manifest/receipts, remote manifest bytes, `HfApi` supplied by caller.
- Produces: `HuggingFaceBatchPublisher`, `PublicationReceipt`, conflict-safe commit plans.

- [ ] **Step 1: Write failing cadence and retry tests**

```python
def test_publish_due_only_after_300_seconds_or_force():
    publisher = make_publisher(last_success_s=100.0)
    assert publisher.publish_due(now_s=399.9, force=False) is None
    assert publisher.publish_due(now_s=400.0, force=False).job_ids == ("job-1",)

def test_failed_commit_keeps_jobs_unpublished():
    api = FakeApi(fail_create_commit=True)
    publisher = make_publisher(api=api)
    with pytest.raises(RuntimeError, match="429"):
        publisher.publish_due(now_s=400.0, force=False)
    assert publisher.pending_job_ids() == ("job-1",)
```

- [ ] **Step 2: Run publication tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_publication.py -q`

Expected: import failure for `distributed_publication`.

- [ ] **Step 3: Implement batch planning and three-way identity merge**

Implement exact `HuggingFaceBatchPublisher` signatures:
`__init__(run: RunConfig, api: Any, token: str, queue: FilesystemJobQueue) -> None`,
`pending_job_ids() -> Sequence[str]`,
`plan_batch(local_manifest: Mapping[str, Any], remote_manifest: Mapping[str, Any]) -> PublicationPlan`, and
`publish_due(*, now_s: float, force: bool, complete: bool = False) -> PublicationReceipt | None`.

Build `CommitOperationAdd` only for closed files under `primary_v3/`. Merge rows
by immutable ID; equal rows are idempotent, unequal rows raise before API calls.
Include the manifest and resume receipt in every nonempty commit. Store token
only on the publisher object and never serialize/repr it.

- [ ] **Step 4: Implement remote hash verification tests**

Use `FakeApi`/fake download bytes to assert that a successful commit is not
marked published until every remote hash matches. Test `force=True` final flush
before 300 seconds.

- [ ] **Step 5: Run publisher tests**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_publication.py -q`

Expected: PASS, zero skipped.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/icgs/data/collection/v3/distributed_publication.py \
  tests/test_v3_distributed_publication.py
git commit -m "feat(data): publish verified v3 batches"
```

### Task 6: Credential-free persistent worker

**Files:**
- Create: `scripts/colab_v3_distributed_worker.py`
- Create: `tests/test_v3_distributed_worker.py`

**Interfaces:**
- Consumes: `FilesystemJobQueue`, `execute_raw_attempt`, `materialize_raw_attempt`, `write_closed_attempt_result`.
- Produces: `run_worker(worker_id, queue, env_factory, task_loader)` and CLI worker loop.

- [ ] **Step 1: Write failing worker claim/result/credential tests**

```python
def test_worker_claims_exact_job_and_publishes_valid_failure(tmp_path):
    queue = queue_with_one_job(tmp_path)
    run_worker("007", queue, fake_env_factory(valid_failure=True), fake_task_loader, once=True)
    ready = queue.iter_ready()
    assert len(ready) == 1
    assert ready[0].outcome == "valid_failure"

def test_worker_source_has_no_hf_token_or_api_access():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text()
    assert ".icgs_hf_token" not in text
    assert "HfApi" not in text
    assert "huggingface_hub" not in text
```

- [ ] **Step 2: Run worker tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_worker.py -q`

Expected: missing worker script/module.

- [ ] **Step 3: Implement the persistent loop**

Implement the exact signature
`run_worker(worker_id: str, queue: FilesystemJobQueue, env_factory: Callable[[GenerationJob], Any], task_loader: Callable[[str], Any], *, once: bool = False, idle_poll_s: float = 1.0) -> int`.

Validate worker ID `000..199`, write heartbeat before/after each state change,
claim atomically, execute exactly the serialized plan, always close the RLBench
environment, publish one closed result and continue. Catch operational errors
into `simulator_crash` results without changing job identity.

- [ ] **Step 4: Add deterministic display CLI**

CLI arguments are `--worker-id`, `--run-root`, `--approved-manifest` and
`--once`. The launcher, not the worker, wraps the command with:

```bash
xvfb-run --server-num "$((200 + worker_id))" -s "-screen 0 1280x1024x24" \
  /content/icgs-data-env/bin/python -B /content/ICGS/scripts/colab_v3_distributed_worker.py \
  --worker-id "$worker_id" --run-root /content/icgs-primary-v3-run \
  --approved-manifest /content/ICGS/artifacts/composition/approved_composition_manifest_v3.json
```

- [ ] **Step 5: Run worker and legacy tests**

Run:

```bash
.venv/bin/python -B -m pytest tests/test_v3_distributed_worker.py \
  tests/test_v3_rlbench_attempt.py tests/test_v3_expert.py -q
```

Expected: PASS, zero skipped.

- [ ] **Step 6: Commit Task 6**

```bash
git add scripts/colab_v3_distributed_worker.py \
  tests/test_v3_distributed_worker.py
git commit -m "feat(data): add credential-free v3 workers"
```

### Task 7: Coordinator, watchdog and exact-200 launcher

**Files:**
- Create: `scripts/colab_v3_distributed_coordinator.py`
- Create: `scripts/colab_v3_distributed_watchdog.py`
- Create: `scripts/colab_v3_distributed_launch.py`
- Create: `tests/test_v3_distributed_control.py`

**Interfaces:**
- Consumes: queue, planner, validator, publisher, token path and run config.
- Produces: coordinator loop, watchdog loop, preflight/launch CLI and progress receipts.

- [ ] **Step 1: Write failing coordinator ingestion and final-flush tests**

```python
def test_coordinator_retains_valid_failure_and_refills_queue(tmp_path):
    control = make_control_plane(tmp_path, ready_outcome="valid_failure")
    control.tick(now_s=10.0)
    assert control.manifest()["episodes"][0]["outcome"] == "valid_failure"
    assert control.queue.counts().pending > 0

def test_complete_quota_forces_final_publish_and_leaves_session_policy_alive(tmp_path):
    control = make_control_plane(tmp_path, quota_complete=True, last_publish_s=9.0)
    control.tick(now_s=10.0)
    assert control.publisher.calls[-1]["force"] is True
    assert control.run_status() == "COMPLETE"
    assert control.stop_session_requested is False
```

- [ ] **Step 2: Run control tests and verify RED**

Run: `.venv/bin/python -B -m pytest tests/test_v3_distributed_control.py -q`

Expected: missing control scripts/module.

- [ ] **Step 3: Implement coordinator tick loop**

Implement:

Implement exact `CoordinatorControlPlane` signatures:
`open(run_config_path: str | Path, *, api_factory: Callable[[], Any]) -> CoordinatorControlPlane`,
`tick(*, now_s: float | None = None) -> None`, and
`run_forever(*, poll_s: float = 1.0) -> int`.

Each tick acquires a single-owner lock, writes heartbeat, recovers stale claims,
validates/ingests every ready result, refills to 400 pending jobs, publishes when
due, writes a sanitized progress receipt and sets terminal state when complete or
attempt-capped.

- [ ] **Step 4: Implement watchdog process inventory**

The watchdog reads PID files and `/proc/<pid>/cmdline`, verifies the expected run
ID before signaling or replacing a process, and maintains one coordinator plus
worker IDs `000..199`. It never starts a second live coordinator lock owner.

- [ ] **Step 5: Implement preflight and launcher CLI**

`colab_v3_distributed_launch.py` exposes:

```text
--run-root /content/icgs-primary-v3-run
--workers 200
--publish-interval-s 300
--hf-repo 33bit/icgs
--hf-subfolder primary_v3
--approved-manifest /content/ICGS/artifacts/composition/approved_composition_manifest_v3.json
--preflight-only
--smoke-workers N
--jobs-per-worker N
--publication-enabled
--detach
```

Reject any full launch when workers is not 200, interval is not 300, token/auth
is absent, any fresh 36-program smoke row is not PASS, any smoke row is SKIPPED,
or remote manifest validation fails. Write `control/run.json` before starting
processes and return coordinator/watchdog/worker PID receipts.

- [ ] **Step 6: Add exact process-count and no-stop tests**

Mock `subprocess.Popen` and assert exactly 200 fixed-display worker commands, one
coordinator and one watchdog. Assert no source contains `colab stop` and the
launcher does not register a completion shutdown callback.

- [ ] **Step 7: Run control tests**

Run:

```bash
.venv/bin/python -B -m pytest tests/test_v3_distributed_control.py \
  tests/test_v3_distributed_queue.py tests/test_v3_distributed_planner.py \
  tests/test_v3_distributed_validation.py tests/test_v3_distributed_publication.py \
  tests/test_v3_distributed_worker.py -q
```

Expected: PASS, zero skipped.

- [ ] **Step 8: Commit Task 7**

```bash
git add scripts/colab_v3_distributed_coordinator.py \
  scripts/colab_v3_distributed_watchdog.py \
  scripts/colab_v3_distributed_launch.py \
  tests/test_v3_distributed_control.py
git commit -m "feat(data): orchestrate 200 v3 workers"
```

### Task 8: Repository validation and operator documentation

**Files:**
- Modify: `docs/audits/2026-09-20-icgs-v3-generation.md`
- Create: `docs/audits/2026-09-21-icgs-v3-full-launch.md`
- Modify: `docs/plans/active/icgs-primary-v3-phase1.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: implemented CLI, receipt schemas and validation commands.
- Produces: exact operator commands, stop/resume semantics and launch evidence template.

- [ ] **Step 1: Document exact local and Colab commands**

Add the complete sequence:

```bash
python3 -B scripts/validate_fast.py
.venv/bin/python -B -m pytest tests/test_v3_distributed_*.py -q
colab new -s icgs-primary-v3-full --tpu v6e1
colab upload -s icgs-primary-v3-full <clean-source-archive> /content/ICGS.tgz
colab upload -s icgs-primary-v3-full .env /content/.env.local
colab exec -s icgs-primary-v3-full --timeout 1800 -f scripts/colab_provision_ephemeral.py
```

Document that `.env.local` is converted to `/content/.icgs_hf_token` and deleted,
that workers never receive the token, and that the session is intentionally left
running.

- [ ] **Step 2: Update active-plan gate status without rewriting history**

Record the earlier place-height failure as historical evidence and state that
only the fresh 36-program smoke for the launch may close the scale gate. Do not
mark full generation complete before the terminal receipt exists.

- [ ] **Step 3: Run all local verification**

Run:

```bash
python3 -B scripts/validate_fast.py
python3 -B -S scripts/validate_fast.py
.venv/bin/python -B -m pytest tests/test_v3_distributed_*.py \
  tests/test_primary_v3.py tests/test_v3_batch.py tests/test_v3_expert.py \
  tests/test_colab_generator_manifest_gate.py -q
git diff --check
```

Expected: all selected tests PASS, zero skipped; both L0 runs PASS 19/19.

- [ ] **Step 4: Commit Task 8**

```bash
git add docs/README.md docs/audits/2026-09-20-icgs-v3-generation.md \
  docs/audits/2026-09-21-icgs-v3-full-launch.md \
  docs/plans/active/icgs-primary-v3-phase1.md
git commit -m "docs(data): document distributed v3 launch"
```

### Task 9: Provision, validate and launch the authorized full run

**Files:**
- Update with real evidence: `docs/audits/2026-09-21-icgs-v3-full-launch.md`

**Interfaces:**
- Consumes: committed clean implementation from Tasks 1–8, Colab CLI, `.env` token, `33bit/icgs`.
- Produces: live v6e-1 session, 200 workers, one coordinator, one watchdog, verified first HF commit and launch receipt.

- [ ] **Step 1: Verify clean committed source and create the session**

Run:

```bash
git status --porcelain=v1 -uall
git rev-parse HEAD
colab new -s icgs-primary-v3-full --tpu v6e1
colab status -s icgs-primary-v3-full
```

Expected: empty Git status and Colab `Hardware: V6E1`, `Status: IDLE`.

- [ ] **Step 2: Upload clean source and credential material**

Create a source archive from `git archive HEAD`, upload it, upload `.env` only as
`/content/.env.local`, run `colab_remote_token_setup.py`, and verify the source
env file is deleted and the token file mode is `0600` without printing token
contents.

- [ ] **Step 3: Provision pinned simulator dependencies**

Run the provisioning entry point with a 1,800-second timeout. Record Python,
CoppeliaSim SHA256, PyRep revision, RLBench revision, CPU count, memory and disk
capacity in the launch audit.

- [ ] **Step 4: Run mandatory parity/build/36-program smoke**

Run the launcher with `--preflight-only`. Record 36 retained episode outcomes
(`success` or `valid_failure`), zero crash/invalid/SKIPPED, timeline counts and
place-predicate distances. If any gate fails, keep the
session alive, record FAIL and stop before worker launch.

- [ ] **Step 5: Run publication-disabled queue smokes**

Run a 2-worker one-job-per-worker smoke, then a 200-worker one-job-per-worker
smoke. Verify fixed display IDs, exactly 200 completed closed results, no
duplicate signatures/IDs, valid-failure retention and zero credential access by
workers.

- [ ] **Step 6: Run and verify one small publication batch**

Enable publication for a bounded two-job run under a `primary_v3/preflight/`
receipt path, wait for the coordinator commit, then download the committed files
at the returned revision and compare every SHA256. This batch is real data only
if it passes the same episode/attempt validators; otherwise publish only the
closed preflight receipt and keep data out of the canonical manifest.

- [ ] **Step 7: Launch the full detached workload**

Run:

```bash
/content/icgs-data-env/bin/python -B \
  /content/ICGS/scripts/colab_v3_distributed_launch.py \
  --run-root /content/icgs-primary-v3-run \
  --workers 200 --publish-interval-s 300 \
  --hf-repo 33bit/icgs --hf-subfolder primary_v3 \
  --approved-manifest /content/ICGS/artifacts/composition/approved_composition_manifest_v3.json \
  --publication-enabled --detach
```

- [ ] **Step 8: Verify live process and queue state**

Read the launch receipt and assert one coordinator, one watchdog and exactly 200
worker slots with fresh heartbeats. Verify run status `RUNNING`, queue depth no
greater than 400, remote manifest identity and no `colab stop` action.

- [ ] **Step 9: Verify the first scheduled five-minute commit**

Wait for the first `publication_receipt.json` after at least 300 seconds. Verify
the remote revision, hashes, manifest count, retained valid failures and per-
program quota progress. Record PASS/FAIL and the exact revision in the launch
audit.

- [ ] **Step 10: Commit the launch receipt without stopping the session**

Update only the audit with session name, hardware, revisions, commands,
PASS/FAIL/SKIPPED/NOT RUN results, PIDs, first commit revision and remaining
risk. Commit locally:

```bash
git add docs/audits/2026-09-21-icgs-v3-full-launch.md
git commit -m "docs(data): record primary v3 full launch"
```

Do not call `colab stop`; report the live session and monitoring paths to the
owner.
