import pytest

from icgs.data.collection.generation.capacity_probe import CapacityProbeConfig
from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig
from scripts import generation_capacity_probe


def _payload():
    return {
        "run_id": "capacity-test",
        "hf_subfolder": "validation/capacity-test",
        "max_total_jobs": 40,
        "max_runtime_s": 600,
        "stages": [
            {"name": "low", "worker_count": 8, "simulator_slots": 2, "max_jobs": 8, "worker_timeout_s": 180},
            {"name": "high", "worker_count": 16, "simulator_slots": 4, "max_jobs": 16, "worker_timeout_s": 180},
        ],
    }


def test_capacity_probe_config_accepts_bounded_stages():
    config = CapacityProbeConfig.from_dict(_payload())
    assert config.stages[1].worker_count == 16
    assert config.as_dict()["stages"][0]["max_jobs"] == 8


@pytest.mark.parametrize("field,value", [("max_total_jobs", 401), ("max_runtime_s", 5401)])
def test_capacity_probe_rejects_global_limits(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_stage_sum_over_global_cap():
    payload = _payload()
    payload["max_total_jobs"] = 10
    with pytest.raises(ValueError, match="exceed"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_non_validation_prefix():
    payload = _payload()
    payload["hf_subfolder"] = "generation/production"
    with pytest.raises(ValueError, match="validation"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_unsafe_stage_path():
    payload = _payload()
    payload["stages"][0]["name"] = "../escape"
    with pytest.raises(ValueError, match="stage name"):
        CapacityProbeConfig.from_dict(payload)


def test_capacity_probe_rejects_unsafe_run_identity():
    payload = _payload()
    payload["run_id"] = "../escape"
    with pytest.raises(ValueError, match="run_id"):
        CapacityProbeConfig.from_dict(payload)


def test_stage_process_cleanup_terminates_only_owned_process_groups(monkeypatch):
    calls = []

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise generation_capacity_probe.subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(generation_capacity_probe.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    generation_capacity_probe.stop_processes([Process()])
    assert calls == [(1234, generation_capacity_probe.signal.SIGTERM), (1234, generation_capacity_probe.signal.SIGKILL)]


def test_stage_planning_never_enqueues_past_stage_cap(tmp_path):
    class Planner:
        def __init__(self):
            self.calls = 0

        def next_job(self):
            self.calls += 1
            return type("Job", (), {"job_id": f"job-{self.calls}"})()

    class Queue:
        def __init__(self):
            self.jobs = []

        def enqueue(self, job):
            self.jobs.append(job.job_id)

    planner, queue = Planner(), Queue()
    generation_capacity_probe.enqueue_fixed_jobs(planner, queue, 8)
    assert planner.calls == 8
    assert queue.jobs == [f"job-{index}" for index in range(1, 9)]


def test_stage_runtime_disables_publication_and_rewrites_identity(tmp_path):
    payload = {
        "machine": {
            "repo_root": str(tmp_path / "repo"),
            "python_executable": str(tmp_path / "python"),
            "simulator_root": str(tmp_path / "sim"),
            "rlbench_root": str(tmp_path / "rlbench"),
            "display_base": 20,
            "display_width": 1024,
            "display_height": 768,
            "simulator_slots": 2,
            "worker_timeout_s": 1200,
        },
        "run": {
            "run_id": "base",
            "run_root": str(tmp_path / "base"),
            "worker_count": 2,
            "hf_subfolder": "validation/capacity-test",
            "publication_enabled": False,
            "validation_mode": True,
        },
    }
    base = GenerationRuntimeConfig.from_dict(payload)
    probe = CapacityProbeConfig.from_dict(_payload())
    runtime = generation_capacity_probe._stage_runtime(base, probe, probe.stages[1], tmp_path / "high")
    assert runtime.run.worker_count == 16
    assert runtime.machine.simulator_slots == 4
    assert runtime.run.run_id == "capacity-test-high"
    assert runtime.run.publication_enabled is False
    assert runtime.machine.worker_timeout_s == 180
    assert len(runtime.machine.worker_ids) == 16
