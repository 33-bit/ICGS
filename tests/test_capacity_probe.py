import pytest

from icgs.data.collection.generation.capacity_probe import CapacityProbeConfig


def _payload():
    return {
        "run_id": "capacity-test",
        "hf_subfolder": "validation/capacity-test",
        "max_total_jobs": 40,
        "max_runtime_s": 600,
        "stages": [
            {"name": "low", "worker_count": 8, "simulator_slots": 2, "max_jobs": 8},
            {"name": "high", "worker_count": 16, "simulator_slots": 4, "max_jobs": 16},
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
