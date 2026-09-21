from __future__ import annotations

from pathlib import Path

from icgs.data.collection.v3.distributed_contracts import GenerationJob
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


def test_simulator_slot_pool_honors_configured_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("ICGS_SIMULATOR_SLOTS", "2")
    from scripts.colab_v3_distributed_worker import SimulatorSlotPool

    pool = SimulatorSlotPool(tmp_path)
    assert pool.slot_count == 2
    with pool:
        assert pool.acquired_slot is not None
        assert pool.acquired_slot.is_file()


def test_worker_source_has_no_hf_token_or_api_access():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text(encoding="utf-8")
    assert ".icgs_hf_token" not in text
    assert "HfApi" not in text
    assert "huggingface_hub" not in text


def test_worker_command_uses_fixed_display_number():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text(encoding="utf-8")
    assert "200 + worker_id" in text
    assert "xvfb-run -a" not in text


def test_worker_invalid_result_hashes_the_candidate_directory():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text(encoding="utf-8")
    assert "file_sha256=_file_hashes(candidate)" in text
    assert "file_sha256=_file_hashes(result_dir)" not in text
