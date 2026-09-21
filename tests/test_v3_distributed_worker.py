from __future__ import annotations

from pathlib import Path

from icgs.data.collection.v3.distributed_contracts import GenerationJob
from icgs.data.collection.v3.distributed_queue import FilesystemJobQueue


def test_worker_source_has_no_hf_token_or_api_access():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text(encoding="utf-8")
    assert ".icgs_hf_token" not in text
    assert "HfApi" not in text
    assert "huggingface_hub" not in text


def test_worker_command_uses_fixed_display_number():
    text = Path("scripts/colab_v3_distributed_worker.py").read_text(encoding="utf-8")
    assert "200 + worker_id" in text
    assert "xvfb-run -a" not in text
