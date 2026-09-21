from __future__ import annotations

from pathlib import Path

from scripts.colab_v3_distributed_launch import build_worker_commands


def test_launcher_builds_exactly_200_fixed_display_workers():
    commands = build_worker_commands(
        workers=200,
        run_root="/content/run",
        approved_manifest="/content/manifest.json",
    )
    assert len(commands) == 200
    assert commands[0][commands[0].index("--server-num") + 1] == "200"
    assert commands[-1][commands[-1].index("--server-num") + 1] == "399"
    assert all("-a" not in command for command in commands)


def test_launcher_rejects_non_200_workers():
    import pytest
    with pytest.raises(ValueError, match="workers must be 200"):
        build_worker_commands(workers=199, run_root="/content/run", approved_manifest="/content/manifest.json")


def test_no_stop_call_in_control_sources():
    for name in (
        "scripts/colab_v3_distributed_launch.py",
        "scripts/colab_v3_distributed_coordinator.py",
        "scripts/colab_v3_distributed_watchdog.py",
    ):
        assert "colab stop" not in Path(name).read_text(encoding="utf-8")
