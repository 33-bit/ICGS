from __future__ import annotations

from pathlib import Path

from scripts.colab_v3_distributed_launch import build_worker_commands, validate_smoke_receipt


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


def test_launcher_exports_pinned_simulator_environment_to_workers():
    text = Path("scripts/colab_v3_distributed_launch.py").read_text(encoding="utf-8")
    assert "COPPELIASIM_ROOT" in text
    assert "LD_LIBRARY_PATH" in text
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" in text
    assert "PYTHONPATH" in text


def test_launcher_requires_real_code_revision_for_run_contract():
    text = Path("scripts/colab_v3_distributed_launch.py").read_text(encoding="utf-8")
    assert '"--code-revision"' in text
    assert '"code_revision": "unknown"' not in text


def test_launch_smoke_accepts_retained_valid_failure(tmp_path):
    import json

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "valid_failure" if index == 0 else "success",
                "timeline_ok": True,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_smoke_receipt(path, expected_program_ids=programs)


def test_launch_smoke_rejects_crash_or_missing_timeline(tmp_path):
    import json
    import pytest

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "simulator_crash" if index == 0 else "success",
                "timeline_ok": index != 1,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="blocking smoke outcomes"):
        validate_smoke_receipt(path, expected_program_ids=programs)
