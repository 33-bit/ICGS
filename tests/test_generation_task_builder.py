from pathlib import Path


def test_task_models_enable_child_dynamics_for_contact_push():
    source = Path("scripts/generation_build_tasks.py").read_text(encoding="utf-8")
    assert "root.set_model_dynamic(True)" in source
    assert "root.set_model_dynamic(False)" not in source


def test_task_builder_uses_configurable_machine_paths():
    source = Path("scripts/generation_build_tasks.py").read_text(encoding="utf-8")
    assert "/content/icgs-ephemeral" not in source
    assert "--rlbench-root" in source
    assert "--output-root" in source


def test_task_builder_rejects_unwritable_task_directories():
    source = Path("scripts/generation_build_tasks.py").read_text(encoding="utf-8")
    assert "os.access" in source
    assert "must be writable" in source
