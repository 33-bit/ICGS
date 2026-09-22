from pathlib import Path


def test_task_models_enable_child_dynamics_for_contact_push():
    source = Path("scripts/generation_build_tasks.py").read_text(encoding="utf-8")
    assert "root.set_model_dynamic(True)" in source
    assert "root.set_model_dynamic(False)" not in source
