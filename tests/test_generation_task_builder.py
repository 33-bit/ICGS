from pathlib import Path

import pytest


def test_verify_all_tasks_rejects_missing_stale_and_empty_assets(tmp_path):
    from scripts import generation_build_tasks as builder
    assert hasattr(builder, "verify_built_tasks"), "generation readiness must verify task assets"
    compiled = builder.compile_generation_catalog()
    tasks = tmp_path / "rlbench" / "tasks"
    models = tmp_path / "rlbench" / "task_ttms"
    tasks.mkdir(parents=True)
    models.mkdir()
    with pytest.raises(ValueError, match="missing"):
        builder.verify_built_tasks(tmp_path)
    for spec in compiled.values():
        (tasks / f"{spec.module}.py").write_text(spec.py_source)
        (models / f"{spec.module}.ttm").write_bytes(b"fixture model")
    assert set(builder.verify_built_tasks(tmp_path)) == set(compiled)
    spec = next(iter(compiled.values()))
    (tasks / f"{spec.module}.py").write_text("old source")
    with pytest.raises(ValueError, match="stale"):
        builder.verify_built_tasks(tmp_path)
    (tasks / f"{spec.module}.py").write_text(spec.py_source)
    (models / f"{spec.module}.ttm").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        builder.verify_built_tasks(tmp_path)


def test_all_programs_cli_builds_complete_catalog_without_changing_pilot_default(tmp_path, monkeypatch):
    from scripts import generation_build_tasks as builder
    import sys
    called = []
    def build(programs, **kwargs):
        called.extend(programs)
        return {key: {} for key in programs}
    monkeypatch.setattr(builder, "build_models", build)
    monkeypatch.setattr(sys, "argv", ["generation_build_tasks.py", "--all-programs", "--rlbench-root", str(tmp_path)])
    builder.main()
    assert set(called) == set(builder.compile_generation_catalog())


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
