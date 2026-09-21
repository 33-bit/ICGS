"""Tests for the 36-composition primitive compiler and task specifications."""

from pathlib import Path
import py_compile
import tempfile
import pytest

from icgs.data.collection.primitive_compiler import compile_catalog


MANIFEST_PATH = Path("artifacts/composition/approved_composition_manifest.json")


def test_compiler_cardinality_and_splits():
    compiled = compile_catalog(MANIFEST_PATH)
    assert len(compiled) == 36

    train_tasks = [pid for pid, spec in compiled.items() if spec.split == "train"]
    dev_tasks = [pid for pid, spec in compiled.items() if spec.split == "development"]
    test_tasks = [pid for pid, spec in compiled.items() if spec.split == "test"]

    assert len(train_tasks) == 20
    assert len(dev_tasks) == 4
    assert len(test_tasks) == 12

    assert set(train_tasks) == {f"T{i:02d}" for i in range(1, 21)}
    assert set(dev_tasks) == {"V01", "V02", "V03", "V04"}
    assert set(test_tasks) == {
        "P1", "P2", "P3", "P4",
        "G1", "G2", "G3", "G4",
        "R1", "R2", "R3", "R4",
    }


def test_all_36_tasks_have_concrete_bundles():
    compiled = compile_catalog(MANIFEST_PATH)
    for pid, spec in compiled.items():
        assert spec.class_name.isidentifier(), f"{pid}: invalid class name {spec.class_name}"
        assert spec.module.isidentifier(), f"{pid}: invalid module name {spec.module}"
        assert len(spec.objects) >= 2, f"{pid}: task must have at least 2 objects"
        assert len(spec.conditions) >= 1, f"{pid}: task must have at least 1 success condition"
        assert len(spec.routine) >= 1, f"{pid}: task must have at least 1 executable routine step"

        # Check object geometry
        for obj_name, (pos, size, color) in spec.objects.items():
            assert len(pos) == 3, f"{pid}.{obj_name}: pos must be 3D"
            assert len(size) == 3, f"{pid}.{obj_name}: size must be 3D"
            assert len(color) == 3, f"{pid}.{obj_name}: color must be 3D"
            assert 0.10 <= pos[0] <= 0.55, f"{pid}.{obj_name}: x out of workspace"
            assert -0.35 <= pos[1] <= 0.35, f"{pid}.{obj_name}: y out of workspace"
            assert 0.70 <= pos[2] <= 1.10, f"{pid}.{obj_name}: z out of workspace"
            assert all(s > 0 for s in size), f"{pid}.{obj_name}: non-positive size"

        # Check conditions reference existing objects
        for obj_a, obj_b, tol in spec.conditions:
            assert obj_a in spec.objects, f"{pid}: condition obj {obj_a} not in objects"
            assert obj_b in spec.objects, f"{pid}: condition target {obj_b} not in objects"
            assert tol > 0, f"{pid}: condition tolerance must be positive"
            assert tol == 0.01, f"{pid}: compiler widened approved predicate tolerance"


def test_all_ordered_steps_map_to_executable_primitives():
    compiled = compile_catalog(MANIFEST_PATH)
    valid_primitive_types = {
        "pick_place",
        "lift",
        "reach",
        "push",
        "grasp_rotate",
        "place",
        "transport_through_aperture",
        "touch_retreat",
    }
    for pid, spec in compiled.items():
        assert len(spec.ordered_steps) >= 2, f"{pid}: expected multiple ordered steps"
        assert len(spec.routine) >= 1, f"{pid}: routine cannot be empty"
        for step_idx, step in enumerate(spec.routine):
            ptype = step.get("type")
            assert ptype in valid_primitive_types, f"{pid} step {step_idx}: unknown type {ptype}"


def test_generated_task_python_sources_compile():
    compiled = compile_catalog(MANIFEST_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        for pid, spec in compiled.items():
            tmp_py = Path(tmpdir) / f"{spec.module}.py"
            tmp_py.write_text(spec.py_source, encoding="utf-8")
            # Must compile cleanly
            py_compile.compile(str(tmp_py), doraise=True)
