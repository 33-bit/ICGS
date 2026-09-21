"""Build all 36 isolated procedural RLBench task models and python classes.

Runs only inside the pinned Colab environment with PyRep/CoppeliaSim.
The generated tasks use parametric primitive assets, explicit names and
waypoint/pose expert control; they are custom ICGS tasks matching the
approved composition manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np

# Add src to path if running inside ICGS repo
_REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/content/ICGS")
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from icgs.data.collection.primitive_compiler import compile_catalog

ROOT = Path("/content/icgs-ephemeral")
RLBench = ROOT / "RLBench"
TASKS_DIR = RLBench / "rlbench" / "tasks"
TTM_DIR = RLBench / "rlbench" / "task_ttms"
OUT = ROOT / "icgs-g2"


def build_all_models(manifest_path: Path) -> dict[str, Any]:
    from pyrep import PyRep
    from pyrep.const import PrimitiveShape
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.panda import Panda
    from rlbench import environment as rl_environment
    from rlbench.backend.const import TTT_FILE

    TTM_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    compiled = compile_catalog(manifest_path)
    print(f"Compiled {len(compiled)} tasks from manifest: {manifest_path}", flush=True)

    base_scene = Path(rl_environment.__file__).parent / TTT_FILE
    sim = PyRep()
    sim.launch(str(base_scene), headless=True)
    sim.start()
    tip_quaternion = np.asarray(Panda().get_tip().get_quaternion(), dtype=float)

    built = {}
    try:
        for program_id, spec in compiled.items():
            root = Dummy.create()
            root.set_name(spec.module)
            root.set_position([0.25, 0.0, 0.752])

            for name, (position, size, color) in spec.objects.items():
                is_movable = (
                    name.startswith("object")
                    or name.startswith("blocker")
                    or name.startswith("spacer")
                    or "handle" in name
                )
                shape = Shape.create(
                    PrimitiveShape.CUBOID,
                    size=size,
                    mass=0.05,
                    static=not is_movable,
                    respondable=is_movable,
                    position=position,
                    color=color,
                )
                shape.set_name(name)
                shape.set_parent(root, keep_in_place=True)

            for index, position in enumerate(spec.waypoints):
                waypoint = Dummy.create(size=0.01)
                waypoint.set_name(f"waypoint{index}")
                waypoint.set_position(position)
                waypoint.set_quaternion(tip_quaternion)
                waypoint.set_parent(root, keep_in_place=True)

            root.set_model(True)
            root.set_model_dynamic(False)
            ttm_path = TTM_DIR / f"{spec.module}.ttm"
            root.save_model(str(ttm_path))
            root.remove()

            # Write .py file
            py_path = TASKS_DIR / f"{spec.module}.py"
            py_path.write_text(spec.py_source, encoding="utf-8")
            built[program_id] = {
                "class_name": spec.class_name,
                "module": spec.module,
                "ttm": str(ttm_path),
                "py": str(py_path),
            }
            print(f"[{program_id}] Built {spec.class_name} -> {ttm_path.name}, {py_path.name}", flush=True)
    finally:
        sim.stop()
        sim.shutdown()

    (OUT / "built_suite_manifest.json").write_text(
        json.dumps({"total_built": len(built), "tasks": built}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ALL_36_MODELS_BUILT_SUCCESSFULLY: {len(built)} tasks written", flush=True)
    return built


def main():
    parser = argparse.ArgumentParser(description="Build 36 procedural task models for RLBench")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "composition" / "approved_composition_manifest.json",
        help="Path to approved composition manifest",
    )
    args = parser.parse_args()
    build_all_models(args.manifest)


if __name__ == "__main__":
    main()
