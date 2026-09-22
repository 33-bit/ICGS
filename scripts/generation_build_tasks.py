"""Build generation procedural RLBench task models from the step-driven compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/content/ICGS")
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL

ROOT = Path("/content/icgs-ephemeral")
RLBench = ROOT / "RLBench"
TASKS_DIR = RLBench / "rlbench" / "tasks"
TTM_DIR = RLBench / "rlbench" / "task_ttms"
OUT = ROOT / "generation-tasks"


def build_models(program_ids: list[str] | None = None) -> dict:
    from pyrep import PyRep
    from pyrep.const import PrimitiveShape
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.panda import Panda
    from rlbench import environment as rl_environment
    from rlbench.backend.const import TTT_FILE
    import numpy as np

    compiled = compile_generation_catalog()
    wanted = list(program_ids or list(compile_generation_catalog()))
    TTM_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    base_scene = Path(rl_environment.__file__).parent / TTT_FILE
    sim = PyRep()
    sim.launch(str(base_scene), headless=True)
    sim.start()
    tip_quaternion = np.asarray(Panda().get_tip().get_quaternion(), dtype=float)
    built = {}
    try:
        for program_id in wanted:
            spec = compiled[program_id]
            root = Dummy.create()
            root.set_name(spec.module)
            root.set_position([0.25, 0.0, 0.752])
            for name, (position, size, color) in spec.objects.items():
                is_marker = (
                    name.startswith("target")
                    or name.endswith("_wp")
                    or name.endswith("_target")
                )
                if is_marker:
                    marker = Dummy.create(size=0.01)
                    marker.set_name(name)
                    marker.set_position(position)
                    marker.set_parent(root, keep_in_place=True)
                    continue
                is_movable = (
                    name.startswith("object")
                    or name.startswith("blocker")
                    or name.startswith("spacer")
                    or "handle" in name
                )
                is_support = name in {"pad", "tray", "holder", "drawer"}
                shape = Shape.create(
                    PrimitiveShape.CUBOID,
                    size=size,
                    mass=0.05,
                    static=not is_movable,
                    respondable=is_movable or is_support,
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
            root.set_model_dynamic(True)
            ttm_path = TTM_DIR / f"{spec.module}.ttm"
            root.save_model(str(ttm_path))
            root.remove()
            py_path = TASKS_DIR / f"{spec.module}.py"
            py_path.write_text(spec.py_source, encoding="utf-8")
            built[program_id] = {"module": spec.module, "ttm": str(ttm_path), "py": str(py_path)}
            print(f"[{program_id}] built {spec.module}", flush=True)
    finally:
        sim.stop()
        sim.shutdown()
    (OUT / "built_suite_manifest.json").write_text(json.dumps({"tasks": built}, indent=2) + "\n", encoding="utf-8")
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", nargs="+", default=list(GENERATION_PROTOCOL.pilot_program_ids))
    args = parser.parse_args()
    built = build_models(args.programs)
    print("GENERATION_MODELS_BUILT", len(built), flush=True)


if __name__ == "__main__":
    main()
