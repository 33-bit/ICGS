"""Build generation procedural RLBench task models from the step-driven compiler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL

def build_models(
    program_ids: list[str] | None = None,
    *,
    rlbench_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict:
    from pyrep import PyRep
    from pyrep.const import PrimitiveShape
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.panda import Panda
    from rlbench import environment as rl_environment
    from rlbench.backend.const import TTT_FILE
    import numpy as np

    installed_rlbench_root = Path(rl_environment.__file__).resolve().parent.parent
    selected_rlbench_root = Path(
        rlbench_root
        or os.environ.get("ICGS_RLBENCH_ROOT")
        or installed_rlbench_root
    ).expanduser().resolve()
    if not (selected_rlbench_root / "rlbench").is_dir():
        raise ValueError(
            "rlbench_root must contain the installed rlbench package: "
            f"{selected_rlbench_root}"
        )
    selected_output_root = Path(
        output_root
        or os.environ.get("ICGS_GENERATION_BUILD_ROOT")
        or (Path.cwd() / "generation-tasks")
    ).expanduser().resolve()

    compiled = compile_generation_catalog()
    wanted = list(program_ids or list(compile_generation_catalog()))
    tasks_dir = selected_rlbench_root / "rlbench" / "tasks"
    ttm_dir = selected_rlbench_root / "rlbench" / "task_ttms"
    for directory, name in (
        (tasks_dir, "RLBench task directory"),
        (ttm_dir, "RLBench task model directory"),
        (selected_output_root, "build output directory"),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"{name} cannot be created: {directory}") from exc
        if not os.access(directory, os.W_OK):
            raise ValueError(f"{name} must be writable: {directory}")

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
            ttm_path = ttm_dir / f"{spec.module}.ttm"
            root.save_model(str(ttm_path))
            root.remove()
            py_path = tasks_dir / f"{spec.module}.py"
            py_path.write_text(spec.py_source, encoding="utf-8")
            built[program_id] = {"module": spec.module, "ttm": str(ttm_path), "py": str(py_path)}
            print(f"[{program_id}] built {spec.module}", flush=True)
    finally:
        sim.stop()
        sim.shutdown()
    (selected_output_root / "built_suite_manifest.json").write_text(
        json.dumps({"tasks": built}, indent=2) + "\n",
        encoding="utf-8",
    )
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", nargs="+", default=list(GENERATION_PROTOCOL.pilot_program_ids))
    parser.add_argument(
        "--rlbench-root",
        default=os.environ.get("ICGS_RLBENCH_ROOT"),
        help="RLBench checkout root (defaults to the installed RLBench package)",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("ICGS_GENERATION_BUILD_ROOT"),
        help="directory for the build manifest (defaults to ./generation-tasks)",
    )
    args = parser.parse_args()
    built = build_models(
        args.programs,
        rlbench_root=args.rlbench_root,
        output_root=args.output_root,
    )
    print("GENERATION_MODELS_BUILT", len(built), flush=True)


if __name__ == "__main__":
    main()
