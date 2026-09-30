"""Geometry signatures of task objects, so asset overlap does not depend on names.

For each task (after one reset, strategy 0 or the first enabled strategy), every
non-generic shape in the task tree plus its graspables gets a signature: mesh
vertex count and bounding-box extent ratios (scale-invariant, because the
always-on object_size factor rescales some containers every reset); absolute
extents are recorded alongside. Two objects
with the same signature are treated as the same asset regardless of TTM name.

  xvfb-run -a python -B scripts/robohiman_asset_inventory.py --task TASK --out DIR/TASK.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from icgs.environments.robohiman.session import RoboHiManSession

_GENERIC = re.compile(r"^(Panda_|ResizableFloor|Wall|diningTable|workspace|boundary_root|Dummy|cam_)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    session = None
    for strategy in (0, 2):
        try:
            session = RoboHiManSession(args.task, strategy_index=strategy, cameras=(), image_size=(64, 64))
            session.build_config()
            break
        except ValueError:
            continue
    session.launch()
    try:
        np.random.seed(1)
        session.reset(0, attempts=10)
        graspable = {o.get_name() for o in session.task_obj.get_graspable_objects()}
        inventory = {}
        for shape in session.task_shapes():
            name = shape.get_name()
            if _GENERIC.match(name):
                continue
            bbox = np.asarray(shape.get_bounding_box(), dtype=np.float64)
            extents = sorted(np.round((bbox[1::2] - bbox[0::2]) * 1000.0).astype(int).tolist())
            try:
                vertices = int(len(shape.get_mesh_data()[0]))
            except Exception:
                vertices = -1
            # Scale-invariant: the always-on object_size factor rescales some containers per reset.
            ratios = [round(e / max(extents), 2) if max(extents) else 0.0 for e in extents]
            signature = hashlib.sha256(json.dumps([ratios, vertices]).encode()).hexdigest()[:12]
            inventory[name] = {"extents_mm_sorted": extents, "vertices": vertices, "signature": signature,
                               "graspable": name in graspable}
        Path(args.out).write_text(json.dumps({"task": args.task, "objects": inventory}, indent=1, sort_keys=True))
        print(json.dumps({"task": args.task, "objects": len(inventory)}))
    finally:
        session.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
