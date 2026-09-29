"""Construct a RoboHiMan task environment exactly as the upstream generators do.

The environment, action mode, observation config and variation-factor config
are built with the same upstream calls as
``colosseum/tools/dataset_generator_{atomic,compositional}.py``. The only
deliberate differences are recorded in ``provenance()``: numpy is seeded
explicitly (upstream workers call ``np.random.seed(None)``) and the full RNG
state before each reset is stored, and camera/image options are explicit.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from icgs.environments.robohiman.pins import NATIVE_CAMERAS, UPSTREAM, level_for, task_family


def _git_revision(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                              check=False, timeout=30).stdout.strip()
    try:
        return {"revision": run("rev-parse", "HEAD"),
                "dirty_paths": [line[3:] for line in run("status", "--porcelain").splitlines()]}
    except Exception as exc:  # pragma: no cover - provenance must not crash collection
        return {"revision": None, "error": type(exc).__name__}


def rng_state_array(state: tuple) -> np.ndarray:
    kind, keys, pos, has_gauss, cached = state
    if kind != "MT19937":
        raise ValueError("unexpected numpy RNG kind")
    return np.concatenate([np.asarray(keys, dtype=np.uint32),
                           np.array([pos, has_gauss], dtype=np.uint32)]).astype(np.uint32), float(cached)


def rng_state_from_array(array: np.ndarray, cached: float) -> tuple:
    array = np.asarray(array, dtype=np.uint32)
    return ("MT19937", array[:624].copy(), int(array[624]), int(array[625]), float(cached))


class RoboHiManSession:
    """One simulator process bound to one RoboHiMan task and collection strategy."""

    def __init__(self, task: str, *, strategy_index: int = 0, image_size: Sequence[int] = (128, 128),
                 cameras: Sequence[str] = NATIVE_CAMERAS, env_seed: int = 42, masks: bool = False,
                 point_cloud: bool = False, headless: bool = True) -> None:
        self.task = task
        self.family = task_family(task)
        self.strategy_index = int(strategy_index)
        self.image_size = [int(v) for v in image_size]
        self.cameras = tuple(cameras)
        self.env_seed = int(env_seed)
        self.masks = bool(masks)
        self.point_cloud = bool(point_cloud)
        self.headless = headless
        self.env = None
        self.task_env = None
        self.cfg = None
        self.strategy: dict[str, Any] | None = None

    # ---------------------------------------------------------------- config
    def _generator_module(self):
        from omegaconf import OmegaConf

        name = f"colosseum.tools.dataset_generator_{self.family}"
        if name in sys.modules:
            return sys.modules[name]
        if OmegaConf.has_resolver("eval"):
            OmegaConf.clear_resolver("eval")  # the upstream module registers it at import
        return importlib.import_module(name)

    def build_config(self):
        import colosseum
        from omegaconf import OmegaConf

        generator = self._generator_module()
        prefix = "ATOMIC" if self.family == "atomic" else "COMPOSITIONAL"
        config_dir = getattr(colosseum, f"ASSETS_{prefix}_CONFIGS_FOLDER")
        json_dir = getattr(colosseum, f"ASSETS_{prefix}_JSON_FOLDER")
        base = OmegaConf.load(os.path.join(config_dir, f"{self.task}.yaml"))
        # Same overrides the native collection scripts pass on the command line.
        base.env.seed = self.env_seed
        base.data.image_size = list(self.image_size)
        base.data.images.rgb = True
        base.data.images.depth = True
        base.data.images.mask = self.masks
        base.data.images.point_cloud = self.point_cloud
        for camera in list(base.data.cameras):
            base.data.cameras[camera] = camera in self.cameras
        with open(os.path.join(json_dir, f"{self.task}.json")) as handle:
            strategy = json.load(handle)
        if not generator.should_collect_task(strategy, self.strategy_index, self.strategy_index):
            raise ValueError(f"strategy {self.strategy_index} is disabled upstream for {self.task}")
        self.strategy = strategy["strategy"][self.strategy_index]
        self.cfg = generator.get_spreadsheet_config(base, strategy, self.strategy_index)
        return self.cfg

    # ---------------------------------------------------------------- launch
    def launch(self) -> None:
        import colosseum
        from colosseum.rlbench.extensions.environment import EnvironmentExt
        from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete

        cfg = self.cfg or self.build_config()
        prefix = "ATOMIC" if self.family == "atomic" else "COMPOSITIONAL"
        self.ttm_folder = getattr(colosseum, f"{prefix}_TASKS_TTM_FOLDER")
        self.py_folder = getattr(colosseum, f"{prefix}_TASKS_PY_FOLDER")
        self.env = EnvironmentExt(
            action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
            obs_config=ObservationConfigExt(cfg.data),
            headless=self.headless,
            path_task_ttms=self.ttm_folder,
            env_config=cfg.env,
        )
        self.env.launch()
        self.task_env = self.env.get_task(name_to_class(self.task, self.py_folder))

    @property
    def pyrep(self):
        return self.env._pyrep

    @property
    def robot(self):
        return self.env._robot

    @property
    def scene(self):
        return self.task_env._scene

    @property
    def task_obj(self):
        return self.task_env._task

    def task_shapes(self) -> list[Any]:
        from pyrep.const import ObjectType

        shapes = list(self.task_obj.get_base().get_objects_in_tree(object_type=ObjectType.SHAPE))
        # Grasped objects are re-parented to the gripper and leave the task tree.
        names = {shape.get_name() for shape in shapes}
        shapes += [obj for obj in self.task_obj.get_graspable_objects() if obj.get_name() not in names]
        return sorted(shapes, key=lambda shape: shape.get_name())

    def task_joints(self) -> list[Any]:
        from pyrep.const import ObjectType

        joints = self.task_obj.get_base().get_objects_in_tree(object_type=ObjectType.JOINT)
        return sorted(joints, key=lambda joint: joint.get_name())

    # ---------------------------------------------------------------- reset
    def reset(self, variation: int, *, rng_state: tuple | None = None, attempts: int = 3) -> dict[str, Any]:
        """Reset like ``TaskEnvironmentExt._get_live_demos``; return RNG lineage.

        Upstream placement can fail for an RNG state that succeeded before
        (robot reset residuals change collision checks). A failed placement is
        retried from the *same* state at most ``attempts`` times and counted;
        upstream generators instead continue with an advanced RNG state.
        """
        from rlbench.backend.exceptions import TaskEnvironmentError

        self.task_env.set_variation(int(variation))
        if rng_state is not None:
            np.random.set_state(rng_state)
        state = np.random.get_state()
        array, cached = rng_state_array(state)
        factors_before = self.factor_rng_states()
        errors = []
        for _ in range(int(attempts)):
            np.random.set_state(state)
            try:
                descriptions, _ = self.task_env.reset()
                break
            except TaskEnvironmentError as error:
                cause = error.__cause__
                errors.append(f"{type(cause).__name__}: {cause}"[:300] if cause else str(error)[:300])
        else:
            raise TaskEnvironmentError(f"placement failed {attempts}x from one RNG state: {errors[-1]}")
        return {
            "variation": int(variation),
            "rng_state": array,
            "rng_cached_gaussian": cached,
            "rng_state_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "descriptions": descriptions,
            "failed_placements_before_success": len(errors),
            "failed_placement_causes": errors,
            "factor_rng_before_reset": factors_before,
            "factors_after_reset": self.factor_rng_states(),
        }

    def variation_factor_state(self) -> list[dict[str, Any]]:
        """Enabled/disabled Colosseum factors as configured (the AP/CP axis)."""
        factors = []
        for factor in self.cfg.env.scene.factors:
            factors.append({
                "type": str(factor.variation),
                "name": str(factor.get("name", "any")),
                "enabled": bool(factor.enabled),
            })
        return factors

    def factor_rng_states(self) -> list[dict[str, Any]]:
        """Bit-generator state of every instantiated Colosseum factor (AP/CP lineage)."""
        manager = getattr(self.scene, "_var_manager", None)
        states = []
        for variation in getattr(manager, "_variations", []) or []:
            rng = getattr(variation, "_rng", None)
            states.append({
                "class": type(variation).__name__,
                "name": getattr(variation, "_name", None),
                "enabled": bool(getattr(variation, "_enabled", False)),
                "targets_found": len(getattr(variation, "_targets", []) or []),
                "bit_generator_state": rng.bit_generator.state if rng is not None else None,
            })
        return states

    def workspace_bounds(self) -> list[list[float]]:
        scene = self.scene
        return [[scene._workspace_minx, scene._workspace_miny, scene._workspace_minz],
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]]

    def provenance(self) -> dict[str, Any]:
        import colosseum
        import pyrep
        import rlbench

        colosseum_root = Path(colosseum.__file__).resolve().parents[2]
        ttm = Path(self.ttm_folder) / f"{self.task}.ttm"
        py = Path(self.py_folder) / f"{self.task}.py"
        return {
            "upstream_pins": json.loads(json.dumps(dict(UPSTREAM))),
            "robohiman_checkout": _git_revision(colosseum_root),
            "pyrep_checkout": _git_revision(Path(pyrep.__file__).resolve().parents[1]),
            "rlbench_checkout": _git_revision(Path(rlbench.__file__).resolve().parents[1]),
            "task_ttm_sha256": hashlib.sha256(ttm.read_bytes()).hexdigest(),
            "task_py_sha256": hashlib.sha256(py.read_bytes()).hexdigest(),
            "coppeliasim_root": os.environ.get("COPPELIASIM_ROOT"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "physics_dt": float(self.pyrep.get_simulation_timestep()),
            "env_seed": self.env_seed,
            "deviations_from_native_generator": [
                "numpy RNG seeded explicitly and full MT19937 state stored before reset "
                "(native workers call np.random.seed(None))",
                "camera set, image size and mask/point-cloud flags are explicit per collection",
                "frames rendered at every physics step (native skips gripper-actuation steps)",
            ],
        }

    def level(self) -> str:
        return level_for(self.task, self.strategy_index)

    def shutdown(self) -> None:
        if self.env is not None:
            self.env.shutdown()
            self.env = None


__all__ = ["RoboHiManSession", "rng_state_array", "rng_state_from_array"]
