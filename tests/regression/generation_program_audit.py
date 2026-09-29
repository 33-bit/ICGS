"""Explicit simulator audit of generation program semantics.

Diagnostic only: no archive, queue, Hugging Face publication, training or
quota collection.  Each attempt runs in its own ``xvfb-run`` process with the
canonical ``scripts/generation_episode_worker.run_program`` plus
instrumentation of routine-step boundaries, measured physics steps per action
and any ``set_position`` on a non-marker body after the first action (a
teleport).  Plans follow the production planner order: nominal plans first,
then perturbed plans after the program's nominal quota.

Run from the repository root on a provisioned generation host:

    .venv/bin/python -B tests/regression/generation_program_audit.py run \\
        --out outputs/program-audit --programs all --nominal 0,1,2 --parallel 6
    .venv/bin/python -B tests/regression/generation_program_audit.py summary outputs/program-audit
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
MANIFEST = REPO / "artifacts" / "composition" / "approved_composition_manifest.json"


def _is_marker(name: str) -> bool:
    return name.startswith("target") or name.endswith("_wp") or name.endswith("_target") or "_target_" in name


def _plan(program_id: str, kind: str, index: int):
    from icgs.data.collection.generation.batch import AttemptPlanner, bounds_from_row
    from icgs.data.collection.generation.quota import quota_for_program

    row = {item["program_id"]: item for item in json.loads(MANIFEST.read_text())["catalog"]}[program_id]
    quota = quota_for_program(program_id)
    planner = AttemptPlanner(
        program_id,
        bounds=bounds_from_row(row),
        asset_family_id=row.get("asset_family_id"),
        n_perturbed=quota.perturbed_attempt_target,
    )
    if kind == "nominal":
        plans = [planner.next_plan("nominal") for _ in range(index + 1)]
    else:
        for _ in range(quota.nominal_success_target):
            planner.next_plan("nominal")
        plans = [planner.next_plan("perturbed") for _ in range(index + 1)]
    return plans[-1]


def first_index_per_perturbation_kind(program_id: str) -> list[int]:
    """Production perturbed-plan index of the first attempt of every kind."""
    from icgs.data.collection.generation.batch import AttemptPlanner, bounds_from_row
    from icgs.data.collection.generation.quota import quota_for_program

    row = {item["program_id"]: item for item in json.loads(MANIFEST.read_text())["catalog"]}[program_id]
    quota = quota_for_program(program_id)
    planner = AttemptPlanner(
        program_id,
        bounds=bounds_from_row(row),
        asset_family_id=row.get("asset_family_id"),
        n_perturbed=quota.perturbed_attempt_target,
    )
    for _ in range(quota.nominal_success_target):
        planner.next_plan("nominal")
    first: dict[str, int] = {}
    for index in range(quota.perturbed_attempt_target):
        first.setdefault(planner.next_plan("perturbed").intervention["kind"], index)
    return sorted(first.values())


def run_child(program_id: str, kind: str, index: int, out_dir: Path) -> int:
    os.environ.setdefault("ICGS_GENERATION_TRACE", "1")
    loader = importlib.util.spec_from_file_location("generation_episode_worker", REPO / "scripts" / "generation_episode_worker.py")
    worker = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(worker)
    import numpy as np
    from pyrep.objects.object import Object
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from icgs.data.collection.generation.compiler import compile_generation_catalog

    spec = compile_generation_catalog(MANIFEST)[program_id]
    plan = _plan(program_id, kind, index)
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)
    obs_config.wrist_camera.point_cloud = True
    obs_config.wrist_camera.depth = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.joint_positions = True
    env = Environment(
        MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaIK(collision_checking=False),
            gripper_action_mode=Discrete(attach_grasped_objects=False),
        ),
        "./",
        obs_config=obs_config,
        headless=True,
    )
    env.launch()
    per_action_steps: list[int] = []
    counter = {"steps": 0}
    original_step = env._pyrep.step

    def counting_step(*args, **kwargs):
        counter["steps"] += 1
        return original_step(*args, **kwargs)

    env._pyrep.step = counting_step
    teleports: list[dict] = []
    original_set_position = Object.set_position

    def logging_set_position(self, position, relative_to=None, reset_dynamics=True):
        if per_action_steps:
            name = self.get_name()
            if not name.startswith("Panda") and not _is_marker(name):
                teleports.append({
                    "name": name,
                    "before": [float(v) for v in self.get_position()],
                    "after": [float(v) for v in position],
                    "action_index": len(per_action_steps),
                })
        return original_set_position(self, position, relative_to, reset_dynamics)

    Object.set_position = logging_set_position
    original_get_task = env.get_task

    def get_task(task_class):
        task = original_get_task(task_class)
        original = task.step

        def counted(action):
            before = counter["steps"]
            result = original(action)
            per_action_steps.append(counter["steps"] - before)
            return result

        task.step = counted
        return task

    env.get_task = get_task
    started = time.monotonic()
    error = None
    try:
        row = worker.run_program(env, spec, plan=plan)
    except Exception:  # noqa: BLE001 - recorded as a simulator crash
        import traceback

        error = traceback.format_exc()
        row = {"result_class": "simulator_crash", "success": False}
    final_poses = worker.live_poses(spec.objects)
    try:
        env.shutdown()
    except Exception:  # noqa: BLE001
        pass
    summary = {
        "program_id": program_id,
        "kind": kind,
        "index": index,
        "scene_seed": plan.scene_seed,
        "episode_id": plan.episode_id,
        "scale": plan.randomization.get("scale"),
        "yaw_deg": plan.randomization.get("object_yaw_deg"),
        "intervention": (plan.intervention or {}).get("kind"),
        "result_class": row.get("result_class"),
        "success": row.get("success"),
        "success_criteria": row.get("success_criteria"),
        "step_outcomes": row.get("step_outcomes"),
        "predicate_distances_m": row.get("predicate_distances_m"),
        "rotation_checks": row.get("rotation_checks"),
        "n_actions": row.get("n_actions"),
        "recorded_sim_time_s": row.get("sim_time_s"),
        "physics_steps": counter["steps"],
        "physics_steps_per_action": {
            "min": min(per_action_steps) if per_action_steps else None,
            "max": max(per_action_steps) if per_action_steps else None,
            "mean": float(np.mean(per_action_steps)) if per_action_steps else None,
        },
        "recorded_substeps_total": int(sum(item["physics_substeps"] for item in row.get("_transition_timing") or ())),
        "teleports": teleports,
        "final_poses": final_poses,
        "elapsed_s": time.monotonic() - started,
        "error": error,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{program_id}-{kind}-{index:03d}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    with gzip.open(out_dir / f"{stem}.trace.json.gz", "wt") as stream:
        json.dump({"trace": row.get("controller_trace") or [], "per_action_steps": per_action_steps}, stream, default=str)
    print("DONE", stem, summary["result_class"], flush=True)
    return 0


def run_parent(args) -> int:
    programs = args.programs
    if programs == ["all"]:
        programs = [row["program_id"] for row in json.loads(MANIFEST.read_text())["catalog"]]
    jobs = [(program, "nominal", int(index)) for program in programs for index in args.nominal.split(",") if index]
    if args.perturbed == "each-kind":
        jobs += [(program, "perturbed", index) for program in programs for index in first_index_per_perturbation_kind(program)]
    else:
        jobs += [(program, "perturbed", int(index)) for program in programs for index in args.perturbed.split(",") if index]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    simulator = str(REPO / "outputs" / "CoppeliaSim")
    environment = dict(os.environ)
    environment.update({
        "COPPELIASIM_ROOT": simulator,
        "LD_LIBRARY_PATH": simulator + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
        "QT_QPA_PLATFORM_PLUGIN_PATH": simulator,
        "QT_QPA_PLATFORM": "xcb",
        "QT_LOGGING_RULES": "*.debug=false",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "PYTHONPATH": os.pathsep.join((str(REPO / "src"), str(REPO / "outputs" / "RLBench"), str(REPO))),
        "PYTHONUNBUFFERED": "1",
    })

    def run(job) -> None:
        program, kind, index = job
        stem = f"{program}-{kind}-{index:03d}"
        if (out_dir / f"{stem}.json").exists() and not args.force:
            return
        command = [
            "xvfb-run", "-a", "-s", "-screen 0 1280x1024x24",
            sys.executable, "-B", str(Path(__file__).resolve()),
            "child", program, kind, str(index), "--out", str(out_dir),
        ]
        with (out_dir / f"{stem}.log").open("w") as log:
            try:
                subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                log.write("\nTIMEOUT\n")
        print("finished", stem, flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(run, jobs))
    return 0


def summarize(out_dir: Path) -> int:
    rows = [json.loads(path.read_text()) for path in sorted(Path(out_dir).glob("*.json"))]
    success = 0
    for item in rows:
        outcomes = item.get("step_outcomes") or []
        failed = [f"{o['type']}:{o.get('obj')}->{o.get('target')}" for o in outcomes if o.get("achieved") is False]
        retries = sum(int(o.get("retries") or 0) for o in outcomes)
        distances = ", ".join(
            f"{p['a']}~{p['b']}={p['distance_m'] * 1000:.1f}mm" if p.get("distance_m") is not None else f"{p['a']}~{p['b']}=None"
            for p in item.get("predicate_distances_m") or ()
        )
        success += item.get("result_class") == "success"
        print(" | ".join(str(value) for value in (
            item["program_id"], item["kind"], item["index"], item.get("intervention") or "-", item.get("result_class"),
            item.get("n_actions"), item.get("physics_steps"), item.get("recorded_substeps_total"),
            f"retries={retries}", f"teleports={len(item.get('teleports') or ())}", ";".join(failed) or "-", distances,
            (item.get("error") or "")[-160:],
        )))
    print(f"TOTAL {len(rows)} success={success} other={len(rows) - success}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--programs", nargs="+", default=["all"])
    run.add_argument("--nominal", default="0")
    run.add_argument("--perturbed", default="", help="comma-separated indices or 'each-kind'")
    run.add_argument("--parallel", type=int, default=4)
    run.add_argument("--timeout", type=int, default=1500)
    run.add_argument("--force", action="store_true")
    child = sub.add_parser("child")
    child.add_argument("program")
    child.add_argument("kind", choices=("nominal", "perturbed"))
    child.add_argument("index", type=int)
    child.add_argument("--out", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("out")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_parent(args)
    if args.command == "child":
        return run_child(args.program, args.kind, args.index, Path(args.out))
    return summarize(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
