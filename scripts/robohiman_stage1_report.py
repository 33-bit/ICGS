"""Offline Stage-1 checks over a RoboHiMan-instrumented store (no simulator).

Builds D_geom/D_dyn/D_task views, confirms D_value is refused, and per episode
reports A0 reconstruction facts, A1 command-vs-achieved separation and timing,
FK-derived canonical EE commands (only if the fitted FK residual passes), B
predicate/label timelines, outcome/perturbation and provenance. Optionally
writes front-camera PNGs around every event occurrence for manual review.

  python -B scripts/robohiman_stage1_report.py --store STORE --out report.json [--frames-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
import struct
from pathlib import Path
from typing import Any

import numpy as np

from icgs.data.stage1.store import canonical_json, iter_episode_dirs, read_episode
from icgs.data.stage1.views import ReferencePolicyRequired, build_views
from icgs.environments.robohiman.camera import fused_cloud, summarize_calibration
from icgs.environments.robohiman.kinematics import canonical_targets, fit_fk_calibration


FK_TOLERANCE = {"position_m": 1e-3, "rotation_deg": 0.1}


def _png(path: Path, image: np.ndarray) -> None:
    image = np.ascontiguousarray(image.astype(np.uint8))
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _edges(trace: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    events = []
    for column, name in enumerate(names):
        values = trace[:, column].astype(np.int8)
        for boundary in np.flatnonzero(np.diff(values)) + 1:
            events.append({"boundary": int(boundary), "predicate": name, "to": bool(values[boundary])})
    return sorted(events, key=lambda item: (item["boundary"], item["predicate"]))


def _a0(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    cameras = manifest["cameras"]["names"]
    if not cameras or "frame_step" not in arrays:
        return {"status": "NOT RUN", "reason": "no camera frames"}
    bounds = np.asarray(manifest["environment"]["workspace_bounds"], dtype=np.float64)
    frames = sorted({0, arrays["frame_step"].shape[0] // 2, arrays["frame_step"].shape[0] - 1})
    samples = []
    for frame in frames:
        points, valid, source = fused_cloud(arrays, frame, cameras)
        inside = np.all((points >= bounds[0] - 0.05) & (points <= bounds[1] + 0.05), axis=1)
        entry = {"frame": frame, "points": int(points.shape[0]), "valid_frac": float(valid.mean()),
                 "valid_within_workspace_5cm_frac": float(inside[valid].mean()),
                 "valid_xyz_min": points[valid].min(axis=0).tolist(),
                 "valid_xyz_max": points[valid].max(axis=0).tolist()}
        live_errors = []
        for index, name in enumerate(cameras):
            key = f"cam_{name}_point_cloud_live"
            if key in arrays:
                live = arrays[key][frame].reshape(-1, 3)
                mask = valid[source == index]
                live_errors.append(float(np.abs(points[source == index][mask] - live[mask]).max()))
        if live_errors:
            entry["numpy_vs_live_pointcloud_max_abs_m"] = max(live_errors)
        samples.append(entry)
    result = {"calibration": summarize_calibration(arrays, cameras), "frames": samples,
              "workspace_bounds": bounds.tolist()}
    try:
        import torch

        from icgs.data.preprocessing.physical import PhysicalPreprocessConfig, prepare_physical_cloud

        points, valid, _ = fused_cloud(arrays, frames[0], cameras,
                                       crop=(bounds[0].tolist(), bounds[1].tolist()))
        config = PhysicalPreprocessConfig(crop_min=tuple(bounds[0]), crop_max=tuple(bounds[1]))
        prepared = prepare_physical_cloud(torch.as_tensor(points), torch.as_tensor(valid), config)
        result["physical_preprocessing"] = {
            "status": "PASS",
            "fields": {name: list(getattr(prepared, name).shape) for name in vars(prepared)
                       if hasattr(getattr(prepared, name), "shape")},
        }
    except ModuleNotFoundError as error:
        result["physical_preprocessing"] = {"status": "SKIPPED", "reason": f"{error.name} unavailable in this venv"}
    return result


def _a1(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    times = arrays["step_sim_time"]
    dt = np.diff(times)
    target_valid = arrays["cmd_arm_joint_target_valid"]
    steps = target_valid.shape[0]
    tracking = np.abs(arrays["cmd_arm_joint_target"][target_valid]
                      - arrays["step_joint_positions"][1:][target_valid])
    result: dict[str, Any] = {
        "physics_dt_declared": manifest["controller"]["physics_dt"],
        "sim_dt_min_max": [float(dt.min()), float(dt.max())],
        "sim_dt_all_equal_declared": bool(np.allclose(  # float32 simulator clock spacing at t_end
            dt, manifest["controller"]["physics_dt"], atol=float(np.spacing(np.float32(times[-1]))) + 1e-9)),
        "steps": int(steps),
        "arm_target_command_rows": int(target_valid.sum()),
        "gripper_velocity_command_rows": int(arrays["cmd_gripper_joint_velocity_valid"].any(axis=1).sum()),
        "rows_without_any_command": int((~(target_valid | arrays["cmd_arm_joint_velocity_valid"]
                                          | arrays["cmd_gripper_joint_velocity_valid"].any(axis=1))).sum()),
        "grasp_events": int((arrays["cmd_grasp_event"] != 0).sum()),
        "kinematic_excursion_rows": int((arrays["cmd_arm_teleport_calls"] > 0).sum()),
        "phase_counts": {str(k): int(v) for k, v in zip(*np.unique(arrays["cmd_phase"], return_counts=True))},
        "command_vs_achieved_next_joint_abs_rad": {
            "mean": float(tracking.mean()) if tracking.size else None,
            "p95": float(np.percentile(tracking, 95)) if tracking.size else None,
            "max": float(tracking.max()) if tracking.size else None,
            "rows_identical": int((tracking.max(axis=1) == 0).sum()) if tracking.size else 0,
        },
        "frames_per_boundary": (float(arrays["frame_step"].shape[0] / (steps + 1))
                                if "frame_step" in arrays else 0.0),
        "wall_s_per_step_mean": float(arrays["cmd_wall_s"].mean()) if "cmd_wall_s" in arrays else None,
    }
    robot = manifest["environment"].get("robot", {})
    if robot.get("arm_base_pose"):
        calibration = fit_fk_calibration(arrays["step_joint_positions"], arrays["step_tip_pose"],
                                         np.asarray(robot["arm_base_pose"]))
        passes = (calibration["residual_position_max_m"] <= FK_TOLERANCE["position_m"]
                  and calibration["residual_rotation_max_deg"] <= FK_TOLERANCE["rotation_deg"])
        fk = {key: value for key, value in calibration.items() if not isinstance(value, np.ndarray)}
        fk["tolerance"] = FK_TOLERANCE
        fk["canonical_commands_derivable"] = bool(passes)
        if passes and target_valid.any():
            canonical = canonical_targets(arrays["cmd_arm_joint_target"][target_valid], calibration)
            achieved = arrays["step_tip_pose"][1:][target_valid]
            gap = np.linalg.norm(canonical[:, :3] - achieved[:, :3], axis=1)
            fk["commanded_ee_vs_achieved_next_ee_m"] = {"mean": float(gap.mean()),
                                                         "p95": float(np.percentile(gap, 95)),
                                                         "max": float(gap.max())}
        result["canonical_ee_command"] = fk
    return result


def _relabel(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[list, dict[str, np.ndarray]] | None:
    """Re-derive labels offline from the stored raw trace with the current reviewed spec."""
    from icgs.data.stage1.labels import derive_event_labels
    from icgs.environments.robohiman.monitors import TASK_SPECS

    spec = TASK_SPECS.get(manifest["source"]["task"])
    if spec is None:
        return None
    events = [dict({"prerequisites": [], "current_requirements": [], "hold_steps": 1,
                    "count_only_when_eligible": False}, **event) for event in spec["events"]]
    labels = derive_event_labels(arrays["step_predicates"], manifest["predicates"]["names"], events)
    return events, {f"label_{k}": v for k, v in labels.items()}


def _b(manifest: dict[str, Any], arrays: dict[str, np.ndarray], relabel: bool = False) -> dict[str, Any]:
    names = manifest["predicates"]["names"]
    specs = manifest["events"]["specs"]
    if relabel:
        derived = _relabel(manifest, arrays)
        if derived is not None:
            specs, labels = derived
            arrays = {**arrays, **labels}
    alpha = arrays["label_alpha"]
    changes = [0] + [int(i) for i in np.flatnonzero(np.diff(alpha)) + 1]
    first = arrays["label_first_occurrence"]
    return {
        "predicate_edges": _edges(arrays["step_predicates"], names),
        "task_success_first_boundary": (int(np.argmax(arrays["step_task_success"]))
                                        if arrays["step_task_success"].any() else None),
        "event_first_occurrence": {spec["event_id"]: (int(first[j]) if first[j] >= 0 else None)
                                   for j, spec in enumerate(specs)},
        "alpha_segments": [{"from_boundary": b, "alpha": int(alpha[b]),
                            "event": specs[int(alpha[b])]["event_id"] if alpha[b] < len(specs) else "null"}
                           for b in changes],
        "final": {"nu": arrays["label_nu"][-1].astype(int).tolist(),
                  "rho": arrays["label_rho"][-1].astype(int).tolist(),
                  "epsilon": arrays["label_epsilon"][-1].astype(int).tolist()},
        "consistency": (manifest["events"]["consistency"] if not relabel else __import__(
            "icgs.data.stage1.labels", fromlist=["label_consistency_report"]).label_consistency_report(
            {k[6:]: v for k, v in arrays.items() if k.startswith("label_")}, specs)),
        "labels_source": "offline relabel with current TASK_SPECS" if relabel else "stored at collection",
    }


def _frames(manifest: dict[str, Any], arrays: dict[str, np.ndarray], out: Path) -> list[str]:
    camera = "front" if "cam_front_rgb" in arrays else None
    if camera is None:
        return []
    out.mkdir(parents=True, exist_ok=True)
    frame_step = arrays["frame_step"]
    written = []
    boundaries = {0, int(frame_step[-1])}
    for value in arrays["label_first_occurrence"]:
        if value >= 0:
            boundaries.update({max(int(value) - 1, 0), int(value)})
    for boundary in sorted(boundaries):
        frame = int(np.searchsorted(frame_step, boundary))
        frame = min(frame, frame_step.shape[0] - 1)
        path = out / f"{manifest['episode_id']}-b{int(frame_step[frame]):04d}.png"
        _png(path, arrays[f"cam_{camera}_rgb"][frame])
        written.append(path.name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir")
    parser.add_argument("--relabel", action="store_true", help="also re-derive labels with current monitor specs")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"store": str(args.store), "episodes": []}
    report["views"] = build_views(args.store)["views"]
    try:
        build_views(args.store, view_names=("D_value",))
        report["d_value_refused"] = False
    except ReferencePolicyRequired:
        report["d_value_refused"] = True
    for episode_dir in iter_episode_dirs(args.store):
        manifest, arrays = read_episode(episode_dir)
        entry = {
            "episode_id": manifest["episode_id"],
            "task": manifest["source"]["task"],
            "level": manifest["source"]["level"],
            "outcome": manifest["outcome"],
            "perturbation": {"families": manifest["perturbation"]["families"],
                             "execution": manifest["perturbation"]["execution"],
                             "enabled_benchmark_factors": [f for f in manifest["perturbation"]["benchmark_factors"]
                                                           if f["enabled"]]},
            "provenance": {"run": manifest["lineage"]["collection_run_id"],
                           "reset_rng_sha256": manifest["lineage"]["reset_rng"]["sha256"],
                           "code": manifest["code"],
                           "robohiman": manifest["environment"]["robohiman_checkout"],
                           "task_ttm_sha256": manifest["environment"]["task_ttm_sha256"],
                           "arrays_sha256": manifest["arrays"]["sha256"]},
            "counts": manifest["counts"],
            "bytes": sum(item.stat().st_size for item in episode_dir.iterdir()),
            "A0": _a0(manifest, arrays),
            "A1": _a1(manifest, arrays),
            "B": _b(manifest, arrays),
            "B_relabel": _b(manifest, arrays, relabel=True) if args.relabel else None,
        }
        if args.frames_dir:
            entry["review_frames"] = _frames(manifest, arrays, Path(args.frames_dir))
        report["episodes"].append(entry)
    Path(args.out).write_text(canonical_json(report))
    print(canonical_json({"views": report["views"], "d_value_refused": report["d_value_refused"],
                          "episodes": [{"id": e["episode_id"], "outcome": e["outcome"]["status"],
                                        "families": e["perturbation"]["families"]} for e in report["episodes"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
