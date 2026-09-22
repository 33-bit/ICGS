"""Phase-1 training views: pointer records only, no copied observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.collection.generation.sampler import mix_training_transitions
from icgs.data.schemas.episode_records import validate_episode


SUPPORTED_GENERATION_VIEWS = GENERATION_PROTOCOL.views
_HELD_OUT_SPLITS = frozenset({"dev", "test", "development"})


def _outcome(provenance: Mapping[str, Any]) -> str:
    return str(provenance.get("outcome") or provenance.get("result_class") or "success")


def _include_record(provenance: Mapping[str, Any], role: str) -> bool:
    if _outcome(provenance) in {"simulator_crash", "invalid_observation"}:
        return False
    split = provenance.get("split")
    subset = provenance.get("subset") or provenance.get("train_subset")
    if role == "train":
        if split in _HELD_OUT_SPLITS:
            return False
        if subset == GENERATION_PROTOCOL.validation:
            return False
        return True
    if role == "validation":
        return split == "train" and subset == GENERATION_PROTOCOL.validation
    if role == "evaluation":
        return split in _HELD_OUT_SPLITS
    return True


def build_generation_view(
    records: Sequence[Mapping[str, Any]],
    view: str,
    *,
    role: str = "train",
    mix: bool | None = None,
) -> list[dict[str, Any]]:
    if view not in SUPPORTED_GENERATION_VIEWS:
        raise ValueError(f"unsupported phase-1 view: {view}")
    samples: list[dict[str, Any]] = []
    apply_mix = GENERATION_PROTOCOL.mixture_measured_on == "training_transitions" if mix is None else mix
    for record in records:
        validate_episode(record)
        provenance = record["provenance"]
        if not _include_record(provenance, role):
            continue
        episode_id = provenance["episode_id"]
        observations = record["online_observations"]
        n_obs = len(observations)
        n_actions = n_obs - 1
        kind = provenance.get("episode_kind", "nominal")
        dt_values = list(record.get("dt") or [item["achieved_duration_s"] for item in record["transitions"]])
        if view == "D_geom":
            for boundary in range(n_obs):
                samples.append({
                    "episode_id": episode_id,
                    "t": boundary,
                    "rgb": (episode_id, "rgb", boundary),
                    "depth": (episode_id, "depth", boundary),
                    "pointcloud": (episode_id, "pointcloud", boundary),
                    "mask": (episode_id, "mask", boundary),
                    "calibration_id": provenance["calibration_id"],
                    "episode_kind": kind,
                    "subset": provenance.get("subset") or provenance.get("train_subset"),
                })
        elif view == "D_temporal":
            history = 4
            events = record.get("events") or ()
            preferred = set()
            for event in events:
                start = int(event.get("start_t", 0))
                preferred.update(range(max(0, start - 1), min(n_actions, start + 2)))
            if not preferred:
                preferred = set(range(n_actions))
            for t in sorted(preferred):
                t0 = max(0, t - history + 1)
                samples.append({
                    "episode_id": episode_id,
                    "t": t,
                    "observations": [(episode_id, "observation", index) for index in range(t0, t + 1)],
                    "actions": [(episode_id, "action", index) for index in range(t0, t)],
                    "dt": dt_values[t] if t < len(dt_values) else None,
                    "around": "contact_or_event",
                    "episode_kind": kind,
                })
        elif view == "D_dyn":
            intervention = record.get("intervention") or {}
            scope = intervention.get("application_scope", provenance.get("application_scope"))
            app_t = intervention.get("application_t", provenance.get("application_t"))
            if app_t is None:
                app_t = intervention.get("intervention_frame", provenance.get("intervention_frame"))
            episode_external = bool(
                provenance.get("episode_has_external_intervention",
                    intervention.get("episode_has_external_intervention",
                        intervention.get("external_intervention", provenance.get("external_intervention"))))
            )
            intervention_id = intervention.get("intervention_id") or provenance.get("intervention_id")
            for t in range(n_actions):
                history = list(range(max(0, t - 4), t))
                transition_external = (
                    episode_external
                    and scope in {"timestep", "event"}
                    and app_t is not None
                    and int(app_t) == t
                )
                samples.append({
                    "episode_id": episode_id,
                    "t": t,
                    "history": history,
                    "action_t": t,
                    "observation_t": t,
                    "observation_t1": t + 1,
                    "robot_state_t": t,
                    "object_state_t": t,
                    "object_state_t1": t + 1,
                    "dt": dt_values[t],
                    "episode_kind": kind,
                    "episode_has_external_intervention": episode_external,
                    "transition_has_external_intervention": transition_external,
                    "external_intervention": transition_external,
                    "initial_scene_intervention_id": (
                        intervention_id if scope == "initial_scene" else None
                    ),
                    "intervention_id": intervention_id if transition_external else None,
                })
        elif view == "D_task":
            events = record.get("events") or ()
            rho = record.get("rho")
            nu = record.get("nu")
            epsilon = record.get("epsilon")
            for t in range(n_obs):
                samples.append({
                    "episode_id": episode_id,
                    "t": t,
                    "events": [(episode_id, "event", event.get("step_id")) for event in events],
                    "rho": (episode_id, "rho", t) if rho is not None else None,
                    "nu": (episode_id, "nu", t) if nu is not None else None,
                    "epsilon": (episode_id, "epsilon", t) if epsilon is not None else None,
                })
    if role == "train" and apply_mix:
        samples = mix_training_transitions(samples, view=view)
    return samples
