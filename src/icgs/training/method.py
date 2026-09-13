"""Staged training runner, full-history replay slicing, batching, and exact resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from icgs.algorithms.objectives.physical import (
    bootstrap_mask,
    masked_normalized_chamfer_distance,
    physical_loss,
    rollout_loss,
)
from icgs.algorithms.rollout.physical import PhysicalRollout
from icgs.artifacts.method import reference_fingerprint, validate_method_manifest, validate_resume
from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, PhysicalPrediction, TimedCommand, TimedObservation
from icgs.contracts.records import Observation
from icgs.data.datasets.episodes import validate_split_lineage
from icgs.models.decoders.physical import PhysicalDecoder
from icgs.models.encoders.physical import PhysicalEncoder
from icgs.models.memories.physical import action_descriptor, proprioception
from icgs.state.physical import PhysicalState
from icgs.training.stages.method import (
    apply_freeze_boundary,
    build_stage_optimizer,
    build_stage_scheduler,
    get_geometry_encoder_decoder,
    rollout_horizon,
    trainable_components,
)


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a training execution run."""

    stage: str
    updates_completed: int
    loss_history: tuple[float, ...]
    eval_metrics: dict[str, Any]
    stopped_early: bool
    checkpoint_path: str | None = None
    training_seeds: tuple[int, ...] | None = None
    selected_seed: int | None = None
    status: str = "completed"
    scheduler_state: dict[str, Any] | None = None
    accumulated_loss_history: tuple[float, ...] = ()
    selection_schema: tuple[str, ...] | None = None
    best_metric_vector: tuple[float, ...] | None = None
    best_update: int | None = None
    selection_status: str | None = None
    total_updates_completed: int = 0


def history_slices(
    boundary: int,
    burnin: int,
    *,
    supervised_intervals: int | None = None,
) -> tuple[slice, slice, slice]:
    """Compute causal history slices for prefix replay, burn-in, and supervision.

    Parameters
    ----------
    boundary:
        The current boundary index separating past history from supervision.
    burnin:
        Number of intervals immediately preceding ``boundary`` for state warmup.
    supervised_intervals:
        Optional length of the supervised window. If None, extends to end of sequence.

    Returns
    -------
    prefix:
        Slice from 0 to burn-in start (replayed without gradients).
    burn:
        Slice from burn-in start to boundary (warmup window).
    supervised:
        Slice starting at boundary for loss calculation.
    """
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise ValueError("boundary must be a nonnegative integer")
    if isinstance(burnin, bool) or not isinstance(burnin, int) or burnin < 0:
        raise ValueError("burnin must be a nonnegative integer")
    if supervised_intervals is not None:
        if isinstance(supervised_intervals, bool) or not isinstance(supervised_intervals, int) or supervised_intervals < 1:
            raise ValueError("supervised_intervals must be a positive integer")

    burn_start = max(0, boundary - burnin)
    prefix = slice(0, burn_start)
    burn = slice(burn_start, boundary)
    supervised_end = boundary + supervised_intervals if supervised_intervals is not None else None
    supervised = slice(boundary, supervised_end)
    return prefix, burn, supervised


def sample_anchor_horizon(groups: Sequence[Any], rng: Any) -> tuple[Any, int]:
    """Sample an anchor group and a valid rollout horizon using the provided RNG."""
    if not isinstance(groups, Sequence) or len(groups) == 0:
        raise ValueError("groups must be a nonempty sequence")

    # Sample anchor group
    seq_tuple = tuple(groups)
    if hasattr(rng, "choice"):
        try:
            anchor = rng.choice(seq_tuple)
        except (ValueError, TypeError):
            idx = int(rng.choice(len(seq_tuple)))
            anchor = seq_tuple[idx]
    else:
        idx = rng.randint(0, len(seq_tuple) - 1)
        anchor = seq_tuple[idx]

    # Extract valid horizons
    valid_horizons = getattr(anchor, "valid_horizons", None)
    if valid_horizons is None and isinstance(anchor, Mapping):
        valid_horizons = anchor.get("valid_horizons")
    if not valid_horizons:
        raise ValueError("selected anchor has no valid_horizons")

    horizons_tuple = tuple(valid_horizons)
    if hasattr(rng, "choice"):
        try:
            horizon = int(rng.choice(horizons_tuple))
        except (ValueError, TypeError):
            idx = int(rng.choice(len(horizons_tuple)))
            horizon = int(horizons_tuple[idx])
    else:
        idx = rng.randint(0, len(horizons_tuple) - 1)
        horizon = int(horizons_tuple[idx])

    return anchor, horizon


def build_transition_descriptor_and_achieved(
    before_T_w_e: Tensor,
    command: TimedCommand,
    after_T_w_e: Tensor,
    *,
    config: MethodConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute training input descriptor from before-pose + command, and achieved translation target.

    Invariant:
    - input descriptor is computed from before-transition achieved pose and command target_w.
    - achieved translation target is computed from after-transition pose relative to before-transition pose.
    """
    cfg = config if config is not None else MethodConfig()
    if before_T_w_e.ndim == 2:
        before_batch = before_T_w_e.unsqueeze(0)
    else:
        before_batch = before_T_w_e

    descriptor = action_descriptor(before_batch, command, config=cfg)
    if before_T_w_e.ndim == 2:
        descriptor = descriptor.squeeze(0)

    # For translation residual target:
    achieved_translation = after_T_w_e[..., :3, 3] - before_T_w_e[..., :3, 3]
    return descriptor, achieved_translation


def _get_rng_state(runner_rng: Any | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if runner_rng is not None:
        if hasattr(runner_rng, "bit_generator") and hasattr(runner_rng.bit_generator, "state"):
            state["runner_rng"] = copy.deepcopy(runner_rng.bit_generator.state)
        elif hasattr(runner_rng, "getstate"):
            state["runner_rng"] = copy.deepcopy(runner_rng.getstate())
        elif hasattr(runner_rng, "state"):
            state["runner_rng"] = copy.deepcopy(runner_rng.state)
        else:
            raise TypeError(f"unsupported runner_rng type: {type(runner_rng).__name__}")
    return state


def _set_rng_state(state: Mapping[str, Any], runner_rng: Any | None = None) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"rng state must be a mapping, got {type(state).__name__}")
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise KeyError(f"checkpoint rng state missing required {key!r} state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if runner_rng is not None:
        if "runner_rng" not in state:
            raise KeyError("checkpoint rng state missing required 'runner_rng' state")
        if hasattr(runner_rng, "bit_generator") and hasattr(runner_rng.bit_generator, "state"):
            runner_rng.bit_generator.state = copy.deepcopy(state["runner_rng"])
        elif hasattr(runner_rng, "setstate"):
            runner_rng.setstate(copy.deepcopy(state["runner_rng"]))
        elif hasattr(runner_rng, "state"):
            runner_rng.state = copy.deepcopy(state["runner_rng"])
        else:
            raise TypeError(f"unsupported runner_rng type: {type(runner_rng).__name__}")


def save_checkpoint(
    checkpoint_path: str | Path,
    *,
    stage: str,
    update: int,
    total_updates: int,
    components: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    config: MethodConfig,
    dataset_id: str,
    reference_id: str,
    selected_seed: int,
    rng: Any | None = None,
    sampler_state: Any | None = None,
    selection_state: Any | None = None,
    accumulated_loss_history: Sequence[float] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Save an atomic training checkpoint containing exact state and validation manifest."""
    if not isinstance(dataset_id, str) or not dataset_id.strip() or dataset_id == "dataset-synthetic":
        raise ValueError("save_checkpoint requires explicit non-empty dataset_id; default 'dataset-synthetic' is rejected")
    if not isinstance(reference_id, str) or not reference_id.strip() or reference_id == "reference-synthetic":
        raise ValueError("save_checkpoint requires explicit non-empty reference_id; default 'reference-synthetic' is rejected")
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int) or selected_seed < 0:
        raise ValueError(f"save_checkpoint requires non-negative integer selected_seed, got {selected_seed!r}")

    target_path = Path(checkpoint_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "stage": stage,
        "reference_id": reference_id,
        "dataset_id": dataset_id,
        "config_id": config.fingerprint(),
        "total_updates": total_updates,
        "update": update,
        "selected_seed": selected_seed,
    }

    if selection_state is None:
        canonical_sel_state: dict[str, Any] = {
            "best_metric": None,
            "best_metric_vector": None,
            "selection_schema": None,
            "best_update": None,
            "patience_counter": 0,
            "eval_history": [],
            "selection_status": "no_evaluation",
        }
    else:
        if not isinstance(selection_state, Mapping):
            raise ValueError(f"selection_state must be a mapping, got {type(selection_state).__name__}")
        canonical_sel_state = dict(selection_state)
        if "best_metric" not in canonical_sel_state:
            vec = canonical_sel_state.get("best_metric_vector")
            canonical_sel_state["best_metric"] = vec[0] if vec is not None and len(vec) > 0 else None
        if "selection_status" not in canonical_sel_state:
            if canonical_sel_state.get("selection_schema") is None:
                canonical_sel_state["selection_status"] = "no_evaluation"
            elif canonical_sel_state.get("best_metric_vector") is None:
                canonical_sel_state["selection_status"] = (
                    "not_certified (bridge gate unavailable)"
                    if stage == "A0"
                    else "not_certified"
                )
            elif stage == "A0":
                canonical_sel_state["selection_status"] = "not_certified (bridge gate unavailable)"
            elif canonical_sel_state.get("selection_schema") == ("physical_validation_loss", "dev_ranking"):
                canonical_sel_state["selection_status"] = "diagnostic_external"
            else:
                canonical_sel_state["selection_status"] = "certified"

    _validate_selection_state(canonical_sel_state, stage=stage, checkpoint_update=update)

    state = {
        "manifest": manifest,
        "update": update,
        "total_updates": total_updates,
        "selected_seed": selected_seed,
        "components": {
            name: mod.state_dict() for name, mod in components.items() if isinstance(mod, nn.Module)
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "rng": _get_rng_state(runner_rng=rng),
        "config": config.resolved_config(),
        "sampler_state": sampler_state,
        "selection_state": canonical_sel_state,
        "accumulated_loss_history": list(accumulated_loss_history or []),
        "metadata": dict(extra_metadata or {}),
    }

    temp_fd, temp_file = tempfile.mkstemp(dir=target_path.parent, prefix="ckpt_", suffix=".pt.tmp")
    os.close(temp_fd)
    try:
        torch.save(state, temp_file)
        os.replace(temp_file, target_path)
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass

    return str(target_path)


def load_checkpoint(
    checkpoint_path: str | Path | Mapping[str, Any],
    *,
    components: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    requested_manifest: Mapping[str, Any] | None = None,
    restore_rng: bool = True,
    runner_rng: Any | None = None,
    datasets: Any | None = None,
) -> dict[str, Any]:
    """Load and validate an exact checkpoint with strict atomic preflight."""
    if isinstance(checkpoint_path, Mapping):
        state = dict(checkpoint_path)
    else:
        target_path = Path(checkpoint_path).expanduser().resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {target_path}")
        state = torch.load(target_path, map_location="cpu")

    if not isinstance(state, Mapping):
        raise ValueError("checkpoint missing state dict mapping")

    # Strict top-level keys check
    required_top_keys = (
        "manifest",
        "update",
        "selected_seed",
        "total_updates",
        "components",
        "optimizer",
        "scheduler",
        "rng",
        "config",
        "sampler_state",
        "selection_state",
        "accumulated_loss_history",
    )
    for k in required_top_keys:
        if k not in state:
            raise KeyError(f"checkpoint missing required key: {k!r}")

    # Validate manifest and config envelope
    saved_manifest = state["manifest"]
    if not isinstance(saved_manifest, Mapping):
        raise ValueError("checkpoint manifest must be a mapping")
    if requested_manifest is not None:
        validate_resume(saved_manifest, requested_manifest)
        req_cfg_id = requested_manifest.get("config_id")
        if req_cfg_id is not None:
            saved_cfg = state["config"]
            if not isinstance(saved_cfg, Mapping):
                raise ValueError("checkpoint config must be a mapping")
            try:
                if "config" in saved_cfg and "config_sha256" in saved_cfg:
                    saved_sha = saved_cfg["config_sha256"]
                    if saved_sha != req_cfg_id:
                        raise ValueError(f"checkpoint config fingerprint mismatch: {saved_sha} != {req_cfg_id}")
                    MethodConfig.from_dict(dict(saved_cfg["config"]))
                else:
                    rec_cfg = MethodConfig.from_dict(dict(saved_cfg))
                    if rec_cfg.fingerprint() != req_cfg_id:
                        raise ValueError(f"checkpoint config fingerprint mismatch")
            except Exception as e:
                raise ValueError(f"checkpoint config invalid: {e}") from e

    saved_update = state["update"]
    saved_total_updates = state["total_updates"]
    if isinstance(saved_update, bool) or not isinstance(saved_update, int) or saved_update < 0:
        raise ValueError(f"checkpoint update must be a non-negative integer, got {saved_update!r}")
    if isinstance(saved_total_updates, bool) or not isinstance(saved_total_updates, int) or saved_total_updates < 1:
        raise ValueError(f"checkpoint total_updates must be a positive integer, got {saved_total_updates!r}")
    if saved_update > saved_total_updates:
        raise ValueError(f"checkpoint update {saved_update} exceeds total_updates {saved_total_updates}")

    # Validate agreement between manifest and top-level update / total_updates
    if "update" not in saved_manifest:
        raise KeyError("checkpoint manifest missing required 'update'")
    if "total_updates" not in saved_manifest:
        raise KeyError("checkpoint manifest missing required 'total_updates'")

    man_update = saved_manifest["update"]
    if isinstance(man_update, bool) or not isinstance(man_update, int) or man_update < 0:
        raise ValueError(f"checkpoint manifest update must be a non-negative integer, got {man_update!r}")
    if man_update != saved_update:
        raise ValueError(
            f"checkpoint top-level update ({saved_update}) does not match manifest update ({man_update})"
        )

    man_total = saved_manifest["total_updates"]
    if isinstance(man_total, bool) or not isinstance(man_total, int) or man_total < 1:
        raise ValueError(f"checkpoint manifest total_updates must be a positive integer, got {man_total!r}")
    if man_total != saved_total_updates:
        raise ValueError(
            f"checkpoint top-level total_updates ({saved_total_updates}) does not match manifest total_updates ({man_total})"
        )
    if man_update > man_total:
        raise ValueError(f"checkpoint manifest update {man_update} exceeds total_updates {man_total}")

    # Validate agreement between manifest and top-level selected_seed
    saved_seed = state["selected_seed"]
    if isinstance(saved_seed, bool) or not isinstance(saved_seed, int) or saved_seed < 0:
        raise ValueError(f"checkpoint selected_seed must be a non-negative integer, got {saved_seed!r}")
    if "selected_seed" not in saved_manifest:
        raise KeyError("checkpoint manifest missing required 'selected_seed'")
    man_seed = saved_manifest["selected_seed"]
    if isinstance(man_seed, bool) or not isinstance(man_seed, int) or man_seed < 0:
        raise ValueError(f"checkpoint manifest selected_seed must be a non-negative integer, got {man_seed!r}")
    if man_seed != saved_seed:
        raise ValueError(f"checkpoint top-level selected_seed ({saved_seed}) does not match manifest selected_seed ({man_seed})")

    # Validate accumulated loss history
    saved_acc_loss = state["accumulated_loss_history"]
    if not isinstance(saved_acc_loss, (list, tuple)):
        raise TypeError(f"checkpoint accumulated_loss_history must be a list or tuple, got {type(saved_acc_loss).__name__}")
    if len(saved_acc_loss) != saved_update:
        raise ValueError(
            f"checkpoint accumulated_loss_history length mismatch: expected {saved_update}, got {len(saved_acc_loss)}"
        )
    for idx, loss_val in enumerate(saved_acc_loss):
        if isinstance(loss_val, bool) or not isinstance(loss_val, (int, float)) or not math.isfinite(loss_val):
            raise ValueError(f"checkpoint accumulated_loss_history[{idx}] is not a finite number: {loss_val!r}")

    # Validate exact component key sets and tensor names, shapes, and dtypes
    saved_components = state["components"]
    if not isinstance(saved_components, Mapping):
        raise ValueError("checkpoint missing components state dict mapping")

    live_module_names = {name for name, mod in components.items() if isinstance(mod, nn.Module)}
    saved_module_names = set(saved_components.keys())

    if live_module_names != saved_module_names:
        missing_comp = live_module_names - saved_module_names
        extra_comp = saved_module_names - live_module_names
        if missing_comp:
            raise KeyError(f"checkpoint missing required component(s): {sorted(missing_comp)}")
        if extra_comp:
            raise KeyError(f"checkpoint has unexpected extra component(s): {sorted(extra_comp)}")

    for name in sorted(live_module_names):
        module = components[name]
        saved_dict = saved_components[name]
        if not isinstance(saved_dict, Mapping):
            raise ValueError(f"checkpoint component {name!r} state must be a mapping")
        live_dict = module.state_dict()
        live_keys = set(live_dict.keys())
        saved_keys = set(saved_dict.keys())
        if live_keys != saved_keys:
            missing_t = live_keys - saved_keys
            extra_t = saved_keys - live_keys
            if missing_t:
                raise KeyError(f"checkpoint component {name!r} missing expected tensor keys: {sorted(missing_t)}")
            if extra_t:
                raise KeyError(f"checkpoint component {name!r} has unexpected extra tensor keys: {sorted(extra_t)}")
        for t_name, live_t in live_dict.items():
            saved_t = saved_dict[t_name]
            if not torch.is_tensor(saved_t):
                raise TypeError(f"checkpoint component {name!r} tensor {t_name!r} is not a torch.Tensor")
            if saved_t.shape != live_t.shape:
                raise ValueError(
                    f"checkpoint component {name!r} tensor {t_name!r} shape mismatch: "
                    f"expected {live_t.shape}, got {saved_t.shape}"
                )
            if saved_t.dtype != live_t.dtype:
                raise TypeError(
                    f"checkpoint component {name!r} tensor {t_name!r} dtype mismatch: "
                    f"expected {live_t.dtype}, got {saved_t.dtype}"
                )

    # Validate optimizer param layout and per-parameter AdamW state
    if optimizer is not None:
        saved_opt = state["optimizer"]
        if saved_opt is None or not isinstance(saved_opt, Mapping):
            raise KeyError("checkpoint missing required optimizer state mapping")
        if "param_groups" not in saved_opt or "state" not in saved_opt:
            raise ValueError("checkpoint optimizer state missing 'param_groups' or 'state'")
        saved_groups = saved_opt["param_groups"]
        live_opt_sd = optimizer.state_dict()
        live_sd_groups = live_opt_sd["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(live_sd_groups):
            raise ValueError(
                f"checkpoint optimizer param_groups count mismatch: "
                f"expected {len(live_sd_groups)}, got {len(saved_groups) if isinstance(saved_groups, list) else type(saved_groups).__name__}"
            )

        # Build canonical ID -> Parameter mapping from live optimizer state dict
        id_to_param: dict[int, torch.nn.Parameter] = {}
        for g_idx, (live_opt_g, live_sd_g, saved_g) in enumerate(zip(optimizer.param_groups, live_sd_groups, saved_groups)):
            if not isinstance(saved_g, Mapping) or "params" not in saved_g:
                raise ValueError(f"checkpoint optimizer param_group {g_idx} missing 'params'")
            saved_params = list(saved_g["params"])
            expected_params = list(live_sd_g["params"])
            if saved_params != expected_params:
                raise ValueError(
                    f"checkpoint optimizer param_group {g_idx} params sequence mismatch: "
                    f"expected {expected_params}, got {saved_params}"
                )
            for p_obj, p_id in zip(live_opt_g["params"], live_sd_g["params"]):
                if p_id in id_to_param and id_to_param[p_id] is not p_obj:
                    raise ValueError(f"conflicting parameter mapping for param ID {p_id}")
                id_to_param[p_id] = p_obj

        saved_opt_state = saved_opt["state"]
        if not isinstance(saved_opt_state, Mapping):
            raise ValueError("checkpoint optimizer state['state'] must be a mapping")

        # Validate that all keys in saved_opt_state are valid integer param IDs
        for k in saved_opt_state.keys():
            try:
                k_int = int(k)
            except (ValueError, TypeError):
                raise ValueError(f"checkpoint optimizer state contains non-integer param key: {k!r}")
            if k_int not in id_to_param:
                raise KeyError(f"checkpoint optimizer state contains unknown param ID: {k_int}")

        # Validate per-parameter AdamW state compatibility via validated actual ID mappings
        for p_id, p in id_to_param.items():
            if p_id in saved_opt_state:
                p_state = saved_opt_state[p_id]
            elif str(p_id) in saved_opt_state:
                p_state = saved_opt_state[str(p_id)]
            else:
                p_state = None
            if p_state is not None:
                if not isinstance(p_state, Mapping):
                    raise ValueError(f"checkpoint optimizer state for param {p_id} must be a mapping")
                if "step" in p_state:
                    step_val = p_state["step"]
                    if torch.is_tensor(step_val):
                        if step_val.numel() != 1:
                            raise ValueError(f"checkpoint optimizer param {p_id} step tensor must be a scalar")
                    elif isinstance(step_val, bool) or not isinstance(step_val, (int, float)):
                        raise TypeError(f"checkpoint optimizer param {p_id} step must be scalar int or float, got {type(step_val).__name__}")
                if "exp_avg" in p_state:
                    exp_avg = p_state["exp_avg"]
                    if not torch.is_tensor(exp_avg):
                        raise TypeError(f"checkpoint optimizer param {p_id} exp_avg must be a torch.Tensor")
                    if exp_avg.shape != p.shape:
                        raise ValueError(
                            f"checkpoint optimizer param {p_id} exp_avg shape mismatch: "
                            f"expected {p.shape}, got {exp_avg.shape}"
                        )
                    if exp_avg.dtype != p.dtype:
                        raise TypeError(
                            f"checkpoint optimizer param {p_id} exp_avg dtype mismatch: "
                            f"expected {p.dtype}, got {exp_avg.dtype}"
                        )
                if "exp_avg_sq" in p_state:
                    exp_avg_sq = p_state["exp_avg_sq"]
                    if not torch.is_tensor(exp_avg_sq):
                        raise TypeError(f"checkpoint optimizer param {p_id} exp_avg_sq must be a torch.Tensor")
                    if exp_avg_sq.shape != p.shape:
                        raise ValueError(
                            f"checkpoint optimizer param {p_id} exp_avg_sq shape mismatch: "
                            f"expected {p.shape}, got {exp_avg_sq.shape}"
                        )
                    if exp_avg_sq.dtype != p.dtype:
                        raise TypeError(
                            f"checkpoint optimizer param {p_id} exp_avg_sq dtype mismatch: "
                            f"expected {p.dtype}, got {exp_avg_sq.dtype}"
                        )

    # Validate scheduler state
    if scheduler is not None and hasattr(scheduler, "load_state_dict"):
        saved_sched = state["scheduler"]
        if saved_sched is None or not isinstance(saved_sched, Mapping):
            raise KeyError("checkpoint missing required scheduler state mapping")
        if hasattr(scheduler, "state_dict"):
            live_sched_keys = set(scheduler.state_dict().keys())
            saved_sched_keys = set(saved_sched.keys())
            if not live_sched_keys.issubset(saved_sched_keys):
                missing_sched = live_sched_keys - saved_sched_keys
                raise KeyError(f"checkpoint scheduler missing expected keys: {sorted(missing_sched)}")

    # Validate RNG state
    if restore_rng:
        saved_rng = state["rng"]
        if saved_rng is None or not isinstance(saved_rng, Mapping):
            raise KeyError("checkpoint missing required rng state mapping")
        for rng_k in ("python", "numpy", "torch"):
            if rng_k not in saved_rng:
                raise KeyError(f"checkpoint rng state missing required {rng_k!r} state")
        if runner_rng is not None and "runner_rng" not in saved_rng:
            raise KeyError("checkpoint rng state missing required 'runner_rng' state")

    # Validate selection state
    _validate_selection_state(state.get("selection_state"), checkpoint_update=state.get("update"))

    # Validate sampler state for stateful sampler
    if datasets is not None:
        if hasattr(datasets, "sample_batch"):
            has_sd = callable(getattr(datasets, "state_dict", None))
            has_lsd = callable(getattr(datasets, "load_state_dict", None))
            if not (has_sd and has_lsd):
                missing = []
                if not has_sd:
                    missing.append("state_dict")
                if not has_lsd:
                    missing.append("load_state_dict")
                raise TypeError(
                    f"dataset object implementing sample_batch must provide callable 'state_dict' and 'load_state_dict', missing: {missing}"
                )
            if state.get("sampler_state") is None or not isinstance(state["sampler_state"], Mapping):
                raise KeyError("checkpoint missing required sampler_state mapping for stateful sampler")
        elif isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
            pass
        elif callable(datasets):
            raise TypeError("un-restorable stateful callable sampler is not supported; use an object implementing sample_batch with state_dict/load_state_dict or an immutable sequence")

    # --- ATOMIC APPLY PHASE WITH ROLLBACK ---
    # Take real snapshot (cloned tensors and deepcopied states) for clean rollback on late apply failure
    initial_components = {
        name: {k: v.clone() for k, v in components[name].state_dict().items()}
        for name in live_module_names
    }
    initial_optimizer = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    initial_scheduler = (
        copy.deepcopy(scheduler.state_dict())
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )
    initial_sampler = (
        copy.deepcopy(datasets.state_dict())
        if datasets is not None and hasattr(datasets, "state_dict")
        else None
    )
    initial_rng = _get_rng_state(runner_rng=runner_rng) if restore_rng else None

    try:
        # Mutate live component weights
        for name in sorted(live_module_names):
            components[name].load_state_dict(saved_components[name])

        # Restore optimizer
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])

        # Restore scheduler
        if scheduler is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler"])

        # Restore RNG state
        if restore_rng:
            _set_rng_state(state["rng"], runner_rng=runner_rng)

        # Restore sampler state
        if datasets is not None and hasattr(datasets, "load_state_dict") and state.get("sampler_state") is not None:
            datasets.load_state_dict(state["sampler_state"])
    except Exception:
        # Atomic rollback on unexpected apply failure
        for name, init_sd in initial_components.items():
            components[name].load_state_dict(init_sd)
        if optimizer is not None and initial_optimizer is not None:
            optimizer.load_state_dict(initial_optimizer)
        if scheduler is not None and initial_scheduler is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(initial_scheduler)
        if datasets is not None and initial_sampler is not None and hasattr(datasets, "load_state_dict"):
            datasets.load_state_dict(initial_sampler)
        if initial_rng is not None:
            _set_rng_state(initial_rng, runner_rng=runner_rng)
        raise

    return state

ALLOWED_SELECTION_STATUSES = frozenset({
    "no_evaluation",
    "certified",
    "not_certified",
    "diagnostic_external",
    "not_certified (bridge gate unavailable)",
})


def _validate_selection_state(
    selection_state: Any,
    *,
    stage: str | None = None,
    expected_schema: tuple[str, ...] | None = None,
    checkpoint_update: int | None = None,
) -> None:
    """Validate checkpoint selection_state structure, finiteness, and schema agreement."""
    if selection_state is None or not isinstance(selection_state, Mapping):
        raise KeyError("checkpoint missing required 'selection_state' mapping")
    for k in ("best_metric", "best_metric_vector", "selection_schema", "best_update", "patience_counter", "eval_history"):
        if k not in selection_state:
            raise KeyError(f"checkpoint selection_state missing required {k!r}")
    if "selection_status" not in selection_state:
        raise KeyError("checkpoint selection_state missing required 'selection_status'")

    status = selection_state["selection_status"]
    if status not in ALLOWED_SELECTION_STATUSES:
        raise ValueError(
            f"checkpoint selection_state has invalid/undocumented selection_status: {status!r}; "
            f"allowed values: {sorted(ALLOWED_SELECTION_STATUSES)}"
        )

    patience = selection_state["patience_counter"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ValueError(f"checkpoint selection_state patience_counter must be a non-negative integer, got {patience!r}")

    history = selection_state["eval_history"]
    if not isinstance(history, list):
        raise TypeError(f"checkpoint selection_state eval_history must be a list, got {type(history).__name__}")

    schema = selection_state["selection_schema"]
    best_vec = selection_state["best_metric_vector"]
    best_up = selection_state["best_update"]
    best_m = selection_state["best_metric"]

    if schema is not None:
        if not isinstance(schema, (list, tuple)) or len(schema) == 0:
            raise ValueError(f"checkpoint selection_schema must be a non-empty sequence of metric names, got {schema!r}")
        schema_tuple = tuple(str(x) for x in schema)
        if stage == "A0":
            if schema_tuple != ("normalized_reconstruction_loss",):
                raise ValueError(f"Stage A0 requires selection_schema ('normalized_reconstruction_loss',), got {schema_tuple!r}")
            if status == "certified":
                raise ValueError("Stage A0 cannot have selection_status 'certified' while bridge gate is unavailable")
        if stage == "A1" and schema_tuple not in (("physical_validation_loss",), ("physical_validation_loss", "dev_ranking")):
            raise ValueError(f"Stage A1 requires selection_schema ('physical_validation_loss',) or ('physical_validation_loss', 'dev_ranking'), got {schema_tuple!r}")
        if schema_tuple == ("physical_validation_loss", "dev_ranking") and status == "certified":
            raise ValueError("Stage A1 with external dev_ranking diagnostic cannot have selection_status 'certified'")

        # Validate every entry in eval_history
        for idx, entry in enumerate(history):
            if not isinstance(entry, Mapping):
                raise TypeError(f"eval_history[{idx}] must be a mapping, got {type(entry).__name__}")
            if "metric_vector" not in entry:
                raise KeyError(f"eval_history[{idx}] missing required 'metric_vector'")
            entry_vec = entry["metric_vector"]
            if not isinstance(entry_vec, (list, tuple)) or len(entry_vec) != len(schema_tuple):
                raise ValueError(
                    f"eval_history[{idx}] metric_vector arity mismatch: expected {len(schema_tuple)}, got {entry_vec!r}"
                )
            for val_idx, val in enumerate(entry_vec):
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"eval_history[{idx}] metric_vector[{val_idx}] is non-finite or invalid: {val!r}")
        if expected_schema is not None and schema_tuple != expected_schema:
            raise ValueError(f"checkpoint selection_schema {schema_tuple!r} does not match expected run schema {expected_schema!r}")

        if best_vec is not None:
            if status == "no_evaluation":
                raise ValueError("checkpoint with best_metric_vector cannot have selection_status 'no_evaluation'")
            if not isinstance(best_vec, (list, tuple)) or len(best_vec) != len(schema_tuple):
                raise ValueError(
                    f"checkpoint best_metric_vector arity mismatch: expected {len(schema_tuple)} elements matching schema, got {best_vec!r}"
                )
            for val in best_vec:
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"checkpoint best_metric_vector element is non-finite or invalid: {val!r}")
            if best_up is None or isinstance(best_up, bool) or not isinstance(best_up, int) or best_up < 1:
                raise ValueError(f"checkpoint best_update must be an integer >= 1 when best_metric_vector is present, got {best_up!r}")
            if checkpoint_update is not None and best_up > checkpoint_update:
                raise ValueError(f"checkpoint best_update ({best_up}) exceeds checkpoint update ({checkpoint_update})")

            if best_m is not None:
                if isinstance(best_m, bool) or not isinstance(best_m, (int, float)) or not math.isfinite(best_m):
                    raise ValueError(f"checkpoint best_metric must be a finite float, got {best_m!r}")
                if float(best_m) != float(best_vec[0]):
                    raise ValueError(f"checkpoint best_metric ({best_m}) does not match primary best_metric_vector[0] ({best_vec[0]}) exactly")
        else:
            # Schema is present, but no evaluations have completed before run end (e.g. max_updates < eval_interval)
            if status == "certified":
                raise ValueError("checkpoint selection_status cannot be 'certified' when best_metric_vector is None; use 'not_certified'")
            if status == "no_evaluation":
                raise ValueError("checkpoint with selection_schema cannot have selection_status 'no_evaluation'; use 'not_certified'")
            if patience != 0:
                raise ValueError(f"checkpoint patience_counter must be 0 when best_metric_vector is None, got {patience!r}")
            if len(history) != 0:
                raise ValueError(f"checkpoint eval_history must be empty when best_metric_vector is None, got {len(history)} entries")
            if best_up is not None:
                raise ValueError("checkpoint best_update must be None when best_metric_vector is None")
            if best_m is not None:
                raise ValueError("checkpoint best_metric must be None when best_metric_vector is None")
    else:
        if status != "no_evaluation":
            raise ValueError(f"checkpoint selection_state without selection_schema must have selection_status 'no_evaluation', got {status!r}")
        if patience != 0:
            raise ValueError(f"checkpoint selection_state patience_counter must be 0 when selection_schema is None, got {patience!r}")
        if len(history) != 0:
            raise ValueError(f"checkpoint selection_state eval_history must be empty when selection_schema is None, got {len(history)} entries")
        if best_vec is not None:
            raise ValueError("checkpoint best_metric_vector must be None when selection_schema is None")
        if best_up is not None:
            raise ValueError("checkpoint best_update must be None when selection_schema is None")
        if best_m is not None:
            raise ValueError("checkpoint best_metric must be None when selection_schema is None")
        if expected_schema is not None:
            raise ValueError(f"checkpoint has no selection_schema, but current run requires evaluation schema {expected_schema!r}")


def _preflight_resume_checkpoint(
    checkpoint_path: str | Path,
    requested_manifest: Mapping[str, Any],
    *,
    stage: str | None = None,
    expected_schema: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Inspect and validate resume checkpoint identity metadata before any live mutations occur."""
    target_path = Path(checkpoint_path).expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {target_path}")

    state = torch.load(target_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint missing state dict mapping")

    if "manifest" not in state:
        raise KeyError("checkpoint missing required key: 'manifest'")
    saved_manifest = state["manifest"]
    if not isinstance(saved_manifest, Mapping):
        raise ValueError("checkpoint manifest must be a mapping")

    # Validate resume manifest identity before any live mutation
    validate_resume(saved_manifest, requested_manifest)

    # Validate config envelope agreement
    if "config" not in state:
        raise KeyError("checkpoint missing required key: 'config'")
    saved_cfg = state["config"]
    if not isinstance(saved_cfg, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    req_cfg_id = requested_manifest.get("config_id")
    if req_cfg_id is not None:
        try:
            if "config" in saved_cfg and "config_sha256" in saved_cfg:
                saved_sha = saved_cfg["config_sha256"]
                if saved_sha != req_cfg_id:
                    raise ValueError(f"checkpoint config fingerprint mismatch: {saved_sha} != {req_cfg_id}")
            else:
                rec_cfg = MethodConfig.from_dict(dict(saved_cfg))
                if rec_cfg.fingerprint() != req_cfg_id:
                    raise ValueError("checkpoint config fingerprint mismatch")
        except Exception as e:
            raise ValueError(f"checkpoint config invalid: {e}") from e

    if "selected_seed" not in state:
        raise KeyError("checkpoint missing required key: 'selected_seed'")
    saved_seed = state["selected_seed"]
    if isinstance(saved_seed, bool) or not isinstance(saved_seed, int) or saved_seed < 0:
        raise ValueError(f"checkpoint selected_seed must be a non-negative integer, got {saved_seed!r}")
    if "selected_seed" not in saved_manifest:
        raise KeyError("checkpoint manifest missing required 'selected_seed'")
    man_seed = saved_manifest["selected_seed"]
    if isinstance(man_seed, bool) or not isinstance(man_seed, int) or man_seed < 0:
        raise ValueError(f"checkpoint manifest selected_seed must be a non-negative integer, got {man_seed!r}")
    if man_seed != saved_seed:
        raise ValueError(
            f"checkpoint top-level selected_seed ({saved_seed}) does not match manifest selected_seed ({man_seed})")

    if "selection_state" not in state:
        raise KeyError("checkpoint missing required key: 'selection_state'")
    _validate_selection_state(
        state["selection_state"],
        stage=stage,
        expected_schema=expected_schema,
        checkpoint_update=state.get("update"),
    )
    return dict(state)


def _validate_record_provenance(record: Any, *, is_eval: bool) -> tuple[str, str]:
    """Validate record provenance: strictly require split, episode_id, and lineage_id nonblank."""
    if not isinstance(record, Mapping):
        raise TypeError(f"dataset record must be a Mapping, got {type(record)!r}")
    if "provenance" not in record:
        raise ValueError("record missing required 'provenance' metadata")
    prov = record["provenance"]
    if not isinstance(prov, Mapping):
        raise TypeError(f"record provenance must be a Mapping, got {type(prov)!r}")

    if "split" not in prov:
        raise ValueError("record provenance missing required 'split' field; no optional split permitted")
    split = prov["split"]
    if not isinstance(split, str):
        raise TypeError(f"record provenance 'split' must be a str, got {type(split)!r}")

    expected_split = "dev" if is_eval else "train"
    if split != expected_split:
        raise ValueError(
            f"record provenance {split!r} is a forbidden/invalid split for "
            f"{'evaluation' if is_eval else 'training'}; requires {expected_split!r} partition only"
        )

    if "episode_id" not in prov:
        raise ValueError("record provenance missing required 'episode_id'")
    ep_id = str(prov["episode_id"]).strip()
    if not ep_id:
        raise ValueError("record provenance 'episode_id' must be a non-empty, non-whitespace string")

    if "lineage_id" not in prov:
        raise ValueError("record provenance missing required 'lineage_id'")
    lin_id = str(prov["lineage_id"]).strip()
    if not lin_id:
        raise ValueError("record provenance 'lineage_id' must be a non-empty, non-whitespace string")

    return ep_id, lin_id


def _sample_records(
    datasets: Any,
    *,
    stage: str,
    update: int,
    curriculum_k: int,
    rng: Any | None,
    batch_size: int,
    train_provenance_pairs: set[tuple[str, str]] | None = None,
) -> list[Any]:
    """Sample batch_size records using stateful sampler or deterministic sequence sampling."""
    if hasattr(datasets, "sample_batch"):
        # Exactly one call to sample_batch; internal TypeError propagates directly
        res = datasets.sample_batch(
            stage=stage,
            update=update,
            curriculum_k=curriculum_k,
            rng=rng,
            batch_size=batch_size,
        )
        if isinstance(res, Sequence) and not isinstance(res, (str, bytes, Mapping)):
            records = list(res)
        elif isinstance(res, Mapping):
            records = [res]
        else:
            raise TypeError(f"sample_batch returned unsupported type: {type(res)!r}")
        if len(records) != batch_size:
            raise ValueError(
                f"sample_batch returned {len(records)} records, expected exactly batch_size={batch_size}"
            )
    elif isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        if len(datasets) == 0:
            raise ValueError(f"dataset sequence is empty for stage {stage}")
        records = []
        for _ in range(batch_size):
            if rng is not None:
                if hasattr(rng, "integers"):
                    idx = int(rng.integers(0, len(datasets)))
                elif hasattr(rng, "randrange"):
                    idx = rng.randrange(len(datasets))
                elif hasattr(rng, "randint"):
                    idx = rng.randint(0, len(datasets) - 1)
                else:
                    idx = int(rng.choice(len(datasets)))
            else:
                idx = 0
            records.append(datasets[idx])
    elif callable(datasets):
        raise TypeError(
            "un-restorable stateful callable sampler is not supported; use an object implementing "
            "sample_batch with state_dict/load_state_dict or an immutable sequence"
        )
    else:
        raise ValueError(f"dataset is missing or invalid for stage {stage}")

    for rec in records:
        ep_id, lin_id = _validate_record_provenance(rec, is_eval=False)
        if train_provenance_pairs is not None and (ep_id, lin_id) not in train_provenance_pairs:
            raise ValueError(
                f"sampled record provenance pair {(ep_id, lin_id)!r} does not belong to predeclared training dataset provenance"
            )

    return records


def run_method_training(
    config: MethodConfig | Mapping[str, Any],
    components: Mapping[str, nn.Module],
    datasets: Any,
    *,
    limits: Mapping[str, Any] | None = None,
    stage: str | None = None,
    resume_checkpoint: str | Path | None = None,
    rng: Any | None = None,
    stop_after: int | None = None,
    output_dir: str | Path | None = None,
    dataset_id: str | None = None,
    reference_id: str | None = None,
    reference_manifest: Mapping[str, Any] | None = None,
    evaluation_dataset: Any | None = None,
    run_seed: int | None = None,
) -> TrainingReport:
    """Run staged method training bounded by limits for tiny synthetic checks or full training.

    Models must be passed already constructed; run_method_training does not reinitialize
    model weights or architectures. Caller owns model initialization seed and initial weights.

    Injected RNG contract: supports only numpy.random.Generator, which is reseeded to the
    selected run_seed on a fresh run, and restored from checkpoint on resume. Unsupported
    injected RNG objects are rejected with TypeError.

    Public runner support: only stages A0 and A1 are supported on the public runner.
    Stage D1 is gated with NotImplementedError pending P06/C verified frozen reference binding;
    D1 multi-record global selected-head/step normalization remains unresolved and uncertified.
    Stages B, C, D2, and E raise early NotImplementedError pending missing upstream dependencies.
    """
    cfg = config if isinstance(config, MethodConfig) else MethodConfig.from_dict(dict(config))
    active_stage = stage or "A1"

    # Early rejection of un-implemented/upstream-blocked stages
    if active_stage == "B":
        raise NotImplementedError(
            "Stage B training requires upstream event/task objective APIs "
            "(src/icgs/algorithms/objectives/task.py owned by P06 Task1C), which are not yet available."
        )
    if active_stage == "C":
        raise NotImplementedError(
            "Stage C reference certification requires upstream reference router / model weights binding "
            "(owned by P06), which is not yet available."
        )
    if active_stage == "D1":
        raise NotImplementedError(
            "Stage 'D1' is unavailable in public runtime pending P06/C verified "
            "frozen-reference binding; D1 multi-record global selected-head/step normalization unresolved, not certified."
        )
    if active_stage == "D2":
        raise NotImplementedError(
            "Stage D2 training requires upstream evaluator/terminal objective APIs "
            "(src/icgs/algorithms/objectives/evaluation.py owned by P09), which are not yet available."
        )
    if active_stage == "E":
        raise NotImplementedError(
            "Stage E temperature calibration requires upstream calibration API "
            "(owned by P09), which is not yet available."
        )

    # Resolve limits and strict ceilings (limits only permits max_updates; stop_after must be keyword arg)
    limits_map = dict(limits or {})
    if "stop_after" in limits_map:
        raise ValueError(
            "limits['stop_after'] is not permitted; use explicit stop_after keyword argument"
        )
    for k in limits_map:
        if k != "max_updates":
            raise ValueError(
                f"limits contains unrecorded/disallowed limit {k!r}; "
                "tunable overrides must be recorded in config, limits only permits 'max_updates'"
            )

    # Readiness: require declared training_seeds and metadata in config; caller-owned RNG cannot bypass
    training_seeds = cfg.stages.training_seeds
    if active_stage not in ("Test",):
        if training_seeds is None:
            raise ValueError(
                f"Stage {active_stage} requires declared stages.training_seeds in config; "
                "caller-owned rng cannot bypass seed readiness"
            )
        if (
            not isinstance(training_seeds, (tuple, list))
            or len(training_seeds) != cfg.stages.training_seed_count
            or len(set(training_seeds)) != len(training_seeds)
        ):
            raise ValueError(
                f"stages.training_seeds must contain {cfg.stages.training_seed_count} distinct seeds, got {training_seeds!r}"
            )
        for s_idx, s_val in enumerate(training_seeds):
            if isinstance(s_val, bool) or not isinstance(s_val, int) or s_val < 0:
                raise ValueError(f"stages.training_seeds[{s_idx}] must be a non-negative integer, got {s_val!r}")

        # Require generator_seed, reset_seed, and action_seed metadata
        for s_name in ("generator_seed", "reset_seed", "action_seed"):
            s_val = getattr(cfg.stages, s_name, None)
            if s_val is None or isinstance(s_val, bool) or not isinstance(s_val, int) or s_val < 0:
                raise ValueError(
                    f"Stage {active_stage} requires declared stages.{s_name} non-negative integer in config, got {s_val!r}"
                )

        selected_seed = run_seed if run_seed is not None else training_seeds[0]
        if selected_seed not in training_seeds:
            raise ValueError(
                f"selected run_seed {selected_seed} is not in configured stages.training_seeds: {training_seeds}"
            )
    else:
        selected_seed = run_seed

    # Trainable stages require datasets
    if datasets is None and active_stage not in ("Test",):
        raise ValueError(f"Stage {active_stage} requires a dataset, got None")

    # Explicit dataset_id requirement (reject missing or synthetic default)
    resolved_dataset_id = dataset_id
    if resolved_dataset_id is None:
        if hasattr(datasets, "dataset_id"):
            resolved_dataset_id = getattr(datasets, "dataset_id")
        elif isinstance(datasets, Mapping) and "dataset_id" in datasets:
            resolved_dataset_id = datasets["dataset_id"]

    if (
        resolved_dataset_id is None
        or not isinstance(resolved_dataset_id, str)
        or not resolved_dataset_id.strip()
        or resolved_dataset_id == "dataset-synthetic"
    ):
        raise ValueError(
            f"Stage {active_stage} requires explicit dataset_id; default 'dataset-synthetic' is rejected"
        )

    # Pre-reference stage identity protocol
    resolved_reference_id = reference_id
    if resolved_reference_id == "reference-synthetic":
        raise ValueError(
            f"Stage {active_stage} rejects default synthetic reference_id 'reference-synthetic'"
        )
    if resolved_reference_id is None:
        if active_stage in ("A0", "A1"):
            resolved_reference_id = f"pre-reference-{active_stage}"
        else:
            raise ValueError(f"Stage {active_stage} requires explicit reference_id")

    stage_cfg = getattr(cfg.stages, active_stage, None)
    configured_stage_max = getattr(stage_cfg, "max_updates", 1) if stage_cfg is not None else 1

    stage_max_updates = limits_map.get("max_updates")
    if stage_max_updates is None:
        stage_max_updates = configured_stage_max
    else:
        if isinstance(stage_max_updates, bool) or not isinstance(stage_max_updates, int) or stage_max_updates < 1:
            raise ValueError(f"limits['max_updates'] must be a positive integer, got {stage_max_updates!r}")
        if stage_max_updates > configured_stage_max:
            raise ValueError(
                f"requested max_updates {stage_max_updates} exceeds configured stage ceiling {configured_stage_max}"
            )

    eval_interval = cfg.stages.evaluation_interval_updates
    if isinstance(eval_interval, bool) or not isinstance(eval_interval, int) or eval_interval < 1:
        raise ValueError(f"stages.evaluation_interval_updates must be a positive integer, got {eval_interval!r}")

    save_every = eval_interval
    if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
        raise ValueError(f"save cadence must be a positive integer, got {save_every!r}")

    # Preflight datasets / sampler contract and predeclared provenance BEFORE any component mutation
    train_episode_ids: set[str] = set()
    train_lineage_ids: set[str] = set()
    train_provenance_pairs: set[tuple[str, str]] = set()

    if datasets is not None:
        if hasattr(datasets, "sample_batch"):
            if not callable(datasets.sample_batch):
                raise TypeError("dataset.sample_batch must be callable")
            sig = inspect.signature(datasets.sample_batch)
            try:
                sig.bind("A1", 0, 1, rng=None, batch_size=1)
            except TypeError as exc:
                raise TypeError(
                    f"dataset.sample_batch signature must accept (stage, update, curriculum_k, *, rng, batch_size): {exc}"
                ) from exc
            has_sd = callable(getattr(datasets, "state_dict", None))
            has_lsd = callable(getattr(datasets, "load_state_dict", None))
            if not (has_sd and has_lsd):
                missing = []
                if not has_sd:
                    missing.append("state_dict")
                if not has_lsd:
                    missing.append("load_state_dict")
                raise TypeError(
                    f"dataset object implementing sample_batch must provide callable 'state_dict' and 'load_state_dict', missing: {missing}"
                )
            if not hasattr(datasets, "records") or not isinstance(datasets.records, Sequence) or isinstance(datasets.records, (str, bytes)):
                raise TypeError(
                    "stateful train sampler must provide inspectable .records Sequence for provenance validation"
                )
            train_records_source = datasets.records
        elif isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
            train_records_source = datasets
        elif callable(datasets):
            raise TypeError(
                "un-restorable stateful callable sampler is not supported; use an object implementing sample_batch with state_dict/load_state_dict or an immutable sequence"
            )
        else:
            raise TypeError(f"unsupported dataset type: {type(datasets)!r}")

        if len(train_records_source) == 0:
            raise ValueError("training dataset records source is empty")

        for rec in train_records_source:
            ep_id, lin_id = _validate_record_provenance(rec, is_eval=False)
            train_episode_ids.add(ep_id)
            train_lineage_ids.add(lin_id)
            train_provenance_pairs.add((ep_id, lin_id))

    # Evaluation dataset validation (immutable EVAL records only, reject stateful evaluation)
    eval_episode_ids: set[str] = set()
    eval_lineage_ids: set[str] = set()
    eval_snapshot: tuple[Any, ...] | None = None

    if evaluation_dataset is not None:
        # Snapshot finite evaluation sequence into tuple once before updates and use throughout
        if hasattr(evaluation_dataset, "sample_batch") or callable(evaluation_dataset):
            raise TypeError(
                "stateful evaluation dataset is not supported; immutable evaluation records sequence required"
            )
        if not isinstance(evaluation_dataset, Sequence) or isinstance(evaluation_dataset, (str, bytes)):
            raise TypeError("evaluation dataset must be an immutable Sequence of records")
        if len(evaluation_dataset) == 0:
            raise ValueError("evaluation dataset is empty")
        eval_snapshot = tuple(evaluation_dataset)
        for rec in eval_snapshot:
            ep_id, lin_id = _validate_record_provenance(rec, is_eval=True)
            eval_episode_ids.add(ep_id)
            eval_lineage_ids.add(lin_id)

        overlap_ep = train_episode_ids & eval_episode_ids
        if overlap_ep:
            raise ValueError(
                f"train and evaluation datasets share overlapping episode IDs: {sorted(overlap_ep)}"
            )
        # Validate literal lineage split disjointness via P02 validate_split_lineage helper
        # (Note: P02 does not provide descendant closure, so literal IDs are validated directly).
        lineage_rows = [{"lineage_id": lin, "split": "train"} for lin in train_lineage_ids] + [
            {"lineage_id": lin, "split": "dev"} for lin in eval_lineage_ids
        ]
        try:
            validate_split_lineage(lineage_rows)
        except ValueError as exc:
            raise ValueError(
                f"train and evaluation datasets share overlapping lineage IDs: {sorted(train_lineage_ids & eval_lineage_ids)}"
            ) from exc

    # Preflight resume checkpoint identity before ANY live mutation (output writes, RNG, freeze, optimizer)
    preloaded_checkpoint: dict[str, Any] | None = None

    # Predeclare selection schema before any update begins
    selection_schema: tuple[str, ...] | None = None
    if eval_snapshot is not None:
        if active_stage == "A0":
            for rec in eval_snapshot:
                if isinstance(rec, Mapping) and "dev_ranking" in rec:
                    raise ValueError("Stage A0 does not support dev_ranking in evaluation dataset")
            selection_schema = ("normalized_reconstruction_loss",)
        elif active_stage == "A1":
            dev_ranking_values: list[float] = []
            for rec in eval_snapshot:
                if isinstance(rec, Mapping) and "dev_ranking" in rec:
                    raw_r = rec["dev_ranking"]
                    if raw_r is not None:
                        if isinstance(raw_r, bool) or not isinstance(raw_r, (int, float)) or not math.isfinite(raw_r):
                            raise ValueError(f"dev_ranking must be a finite float, got {raw_r!r}")
                        dev_ranking_values.append(float(raw_r))

            if len(dev_ranking_values) == len(eval_snapshot) and len(eval_snapshot) > 0:
                first_val = dev_ranking_values[0]
                if not all(v == first_val for v in dev_ranking_values):
                    raise ValueError(
                        f"varying dev_ranking values across evaluation records rejected; "
                        f"supplied diagnostic requires identical exact value across all records, got {dev_ranking_values!r}"
                    )
                selection_schema = ("physical_validation_loss", "dev_ranking")
            elif len(dev_ranking_values) == 0:
                selection_schema = ("physical_validation_loss",)
            else:
                raise ValueError(
                    "partial/mixed dev_ranking across evaluation dataset records is rejected; "
                    "dev_ranking must be uniformly supplied across all records with identical value, or omitted completely"
                )
    if resume_checkpoint is not None:
        requested_manifest = {
            "schema_version": 1,
            "stage": active_stage,
            "reference_id": resolved_reference_id,
            "dataset_id": resolved_dataset_id,
            "config_id": cfg.fingerprint(),
            "total_updates": stage_max_updates,
            "selected_seed": selected_seed,
        }
        preloaded_checkpoint = _preflight_resume_checkpoint(
            resume_checkpoint,
            requested_manifest,
            stage=active_stage,
            expected_schema=selection_schema,
        )

    target_out_dir = Path(output_dir) if output_dir is not None else (
        Path(cfg.training.output_dir) if cfg.training.output_dir is not None else None
    )

    # Write resolved_config.json before execution if output requested (envelope equality checked)
    resolved_cfg_dict = cfg.resolved_config()
    if target_out_dir is not None:
        target_out_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg_file = target_out_dir / "resolved_config.json"
        resolved_cfg_file.write_text(
            json.dumps(resolved_cfg_dict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        loaded = json.loads(resolved_cfg_file.read_text(encoding="utf-8"))
        if loaded != resolved_cfg_dict:
            raise RuntimeError("resolved_config.json persistence mismatch against cfg.resolved_config()")

    if resume_checkpoint is None:
        if selected_seed is not None:
            random.seed(selected_seed)
            np.random.seed(selected_seed)
            torch.manual_seed(selected_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(selected_seed)
            if rng is not None:
                if not isinstance(rng, np.random.Generator):
                    raise TypeError(f"injected rng must be a numpy.random.Generator, got {type(rng)!r}")
                rng.__setstate__(np.random.default_rng(selected_seed).__getstate__())
                runner_rng = rng
            else:
                runner_rng = np.random.default_rng(selected_seed)
        else:
            runner_rng = rng if rng is not None else np.random.default_rng()
    else:
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError(f"injected rng must be a numpy.random.Generator, got {type(rng)!r}")
        runner_rng = rng if rng is not None else np.random.default_rng()

    trainable = trainable_components(active_stage)

    # Freeze non-trainable components and validate role requirements
    apply_freeze_boundary(active_stage, components)

    if active_stage in ("Test",) or not trainable:
        return TrainingReport(
            stage=active_stage,
            updates_completed=0,
            loss_history=(),
            eval_metrics={},
            stopped_early=False,
            training_seeds=cfg.stages.training_seeds,
            selected_seed=selected_seed,
        )

    # Trainable stages (A0, A1, D1)
    optimizer = build_stage_optimizer(active_stage, components, cfg)
    if optimizer is None:
        raise RuntimeError(f"Failed to build optimizer for trainable stage {active_stage}")

    scheduler = build_stage_scheduler(optimizer, cfg, max_updates=stage_max_updates)

    stage_batch_size = getattr(stage_cfg, "batch_size", 1) if stage_cfg is not None else 1
    if isinstance(stage_batch_size, bool) or not isinstance(stage_batch_size, int) or stage_batch_size < 1:
        raise ValueError(f"stages.{active_stage}.batch_size must be a positive integer, got {stage_batch_size!r}")

    start_update = 0
    best_metric_vector: tuple[float, ...] | None = None
    best_update: int | None = None
    patience_counter = 0
    eval_history: list[dict[str, Any]] = []
    accumulated_loss_history: list[float] = []
    patience_limit = int(cfg.stages.early_stopping_patience)
    stopped_early = False
    if resume_checkpoint is not None:
        loaded_state = load_checkpoint(
            preloaded_checkpoint,
            components=components,
            optimizer=optimizer,
            scheduler=scheduler,
            requested_manifest=requested_manifest,
            restore_rng=True,
            runner_rng=runner_rng,
            datasets=datasets,
        )
        start_update = int(loaded_state.get("update", 0))
        sel_state = loaded_state.get("selection_state") or {}
        if "best_metric_vector" in sel_state and sel_state["best_metric_vector"] is not None:
            best_metric_vector = tuple(float(x) for x in sel_state["best_metric_vector"])
        if "best_update" in sel_state and sel_state["best_update"] is not None:
            best_update = int(sel_state["best_update"])
        patience_counter = int(sel_state.get("patience_counter", 0))
        eval_history = list(sel_state.get("eval_history", []))
        if loaded_state.get("accumulated_loss_history"):
            accumulated_loss_history = list(loaded_state["accumulated_loss_history"])
        if patience_counter >= patience_limit:
            stopped_early = True

    if stopped_early:
        return TrainingReport(
            stage=active_stage,
            updates_completed=0,
            loss_history=(),
            eval_metrics=eval_history[-1] if eval_history else {},
            stopped_early=True,
            checkpoint_path=str(Path(resume_checkpoint).expanduser().resolve()) if resume_checkpoint is not None else None,
            training_seeds=cfg.stages.training_seeds,
            selected_seed=selected_seed,
            status="stopped_early",
            scheduler_state=scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            accumulated_loss_history=tuple(accumulated_loss_history),
            selection_schema=selection_schema,
            best_metric_vector=best_metric_vector,
            best_update=best_update,
            selection_status=(
                "no_evaluation"
                if selection_schema is None
                else (
                    "not_certified (bridge gate unavailable)"
                    if active_stage == "A0"
                    else ("diagnostic_external" if selection_schema == ("physical_validation_loss", "dev_ranking") else ("certified" if best_metric_vector is not None else "not_certified"))
                )
            ),
            total_updates_completed=start_update,
        )

    run_ceiling = stage_max_updates
    if stop_after is not None:
        if isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 1 or stop_after > stage_max_updates:
            raise ValueError(f"stop_after must be a positive integer <= max_updates ({stage_max_updates}), got {stop_after!r}")
        run_ceiling = stop_after

    loss_history: list[float] = []
    last_saved_path: str | None = None

    grad_clip = float(cfg.optimizer.gradient_clip_norm)

    def _current_sampler_state(ds: Any, next_u: int) -> Any:
        if hasattr(ds, "state_dict"):
            return ds.state_dict()
        elif isinstance(ds, Sequence) and not isinstance(ds, (str, bytes)):
            return {"type": "sequence", "next_update": next_u}
        return None

    for current_update in range(start_update, run_ceiling):
        curriculum_k = (
            rollout_horizon(active_stage, current_update, stage_max_updates, config=cfg)
            if active_stage in ("A1", "D1")
            else 1
        )

        loss_val = _execute_training_step(
            stage=active_stage,
            components=components,
            optimizer=optimizer,
            scheduler=scheduler,
            datasets=datasets,
            curriculum_k=curriculum_k,
            grad_clip=grad_clip,
            update=current_update,
            config=cfg,
            rng=runner_rng,
            batch_size=stage_batch_size,
            train_provenance_pairs=train_provenance_pairs,
        )
        loss_history.append(loss_val)
        accumulated_loss_history.append(loss_val)

        # Lexicographic evaluation and early stopping
        if eval_snapshot is not None and ((current_update + 1) % eval_interval == 0):
            eval_metrics = _evaluate_stage(active_stage, components, eval_snapshot, cfg, expected_schema=selection_schema)
            eval_history.append(eval_metrics)
            candidate_vector = tuple(eval_metrics["metric_vector"])
            for mv in candidate_vector:
                if isinstance(mv, bool) or not isinstance(mv, (int, float)) or not math.isfinite(mv):
                    raise ValueError(f"evaluation metric in candidate_vector is non-finite or invalid: {mv!r}")

            is_strictly_better = False
            if best_metric_vector is None:
                is_strictly_better = True
            else:
                is_strictly_better = (candidate_vector < best_metric_vector)

            if is_strictly_better:
                best_metric_vector = candidate_vector
                best_update = current_update + 1
                patience_counter = 0
                if active_stage == "A0":
                    current_selection_status = "not_certified (bridge gate unavailable)"
                elif selection_schema == ("physical_validation_loss", "dev_ranking"):
                    current_selection_status = "diagnostic_external"
                else:
                    current_selection_status = "certified"

                # Only write _best.pt if stage certification gate passes (must be certified)
                if target_out_dir is not None and current_selection_status == "certified":
                    save_checkpoint(
                        target_out_dir / f"checkpoint_{active_stage}_best.pt",
                        stage=active_stage,
                        update=current_update + 1,
                        total_updates=stage_max_updates,
                        components=components,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=cfg,
                        dataset_id=resolved_dataset_id,
                        reference_id=resolved_reference_id,
                        selected_seed=selected_seed,
                        rng=runner_rng,
                        sampler_state=_current_sampler_state(datasets, current_update + 1),
                        selection_state={
                            "best_metric": best_metric_vector[0],
                            "best_metric_vector": list(best_metric_vector),
                            "selection_schema": list(selection_schema) if selection_schema else None,
                            "best_update": best_update,
                            "patience_counter": patience_counter,
                            "eval_history": eval_history,
                            "selection_status": "certified",
                        },
                        accumulated_loss_history=accumulated_loss_history,
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    stopped_early = True

        # Checkpoint saving
        if target_out_dir is not None and (
            (current_update + 1) == run_ceiling
            or (current_update + 1) % save_every == 0
            or stopped_early
        ):
            ckpt_file = target_out_dir / f"checkpoint_{active_stage}_{current_update + 1}.pt"
            if selection_schema is None:
                stage_sel_status = "no_evaluation"
            elif best_metric_vector is None:
                stage_sel_status = (
                    "not_certified (bridge gate unavailable)"
                    if active_stage == "A0"
                    else "not_certified"
                )
            elif active_stage == "A0":
                stage_sel_status = "not_certified (bridge gate unavailable)"
            elif selection_schema == ("physical_validation_loss", "dev_ranking"):
                stage_sel_status = "diagnostic_external"
            else:
                stage_sel_status = "certified"
            last_saved_path = save_checkpoint(
                ckpt_file,
                stage=active_stage,
                update=current_update + 1,
                total_updates=stage_max_updates,
                components=components,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                dataset_id=resolved_dataset_id,
                reference_id=resolved_reference_id,
                selected_seed=selected_seed,
                rng=runner_rng,
                sampler_state=_current_sampler_state(datasets, current_update + 1),
                selection_state={
                    "best_metric": best_metric_vector[0] if best_metric_vector is not None else None,
                    "best_metric_vector": list(best_metric_vector) if best_metric_vector is not None else None,
                    "selection_schema": list(selection_schema) if selection_schema else None,
                    "best_update": best_update,
                    "patience_counter": patience_counter,
                    "eval_history": eval_history,
                    "selection_status": stage_sel_status,
                },
                accumulated_loss_history=accumulated_loss_history,
            )

        if stopped_early:
            break

    return TrainingReport(
        stage=active_stage,
        updates_completed=len(loss_history),
        loss_history=tuple(loss_history),
        eval_metrics=eval_history[-1] if eval_history else {},
        stopped_early=stopped_early,
        checkpoint_path=last_saved_path,
        training_seeds=cfg.stages.training_seeds,
        selected_seed=selected_seed,
        status="stopped_early" if stopped_early else "completed",
        scheduler_state=scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        accumulated_loss_history=tuple(accumulated_loss_history),
        selection_schema=selection_schema,
        best_metric_vector=best_metric_vector,
        best_update=best_update,
        selection_status=(
            "no_evaluation"
            if selection_schema is None
            else (
                "not_certified (bridge gate unavailable)"
                if active_stage == "A0"
                else ("diagnostic_external" if selection_schema == ("physical_validation_loss", "dev_ranking") else ("certified" if best_metric_vector is not None else "not_certified"))
            )
        ),
        total_updates_completed=start_update + len(loss_history),
    )


def _execute_training_step(
    *,
    stage: str,
    components: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    datasets: Any,
    curriculum_k: int,
    grad_clip: float,
    update: int,
    config: MethodConfig,
    rng: Any | None,
    batch_size: int = 1,
    train_provenance_pairs: set[tuple[str, str]] | None = None,
) -> float:
    """Execute a single step consuming configured batch_size independent records."""
    optimizer.zero_grad()

    records = _sample_records(
        datasets,
        stage=stage,
        update=update,
        curriculum_k=curriculum_k,
        rng=rng,
        batch_size=batch_size,
        train_provenance_pairs=train_provenance_pairs,
    )

    trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]

    total_loss = None
    valid_count = 0
    for record in records:
        _validate_record_provenance(record, is_eval=False)

        if stage == "A1":
            step_loss = _step_a1_rollout(components, record, curriculum_k, config)
        elif stage == "A0":
            step_loss = _step_a0_reconstruction(components, record, config)
        elif stage == "D1":
            step_loss = _step_d1_rollout(components, record, curriculum_k, config)
        elif stage == "B":
            raise NotImplementedError(
                "Stage B training requires upstream event/task objective APIs "
                "(src/icgs/algorithms/objectives/task.py owned by P06 Task1C), which are not yet available."
            )
        elif stage == "D2":
            raise NotImplementedError(
                "Stage D2 training requires upstream evaluator/terminal objective APIs "
                "(src/icgs/algorithms/objectives/evaluation.py owned by P09), which are not yet available."
            )
        else:
            raise ValueError(f"unsupported trainable stage: {stage}")

        if not torch.is_tensor(step_loss) or not torch.isfinite(step_loss).all():
            raise FloatingPointError(f"nonfinite loss in stage {stage} at update {update}: {step_loss}")

        if total_loss is None:
            total_loss = step_loss
        else:
            total_loss = total_loss + step_loss
        valid_count += 1

    if valid_count == 0 or total_loss is None:
        raise ValueError(f"no valid records in batch for stage {stage}")

    avg_loss = total_loss / float(valid_count)

    avg_loss.backward()
    if grad_clip > 0 and trainable_params:
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
    optimizer.step()
    if scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step()

    return float(avg_loss.detach().item())


def _validate_transitions_and_reconstruct_reset(
    transitions: Sequence[ExecutedTransition],
    encoder: nn.Module,
    config: MethodConfig,
    *,
    reset_state: PhysicalState | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[PhysicalState, Sequence[ExecutedTransition]]:
    """Validate full ordered ExecutedTransition sequence and reconstruct reset state."""
    if not isinstance(transitions, Sequence) or len(transitions) == 0:
        raise ValueError("transitions must be a nonempty sequence of ExecutedTransition")

    t0 = transitions[0]
    if not isinstance(t0, ExecutedTransition):
        raise TypeError("transitions entries must be ExecutedTransition records")
    if t0.before.boundary != 0:
        raise ValueError(f"transitions must start at boundary 0, got {t0.before.boundary}")

    # Inspect/validate passed reset_state (required calibrated gravity carrier)
    if reset_state is None or not isinstance(reset_state, PhysicalState):
        raise TypeError(
            "reset_state is required as the calibrated gravity carrier and must be a PhysicalState, "
            f"got {type(reset_state).__name__ if reset_state is not None else 'None'}"
        )
    if reset_state.boundary != 0:
        raise ValueError(f"reset_state must start at boundary 0, got {reset_state.boundary}")
    if bool((reset_state.memory.abs() > 0).any().item()):
        raise ValueError("reset state hidden memory must be zero")
    if (
        reset_state.p is None
        or not torch.is_tensor(reset_state.p)
        or reset_state.p.ndim != 2
        or reset_state.p.shape[-1] != 13
        or not bool(torch.isfinite(reset_state.p).all().item())
    ):
        raise ValueError("reset_state.p must be a finite 2D tensor with last dimension 13 carrying calibrated gravity")
    gravity = reset_state.p[:, 10:13].to(device=device, dtype=dtype)

    # Validate full ordered ExecutedTransition continuity and stored before pose
    for i, trans in enumerate(transitions):
        if not isinstance(trans, ExecutedTransition):
            raise TypeError(f"transition {i} must be an ExecutedTransition")
        if trans.before.boundary != i:
            raise ValueError(f"transition {i} before boundary is {trans.before.boundary}, expected {i}")
        if trans.after.boundary != i + 1:
            raise ValueError(f"transition {i} after boundary is {trans.after.boundary}, expected {i + 1}")

        before_obs = trans.before.observation
        before_pose = before_obs.T_w_e
        if not isinstance(before_pose, np.ndarray) or before_pose.shape != (4, 4) or not np.all(np.isfinite(before_pose)):
            raise ValueError(f"transition {i} before observation must have explicit finite 4x4 pose")

        if i > 0:
            prev_after = transitions[i - 1].after.observation
            if not np.array_equal(prev_after.T_w_e, before_obs.T_w_e):
                raise ValueError(f"discontinuous pose between transition {i-1} after and {i} before")
            if float(prev_after.grip) != float(before_obs.grip):
                raise ValueError(f"discontinuous grip between transition {i-1} after and {i} before")
            if not np.array_equal(prev_after.points, before_obs.points):
                raise ValueError(f"discontinuous points between transition {i-1} after and {i} before")

    # Reconstruct RESET encoded geometry/proprioception from t0.before measured observation under current encoder
    obs0 = t0.before.observation
    pts = torch.tensor(obs0.points, dtype=dtype, device=device)
    if pts.ndim == 2:
        pts = pts.unsqueeze(0)
    pts_valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)

    with torch.no_grad():
        encoded = encoder(pts, pts_valid)

    pose = torch.tensor(obs0.T_w_e, dtype=dtype, device=device)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    grip = torch.as_tensor([[float(obs0.grip)]], dtype=dtype, device=device)
    p = proprioception(pose, grip, gravity, config=config)

    # Initialize physical memory per P04 reset semantics: zero memory
    memory = torch.zeros((1, config.memory.slots, config.memory.width), dtype=dtype, device=device)

    reconstructed_reset = PhysicalState(
        X=encoded.X,
        x=encoded.x,
        valid=encoded.anchor_valid,
        p=p,
        memory=memory,
        T_w_e=pose,
        grip=grip,
        cached_world_cloud=pts,
        cached_world_cloud_valid=pts_valid,
        boundary=0,
        encoder_lineage="current",
        memory_lineage="current",
        origin="real",
    )
    return reconstructed_reset, transitions


def _replay_history_and_detach(
    *,
    reset_state: PhysicalState,
    transitions: Sequence[ExecutedTransition],
    boundary: int,
    encoder: nn.Module,
    physical_memory: Any,
    config: MethodConfig,
    step_listener: Any | None = None,
) -> PhysicalState:
    """Reconstruct causal history from reset observation and ordered executed transitions.

    Replays all pre-supervision measured transitions under torch.no_grad(),
    and detaches hidden state at the explicit supervision boundary.
    """
    if reset_state is None or not isinstance(reset_state, PhysicalState):
        raise TypeError(
            "reset_state is required as the calibrated gravity carrier and must be a PhysicalState, "
            f"got {type(reset_state).__name__ if reset_state is not None else 'None'}"
        )
    if reset_state.boundary != 0:
        raise ValueError(f"reset_state must start at boundary 0, got {reset_state.boundary}")
    if (
        reset_state.p is None
        or not torch.is_tensor(reset_state.p)
        or reset_state.p.ndim != 2
        or reset_state.p.shape[-1] != 13
        or not bool(torch.isfinite(reset_state.p).all().item())
    ):
        raise ValueError("reset_state.p must be a finite 2D tensor with last dimension 13 carrying calibrated gravity")
    if not isinstance(boundary, int) or boundary < 0:
        raise ValueError(f"boundary must be a nonnegative integer, got {boundary!r}")
    if len(transitions) < boundary:
        raise ValueError(
            f"transitions sequence length {len(transitions)} insufficient for boundary {boundary}"
        )

    current_state = reset_state.branch_copy()

    with torch.no_grad():
        for step_idx in range(0, boundary):
            if step_listener is not None:
                step_listener(
                    step_idx=step_idx,
                    is_prefix=True,
                    grad_enabled=torch.is_grad_enabled(),
                )

            trans = transitions[step_idx]
            before_pose = current_state.T_w_e
            u_t = action_descriptor(before_pose, trans.command, config=config)

            obs_after = trans.after.observation
            pts = torch.tensor(obs_after.points, dtype=current_state.X.dtype, device=current_state.X.device)
            if pts.ndim == 2:
                pts = pts.unsqueeze(0)
            pts_valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=current_state.X.device)

            encoded_next = encoder(pts, pts_valid)

            pose_next = torch.tensor(obs_after.T_w_e, dtype=current_state.X.dtype, device=current_state.X.device)
            if pose_next.ndim == 2:
                pose_next = pose_next.unsqueeze(0)

            grip_next = torch.as_tensor(obs_after.grip, dtype=current_state.X.dtype, device=current_state.X.device).reshape(-1, 1)

            gravity = current_state.p[:, 10:13]
            p_next = proprioception(pose_next, grip_next, gravity, config=config)

            memory_next = physical_memory(
                encoded_next.X,
                encoded_next.anchor_valid,
                p_next,
                u_t,
                current_state.memory,
            )

            current_state = PhysicalState(
                X=encoded_next.X,
                x=encoded_next.x,
                valid=encoded_next.anchor_valid,
                p=p_next,
                memory=memory_next,
                T_w_e=pose_next,
                grip=grip_next,
                cached_world_cloud=encoded_next.x,
                cached_world_cloud_valid=encoded_next.anchor_valid,
                boundary=step_idx + 1,
                encoder_lineage=current_state.encoder_lineage,
                memory_lineage=current_state.memory_lineage,
                origin="real",
            )

    # Detach hidden state at the explicit supervision boundary
    state_at_boundary = PhysicalState(
        X=current_state.X.detach(),
        x=current_state.x.detach(),
        valid=current_state.valid,
        p=current_state.p.detach(),
        memory=current_state.memory.detach(),
        T_w_e=current_state.T_w_e.detach(),
        grip=current_state.grip.detach(),
        cached_world_cloud=None if current_state.cached_world_cloud is None else current_state.cached_world_cloud.detach(),
        cached_world_cloud_valid=current_state.cached_world_cloud_valid,
        boundary=current_state.boundary,
        encoder_lineage=current_state.encoder_lineage,
        memory_lineage=current_state.memory_lineage,
        origin="imagined",
    )
    return state_at_boundary


def _step_a1_rollout(
    components: Mapping[str, nn.Module],
    batch: Any,
    curriculum_k: int,
    config: MethodConfig,
) -> Tensor:
    """Execute Stage A1 with prefix replay under no_grad, memory detach, and supervised unroll."""
    memory = components.get("physical_memory")
    dynamics = components.get("dynamics")
    geometry = components.get("geometry")

    if memory is None or dynamics is None or geometry is None:
        raise KeyError("Stage A1 requires 'geometry', 'physical_memory', and 'dynamics' components")

    encoder, decoder = get_geometry_encoder_decoder(geometry)

    num_supervised = config.stages.A1.supervised_intervals
    burnin = config.stages.A1.burnin_intervals

    if isinstance(batch, Mapping):
        if "supervised_intervals" in batch and batch["supervised_intervals"] != num_supervised:
            raise ValueError(
                f"batch supervised_intervals ({batch['supervised_intervals']}) conflicts with configured stage value ({num_supervised})"
            )
        if "burnin" in batch and batch["burnin"] != burnin:
            raise ValueError(
                f"batch burnin ({batch['burnin']}) conflicts with configured stage value ({burnin})"
            )
        passed_reset_state = batch.get("reset_state") or batch.get("state")
        transitions = batch.get("transitions")
        boundary = batch.get("boundary", 0)
        step_listener = batch.get("step_listener")
    else:
        passed_reset_state = getattr(batch, "reset_state", None) or getattr(batch, "state", None)
        transitions = getattr(batch, "transitions", None)
        boundary = getattr(batch, "boundary", 0)
        step_listener = getattr(batch, "step_listener", None)

    if transitions is None:
        raise ValueError("Stage A1 batch must contain 'transitions' sequence")

    if len(transitions) < boundary + num_supervised:
        raise ValueError(
            f"transitions sequence length {len(transitions)} is insufficient for boundary {boundary} + supervised_intervals {num_supervised}"
        )

    encoder_param = next(encoder.parameters(), None)
    device = encoder_param.device if encoder_param is not None else torch.device("cpu")
    dtype = encoder_param.dtype if encoder_param is not None else torch.float32

    reset_state, transitions = _validate_transitions_and_reconstruct_reset(
        transitions,
        encoder,
        config,
        reset_state=passed_reset_state,
        device=device,
        dtype=dtype,
    )

    state_at_boundary = _replay_history_and_detach(
        reset_state=reset_state,
        transitions=transitions,
        boundary=boundary,
        encoder=encoder,
        physical_memory=memory,
        config=config,
        step_listener=step_listener,
    )

    rollout = getattr(batch, "rollout", None)
    if rollout is None:
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=config)

    # Observed sequence roots vs K-step recursive rollout
    root_offsets = list(range(0, num_supervised - curriculum_k + 1, curriculum_k))
    if not root_offsets:
        root_offsets = [0]

    all_predictions = []
    all_targets = []
    for root_offset in root_offsets:
        if root_offset == 0:
            root_state = state_at_boundary
        else:
            root_state = _replay_history_and_detach(
                reset_state=reset_state,
                transitions=transitions,
                boundary=boundary + root_offset,
                encoder=encoder,
                physical_memory=memory,
                config=config,
            )

        target_steps = [transitions[boundary + root_offset + step] for step in range(curriculum_k)]
        commands = [target_trans.command for target_trans in target_steps]
        preds = rollout.rollout(root_state, commands, head_id=0)

        all_predictions.append([list(preds), [], []])
        all_targets.append(target_steps)

    # Reconstruction loss on measured points under current encoder and decoder
    t_meas = transitions[boundary].after.observation
    pts_meas = torch.tensor(t_meas.points, dtype=dtype, device=device)
    if pts_meas.ndim == 2:
        pts_meas = pts_meas.unsqueeze(0)
    pts_meas_valid = torch.ones(pts_meas.shape[:2], dtype=torch.bool, device=device)
    encoded_meas = encoder(pts_meas, pts_meas_valid)
    decoded_meas = decoder(encoded_meas)
    rec_loss = masked_normalized_chamfer_distance(
        decoded_meas.points_w,
        pts_meas,
        decoded_meas.point_valid,
        pts_meas_valid,
        config=config,
    )

    R = len(root_offsets)
    bootstrap = torch.tensor([[True, False, False]]).expand(R, 3)
    supervision = torch.ones((R, curriculum_k), dtype=torch.bool, device=device)
    return rollout_loss(
        all_predictions,
        all_targets,
        bootstrap,
        valid=supervision,
        reconstruction_loss=rec_loss,
        encoder_decoder_trainable=True,
        config=config,
    )


def _step_a0_reconstruction(
    components: Mapping[str, nn.Module],
    batch: Any,
    config: MethodConfig,
) -> Tensor:
    """Stage A0 point-cloud autoencoder reconstruction."""
    geometry = components.get("geometry")
    if geometry is None:
        raise KeyError("Stage A0 requires 'geometry' component")

    encoder, decoder = get_geometry_encoder_decoder(geometry)

    points = batch.get("points") if isinstance(batch, Mapping) else getattr(batch, "points", None)
    if points is None:
        raise ValueError("Stage A0 batch requires 'points'")

    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)
    if points.ndim == 2:
        points = points.unsqueeze(0)

    raw_valid = batch.get("valid") if isinstance(batch, Mapping) else getattr(batch, "valid", None)
    if raw_valid is None:
        point_valid = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    else:
        point_valid = torch.as_tensor(raw_valid, dtype=torch.bool, device=points.device)
        if point_valid.ndim == 1:
            point_valid = point_valid.unsqueeze(0)
        if point_valid.shape != points.shape[:2]:
            raise ValueError("Stage A0 valid mask shape must match points")

    encoded = encoder(points, point_valid)
    decoded = decoder(encoded)
    pred_points = getattr(decoded, "points_w", decoded)
    pred_valid = getattr(decoded, "point_valid", None)
    if pred_valid is None:
        pred_valid = torch.ones(pred_points.shape[:2], dtype=torch.bool, device=pred_points.device)
    return masked_normalized_chamfer_distance(
        pred_points,
        points,
        pred_valid,
        point_valid,
        config=config,
    )


def _step_d1_rollout(
    components: Mapping[str, nn.Module],
    batch: Any,
    curriculum_k: int,
    config: MethodConfig,
) -> Tensor:
    """Execute Stage D1 dynamics training over bootstrap heads with frozen geometry and memory.

    Note: Stage D1 is not supported in the public runner pending P06/C verified frozen-reference
    binding; multi-record global selected-head/step normalization across batches remains
    unresolved and is not certified.
    """
    memory = components.get("physical_memory")
    dynamics = components.get("dynamics")
    geometry = components.get("geometry")

    if memory is None or dynamics is None or geometry is None:
        raise KeyError("Stage D1 requires 'geometry', 'physical_memory', and 'dynamics' components")

    encoder, decoder = get_geometry_encoder_decoder(geometry)

    num_supervised = config.stages.D1.supervised_intervals

    if isinstance(batch, Mapping):
        if "supervised_intervals" in batch and batch["supervised_intervals"] != num_supervised:
            raise ValueError(
                f"batch supervised_intervals ({batch['supervised_intervals']}) conflicts with configured stage value ({num_supervised})"
            )
        if "burnin" in batch:
            raise ValueError("D1 has no configured burnin; batch burnin must not be provided")
        if "seed" in batch and "bootstrap_seed" not in batch:
            raise ValueError("Stage D1 requires explicit 'bootstrap_seed', generic 'seed' is not accepted")
        passed_reset_state = batch.get("reset_state") or batch.get("state")
        transitions = batch.get("transitions")
        boundary = batch.get("boundary", 0)
        episode_id = batch.get("episode_id")
        bootstrap_seed = batch.get("bootstrap_seed")
        raw_valid = batch.get("valid")
    else:
        if hasattr(batch, "seed") and not hasattr(batch, "bootstrap_seed"):
            raise ValueError("Stage D1 requires explicit 'bootstrap_seed', generic 'seed' is not accepted")
        passed_reset_state = getattr(batch, "reset_state", None) or getattr(batch, "state", None)
        transitions = getattr(batch, "transitions", None)
        boundary = getattr(batch, "boundary", 0)
        episode_id = getattr(batch, "episode_id", None)
        bootstrap_seed = getattr(batch, "bootstrap_seed", None)
        raw_valid = getattr(batch, "valid", None)

    # D1 requires explicit nonempty episode identity + bootstrap seed with provenance
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("Stage D1 requires explicit nonempty episode identity ('episode_id')")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("Stage D1 requires explicit bootstrap seed with provenance ('bootstrap_seed')")

    if transitions is None:
        raise ValueError("Stage D1 batch must contain 'transitions' sequence")

    if len(transitions) < boundary + num_supervised:
        raise ValueError(
            f"transitions sequence length {len(transitions)} is insufficient for boundary {boundary} + supervised_intervals {num_supervised}"
        )

    dyn_param = next(dynamics.parameters(), None)
    device = dyn_param.device if dyn_param is not None else torch.device("cpu")
    dtype = dyn_param.dtype if dyn_param is not None else torch.float32

    # Require exact boolean validity shape [1, num_supervised]
    if raw_valid is None:
        raise ValueError(f"Stage D1 requires explicitly supplied 'valid' mask of shape [1, {num_supervised}] and boolean dtype")
    if not torch.is_tensor(raw_valid) or raw_valid.dtype != torch.bool or raw_valid.shape != (1, num_supervised):
        raise ValueError(f"D1 valid mask must have shape [1, {num_supervised}] and boolean dtype")
    valid_mask = raw_valid.to(device=device)

    reset_state, transitions = _validate_transitions_and_reconstruct_reset(
        transitions,
        encoder,
        config,
        reset_state=passed_reset_state,
        device=device,
        dtype=dtype,
    )

    state_at_boundary = _replay_history_and_detach(
        reset_state=reset_state,
        transitions=transitions,
        boundary=boundary,
        encoder=encoder,
        physical_memory=memory,
        config=config,
    )

    mask = bootstrap_mask([episode_id], seed=bootstrap_seed, config=config)

    rollout = getattr(batch, "rollout", None)
    if rollout is None:
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=config)

    root_offsets = list(range(0, num_supervised - curriculum_k + 1, curriculum_k))
    if not root_offsets:
        root_offsets = [0]

    all_predictions = []
    all_targets = []
    for root_offset in root_offsets:
        if root_offset == 0:
            root_state = state_at_boundary
        else:
            root_state = _replay_history_and_detach(
                reset_state=reset_state,
                transitions=transitions,
                boundary=boundary + root_offset,
                encoder=encoder,
                physical_memory=memory,
                config=config,
            )

        target_steps = [transitions[boundary + root_offset + step] for step in range(curriculum_k)]
        commands = [target_trans.command for target_trans in target_steps]

        episode_heads_preds = [[], [], []]
        for head_idx in range(3):
            if bool(mask[0, head_idx].item()):
                head_preds = rollout.rollout(root_state, commands, head_id=head_idx)
                episode_heads_preds[head_idx] = list(head_preds)

        all_predictions.append(episode_heads_preds)
        all_targets.append(target_steps)

    R = len(root_offsets)
    root_bootstrap = mask.expand(R, 3)
    supervision = torch.ones((R, curriculum_k), dtype=torch.bool, device=device)
    for r_idx, root_offset in enumerate(root_offsets):
        supervision[r_idx] = valid_mask[0, root_offset:root_offset + curriculum_k]

    return rollout_loss(
        all_predictions,
        all_targets,
        root_bootstrap,
        valid=supervision,
        encoder_decoder_trainable=False,
        config=config,
    )


def _evaluate_stage(
    stage: str,
    components: Mapping[str, nn.Module],
    dataset: Any,
    config: MethodConfig,
    curriculum_k: int = 1,
    expected_schema: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate components on validation partition.

    Temporarily switches all evaluated components to eval() mode and restores
    their original training mode in a finally block, even if an exception occurs.
    Rejects invalid/empty datasets or batches explicitly instead of returning zero.
    """
    if stage == "B":
        raise NotImplementedError(
            "Stage B evaluation requires upstream event/task objective APIs "
            "(src/icgs/algorithms/objectives/task.py owned by P06 Task1C), which are not yet available."
        )
    if stage == "C":
        raise NotImplementedError(
            "Stage C reference certification requires upstream reference router / model weights binding "
            "(owned by P06), which is not yet available."
        )
    if stage == "D2":
        raise NotImplementedError(
            "Stage D2 evaluation requires upstream evaluator/terminal objective APIs "
            "(src/icgs/algorithms/objectives/evaluation.py owned by P09), which are not yet available."
        )
    if stage == "E":
        raise NotImplementedError(
            "Stage E temperature calibration requires upstream calibration API "
            "(owned by P09), which is not yet available."
        )

    if dataset is None:
        raise ValueError("evaluation dataset cannot be None")

    # Record previous training mode of each module
    previous_modes: dict[int, bool] = {
        id(mod): mod.training for mod in components.values() if isinstance(mod, nn.Module)
    }

    try:
        # Temporarily switch all evaluated components to eval mode
        for mod in components.values():
            if isinstance(mod, nn.Module):
                mod.eval()

        dev_ranking_val: float | None = None
        if hasattr(dataset, "sample_batch") or callable(dataset):
            raise TypeError(
                "stateful evaluation dataset is not supported; immutable evaluation records sequence required"
            )
        if not isinstance(dataset, Sequence) or isinstance(dataset, (str, bytes)):
            raise TypeError("evaluation dataset must be an immutable Sequence of records")
        if len(dataset) == 0:
            raise ValueError("evaluation dataset cannot be empty")
        batches = [dataset[i] for i in range(len(dataset))]

        dev_ranking_vals: list[float] = []
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise ValueError("validation batch must be a mapping")
            _validate_record_provenance(batch, is_eval=True)
            if "dev_ranking" in batch:
                raw_r = batch["dev_ranking"]
                if raw_r is not None:
                    if isinstance(raw_r, bool) or not isinstance(raw_r, (int, float)) or not math.isfinite(raw_r):
                        raise ValueError(f"dev_ranking must be a finite float, got {raw_r!r}")
                    dev_ranking_vals.append(float(raw_r))

        if len(dev_ranking_vals) > 0:
            if len(dev_ranking_vals) != len(batches):
                raise ValueError(
                    "partial/mixed dev_ranking across evaluation dataset records is rejected; "
                    "dev_ranking must be uniformly supplied across all records with identical value, or omitted completely"
                )
            if not all(v == dev_ranking_vals[0] for v in dev_ranking_vals):
                raise ValueError(
                    f"varying dev_ranking values across evaluation records rejected; "
                    f"supplied diagnostic requires identical exact value across all records, got {dev_ranking_vals!r}"
                )
            dev_ranking_val = dev_ranking_vals[0]

        with torch.no_grad():
            total_loss = 0.0
            count = 0
            for batch in batches:
                if stage == "A1":
                    step_loss = _step_a1_rollout(components, batch, curriculum_k, config)
                elif stage == "D1":
                    step_loss = _step_d1_rollout(components, batch, curriculum_k, config)
                elif stage == "A0":
                    step_loss = _step_a0_reconstruction(components, batch, config)
                else:
                    raise ValueError(f"stage {stage!r} does not support evaluation")

                if not torch.is_tensor(step_loss) or not torch.isfinite(step_loss).all():
                    raise FloatingPointError(f"nonfinite loss during evaluation in stage {stage}: {step_loss}")

                total_loss += float(step_loss.item())
                count += 1

            if count == 0:
                raise ValueError("no validation batches were evaluated")
            val_loss = total_loss / count
            if not math.isfinite(val_loss):
                raise FloatingPointError("non-finite validation loss")

        if stage == "A0":
            if expected_schema is not None and expected_schema != ("normalized_reconstruction_loss",):
                raise ValueError(f"Stage A0 evaluation schema mismatch: {expected_schema}")
            if dev_ranking_val is not None:
                raise ValueError("Stage A0 does not support dev_ranking in evaluation dataset")
            return {
                "val_loss": val_loss,
                "normalized_reconstruction_loss": val_loss,
                "metric_priority": ("normalized_reconstruction_loss",),
                "metric_vector": (val_loss,),
                "selection_schema": ("normalized_reconstruction_loss",),
                "fidelity_gate": "NOT RUN (unavailable)",
                "bridge_gate_passed": False,
                "selection_status": "not_certified (bridge gate unavailable)",
            }
        else:
            if expected_schema is not None:
                if "dev_ranking" in expected_schema:
                    if dev_ranking_val is None:
                        raise ValueError(f"expected_schema {expected_schema} requires dev_ranking in evaluation dataset")
                    return {
                        "val_loss": val_loss,
                        "physical_validation_loss": val_loss,
                        "dev_ranking": dev_ranking_val,
                        "evaluator": "NOT RUN (diagnostic external supplied)",
                        "metric_priority": ("physical_validation_loss", "dev_ranking"),
                        "metric_vector": (val_loss, dev_ranking_val),
                        "selection_schema": ("physical_validation_loss", "dev_ranking"),
                        "selection_status": "diagnostic_external",
                    }
                else:
                    return {
                        "val_loss": val_loss,
                        "physical_validation_loss": val_loss,
                        "dev_ranking": "NOT RUN (unavailable)",
                        "evaluator": "NOT RUN",
                        "metric_priority": ("physical_validation_loss",),
                        "metric_vector": (val_loss,),
                        "selection_schema": ("physical_validation_loss",),
                        "selection_status": "certified",
                    }
            if dev_ranking_val is not None:
                return {
                    "val_loss": val_loss,
                    "physical_validation_loss": val_loss,
                    "dev_ranking": dev_ranking_val,
                    "evaluator": "NOT RUN (diagnostic external supplied)",
                    "metric_priority": ("physical_validation_loss", "dev_ranking"),
                    "metric_vector": (val_loss, dev_ranking_val),
                    "selection_schema": ("physical_validation_loss", "dev_ranking"),
                    "selection_status": "diagnostic_external",
                }
            else:
                return {
                    "val_loss": val_loss,
                    "physical_validation_loss": val_loss,
                    "dev_ranking": "NOT RUN (unavailable)",
                    "evaluator": "NOT RUN",
                    "metric_priority": ("physical_validation_loss",),
                    "metric_vector": (val_loss,),
                    "selection_schema": ("physical_validation_loss",),
                    "selection_status": "certified",
                }
    finally:
        # Restore previous mode of each component
        for mod in components.values():
            if isinstance(mod, nn.Module):
                orig_mode = previous_modes.get(id(mod), True)
                mod.train(orig_mode)


__all__ = [
    "TrainingReport",
    "build_transition_descriptor_and_achieved",
    "history_slices",
    "load_checkpoint",
    "run_method_training",
    "sample_anchor_horizon",
    "save_checkpoint",
]
