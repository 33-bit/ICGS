"""Belief state, hypothesis management, mass conservation, and representative selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence
import numpy as np

from icgs.configuration.method import MethodConfig, NumericsConfig
from icgs.state.physical import PhysicalState


def _mass_atol(
    values: Sequence[np.ndarray | float],
    numerics: NumericsConfig,
) -> float:
    is_float32 = any(
        getattr(v, "dtype", None) == np.float32
        for v in values
        if isinstance(v, np.ndarray)
    )
    return numerics.mass_sum_atol_float32 if is_float32 else numerics.mass_sum_atol_float64


def propagate_mass(
    U: float,
    F: float,
    weights: np.ndarray,
    event_probabilities: np.ndarray,
    cfg: MethodConfig | NumericsConfig,
) -> tuple[float, float, np.ndarray]:
    """Propagate absorbed and continuation mass across hypotheses according to predicted event hazards."""
    if isinstance(cfg, MethodConfig):
        numerics = cfg.numerics
    elif isinstance(cfg, NumericsConfig):
        numerics = cfg
    else:
        raise TypeError("cfg must be a MethodConfig or NumericsConfig instance")

    if isinstance(U, bool) or not isinstance(U, (int, float, np.floating, np.integer)):
        raise TypeError("U must be a numeric float")
    if isinstance(F, bool) or not isinstance(F, (int, float, np.floating, np.integer)):
        raise TypeError("F must be a numeric float")
    U = float(U)
    F = float(F)
    if not (np.isfinite(U) and np.isfinite(F)):
        raise ValueError("U and F must be finite numbers")
    if U < 0.0 or F < 0.0:
        raise ValueError("U and F must be nonnegative")

    w_arr = np.asarray(weights)
    ev_arr = np.asarray(event_probabilities)

    if w_arr.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if not np.isfinite(w_arr).all():
        raise ValueError("weights must contain only finite numbers")
    if (w_arr < 0.0).any():
        raise ValueError("weights must be nonnegative")

    is_float32 = w_arr.dtype == np.float32 or ev_arr.dtype == np.float32
    atol = _mass_atol((w_arr, ev_arr), numerics)

    prior_mass = U + F + float(w_arr.sum())
    if abs(prior_mass - 1.0) > atol:
        raise ValueError(f"prior mass must sum to 1.0 within {atol}, got {prior_mass}")

    if ev_arr.ndim != 2 or ev_arr.shape[1] != 3:
        raise ValueError("event_probabilities must have shape [M, 3]")
    if ev_arr.shape[0] != len(w_arr):
        raise ValueError("event_probabilities rows must match weights length")
    if not np.isfinite(ev_arr).all():
        raise ValueError("event_probabilities must contain only finite values")
    if (ev_arr < 0.0).any() or (ev_arr > 1.0).any():
        raise ValueError("event_probabilities elements must be strictly in [0, 1]")

    row_sums = ev_arr.sum(axis=-1)
    if np.any(np.abs(row_sums - 1.0) > atol):
        raise ValueError(f"event_probabilities rows must sum to 1.0 within {atol}")

    success = ev_arr[:, 0]
    failure = ev_arr[:, 1]
    cont = ev_arr[:, 2]

    if is_float32:
        next_U = np.float32(U + (w_arr * success).sum())
        next_F = np.float32(F + (w_arr * failure).sum())
    else:
        next_U = float(U + (w_arr * success).sum())
        next_F = float(F + (w_arr * failure).sum())
    next_weights = w_arr * cont

    # Zero active mass means exactly zero
    next_weights = np.where(next_weights == 0.0, 0.0, next_weights)

    posterior_mass = next_U + next_F + float(next_weights.sum())
    if abs(posterior_mass - 1.0) > atol:
        raise ValueError(f"posterior mass must sum to 1.0 within {atol}, got {posterior_mass}")

    return next_U, next_F, next_weights


def leaf_return(
    U: float,
    weights: np.ndarray,
    active_values: np.ndarray,
    remaining: int,
) -> float:
    """Compute expected leaf return, returning strictly U when horizon remaining is zero."""
    if isinstance(remaining, bool) or not isinstance(remaining, (int, np.integer)):
        raise TypeError("remaining must be an integer")
    remaining = int(remaining)
    if remaining < 0:
        raise ValueError("remaining horizon must be nonnegative")

    if isinstance(U, bool) or not isinstance(U, (int, float, np.floating, np.integer)):
        raise TypeError("U must be a numeric float")
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")

    w_arr = np.asarray(weights)
    v_arr = np.asarray(active_values)
    if w_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError("weights and active_values must be 1D arrays")
    if len(w_arr) != len(v_arr):
        raise ValueError("weights and active_values must have matching lengths")
    if not (np.isfinite(w_arr).all() and np.isfinite(v_arr).all()):
        raise ValueError("weights and active_values must contain only finite numbers")

    if remaining == 0:
        return U
    return float(U + (w_arr * v_arr).sum())


@dataclass(frozen=True)
class Hypothesis:
    """Single predictive hypothesis on one fixed head with associated active mass."""

    head_id: int
    state: PhysicalState
    task: Any
    weight: float | np.floating

    def __post_init__(self) -> None:
        if isinstance(self.head_id, bool) or not isinstance(self.head_id, (int, np.integer)):
            raise TypeError("head_id must be an integer")
        head_id = int(self.head_id)
        if head_id not in (0, 1, 2):
            raise ValueError("head_id must be one of three fixed heads: 0, 1, or 2")
        object.__setattr__(self, "head_id", head_id)

        if not isinstance(self.state, PhysicalState):
            raise TypeError(f"state must be PhysicalState, got {type(self.state)}")

        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float, np.floating, np.integer)):
            raise TypeError("weight must be a float")
        weight = self.weight
        if isinstance(weight, (int, np.integer)):
            weight = float(weight)
        if not np.isfinite(weight):
            raise ValueError("weight must be finite")
        if weight < 0.0:
            raise ValueError("weight (active mass) must be nonnegative")
        object.__setattr__(self, "weight", weight)


@dataclass
class BeliefNode:
    """Belief node tracking absorbed terminal masses and predictive hypotheses across fixed heads."""

    tau: int
    H_root: int
    U: float | np.floating
    F: float | np.floating
    hypotheses: tuple[Hypothesis, ...]
    cfg: MethodConfig
    candidate_edges: list[Any] = field(default_factory=list)
    visits: int = 0
    _weights_dtype: Any = field(init=False, repr=False, default=np.float64)

    def __post_init__(self) -> None:
        if not isinstance(self.cfg, MethodConfig):
            raise TypeError("cfg must be a MethodConfig")

        if isinstance(self.tau, bool) or not isinstance(self.tau, (int, np.integer)) or self.tau < 0:
            raise ValueError("tau must be a nonnegative integer")
        if isinstance(self.H_root, bool) or not isinstance(self.H_root, (int, np.integer)) or self.H_root < 0:
            raise ValueError("H_root must be a nonnegative integer")
        if self.H_root > self.cfg.planning.H:
            raise ValueError(f"H_root ({self.H_root}) cannot exceed cfg.planning.H ({self.cfg.planning.H})")
        if self.tau > self.H_root:
            raise ValueError("tau cannot exceed H_root")

        if isinstance(self.U, bool) or not isinstance(self.U, (int, float, np.floating, np.integer)):
            raise TypeError("U must be float")
        if isinstance(self.F, bool) or not isinstance(self.F, (int, float, np.floating, np.integer)):
            raise TypeError("F must be float")
        U = self.U if isinstance(self.U, np.floating) else float(self.U)
        F = self.F if isinstance(self.F, np.floating) else float(self.F)
        if not (np.isfinite(U) and np.isfinite(F)):
            raise ValueError("U and F must be finite")
        if U < 0.0 or F < 0.0:
            raise ValueError("U and F must be nonnegative")
        self.U = U
        self.F = F

        if not isinstance(self.hypotheses, (tuple, list)) or len(self.hypotheses) != 3:
            raise ValueError("hypotheses must contain exactly three hypotheses")
        self.hypotheses = tuple(self.hypotheses)
        for i, h in enumerate(self.hypotheses):
            if not isinstance(h, Hypothesis) or h.head_id != i:
                raise ValueError(f"Hypothesis {i} must be a Hypothesis with head_id {i}")

        is_float32 = (
            getattr(self.U, "dtype", None) == np.float32
            or getattr(self.F, "dtype", None) == np.float32
            or any(getattr(h.weight, "dtype", None) == np.float32 for h in self.hypotheses)
        )
        self._weights_dtype = np.float32 if is_float32 else np.float64
        atol = self.cfg.numerics.mass_sum_atol_float32 if is_float32 else self.cfg.numerics.mass_sum_atol_float64
        total_mass = float(self.U) + float(self.F) + sum(float(h.weight) for h in self.hypotheses)
        if abs(total_mass - 1.0) > atol:
            raise ValueError(f"BeliefNode total mass must sum to 1.0 within {atol}, got {total_mass}")

        if isinstance(self.visits, bool) or not isinstance(self.visits, (int, np.integer)) or self.visits < 0:
            raise ValueError("visits must be a nonnegative integer")

    @property
    def weights(self) -> np.ndarray:
        return np.array([h.weight for h in self.hypotheses], dtype=self._weights_dtype)

    @property
    def is_terminal(self) -> bool:
        return all(h.weight == 0.0 for h in self.hypotheses) or self.tau >= self.H_root

    def select_representative(self) -> Hypothesis:
        return select_representative(self.hypotheses, self.cfg)


def _extract_geometry(state: PhysicalState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(state, PhysicalState):
        raise TypeError(f"state must be PhysicalState, got {type(state)}")

    pose = state.T_w_e
    if pose.ndim == 3:
        if pose.shape[0] != 1:
            raise ValueError("PhysicalState batch size must be 1")
        pose = pose[0]
    pose_np = pose.detach().cpu().numpy()
    t = pose_np[:3, 3]
    R = pose_np[:3, :3]

    if state.cached_world_cloud is not None and state.cached_world_cloud_valid is not None:
        cloud = state.cached_world_cloud
        valid = state.cached_world_cloud_valid
        if cloud.ndim == 3:
            if cloud.shape[0] != 1:
                raise ValueError("PhysicalState batch size must be 1")
            cloud = cloud[0]
            valid = valid[0]
        pts = cloud[valid].detach().cpu().numpy()
    else:
        cloud = state.x
        valid = state.valid
        if cloud.ndim == 3:
            if cloud.shape[0] != 1:
                raise ValueError("PhysicalState batch size must be 1")
            cloud = cloud[0]
            valid = valid[0]
        pts = cloud[valid].detach().cpu().numpy()

    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("Hypothesis physical state contains no valid geometry points")

    return pts, t, R


def select_representative(
    hypotheses: Sequence[Hypothesis],
    cfg: MethodConfig,
) -> Hypothesis:
    """Select the medoid representative hypothesis minimizing pairwise distance to active heads."""
    if not isinstance(cfg, MethodConfig):
        raise TypeError("cfg must be a MethodConfig")

    active = [h for h in hypotheses if h.weight > 0.0]
    if not active:
        raise ValueError("No hypothesis has positive active mass")

    geoms = {h.head_id: _extract_geometry(h.state) for h in active}
    if len(active) == 1:
        return active[0]

    cloud_scale = cfg.planning.medoid_cloud_scale_m ** 2
    trans_scale = cfg.planning.medoid_translation_scale_m ** 2
    rot_scale = math.radians(cfg.planning.medoid_rotation_scale_deg) ** 2

    def _pairwise_dist(geom_i: tuple[np.ndarray, np.ndarray, np.ndarray], geom_j: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        pts_i, t_i, R_i = geom_i
        pts_j, t_j, R_j = geom_j

        diff = pts_i[:, None, :] - pts_j[None, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        cd = float(np.mean(np.min(sq_dist, axis=1)) + np.mean(np.min(sq_dist, axis=0)))

        trans = float(np.sum((t_i - t_j) ** 2))

        rel_R = R_i.T @ R_j
        cos_theta = float(np.clip((np.trace(rel_R) - 1.0) / 2.0, -1.0, 1.0))
        rot = math.acos(cos_theta) ** 2

        return cd / cloud_scale + trans / trans_scale + rot / rot_scale

    total_distances: dict[int, float] = {}
    for h_i in active:
        tot = 0.0
        for h_j in active:
            if h_i.head_id == h_j.head_id:
                continue
            tot += _pairwise_dist(geoms[h_i.head_id], geoms[h_j.head_id])
        total_distances[h_i.head_id] = tot

    best_h = active[0]
    best_dist = total_distances[best_h.head_id]
    for h in active[1:]:
        dist = total_distances[h.head_id]
        if dist < best_dist:
            best_dist = dist
            best_h = h
        elif dist == best_dist:
            if h.head_id < best_h.head_id:
                best_h = h

    return best_h
