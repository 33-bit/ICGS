"""Training-time transition mixture. Separate from collection quota."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Mapping

from icgs.data.collection.v3.protocol import V3_PROTOCOL


def mix_training_transitions(
    samples: Sequence[Mapping[str, Any]],
    *,
    view: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Keep 70/30 nominal/perturbed transitions for D_temporal and D_dyn.

    D_geom and other views pass through unchanged. Collection quota is not
    rewritten here.
    """
    rows = [dict(sample) for sample in samples]
    if view not in V3_PROTOCOL.mixture_views:
        return rows
    nominal = [row for row in rows if row.get("episode_kind") != "perturbed"]
    perturbed = [row for row in rows if row.get("episode_kind") == "perturbed"]
    if not perturbed or not nominal:
        return rows
    nom_frac, pert_frac = V3_PROTOCOL.warmup_mixture
    rng_seed = V3_PROTOCOL.collection_seed if seed is None else seed
    keep_nominal = int(round(nom_frac / pert_frac * len(perturbed)))
    if keep_nominal <= len(nominal):
        return _take(nominal, keep_nominal, rng_seed, "nominal") + perturbed
    keep_perturbed = int(round(pert_frac / nom_frac * len(nominal)))
    return nominal + _take(perturbed, keep_perturbed, rng_seed, "perturbed")


def _take(rows: list[dict[str, Any]], count: int, seed: int, channel: str) -> list[dict[str, Any]]:
    if count >= len(rows):
        return rows
    ranked = []
    for index, row in enumerate(rows):
        payload = f"{seed}|{channel}|{row.get('episode_id')}|{row.get('t')}|{index}".encode("utf-8")
        ranked.append((int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"), index, row))
    ranked.sort()
    return [item[2] for item in ranked[:count]]
