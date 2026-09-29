"""Static train/test overlap audit of RoboHiMan's native collection protocol.

Reads the pinned RoboHiMan checkout (no simulator) and reports, for every
native train/test split pair, which factors overlap: exact task, primitive
skills, scene (TTM bytes), assets (named shapes), success-predicate classes,
dependency structure (ICGS monitor where reviewed), perturbation family
(enabled Colosseum factor types) and seed lineage. It also re-reads the
collection shell scripts and fails if ``pins.NATIVE_SPLITS`` drifted.

Primitive skills come from the task's ``oracle_half`` decomposition text; they
are split metadata only, never supervision labels.

  python -B scripts/robohiman_split_audit.py --robohiman ROOT --out split_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

from icgs.environments.robohiman.pins import NATIVE_SPLITS


_OPTION_WORDS = re.compile(r"\b(bottom|middle|top|left|right|red|green|blue|first|second|two|the|a|an|of|on|in|into|to|from|it)\b")


def _script_fields(path: Path) -> dict[str, Any]:
    text = path.read_text()
    block = re.search(r"tasks=\((.*?)\)", text, re.S)
    tasks = tuple(re.findall(r'"([a-z_]+)"', block.group(1))) if block else ()
    def value(name: str) -> str:
        match = re.search(rf"^{name}=(.*)$", text, re.M)
        return match.group(1).split("#")[0].strip() if match else ""
    size = re.findall(r"\d+", value("IMAGE_SIZE"))
    return {"tasks": tasks, "idx": int(value("IDX_TO_COLLECT")), "episodes": int(value("NUMBER_OF_EPISODES")),
            "image_size": tuple(int(v) for v in size), "seed": int(value("SEED")),
            "generator": "atomic" if "dataset_generator_atomic" in text else "compositional",
            "cameras": {cam.lower(): value(f"CAMERAS_USE_{cam}") == '"True"'
                        for cam in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "OVERHEAD", "WRIST", "FRONT")},
            "images": {kind.lower(): value(f"IMAGES_USE_{kind}") == '"True"'
                       for kind in ("RGB", "DEPTH", "MASK", "POINTCLOUD")}}


def _primitives(py_text: str) -> list[str]:
    match = re.search(r'"oracle_half"\s*:\s*\[\s*f?"(.*?)"\s*\]', py_text, re.S)
    if not match:
        return []
    steps = []
    for line in match.group(1).split("\\n"):
        line = re.sub(r"\{[^}]*\}", "", line.lower())
        line = _OPTION_WORDS.sub(" ", line)
        steps.append(" ".join(line.split()))
    return steps


def _task_facts(root: Path, task: str, family: str) -> dict[str, Any]:
    base = root / "HiMan-Bench/robot-colosseum/colosseum"
    py = base / f"rlbench/{family}_tasks/{task}.py"
    ttm = base / f"rlbench/{family}_task_ttms/{task}.ttm"
    strategy = json.loads((base / f"assets/{family}_json/{task}.json").read_text())["strategy"]
    text = py.read_text()
    return {
        "py_sha256": hashlib.sha256(py.read_bytes()).hexdigest(),
        "ttm_sha256": hashlib.sha256(ttm.read_bytes()).hexdigest(),
        "shapes": sorted(set(re.findall(r"Shape\(['\"]([A-Za-z0-9_]+)", text))),
        "conditions": sorted(set(re.findall(r"([A-Z][A-Za-z]*Condition[s]?)\(", text))),
        "primitives": _primitives(text),
        "strategies": [{"index": s["spreadsheet_idx"], "name": s["variation_name"], "enabled": s["enabled"],
                        "factor_types": sorted({v["type"] for v in s["variations"] if v["enabled"]})}
                       for s in strategy],
    }


def _split_members(split: dict[str, Any], facts: dict[str, dict[str, Any]]) -> dict[str, set]:
    members: dict[str, set] = {k: set() for k in ("task", "primitive", "scene", "asset", "condition",
                                                   "perturbation_family", "strategy")}
    for task in split["tasks"]:
        fact = facts[task]
        members["task"].add(task)
        members["primitive"].update(fact["primitives"])
        members["scene"].add(fact["ttm_sha256"])
        members["asset"].update(fact["shapes"])
        members["condition"].update(fact["conditions"])
        for strategy in fact["strategies"]:
            if strategy["enabled"] and (split["idx"] == -1 or split["idx"] == strategy["index"]):
                members["strategy"].add(strategy["name"])
                members["perturbation_family"].update(strategy["factor_types"] or ["none"])
    return members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robohiman", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.robohiman)
    scripts = root / "HiMan-Bench/robot-colosseum"
    splits, drift = {}, []
    for name, (script, tasks, generator, idx, episodes, size, seed) in NATIVE_SPLITS.items():
        fields = _script_fields(scripts / script)
        expected = {"tasks": tasks, "idx": idx, "episodes": episodes, "image_size": size, "seed": seed,
                    "generator": generator}
        for key, value in expected.items():
            if fields[key] != value:
                drift.append({"split": name, "field": key, "pins": value, "script": fields[key]})
        splits[name] = fields
    families = {}
    for fields in splits.values():
        for task in fields["tasks"]:
            families[task] = fields["generator"]
    facts = {task: _task_facts(root, task, family) for task, family in families.items()}
    strategy0 = {task: fact["strategies"][0]["name"] for task, fact in facts.items()}
    members = {name: _split_members(fields, facts) for name, fields in splits.items()}
    train = [n for n in splits if n.startswith("train")]
    test = [n for n in splits if n.startswith("test")]
    # RoboHiMan's L4 training set is the union of A, AP, C and CP.
    members["train_L4_union"] = {factor: set().union(*(members[n][factor] for n in train))
                                 for factor in members[train[0]]}
    train.append("train_L4_union")
    pairs = []
    for a, b in product(train, test):
        overlap = {}
        for factor in members[a]:
            shared = members[a][factor] & members[b][factor]
            overlap[factor] = {"shared": len(shared), "test_total": len(members[b][factor]),
                               "test_fraction_seen_in_train": (len(shared) / len(members[b][factor])
                                                                if members[b][factor] else None),
                               "examples": sorted(shared)[:6]}
        pairs.append({"train": a, "test": b, "overlap": overlap})
    unseen_test_tasks = sorted(members["test_compositional"]["task"]
                               - members["train_L4_union"]["task"])
    report = {
        "robohiman_root": str(root),
        "pins_drift": drift,
        "strategy_index_0_names": sorted(set(strategy0.values())),
        "splits": {name: {**fields, "tasks": list(fields["tasks"])} for name, fields in splits.items()},
        "tasks": facts,
        "pairs": pairs,
        "test_compositional_tasks_unseen_in_any_train_split": unseen_test_tasks,
        "seed_lineage": {
            "script_env_seed": {name: fields["seed"] for name, fields in splits.items()},
            "finding": ("dataset generators call np.random.seed(None) in every worker, so env.seed only seeds "
                        "Colosseum factor sampling; episode placement lineage is not recoverable from the "
                        "native scripts. ICGS collection stores the full numpy MT19937 state per episode."),
        },
        "claims_supported_by_native_split": {
            "exact_task_heldout": bool(unseen_test_tasks),
            "asset_heldout": False,
            "note": "per-factor fractions above; a claim needs the corresponding factor held out",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    summary = {"pins_drift": drift, "unseen_test_compositional_tasks": unseen_test_tasks,
               "strategy_index_0_names": report["strategy_index_0_names"]}
    for pair in pairs:
        summary[f"{pair['train']}→{pair['test']}"] = {
            factor: value["test_fraction_seen_in_train"] for factor, value in pair["overlap"].items()}
    print(json.dumps(summary, indent=1))
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
