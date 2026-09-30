"""Build, audit and freeze the ICGS Stage-1 split over RoboHiMan tasks.

Split decisions are the explicit constants below (reviewed in
docs/decisions/0017-icgs-stage1-split.md). Everything else is derived:
strategy indices from the pinned upstream strategy JSON, variation counts from
the task sources, assets from pre-flight episode manifests, and the overlap
matrix from those facts. The script refuses to lock if a planned task is not
PASS/PARTIAL in the pre-flight compatibility reports, and it never overwrites
an existing lock with different content.

  python -B scripts/robohiman_split_lock.py --robohiman ROOT --compat-dir DIR \
      --store DIR --out-dir artifacts/robohiman --report overlap.json
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

from icgs.data.stage1.split import manifest_sha256, validate_split_manifest
from icgs.environments.robohiman.monitors import MONITOR_VERSION
from icgs.environments.robohiman.pins import UPSTREAM, quirks_for, task_family

SPLIT_ID = "icgs-robohiman-stage1-split-v1"
FROZEN_ON = "2026-09-30"

TRAIN_ATOMIC = ("box_in_cupboard", "box_out_of_cupboard", "box_out_of_opened_drawer", "broom_out_of_cupboard",
                "close_drawer", "open_drawer", "put_in_opened_drawer", "rubbish_in_dustpan", "sweep_to_dustpan",
                "take_out_of_opened_drawer")
TRAIN_COMPOSITIONAL = ("put_in_without_close", "sweep_and_drop", "take_out_without_close", "transfer_box")
DEV_HELD_OUT = ("take_out_and_close", "take_two_out_of_different", "take_two_out_of_same")
TEST_HELD_OUT = ("box_exchange", "put_in_and_close", "put_two_in_different", "put_two_in_same",
                 "retrieve_and_sweep")
DEPENDENCY_STRESS = ("put_in_and_close", "put_two_in_different")

# Colosseum strategy names withheld from every TRAIN/DEV strategy set; used only
# by the unseen-perturbation test track. all_mixed contains them, so it is
# withheld too.
HELD_OUT_PERTURBATIONS = ("distractor", "background_texture", "all_mixed")

SEED_RANGES = {"train": [100_000_000, 199_999_999], "dev": [200_000_000, 299_999_999],
               "test": [300_000_000, 399_999_999]}
TEST_CONTEXT_SEEDS = [390_000_000, 399_999_999]  # inference-time demonstration contexts only
FACTOR_ENV_SEEDS = {"train": 42, "dev": 4242, "test": 244}  # 42/244 = native train/test

# Execution-order structure signatures (monitor events + the upstream expert's
# waypoint order; reviewed, see ADR 0017). drawer_in / surface / table /
# cupboard / dustpan / holder are the places involved.
STRUCTURE = {
    "open_drawer": ["OPEN"],
    "close_drawer": ["CLOSE"],
    "put_in_opened_drawer": ["PICK:surface", "PLACE:drawer_in"],
    "take_out_of_opened_drawer": ["PICK:drawer_in", "PLACE:surface"],
    "box_out_of_opened_drawer": ["PICK:drawer_in", "PLACE:surface"],
    "box_in_cupboard": ["PICK:table", "PLACE:cupboard"],
    "box_out_of_cupboard": ["PICK:cupboard", "PLACE:table"],
    "broom_out_of_cupboard": ["PICK_TOOL:cupboard", "PLACE_TOOL:table"],
    "sweep_to_dustpan": ["PICK_TOOL:table", "SWEEP:dustpan"],
    "rubbish_in_dustpan": ["PICK:table", "DROP:dustpan"],
    "put_in_without_close": ["OPEN", "PICK:surface", "PLACE:drawer_in"],
    "put_in_and_close": ["OPEN", "PICK:surface", "PLACE:drawer_in", "CLOSE"],
    "take_out_without_close": ["OPEN", "PICK:drawer_in", "PLACE:surface"],
    "take_out_and_close": ["OPEN", "PICK:drawer_in", "PLACE:surface", "CLOSE"],
    "put_two_in_same": ["OPEN", "PICK:surface", "PLACE:drawer_in", "PICK:surface", "PLACE:drawer_in"],
    "take_two_out_of_same": ["OPEN", "PICK:drawer_in", "PLACE:surface", "PICK:drawer_in", "PLACE:surface"],
    "put_two_in_different": ["OPEN", "PICK:surface", "PLACE:drawer_in", "CLOSE", "OPEN", "PICK:surface",
                             "PLACE:drawer_in"],
    "take_two_out_of_different": ["OPEN", "PICK:drawer_in", "PLACE:surface", "CLOSE", "OPEN", "PICK:drawer_in",
                                  "PLACE:surface"],
    "transfer_box": ["OPEN", "PICK:drawer_in", "PLACE:cupboard"],
    "box_exchange": ["PICK:cupboard", "PLACE:table", "PICK:table", "PLACE:cupboard"],
    "sweep_and_drop": ["PICK_TOOL:table", "SWEEP:dustpan", "PLACE_TOOL:holder", "PICK:table", "DROP:dustpan"],
    "retrieve_and_sweep": ["PICK_TOOL:cupboard", "SWEEP:dustpan"],
}
# Scene shapes that exist in every RLBench scene and carry no task identity.
_GENERIC = re.compile(r"^(Panda_|ResizableFloor|Wall|diningTable|workspace|boundary_root|Dummy|cam_)")

UNSUPPORTED_CLAIMS = {
    "asset_generalization": "held-out tasks reuse TRAIN object geometry (see overlap: asset_geometry); "
                            "RoboHiMan provides no held-out asset factor. Not claimable without new assets.",
    "unseen_primitive_generalization": "all DEV/TEST primitive steps occur in TRAIN tasks.",
    "dependency_mechanism_generalization": "the n=20 dependency audit found one consequential band "
                                           "(put_in_without_close, TRAIN) and none in audited DEV/TEST tasks.",
}


def _strategies(root: Path, task: str) -> list[dict[str, Any]]:
    family = task_family(task)
    data = json.loads((root / f"HiMan-Bench/robot-colosseum/colosseum/assets/{family}_json/{task}.json").read_text())
    return [{"index": s["spreadsheet_idx"], "name": s["variation_name"], "enabled": s["enabled"]}
            for s in data["strategy"]]


def _variation_count(root: Path, task: str) -> int:
    family = task_family(task)
    text = (root / f"HiMan-Bench/robot-colosseum/colosseum/rlbench/{family}_tasks/{task}.py").read_text()
    body = text[text.index("def variation_count"):].split("\n")[1]
    value = body.strip().removeprefix("return").strip()
    if value == "len(GROCERY_NAMES)":
        return len(re.findall(r"^\s+'[^']+',?$", text[text.index("GROCERY_NAMES"):text.index("]")], re.M))
    return int(value)


def _allowed_strategies(root: Path, task: str, *, held_out_perturbations: bool) -> list[int]:
    allowed = []
    for entry in _strategies(root, task):
        if not entry["enabled"]:
            continue
        if held_out_perturbations != (entry["name"] in HELD_OUT_PERTURBATIONS):
            continue
        allowed.append(entry["index"])
    return allowed


def _asset_signatures(inventory_dir: Path) -> dict[str, dict[str, str]]:
    """task -> {object name: geometry signature} from robohiman_asset_inventory.py."""
    result = {}
    for path in inventory_dir.glob("*.json"):
        data = json.loads(path.read_text())
        result[data["task"]] = {name: obj["signature"] for name, obj in data["objects"].items()}
    return result


def _assets(store: Path) -> dict[str, set[str]]:
    assets: dict[str, set[str]] = {}
    for manifest_path in store.glob("episodes/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        names = {n for n in manifest["predicates"]["object_names"] if not _GENERIC.match(n)}
        assets.setdefault(manifest["source"]["task"], set()).update(names)
    return assets


def _primitives(root: Path, task: str) -> list[str]:
    family = task_family(task)
    text = (root / f"HiMan-Bench/robot-colosseum/colosseum/rlbench/{family}_tasks/{task}.py").read_text()
    match = re.search(r'"oracle_half"\s*:\s*\[\s*f?"(.*?)"\s*\]', text, re.S)
    steps = []
    for line in (match.group(1).split("\\n") if match else []):
        line = re.sub(r"\{[^}]*\}", "", line.lower())
        line = re.sub(r"\b(bottom|middle|top|the|a|an|of|on|in|into|to|from|it|one|other)\b", " ", line)
        # Action + place only: manipulated-object nouns are templated in some tasks and
        # literal in others ("pick up the sugar"), so they are removed for comparison.
        line = re.sub(r"\b(block|blocks|spam|sugar|strawberry|jello|rubbish|broom|dirt)\b", " ", line)
        steps.append(" ".join(line.split()))
    return steps


def _bigrams(sequence: list[str]) -> set[str]:
    return {f"{a}>{b}" for a, b in zip(sequence, sequence[1:])}


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(args.robohiman)
    compat = {p.stem: json.loads(p.read_text()) for p in Path(args.compat_dir).glob("*.json")}
    planned = TRAIN_ATOMIC + TRAIN_COMPOSITIONAL + DEV_HELD_OUT + TEST_HELD_OUT
    problems = [f"{task}: {compat.get(task, {}).get('verdict', 'MISSING')}" for task in planned
                if compat.get(task, {}).get("verdict") not in ("PASS", "PARTIAL")
                and not (task == "rubbish_in_dustpan" and compat.get(task, {}).get("ap_only_verdict") in ("PASS", "PARTIAL"))]
    if problems:
        raise SystemExit(f"refusing to lock: planned tasks not compatible: {problems}")

    def rule(task: str, *, held_out_perturbations: bool = False) -> dict[str, Any]:
        return {"strategies": _allowed_strategies(root, task, held_out_perturbations=held_out_perturbations),
                "variations": list(range(_variation_count(root, task))),
                "compat_verdict": compat[task]["verdict"],
                "known_upstream_quirks": quirks_for(task)}

    train_tasks = TRAIN_ATOMIC + TRAIN_COMPOSITIONAL
    splits = {
        "train": {"tasks": {t: rule(t) for t in train_tasks}},
        "dev": {"tasks": {**{t: rule(t) for t in train_tasks}, **{t: rule(t) for t in DEV_HELD_OUT}}},
        "test": {"tasks": {**{t: {**rule(t), "strategies": sorted(set(rule(t)["strategies"])
                                                                  | set(_allowed_strategies(root, t, held_out_perturbations=True)))}
                              for t in train_tasks},
                           **{t: rule(t) for t in TEST_HELD_OUT}},
                 "context_seed_range": TEST_CONTEXT_SEEDS},
    }
    for name in splits:
        splits[name]["numpy_seed_range"] = SEED_RANGES[name]
        splits[name]["factor_env_seed"] = FACTOR_ENV_SEEDS[name]
    tracks = [
        {"track_id": "DEV-config", "split": "dev", "task_novelty": "seen_task", "tasks": list(train_tasks),
         "strategies": "same as TRAIN", "use": "model selection / early stopping",
         "claim": "selection signal for known-task configuration robustness",
         "not_claimed": "any generalization result; DEV is never reported as a test"},
        {"track_id": "DEV-composition", "split": "dev", "task_novelty": "held_out_task", "tasks": list(DEV_HELD_OUT),
         "strategies": "strategy 0 + TRAIN perturbation families", "use": "model selection for held-out composition",
         "claim": "selection signal for composition of seen primitives into unseen task structures",
         "not_claimed": "test performance; no gradient from these tasks"},
        {"track_id": "TEST-config", "split": "test", "task_novelty": "seen_task", "tasks": list(train_tasks),
         "strategies": "same as TRAIN", "claim": "known-task robustness to new placements and new samples of "
                                                 "TRAIN perturbation families (disjoint seeds and factor seed)",
         "not_claimed": "task, composition, asset or perturbation-family generalization"},
        {"track_id": "TEST-unseen-perturbation", "split": "test", "task_novelty": "seen_task",
         "tasks": list(train_tasks), "strategies": list(HELD_OUT_PERTURBATIONS),
         "claim": "robustness of known tasks to Colosseum perturbation families absent from TRAIN",
         "not_claimed": "new tasks; distractor YCB objects are new only as distractors, not as manipulated assets"},
        {"track_id": "TEST-held-out-composition", "split": "test", "task_novelty": "held_out_task",
         "tasks": list(TEST_HELD_OUT), "strategies": "strategy 0 + TRAIN perturbation families",
         "claim": "composition of seen primitives and assets into task structures unseen in TRAIN and DEV",
         "not_claimed": "unseen primitives or assets (overlap matrix shows both are seen)"},
        {"track_id": "TEST-dependency-stress", "split": "test", "task_novelty": "held_out_task",
         "tasks": list(DEPENDENCY_STRESS), "strategies": "strategy 0",
         "claim": "behaviour on held-out tasks with explicit prerequisite chains (open→place→close; "
                  "close first drawer before second)",
         "not_claimed": "generalization of consequential decisions (audit: consequences are sparse)"},
    ]
    excluded = [
        {"task": "custom-36-program-generator (T01–T20, V01–V04, P/G/R)",
         "reason": "legacy ICGS custom benchmark; not RoboHiMan semantics; known physical abstractions "
                   "(handle-block drawers, marker predicates). Kept recoverable as a possible mechanistic "
                   "diagnostic suite; no scientific reason found to add it to TRAIN."},
        {"task": "colosseum original 20 tasks (colosseum/rlbench/tasks)",
         "reason": "not part of HiMan-Bench A/AP/C/CP; outside the benchmark scope."},
    ]
    lineage_rules = [
        "every episode carries split_id, split, per-episode numpy seed and factor env seed; the collector "
        "refuses (task, strategy, variation, seed) outside the split",
        "Stage-2 anchors, branches and continuations inherit the split of their source episode",
        "no DEV or TEST episode, failure, perturbation, branch or statistic enters gradient training or "
        "normalization statistics",
        "held-out tasks (DEV-composition, TEST-held-out-composition) never appear in TRAIN in any form",
        "TEST demonstrations used as inference-time context come only from test.context_seed_range",
        "changing this manifest requires a new split_id and a new ADR; the lock file hash is checked by the collector",
    ]
    manifest = {
        "split_id": SPLIT_ID, "status": "frozen", "frozen_on": FROZEN_ON,
        "benchmark": {k: dict(v) for k, v in UPSTREAM.items()},
        "monitor_version": MONITOR_VERSION,
        "held_out_perturbation_strategies": list(HELD_OUT_PERTURBATIONS),
        "splits": splits, "tracks": tracks, "excluded": excluded, "lineage_rules": lineage_rules,
        "unsupported_claims": UNSUPPORTED_CLAIMS,
    }
    validate_split_manifest(manifest)

    # ---------------------------------------------------------------- overlap
    assets = _assets(Path(args.store))
    signatures = _asset_signatures(Path(args.asset_inventory))
    groups = {"TRAIN": list(train_tasks), "DEV-composition": list(DEV_HELD_OUT),
              "TEST-held-out-composition": list(TEST_HELD_OUT), "TEST-dependency-stress": list(DEPENDENCY_STRESS)}

    def facts(tasks: list[str]) -> dict[str, set[str]]:
        return {
            "task": set(tasks),
            "primitive": {p for t in tasks for p in _primitives(root, t)},
            "scene_ttm": {hashlib.sha256((root / f"HiMan-Bench/robot-colosseum/colosseum/rlbench/"
                                                  f"{task_family(t)}_task_ttms/{t}.ttm").read_bytes()).hexdigest()[:16]
                          for t in tasks},
            "asset_name": {a for t in tasks for a in assets.get(t, set())},
            "asset_geometry": {sig for t in tasks for sig in signatures.get(t, {}).values()},
            "structure_sequence": {">".join(STRUCTURE[t]) for t in tasks},
            "structure_bigram": {b for t in tasks for b in _bigrams(STRUCTURE[t])},
            "structure_step": {s for t in tasks for s in STRUCTURE[t]},
        }

    group_facts = {name: facts(tasks) for name, tasks in groups.items()}
    matrix = {}
    for a, b in product(["TRAIN"], [g for g in groups if g != "TRAIN"]):
        matrix[f"{a} vs {b}"] = {
            factor: {"shared": sorted(group_facts[a][factor] & group_facts[b][factor])[:12],
                     "fraction_of_eval_seen_in_train": (len(group_facts[a][factor] & group_facts[b][factor])
                                                        / len(group_facts[b][factor])) if group_facts[b][factor] else None,
                     "eval_only": sorted(group_facts[b][factor] - group_facts[a][factor])}
            for factor in group_facts[a]}
    matrix["DEV-composition vs TEST-held-out-composition"] = {
        factor: {"fraction_of_test_seen_in_dev": (len(group_facts["DEV-composition"][factor]
                                                      & group_facts["TEST-held-out-composition"][factor])
                                                  / len(group_facts["TEST-held-out-composition"][factor]))
                 if group_facts["TEST-held-out-composition"][factor] else None}
        for factor in group_facts["TRAIN"]}
    per_task = {}
    train_bigrams = group_facts["TRAIN"]["structure_bigram"]
    for task in DEV_HELD_OUT + TEST_HELD_OUT:
        nearest = max(train_tasks, key=lambda t: len(_bigrams(STRUCTURE[t]) & _bigrams(STRUCTURE[task]))
                      + (STRUCTURE[t] == STRUCTURE[task]) * 100)
        per_task[task] = {
            "structure": ">".join(STRUCTURE[task]),
            "unseen_bigrams": sorted(_bigrams(STRUCTURE[task]) - train_bigrams),
            "exact_structure_in_train": any(STRUCTURE[t] == STRUCTURE[task] for t in train_tasks),
            "nearest_train_task_by_structure": nearest,
            "asset_names_unseen_in_train": sorted(assets.get(task, set()) - group_facts["TRAIN"]["asset_name"]),
            "assets_unseen_in_train_by_geometry": sorted(
                name for name, sig in signatures.get(task, {}).items() if sig not in group_facts["TRAIN"]["asset_geometry"]),
            "primitives_unseen_in_train": sorted(set(_primitives(root, task)) - group_facts["TRAIN"]["primitive"]),
        }
    perturbation = {
        "train_strategy_names": sorted({s["name"] for t in train_tasks for s in _strategies(root, t)
                                        if s["enabled"] and s["name"] not in HELD_OUT_PERTURBATIONS}),
        "held_out_strategy_names": list(HELD_OUT_PERTURBATIONS),
        "always_on_quirk": "13 tasks keep a misnamed object_size factor enabled in every strategy; present in all splits",
    }
    lineage = {"seed_ranges": SEED_RANGES, "test_context_seeds": TEST_CONTEXT_SEEDS,
               "factor_env_seeds": FACTOR_ENV_SEEDS,
               "native_note": "native RoboHiMan train/test used env seeds 42/244 with np.random.seed(None); "
                              "native episodes carry no recoverable lineage and are not part of this split"}
    report = {"split_id": SPLIT_ID, "matrix": matrix, "per_held_out_task": per_task, "perturbation": perturbation,
              "lineage": lineage, "assets_by_task": {t: sorted(v) for t, v in sorted(assets.items())},
              "structure": STRUCTURE}
    return manifest, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robohiman", required=True)
    parser.add_argument("--compat-dir", required=True)
    parser.add_argument("--store", required=True, help="pre-flight store (asset names)")
    parser.add_argument("--asset-inventory", required=True, help="robohiman_asset_inventory.py output dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    manifest, report = build(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{SPLIT_ID.replace('-', '_')}.json"
    lock = path.with_suffix(".lock")
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if lock.exists():
        existing = json.loads(lock.read_text())
        if existing["manifest_sha256"] != hashlib.sha256(text.encode()).hexdigest():
            raise SystemExit(f"{lock} exists with different content; a changed split needs a new split_id")
    path.write_text(text)
    lock.write_text(json.dumps({"split_id": SPLIT_ID, "manifest": path.name,
                                "manifest_sha256": manifest_sha256(path), "frozen_on": FROZEN_ON}, indent=1) + "\n")
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "sha256": manifest_sha256(path),
                      "per_held_out_task": report["per_held_out_task"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
