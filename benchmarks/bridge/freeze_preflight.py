"""Freeze BRIDGE development and candidate test IDs before larger dev smoke."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json"
DEVELOPMENT = ("MiniGrid-LavaGapS5-v0", "MiniGrid-SimpleCrossingS9N1-v0")
SOURCE_FILES = (
    "benchmarks/bridge/adapter.py", "benchmarks/bridge/finite_horizon.py",
    "benchmarks/bridge/labels.py", "code/oeh/predictive.py",
    "scripts/analysis/oeh_final_forecast.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    candidates = [r for r in registry if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    names = {r["mdp_name"] for r in candidates}
    if len(names) != 16 or not set(DEVELOPMENT).issubset(names):
        raise RuntimeError("Frozen candidate registry changed")
    small = min((r for r in candidates if int(r["actions"]) == 6),
                key=lambda r: int(r["sticky_states"]))["mdp_name"]
    large = max((r for r in candidates if int(r["actions"]) == 6),
                key=lambda r: int(r["sticky_states"]))["mdp_name"]
    if (small, large) != DEVELOPMENT:
        raise RuntimeError("Development tasks no longer follow metadata size rule")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "PREFLIGHT_SPLIT_FROZEN_NOT_BENCHMARK_GO",
        "selection_rule": "Smallest and largest six-action candidate by published sticky state count",
        "development_tasks": list(DEVELOPMENT),
        "candidate_test_tasks": sorted(names - set(DEVELOPMENT)),
        "source": "BRIDGE v1.1.1 published MDP arrays; group-generated offline logs and labels",
        "pilot_multiplier": 10, "pilot_training_independent_seed_namespace": True,
        "budgets": [1, 2, 5, 10, 20, 50], "training_seeds": 10,
        "sustained_minimum": 8, "success_threshold": .95,
        "horizon": 100, "gamma": .99,
        "budget_state_count": "reachable states of the group adapter, including reset sentinel and directly reachable terminal slots",
        "primary_learners": ["BC", "CFQI", "IQL"],
        "primary_forecasters": ["B2 matched", "OEH full 108"],
        "other_internal_arms": ["B2+ERI adaptation", "B2+BWD adaptation",
                                "OEH without direct OEH-B", "OEH without OEH-H/risk",
                                "OEH without legacy diagnostics"],
        "score_only_original_formulas": ["ERI", "TQ privileged", "SACo privileged"],
        "decision_quantile": .8,
        "test_label_read_rule": "No candidate test labels before complete preoutcome predictions and final test protocol seal",
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "registry_sha256": sha256(ROOT / "benchmarks/bridge/candidate_registry.csv"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(DEVELOPMENT)} development and {len(names)-len(DEVELOPMENT)} candidate test tasks", flush=True)


if __name__ == "__main__":
    main()
