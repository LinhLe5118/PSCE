"""Freeze the development-selected BRIDGE posterior range filter pre-test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEEK8 = ROOT / "artifacts/week8"
OUT = WEEK8 / "bridge_transfer_range_addendum_2026-10-07.json"
SMALL = WEEK8 / "bridge_oeh_range_MiniGrid-LavaGapS5-v0_dev_2026-10-07.json"
LARGE = WEEK8 / "bridge_oeh_range_MiniGrid-SimpleCrossingS9N1-v0_dev_2026-10-07.json"
FREEZE = WEEK8 / "bridge_transfer_preflight_freeze_2026-10-07.json"
PILOTS = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1/manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "PREFLIGHT_SPLIT_FROZEN_NOT_BENCHMARK_GO":
        raise RuntimeError("Task split not frozen")
    for path in (SMALL, LARGE):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if (doc["status"] != "DEVELOPMENT_RANGE_FILTER_PREFLIGHT_ONLY"
                or doc["task"] not in freeze["development_tasks"]
                or doc["filter_minimum_dynamic_range"] != .2
                or doc["posterior_sampling"]["draws_returned"] != 1):
            raise RuntimeError(f"Range preflight failed: {path.name}")
    pilots = json.loads(PILOTS.read_text(encoding="utf-8"))
    if pilots["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS":
        raise RuntimeError("Pilot seal missing")
    OUT.write_text(json.dumps({
        "status": "DEVELOPMENT_METHOD_ADDENDUM_FROZEN_NO_TEST_LABELS",
        "minimum_dynamic_range": .2,
        "rationale": "Preserve the absolute filter 0.01/(1-0.95)=0.2 used in synthetic meta-training. Recomputing from BRIDGE gamma=0.99 gives 1.0 and rejected all 64 large-task development proposals; the 0.2 filter accepted the first proposal on both metadata-selected development tasks.",
        "adaptation_disclosure": "Chosen after development preflight but before any candidate test difficulty label; report separately from the frozen synthetic confirmatory method.",
        "posterior_structure": {"reward_prior": "unseen_zero", "transition_base": "uniform",
                                "behavior_prior": "mixture"},
        "full_forecast_draws_if_feasible": 64, "full_forecast_training_seeds": 10,
        "full_forecast_budgets": [1, 2, 5, 10, 20, 50],
        "source_sha256": {path.name: sha256(path) for path in (FREEZE, SMALL, LARGE, PILOTS)},
        "benchmark_gate": "NO_GO_UNTIL_ALL_108_FEATURES_AND_MATCHED_PREDICTIONS_SEALED",
        "test_labels_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("Frozen BRIDGE development range-filter addendum at 0.2", flush=True)


if __name__ == "__main__":
    main()
