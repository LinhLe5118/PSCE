"""Read-only timing of one posterior draw on a published pilot, without labels."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import posterior, predictive, tabular  # noqa: E402
from .fast_backend import install  # noqa: E402


def main():
    data = ROOT / "datasets/bridge_v1_1_1/offline_pilot_expanded_2026-10-08"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in manifest["pilot_groups"]
               if r["mdp_id"] == "MiniGrid-DoorKey-5x5-v0"
               and r["policy_family"] == "q_star_softmax")
    with np.load(data / row["file"], allow_pickle=False) as archive:
        pilot = {key: archive[key] for key in archive.files}
    install()
    seconds = defaultdict(float)
    calls = defaultdict(int)
    originals = []

    def instrument(module, function):
        original = getattr(module, function)
        key = function

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                seconds[key] += time.perf_counter() - start
                calls[key] += 1

        setattr(module, function, wrapper)
        originals.append((module, function, original))

    for module, names in (
        (posterior, ("pilot_statistics", "fit_posterior", "sample_mdps",
                     "simulate_streams")),
        (predictive, ("count_model", "finite_horizon_references",
                      "finite_horizon_policy_returns")),
        (tabular, ("cfqi", "iql", "behavior_cloning")),
    ):
        for name in names:
            instrument(module, name)
    started = time.perf_counter()
    try:
        result = predictive.posterior_outcomes(
            pilot, row["initial_state_distribution"],
            num_states=int(row["num_states"]),
            num_actions=int(row["num_actions"]), gamma=.99,
            draws=1, seeds=2, budgets=(1, 2, 5, 10, 20, 50),
            horizon=100, evaluation_horizon=100, seed=1842,
            minimum_dynamic_range=0.0, reward_bound=1.0,
            reward_prior="unseen_zero", transition_base="uniform",
            behavior_prior="mixture")
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
    elapsed = time.perf_counter() - started
    report = {"status": "PRELABEL_TIMING_ONLY", "mdp_id": row["mdp_id"],
              "policy_family": row["policy_family"], "draws": 1, "seeds": 2,
              "elapsed_seconds": round(elapsed, 3),
              "draws_returned": result.sampling["draws_returned"],
              "timings": {key: {"seconds": round(seconds[key], 3), "calls": calls[key]}
                          for key in sorted(seconds)},
              "test_labels_read": False}
    report_path = ROOT / "artifacts/week8/bridge_oehb_prelabel_profile_2026-10-10.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
