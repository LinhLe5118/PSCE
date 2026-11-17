"""Full 64-draw OEH-B forecast on the smallest BRIDGE development task."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402

TASK = "MiniGrid-LavaGapS5-v0"
OUT = ROOT / "artifacts/week8/bridge_full_oehb_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    addendum = json.loads((ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json")
                          .read_text(encoding="utf-8"))
    if addendum["status"] != "DEVELOPMENT_METHOD_ADDENDUM_FROZEN_NO_TEST_LABELS":
        raise RuntimeError("BRIDGE range filter not frozen")
    mdp = load_sticky_mdp(TASK)
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    start = time.perf_counter()
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=mdp.gamma,
        draws=64, seeds=10, budgets=(1, 2, 5, 10, 20, 50),
        evaluation_horizon=100,
        seed=[20261007], minimum_dynamic_range=.2,
        **addendum["posterior_structure"])
    forecast = predictive.requirement_forecast(outcomes)
    elapsed = time.perf_counter() - start
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "DEVELOPMENT_FULL_OEHB_SMOKE_ONLY",
        "task": TASK, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "pilot_transitions": len(pilot["state"]),
        "filter_minimum_dynamic_range": .2, "evaluation_horizon": 100,
        "draws": 64, "training_seeds": 10, "budgets": [1, 2, 5, 10, 20, 50],
        "posterior_sampling": outcomes.sampling,
        "probabilities": {name: np.asarray(value["probabilities"]).tolist()
                          for name, value in forecast.items()},
        "elapsed_seconds": elapsed,
        "warning": "Full OEH-B only on excluded development task; no 108-feature or test prediction."},
        indent=2) + "\n", encoding="utf-8")
    print(f"Full OEH-B BRIDGE dev smoke complete in {elapsed:.1f}s; "
          f"accepted={outcomes.sampling['draws_returned']}/64", flush=True)


if __name__ == "__main__":
    main()
