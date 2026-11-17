"""Measure the corrected BRIDGE posterior scorer only on excluded development tasks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("MiniGrid-LavaGapS5-v0", "MiniGrid-SimpleCrossingS9N1-v0"))
    parser.add_argument("--full-seeds-and-budgets", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    mode = ("_grid" if args.full_seeds_and_budgets else "") + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "artifacts/week8" / f"bridge_finite_posterior_{args.task}{mode}_dev_2026-10-07.json"
    if out.exists():
        raise FileExistsError(out)
    addendum = json.loads((ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json")
                          .read_text(encoding="utf-8"))
    mdp = load_sticky_mdp(args.task)
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    start = time.perf_counter()
    seeds = 10 if args.full_seeds_and_budgets else 1
    budgets = (1, 2, 5, 10, 20, 50) if args.full_seeds_and_budgets else (1,)
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=.99, draws=1, seeds=seeds,
        budgets=budgets, seed=[20261007], minimum_dynamic_range=.2,
        evaluation_horizon=100, **addendum["posterior_structure"])
    elapsed = time.perf_counter() - start
    out.write_text(json.dumps({
        "status": "FINITE_POSTERIOR_RESOURCE_SMOKE_DEVELOPMENT_ONLY",
        "task": args.task, "states": mdp.num_states, "draws": 1,
        "seeds": seeds, "budgets": list(budgets), "evaluation_horizon": 100,
        "sampling": outcomes.sampling, "runtime_seconds": elapsed,
        "test_labels_read": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Finite posterior development smoke {args.task}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
