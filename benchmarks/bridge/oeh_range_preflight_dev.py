"""Development-only audit of OEH-B's posterior dynamic-range filter on BRIDGE."""
from __future__ import annotations

import argparse
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

DEVELOPMENT = ("MiniGrid-LavaGapS5-v0", "MiniGrid-SimpleCrossingS9N1-v0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=DEVELOPMENT)
    args = parser.parse_args()
    out = ROOT / "artifacts/week8" / f"bridge_oeh_range_{args.task}_dev_2026-10-07.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    frozen = json.loads((ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json")
                        .read_text(encoding="utf-8"))
    if args.task not in frozen["development_tasks"]:
        raise RuntimeError("Only development tasks are allowed")
    mdp = load_sticky_mdp(args.task)
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    structure = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json")
                           .read_text(encoding="utf-8"))["oeh_final"]["structure"]
    start = time.perf_counter()
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=mdp.gamma,
        draws=1, seeds=1, budgets=(10,), seed=[20261007],
        minimum_dynamic_range=.2, **structure)
    elapsed = time.perf_counter() - start
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "DEVELOPMENT_RANGE_FILTER_PREFLIGHT_ONLY",
        "task": args.task, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "gamma": mdp.gamma, "filter_minimum_dynamic_range": .2,
        "filter_reason": "Transfer the absolute 0.01/(1-0.95)=0.2 filter from synthetic training rather than recomputing it as 1.0 at BRIDGE gamma=0.99",
        "primitive_draws": 1, "primitive_seeds": 1, "primitive_budgets": [10],
        "posterior_sampling": outcomes.sampling,
        "returned_normalized_shape": list(outcomes.normalized_return.shape),
        "returned_normalized": np.asarray(outcomes.normalized_return).tolist(),
        "elapsed_seconds": elapsed,
        "warning": "Development-only protocol adaptation; no full 64-draw forecast or test outcome."},
        indent=2) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
