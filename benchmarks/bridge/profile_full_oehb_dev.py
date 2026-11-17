"""Development-only OEH-B profiler on an excluded BRIDGE task.

Runs one posterior draw with the full seed/budget grid. It reads no test labels.
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402

TASK = "MiniGrid-SimpleCrossingS9N1-v0"
OUTPUT = ROOT / "artifacts/week8/bridge_full_oehb_profile_dev_2026-10-08.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = json.loads((ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json")
                        .read_text(encoding="utf-8"))
    if TASK not in frozen["development_tasks"]:
        raise RuntimeError("Profile task must be development only")
    mdp = load_sticky_mdp(TASK)
    beta = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, beta, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    structure = json.loads((ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json")
                           .read_text(encoding="utf-8"))["posterior_structure"]
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=mdp.gamma, draws=1, seeds=10,
        budgets=(1, 2, 5, 10, 20, 50), evaluation_horizon=100,
        seed=[20261007], minimum_dynamic_range=.2, **structure)
    profiler.disable()
    elapsed = time.perf_counter() - started
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(35)
    OUTPUT.write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_NO_TEST_LABELS",
        "task": TASK, "states": mdp.num_states,
        "draws_requested": 1, "draws_returned": outcomes.sampling["draws_returned"],
        "seeds": 10, "budgets": [1, 2, 5, 10, 20, 50],
        "runtime_seconds": elapsed, "top_cumulative_functions": buffer.getvalue(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Development OEH-B profile: {elapsed:.1f}s; saved {OUTPUT}", flush=True)
    print(buffer.getvalue(), flush=True)


if __name__ == "__main__":
    main()
