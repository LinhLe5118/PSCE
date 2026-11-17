"""Measure current dense infinite-horizon scorer cost at BRIDGE state sizes.

Synthetic MDPs only. This is a runtime comparison, not BRIDGE evaluation or
the planned finite-horizon sticky-action score.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
from environments.tabular_environments import TabularMDP  # noqa: E402
from exact_solver import evaluate_policy  # noqa: E402


def main() -> None:
    rows = []
    for states, actions in ((97, 3), (485, 6), (1010, 6)):
        started = time.perf_counter()
        rng = np.random.default_rng(20261007 + states)
        p = np.zeros((states, actions, states), dtype=np.float64)
        r = np.zeros_like(p)
        s = np.arange(states - 1)[:, None]
        a = np.arange(actions)[None, :]
        next_one = rng.integers(0, states, size=(states - 1, actions))
        next_two = rng.integers(0, states, size=(states - 1, actions))
        p[s, a, next_one] += 0.75
        p[s, a, next_two] += 0.25
        r[s, a, next_one] = (next_one == states - 1).astype(float)
        p[-1, :, -1] = 1.0
        rho = np.zeros(states)
        rho[0] = 1.0
        mdp = TabularMDP(f"synthetic_score_{states}", p, r, rho, (states - 1,), 0.99, {})
        policy = np.full((states, actions), 1 / actions)
        built = time.perf_counter()
        value = evaluate_policy(mdp, policy).expected_return
        scored = time.perf_counter()
        row = {"states": states, "actions": actions,
               "build_seconds": round(built - started, 3),
               "score_seconds": round(scored - built, 3),
               "finite_return": bool(np.isfinite(value))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = ROOT / "datasets/bridge_v1_1_1/score_runtime_probe.json"
    output.write_text(json.dumps({"scope": "synthetic_dense_infinite_horizon_proxy_only", "cases": rows}, indent=2),
                      encoding="utf-8")


if __name__ == "__main__":
    main()
