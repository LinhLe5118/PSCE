"""Runtime-only learner-fit probe at BRIDGE candidate state/budget sizes.

Synthetic transitions measure the existing dense learner implementation. They
are not BRIDGE trajectories, a sticky adapter, an outcome, or a benchmark score.
"""

from __future__ import annotations

import json
import argparse
import csv
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular  # noqa: E402


Transition = namedtuple("Transition", "state action reward next_state terminal")
CASES = ((117, 6, 10), (117, 6, 50), (1010, 6, 10), (1010, 6, 50))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    if args.all_candidates:
        with (ROOT / "benchmarks/bridge/candidate_registry.csv").open(newline="", encoding="utf-8") as handle:
            registry = [row for row in csv.DictReader(handle)
                        if row["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
        cases = [(row["mdp_name"], int(row["sticky_states"]), int(row["actions"]), multiplier)
                 for row in registry for multiplier in (1, 2, 5, 10, 20, 50)]
    else:
        cases = [("synthetic_size_probe", states, actions, multiplier)
                 for states, actions, multiplier in CASES]
    rows = []
    for name, states, actions, multiplier in cases:
        n = states * actions * multiplier
        rng = np.random.default_rng(20261007 + n)
        start = time.perf_counter()
        s = rng.integers(0, states - 1, size=n)
        a = rng.integers(0, actions, size=n)
        next_s = rng.integers(0, states, size=n)
        done = next_s == states - 1
        transitions = [Transition(int(x), int(y), float(z), int(w), bool(t))
                       for x, y, z, w, t in zip(s, a, done, next_s, done)]
        generated = time.perf_counter()
        fits = {}
        for algorithm, learner in (
            ("BC", lambda: behavior_cloning_tabular(transitions, num_states=states, num_actions=actions)),
            ("CFQI", lambda: cfqi_tabular(transitions, num_states=states, num_actions=actions,
                                         gamma=0.99, iterations=100, tolerance=1e-10)),
            ("IQL", lambda: iql_tabular(transitions, num_states=states, num_actions=actions,
                                       gamma=0.99, iterations=100, tolerance=1e-10,
                                       expectile=0.7, inverse_temperature=3.0)),
        ):
            tick = time.perf_counter()
            policy = learner()
            fits[algorithm] = round(time.perf_counter() - tick, 3)
            if policy.shape != (states, actions):
                raise ValueError("Wrong policy shape")
        row = {"mdp_name": name, "states": states, "actions": actions, "multiplier": multiplier,
               "transitions": n, "generate_seconds": round(generated - start, 3),
               "fit_seconds": fits}
        rows.append(row)
        if not args.all_candidates or multiplier == 50:
            print(json.dumps(row), flush=True)
    output = ROOT / "datasets/bridge_v1_1_1" / (
        "learner_runtime_all_candidates.json" if args.all_candidates else "learner_runtime_probe.json"
    )
    output.write_text(json.dumps({"scope": "synthetic_runtime_only_no_BRIDGE_outcomes", "cases": rows}, indent=2),
                      encoding="utf-8")


if __name__ == "__main__":
    main()
