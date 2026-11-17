"""Development-only exact finite-horizon scoring of three offline learners."""
from __future__ import annotations

import json
import sys
import time
from collections import namedtuple
from pathlib import Path

from benchmarks.bridge.adapter import load_sticky_mdp
from benchmarks.bridge.finite_horizon import FiniteHorizonEvaluator
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular  # noqa: E402

OUTPUT = ROOT / "artifacts/week8/bridge_finite_horizon_score_smoke_dev_2026-10-07.json"
Transition = namedtuple("Transition", "state action reward next_state terminal")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mdp = load_sticky_mdp("MiniGrid-LavaGapS5-v0")
    start = time.perf_counter()
    evaluator = FiniteHorizonEvaluator(mdp, 100)
    random_return, optimal_return = evaluator.reference()
    policy = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, policy, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    transitions = [Transition(int(s), int(a), float(r), int(n), bool(d)) for s, a, r, n, d in zip(
        pilot["state"], pilot["action"], pilot["reward"], pilot["next_state"], pilot["terminal"])]
    scores = {}
    for name, learner in (
        ("BC", lambda: behavior_cloning_tabular(transitions, num_states=mdp.num_states,
                                                 num_actions=mdp.num_actions)),
        ("CFQI", lambda: cfqi_tabular(transitions, num_states=mdp.num_states,
                                      num_actions=mdp.num_actions, gamma=.99,
                                      iterations=100, tolerance=1e-10)),
        ("IQL", lambda: iql_tabular(transitions, num_states=mdp.num_states,
                                    num_actions=mdp.num_actions, gamma=.99,
                                    iterations=100, tolerance=1e-10,
                                    expectile=.7, inverse_temperature=3.0)),
    ):
        fitted = learner()
        scores[name] = {"return": evaluator.value(fitted),
                        "normalized_return": evaluator.normalized(fitted)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "DEVELOPMENT_SCORE_SMOKE_ONLY",
        "task": "MiniGrid-LavaGapS5-v0", "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "horizon": 100, "gamma": .99,
        "random_return": random_return, "optimal_time_dependent_return": optimal_return,
        "scores": scores, "seconds": round(time.perf_counter() - start, 3),
        "warning": "One development pilot and one learner fit per method; not an E_l label or test result."},
        indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
