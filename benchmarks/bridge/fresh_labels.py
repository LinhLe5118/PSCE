"""Fresh BRIDGE empirical requirement labels; import only after prelabel seal."""
from __future__ import annotations

import hashlib
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from benchmarks.bridge.finite_horizon import FiniteHorizonEvaluator
from environments.behavior_policies import BehaviorPolicy
from environments.tabular_environments import TabularMDP
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular  # noqa: E402

BUDGETS = (1, 2, 5, 10, 20, 50)
LEARNERS = ("BC", "CFQI", "IQL")
Transition = namedtuple("Transition", "state action reward next_state terminal")


def sustained_event(reached: np.ndarray, minimum: int = 8) -> int:
    if reached.shape != (len(BUDGETS),) or not np.issubdtype(reached.dtype, np.integer):
        raise ValueError("Expected six success counts")
    for j in range(len(BUDGETS)):
        if np.all(reached[j:] >= minimum):
            return j
    return len(BUDGETS)


def seed(task: str, behavior: str, index: int) -> int:
    digest = hashlib.sha256(f"bridge-fresh-training-v1|{task}|{behavior}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def one_group(mdp: TabularMDP, behavior: BehaviorPolicy, *, horizon: int,
              seeds: int = 10, tau: float = .95, minimum: int = 8) -> dict:
    if seeds != 10 or minimum != 8 or horizon != mdp.metadata["horizon"]:
        raise ValueError("Frozen label protocol mismatch")
    evaluator = FiniteHorizonEvaluator(mdp, horizon=horizon)
    random_return, optimal_return = evaluator.reference()
    if optimal_return - random_return <= 1e-8:
        raise ValueError("Degenerate normalized-return span")
    base = mdp.num_states * mdp.num_actions
    values = np.empty((seeds, len(BUDGETS), len(LEARNERS)), dtype=float)
    start = time.perf_counter()
    for i in range(seeds):
        buffer = collect_trajectories(mdp, behavior,
                                      seed=seed(mdp.environment_id, behavior.family, i),
                                      size=BUDGETS[-1] * base, horizon=horizon)
        for j, budget in enumerate(BUDGETS):
            n = budget * base
            transitions = [Transition(int(s), int(a), float(r), int(nx), bool(done))
                           for s, a, r, nx, done in zip(buffer["state"][:n], buffer["action"][:n],
                                                         buffer["reward"][:n], buffer["next_state"][:n],
                                                         buffer["terminal"][:n])]
            policies = (
                behavior_cloning_tabular(transitions, num_states=mdp.num_states,
                                         num_actions=mdp.num_actions),
                cfqi_tabular(transitions, num_states=mdp.num_states,
                             num_actions=mdp.num_actions, gamma=mdp.gamma,
                             iterations=100, tolerance=1e-10),
                iql_tabular(transitions, num_states=mdp.num_states,
                            num_actions=mdp.num_actions, gamma=mdp.gamma,
                            iterations=100, tolerance=1e-10,
                            expectile=.7, inverse_temperature=3.0),
            )
            for k, policy in enumerate(policies):
                values[i, j, k] = evaluator.normalized(policy)
    successes = (values >= tau).sum(axis=0)
    events = {name: sustained_event(successes[:, k], minimum) for k, name in enumerate(LEARNERS)}
    return {"task": mdp.environment_id, "behavior": behavior.family,
            "source_npz_sha256": mdp.metadata["source_sha256"],
            "states": mdp.num_states, "actions": mdp.num_actions,
            "horizon": horizon, "gamma": mdp.gamma, "tau": tau,
            "budgets": list(BUDGETS), "seeds": seeds, "minimum": minimum,
            "random_return": random_return, "optimal_return": optimal_return,
            "normalized_returns": {name: values[:, :, k].tolist() for k, name in enumerate(LEARNERS)},
            "success_counts": {name: successes[:, k].tolist() for k, name in enumerate(LEARNERS)},
            "event": events, "elapsed_seconds": time.perf_counter() - start}
