"""Development-only collector and learner smoke on a prespecified small task."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories, validate_buffer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular  # noqa: E402

TASK = "MiniGrid-LavaGapS5-v0"  # metadata-selected development task, excluded from later test
OUTPUT = ROOT / "artifacts/week8/bridge_adapter_smoke_dev_v2_2026-10-07.json"
Transition = namedtuple("Transition", "state action reward next_state terminal")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    start = time.perf_counter()
    mdp = load_sticky_mdp(TASK)
    adapter_seconds = time.perf_counter() - start
    behavior = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot_size = 10 * mdp.num_states * mdp.num_actions
    buffer = collect_trajectories(mdp, behavior, seed=2026100701, size=pilot_size, horizon=100)
    validate_buffer(mdp, buffer, horizon=100)
    collect_seconds = time.perf_counter() - start - adapter_seconds
    transitions = [Transition(int(s), int(a), float(r), int(n), bool(d)) for s, a, r, n, d in zip(
        buffer["state"], buffer["action"], buffer["reward"], buffer["next_state"], buffer["terminal"])]
    fits = {}
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
        tick = time.perf_counter()
        policy = learner()
        if policy.shape != (mdp.num_states, mdp.num_actions) or not np.allclose(policy.sum(1), 1):
            raise RuntimeError(f"Bad {name} policy")
        fits[name] = round(time.perf_counter() - tick, 3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "DEVELOPMENT_SMOKE_ONLY_NO_DIFFICULTY_LABEL",
        "task": TASK, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "adapter_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "states": mdp.num_states, "actions": mdp.num_actions,
        "pilot_transitions": pilot_size, "observed_nonzero_reward": int(np.count_nonzero(buffer["reward"])),
        "observed_terminal": int(np.count_nonzero(buffer["terminal"])),
        "adapter_seconds": round(adapter_seconds, 3), "collect_seconds": round(collect_seconds, 3),
        "fit_seconds": fits,
        "remaining": ["finite-horizon normalized-return scoring", "full OEH feature extraction",
                      "all budgets, seeds, behaviors and test MDPs"]}, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
