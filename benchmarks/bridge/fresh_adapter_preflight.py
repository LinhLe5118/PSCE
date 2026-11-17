"""Prelabel sticky-action/horizon check on the frozen BRIDGE panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .fresh_adapter import DATA, PANEL, load_fresh_sticky_mdp, registry
from .sticky_preflight import ArrayEnv, load_wrapper, local_step

OUT = PANEL / "adapter_preflight.json"
WRAPPER = DATA.parent / "official_source/effective_horizon/envs/wrappers.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    wrapper = load_wrapper()
    rows = []
    for name, spec in registry().items():
        mdp = load_fresh_sticky_mdp(name)
        with np.load(DATA / name / "consolidated.npz", allow_pickle=False) as raw:
            transitions, rewards = raw["transitions"], raw["rewards"]
        horizon = int(spec["horizon"])
        for seed in (0, 17, 104729):
            env = wrapper(ArrayEnv(transitions, rewards, horizon), repeat_action_probability=.25)
            observed, _ = env.reset(seed=seed)
            if observed != 0 or env.last_action is not None:
                raise AssertionError(f"Reset mismatch: {name}")
            local_rng = np.random.default_rng(seed)
            command_rng = np.random.default_rng(seed + 1000000)
            state, previous = 0, None
            for step in range(horizon):
                command = int(command_rng.integers(mdp.num_actions))
                expected, executed, reward, terminal, truncated = local_step(
                    transitions, rewards, state, previous, step, command, local_rng, horizon)
                got, got_reward, got_terminal, got_truncated, _ = env.step(command)
                if (got != expected or env.last_action != executed or
                        abs(got_reward - reward) > 1e-6 or
                        got_terminal != terminal or got_truncated != truncated):
                    raise AssertionError(f"Author wrapper mismatch: {name}, seed={seed}, t={step}")
                state, previous = expected, executed
                if terminal or truncated:
                    break
            env.close()
        # Compare each adapter transition row to the published base kernel and
        # sticky mixture. The reset sentinel is state 0 by construction.
        for command in range(mdp.num_actions):
            target = int(transitions[0, command])
            reward = float(rewards[0, command])
            support = np.flatnonzero(mdp.P[0, command] > 0)
            if len(support) != 1 or not np.isclose(mdp.R[0, command, support[0]], reward):
                raise AssertionError(f"First-step adapter mismatch: {name}")
            if (target < 0) != (int(support[0]) in mdp.terminal_states):
                raise AssertionError(f"First-step terminal mismatch: {name}")
        rows.append({"mdp_name": name, "origin": spec["origin"],
                     "source_npz_sha256": sha256(DATA / name / "consolidated.npz"),
                     "adapter_states": mdp.num_states,
                     "published_sticky_states": int(spec["sticky_states"]),
                     "horizon": horizon, "wrapper_trials": 3,
                     "status": "PASS_AUTHOR_STICKY_WRAPPER_AND_FIRST_STEP"})
        print(f"Adapter preflight {name}: PASS", flush=True)
    OUT.write_text(json.dumps({"status": "PRELABEL_ADAPTER_PASS", "source_wrapper_sha256": sha256(WRAPPER),
                               "rows": rows, "test_labels_read": False,
                               "limit": "Checks array-backed author wrapper and first-step adapter; does not reproduce original Atari/Procgen simulators or certify OEH quality."},
                              indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
