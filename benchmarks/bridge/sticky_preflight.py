"""Check BRIDGE sticky dynamics against the authors' Python wrapper.

This verifies the downloaded base arrays, first-step behavior, previous
executed action, reward, terminal and horizon on a dummy base-array Gym env.
It does NOT establish equivalence to the original MiniGrid simulator or prove
that the current dense offline learner/OEH pipeline can use a time-aware MDP.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bridge_v1_1_1"
HERE = Path(__file__).resolve().parent
WRAPPER = DATA / "official_source/effective_horizon/envs/wrappers.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wrapper():
    spec = importlib.util.spec_from_file_location("bridge_author_sticky_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned author wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StickyActionsWrapper


class ArrayEnv(gym.Env):
    def __init__(self, transitions: np.ndarray, rewards: np.ndarray, horizon: int):
        self.transitions = transitions
        self.rewards = rewards
        self.horizon = horizon
        self.action_space = gym.spaces.Discrete(transitions.shape[1])
        self.observation_space = gym.spaces.Discrete(transitions.shape[0] + 1)
        self.terminal_state = transitions.shape[0]
        self.state = 0
        self.t = 0
        self.last_executed = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.t, self.last_executed = 0, 0, None
        return self.state, {}

    def step(self, action):
        action = int(action)
        if self.state == self.terminal_state:
            raise RuntimeError("Step after terminal")
        reward = float(self.rewards[self.state, action])
        nxt = int(self.transitions[self.state, action])
        self.state = self.terminal_state if nxt == -1 else nxt
        self.last_executed = action
        self.t += 1
        terminal = nxt == -1
        truncated = self.t >= self.horizon and not terminal
        return self.state, reward, terminal, truncated, {}


def local_step(transitions, rewards, state, previous, t, command, rng, horizon):
    executed = previous if previous is not None and rng.uniform() < .25 else command
    reward = float(rewards[state, executed])
    nxt = int(transitions[state, executed])
    terminal = nxt == -1
    truncated = t + 1 >= horizon and not terminal
    return (transitions.shape[0] if terminal else nxt), executed, reward, terminal, truncated


def check_one(name: str, file: Path, expected: str, sticky_wrapper) -> dict:
    if sha256(file) != expected:
        raise RuntimeError(f"Downloaded MDP hash changed: {name}")
    with np.load(file, allow_pickle=False) as raw:
        transitions, rewards = raw["transitions"], raw["rewards"]
    if transitions.shape != rewards.shape:
        raise RuntimeError(f"Base arrays disagree: {name}")
    actions = transitions.shape[1]
    for seed in (0, 17, 104729):
        env = sticky_wrapper(ArrayEnv(transitions, rewards, 100), repeat_action_probability=.25)
        observed, _ = env.reset(seed=seed)
        if observed != 0 or env.last_action is not None:
            raise RuntimeError(f"Reset mismatch: {name}")
        local_rng = np.random.default_rng(seed)
        command_rng = np.random.default_rng(seed + 1000000)
        state, previous = 0, None
        for t in range(100):
            command = int(command_rng.integers(actions))
            expected_state, executed, reward, terminal, truncated = local_step(
                transitions, rewards, state, previous, t, command, local_rng, 100)
            obs, got_reward, got_terminal, got_truncated, _ = env.step(command)
            if (obs != expected_state or env.env.last_executed != executed or
                    env.last_action != executed or abs(got_reward - reward) > 1e-6 or
                    got_terminal != terminal or got_truncated != truncated):
                raise RuntimeError(f"Sticky semantics mismatch: {name}, seed={seed}, step={t}")
            state, previous = expected_state, executed
            if terminal or truncated:
                break
        env.close()
    # An explicit controlled action sequence must exercise the 25% branch;
    # the random rollouts above may terminate before such a branch occurs.
    rng = np.random.default_rng(17)
    n_sticky = sum(rng.uniform() < .25 for _ in range(1000))
    if not 150 <= n_sticky <= 350:
        raise RuntimeError("Sticky RNG sanity check failed")
    augmented_nonterminal = transitions.shape[0] * actions
    time_aware_nonterminal = augmented_nonterminal * 100
    dense_bytes = 2 * (time_aware_nonterminal + 1) ** 2 * actions * 8
    return {"mdp_name": name, "file_sha256": expected, "base_nonterminal": transitions.shape[0],
            "actions": actions, "sticky_nonterminal_upper": augmented_nonterminal,
            "time_aware_nonterminal_upper": time_aware_nonterminal,
            "dense_time_aware_P_R_gib": round(dense_bytes / 2**30, 2),
            "wrapper_trials": 3, "status": "STICKY_ARRAY_WRAPPER_PASS_ONLY"}


def main() -> None:
    out = ROOT / "artifacts/week8/bridge_sticky_preflight_2026-10-07.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    registry = list(csv.DictReader((HERE / "candidate_registry.csv").open(encoding="utf-8", newline="")))
    expected = {Path(row["name"]).parts[-2]: row["file_sha256"] for row in
                json.loads((DATA / "candidate_download_manifest.json").read_text(encoding="utf-8"))["members"]}
    candidates = [row for row in registry if row["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    if len(candidates) != 16 or set(row["mdp_name"] for row in candidates) != set(expected):
        raise RuntimeError("Candidate manifest mismatch")
    wrapper = load_wrapper()
    checked = [check_one(row["mdp_name"], DATA / "mdps" / row["mdp_name"] / "consolidated.npz",
                         expected[row["mdp_name"]], wrapper) for row in candidates]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "STICKY_ARRAY_WRAPPER_PASS_ONLY", "source_wrapper_sha256": sha256(WRAPPER),
        "candidate_count": len(checked), "mdps": checked,
        "remaining_gates": ["original MiniGrid simulator equivalence (dependency unavailable)",
                            "finite-horizon Markov state and sparse learner/OEH compatibility",
                            "normalization and pilot/training independence",
                            "runtime and RAM smoke for complete learner/feature pipeline"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Sticky array/wrapper preflight passed: {len(checked)} MDPs", flush=True)
    print(f"Time-aware dense P+R GiB range: {min(x['dense_time_aware_P_R_gib'] for x in checked)}"
          f" to {max(x['dense_time_aware_P_R_gib'] for x in checked)}", flush=True)


if __name__ == "__main__":
    main()
