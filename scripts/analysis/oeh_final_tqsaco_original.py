"""TQ and SACo computed with the authors' own code (ml-jku/OfflineRL, commit 8cc8eec).

Reproduction of Schweighofer et al., "A Dataset Perspective on Offline
Reinforcement Learning" (CoLLAs 2022). Per the authors' ``train_online.py`` and
``create_plots.py``:

* an online DQN agent (their ``source.agents.dqn.DQN`` with their fixed online
  settings: batch 32, lr 1e-4, epsilon 1 -> 0.01 over 1000 steps, target copy
  every 100 updates, train from step 33) is trained for as many transitions as
  the dataset contains; every experienced transition fills the ER ("Replay")
  buffer and the agent is evaluated greedily (10 episodes) every
  ``transitions // 500`` steps with their ``evaluate``;
* a uniform random policy collects the "random" dataset of the same size;
* ``TQ = (mean return(D) - mean return(random)) / (max online return - mean return(random))``;
* ``SACo = u_sa(D) / u_sa(ER buffer)`` with their exact unique-pair count.

The only addition is a gym-style adapter for a tabular MDP: one-hot
observations, episodes that end at a terminal state or after 100 steps, and
restarts from the initial-state distribution. The authors' Evaluator module
imports the optional ``hyperloglog`` package; its exact counter and return
routine are reproduced verbatim below instead of importing it.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "tmp/public_offlinerl"
AUTHOR_COMMIT = "8cc8eec"


class TabularEnv:
    """Gym-style view of a tabular MDP for the authors' agent and evaluator."""

    def __init__(self, mdp, horizon: int = 100):
        self.mdp, self.horizon = mdp, horizon
        self.rng = np.random.default_rng(0)
        self.terminal = set(int(s) for s in mdp.terminal_states)
        self.state, self.t = 0, 0

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        obs = np.zeros(self.mdp.num_states, dtype=np.float32)
        obs[self.state] = 1.0
        return obs

    def reset(self):
        self.state = int(self.rng.choice(self.mdp.num_states, p=self.mdp.initial_state_distribution))
        self.t = 0
        return self._obs()

    def step(self, action):
        action = int(action)
        nxt = int(self.rng.choice(self.mdp.num_states, p=self.mdp.P[self.state, action]))
        reward = float(self.mdp.R[self.state, action, nxt])
        self.state, self.t = nxt, self.t + 1
        done = nxt in self.terminal or self.t >= self.horizon
        return self._obs(), reward, done, {}


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def get_returns(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Verbatim logic of the authors' ``Evaluator.get_returns``."""
    rewards, dones = np.asarray(rewards).reshape(-1), np.asarray(dones).reshape(-1)
    returns, ret = [], 0.0
    for r, d in zip(rewards, dones):
        ret += r
        if d:
            returns.append(ret)
            ret = 0.0
    return np.asarray(returns if returns else [ret])


def unique_state_actions_exact(states: np.ndarray, actions: np.ndarray) -> int:
    """Verbatim logic of the authors' ``get_unique_state_actions_exact``."""
    states = np.concatenate([np.asarray(states), np.asarray(actions).reshape(-1, 1)], axis=1)
    unique = set()
    for i in range(len(states)):
        unique.add(",".join([str(s) for s in states[i]]))
    return len(unique)


def fast_evaluate(env, agent, all_rewards: list) -> list:
    """Exact equivalent of the authors' ``evaluate`` for a tabular one-hot env.

    With ``eval=True`` their ``DQN.policy`` uses epsilon 0, draws one uniform
    number per step and takes ``argmax Q(one_hot(s))``. The greedy action of
    every state is computed once per evaluation with the same single-state
    forward pass, and one uniform number is consumed per step so the agent's
    random stream (and hence later online training) is unchanged.
    """
    import torch

    agent.Q.eval()
    eye = np.eye(env.mdp.num_states, dtype=np.float32)
    with torch.no_grad():
        # Evaluating all tabular states together removes thousands of Python
        # forward-call overheads while preserving the author's greedy policy.
        greedy = agent.Q(torch.from_numpy(eye).to(agent.device)).argmax(dim=1).cpu().tolist()
    returns = []
    for seed in range(10):
        env.seed(seed)
        env.reset()
        done, rewards = False, []
        while not done:
            agent.rng.uniform(0, 1)
            _, reward, done, _ = env.step(greedy[env.state])
            rewards.append(reward)
        returns.append(sum(rewards))  # built-in sum, as in the authors' code (compensated on 3.12)
    all_rewards.append(np.mean(returns))
    return all_rewards


def online_reference(mdp, transitions: int, seed: int, exact_evaluate: bool = False) -> tuple[float, int]:
    """The authors' online DQN run: max greedy return and ER-buffer pair count."""
    sys.path.insert(0, str(AUTHOR))
    import torch
    from source.agents.dqn import DQN
    from source.utils.buffer import ReplayBuffer
    from source.utils.evaluation import evaluate

    batch_size, lr, train_every, train_start_iter = 32, 1e-4, 1, 32
    evaluate_every = max(transitions // 500, 1)
    env, eval_env = TabularEnv(mdp), TabularEnv(mdp)
    obs_space = mdp.num_states
    agent = DQN(obs_space, mdp.num_actions, 0.95, lr, seed=seed)
    buffer = ReplayBuffer(obs_space, min(50000, transitions), batch_size, seed=seed)
    states, actions = [], []
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    writer, all_rewards, all_avds = _NullWriter(), [], []
    done = True
    for iteration in range(transitions):
        if done:
            state = env.reset()
        action, _, _ = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, done)
        states.append(state)
        actions.append(action)
        state = next_state
        if (iteration + 1) % train_every == 0 and iteration > train_start_iter:
            agent.train(buffer, writer)
        if (iteration + 1) % evaluate_every == 0:
            if exact_evaluate:
                all_rewards, all_avds = evaluate(eval_env, agent, writer, all_rewards, all_avds)
            else:
                all_rewards = fast_evaluate(eval_env, agent, all_rewards)
    return float(np.max(all_rewards)), unique_state_actions_exact(np.asarray(states), np.asarray(actions)), all_rewards


def _task(task: dict) -> dict:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # tiny MLP: CPU is faster and keeps the GPU free
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import make_benchmark_mdp
    from scripts.analysis.oeh_final_published import episode_returns

    instance, rows = task["instance"], task["rows"]
    mdp = make_benchmark_mdp(type("Spec", (), instance)())
    size = int(rows[0]["size"])
    digest = hashlib.sha256(f"tq-saco-original|{instance['instance_id']}".encode()).digest()
    seed = int.from_bytes(digest[:4], "little") % (2**31 - 1)
    max_online, online_usap, _ = online_reference(mdp, size, seed)
    # Random reference dataset of the same size (uniform behavior).
    env = TabularEnv(mdp)
    env.seed(seed + 1)
    rewards, dones, done = [], [], True
    for _ in range(size):
        if done:
            env.reset()
        _, reward, done, _ = env.step(int(env.rng.integers(mdp.num_actions)))
        rewards.append(reward)
        dones.append(done)
    random_return = float(get_returns(np.asarray(rewards), np.asarray(dones)).mean())
    out = []
    for row in rows:
        with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
            pilot = {name: loaded[name][:int(row["size"])] for name in loaded.files}
        mean_return = float(episode_returns(pilot).mean())
        one_hot = np.eye(mdp.num_states, dtype=np.float32)[pilot["state"]]
        usap = unique_state_actions_exact(one_hot, pilot["action"])
        denominator = max_online - random_return
        out.append({"mdp_id": row["instance_id"], "policy_family": row["policy_family"],
                    "tq_original": (mean_return - random_return) / denominator if denominator != 0 else 0.0,
                    "saco_original": usap / max(online_usap, 1),
                    "tq_reference_online_max": max_online, "tq_reference_random": random_return,
                    "saco_reference_pairs": online_usap})
    return out


def compute(dataset: Path, out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = [row for row in manifest["datasets"]
            if row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10
            and row.get("benchmark_split", "evaluation") == "evaluation"]
    by_mdp = {}
    for row in sorted(rows, key=lambda r: (r["instance_id"], r["policy_family"])):
        by_mdp.setdefault(row["instance_id"], []).append(row)
    tasks = [{"instance": instances[m], "rows": r, "dataset": str(dataset)} for m, r in by_mdp.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame([item for chunk in pool.map(_task, tasks) for item in chunk])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    frame = compute(ROOT / args.dataset, ROOT / args.output, args.workers)
    print(f"{len(frame)} rows -> {args.output}")
