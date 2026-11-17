"""Reference-faithful published dataset-difficulty scores for tabular pilots.

* TQ and SACo (Schweighofer et al., CoLLAs 2022): trajectory quality
  ``(g_D - g_random) / (g_expert - g_random)`` from mean undiscounted episode
  returns, and state-action coverage ``u_sa(D) / u_sa(D_reference)``. The
  random and expert reference datasets are generated from the true MDP with
  the uniform policy and an optimal policy, using the pilot's size, horizon and
  initial distribution; the coverage reference is the uniform-random dataset.
  This is privileged information unavailable to OEH, reported as such.
* ERI (Swazinna et al., 2021): ``(max episode return - mean) / mean``, defined
  only for positive mean return; an eligibility flag is reported.
* BWD (Bellman-Wasserstein distance, ICLR 2026), tabular instantiation: the
  behavioral critic is the tabular SARSA fixed point (empirical-model value of
  the empirical behavior, zero for unvisited pairs); actions are one-hot, so
  ``||a' - a||^2 = 2 [a != a']``; the reference is the uniform random policy on
  the dataset's state marginal; the state-conditional transport problem with
  cost ``Q(s,a') - ||a' - a||^2`` is solved exactly (the ``epsilon -> 0`` limit
  of the entropic dual) and averaged over dataset states.

Only complete episodes (terminal or horizon-truncated) of the x10 pilot count.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def episode_returns(arrays: dict) -> np.ndarray:
    ends = np.asarray(arrays["terminal"], bool) | np.asarray(arrays["truncated"], bool)
    reward = np.asarray(arrays["reward"], float)
    episode = np.concatenate([[0], np.cumsum(ends)[:-1]])
    complete = episode < (ends.sum())
    if not complete.any():
        return np.array([reward.sum()])
    return np.bincount(episode[complete], weights=reward[complete])


def unique_pairs(arrays: dict) -> int:
    return len(set(zip(np.asarray(arrays["state"]).tolist(), np.asarray(arrays["action"]).tolist())))


def exact_transport(cost: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    """Minimum-cost coupling of two discrete marginals (exact linear program)."""
    from scipy.optimize import linprog

    n, m = cost.shape
    equality = np.zeros((n + m, n * m))
    for i in range(n):
        equality[i, i * m:(i + 1) * m] = 1.0
    for j in range(m):
        equality[n + j, j::m] = 1.0
    result = linprog(cost.ravel(), A_eq=equality, b_eq=np.concatenate([source, target]),
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError("Transport linear program failed")
    return float(result.fun)


def bellman_wasserstein(arrays: dict, num_states: int, num_actions: int, gamma: float) -> float:
    """Tabular BWD of a dataset (see module docstring)."""
    s = np.asarray(arrays["state"], dtype=np.int64)
    a = np.asarray(arrays["action"], dtype=np.int64)
    n = np.asarray(arrays["next_state"], dtype=np.int64)
    r = np.asarray(arrays["reward"], dtype=float)
    t = np.asarray(arrays["terminal"], dtype=bool)
    counts = np.zeros((num_states, num_actions))
    np.add.at(counts, (s, a), 1.0)
    rewards = np.zeros_like(counts)
    np.add.at(rewards, (s, a), r)
    nxt = np.zeros((num_states, num_actions, num_states))
    np.add.at(nxt, (s[~t], a[~t], n[~t]), 1.0)
    seen = counts > 0
    p_hat = np.divide(nxt, counts[..., None], out=np.zeros_like(nxt), where=counts[..., None] > 0)
    r_hat = np.divide(rewards, counts, out=np.zeros_like(rewards), where=seen)
    totals = counts.sum(1, keepdims=True)
    beta = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    # SARSA fixed point: Q = r + gamma P beta Q on visited pairs, zero elsewhere.
    transfer = np.einsum("san,nb->sanb", p_hat, beta).reshape(num_states * num_actions, -1)
    q = np.linalg.solve(np.eye(transfer.shape[0]) - gamma * transfer, r_hat.ravel()).reshape(counts.shape)
    q[~seen] = 0.0
    distance = 2.0 * (1.0 - np.eye(num_actions))          # one-hot squared Euclidean
    uniform = np.full(num_actions, 1.0 / num_actions)
    weights = totals[:, 0] / totals.sum()
    value = 0.0
    for state in np.flatnonzero(weights > 0):
        cost = q[state][None, :] - distance                 # rows: behavior a, cols: random a'
        value += weights[state] * exact_transport(cost, beta[state], uniform)
    return float(value)


def _task(task: dict) -> dict:
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import make_benchmark_mdp
    from scripts.data.generate_offline_datasets import collect_trajectories
    from oeh.tabular import optimal_values
    from types import SimpleNamespace

    instance, row = task["instance"], task["row"]
    mdp = make_benchmark_mdp(type("Spec", (), instance)())
    with np.load(Path(task["dataset"]) / row["file"], allow_pickle=False) as loaded:
        pilot = {name: loaded[name][:int(row["size"])] for name in loaded.files}
    size = int(row["size"])
    r = np.einsum("san,san->sa", mdp.P, mdp.R)
    _, best = optimal_values(mdp.P, r, mdp.gamma)
    references = {}
    for name, probabilities in (("random", np.full((mdp.num_states, mdp.num_actions), 1.0 / mdp.num_actions)),
                                ("expert", np.eye(mdp.num_actions)[best])):
        policy = SimpleNamespace(probabilities=probabilities, family=name, validate=lambda _m: None)
        digest = hashlib.sha256(f"tq-saco-reference|{instance['instance_id']}|{name}".encode()).digest()
        references[name] = collect_trajectories(mdp, policy, seed=int.from_bytes(digest[:4], "little"),
                                                size=size, horizon=100)
    g = {name: float(episode_returns(data).mean()) for name, data in (("pilot", pilot), *references.items())}
    returns = episode_returns(pilot)
    mean = float(returns.mean())
    denominator = g["expert"] - g["random"]
    return {"mdp_id": row["instance_id"], "policy_family": row["policy_family"],
            "tq": (g["pilot"] - g["random"]) / denominator if abs(denominator) > 0 else 0.0,
            "saco": unique_pairs(pilot) / max(unique_pairs(references["random"]), 1),
            "eri": (float(returns.max()) - mean) / mean if mean > 0 else 0.0,
            "eri_eligible": float(mean > 0),
            "bwd": bellman_wasserstein(pilot, mdp.num_states, mdp.num_actions, mdp.gamma)}


def compute(dataset: Path, out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        frame = pd.read_csv(out)
        if "bwd" in frame.columns:
            return frame
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = sorted((row for row in manifest["datasets"]
                   if row.get("stream") == "pilot" and int(row.get("size_multiplier", -1)) == 10
                   and row.get("benchmark_split", "evaluation") == "evaluation"),
                  key=lambda row: (row["instance_id"], row["policy_family"]))
    tasks = [{"instance": instances[row["instance_id"]], "row": row, "dataset": str(dataset)} for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(_task, tasks, chunksize=8)))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    frame = compute(ROOT / args.dataset, ROOT / args.output, args.workers)
    print(f"{len(frame)} published-score rows -> {args.output}")


if __name__ == "__main__":
    main()
