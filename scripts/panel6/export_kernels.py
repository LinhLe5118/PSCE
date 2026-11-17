"""Export panel-6 candidate kernels from their source libraries (task card T8; run in the isolated venv).

Runs outside the project environment, because MDP Playground pins its own Gymnasium and ale-py
(the project keeps Gymnasium 1.0.0 for the sealed FrozenLake panel). Only numpy and the source libraries are imported.

    D:/Tools/venvs/mdpp/Scripts/python.exe -I scripts/panel6/export_kernels.py \
        --candidates <candidates.json> --out <dir> [--verify 400]

``candidates.json`` is written by ``scripts.analysis.panel6_families`` and lists ``{"mdp_id", "family", ...parameters}``.
Each kernel is written to ``<out>/<mdp_id>.npz`` with ``T`` (S, A, S), ``R`` (S, A), ``rho`` (S,) and ``terminal`` (indices),
in the source's own (episodic) form; the continuing conversion is done in the project environment.
``--verify n`` checks every MDP Playground kernel against ``n`` calls of the environment's own ``step`` per state-action pair.
``export_meta.json`` also keeps each kernel's source tables (successor table, noise, next-state reward and start law for MDP
Playground; the text layout for labmaze), so the project environment can rebuild every kernel without the libraries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import warnings

import numpy as np

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # north, east, south, west


# --------------------------------------------------------------------------- #
# MDP Playground (Rajan et al., 2021), discrete toy MDPs
# --------------------------------------------------------------------------- #

def mdpp_env(c: dict):
    import io
    import contextlib
    from mdp_playground.envs import RLToyEnv

    config = {"state_space_type": "discrete", "action_space_size": c["actions"], "diameter": c["diameter"],
              "delay": 0, "sequence_length": 1, "reward_density": c["reward_density"], "make_denser": False,
              "terminal_state_density": c["terminal_state_density"], "transition_noise": c["transition_noise"],
              "reward_noise": 0, "reward_scale": 1.0, "reward_shift": 0.0, "repeats_in_sequences": False,
              "maximally_connected": True, "seed": c["source_seed"], "log_level": logging.ERROR}
    with contextlib.redirect_stdout(io.StringIO()):  # the library prints its matrices
        return RLToyEnv(**config)


def successor_kernel(succ: np.ndarray, eta: float) -> np.ndarray:
    """MDP Playground's discrete dynamics: the successor ``succ[s, a]`` with probability 1 - eta, else uniform over the
    other S - 1 states (``RLToyEnv.transition_function``)."""
    S, A = succ.shape
    T = np.full((S, A, S), eta / (S - 1)) if eta else np.zeros((S, A, S))
    T[np.arange(S)[:, None], np.arange(A)[None, :], succ] = 1.0 - eta
    return T


def next_state_reward(env) -> np.ndarray:
    """With sequence length 1 and no delay, the reward depends on the next state only."""
    reward = np.zeros(int(env.state_space_size[0]))
    for seq, value in env.rewardable_sequences.items():
        reward[int(seq[-1])] = float(value) * env.reward_scale + env.reward_shift
    return reward


def mdpp_kernel(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    env = mdpp_env(c)
    S, A = int(env.state_space_size[0]), int(env.action_space_size[0])
    succ = np.array([[int(env.transition_matrix[s, a]) for a in range(A)] for s in range(S)])
    eta, reward = float(env.transition_noise or 0.0), next_state_reward(env)
    T = successor_kernel(succ, eta)
    terminal = np.array(sorted(int(s) for s in np.asarray(env.config["terminal_states"]).ravel()), dtype=int)
    rho = np.asarray(env.config["relevant_init_state_dist"], float)
    source = {"succ": succ.tolist(), "eta": eta, "next_reward": reward.tolist(), "rho": rho.tolist(), "terminal": terminal.tolist()}
    return T, T @ reward, rho / rho.sum(), terminal, source


def mdpp_verify(c: dict, T: np.ndarray, R: np.ndarray, n: int) -> dict:
    """Compare the kernel with ``n`` calls of the environment's own ``step`` per state-action pair.

    Returns, over the cells with 0 < T < 1, the Bonferroni-adjusted smallest two-sided exact binomial p-value of the
    observed count (and the largest normal z for reference); whether the visited next states agree with the support of
    T; and the largest |observed reward - r(next state)|.
    """
    import io
    import contextlib
    from scipy.stats import binom

    env = mdpp_env(c)
    reward = next_state_reward(env)
    S, A = R.shape
    z, support, reward_error, p_min, cells = 0.0, True, 0.0, 1.0, 0
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.reset()
        for s in range(S):
            for a in range(A):
                counts = np.zeros(S)
                for _ in range(n):
                    env.set_markov_state(env.dtype_s(s))
                    nxt, r, *_ = env.step(a)
                    nxt = int(np.asarray(nxt).ravel()[0])
                    counts[nxt] += 1
                    reward_error = max(reward_error, abs(float(r) - reward[nxt]))
                inner = (T[s, a] > 0) & (T[s, a] < 1)
                se = np.sqrt(T[s, a][inner] * (1 - T[s, a][inner]) / n)
                z = max(z, float((np.abs(counts[inner] / n - T[s, a][inner]) / se).max(initial=0.0)))
                k, q = counts[inner], T[s, a][inner]
                if len(k):
                    p_min = min(p_min, float(np.minimum(1, 2 * np.minimum(binom.cdf(k, n, q), binom.sf(k - 1, n, q))).min()))
                    cells += len(k)
                support &= bool((counts[T[s, a] == 0] == 0).all() and (counts[T[s, a] == 1] == n).all())
    return {"bonferroni_p": min(1.0, p_min * max(cells, 1)), "max_z": z, "support_ok": support,
            "max_abs_reward_error": reward_error}


# --------------------------------------------------------------------------- #
# labmaze (DeepMind Lab text-maze generator) with slippery four-way moves
# --------------------------------------------------------------------------- #

def maze_layout(c: dict) -> np.ndarray:
    import labmaze
    maze = labmaze.RandomMaze(height=c["height"], width=c["width"], max_rooms=c["max_rooms"], room_min_size=c["room_min_size"],
                              room_max_size=c["room_max_size"], extra_connection_probability=c["extra_connection_probability"],
                              spawns_per_room=1, objects_per_room=1, simplify=False, random_seed=c["source_seed"])
    return np.array([list(row) for row in str(maze.entity_layer).splitlines()])


def maze_kernel(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    grid = maze_layout(c)
    return (*grid_kernel(grid, c["success"]), {"layout": ["".join(row) for row in grid], "success": c["success"]})


def grid_kernel(grid: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four-way moves on the free cells of a text maze: intended direction with probability p, each other direction with
    (1 - p) / 3; walls ('*') block; reward 1 on entering a goal cell ('G'); start uniform over spawn cells ('P')."""
    cells = [tuple(x) for x in np.argwhere(grid != "*")]
    index = {cell: i for i, cell in enumerate(cells)}
    S, A = len(cells), 4
    T, R = np.zeros((S, A, S)), np.zeros((S, A))
    goal = np.array([grid[cell] == "G" for cell in cells])
    for i, (y, x) in enumerate(cells):
        for a in range(A):
            for move in range(A):
                prob = p if move == a else (1 - p) / (A - 1)
                dy, dx = MOVES[move]
                j = index.get((y + dy, x + dx), i)  # walls block
                T[i, a, j] += prob
                R[i, a] += prob * goal[j]
    spawn = np.array([grid[cell] == "P" for cell in cells], float)
    terminal = np.flatnonzero(goal)
    return T, R, spawn / spawn.sum(), terminal


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", type=int, default=0)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"python": platform.python_version(), "numpy": np.__version__, "kernels": {}}
    for name in ("mdp_playground", "labmaze", "gymnasium"):
        try:
            from importlib.metadata import version
            meta[name] = version(name.replace("_", "-"))
        except Exception:
            meta[name] = None
    for c in candidates:
        builder = {"mdpp": mdpp_kernel, "labmaze": maze_kernel}[c["family"]]
        T, R, rho, terminal, source = builder(c)
        if not (np.allclose(T.sum(-1), 1) and (T >= 0).all() and np.isclose(rho.sum(), 1) and (R >= 0).all() and (R <= 1).all()):
            raise RuntimeError(f"invalid kernel {c['mdp_id']}")
        path = args.out / f"{c['mdp_id']}.npz"
        np.savez(path, T=T, R=R, rho=rho, terminal=terminal)
        entry = {"states": int(T.shape[0]), "actions": int(T.shape[1]), "sha256": sha(path), "source": source}
        if args.verify and c["family"] == "mdpp":
            entry["verify"] = {"steps_per_pair": args.verify, **mdpp_verify(c, T, R, args.verify)}
        meta["kernels"][c["mdp_id"]] = entry
    (args.out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {len(candidates)} kernels to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
