"""Compare published BRIDGE MiniGrid NPZ rollouts with the authors' Gym code.

This is a partial fidelity check on random action paths, not an exhaustive
state-bijection proof or an offline learning benchmark. The official MiniGrid
module is loaded unchanged; a process-local Gymnasium attribute shim restores
the older Wrapper behavior expected by its pinned MiniGrid 2.2.1 dependency.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bridge_v1_1_1"
WHEEL = DATA / "vendor_download/minigrid-2.2.1-py3-none-any.whl"
OFFICIAL = DATA / "official_source/effective_horizon/envs/minigrid.py"
OUTPUT = ROOT / "artifacts/week8/bridge_official_env_preflight_2026-10-07.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def author_env_module():
    sys.path.insert(0, str(WHEEL))
    import minigrid

    if minigrid.__version__ != "2.2.1":
        raise RuntimeError("Wrong MiniGrid version")
    minigrid.register_minigrid_envs()
    # Gymnasium's newer Wrapper no longer forwards unknown attributes to env.
    # The author source and MiniGrid package are unchanged on disk.
    gym.Wrapper.__getattr__ = lambda self, name: getattr(
        object.__getattribute__(self, "env"), name)
    spec = importlib.util.spec_from_file_location("bridge_official_minigrid", OFFICIAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load official MiniGrid module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_one(name: str, module) -> dict:
    path = DATA / "mdps" / name / "consolidated.npz"
    with np.load(path, allow_pickle=False) as raw:
        transitions, rewards = raw["transitions"], raw["rewards"]
    if transitions.shape != rewards.shape:
        raise RuntimeError(f"Bad NPZ shape: {name}")
    env = module.build_env_maker(name)()
    if env.action_space.n != transitions.shape[1] or env.horizon != 100:
        raise RuntimeError(f"Actions/horizon mismatch: {name}")
    trial_rows = []
    for seed in (0, 17, 104729):
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        state, reached_terminal, reward_events = 0, False, 0
        for t in range(100):
            action = int(rng.integers(transitions.shape[1]))
            expected_reward = float(rewards[state, action])
            next_state = int(transitions[state, action])
            _, reward, terminated, truncated, _ = env.step(action)
            if abs(reward - expected_reward) > 1e-4:
                raise RuntimeError(f"Reward mismatch: {name}, seed={seed}, step={t}")
            if abs(reward) > 1e-12:
                reward_events += 1
            if terminated or truncated:
                # The authors' test also accepts a boundary state whose every
                # outgoing edge reaches the absorbing state.
                boundary = next_state >= 0 and np.all(transitions[next_state] == -1)
                if next_state != -1 and not boundary and t != 99:
                    raise RuntimeError(f"Terminal mismatch: {name}, seed={seed}, step={t}")
                reached_terminal = True
                trial_rows.append({"seed": seed, "steps": t + 1, "terminated": True,
                                   "reward_events": reward_events})
                break
            if next_state == -1:
                raise RuntimeError(f"NPZ terminal edge not terminal in Gym: {name}")
            state = next_state
        else:
            trial_rows.append({"seed": seed, "steps": 100, "terminated": reached_terminal,
                               "reward_events": reward_events})
    env.close()
    return {"mdp_name": name, "npz_sha256": sha256(path), "trials": trial_rows,
            "status": "AUTHOR_GYM_RANDOM_PATH_PASS_ONLY"}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    names = [r["mdp_name"] for r in registry if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    if len(names) != 16:
        raise RuntimeError("Frozen BRIDGE candidate count changed")
    module = author_env_module()
    rows = []
    for name in names:
        item = check_one(name, module)
        rows.append(item)
        print(f"{name}: {len(item['trials'])} author Gym paths pass", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "AUTHOR_GYM_RANDOM_PATH_PASS_ONLY",
        "candidate_count": len(rows), "gymnasium_version": gym.__version__,
        "minigrid_wheel_sha256": sha256(WHEEL), "author_module_sha256": sha256(OFFICIAL),
        "compatibility_shim": "process-local Wrapper.__getattr__ forwarding",
        "remaining": ["exhaustive state mapping", "offline behavior/label generation",
                      "full OEH feature and learner resource gate"], "mdps": rows}, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
