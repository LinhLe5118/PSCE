"""Exercise reward-seeking BRIDGE NPZ paths against the authors' MiniGrid env.

The policy is computed from the published true MDP only for fidelity testing;
it is privileged and is never supplied to OEH or an offline learner.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from benchmarks.bridge.official_env_preflight import DATA, OUTPUT as RANDOM_OUTPUT
from benchmarks.bridge.official_env_preflight import ROOT, author_env_module, sha256

OUTPUT = ROOT / "artifacts/week8/bridge_official_env_guided_preflight_2026-10-07.json"


def guided_policy(transitions: np.ndarray, rewards: np.ndarray, horizon: int) -> np.ndarray:
    state_count, actions = transitions.shape
    future = np.zeros(state_count + 1, dtype=float)
    index = np.where(transitions == -1, state_count, transitions)
    policy = np.empty((horizon, state_count), dtype=np.int16)
    for t in range(horizon - 1, -1, -1):
        q = rewards + future[index]
        policy[t] = np.argmax(q, axis=1)
        future = np.r_[np.max(q, axis=1), 0.0]
    return policy


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    random_check = json.loads(RANDOM_OUTPUT.read_text(encoding="utf-8"))
    if random_check["status"] != "AUTHOR_GYM_RANDOM_PATH_PASS_ONLY":
        raise RuntimeError("Random path preflight incomplete")
    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    names = [r["mdp_name"] for r in registry if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    if len(names) != 16:
        raise RuntimeError("Frozen candidate count changed")
    module = author_env_module()
    rows = []
    for name in names:
        path = DATA / "mdps" / name / "consolidated.npz"
        with np.load(path, allow_pickle=False) as raw:
            transitions, rewards = raw["transitions"], raw["rewards"]
        policy = guided_policy(transitions, rewards, 100)
        env = module.build_env_maker(name)()
        env.reset()
        state, reward_events, total = 0, 0, 0.0
        for t in range(100):
            action = int(policy[t, state])
            expected = float(rewards[state, action])
            nxt = int(transitions[state, action])
            _, got, terminated, truncated, _ = env.step(action)
            if abs(got - expected) > 1e-4:
                raise RuntimeError(f"Guided reward mismatch: {name}, t={t}")
            if abs(got) > 1e-12:
                reward_events += 1
            total += got
            if terminated or truncated:
                boundary = nxt >= 0 and np.all(transitions[nxt] == -1)
                if nxt != -1 and not boundary and t != 99:
                    raise RuntimeError(f"Guided terminal mismatch: {name}, t={t}")
                break
            if nxt == -1:
                raise RuntimeError(f"Guided NPZ terminal edge not terminal in Gym: {name}")
            state = nxt
        env.close()
        rows.append({"mdp_name": name, "npz_sha256": sha256(path), "steps": t + 1,
                     "reward_events": reward_events, "return": total,
                     "status": "AUTHOR_GYM_GUIDED_PATH_PASS_ONLY"})
        print(f"{name}: guided steps={t+1}, nonzero rewards={reward_events}", flush=True)
    OUTPUT.write_text(json.dumps({"status": "AUTHOR_GYM_GUIDED_PATH_PASS_ONLY",
        "random_path_manifest_sha256": sha256(RANDOM_OUTPUT), "mdps": rows,
        "reward_hit_mdps": sum(x["reward_events"] > 0 for x in rows),
        "remaining": ["exhaustive state mapping", "offline log/label generation",
                      "full OEH feature and learner resource gate"]}, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
