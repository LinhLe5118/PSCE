"""Time a simple sticky random-policy rollout loop on a published base MDP.

This only estimates collection throughput. It is not the validated BRIDGE
adapter and it creates no pilot, training buffer, label, or experimental result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bridge_v1_1_1"
MDP = DATA / "mdps/MiniGrid-SimpleCrossingS9N1-v0/consolidated.npz"


def main() -> None:
    n = 1_000_000
    with np.load(MDP, allow_pickle=False) as source:
        transitions = source["transitions"]
        rewards = source["rewards"]
    actions = transitions.shape[1]
    rng = np.random.default_rng(20261007)
    state = np.empty(n, dtype=np.int32)
    requested = np.empty(n, dtype=np.int16)
    reward = np.empty(n, dtype=np.float32)
    next_state = np.empty(n, dtype=np.int32)
    terminal = np.empty(n, dtype=np.bool_)
    current, previous, timestep = 0, -1, 0
    started = time.perf_counter()
    for i in range(n):
        action = int(rng.integers(actions))
        executed = previous if previous >= 0 and rng.random() < 0.25 else action
        successor = int(transitions[current, executed])
        done = successor == -1 or timestep + 1 >= 100
        state[i] = current
        requested[i] = action
        reward[i] = rewards[current, executed]
        next_state[i] = successor
        terminal[i] = done
        current = 0 if done else successor
        previous = -1 if done else executed
        timestep = 0 if done else timestep + 1
    elapsed = time.perf_counter() - started
    result = {"scope": "runtime_proxy_random_policy_no_adapter_no_disk_write",
              "mdp_name": MDP.parent.name, "transitions": n,
              "seconds": round(elapsed, 3), "transitions_per_second": round(n / elapsed),
              "terminal_count": int(terminal.sum()),
              "arrays_bytes": sum(x.nbytes for x in (state, requested, reward, next_state, terminal))}
    (DATA / "rollout_runtime_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
