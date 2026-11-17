"""Verify all five frozen behavior generators on a BRIDGE development task."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import collect_trajectories, validate_buffer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/week8/bridge_behavior_suite_smoke_dev_2026-10-07.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mdp = load_sticky_mdp("MiniGrid-LavaGapS5-v0")
    start = time.perf_counter()
    policies = behavior_policy_suite(mdp, seed=20261007, variant_index=0)
    fit_seconds = time.perf_counter() - start
    rows = []
    for index, policy in enumerate(policies):
        pilot = collect_trajectories(mdp, policy, seed=2026100701 + index,
                                     size=10 * mdp.num_states * mdp.num_actions,
                                     horizon=100)
        validate_buffer(mdp, pilot, horizon=100)
        rows.append({"family": policy.family, "uses_q_star_for_behavior_generation": policy.uses_q_star,
                     "probability_sha256": hashlib.sha256(np.ascontiguousarray(
                         policy.probabilities, dtype="<f8").tobytes()).hexdigest(),
                     "pilot_transitions": len(pilot["state"]),
                     "nonzero_rewards": int(np.count_nonzero(pilot["reward"])),
                     "terminals": int(np.count_nonzero(pilot["terminal"]))})
    if len(rows) != 5:
        raise RuntimeError("Behavior suite incomplete")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "DEVELOPMENT_BEHAVIOR_SUITE_SMOKE_ONLY",
        "task": "MiniGrid-LavaGapS5-v0", "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - start,
        "behaviors": rows}, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
