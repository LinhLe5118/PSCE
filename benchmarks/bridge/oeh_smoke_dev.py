"""Measure frozen OEH-B posterior on the BRIDGE development task only."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import structured_random_policy
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402

TASK = "MiniGrid-LavaGapS5-v0"
OUTPUT = ROOT / "artifacts/week8/bridge_oehb_smoke_dev_v2_2026-10-07.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mdp = load_sticky_mdp(TASK)
    policy = structured_random_policy(mdp, epsilon=.2, seed=20261007)
    pilot = collect_trajectories(mdp, policy, seed=2026100701,
                                 size=10 * mdp.num_states * mdp.num_actions, horizon=100)
    frozen = json.loads((ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json")
                        .read_text(encoding="utf-8"))["oeh_final"]
    start = time.perf_counter()
    outcome = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution, num_states=mdp.num_states,
        num_actions=mdp.num_actions, gamma=mdp.gamma, draws=8,
        seed=[20261007], **frozen["structure"])
    forecast = predictive.requirement_forecast(outcome)
    elapsed = time.perf_counter() - start
    payload = {"status": "DEVELOPMENT_OEHB_SMOKE_ONLY_NO_TEST_LABEL",
        "task": TASK, "exclude_from_independent_test": True,
        "source_npz_sha256": mdp.metadata["source_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pilot_transitions": len(pilot["state"]), "draws": 8,
        "gamma": mdp.gamma, "structure": frozen["structure"],
        "seconds": round(elapsed, 3),
        "learner_law": {name: np.asarray(value["probabilities"]).tolist()
                        for name, value in forecast.items()},
        "warning": "Smoke draw count 8 differs from frozen 64; no independent test outcome read."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"OEH-B BRIDGE development smoke complete in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
