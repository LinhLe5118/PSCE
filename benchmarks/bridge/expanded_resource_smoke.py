"""One-draw, one-seed label-free OEH-B timing on a new BRIDGE MDP.

This is a resource probe only. It neither fits the final 64-draw predictor nor
reads empirical learner-requirement labels. Each task writes its own record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import collect_trajectories

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402
from .expanded_adapter import PANEL, registry  # noqa: E402
from .expanded_preflight import load_eligible_mdp  # noqa: E402
from .fast_backend import install  # noqa: E402

OUT = PANEL / "resource_smoke"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(registry()))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    output = OUT / f"{args.task}.json"
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    mdp = load_eligible_mdp(args.task)
    horizon = int(mdp.metadata["horizon"])
    ordinal = list(registry()).index(args.task)
    policies = behavior_policy_suite(mdp, seed=seed("bridge-expanded-behavior-v1", args.task),
                                     variant_index=ordinal)
    policy = next(p for p in policies if p.family == "structured_random")
    pilot = collect_trajectories(mdp, policy,
                                 seed=seed("bridge-expanded-resource-pilot", args.task),
                                 size=10 * mdp.num_states * mdp.num_actions,
                                 horizon=horizon)
    pilot_seconds = time.perf_counter() - started
    install()
    draw_started = time.perf_counter()
    outcomes = predictive.posterior_outcomes(
        pilot, mdp.initial_state_distribution.tolist(),
        num_states=mdp.num_states, num_actions=mdp.num_actions,
        gamma=.99, draws=1, seeds=1, budgets=(1, 2, 5, 10, 20, 50),
        horizon=horizon, evaluation_horizon=horizon,
        seed=[seed("bridge-expanded-resource-draw", args.task)],
        minimum_dynamic_range=.2, reward_bound=1.0,
        reward_prior="unseen_zero", transition_base="uniform", behavior_prior="mixture")
    draw_seconds = time.perf_counter() - draw_started
    payload = {"status": "EXPANDED_RESOURCE_SMOKE_NO_LABELS", "mdp_id": args.task,
               "policy_family": policy.family, "states": mdp.num_states,
               "actions": mdp.num_actions, "horizon": horizon,
               "pilot_nonzero_rewards": int(np.count_nonzero(pilot["reward"])),
               "pilot_seconds_including_adapter_and_behavior": pilot_seconds,
               "one_draw_one_seed_seconds": draw_seconds,
               "observed_output_shape": list(outcomes.normalized_return.shape),
               "methodology": "Not a valid 64-draw timing estimate; report only as lower-bound resource probe",
               "source_sha256": {"script": sha256(Path(__file__)),
                                 "npz": mdp.metadata["source_sha256"]},
               "test_labels_read": False}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task": args.task, "S": mdp.num_states,
                      "pilot_seconds": round(pilot_seconds, 1),
                      "one_draw_one_seed_seconds": round(draw_seconds, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
