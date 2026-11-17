"""Calibrate one-draw OEH-B smoke against a completed sealed task."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402
from .fast_backend import install  # noqa: E402


def main() -> None:
    pilot_dir = ROOT / "datasets/bridge_v1_1_1/offline_pilot_fresh_2026-10-08"
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in manifest["pilot_groups"]
               if r["mdp_id"] == "atlantis_20_fs30" and r["policy_family"] == "structured_random")
    with np.load(pilot_dir / row["file"], allow_pickle=False) as loaded:
        pilot = {key: loaded[key] for key in loaded.files}
    install()
    started = time.perf_counter()
    predictive.posterior_outcomes(
        pilot, row["initial_state_distribution"], num_states=row["num_states"],
        num_actions=row["num_actions"], gamma=.99, draws=1, seeds=1,
        budgets=(1, 2, 5, 10, 20, 50), horizon=20, evaluation_horizon=20,
        seed=[int.from_bytes(hashlib.sha256(b"expanded-resource-calibration").digest()[:8], "little")],
        minimum_dynamic_range=.2, reward_bound=1.0,
        reward_prior="unseen_zero", transition_base="uniform", behavior_prior="mixture")
    smoke_seconds = time.perf_counter() - started
    groups = ROOT / "artifacts/week8/bridge_fresh_full_oehb_preoutcome_2026-10-08/groups"
    full = None
    for file in groups.glob("*.json"):
        row_full = json.loads(file.read_text(encoding="utf-8"))
        if row_full["mdp_id"] == "atlantis_20_fs30" and row_full["policy_family"] == "structured_random":
            full = row_full
            break
    if full is None:
        raise RuntimeError("Sealed calibration group missing")
    print(json.dumps({"task": row["mdp_id"], "states": row["num_states"],
                      "one_draw_one_seed_seconds": smoke_seconds,
                      "sealed_full_64_draw_10_seed_seconds": full["runtime_seconds"],
                      "observed_ratio": full["runtime_seconds"] / smoke_seconds}, indent=2))


if __name__ == "__main__":
    main()
