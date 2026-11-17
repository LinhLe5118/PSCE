"""Complete full OEH-B on 14 previously opened BRIDGE MiniGrid MDPs.

This is a post-outcome exploratory extension. It deliberately does not use the
prelabel-sealed status names. The fixed 64-draw/10-seed/gamma=.99/H=100 method
and the 70 pilot logs are reused without reading the labels in this stage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import predictive  # noqa: E402
from .fast_backend import install  # noqa: E402

DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
ADDENDUM = ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json"
OUTPUT = ROOT / "artifacts/week8/bridge_opened_minigrid_full_oehb_exploratory_2026-10-09"
BUDGETS = (1, 2, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one(row: dict, structure: dict, source_hash: str, backend_hash: str,
            predictive_hash: str) -> dict:
    stem = hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
    path = OUTPUT / "groups" / f"{stem}.json"
    identity = {
        "mdp_id": row["mdp_id"], "policy_family": row["policy_family"],
        "pilot_sha256": row["sha256"], "source_script_sha256": source_hash,
        "backend_sha256": backend_hash, "predictive_sha256": predictive_hash,
    }
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Stale exploratory checkpoint: {path.name}")
        return saved
    started = time.perf_counter()
    try:
        pilot_file = DATA / row["file"]
        if sha256(pilot_file) != row["sha256"]:
            raise RuntimeError(f"Pilot hash mismatch: {pilot_file}")
        with np.load(pilot_file, allow_pickle=False) as archive:
            pilot = {key: archive[key] for key in archive.files}
        states, actions = int(row["num_states"]), int(row["num_actions"])
        if len(pilot["state"]) != 10 * states * actions or int(row["horizon"]) != 100:
            raise RuntimeError("Pilot size or horizon mismatch")
        seed = hashlib.sha256(
            f"bridge-oehb-v1|{row['mdp_id']}|{row['policy_family']}".encode()).digest()
        outcomes = predictive.posterior_outcomes(
            pilot, row["initial_state_distribution"], num_states=states,
            num_actions=actions, gamma=.99, draws=64, seeds=10,
            budgets=BUDGETS, horizon=100, evaluation_horizon=100,
            seed=[int.from_bytes(seed[:8], "little")],
            minimum_dynamic_range=.2, reward_bound=1.0, **structure,
        )
        forecast = predictive.requirement_forecast(outcomes)
        nr = outcomes.normalized_return
        result = {
            **identity, "status": "EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN",
            "states": states, "actions": actions, "horizon": 100,
            "source_npz_sha256": row["source_npz_sha256"],
            "posterior": outcomes.posterior, "sampling": outcomes.sampling,
            "learners": {name: {
                "probabilities": value["probabilities"].tolist(),
                "events": value["events"].tolist(),
                "median_event": value["median_event"],
                "censored_fraction": value["censored_fraction"],
                "mean_normalized_return": nr[:, :, :, i].mean((0, 1)).tolist() if len(nr) else None,
                "success_rate": (nr[:, :, :, i] >= .95).mean((0, 1)).tolist() if len(nr) else None,
            } for i, (name, value) in enumerate(forecast.items())},
            "runtime_seconds": time.perf_counter() - started,
            "labels_already_opened_at_run": True,
        }
    except Exception as error:
        result = {**identity, "status": "EXPLORATORY_FULL_OEHB_FAILURE",
                  "error": repr(error), "traceback": traceback.format_exc(),
                  "runtime_seconds": time.perf_counter() - started,
                  "labels_already_opened_at_run": True}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=70,
                        help="Process the first N groups for checkpointed smoke/resume")
    args = parser.parse_args()
    if not 1 <= args.limit <= 70:
        raise ValueError("--limit must be in 1..70")
    manifest_file = DATA / "manifest.json"
    pilot = json.loads(manifest_file.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    rows = pilot["pilot_groups"]
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or len(rows) != 70 or
            len(set(row["mdp_id"] for row in rows)) != 14 or
            addendum["posterior_structure"] != {
                "reward_prior": "unseen_zero", "transition_base": "uniform",
                "behavior_prior": "mixture"}):
        raise RuntimeError("Opened MiniGrid pilot/addendum mismatch")
    source_hash = sha256(Path(__file__))
    backend_hash = sha256(ROOT / "benchmarks/bridge/fast_backend.py")
    predictive_hash = sha256(Path(predictive.__file__))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "groups").mkdir(exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Exploratory full OEH-B already complete")
    install()
    results = []
    for index, row in enumerate(rows[:args.limit], 1):
        result = run_one(row, addendum["posterior_structure"], source_hash,
                         backend_hash, predictive_hash)
        results.append(result)
        print(f"Opened MiniGrid full OEH-B {index}/{args.limit}: "
              f"{row['mdp_id']} / {row['policy_family']} "
              f"{result['status']} {result['runtime_seconds']:.1f}s", flush=True)
        if result["status"] != "EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN":
            raise RuntimeError(f"OEH-B failed on {row['mdp_id']} / {row['policy_family']}")
        gc.collect()
    if args.limit == 70:
        files = sorted((OUTPUT / "groups").glob("*.json"))
        if len(files) != 70:
            raise RuntimeError("Incomplete 70-group exploratory panel")
        summary = {
            "status": "EXPLORATORY_FULL_OEHB_COMPLETE_AFTER_LABEL_OPEN",
            "groups": 70, "tasks": 14, "labels_already_opened_at_run": True,
            "pilot_manifest_sha256": sha256(manifest_file),
            "method_addendum_sha256": sha256(ADDENDUM),
            "source_script_sha256": source_hash, "backend_sha256": backend_hash,
            "predictive_sha256": predictive_hash,
            "runtime_seconds_sum": sum(item["runtime_seconds"] for item in results),
            "group_sha256": {file.name: sha256(file) for file in files},
        }
        (OUTPUT / "manifest.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
