"""Checkpointed 64-draw OEH-B preoutcome panel for BRIDGE test pilots."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
ADDENDUM = ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json"
HORIZON_ADDENDUM = ROOT / "artifacts/week8/bridge_transfer_horizon_addendum_2026-10-07.json"
OUTPUT = ROOT / "artifacts/week8/bridge_oehb_preoutcome_2026-10-07"
BUDGETS = (1, 2, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(task: dict) -> dict:
    row = task["row"]
    output = Path(task["output"])
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if (saved.get("pilot_sha256") != row["sha256"] or
                saved.get("source_script_sha256") != task["source_script_sha256"] or
                saved.get("range_addendum_sha256") != task["range_addendum_sha256"] or
                saved.get("horizon_addendum_sha256") != task["horizon_addendum_sha256"]):
            raise RuntimeError(f"Stale OEH-B checkpoint: {output.name}")
        return {"status": saved["status"], "task": row["mdp_id"],
                "behavior": row["policy_family"], "seconds": saved.get("runtime_seconds", 0)}
    start = time.perf_counter()
    try:
        pilot_path = DATA / row["file"]
        if sha256(pilot_path) != row["sha256"]:
            raise RuntimeError("Pilot hash changed")
        with np.load(pilot_path, allow_pickle=False) as loaded:
            pilot = {key: loaded[key] for key in loaded.files}
        if len(pilot["state"]) != 10 * row["num_states"] * row["num_actions"]:
            raise RuntimeError("Pilot x10 size mismatch")
        seed_digest = hashlib.sha256(
            f"bridge-oehb-v1|{row['mdp_id']}|{row['policy_family']}".encode()).digest()
        outcomes = predictive.posterior_outcomes(
            pilot, row["initial_state_distribution"],
            num_states=int(row["num_states"]), num_actions=int(row["num_actions"]),
            gamma=.99, draws=64, seeds=10, budgets=BUDGETS,
            evaluation_horizon=100,
            seed=[int.from_bytes(seed_digest[:8], "little")],
            minimum_dynamic_range=.2,
            **task["posterior_structure"])
        forecast = predictive.requirement_forecast(outcomes)
        nr = outcomes.normalized_return
        document = {"status": "OEHB_PREOUTCOME_ONLY", "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"], "pilot_sha256": row["sha256"],
            "source_npz_sha256": row["source_npz_sha256"],
            "source_script_sha256": task["source_script_sha256"],
            "range_addendum_sha256": task["range_addendum_sha256"],
            "horizon_addendum_sha256": task["horizon_addendum_sha256"],
            "structure": task["posterior_structure"], "draws": 64,
            "gamma": .99, "minimum_dynamic_range": .2,
            "evaluation_horizon": 100,
            "posterior": outcomes.posterior, "sampling": outcomes.sampling,
            "learners": {name: {
                "probabilities": value["probabilities"].tolist(),
                "events": value["events"].tolist(),
                "median_event": value["median_event"],
                "censored_fraction": value["censored_fraction"],
                "mean_normalized_return": nr[:, :, :, i].mean((0, 1)).tolist() if len(nr) else None,
                "success_rate": (nr[:, :, :, i] >= .95).mean((0, 1)).tolist() if len(nr) else None,
            } for i, (name, value) in enumerate(forecast.items())},
            "runtime_seconds": time.perf_counter() - start,
            "test_labels_read": False}
    except Exception as exc:
        document = {"status": "OEHB_PREOUTCOME_FAILURE", "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"], "pilot_sha256": row["sha256"],
            "source_script_sha256": task["source_script_sha256"],
            "range_addendum_sha256": task["range_addendum_sha256"],
            "horizon_addendum_sha256": task["horizon_addendum_sha256"],
            "error": repr(exc), "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - start, "test_labels_read": False}
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return {"status": document["status"], "task": row["mdp_id"],
            "behavior": row["policy_family"], "seconds": document["runtime_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1-4 workers on this 24 GiB laptop")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("OEH-B preoutcome panel already sealed")
    pilot_manifest = DATA / "manifest.json"
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    horizon_addendum = json.loads(HORIZON_ADDENDUM.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or
            addendum["status"] != "DEVELOPMENT_METHOD_ADDENDUM_FROZEN_NO_TEST_LABELS" or
            len(pilot["pilot_groups"]) != 70 or
            horizon_addendum["status"] != "DEVELOPMENT_SCORER_CORRECTION_FROZEN_NO_TEST_LABELS" or
            horizon_addendum["bridge_posterior_evaluation_horizon"] != 100):
        raise RuntimeError("Preoutcome inputs not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_dir = OUTPUT / "groups"
    group_dir.mkdir(exist_ok=True)
    script_hash = sha256(Path(__file__))
    addendum_hash = sha256(ADDENDUM)
    horizon_addendum_hash = sha256(HORIZON_ADDENDUM)
    tasks = []
    for row in pilot["pilot_groups"]:
        stem = hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
        tasks.append({"row": row, "output": str(group_dir / f"{stem}.json"),
                      "source_script_sha256": script_hash,
                      "range_addendum_sha256": addendum_hash,
                      "horizon_addendum_sha256": horizon_addendum_hash,
                      "posterior_structure": addendum["posterior_structure"]})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j, result in enumerate(pool.map(run_one, tasks), 1):
            results.append(result)
            print(f"OEH-B BRIDGE {j}/70 {result['task']} / {result['behavior']}: "
                  f"{result['status']} {result['seconds']:.1f}s", flush=True)
    failures = [r for r in results if r["status"] != "OEHB_PREOUTCOME_ONLY"]
    if failures:
        (OUTPUT / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_OEHB_PREOUTCOME", "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"OEH-B failed for {len(failures)} pilot groups")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_OEHB_PREOUTCOME_NO_TEST_LABELS",
        "pilot_manifest_sha256": sha256(pilot_manifest),
        "range_addendum_sha256": addendum_hash,
        "horizon_addendum_sha256": horizon_addendum_hash,
        "source_script_sha256": script_hash,
        "groups": len(results), "test_labels_read": False,
        "runtime_seconds": sum(r["seconds"] for r in results),
        "group_sha256": {Path(t["output"]).name: sha256(Path(t["output"])) for t in tasks},
    }, indent=2) + "\n", encoding="utf-8")
    print("Sealed 70 BRIDGE OEH-B preoutcome groups", flush=True)


if __name__ == "__main__":
    main()
