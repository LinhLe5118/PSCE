"""Checkpointed OEH-H/risk profile from BRIDGE pilots, without test labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh import horizon as H  # noqa: E402
from scripts.analysis.develop_oeh_horizon import posterior_mean_model  # noqa: E402

DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
ADDENDUM = ROOT / "artifacts/week8/bridge_transfer_range_addendum_2026-10-07.json"
OUTPUT = ROOT / "artifacts/week8/bridge_horizon_preoutcome_2026-10-07"
BUDGETS = (1, 2, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def one(task: dict) -> dict:
    row = task["row"]
    path = Path(task["output"])
    if path.exists():
        item = json.loads(path.read_text(encoding="utf-8"))
        if (item.get("pilot_sha256") != row["sha256"] or
                item.get("source_script_sha256") != task["source_script_sha256"]):
            raise RuntimeError(f"Stale horizon checkpoint: {path.name}")
        return {"status": item["status"], "task": row["mdp_id"],
                "behavior": row["policy_family"], "seconds": item.get("runtime_seconds", 0)}
    start = time.perf_counter()
    try:
        pilot_path = DATA / row["file"]
        if sha256(pilot_path) != row["sha256"]:
            raise RuntimeError("Pilot hash changed")
        with np.load(pilot_path, allow_pickle=False) as loaded:
            pilot = {key: loaded[key] for key in loaded.files}
        S, A = int(row["num_states"]), int(row["num_actions"])
        rho = np.asarray(row["initial_state_distribution"], dtype=float)
        P, R, beta, terminal = posterior_mean_model(
            pilot, rho, S, A, .99, **task["posterior_structure"])
        estimate = H.offline_effective_horizon(P, R, beta, rho, .99,
                                               terminal=terminal, tau=.95,
                                               horizon=100, form="risk")
        theory = H.offline_theory_measures(P, R, beta, rho, .99,
                                           terminal=terminal, horizon=100)
        observed = np.zeros((S, A), dtype=bool)
        observed[pilot["state"], pilot["action"]] = True
        theory.update(H.asymptotic_ceilings(P, R, beta, rho, .99,
                                            supported=observed, terminal=terminal))
        profile = H.risk_profile(P, R, beta, rho, .99,
                                 terminal=terminal, tau=.95, horizon=100,
                                 budgets=np.asarray(BUDGETS) * S * A)
        theory.update({f"risk_x{b}": float(value) for b, value in zip(BUDGETS, profile["noise_regret"])})
        theory.update({f"miss_x{b}": float(value) for b, value in zip(BUDGETS, profile["missing_mass"])})
        theory["risk_slack"] = profile["slack"]
        values = {"mdp_id": row["mdp_id"], "policy_family": row["policy_family"],
                  **{f"data_{key}": clean(value) for key, value in estimate.as_dict().items()},
                  **{f"data_{key}": clean(value) for key, value in theory.items()}}
        item = {"status": "HORIZON_PREOUTCOME_ONLY", "pilot_sha256": row["sha256"],
                "source_script_sha256": task["source_script_sha256"],
                "row": values, "runtime_seconds": time.perf_counter() - start,
                "test_labels_read": False}
    except Exception as exc:
        item = {"status": "HORIZON_PREOUTCOME_FAILURE", "pilot_sha256": row["sha256"],
                "source_script_sha256": task["source_script_sha256"],
                "error": repr(exc), "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - start,
                "test_labels_read": False}
    path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    return {"status": item["status"], "task": row["mdp_id"],
            "behavior": row["policy_family"], "seconds": item["runtime_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1-4 workers on this laptop")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Horizon preoutcome panel already sealed")
    pilot_path = DATA / "manifest.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or
            addendum["status"] != "DEVELOPMENT_METHOD_ADDENDUM_FROZEN_NO_TEST_LABELS" or
            len(pilot["pilot_groups"]) != 70):
        raise RuntimeError("Preoutcome inputs not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_dir = OUTPUT / "groups"
    group_dir.mkdir(exist_ok=True)
    script_hash = sha256(Path(__file__))
    tasks = []
    for row in pilot["pilot_groups"]:
        stem = hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
        tasks.append({"row": row, "output": str(group_dir / f"{stem}.json"),
                      "source_script_sha256": script_hash,
                      "posterior_structure": addendum["posterior_structure"]})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j, result in enumerate(pool.map(one, tasks), 1):
            results.append(result)
            print(f"OEH-H BRIDGE {j}/70 {result['task']} / {result['behavior']}: "
                  f"{result['status']} {result['seconds']:.1f}s", flush=True)
    failures = [r for r in results if r["status"] != "HORIZON_PREOUTCOME_ONLY"]
    if failures:
        (OUTPUT / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_HORIZON_PREOUTCOME", "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"OEH-H failed for {len(failures)} pilot groups")
    rows = [json.loads(Path(t["output"]).read_text(encoding="utf-8"))["row"] for t in tasks]
    frame = pd.DataFrame(rows).sort_values(["mdp_id", "policy_family"])
    if frame[["mdp_id", "policy_family"]].duplicated().any() or len(frame) != 70:
        raise RuntimeError("Duplicate/missing horizon row")
    frame.to_csv(OUTPUT / "horizon_preoutcome.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_HORIZON_PREOUTCOME_NO_TEST_LABELS",
        "pilot_manifest_sha256": sha256(pilot_path),
        "range_addendum_sha256": sha256(ADDENDUM),
        "source_script_sha256": script_hash, "groups": len(rows),
        "horizon_csv_sha256": sha256(OUTPUT / "horizon_preoutcome.csv"),
        "runtime_seconds": sum(r["seconds"] for r in results),
        "test_labels_read": False}, indent=2) + "\n", encoding="utf-8")
    print("Sealed 70 BRIDGE OEH-H preoutcome groups", flush=True)


if __name__ == "__main__":
    main()
