"""Checkpointed OEH-H pilot descriptor on fresh task-specific horizons."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .fresh_adapter import PANEL
from .fresh_prepare_pilots import OUTPUT as DATA

OUTPUT = ROOT / "artifacts/week8/bridge_fresh_horizon_preoutcome_2026-10-08"
BUDGETS = (1, 2, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def one(task: dict) -> dict:
    row, protocol = task["row"], task["protocol"]
    path = Path(task["output"])
    if path.exists():
        item = json.loads(path.read_text(encoding="utf-8"))
        if (item.get("pilot_sha256") != row["sha256"] or
                item.get("source_script_sha256") != task["source_script_sha256"] or
                item.get("protocol_sha256") != task["protocol_sha256"]):
            raise RuntimeError(f"Stale horizon checkpoint: {path.name}")
        return {"status": item["status"], "mdp_id": row["mdp_id"],
                "policy_family": row["policy_family"], "seconds": item.get("runtime_seconds", 0)}
    start = time.perf_counter()
    try:
        pilot_path = DATA / row["file"]
        if sha256(pilot_path) != row["sha256"]:
            raise RuntimeError("Pilot hash changed")
        with np.load(pilot_path, allow_pickle=False) as loaded:
            pilot = {key: loaded[key] for key in loaded.files}
        S, A, horizon = int(row["num_states"]), int(row["num_actions"]), int(row["horizon"])
        if horizon != int(protocol["episode_horizon"][row["mdp_id"]]):
            raise RuntimeError("Task horizon changed")
        rho = np.asarray(row["initial_state_distribution"], dtype=float)
        P, R, beta, terminal = posterior_mean_model(
            pilot, rho, S, A, float(protocol["gamma"]),
            **protocol["posterior"]["structure"])
        estimate = H.offline_effective_horizon(P, R, beta, rho, float(protocol["gamma"]),
                                               terminal=terminal, tau=float(protocol["tau"]),
                                               horizon=horizon, form="risk")
        theory = H.offline_theory_measures(P, R, beta, rho, float(protocol["gamma"]),
                                           terminal=terminal, horizon=horizon)
        observed = np.zeros((S, A), dtype=bool)
        observed[pilot["state"], pilot["action"]] = True
        theory.update(H.asymptotic_ceilings(P, R, beta, rho, float(protocol["gamma"]),
                                            supported=observed, terminal=terminal))
        profile = H.risk_profile(P, R, beta, rho, float(protocol["gamma"]),
                                 terminal=terminal, tau=float(protocol["tau"]),
                                 horizon=horizon, budgets=np.asarray(BUDGETS) * S * A)
        theory.update({f"risk_x{budget}": float(value) for budget, value in
                       zip(BUDGETS, profile["noise_regret"])})
        theory.update({f"miss_x{budget}": float(value) for budget, value in
                       zip(BUDGETS, profile["missing_mass"])})
        theory["risk_slack"] = profile["slack"]
        values = {"mdp_id": row["mdp_id"], "policy_family": row["policy_family"],
                  "public_horizon": horizon,
                  **{f"data_{key}": clean(value) for key, value in estimate.as_dict().items()},
                  **{f"data_{key}": clean(value) for key, value in theory.items()}}
        item = {"status": "FRESH_HORIZON_PREOUTCOME_ONLY", "pilot_sha256": row["sha256"],
                "protocol_sha256": task["protocol_sha256"],
                "source_script_sha256": task["source_script_sha256"],
                "row": values, "runtime_seconds": time.perf_counter() - start,
                "test_labels_read": False}
    except Exception as exc:
        item = {"status": "FRESH_HORIZON_PREOUTCOME_FAILURE", "pilot_sha256": row["sha256"],
                "protocol_sha256": task["protocol_sha256"],
                "source_script_sha256": task["source_script_sha256"],
                "error": repr(exc), "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - start, "test_labels_read": False}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {"status": item["status"], "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"], "seconds": item["runtime_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("Use one or two workers while OEH-B uses the GPU")
    pilot_file, protocol_file = DATA / "manifest.json", PANEL / "protocol.json"
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["protocol_sha256"] != sha256(protocol_file) or
            len(pilot["pilot_groups"]) != 25):
        raise RuntimeError("Fresh horizon preoutcome gate failed")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Fresh horizon already sealed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = OUTPUT / "groups"
    groups.mkdir(exist_ok=True)
    tasks = []
    for row in pilot["pilot_groups"]:
        stem = hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
        tasks.append({"row": row, "protocol": protocol, "output": str(groups / f"{stem}.json"),
                      "protocol_sha256": sha256(protocol_file),
                      "source_script_sha256": sha256(Path(__file__))})
    tasks.sort(key=lambda task: int(task["row"]["num_states"]), reverse=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Fresh OEH-H {len(results)}/25 {result['mdp_id']} / {result['policy_family']}: "
                  f"{result['status']} {result['seconds']:.1f}s", flush=True)
    failures = [row for row in results if row["status"] != "FRESH_HORIZON_PREOUTCOME_ONLY"]
    if failures:
        (OUTPUT / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_FRESH_HORIZON", "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Fresh horizon failed for {len(failures)} groups")
    rows = [json.loads(Path(task["output"]).read_text(encoding="utf-8"))["row"] for task in tasks]
    frame = pd.DataFrame(rows).sort_values(["mdp_id", "policy_family"])
    if len(frame) != 25 or frame[["mdp_id", "policy_family"]].duplicated().any():
        raise RuntimeError("Incomplete fresh horizon rows")
    frame.to_csv(OUTPUT / "horizon_preoutcome.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_HORIZON_PREOUTCOME_NO_TEST_LABELS",
        "pilot_manifest_sha256": sha256(pilot_file), "protocol_sha256": sha256(protocol_file),
        "source_script_sha256": sha256(Path(__file__)), "groups": 25,
        "horizon_csv_sha256": sha256(OUTPUT / "horizon_preoutcome.csv"),
        "runtime_seconds_sum": sum(row["seconds"] for row in results), "test_labels_read": False},
        indent=2) + "\n", encoding="utf-8")
    print("Sealed 25/25 fresh OEH-H pilot descriptors", flush=True)


if __name__ == "__main__":
    main()
