"""Pilot-only B2/OEH3/LCOEH4 features on frozen BRIDGE candidate tasks.

No learner training stream or test requirement label is opened in this phase.
Each group is checkpointed so a long run can resume without changing inputs.
"""
from __future__ import annotations

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
from lcoeh_v4 import LC_OEH_V4_FEATURES, estimate_lcoeh_v4  # noqa: E402
from oeh_risk_v3 import estimate_oeh_risk_v3  # noqa: E402
from stochastic_features import aggregate_oeh_risk_v3_profile, extract_stochastic_features  # noqa: E402

DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
OUTPUT = ROOT / "artifacts/week8/bridge_pilot_features_preoutcome_2026-10-07"
ALGORITHMS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate(record: dict, names: list[str]) -> dict:
    path = DATA / record["file"]
    if sha256(path) != record["sha256"]:
        raise RuntimeError("Pilot NPZ hash changed")
    with np.load(path, allow_pickle=False) as loaded:
        pilot = {key: loaded[key] for key in loaded.files}
    states, actions = int(record["num_states"]), int(record["num_actions"])
    if len(pilot["state"]) != 10 * states * actions:
        raise RuntimeError("Pilot x10 size mismatch")
    prefixes = {b: {key: value[:b * states * actions] for key, value in pilot.items()}
                for b in (2, 5, 10)}
    rho = record["initial_state_distribution"]
    group_key = f"{record['mdp_id']}|{record['policy_family']}"
    start = time.perf_counter()
    base = extract_stochastic_features(pilot, num_states=states, num_actions=actions,
                                       gamma=.99, bootstrap_replicates=200, seed_key=group_key)
    base_seconds = time.perf_counter() - start
    risk = estimate_oeh_risk_v3(pilot, rho, num_states=states, num_actions=actions,
                                gamma=.99, k_max=100)
    by_depth = {int(item["depth"]): item for item in risk["records"]}
    profile = [{**item, "risk_at_pilot": by_depth[int(item["depth"])]["risk_at_pilot"],
                "policy_instability": by_depth[int(item["depth"])]["policy_instability"]}
               for item in risk["profile"]]
    oeh = aggregate_oeh_risk_v3_profile(profile)
    risk_seconds = time.perf_counter() - start - base_seconds
    learners = estimate_lcoeh_v4(prefixes, rho, num_states=states, num_actions=actions,
                                 gamma=.99, seed_key=group_key)
    context_seconds = time.perf_counter() - start - base_seconds - risk_seconds
    rows = []
    for algorithm in ALGORITHMS:
        values = {**base, **oeh,
                  **{name: learners[algorithm][name] for name in LC_OEH_V4_FEATURES}}
        missing = set(names) - set(values)
        if missing or any(not math.isfinite(float(values[name])) for name in names):
            raise RuntimeError(f"Missing/nonfinite feature: {algorithm}, {sorted(missing)}")
        rows.append({"mdp_id": record["mdp_id"], "policy_family": record["policy_family"],
                     "algorithm": algorithm, **{name: float(values[name]) for name in names}})
    return {"status": "PILOT_FEATURES_PREOUTCOME_ONLY", "pilot_sha256": record["sha256"],
            "source_npz_sha256": record["source_npz_sha256"],
            "rows": rows, "base_seconds": base_seconds, "legacy_risk_seconds": risk_seconds,
            "learner_context_seconds": context_seconds,
            "total_seconds": time.perf_counter() - start}


def main() -> None:
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Preoutcome feature manifest already exists")
    pilot_manifest = DATA / "manifest.json"
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or len(pilot["pilot_groups"]) != 70:
        raise RuntimeError("Sealed BRIDGE pilots incomplete")
    names = json.loads((ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json")
                       .read_text(encoding="utf-8"))["feature_sets"]["B2_OEH3_LCOEH4"]
    script_hash = sha256(Path(__file__))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_dir = OUTPUT / "groups"
    group_dir.mkdir(exist_ok=True)
    results, failures = [], []
    for j, record in enumerate(pilot["pilot_groups"], 1):
        stem = hashlib.sha256(f"{record['mdp_id']}|{record['policy_family']}".encode()).hexdigest()[:24]
        path = group_dir / f"{stem}.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if (item.get("pilot_sha256") != record["sha256"] or
                    item.get("source_script_sha256") != script_hash):
                raise RuntimeError(f"Stale feature checkpoint: {path.name}")
        else:
            try:
                item = calculate(record, names)
            except Exception as exc:
                item = {"status": "FEATURE_FAILURE_PREOUTCOME", "pilot_sha256": record["sha256"],
                        "source_npz_sha256": record["source_npz_sha256"],
                        "error": repr(exc), "traceback": traceback.format_exc()}
            item["source_script_sha256"] = script_hash
            path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
        if item["status"] == "PILOT_FEATURES_PREOUTCOME_ONLY":
            results.extend(item["rows"])
            print(f"BRIDGE pilot features {j}/70: {record['mdp_id']} / "
                  f"{record['policy_family']} ({item['total_seconds']:.1f}s)", flush=True)
        else:
            failures.append({"task": record["mdp_id"], "behavior": record["policy_family"],
                             "checkpoint": str(path.relative_to(ROOT)), "error": item["error"]})
            print(f"BRIDGE pilot features FAILED {j}/70: {record['mdp_id']} / "
                  f"{record['policy_family']}", flush=True)
    if len(results) != 210 or failures:
        (OUTPUT / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_PREOUTCOME_FEATURES", "successful_rows": len(results),
            "failures": failures}, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Incomplete BRIDGE pilot feature panel: {len(failures)} failures")
    frame = pd.DataFrame(results).sort_values(["mdp_id", "policy_family", "algorithm"])
    if frame[["mdp_id", "policy_family", "algorithm"]].duplicated().any():
        raise RuntimeError("Duplicate pilot feature row")
    frame.to_csv(OUTPUT / "features.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS",
        "pilot_manifest_sha256": sha256(pilot_manifest),
        "source_script_sha256": script_hash, "groups": 70, "rows": len(frame),
        "feature_columns": len(names), "gamma": .99,
        "features_sha256": sha256(OUTPUT / "features.csv"),
        "test_labels_read": False}, indent=2) + "\n", encoding="utf-8")
    print("Sealed 210 BRIDGE pilot-only feature rows", flush=True)


if __name__ == "__main__":
    main()
