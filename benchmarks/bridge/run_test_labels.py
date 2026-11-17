"""Open BRIDGE test labels once, after the selected comparison is sealed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import policy_sha256

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
FREEZE = ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json"
B2 = ROOT / "artifacts/week8/bridge_b2_forecast_preoutcome_2026-10-07"
FULL = ROOT / "artifacts/week8/bridge_full_forecast_preoutcome_2026-10-07"
REDUCED = ROOT / "artifacts/week8/bridge_reduced_forecast_preoutcome_2026-10-07"
RESOURCE = ROOT / "artifacts/week8/bridge_resource_gate_addendum_2026-10-07.json"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
LABEL_TO_REGISTERED = {
    "BC": "behavior_cloning_tabular",
    "CFQI": "cfqi_tabular",
    "IQL": "iql_tabular",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def checked_predictions(directory: Path, status: str) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "predictions_before_label_read.csv"
    if (manifest["status"] != status or manifest["test_labels_read"] or
            manifest["rows"] != 210 or manifest["task_count"] != 14 or
            sha256(path) != manifest["output_sha256"]["predictions"]):
        raise RuntimeError(f"Prediction seal failed: {directory}")
    frame = pd.read_csv(path)
    if (len(frame) != 210 or frame[KEYS].duplicated().any() or
            not np.isfinite(frame[P].to_numpy(float)).all() or
            not np.allclose(frame[P].sum(axis=1), 1)):
        raise RuntimeError(f"Invalid predicted laws: {directory}")
    return frame, manifest


def prelabel_gate(mode: str) -> tuple[list[dict], list[str], dict]:
    pilot = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["test_labels_read"] or len(pilot["pilot_groups"]) != 70 or
            frozen["status"] != "PREFLIGHT_SPLIT_FROZEN_NOT_BENCHMARK_GO"):
        raise RuntimeError("Pilot/split prelabel gate failed")
    b2, _ = checked_predictions(B2, "SEALED_B2_FORECAST_NO_TEST_LABELS")
    comparison_dir = FULL if mode == "full" else REDUCED
    comparison_status = ("SEALED_FULL_108_FORECAST_NO_TEST_LABELS" if mode == "full"
                         else "SEALED_REDUCED_OEH_ABLATION_NO_TEST_LABELS")
    comparison_method = "oeh_full_108" if mode == "full" else "oeh_without_direct_oehb"
    if mode == "reduced":
        resource = json.loads(RESOURCE.read_text(encoding="utf-8"))
        if (resource["status"] != "PRELABEL_RESOURCE_NO_GO_FULL_14_TASK_OEHB" or
                resource["test_labels_read"]):
            raise RuntimeError("Reduced comparison was not frozen before test labels")
    full, _ = checked_predictions(comparison_dir, comparison_status)
    expected = {(r["mdp_id"], r["policy_family"], learner)
                for r in pilot["pilot_groups"] for learner in LABEL_TO_REGISTERED.values()}
    if (set(b2[KEYS].itertuples(index=False, name=None)) != expected or
            set(full[KEYS].itertuples(index=False, name=None)) != expected or
            b2.method.unique().tolist() != ["b2_matched"] or
            full.method.unique().tolist() != [comparison_method]):
        raise RuntimeError("Prediction groups differ from pilot groups")
    tests = frozen["candidate_test_tasks"]
    if len(tests) != 14 or set(tests) != {r["mdp_id"] for r in pilot["pilot_groups"]}:
        raise RuntimeError("Test task split changed")
    seal = {"status": "FINAL_BRIDGE_TEST_PROTOCOL_SEALED_BEFORE_LABELS",
            "test_labels_read": False, "task_ids": tests, "pilot_groups": 70,
            "learners": list(LABEL_TO_REGISTERED.values()),
            "label_name_map": LABEL_TO_REGISTERED,
            "forecasters": ["b2_matched", comparison_method],
            "comparison_mode": mode,
            "tau": .95, "horizon": 100, "gamma": .99,
            "budgets": [1, 2, 5, 10, 20, 50], "training_seeds": 10,
            "sustained_minimum": 8,
            "source_sha256": {"pilot_manifest": sha256(PILOT / "manifest.json"),
                              "split_freeze": sha256(FREEZE),
                              "b2_predictions": sha256(B2 / "predictions_before_label_read.csv"),
                              "comparison_predictions": sha256(comparison_dir / "predictions_before_label_read.csv"),
                              **({"resource_addendum": sha256(RESOURCE)} if mode == "reduced" else {}),
                              "label_code": sha256(ROOT / "benchmarks/bridge/labels.py"),
                              "runner": sha256(Path(__file__))}}
    return pilot["pilot_groups"], tests, seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "reduced"), default="full")
    args = parser.parse_args()
    output = ROOT / "artifacts/week8" / f"bridge_{args.mode}_test_labels_2026-10-07"
    if (output / "manifest.json").exists():
        raise FileExistsError("Complete label artifact already exists")
    groups, tests, seal = prelabel_gate(args.mode)
    output.mkdir(parents=True, exist_ok=True)
    seal_path = output / "prelabel_protocol_seal.json"
    if seal_path.exists():
        if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
            raise RuntimeError("Prelabel seal changed after a partial run")
    else:
        seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    from benchmarks.bridge.labels import one_group  # Import only after the seal.

    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    candidates = [r for r in registry if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    ordinal = {r["mdp_name"]: i for i, r in enumerate(candidates)}
    lookup = {(r["mdp_id"], r["policy_family"]): r for r in groups}
    if len(lookup) != 70:
        raise RuntimeError("Duplicate pilot group")
    group_dir = output / "groups"
    group_dir.mkdir(exist_ok=True)
    failures = []
    for task in tests:
        mdp = load_sticky_mdp(task)
        policies = behavior_policy_suite(mdp, seed=seed("bridge-behavior-v1", task),
                                         variant_index=ordinal[task])
        for policy in policies:
            source = lookup[(task, policy.family)]
            if policy_sha256(policy) != source["behavior_policy_sha256"]:
                raise RuntimeError(f"Behavior changed: {task}/{policy.family}")
            stem = hashlib.sha256(f"{task}|{policy.family}".encode()).hexdigest()[:24]
            path = group_dir / f"{stem}.json"
            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
                if (item.get("pilot_sha256") != source["sha256"] or
                        item.get("prelabel_protocol_sha256") != sha256(seal_path)):
                    raise RuntimeError(f"Stale label checkpoint: {path.name}")
            else:
                start = time.perf_counter()
                try:
                    result = one_group(mdp, policy)
                    item = {"status": "BRIDGE_TEST_LABEL_COMPLETE", **result,
                            "pilot_sha256": source["sha256"],
                            "prelabel_protocol_sha256": sha256(seal_path)}
                except Exception as exc:
                    item = {"status": "BRIDGE_TEST_LABEL_FAILURE", "task": task,
                            "behavior": policy.family, "pilot_sha256": source["sha256"],
                            "prelabel_protocol_sha256": sha256(seal_path),
                            "runtime_seconds": time.perf_counter() - start,
                            "error": repr(exc), "traceback": traceback.format_exc()}
                path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
            if item["status"] != "BRIDGE_TEST_LABEL_COMPLETE":
                failures.append({"task": task, "behavior": policy.family,
                                 "error": item.get("error")})
            print(f"BRIDGE test label {task}/{policy.family}: {item['status']}", flush=True)
    if failures:
        (output / "failure_report.json").write_text(json.dumps({
            "status": "NO_GO_INCOMPLETE_BRIDGE_TEST_LABELS", "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Incomplete BRIDGE labels: {len(failures)} failures")
    rows = []
    for path in sorted(group_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        for learner, event in item["event"].items():
            rows.append({"mdp_id": item["task"], "policy_family": item["behavior"],
                         "algorithm": LABEL_TO_REGISTERED[learner], "event": event,
                         "censored": int(event == 6),
                         "label_runtime_seconds": item["elapsed_seconds"]})
    frame = pd.DataFrame(rows).sort_values(KEYS)
    if len(frame) != 210 or frame[KEYS].duplicated().any():
        raise RuntimeError("Incomplete seven-class BRIDGE label table")
    frame.to_csv(output / "labels.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": f"BRIDGE_{args.mode.upper()}_TEST_LABELS_COMPLETE",
        "test_labels_read": True, "groups": 70, "rows": 210,
        "prelabel_protocol_sha256": sha256(seal_path),
        "labels_sha256": sha256(output / "labels.csv"),
        "group_sha256": {path.name: sha256(path) for path in sorted(group_dir.glob("*.json"))},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Complete 210 BRIDGE {args.mode} test labels after both predictions sealed", flush=True)


if __name__ == "__main__":
    main()
