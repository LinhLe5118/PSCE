# Generated from the sealed fresh pipeline by materialize_expanded_pipeline.py.
"""Run exactly one fresh BRIDGE learner-label pass after the prelabel seal."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import pandas as pd

from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import policy_sha256
from .expanded_adapter import PANEL, registry
from .fresh_labels import LEARNERS, one_group
from .expanded_prepare_pilots import OUTPUT as PILOT, seed
from .expanded_preflight import load_eligible_mdp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/week8/bridge_expanded_test_labels_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_task(task: dict) -> dict:
    name, rows, ordinal = task["name"], task["pilot_rows"], task["ordinal"]
    mdp = load_eligible_mdp(name)
    policies = behavior_policy_suite(mdp, seed=seed("bridge-fresh-behavior-v1", name),
                                     variant_index=ordinal)
    policy_map = {policy.family: policy for policy in policies}
    if len(policies) != 5 or set(policy_map) != {row["policy_family"] for row in rows}:
        raise RuntimeError(f"Behavior suite changed: {name}")
    checkpoints = []
    for row in rows:
        policy = policy_map[row["policy_family"]]
        if policy_sha256(policy) != row["behavior_policy_sha256"]:
            raise RuntimeError(f"Behavior policy changed: {name}/{policy.family}")
        stem = hashlib.sha256(f"{name}|{policy.family}".encode()).hexdigest()[:24]
        path = OUT / "groups" / f"{stem}.json"
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if (document["status"] != "FRESH_LABEL_GROUP_COMPLETE" or
                    document["pilot_sha256"] != row["sha256"] or
                    document["prelabel_seal_sha256"] != task["seal_sha256"] or
                    document["label_kernel_sha256"] != task["kernel_sha256"]):
                raise RuntimeError(f"Stale label checkpoint: {path.name}")
        else:
            result = one_group(mdp, policy, horizon=int(row["horizon"]))
            document = {"status": "FRESH_LABEL_GROUP_COMPLETE",
                        "pilot_sha256": row["sha256"],
                        "prelabel_seal_sha256": task["seal_sha256"],
                        "label_kernel_sha256": task["kernel_sha256"],
                        "result": result}
            temporary = path.with_suffix(".json.part")
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        checkpoints.append(path)
        print(f"Fresh labels {name} / {policy.family}: COMPLETE", flush=True)
    return {"task": name, "checkpoints": [str(p) for p in checkpoints]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use one to three CPU workers")
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Fresh labels already sealed")
    seal_file = PANEL / "prelabel_seal.json"
    seal = json.loads(seal_file.read_text(encoding="utf-8"))
    if seal["status"] != "FRESH_PRELABEL_SEAL_READY" or seal["test_labels_read"]:
        raise RuntimeError("Fresh prelabel seal missing")
    required = {"label_kernel": ROOT / "benchmarks/bridge/fresh_labels.py",
                "label_runner": Path(__file__),
                "protocol": PANEL / "protocol.json",
                "reward_interpretation_addendum": PANEL / "reward_interpretation_addendum.json",
                "published_survival_comparator_addendum": PANEL / "published_survival_comparator_addendum.json",
                "implementation_addendum": PANEL / "prelabel_implementation_addendum.json",
                "posterior_memory_addendum": PANEL / "prelabel_posterior_memory_addendum.json",
                "oehb_manifest": ROOT / "artifacts/week8/bridge_expanded_full_oehb_preoutcome_2026-10-08/manifest.json",
                "pilot_manifest": PILOT / "manifest.json",
                "oeh_full_manifest": ROOT / "artifacts/week8/bridge_expanded_full_forecast_preoutcome_2026-10-08/manifest.json",
                "oeh_full_predictions": ROOT / "artifacts/week8/bridge_expanded_full_forecast_preoutcome_2026-10-08/predictions_before_label_read.csv",
                "b2_internal_manifest": ROOT / "artifacts/week8/bridge_expanded_b2_forecast_preoutcome_2026-10-08/manifest.json",
                "b2_internal_predictions": ROOT / "artifacts/week8/bridge_expanded_b2_forecast_preoutcome_2026-10-08/predictions_before_label_read.csv",
                "published_cf_manifest": ROOT / "artifacts/week8/bridge_expanded_published_xgboost_cf_preoutcome_2026-10-08/manifest.json",
                "published_cf_predictions": ROOT / "artifacts/week8/bridge_expanded_published_xgboost_cf_preoutcome_2026-10-08/predictions_before_label_read.csv",
                "published_cf_full108_manifest": ROOT / "artifacts/week8/bridge_expanded_published_xgboost_cf_full108_preoutcome_2026-10-08/manifest.json",
                "published_cf_full108_predictions": ROOT / "artifacts/week8/bridge_expanded_published_xgboost_cf_full108_preoutcome_2026-10-08/predictions_before_label_read.csv",
                "original_manifest": ROOT / "artifacts/week8/bridge_expanded_original_scores_preoutcome_2026-10-08/manifest.json",
                "original_scores": ROOT / "artifacts/week8/bridge_expanded_original_scores_preoutcome_2026-10-08/scores_before_label_read.csv"}
    for key, path in required.items():
        if seal["sha256"][key] != sha256(path):
            raise RuntimeError(f"Prelabel sealed input changed: {key}")
    protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    pilot = json.loads(required["pilot_manifest"].read_text(encoding="utf-8"))
    if (len(pilot["pilot_groups"]) != 25 or
            pilot["tasks"] != protocol["evaluation_tasks"] or
            seal["tasks"] != protocol["evaluation_tasks"]):
        raise RuntimeError("Frozen task grid changed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "groups").mkdir(exist_ok=True)
    ordinal = {name: i for i, name in enumerate(registry())}
    tasks = [{"name": name, "ordinal": ordinal[name],
              "pilot_rows": [row for row in pilot["pilot_groups"] if row["mdp_id"] == name],
              "seal_sha256": sha256(seal_file), "kernel_sha256": sha256(required["label_kernel"])}
             for name in protocol["evaluation_tasks"]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_task, task): task["name"] for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(f"Fresh labels task {len(results)}/5 {result['task']}: 5/5 groups", flush=True)
    files = sorted((OUT / "groups").glob("*.json"))
    if len(files) != 25:
        raise RuntimeError("Incomplete fresh label checkpoint grid")
    rows = []
    for file in files:
        document = json.loads(file.read_text(encoding="utf-8"))
        if (document["status"] != "FRESH_LABEL_GROUP_COMPLETE" or
                document["prelabel_seal_sha256"] != sha256(seal_file)):
            raise RuntimeError(f"Bad label checkpoint: {file.name}")
        result = document["result"]
        task_name = result["task"].removesuffix("_sticky25")
        for learner in LEARNERS:
            rows.append({"mdp_id": task_name, "label_environment_id": result["task"],
                         "policy_family": result["behavior"], "algorithm": {
                             "BC": "behavior_cloning_tabular", "CFQI": "cfqi_tabular",
                             "IQL": "iql_tabular"}[learner],
                         "event": result["event"][learner],
                         "success_counts": json.dumps(result["success_counts"][learner]),
                         "horizon": result["horizon"]})
    frame = pd.DataFrame(rows).sort_values(["mdp_id", "policy_family", "algorithm"])
    if len(frame) != 75 or frame[["mdp_id", "policy_family", "algorithm"]].duplicated().any():
        raise RuntimeError("Incomplete fresh label rows")
    frame.to_csv(OUT / "labels.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "FRESH_BRIDGE_TEST_LABELS_COMPLETE", "rows": 75,
        "tasks": 5, "groups": 25,
        "prelabel_seal_sha256": sha256(seal_file),
        "label_kernel_sha256": sha256(required["label_kernel"]),
        "label_runner_sha256": sha256(Path(__file__)),
        "group_sha256": {file.name: sha256(file) for file in files},
        "labels_sha256": sha256(OUT / "labels.csv")},
        indent=2) + "\n", encoding="utf-8")
    print("Sealed 75/75 fresh BRIDGE learner labels", flush=True)


if __name__ == "__main__":
    main()
