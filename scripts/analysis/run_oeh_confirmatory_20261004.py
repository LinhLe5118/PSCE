"""Seventh-cohort one-shot confirmation: expanded OEH and fixed-CFQI router.

All model choices come from opened cohorts. ``freeze`` must precede ``prepare``;
no learner outcomes of this cohort are inspected until preoutcome_seal exists.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis import oeh_confirmatory_helpers as H
from scripts.analysis.develop_fixed_selector_20261004 import design, model
from scripts.analysis.run_unseen_validation import load_json, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
DATASET = ROOT / "datasets/oeh_confirmatory_unseen_120_2026-10-04"
NAMESPACE = "oeh-confirmatory-seventh-new-120-v1-20261004"
FOURTH = H.FOURTH
ORIGINAL_META = R.meta_training
ORIGINAL_SEEN = R.seen_kernels


def expanded_meta(cohorts, tag):
    earlier = ORIGINAL_META({k: v for k, v in cohorts.items() if k != "fourth120"}, tag)
    fourth = H.fourth_frame()
    # The frozen fourth and seventh recipes reuse the same human-readable
    # instance IDs, but their audited kernels are distinct. Namespace the
    # opened rows so group CV and overlap checks use kernel identity correctly.
    fourth["mdp_id"] = "opened_fourth120/" + fourth["mdp_id"].astype(str)
    if set(earlier.mdp_id) & set(fourth.mdp_id):
        raise RuntimeError("Opened meta-training MDP overlap")
    return pd.concat([earlier, fourth], ignore_index=True)


def seen_kernels():
    previous = ORIGINAL_SEEN()
    fourth = load_json(FOURTH / "actual_kernel_audit_preoutcome.json")
    return previous | set(fourth["actual_kernel_fingerprints"].values())


def fixed_selector(doc, manifest, workers):
    """Train only on opened fourth outcomes; write choices before new outcomes."""
    if not (R.OUTPUT / "predictions_preoutcome.csv").exists():
        raise RuntimeError("Forecasts must precede learner choices")
    train_key, x_train_all = design(FOURTH)
    outcomes = pd.read_csv(FOURTH / "selection_group_outcomes.csv")
    keys = outcomes[["mdp_id", "policy_family"]]
    positions = pd.MultiIndex.from_frame(train_key).get_indexer(pd.MultiIndex.from_frame(keys))
    if (positions < 0).any():
        raise RuntimeError("Opened selection features missing")
    x_train = x_train_all.iloc[positions].reset_index(drop=True)
    test_key, x_test_all = design(R.OUTPUT)
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    mask = test_key.mdp_id.map(family) != "lock_structured"
    test_key = test_key.loc[mask].reset_index(drop=True)
    x_test = x_test_all.loc[mask].reset_index(drop=True)
    if len(test_key) != 420 or test_key.duplicated().any():
        raise RuntimeError("Selection panel is incomplete or contains duplicate groups")
    old_kernels = set(load_json(FOURTH / "actual_kernel_audit_preoutcome.json")
                      ["actual_kernel_fingerprints"].values())
    new_kernels = set(load_json(R.OUTPUT / "actual_kernel_audit_preoutcome.json")
                      ["actual_kernel_fingerprints"].values())
    if old_kernels & new_kernels:
        raise RuntimeError("New selection kernels overlap opened training kernels")
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    names = [c[6:-4] for c in outcomes if c.startswith("fixed:") and c.endswith("|suc")]
    success = outcomes[[f"fixed:{name}|suc" for name in names]].to_numpy()
    returns = outcomes[[f"fixed:{name}|ret" for name in names]].to_numpy()
    pred_s = model().fit(x_train, success).predict(x_test)
    pred_r = model().fit(x_train, returns).predict(x_test)
    score = pred_s + .15 * pred_r
    cfqi = names.index("cfqi_tabular")
    choice = score.argmax(1)
    choice = np.where(score[np.arange(len(choice)), choice] - score[:, cfqi] > .1, choice, cfqi)
    target = R.OUTPUT / "selection_groups"
    target.mkdir(parents=True, exist_ok=True)
    for row, selected in zip(test_key.itertuples(index=False), choice):
        name = hashlib.sha256(f"{row.mdp_id}|{row.policy_family}".encode()).hexdigest()[:24]
        write_json(target / f"{name}.json", {"mdp_id": row.mdp_id,
                   "policy_family": row.policy_family, "selected": names[int(selected)],
                   "source": "fixed_selector_v1", "opened_training_groups": len(keys)})
    print(f"Sealed selector choices for {len(test_key)} new groups", flush=True)


def configure():
    R.OUTPUT = OUTPUT
    R.DATASET = DATASET
    R.NAMESPACE = NAMESPACE
    R.base.OUTPUT = OUTPUT
    R.base.DATASET = DATASET
    R.base.CANDIDATE = R.CANDIDATE
    R.base.COMPARATORS = R.COMPARATORS
    R.base.FORECAST_METHODS = (*R.COMPARATORS, R.CANDIDATE, *R.REPORTED)
    R.OPENED["fourth120"] = dict(
        labels=FOURTH / "labels.csv", features=FOURTH / "pilot_features.csv",
        oracle=FOURTH / "oracle_descriptors_preoutcome.csv",
        dataset=H.FOURTH_DATASET, mdps=120)
    R.meta_training = expanded_meta
    R.seen_kernels = seen_kernels
    R.run_selection = fixed_selector


def freeze():
    R.freeze(argparse.Namespace())
    path = OUTPUT / "protocol.json"
    doc = load_json(path)
    doc["confirmatory_schema"] = "oeh-seventh-one-shot-v1"
    doc["scope"] = "120 unique untouched kernels; seven opened cohorts for fitting; independent published data are a separate transfer stress test"
    doc["decision_rule"] = ("84 stochastic MDP-cluster units; OEH vs tier-1 shared-information baselines and vs original TQ/SACo reported separately; "
                            "paired integrated Brier and seven-class log-loss gains; simultaneous one-sided 95% lower bounds must be strictly positive; no minimum point gain")
    doc["selection_study"]["rule"] = "ExtraTrees 150 trees, leaf 7, max_features 0.8; predict success + 0.15 normalized return; switch from fixed CFQI only when advantage >0.1"
    doc["selection_study"]["comparators"] = ["best fixed learner on the new panel, selected by predefined tie order", "fixed CFQI tabular"]
    doc["selection_study"]["decision_rule"] = "simultaneous one-sided 95% MDP-cluster lower bounds >0 for success and return against fixed CFQI; best fixed learner reported separately"
    doc["oeh_final"]["meta_training_cohorts"] = list(R.OPENED)
    doc["external_dataset"] = {"name": "D4RL/minigrid/fourrooms-random-v0", "publisher": "Farama Minari",
                               "status": "published offline trajectories; independent transfer stress test; partial observations, outside the tabular MDP claim",
                               "source": "https://minari.farama.org/datasets/D4RL/minigrid/fourrooms-random-v0/"}
    doc["source_sha256"][str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = R.sha256(Path(__file__))
    doc["source_sha256"]["scripts/analysis/develop_fixed_selector_20261004.py"] = R.sha256(ROOT / "scripts/analysis/develop_fixed_selector_20261004.py")
    doc["source_sha256"]["scripts/analysis/oeh_confirmatory_helpers.py"] = R.sha256(ROOT / "scripts/analysis/oeh_confirmatory_helpers.py")
    write_json(path, doc)
    (OUTPUT / "protocol.sha256").write_text(R.sha256(path) + "\n", encoding="ascii")
    print(f"Frozen seventh protocol {R.sha256(path)}", flush=True)


def repair_schema_preoutcome():
    """One-time schema correction before generation or outcomes."""
    if DATASET.exists() or (OUTPUT / "preoutcome_seal.json").exists():
        raise RuntimeError("Schema amendment forbidden after dataset generation")
    path = OUTPUT / "protocol.json"
    doc = load_json(path)
    if doc["schema"] != "oeh-seventh-one-shot-v1":
        raise RuntimeError("Unexpected schema; no amendment made")
    doc["schema"] = "oeh-unseen-120-difficulty-v1"
    doc["confirmatory_schema"] = "oeh-seventh-one-shot-v1"
    doc["preoutcome_schema_correction"] = {"reason": "base verifier requires its original schema token",
                                            "dataset_buffers_created": 0, "learner_outcomes_observed": 0}
    doc["source_sha256"]["scripts/analysis/run_oeh_confirmatory_20261004.py"] = R.sha256(Path(__file__))
    write_json(path, doc)
    (OUTPUT / "protocol.sha256").write_text(R.sha256(path) + "\n", encoding="ascii")
    print(f"Corrected pre-dataset schema {R.sha256(path)}", flush=True)


def repair_meta_ids_preoutcome():
    """Record an ID-only meta-training fix before forecast seal or learner jobs."""
    if ((OUTPUT / "preoutcome_seal.json").exists()
            or (OUTPUT / "execution_plan.json").exists()
            or (OUTPUT / "external_execution_plan.json").exists()):
        raise RuntimeError("Meta ID amendment forbidden after forecasts or outcomes")
    path = OUTPUT / "protocol.json"
    doc = load_json(path)
    if doc.get("preoutcome_meta_id_correction"):
        raise RuntimeError("Meta ID amendment already recorded")
    previous = OUTPUT / "protocol_before_meta_id_amendment.json"
    previous.write_bytes(path.read_bytes())
    doc["preoutcome_meta_id_correction"] = {
        "reason": "Distinct fourth and seventh kernels shared human-readable instance IDs",
        "opened_fourth_id_prefix": "opened_fourth120/",
        "new_cohort_ids_unchanged": True,
        "learner_outcomes_observed": 0,
        "previous_protocol_sha256": R.sha256(previous),
    }
    doc["source_sha256"]["scripts/analysis/run_oeh_confirmatory_20261004.py"] = R.sha256(Path(__file__))
    write_json(path, doc)
    (OUTPUT / "protocol.sha256").write_text(R.sha256(path) + "\n", encoding="ascii")
    dataset_seal = OUTPUT / "dataset_seal.json"
    seal = load_json(dataset_seal)
    seal["protocol_sha256"] = R.sha256(path)
    seal["preoutcome_meta_id_correction"] = True
    write_json(dataset_seal, seal)
    print(f"Corrected opened-cohort IDs before outcomes {R.sha256(path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "repair-schema", "repair-meta-ids", "prepare", "evaluate-core", "evaluate-external", "complete-core", "analyze"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--selection-workers", type=int, default=4)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    configure()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "repair-schema":
        repair_schema_preoutcome()
    elif args.phase == "repair-meta-ids":
        repair_meta_ids_preoutcome()
    elif args.phase == "prepare":
        R.prepare(args)
    elif args.phase == "evaluate-core":
        R.evaluate_core_shard(args)
    elif args.phase == "evaluate-external":
        R.evaluate_external_shard(args)
    elif args.phase == "complete-core":
        R.base.evaluate(resume=True)
    else:
        R.base.analyze()


if __name__ == "__main__":
    main()
