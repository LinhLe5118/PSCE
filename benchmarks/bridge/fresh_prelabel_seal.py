"""Audit all fresh BRIDGE predictions and original scores before labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fresh_adapter import PANEL
from .fresh_prepare_pilots import OUTPUT as PILOT

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/week8/bridge_fresh_full_forecast_preoutcome_2026-10-08"
B2 = ROOT / "artifacts/week8/bridge_fresh_b2_forecast_preoutcome_2026-10-08"
PUBLISHED_CF = ROOT / "artifacts/week8/bridge_fresh_published_xgboost_cf_preoutcome_2026-10-08"
PUBLISHED_CF_FULL = ROOT / "artifacts/week8/bridge_fresh_published_xgboost_cf_full108_preoutcome_2026-10-08"
ORIGINAL = ROOT / "artifacts/week8/bridge_fresh_original_scores_preoutcome_2026-10-08"
LABELS = ROOT / "artifacts/week8/bridge_fresh_test_labels_2026-10-08"
OUT = PANEL / "prelabel_seal.json"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{j}" for j in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists() or LABELS.exists():
        raise FileExistsError("Seal/labels already exist")
    protocol_file, pilot_file = PANEL / "protocol.json", PILOT / "manifest.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    reward_addendum_file = PANEL / "reward_interpretation_addendum.json"
    reward_addendum = json.loads(reward_addendum_file.read_text(encoding="utf-8"))
    published_addendum_file = PANEL / "published_survival_comparator_addendum.json"
    published_addendum = json.loads(published_addendum_file.read_text(encoding="utf-8"))
    if (protocol["status"] != "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS" or
            pilot["status"] != "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS" or
            pilot["protocol_sha256"] != sha256(protocol_file) or
            reward_addendum["status"] != "PRELABEL_REWARD_SCALE_INTERPRETATION_FROZEN" or
            reward_addendum["protocol_sha256"] != sha256(protocol_file) or
            reward_addendum["test_labels_read"] or
            published_addendum["status"] != "PRELABEL_PUBLISHED_SURVIVAL_COMPARATOR_ADDENDUM_FROZEN" or
            published_addendum["protocol_sha256"] != sha256(protocol_file) or
            published_addendum["test_labels_read"] or
            {r["method"] for r in published_addendum["forecast_comparisons"]} != {
                "published_xgboost_cf_2026_adapted_b2",
                "xgboost_cf_2026_full108_architecture_control"}):
        raise RuntimeError("Fresh protocol/pilot seal failed")
    expected = {(r["mdp_id"], r["policy_family"], learner)
                for r in pilot["pilot_groups"]
                for learner in ("behavior_cloning_tabular", "cfqi_tabular", "iql_tabular")}
    if len(expected) != 75:
        raise RuntimeError("Incomplete pilot keys")
    # The full-input architecture control needs the completed OEH feature
    # assembly. Fit it here, while the label gate is still closed.
    if not PUBLISHED_CF_FULL.exists():
        from .fresh_published_survival_full_forecast import main as fit_full_control
        fit_full_control()
    files = {}
    for name, directory, status in (
            ("oeh_full", FULL, "SEALED_FRESH_FULL_108_FORECAST_NO_TEST_LABELS"),
            ("b2_internal", B2, "SEALED_FRESH_B2_FORECAST_NO_TEST_LABELS"),
            ("published_cf", PUBLISHED_CF,
             "SEALED_FRESH_PUBLISHED_XGBOOST_CF_ADAPTATION_NO_TEST_LABELS"),
            ("published_cf_full108", PUBLISHED_CF_FULL,
             "SEALED_FRESH_XGBOOST_CF_FULL108_ARCHITECTURE_CONTROL_NO_TEST_LABELS")):
        manifest_file = directory / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        file = directory / "predictions_before_label_read.csv"
        if (manifest["status"] != status or manifest["test_labels_read"] or
                manifest["protocol_sha256"] != sha256(protocol_file) or
                manifest["output_sha256"]["predictions"] != sha256(file)):
            raise RuntimeError(f"Bad {name} forecast manifest")
        frame = pd.read_csv(file)
        keys = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
        probability = frame[P].to_numpy(float)
        if (len(frame) != 75 or keys != expected or
                not np.isfinite(probability).all() or
                not np.allclose(probability.sum(axis=1), 1.0)):
            raise RuntimeError(f"Incomplete {name} predictions")
        files[f"{name}_manifest"] = sha256(manifest_file)
        files[f"{name}_predictions"] = sha256(file)
    original_manifest_file = ORIGINAL / "manifest.json"
    original_manifest = json.loads(original_manifest_file.read_text(encoding="utf-8"))
    original_file = ORIGINAL / "scores_before_label_read.csv"
    original = pd.read_csv(original_file)
    behavior_keys = {(r["mdp_id"], r["policy_family"]) for r in pilot["pilot_groups"]}
    if (original_manifest["status"] != "SEALED_FRESH_ORIGINAL_SCALAR_SCORES_NO_TEST_LABELS" or
            original_manifest["test_labels_read"] or
            original_manifest["protocol_sha256"] != sha256(protocol_file) or
            original_manifest["scores_sha256"] != sha256(original_file) or
            set(map(tuple, original[["mdp_id", "policy_family"]].itertuples(index=False, name=None))) != behavior_keys):
        raise RuntimeError("Original scalar scores missing before labels")
    files.update({"original_manifest": sha256(original_manifest_file),
                  "original_scores": sha256(original_file),
                  "pilot_manifest": sha256(pilot_file),
                  "protocol": sha256(protocol_file),
                  "reward_interpretation_addendum": sha256(reward_addendum_file),
                  "published_survival_comparator_addendum": sha256(published_addendum_file),
                  "label_kernel": sha256(ROOT / "benchmarks/bridge/fresh_labels.py"),
                  "label_runner": sha256(ROOT / "benchmarks/bridge/fresh_run_test_labels.py")})
    OUT.write_text(json.dumps({"status": "FRESH_PRELABEL_SEAL_READY",
                               "tasks": protocol["evaluation_tasks"],
                               "pilot_groups": 25, "prediction_rows_per_method": 75,
                               "original_score_rows": 25,
                               "sha256": files, "test_labels_read": False},
                              indent=2) + "\n", encoding="utf-8")
    print("Prelabel seal PASS: 75 each OEH/XGBoost-CF/B2/full-input control and 25 original-score rows", flush=True)


if __name__ == "__main__":
    main()
