"""Gate-conditional, descriptive 120-MDP E1 comparison of the revised estimator.

E1 supplied the meta-training labels and was used in earlier development.
Its MDP-grouped out-of-fold comparison is useful for an exact original-cohort
audit, but is never independent confirmation or a new one-shot evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import sha256, METHODS
from scripts.analysis.confirm_oeh_difficulty import ROOT, OUTPUT as CONFIRMATION
from scripts.analysis.develop_oeh_blend import FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities


E1 = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30"
E1_QUALITY = ROOT / "artifacts/e1_dataset_signals_posthoc_2026-09-30"
FEATURES_FILE = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv"
OUTPUT = ROOT / "artifacts/oeh_difficulty_e1_descriptive_2026-10-01"
ACCEPTANCE = CONFIRMATION / "published_method_5pct_acceptance.json"
KEYS = ["mdp_id", "policy_family", "algorithm", "panel", "fold", "event"]
P = [f"p_{index}" for index in range(7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd

    gate_path = CONFIRMATION / "result_manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "complete" or gate["difficulty_gate"] != "GO":
        raise RuntimeError("Independent difficulty gate is not GO; E1 comparison forbidden")
    # The user added a stricter acceptance criterion after this confirmation
    # had already started. A separate, auditable comparison with original
    # published methods and >=5% relative gain must be recorded first.
    if not ACCEPTANCE.exists():
        raise RuntimeError("Original published-method >=5% acceptance is absent; E1 comparison forbidden")
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    if (acceptance.get("status") != "GO" or acceptance.get("minimum_relative_gain") != .05
            or acceptance.get("confirmation_result_sha256") != sha256(gate_path)):
        raise RuntimeError("Published-method >=5% acceptance is invalid")
    protocol = json.loads((CONFIRMATION / "protocol.json").read_text(encoding="utf-8"))
    if gate["protocol_sha256"] != sha256(CONFIRMATION / "protocol.json"):
        raise RuntimeError("Confirmation protocol/gate hash mismatch")
    for name, digest in gate["artifact_sha256"].items():
        if sha256(CONFIRMATION / name) != digest:
            raise RuntimeError(f"Confirmation artifact changed: {name}")
    original_manifest = json.loads((E1 / "manifest.json").read_text(encoding="utf-8"))
    quality_manifest = json.loads((E1_QUALITY / "manifest.json").read_text(encoding="utf-8"))
    feature_seal = json.loads((FEATURES_FILE.parent / "feature_seal.json").read_text(encoding="utf-8"))
    if (sha256(E1 / "labels.csv") != original_manifest["output_hashes"]["labels.csv"]
            or sha256(E1 / "predictions.csv") != original_manifest["output_hashes"]["predictions.csv"]
            or sha256(FEATURES_FILE) != feature_seal["features_sha256"]
            or sha256(E1_QUALITY / "predictions.csv") != quality_manifest["output_hashes"]["predictions.csv"]):
        raise RuntimeError("Historical E1 input hash changed")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite E1 comparison: {args.output}")
    frame = pd.read_csv(E1 / "labels.csv")
    frame = frame.merge(pd.read_csv(FEATURES_FILE),
                        on=["mdp_id", "policy_family", "algorithm"], validate="one_to_one")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    if len(frame) != 1800 or frame.mdp_id.nunique() != 120:
        raise RuntimeError("Original E1 rows incomplete")
    anchor = pd.read_csv(E1 / "predictions.csv")
    anchor = anchor[anchor.method == "xgboost_aft_b2_oeh"][KEYS + P]
    aligned = frame[KEYS].merge(anchor, on=KEYS, validate="one_to_one")
    if len(aligned) != len(frame) or not aligned[KEYS].equals(frame[KEYS]):
        raise RuntimeError("Original AFT fold/label alignment changed")
    direct = np.full((len(frame), 7), np.nan)
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            group = np.flatnonzero((frame.panel == panel) & (frame.algorithm == algorithm))
            for fold in range(5):
                tr = group[frame.iloc[group].fold.to_numpy() != fold]
                te = group[frame.iloc[group].fold.to_numpy() == fold]
                if not len(tr) or not len(te) or set(frame.iloc[tr].mdp_id) & set(frame.iloc[te].mdp_id):
                    raise RuntimeError("MDP-grouped out-of-fold split failed")
                direct[te] = fit_thresholds(frame.iloc[tr][list(FEATURES)].to_numpy(),
                                            frame.iloc[tr].event.to_numpy(),
                                            frame.iloc[te][list(FEATURES)].to_numpy())
            print(f"E1 descriptive folds: {panel}/{algorithm}", flush=True)
    if not np.isfinite(direct).all():
        raise RuntimeError("Incomplete E1 out-of-fold predictions")
    weight = float(protocol["direct_weight"])
    candidate = weight * direct + (1. - weight) * aligned[P].to_numpy()
    brier, logloss = score_probabilities(candidate, frame.event.to_numpy())
    rows = frame[KEYS].copy()
    rows["method"] = "oeh_blend"
    rows["brier"] = brier
    rows["log_loss"] = logloss
    for index, name in enumerate(P):
        rows[name] = candidate[:, index]
    comparators = pd.read_csv(E1_QUALITY / "predictions.csv")
    reference = pd.read_csv(E1 / "predictions.csv")
    frequency = reference[reference.method == "train_frequency"]
    comparators = pd.concat([comparators, frequency], ignore_index=True)
    names = ("train_frequency", "coverage_adaptation", "quality_adaptation",
             "coverage_quality_adaptation", "b2_generic", "b2_plus_oeh")
    comparators = comparators[comparators.method.isin(names)]
    for name in names:
        part = comparators[comparators.method == name]
        if len(part) != 1800 or part[KEYS].duplicated().any():
            raise RuntimeError(f"Original comparator predictions incomplete: {name}")
    all_rows = pd.concat([rows, comparators[rows.columns]], ignore_index=True)
    summary = all_rows.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("log_loss", "mean"))
    args.output.mkdir(parents=True)
    all_rows.to_csv(args.output / "predictions.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    manifest = {
        "status": "posthoc_original_E1_adaptively_reused_not_confirmation",
        "independent_gate_result_sha256": sha256(gate_path),
        "confirmation_protocol_sha256": sha256(CONFIRMATION / "protocol.json"),
        "original_e1_mdps": 120, "event_rows": 1800,
        "comparison": "MDP-grouped five-fold out-of-fold forecasts on exact original E1 labels/features",
        "limitation": "E1 labels used for meta-training and repeated development; descriptive only; comparator quality methods are tabular adaptations",
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in (E1 / "labels.csv", E1 / "predictions.csv",
                                      E1_QUALITY / "predictions.csv", FEATURES_FILE)},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(args.output / name) for name in ("predictions.csv", "summary.csv")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
