"""Development-only matched AFT comparison of B2+oracle EH and B2+OEH.

Oracle EH uses true MDP dynamics. This tests descriptor transfer to the
offline requirement endpoint, not the original online sample-complexity bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import xgboost as xgb

from scripts.analysis.compare_difficulty import FEATURES_B2, GRID, METHODS, aft_interval
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.independent_difficulty import XGB_PARAMS


ROOT = Path(__file__).resolve().parents[2]
E1_LABELS = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"
E1_FEATURES = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv"
E1_ORACLE = ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30/oracle_diagnostics.csv"
DEV_FEATURES = ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv"
DEV_ORACLE = ROOT / "artifacts/independent_difficulty_2026-09-30/oracle_preoutcome.csv"
DEV_LABELS = ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv"
DEV_PREOUTCOME = ROOT / "artifacts/independent_difficulty_2026-09-30/predictions_preoutcome.csv"
OUTPUT = ROOT / "artifacts/oeh_b2_eh_oracle_aft_development_2026-10-01"
KEYS = ["mdp_id", "policy_family", "algorithm"]
EH = ("stochastic_eh_hbar_h100", "stochastic_eh_min_qvi_k_h100",
      "stochastic_eh_delta_h100")
FIELDS = (*FEATURES_B2, *EH)
P = [f"p_{i}" for i in range(7)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    train = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                          validate="one_to_one")
    train = train.merge(pd.read_csv(E1_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    query = pd.read_csv(DEV_LABELS).merge(pd.read_csv(DEV_FEATURES), on=KEYS,
                                           validate="one_to_one")
    query = query.merge(pd.read_csv(DEV_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    if len(train) != 1800 or len(query) != 900:
        raise RuntimeError("E1 or opened development grid incomplete")
    rows = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            qu = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(tr) != (420 if panel == "primary" else 180) or len(qu) != (210 if panel == "primary" else 90):
                raise RuntimeError("Learner/panel row mismatch")
            x = np.where(np.isfinite(tr[list(FIELDS)].to_numpy(dtype=float)),
                         tr[list(FIELDS)].to_numpy(dtype=float), np.nan)
            q = np.where(np.isfinite(qu[list(FIELDS)].to_numpy(dtype=float)),
                         qu[list(FIELDS)].to_numpy(dtype=float), np.nan)
            lower, upper = np.asarray([aft_interval(int(value)) for value in tr.event], dtype=float).T
            matrix = xgb.DMatrix(x)
            matrix.set_float_info("label_lower_bound", lower)
            matrix.set_float_info("label_upper_bound", upper)
            model = xgb.train(XGB_PARAMS, matrix, num_boost_round=200)
            median = model.predict(xgb.DMatrix(q))
            if not np.isfinite(median).all() or (median <= 0).any():
                raise RuntimeError("Invalid AFT median")
            cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
            probabilities = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            brier, logloss = score_probabilities(probabilities, qu.event.to_numpy(dtype=int))
            frame = qu[KEYS + ["panel", "event"]].copy()
            frame["method"] = "b2_plus_eh_oracle_aft"
            frame["brier"] = brier
            frame["logloss"] = logloss
            for index, column in enumerate(P):
                frame[column] = probabilities[:, index]
            rows.append(frame)
            print(f"Fitted oracle EH AFT: {panel}/{learner}", flush=True)
    forecast = pd.concat(rows, ignore_index=True)
    baseline = pd.read_csv(DEV_PREOUTCOME)
    baseline = baseline[baseline.method == "b2_plus_oeh"]
    joined = query[KEYS + ["panel", "event"]].merge(baseline[KEYS + P], on=KEYS,
                                                       validate="one_to_one")
    if len(joined) != 900:
        raise RuntimeError("Saved B2+OEH AFT predictions incomplete")
    brier, logloss = score_probabilities(joined[P].to_numpy(dtype=float),
                                         joined.event.to_numpy(dtype=int))
    joined["method"] = "b2_plus_oeh_aft"
    joined["brier"] = brier
    joined["logloss"] = logloss
    scored = pd.concat([forecast, joined], ignore_index=True)
    summary = scored.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    manifest = {
        "status": "development_only_opened_60_mdp_not_confirmation",
        "scope": "Same AFT, B2+full-MDP oracle EH versus B2+pilot-only OEH; offline target transfer only",
        "eh_features": EH,
        "aft_config": XGB_PARAMS,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in
                          (E1_LABELS, E1_FEATURES, E1_ORACLE, DEV_FEATURES,
                           DEV_ORACLE, DEV_LABELS, DEV_PREOUTCOME)},
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
