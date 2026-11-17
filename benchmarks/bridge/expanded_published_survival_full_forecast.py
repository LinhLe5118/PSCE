# Generated from the sealed fresh pipeline by materialize_expanded_pipeline.py.
"""Architecture control: published XGBoost-CF with exactly OEH's 108 inputs.

This reuses OEH descriptors and is therefore NOT an independent-method
comparator. It isolates the output-model architecture under identical inputs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis import oeh_final_forecast as F
from scripts.analysis.develop_cross_learner_context import with_context
from .expanded_adapter import PANEL
from .expanded_published_survival_forecast import OFFICIAL, KEYS, P, expanded_train, predict_law, sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_expanded_pilot_features_preoutcome_2026-10-08"
OEHB = ROOT / "artifacts/week8/bridge_expanded_full_oehb_preoutcome_2026-10-08"
FULL = ROOT / "artifacts/week8/bridge_expanded_full_forecast_preoutcome_2026-10-08"
OUTPUT = ROOT / "artifacts/week8/bridge_expanded_published_xgboost_cf_full108_preoutcome_2026-10-08"
METHOD = "xgboost_cf_2026_full108_architecture_control"


def main() -> None:
    if OUTPUT.exists() or (PANEL / "prelabel_seal.json").exists():
        raise FileExistsError("Full-input control output or prelabel seal already exists")
    protocol_file = PANEL / "protocol.json"
    full_manifest_file = FULL / "manifest.json"
    fm = json.loads(full_manifest_file.read_text(encoding="utf-8"))
    if (fm["status"] != "SEALED_FRESH_FULL_108_FORECAST_NO_TEST_LABELS" or
            fm["test_labels_read"] or fm["protocol_sha256"] != sha256(protocol_file) or
            fm["source_sha256"]["pilot_features"] != sha256(FEATURES / "features.csv") or
            fm["source_sha256"]["oehb_manifest"] != sha256(OEHB / "manifest.json")):
        raise RuntimeError("Full OEH input seal failed")
    model_horizon = FULL / "horizon_model_input_before_labels.csv"
    if fm["source_sha256"]["model_horizon"] != sha256(model_horizon):
        raise RuntimeError("Full OEH horizon input changed")
    query = pd.read_csv(FEATURES / "features.csv")
    own = F.oeh_feature_rows(OEHB / "groups", model_horizon)
    query = with_context(query.merge(own, on=KEYS, validate="one_to_one"))
    query["panel"] = "primary"
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    if (len(fields) != 108 or len(query) != 75 or query[KEYS].duplicated().any() or
            set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete 108-feature control query")
    C.configure()
    historical_file = SOURCE / "protocol.json"
    historical = json.loads(historical_file.read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    train = train[train.panel == "primary"].copy()
    if len(train) != 7560 or train.mdp_id.nunique() != 504:
        raise RuntimeError("Primary meta-training grid changed")
    prediction = query[KEYS].copy()
    prediction["method"] = METHOD
    prediction[P] = np.nan
    timing = {}
    for learner in sorted(query.algorithm.unique()):
        tic = time.perf_counter()
        tr = train[train.algorithm == learner]
        qi = query.algorithm == learner
        if len(tr) != 2520 or qi.sum() != 25:
            raise RuntimeError("Incomplete learner stratum")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x = imputer.fit_transform(F._matrix(tr, fields))
        q = imputer.transform(F._matrix(query.loc[qi], fields))
        xt, yt = expanded_train(x, tr.event.to_numpy(int))
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                              random_state=42, eval_metric="logloss", n_jobs=1,
                              tree_method="hist")
        model.fit(xt, yt)
        prediction.loc[qi, P] = predict_law(model, q)
        timing[learner] = {"fit_predict_seconds": time.perf_counter() - tic,
                           "training_rows": len(tr), "expanded_rows": len(xt)}
    probability = prediction[P].to_numpy(float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1):
        raise RuntimeError("Incomplete full-input control")
    OUTPUT.mkdir(parents=True)
    prediction.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "fit_summary.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_XGBOOST_CF_FULL108_ARCHITECTURE_CONTROL_NO_TEST_LABELS",
        "test_labels_read": False, "rows": 75, "meta_training_mdps_primary_used": 504,
        "role": "architecture control using OEH's 108 features; not an independent published-method comparator",
        "publication": "https://arxiv.org/html/2601.22259v2",
        "author_hyperparameters": {"n_estimators": 100, "learning_rate": 0.1,
                                   "max_depth": 3, "random_state": 42},
        "feature_columns": list(fields), "protocol_sha256": sha256(protocol_file),
        "source_sha256": {"full_oeh_manifest": sha256(full_manifest_file),
                          "pilot_features": sha256(FEATURES / "features.csv"),
                          "historical_protocol": sha256(historical_file),
                          "official_author_code": sha256(OFFICIAL),
                          "adapter_script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "fit_summary": sha256(OUTPUT / "fit_summary.json")}}, indent=2) + "\n", encoding="utf-8")
    print("Sealed 75 full-input XGBoost-CF architecture-control forecasts", flush=True)


if __name__ == "__main__":
    main()
