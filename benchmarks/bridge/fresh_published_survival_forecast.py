"""Fit the published XGBoost-CF survival formulation to frozen OEH budgets.

This is an explicitly adapted published *survival* baseline, not a published
offline-RL requirement forecaster. No fresh BRIDGE learner outcome is read.
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
from scripts.analysis.compare_difficulty import FEATURES_B2
from .fresh_adapter import PANEL

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
FEATURES = ROOT / "artifacts/week8/bridge_fresh_pilot_features_preoutcome_2026-10-08"
OUTPUT = ROOT / "artifacts/week8/bridge_fresh_published_xgboost_cf_preoutcome_2026-10-08"
OFFICIAL = ROOT / "benchmarks/bridge/third_party/TabularFM4Survival/static/static_xgboost_final.py"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
BUDGETS = np.array([1., 2., 5., 10., 20., 50.])
METHOD = "published_xgboost_cf_2026_adapted_b2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expanded_train(x: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Author CF: one binary row per time and one extra subject-time row."""
    n, d = x.shape
    if not np.isin(event, np.arange(7)).all():
        raise ValueError("Invalid requirement event")
    times = np.broadcast_to(BUDGETS, (n, len(BUDGETS)))
    labels = (event[:, None] < 6) & (event[:, None] <= np.arange(6))
    base = np.column_stack((np.repeat(x, 6, axis=0), times.reshape(-1)))
    event_times = BUDGETS[np.minimum(event, 5)]
    own = np.column_stack((x, event_times))
    # Original implementation appends each subject's observed/censoring time.
    # Here censoring is administrative at x50; it remains a known negative.
    own_labels = (event < 6).astype(int)
    return np.vstack((base, own)), np.concatenate((labels.astype(int).reshape(-1), own_labels))


def predict_law(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    n = len(x)
    features = np.column_stack((np.repeat(x, 6, axis=0),
                                np.tile(BUDGETS, n)))
    survival = model.predict_proba(features)[:, 0].reshape(n, 6)
    # Same non-increasing survival repair as the authors' implementation.
    survival = np.minimum.accumulate(survival, axis=1)
    cdf = 1.0 - survival
    law = np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), survival[:, -1]))
    if not np.isfinite(law).all() or (law < -1e-7).any() or not np.allclose(law.sum(1), 1):
        raise RuntimeError("Invalid XGBoost-CF event distribution")
    return np.clip(law, 0, 1)


def main() -> None:
    if OUTPUT.exists() or (PANEL / "prelabel_seal.json").exists():
        raise FileExistsError("Forecast output or prelabel seal already exists")
    protocol_file = PANEL / "protocol.json"
    feature_file = FEATURES / "features.csv"
    fm = json.loads((FEATURES / "manifest.json").read_text(encoding="utf-8"))
    if (fm["status"] != "SEALED_FRESH_30_COLUMN_PILOT_FEATURES_NO_TEST_LABELS" or
            fm["test_labels_read"] or fm["protocol_sha256"] != sha256(protocol_file) or
            fm["features_sha256"] != sha256(feature_file)):
        raise RuntimeError("Pilot-only feature seal failed")
    query = pd.read_csv(feature_file).sort_values(KEYS).reset_index(drop=True)
    if (len(query) != 75 or query.mdp_id.nunique() != 5 or
            query[KEYS].duplicated().any() or set(FEATURES_B2) - set(query.columns)):
        raise RuntimeError("Incomplete fresh query")
    C.configure()
    historical_file = SOURCE / "protocol.json"
    historical = json.loads(historical_file.read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training grid changed")
    train = train[train.panel == "primary"].copy()
    if len(train) != 7560 or train.mdp_id.nunique() != 504 or train[KEYS].duplicated().any():
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
            raise RuntimeError(f"Incomplete learner stratum: {learner}")
        imputer = SimpleImputer(strategy="median")
        x = imputer.fit_transform(tr[list(FEATURES_B2)].to_numpy(float))
        q = imputer.transform(query.loc[qi, list(FEATURES_B2)].to_numpy(float))
        xt, yt = expanded_train(x, tr.event.to_numpy(int))
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                              random_state=42, eval_metric="logloss", n_jobs=1,
                              tree_method="hist")
        model.fit(xt, yt)
        prediction.loc[qi, P] = predict_law(model, q)
        timing[learner] = {"fit_predict_seconds": time.perf_counter() - tic,
                           "training_rows": len(tr), "expanded_rows": len(xt),
                           "event_fraction": float(np.mean(tr.event.to_numpy(int) < 6))}
        print(f"XGBoost-CF {learner}: 25 forecasts sealed in memory", flush=True)
    probability = prediction[P].to_numpy(float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1):
        raise RuntimeError("Incomplete XGBoost-CF predictions")
    OUTPUT.mkdir(parents=True)
    prediction.to_csv(OUTPUT / "predictions_before_label_read.csv", index=False)
    (OUTPUT / "fit_summary.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "SEALED_FRESH_PUBLISHED_XGBOOST_CF_ADAPTATION_NO_TEST_LABELS",
        "test_labels_read": False, "rows": 75, "meta_training_mdps_total": 720,
        "meta_training_mdps_primary_used": 504,
        "publication": "Kim, Lai, Zhang (2026), Tabular Foundation Models Can Do Survival Analysis",
        "publication_url": "https://arxiv.org/html/2601.22259v2",
        "official_repository": "https://github.com/dain20000222/TabularFM4Survival",
        "official_repository_commit": "b27d69c27a5c39b87822b1aebf838e3dd900000e",
        "role": "published general-survival XGBoost-CF formulation adapted to offline-RL budget labels; not original offline-RL SOTA",
        "adaptations": ["fixed project x1,x2,x5,x10,x20,x50 horizons", "pilot-only 18 B2 descriptive inputs",
                        "one model per learner", "administrative censoring at x50", "no test-label tuning"],
        "author_hyperparameters": {"n_estimators": 100, "learning_rate": 0.1,
                                   "max_depth": 3, "random_state": 42},
        "protocol_sha256": sha256(protocol_file),
        "source_sha256": {"pilot_features": sha256(feature_file),
                          "historical_protocol": sha256(historical_file),
                          "official_author_code": sha256(OFFICIAL), "adapter_script": sha256(Path(__file__))},
        "output_sha256": {"predictions": sha256(OUTPUT / "predictions_before_label_read.csv"),
                          "fit_summary": sha256(OUTPUT / "fit_summary.json")}}, indent=2) + "\n", encoding="utf-8")
    print("Sealed 75 published-formulation XGBoost-CF forecasts before labels", flush=True)


if __name__ == "__main__":
    main()
