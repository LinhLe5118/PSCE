"""Retrospective comparison of OEH with adapted published XGBoost-CF on 120 MDPs.

The 120 outcomes were opened before this published baseline was selected, so
all intervals in this file are descriptive, not a new confirmatory result.
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

from benchmarks.bridge.fresh_published_survival_forecast import (
    FEATURES_B2, KEYS, METHOD, P, OFFICIAL, expanded_train, predict_law,
)
from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import run_oeh_final_120 as R

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
OUTPUT = ROOT / "artifacts/week8/published_xgboost_cf_120_exploratory_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_law(train: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, dict]:
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(train[list(FEATURES_B2)].to_numpy(float))
    q = imputer.transform(query[list(FEATURES_B2)].to_numpy(float))
    xt, yt = expanded_train(x, train.event.to_numpy(int))
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                          random_state=42, eval_metric="logloss", n_jobs=1,
                          tree_method="hist")
    tic = time.perf_counter()
    model.fit(xt, yt)
    law = predict_law(model, q)
    return law, {"fit_predict_seconds": time.perf_counter() - tic,
                 "meta_training_rows": len(train), "expanded_rows": len(xt),
                 "query_rows": len(query)}


def score(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    p = result[P].to_numpy(float)
    event = result.event.to_numpy(int)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1) or not np.isin(event, range(7)).all():
        raise RuntimeError("Invalid event law or target")
    cdf = np.cumsum(p, axis=1)[:, :6]
    result["brier"] = np.mean((cdf - (event[:, None] <= np.arange(6))) ** 2, axis=1)
    result["log_loss"] = -np.log(np.maximum(p[np.arange(len(p)), event], 1e-12))
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    C.configure()
    historical_file = SOURCE / "protocol.json"
    historical = json.loads(historical_file.read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    if len(train) != 10800 or train.mdp_id.nunique() != 720:
        raise RuntimeError("Historical meta-training grid changed")
    pilot_file = SOURCE / "pilot_features.csv"
    query = pd.read_csv(pilot_file).sort_values(KEYS).reset_index(drop=True)
    if len(query) != 1800 or query.mdp_id.nunique() != 120 or query[KEYS].duplicated().any():
        raise RuntimeError("120 MDP query grid changed")
    # Panel membership is fixed by the already-frozen protocol; check against
    # the sealed preoutcome OEH predictions, never infer it from our outcomes.
    old_file = SOURCE / "predictions_preoutcome.csv"
    old = pd.read_csv(old_file)
    old = old[old.method == "oeh_final"][KEYS + ["panel"] + P]
    if len(old) != 1800 or old[KEYS].duplicated().any():
        raise RuntimeError("Original OEH grid changed")
    query = query.merge(old[KEYS + ["panel"]], on=KEYS, validate="one_to_one")
    prediction = query[KEYS + ["panel"]].copy()
    prediction["method"] = METHOD
    prediction[P] = np.nan
    timing = {}
    for panel in ("primary", "control"):
        for learner in sorted(query.algorithm.unique()):
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            qi = (query.panel == panel) & (query.algorithm == learner)
            if tr.mdp_id.nunique() < 10 or qi.sum() == 0:
                raise RuntimeError("Missing train/query stratum")
            law, elapsed = fit_law(tr, query.loc[qi])
            prediction.loc[qi, P] = law
            timing[f"{panel}/{learner}"] = elapsed
            print(f"XGBoost-CF {panel}/{learner}: {qi.sum()} query rows", flush=True)
    if not np.isfinite(prediction[P].to_numpy(float)).all():
        raise RuntimeError("Incomplete published baseline")
    OUTPUT.mkdir(parents=True)
    pred_file = OUTPUT / "predictions.csv"
    prediction.to_csv(pred_file, index=False)
    # The labels are already opened: this order is for audit, not a claim of
    # prospective or confirmatory sealing.
    label_file = SOURCE / "labels.csv"
    labels = pd.read_csv(label_file)[KEYS + ["panel", "event"]]
    if labels[KEYS].duplicated().any() or len(labels) != 1800:
        raise RuntimeError("Historical label grid changed")
    methods = pd.concat([prediction, old.assign(method="oeh_final")], ignore_index=True)
    joined = methods.merge(labels, on=KEYS + ["panel"], validate="many_to_one")
    if len(joined) != 3600:
        raise RuntimeError("Joined 120 MDP grid changed")
    scored = score(joined)
    summary = (scored.groupby(["panel", "algorithm", "method"], observed=True)
               .agg(rows=("event", "size"), mdps=("mdp_id", "nunique"),
                    brier=("brier", "mean"), log_loss=("log_loss", "mean"))
               .reset_index())
    rng = np.random.default_rng(20261008)
    paired = []
    for (panel, learner), group in scored.groupby(["panel", "algorithm"], observed=True):
        for metric in ("brier", "log_loss"):
            pivot = group.pivot(index=["mdp_id", "policy_family"], columns="method", values=metric)
            delta = pivot["oeh_final"] - pivot[METHOD]
            by_mdp = delta.groupby(level="mdp_id").mean().to_numpy()
            samples = rng.choice(by_mdp, size=(10000, len(by_mdp)), replace=True).mean(axis=1)
            paired.append({"panel": panel, "algorithm": learner, "metric": metric,
                           "mdps": len(by_mdp), "oeh_minus_published_cf": float(by_mdp.mean()),
                           "ci95_low_descriptive": float(np.quantile(samples, .025)),
                           "ci95_high_descriptive": float(np.quantile(samples, .975))})
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(OUTPUT / "paired_mdp_bootstrap.csv", index=False)
    (OUTPUT / "fit_summary.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_RETROSPECTIVE_EXPLORATORY_120_COMPARISON",
        "interpretation": "120 labels were already opened when this comparator was chosen; descriptive only",
        "method": METHOD,
        "publication": "https://arxiv.org/html/2601.22259v2",
        "author_code": "https://github.com/dain20000222/TabularFM4Survival",
        "author_commit": "b27d69c27a5c39b87822b1aebf838e3dd900000e",
        "source_sha256": {"pilot_features": sha256(pilot_file), "labels": sha256(label_file),
                          "old_oeh_predictions": sha256(old_file),
                          "historical_protocol": sha256(historical_file),
                          "official_author_code": sha256(OFFICIAL),
                          "script": sha256(Path(__file__))},
        "output_sha256": {name: sha256(OUTPUT / name) for name in
                          ("predictions.csv", "scored.csv", "summary.csv",
                           "paired_mdp_bootstrap.csv", "fit_summary.json")}}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
