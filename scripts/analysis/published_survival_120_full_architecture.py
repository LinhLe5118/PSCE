"""Retrospective 120-MDP architecture control with OEH's 108 inputs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from benchmarks.bridge.fresh_published_survival_forecast import KEYS, P, OFFICIAL, expanded_train, predict_law
from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis import run_oeh_final_120 as R
from scripts.analysis import oeh_final_forecast as F
from scripts.analysis.develop_cross_learner_context import with_context
from scripts.analysis.published_survival_120_exploratory import score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
OUTPUT = ROOT / "artifacts/week8/published_xgboost_cf_120_full108_architecture_2026-10-08"
METHOD = "xgboost_cf_2026_full108_architecture_control"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    C.configure()
    historical_file = SOURCE / "protocol.json"
    historical = json.loads(historical_file.read_text(encoding="utf-8"))
    train = R.meta_training(R.OPENED, historical["oeh_final"]["structure_tag"])
    pilot_file = SOURCE / "pilot_features.csv"
    horizon_file = SOURCE / "horizon_preoutcome.csv"
    query = pd.read_csv(pilot_file)
    query = with_context(query.merge(F.oeh_feature_rows(SOURCE / "oeh_b_groups", horizon_file),
                                     on=KEYS, validate="one_to_one"))
    old_file = SOURCE / "predictions_preoutcome.csv"
    old = pd.read_csv(old_file)
    old = old[old.method == "oeh_final"][KEYS + ["panel"] + P]
    query = query.merge(old[KEYS + ["panel"]], on=KEYS, validate="one_to_one")
    query = query.sort_values(KEYS).reset_index(drop=True)
    fields = tuple(R.CANDIDATE_VARIANTS["plus_prior_profile"])
    if (len(fields) != 108 or len(query) != 1800 or query.mdp_id.nunique() != 120 or
            query[KEYS].duplicated().any() or set(fields) - set(query.columns)):
        raise RuntimeError("Incomplete 120-MDP full-feature grid")
    prediction = query[KEYS + ["panel"]].copy()
    prediction["method"] = METHOD
    prediction[P] = np.nan
    timing = {}
    for panel in ("primary", "control"):
        for learner in sorted(query.algorithm.unique()):
            tic = time.perf_counter()
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            qi = (query.panel == panel) & (query.algorithm == learner)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x = imputer.fit_transform(F._matrix(tr, fields))
            q = imputer.transform(F._matrix(query.loc[qi], fields))
            xt, yt = expanded_train(x, tr.event.to_numpy(int))
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                  random_state=42, eval_metric="logloss", n_jobs=1,
                                  tree_method="hist")
            model.fit(xt, yt)
            prediction.loc[qi, P] = predict_law(model, q)
            timing[f"{panel}/{learner}"] = {"fit_predict_seconds": time.perf_counter() - tic,
                                             "training_rows": len(tr), "query_rows": int(qi.sum())}
            print(f"Full-input XGBoost-CF {panel}/{learner}: {qi.sum()} rows", flush=True)
    if not np.isfinite(prediction[P].to_numpy(float)).all():
        raise RuntimeError("Incomplete full-input predictions")
    OUTPUT.mkdir(parents=True)
    prediction.to_csv(OUTPUT / "predictions.csv", index=False)
    labels_file = SOURCE / "labels.csv"
    labels = pd.read_csv(labels_file)[KEYS + ["panel", "event"]]
    joined = pd.concat([prediction, old.assign(method="oeh_final")], ignore_index=True)
    joined = joined.merge(labels, on=KEYS + ["panel"], validate="many_to_one")
    if len(joined) != 3600:
        raise RuntimeError("Incomplete 120-MDP score grid")
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
                           "mdps": len(by_mdp), "oeh_minus_full_input_cf": float(by_mdp.mean()),
                           "ci95_low_descriptive": float(np.quantile(samples, .025)),
                           "ci95_high_descriptive": float(np.quantile(samples, .975))})
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(OUTPUT / "paired_mdp_bootstrap.csv", index=False)
    (OUTPUT / "fit_summary.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE_RETROSPECTIVE_120_FULL_INPUT_ARCHITECTURE_CONTROL",
        "interpretation": "Uses OEH descriptors and already-opened 120 labels; architecture control, descriptive only",
        "method": METHOD, "feature_count": 108,
        "publication": "https://arxiv.org/html/2601.22259v2",
        "source_sha256": {"pilot_features": sha256(pilot_file),
                          "horizon_preoutcome": sha256(horizon_file),
                          "labels": sha256(labels_file),
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
