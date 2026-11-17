"""Development-only joint categorical OEH forecast on two opened 60-MDP sets.

This tests a structural alternative to six separately fitted cumulative
classifiers: one seven-class proper-loss model. Tree settings are unchanged,
shared across families and learners, and selected without a new search.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES, KEYS, PROBABILITY_COLUMNS
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, _load, score_probabilities, PARAMS, ROUNDS,
)
from scripts.analysis.confirm_oeh_difficulty import OUTPUT as CONFIRMATION

OUTPUT = ROOT / "artifacts/oeh_difficulty_multiclass_development_2026-10-01"


def fit_multiclass(x, event, query):
    import numpy as np
    import xgboost as xgb

    config = {**PARAMS, "objective": "multi:softprob", "eval_metric": "mlogloss",
              "num_class": 7}
    labels = np.asarray(event, dtype=int)
    if (labels.ndim != 1 or (labels < 0).any() or (labels > 6).any()
            or not np.isfinite(x).all() or not np.isfinite(query).all()):
        raise ValueError("Invalid multiclass training data")
    model = xgb.train(config, xgb.DMatrix(x, label=labels), num_boost_round=ROUNDS)
    probabilities = np.asarray(model.predict(xgb.DMatrix(query)), dtype=float)
    if (probabilities.shape != (len(query), 7) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()):
        raise RuntimeError("Invalid multiclass forecast")
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main():
    import numpy as np
    import pandas as pd

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact: {OUTPUT}")
    train, first, paths = _load()
    result_file = CONFIRMATION / "result_manifest.json"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("jobs") != 54000:
        raise RuntimeError("Second opened cohort is incomplete")
    for name, digest in result["artifact_sha256"].items():
        if sha256(CONFIRMATION / name) != digest:
            raise RuntimeError(f"Second-cohort artifact changed: {name}")
    second = pd.read_csv(CONFIRMATION / "labels.csv").merge(
        pd.read_csv(CONFIRMATION / "pilot_features.csv"),
        on=["mdp_id", "policy_family", "algorithm"], validate="one_to_one")
    if (len(second) != 900 or second.mdp_id.nunique() != 60
            or set(train.mdp_id) & set(second.mdp_id)
            or set(first.mdp_id) & set(second.mdp_id)):
        raise RuntimeError("Second development cohort overlaps or is incomplete")
    queries = {"first_open": first.sort_values(KEYS).reset_index(drop=True),
               "second_open": second.sort_values(KEYS).reset_index(drop=True)}
    train = train.sort_values(KEYS).reset_index(drop=True)
    predictions = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            if len(tr) != (420 if panel == "primary" else 180):
                raise RuntimeError("Training stratum incomplete")
            for method, fields in (("b2_multiclass", FEATURES_B2),
                                   ("oeh_multiclass", OEH_FEATURES)):
                for cohort, frame in queries.items():
                    qu = frame[(frame.panel == panel) & (frame.algorithm == learner)]
                    if len(qu) != (210 if panel == "primary" else 90):
                        raise RuntimeError("Query stratum incomplete")
                    probabilities = fit_multiclass(
                        tr[list(fields)].to_numpy(dtype=float),
                        tr.event.to_numpy(dtype=int),
                        qu[list(fields)].to_numpy(dtype=float))
                    brier, logloss = score_probabilities(
                        probabilities, qu.event.to_numpy(dtype=int))
                    for index, (_, item) in enumerate(qu.iterrows()):
                        predictions.append({**{key: item[key] for key in KEYS},
                                            "cohort": cohort, "event": int(item.event),
                                            "method": method, "brier": float(brier[index]),
                                            "logloss": float(logloss[index]),
                                            **{name: float(probabilities[index, category])
                                               for category, name in enumerate(PROBABILITY_COLUMNS)}})
            print(f"Multiclass development: {panel}/{learner}", flush=True)
    detail = pd.DataFrame(predictions)
    if len(detail) != 3600:
        raise RuntimeError("Incomplete multiclass development forecasts")
    scores = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_60_mdp_cohorts_not_confirmation",
        "candidate": "joint_seven_class_xgboost_pilot_oeh",
        "promoted": False,
        "feature_sets": {"b2_multiclass": FEATURES_B2,
                         "oeh_multiclass": OEH_FEATURES},
        "fit": {"params": {**PARAMS, "objective": "multi:softprob",
                           "eval_metric": "mlogloss", "num_class": 7},
                "rounds": ROUNDS},
        "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                         "second_result": sha256(result_file),
                         "second_pilot": sha256(CONFIRMATION / "pilot_features.csv")},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("predictions.csv", "scores.csv")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
