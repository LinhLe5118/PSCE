"""Development-only discrete-time survival forecast for dataset difficulty.

The empirical event is an ordered first-success budget. Six conditional
hazards, trained with the same fixed binary tree settings, share its survival
structure and yield a proper seven-category distribution without CDF repair.
Both B2 and OEH use this exact architecture and all environment families.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.confirm_oeh_difficulty import OUTPUT as CONFIRMATION
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES, KEYS, PROBABILITY_COLUMNS
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, _load, score_probabilities, PARAMS, ROUNDS,
)

OUTPUT = ROOT / "artifacts/oeh_difficulty_hazard_development_2026-10-01"


def fit_hazards(x, event, query):
    import numpy as np
    import xgboost as xgb

    x = np.asarray(x, dtype=float)
    y = np.asarray(event, dtype=int)
    q = np.asarray(query, dtype=float)
    if (x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]
            or len(x) != len(y) or (y < 0).any() or (y > 6).any()
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError("Invalid discrete-hazard data")
    survival = np.ones(len(q), dtype=float)
    probabilities = np.zeros((len(q), 7), dtype=float)
    for budget in range(6):
        at_risk = y >= budget
        events = (y[at_risk] == budget).astype(float)
        if not len(events):
            hazard = np.zeros(len(q))
        elif np.unique(events).size == 1:
            hazard = np.full(len(q), events[0])
        else:
            model = xgb.train(PARAMS, xgb.DMatrix(x[at_risk], label=events),
                              num_boost_round=ROUNDS)
            hazard = model.predict(xgb.DMatrix(q))
        if not np.isfinite(hazard).all() or (hazard < 0).any() or (hazard > 1).any():
            raise RuntimeError("Invalid conditional hazard")
        probabilities[:, budget] = survival * hazard
        survival *= 1. - hazard
    probabilities[:, 6] = survival
    if not np.allclose(probabilities.sum(axis=1), 1.):
        raise RuntimeError("Hazards do not form a proper event distribution")
    return probabilities


def main():
    import pandas as pd

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    train, first, paths = _load()
    result_path = CONFIRMATION / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result["status"] != "complete" or result["jobs"] != 54000
            or sha256(CONFIRMATION / "labels.csv") != result["artifact_sha256"]["labels.csv"]):
        raise RuntimeError("Second opened cohort is incomplete or changed")
    second = pd.read_csv(CONFIRMATION / "labels.csv").merge(
        pd.read_csv(CONFIRMATION / "pilot_features.csv"),
        on=["mdp_id", "policy_family", "algorithm"], validate="one_to_one")
    queries = {"first_open": first, "second_open": second}
    if len(train) != 1800 or any(len(frame) != 900 for frame in queries.values()):
        raise RuntimeError("Incorrect cohort size")
    all_rows = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == learner)]
            for method, fields in (("b2_hazard", FEATURES_B2),
                                   ("oeh_hazard", OEH_FEATURES)):
                for cohort, frame in queries.items():
                    qu = frame[(frame.panel == panel) & (frame.algorithm == learner)]
                    probabilities = fit_hazards(
                        tr[list(fields)].to_numpy(dtype=float),
                        tr.event.to_numpy(dtype=int),
                        qu[list(fields)].to_numpy(dtype=float))
                    brier, logloss = score_probabilities(
                        probabilities, qu.event.to_numpy(dtype=int))
                    for index, (_, item) in enumerate(qu.iterrows()):
                        all_rows.append({**{key: item[key] for key in KEYS},
                                         "cohort": cohort, "event": int(item.event),
                                         "method": method, "brier": float(brier[index]),
                                         "logloss": float(logloss[index]),
                                         **{name: float(probabilities[index, category])
                                            for category, name in enumerate(PROBABILITY_COLUMNS)}})
            print(f"Discrete-hazard development: {panel}/{learner}", flush=True)
    detail = pd.DataFrame(all_rows)
    if len(detail) != 3600:
        raise RuntimeError("Incomplete discrete-hazard predictions")
    scores = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_60_mdp_cohorts_not_confirmation",
        "candidate": "six_conditional_discrete_hazards_same_fixed_binary_tree_settings",
        "promoted": False,
        "feature_sets": {"b2_hazard": FEATURES_B2, "oeh_hazard": OEH_FEATURES},
        "fit": {"params": PARAMS, "rounds": ROUNDS,
                "at_risk_training": "event >= each budget index"},
        "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                         "second_result": sha256(result_path),
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
