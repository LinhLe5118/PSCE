"""Explore direct seven-class difficulty forecasts on already-opened cohorts.

All predictions are exploratory. Training uses historical E1 only; no new
evaluation instances are generated or opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_oeh_difficulty import OEH_MECHANISM, score_probabilities
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, P, ROOT, sha256


QUERIES = {
    "first_open": (
        ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv",
        ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv",
    ),
    "second_open": (
        ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/labels.csv",
        ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv",
    ),
}
OUTPUT = ROOT / "artifacts/oeh_multiclass_development_2026-10-01"
FIELDS = {
    "b2": FEATURES_B2,
    "oeh": OEH_MECHANISM,
    "b2_plus_oeh": tuple(dict.fromkeys((*FEATURES_B2, *OEH_MECHANISM))),
}
PARAMS = {
    "objective": "multi:softprob", "num_class": 7, "tree_method": "hist",
    "learning_rate": .03, "max_depth": 2, "min_child_weight": 12,
    "lambda": 3, "seed": 20261001, "nthread": 1, "verbosity": 0,
}
ROUNDS = 100


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    train = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                         validate="one_to_one")
    queries = {
        name: pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS,
                                       validate="one_to_one")
        for name, (labels, features) in QUERIES.items()
    }
    if len(train) != 1800 or any(len(frame) != 900 for frame in queries.values()):
        raise RuntimeError("Incomplete E1 or development labels")
    rows = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == learner)]
            for method, fields in FIELDS.items():
                matrix = xgb.DMatrix(source[list(fields)].to_numpy(dtype=float),
                                     label=source.event.to_numpy(dtype=int))
                model = xgb.train(PARAMS, matrix, num_boost_round=ROUNDS)
                for cohort, frame in queries.items():
                    query = frame[(frame.panel == panel) & (frame.algorithm == learner)]
                    probs = model.predict(xgb.DMatrix(query[list(fields)].to_numpy(dtype=float)))
                    if probs.shape != (len(query), 7):
                        raise RuntimeError("Invalid multiclass prediction shape")
                    brier, logloss = score_probabilities(probs, query.event.to_numpy(dtype=int))
                    scored = query[KEYS + ["panel", "event"]].copy()
                    scored["cohort"] = cohort
                    scored["method"] = method
                    scored["brier"] = brier
                    scored["logloss"] = logloss
                    for i, col in enumerate(P):
                        scored[col] = probs[:, i]
                    rows.append(scored)
            print(f"Fitted multiclass: {panel}/{learner}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "development_only_two_opened_cohorts_not_confirmation",
        "model": "xgboost_direct_seven_class", "params": PARAMS,
        "rounds": ROUNDS, "features": FIELDS,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in (E1_LABELS, E1_FEATURES, *sum(QUERIES.values(), ()))},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
