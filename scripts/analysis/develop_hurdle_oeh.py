"""Development-only two-stage censored-budget estimator.

First predict success by x50, then the first successful budget conditional on
success. This separates the frequent censored class from rare ordered events.
The architecture and fixed tree settings are identical for B2 and OEH.
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

OUTPUT = ROOT / "artifacts/oeh_difficulty_hurdle_development_2026-10-01"


def design(frame, fields):
    import numpy as np

    if set(frame.algorithm) - set(METHODS):
        raise ValueError("Unknown learner identity")
    raw = frame[list(fields)].to_numpy(dtype=float)
    identity = np.column_stack([(frame.algorithm == method).to_numpy(dtype=float)
                                for method in METHODS])
    return np.column_stack((raw, identity))


def fit_hurdle(x, event, query):
    import numpy as np
    import xgboost as xgb

    x = np.asarray(x, dtype=float)
    y = np.asarray(event, dtype=int)
    q = np.asarray(query, dtype=float)
    if (x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1]
            or len(x) != len(y) or (y < 0).any() or (y > 6).any()
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError("Invalid hurdle training data")
    success = (y < 6).astype(float)
    gate = xgb.train(PARAMS, xgb.DMatrix(x, label=success),
                     num_boost_round=ROUNDS)
    p_success = gate.predict(xgb.DMatrix(q))
    conditional = xgb.train(
        {**PARAMS, "objective": "multi:softprob", "eval_metric": "mlogloss",
         "num_class": 6},
        xgb.DMatrix(x[success == 1], label=y[success == 1]),
        num_boost_round=ROUNDS)
    p_budget = conditional.predict(xgb.DMatrix(q))
    probabilities = np.column_stack((p_success[:, None] * p_budget,
                                     1. - p_success))
    if (probabilities.shape != (len(q), 7)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.)):
        raise RuntimeError("Invalid hurdle event probabilities")
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
        tr = train[train.panel == panel]
        for method, fields in (("b2_hurdle", FEATURES_B2),
                               ("oeh_hurdle", OEH_FEATURES)):
            for cohort, frame in queries.items():
                qu = frame[frame.panel == panel]
                if len(tr) != (1260 if panel == "primary" else 540) or len(qu) != (630 if panel == "primary" else 270):
                    raise RuntimeError("Panel row count mismatch")
                probabilities = fit_hurdle(design(tr, fields),
                                           tr.event.to_numpy(dtype=int),
                                           design(qu, fields))
                brier, logloss = score_probabilities(probabilities,
                                                     qu.event.to_numpy(dtype=int))
                for index, (_, item) in enumerate(qu.iterrows()):
                    all_rows.append({**{key: item[key] for key in KEYS},
                                     "cohort": cohort, "event": int(item.event),
                                     "method": method, "brier": float(brier[index]),
                                     "logloss": float(logloss[index]),
                                     **{name: float(probabilities[index, category])
                                        for category, name in enumerate(PROBABILITY_COLUMNS)}})
        print(f"Hurdle development: {panel}", flush=True)
    detail = pd.DataFrame(all_rows)
    if len(detail) != 3600:
        raise RuntimeError("Incomplete hurdle forecast panel")
    scores = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_60_mdp_cohorts_not_confirmation",
        "candidate": "binary_success_by_x50_then_six_class_conditional_budget",
        "promoted": False,
        "feature_sets": {"b2_hurdle": FEATURES_B2, "oeh_hurdle": OEH_FEATURES},
        "fit": {"params": PARAMS, "rounds": ROUNDS,
                "learner_encoding": {name: index for index, name in enumerate(METHODS)}},
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
