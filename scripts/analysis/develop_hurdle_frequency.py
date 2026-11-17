"""Exploratory censoring gate with a smoothed conditional budget distribution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import PARAMS, ROUNDS, score_probabilities


OUTPUT = ROOT / "artifacts/oeh_hurdle_frequency_development_2026-10-01"


def predict(source: pd.DataFrame, target: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    x = np.nan_to_num(source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    q = np.nan_to_num(target[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    y = source.event.to_numpy(dtype=int)
    success = (y < 6).astype(float)
    gate = xgb.train(PARAMS, xgb.DMatrix(x, label=success), num_boost_round=ROUNDS)
    p_success = np.clip(gate.predict(xgb.DMatrix(q)), 1e-6, 1. - 1e-6)
    conditional = np.bincount(y[y < 6], minlength=6).astype(float) + 1.
    conditional /= conditional.sum()
    result = np.column_stack((p_success[:, None] * conditional[None, :], 1. - p_success))
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Hurdle-frequency development artifact exists")
    e1 = load_pair(E1_LABELS, E1_FEATURES)
    opened = {name: load_pair(*pair) for name, pair in OPENED.items()}
    rows = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            for name, fields in (("b2_hurdle_frequency", FEATURES_B2),
                                 ("oeh_hurdle_frequency", OEH_FEATURES)):
                p = predict(source, target, fields)
                brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                result = target[[*KEYS, "event"]].copy()
                result["cohort"] = query_name
                result["method"] = name
                result["integrated_brier"] = brier
                result["seven_class_log_loss"] = logloss
                rows.append(result)
        print(f"Hurdle frequency: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "model": "learner-specific x50-success XGBoost gate, Laplace-smoothed six-budget frequency",
        "xgb_params": PARAMS, "rounds": ROUNDS,
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in (E1_LABELS, E1_FEATURES,
                                      *(p for pair in OPENED.values() for p in pair))},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
