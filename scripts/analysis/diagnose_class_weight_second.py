"""Exploratory matched rare-event class weighting on the opened second suite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import EH, FIELDS, KEYS, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import PARAMS, fit_thresholds, score_probabilities
from scripts.analysis.run_oeh_120_cross_capacity import (
    OUTPUT as THIRD, SECOND_OUTPUT as SECOND, training,
)


OUTPUT = ROOT / "artifacts/oeh_class_weight_second_posthoc_2026-10-01"
MODES = [(factor, corrected) for factor in (1., 1.5, 2., 3., 4., 6.)
         for corrected in (False, True) if factor != 1. or not corrected]
WEIGHTS = (.15, .10, .75)


def weighted_multiclass(x: np.ndarray, y: np.ndarray, q: np.ndarray,
                        factor: float, corrected: bool) -> np.ndarray:
    class_weight = np.array([factor] * 6 + [1.])
    params = {**PARAMS, "objective": "multi:softprob", "eval_metric": "mlogloss",
              "num_class": 7, "max_depth": 3, "min_child_weight": 10}
    model = xgb.train(params, xgb.DMatrix(x, label=y, weight=class_weight[y]),
                      num_boost_round=200)
    p = model.predict(xgb.DMatrix(q))
    if corrected:
        p = p / class_weight[None, :]
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Blend diagnostic exists")
    protocol_path = THIRD / "protocol.json"
    doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    train = training()
    query = pd.read_csv(SECOND / "pilot_features.csv").merge(
        pd.read_csv(SECOND / "oracle_descriptors_preoutcome.csv")[["mdp_id", *EH]],
        on="mdp_id", validate="many_to_one")
    query = with_context(query)
    query = query.merge(pd.read_csv(SECOND / "labels.csv")[KEYS + ["panel", "event"]],
                        on=KEYS, validate="one_to_one")
    if len(query) != 1800 or set(train.mdp_id) & set(query.mdp_id):
        raise RuntimeError("Invalid opened cohort")
    rows = []
    field_sets = {"b2_three_way": FEATURES_B2,
                  "b2_plus_eh_oracle_three_way": FIELDS,
                  "oeh_cross_capacity": (*OEH_FEATURES, *CONTEXT)}
    for learner in METHODS:
        source = train[(train.panel == "primary") & (train.algorithm == learner)]
        target = query[(query.panel == "primary") & (query.algorithm == learner)]
        y = source.event.to_numpy(dtype=int)
        truth = target.event.to_numpy(dtype=int)
        for method, fields in field_sets.items():
            x = numeric(source) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
            q = numeric(target) if method == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                target[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
            aft = aft_probabilities(x, y, q)
            direct = fit_thresholds(x, y, q)
            for factor, corrected in MODES:
                multi = weighted_multiclass(x, y, q, factor, corrected)
                p = WEIGHTS[0] * aft + WEIGHTS[1] * direct + WEIGHTS[2] * multi
                brier, logloss = score_probabilities(p, truth)
                frame = target[[*KEYS]].copy()
                frame["method"] = method
                frame["uncensored_weight"] = factor
                frame["prior_corrected"] = corrected
                frame["integrated_brier"] = brier
                frame["seven_class_log_loss"] = logloss
                rows.append(frame)
        print(f"Blend component diagnostic: {learner}", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    summary = scores.groupby(["method", "uncensored_weight", "prior_corrected"],
                             as_index=False)[["integrated_brier", "seven_class_log_loss"]].mean()
    OUTPUT.mkdir(parents=True)
    scores.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "posthoc_grid_on_opened_second_suite_not_confirmation",
        "class_weight_modes": MODES, "common_weights": WEIGHTS,
        "third_protocol_sha256": sha256(protocol_path),
        "second_result_manifest_sha256": sha256(SECOND / "result_manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("Best posthoc class-weight mode per method:\n" + summary.loc[
        summary.groupby("method").seven_class_log_loss.idxmin()].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
