"""Bounded development audit of multiclass tree capacity for matched forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import PARAMS, fit_thresholds, score_probabilities
from scripts.analysis.run_oeh_120_successor import FIRST_OUTPUT


OUTPUT = ROOT / "artifacts/oeh_tree_capacity_development_2026-10-01"
VARIANTS = {
    "shallow_300": (2, 10, 300),
    "depth3_200": (3, 10, 200),
    "depth3_150": (3, 5, 150),
}


def matrix(frame: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    return np.nan_to_num(frame[list(fields)].to_numpy(dtype=float),
                         nan=0., posinf=1e6, neginf=-1e6)


def multiclass(x: np.ndarray, y: np.ndarray, q: np.ndarray,
               config: tuple[int, int, int]) -> np.ndarray:
    depth, child, rounds = config
    params = {**PARAMS, "objective": "multi:softprob", "eval_metric": "mlogloss",
              "num_class": 7, "max_depth": depth, "min_child_weight": child}
    model = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=rounds)
    p = model.predict(xgb.DMatrix(q))
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Tree-capacity development artifact exists")
    e1 = with_context(load_pair(E1_LABELS, E1_FEATURES))
    opened = {name: with_context(load_pair(*pair)) for name, pair in OPENED.items()}
    weights = json.loads((FIRST_OUTPUT / "protocol.json").read_text(encoding="utf-8"))["weights"]
    rows = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            y = source.event.to_numpy(dtype=int)
            for name, fields, weight_name in (
                ("b2", FEATURES_B2, "b2_three_way"),
                ("oeh_cross", (*OEH_FEATURES, *CONTEXT), "oeh_three_way"),
            ):
                x, q = matrix(source, fields), matrix(target, fields)
                aft = aft_probabilities(x, y, q)
                direct = fit_thresholds(x, y, q)
                for variant, config in VARIANTS.items():
                    multi = multiclass(x, y, q, config)
                    p = np.einsum("i,ijk->jk", weights[weight_name],
                                  np.asarray((aft, direct, multi)))
                    brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                    result = target[[*KEYS, "event"]].copy()
                    result["cohort"] = query_name
                    result["method"] = name
                    result["variant"] = variant
                    result["integrated_brier"] = brier
                    result["seven_class_log_loss"] = logloss
                    rows.append(result)
        print(f"Capacity comparison: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "method", "variant"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "variants_depth_child_rounds": VARIANTS,
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
