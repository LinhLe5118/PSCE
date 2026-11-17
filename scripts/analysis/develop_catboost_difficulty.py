"""Exploratory CatBoost categorical OEH forecast on three opened cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, P, ROOT, sha256
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair


OUTPUT = ROOT / "artifacts/oeh_catboost_development_2026-10-01"
PARAMS = dict(iterations=300, depth=4, learning_rate=.03, l2_leaf_reg=5.,
              loss_function="MultiClass", random_seed=20261001,
              thread_count=4, verbose=False, allow_writing_files=False)


def fit(train: pd.DataFrame, query: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    x = np.nan_to_num(train[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    q = np.nan_to_num(query[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    labels = train.event.to_numpy(dtype=int)
    model = CatBoostClassifier(**PARAMS)
    model.fit(x, labels)
    raw = model.predict_proba(q)
    result = np.full((len(q), 7), 1e-8, dtype=float)
    for col, event in enumerate(model.classes_):
        result[:, int(event)] = raw[:, col]
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("CatBoost development artifact exists")
    e1 = load_pair(E1_LABELS, E1_FEATURES)
    opened = {name: load_pair(*paths) for name, paths in OPENED.items()}
    if len(e1) != 1800 or [len(frame) for frame in opened.values()] != [900, 900, 1800]:
        raise RuntimeError("Incomplete opened development cohorts")
    rows = []
    for query_name, query in opened.items():
        source = pd.concat((e1, *(frame for name, frame in opened.items()
                                  if name != query_name)), ignore_index=True)
        for panel in ("primary", "control"):
            for learner in METHODS:
                train = source[(source.panel == panel) & (source.algorithm == learner)]
                target = query[(query.panel == panel) & (query.algorithm == learner)]
                for name, fields in (("b2_catboost", FEATURES_B2),
                                     ("oeh_catboost", OEH_FEATURES)):
                    probabilities = fit(train, target, fields)
                    brier, logloss = score_probabilities(
                        probabilities, target.event.to_numpy(dtype=int))
                    result = target[[*KEYS, "panel", "event"]].copy()
                    result["cohort"] = query_name
                    result["method"] = name
                    result["integrated_brier"] = brier
                    result["seven_class_log_loss"] = logloss
                    for index, col in enumerate(P):
                        result[col] = probabilities[:, index]
                    rows.append(result)
            print(f"CatBoost development: {query_name}/{panel}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "params": PARAMS,
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
