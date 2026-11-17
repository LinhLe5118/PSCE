"""Exploratory CatBoost OEH with cross-learner pilot context."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_catboost_difficulty import PARAMS, fit
from scripts.analysis.develop_cross_learner_context import CONTEXT, with_context
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import score_probabilities


OUTPUT = ROOT / "artifacts/oeh_catboost_cross_context_development_2026-10-01"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("CatBoost cross-context artifact exists")
    e1 = with_context(load_pair(E1_LABELS, E1_FEATURES))
    opened = {name: with_context(load_pair(*pair)) for name, pair in OPENED.items()}
    rows = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        if set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Cross-cohort leak")
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            for name, fields in (("oeh_catboost_plain", OEH_FEATURES),
                                 ("oeh_catboost_cross", (*OEH_FEATURES, *CONTEXT))):
                p = fit(source, target, fields)
                brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                result = target[[*KEYS, "event"]].copy()
                result["cohort"] = query_name
                result["method"] = name
                result["integrated_brier"] = brier
                result["seven_class_log_loss"] = logloss
                rows.append(result)
        print(f"CatBoost cross context: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "params": PARAMS, "context_features": CONTEXT,
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
