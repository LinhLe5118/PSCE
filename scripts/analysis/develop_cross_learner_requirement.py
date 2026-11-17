"""Exploratory cross-learner OEH with censored pilot requirement coordinates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_cross_learner_context import CONTEXT, model, with_context
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.run_oeh_120_successor import FIRST_OUTPUT


OUTPUT = ROOT / "artifacts/oeh_cross_learner_requirement_development_2026-10-01"
EXTRA = ("lcoeh4_log1p_requirement", "lcoeh4_requirement_censored")
OTHER_EXTRA = tuple(f"other_{method}_{name}" for method in METHODS for name in EXTRA)


def augment(frame: pd.DataFrame) -> pd.DataFrame:
    result = with_context(frame)
    for learner in METHODS:
        other = frame[frame.algorithm == learner][["mdp_id", "policy_family", *EXTRA]].copy()
        other = other.rename(columns={name: f"other_{learner}_{name}" for name in EXTRA})
        result = result.merge(other, on=["mdp_id", "policy_family"], validate="many_to_one")
    if len(result) != len(frame):
        raise RuntimeError("Incomplete requirement context")
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Cross-learner requirement artifact exists")
    e1 = augment(load_pair(E1_LABELS, E1_FEATURES))
    opened = {name: augment(load_pair(*pair)) for name, pair in OPENED.items()}
    weights = json.loads((FIRST_OUTPUT / "protocol.json").read_text(encoding="utf-8"))["weights"]["oeh_three_way"]
    rows = []
    fields = (*OEH_FEATURES, *CONTEXT, *EXTRA, *OTHER_EXTRA)
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        if set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Cross-cohort leak")
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            p = model(source, target, fields, weights)
            brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
            result = target[[*KEYS, "panel", "event"]].copy()
            result["cohort"] = query_name
            result["integrated_brier"] = brier
            result["seven_class_log_loss"] = logloss
            rows.append(result)
        print(f"Cross-learner requirement: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("cohort", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "feature_count": len(fields), "features": fields, "weights": weights,
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
