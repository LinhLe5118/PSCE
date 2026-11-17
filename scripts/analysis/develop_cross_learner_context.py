"""Exploratory OEH forecasts with other learners' pilot diagnostics as context."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
from scripts.analysis.develop_joint_learner_multiclass import OPENED, load_pair
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.run_oeh_120_successor import FIRST_OUTPUT


OUTPUT = ROOT / "artifacts/oeh_cross_learner_context_development_2026-10-01"
SHARED = ("mdp_id", "policy_family")
LC = tuple(name for name in OEH_FEATURES if name.startswith("lcoeh4_"))
CONTEXT = tuple(f"other_{method}_{name}" for method in METHODS for name in LC)


def with_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for method in METHODS:
        context = frame[frame.algorithm == method][[*SHARED, *LC]].copy()
        if context.duplicated(list(SHARED)).any():
            raise RuntimeError("Duplicate learner context")
        context = context.rename(columns={name: f"other_{method}_{name}" for name in LC})
        result = result.merge(context, on=list(SHARED), validate="many_to_one")
    if len(result) != len(frame):
        raise RuntimeError("Missing learner pilot context")
    return result


def model(source: pd.DataFrame, target: pd.DataFrame,
          fields: tuple[str, ...], weights: list[float]) -> np.ndarray:
    x = np.nan_to_num(source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    q = np.nan_to_num(target[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    y = source.event.to_numpy(dtype=int)
    parts = np.asarray((aft_probabilities(x, y, q),
                        fit_thresholds(x, y, q), fit_multiclass(x, y, q)))
    return np.einsum("i,ijk->jk", weights, parts)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Cross-learner development artifact exists")
    e1 = with_context(load_pair(E1_LABELS, E1_FEATURES))
    opened = {name: with_context(load_pair(*pair)) for name, pair in OPENED.items()}
    weights = json.loads((FIRST_OUTPUT / "protocol.json").read_text(encoding="utf-8"))["weights"]["oeh_three_way"]
    rows = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        if set(train.mdp_id) & set(query.mdp_id):
            raise RuntimeError("Cross-cohort leak")
        for learner in METHODS:
            source = train[(train.panel == "primary") & (train.algorithm == learner)]
            target = query[(query.panel == "primary") & (query.algorithm == learner)]
            for name, fields in (("oeh_plain", OEH_FEATURES),
                                 ("oeh_cross_learner", (*OEH_FEATURES, *CONTEXT))):
                p = model(source, target, fields, weights)
                brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                result = target[[*KEYS, "panel", "event"]].copy()
                result["cohort"] = query_name
                result["method"] = name
                result["integrated_brier"] = brier
                result["seven_class_log_loss"] = logloss
                rows.append(result)
        print(f"Cross-learner context: {query_name}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby(["cohort", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "context_features": CONTEXT,
        "weights": weights,
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
