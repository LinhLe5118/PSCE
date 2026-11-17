"""Exploratory cross-cohort OEH meta-training expansion on opened data only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, P, ROOT, sha256
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities


FIRST = (ROOT / "artifacts/independent_difficulty_2026-09-30/labels.csv",
         ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv")
SECOND = (ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/labels.csv",
          ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv")
BASE_MANIFEST = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/manifest.json"
BASE_PREDICTIONS = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_pooled_meta_training_development_2026-10-01"


def joined(pair: tuple[Path, Path]) -> pd.DataFrame:
    labels, features = pair
    result = pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS,
                                      validate="one_to_one")
    if len(result) != 900 or result.mdp_id.nunique() != 60:
        raise RuntimeError(f"Opened cohort incomplete: {labels}")
    return result


def forecast(train: pd.DataFrame, query: pd.DataFrame, panel: str,
             learner: str, fields: tuple[str, ...], weights: np.ndarray) -> np.ndarray:
    source = train[(train.panel == panel) & (train.algorithm == learner)]
    target = query[(query.panel == panel) & (query.algorithm == learner)]
    if len(source) != (630 if panel == "primary" else 270) or len(target) != (210 if panel == "primary" else 90):
        raise RuntimeError("Wrong expanded meta-training stratum")
    if set(source.mdp_id) & set(target.mdp_id):
        raise RuntimeError("Training/query MDP overlap")
    x = np.nan_to_num(source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    q = np.nan_to_num(target[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
    y = source.event.to_numpy(dtype=int)
    parts = np.asarray((aft_probabilities(x, y, q),
                        fit_thresholds(x, y, q), fit_multiclass(x, y, q)))
    return np.einsum("i,ijk->jk", weights, parts)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    e1 = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                      validate="one_to_one")
    first, second = joined(FIRST), joined(SECOND)
    if (len(e1) != 1800 or set(e1.mdp_id) & set(first.mdp_id)
            or set(e1.mdp_id) & set(second.mdp_id)
            or set(first.mdp_id) & set(second.mdp_id)):
        raise RuntimeError("Meta-training cohort identities overlap")
    weights = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))["weights"]
    methods = (("b2_pooled_meta", FEATURES_B2, "b2_three_way"),
               ("oeh_pooled_meta", OEH_FEATURES, "oeh_three_way"))
    rows = []
    for cohort, additional, target in (("second_open", first, second),
                                       ("first_open", second, first)):
        source = pd.concat((e1, additional), ignore_index=True)
        for panel in ("primary", "control"):
            for learner in METHODS:
                selected = target[(target.panel == panel) & (target.algorithm == learner)]
                for name, fields, weight_name in methods:
                    p = forecast(source, target, panel, learner, fields,
                                 np.asarray(weights[weight_name]["aft_direct_multi"], dtype=float))
                    brier, logloss = score_probabilities(p, selected.event.to_numpy(dtype=int))
                    scored = selected[KEYS + ["panel", "event"]].copy()
                    scored["cohort"] = cohort
                    scored["method"] = name
                    scored["brier"] = brier
                    scored["logloss"] = logloss
                    for i, col in enumerate(P):
                        scored[col] = p[:, i]
                    rows.append(scored)
            print(f"Expanded meta-training: {cohort}/{panel}", flush=True)
    scored = pd.concat(rows, ignore_index=True)
    if len(scored) != 3600 or scored.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("Incomplete expanded meta-training forecasts")
    summary = scored.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_two_opened_cohorts_cross_fit_not_independent_confirmation",
        "training": "historical E1 plus the other opened 60-MDP cohort; no query MDP overlaps",
        "weights": weights,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (E1_LABELS, E1_FEATURES, *FIRST, *SECOND,
                                       BASE_MANIFEST, BASE_PREDICTIONS)},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
