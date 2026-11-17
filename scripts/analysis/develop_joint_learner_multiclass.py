"""Exploratory shared-learner categorical difficulty forecasts on opened MDPs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, KEYS, ROOT, sha256
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.run_oeh_120_successor import DEV, FIRST_OUTPUT


OUTPUT = ROOT / "artifacts/oeh_joint_learner_multiclass_development_2026-10-01"
OPENED = {
    "dev_first": DEV[0][:2],
    "dev_second": DEV[1][:2],
    "failed_120": (FIRST_OUTPUT / "labels.csv", FIRST_OUTPUT / "pilot_features.csv"),
}


def load_pair(labels: Path, features: Path) -> pd.DataFrame:
    frame = pd.read_csv(labels).merge(pd.read_csv(features), on=KEYS,
                                       validate="one_to_one")
    if frame[KEYS].duplicated().any():
        raise RuntimeError("Duplicate development task")
    return frame


def numeric(frame: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    values = np.nan_to_num(frame[list(fields)].to_numpy(dtype=float),
                           nan=0., posinf=1e6, neginf=-1e6)
    learner = np.column_stack([(frame.algorithm == method).to_numpy(dtype=float)
                               for method in METHODS])
    return np.column_stack((values, learner))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Joint-learner development artifact exists")
    e1 = load_pair(E1_LABELS, E1_FEATURES)
    opened = {name: load_pair(*pair) for name, pair in OPENED.items()}
    all_ids = [set(frame.mdp_id) for frame in (e1, *opened.values())]
    if (len(e1) != 1800 or [len(frame) for frame in opened.values()] != [900, 900, 1800]
            or sum(map(len, all_ids)) != len(set().union(*all_ids))):
        raise RuntimeError("Opened meta-training cohorts overlap or are incomplete")
    scores = []
    for query_name, query in opened.items():
        train = pd.concat((e1, *(frame for name, frame in opened.items()
                                 if name != query_name)), ignore_index=True)
        for panel in ("primary", "control"):
            source = train[train.panel == panel]
            target = query[query.panel == panel]
            if len(target) != (1260 if query_name == "failed_120" and panel == "primary"
                               else 540 if query_name == "failed_120" else
                               630 if panel == "primary" else 270):
                raise RuntimeError("Wrong joint-learner query stratum")
            for method, fields in (("b2_joint_multi", FEATURES_B2),
                                   ("oeh_joint_multi", OEH_FEATURES)):
                p = fit_multiclass(numeric(source, fields),
                                   source.event.to_numpy(dtype=int), numeric(target, fields))
                brier, logloss = score_probabilities(p, target.event.to_numpy(dtype=int))
                result = target[[*KEYS, "panel", "event"]].copy()
                result["cohort"] = query_name
                result["method"] = method
                result["integrated_brier"] = brier
                result["seven_class_log_loss"] = logloss
                scores.append(result)
            print(f"Joint-learner multiclass: {query_name}/{panel}", flush=True)
    detail = pd.concat(scores, ignore_index=True)
    summary = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "scored.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_three_opened_cohorts_not_independent",
        "model": "single seven-class XGBoost model per panel, shared across three learner IDs",
        "training": "historical E1 plus the other two opened cohorts",
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (E1_LABELS, E1_FEATURES,
                                       *(p for pair in OPENED.values() for p in pair))},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("scored.csv", "summary.csv")},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
