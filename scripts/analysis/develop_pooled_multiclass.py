"""Development-only learner-pooled categorical difficulty forecaster.

The same fixed XGBoost settings are used for B2 and OEH features. Learner
identity is represented by a three-column indicator, allowing rare event
categories to share statistical strength without a family-specific rule.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.confirm_oeh_difficulty import OUTPUT as CONFIRMATION
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES, KEYS, PROBABILITY_COLUMNS
from scripts.analysis.develop_oeh_difficulty import ROOT, _load, score_probabilities

OUTPUT = ROOT / "artifacts/oeh_difficulty_pooled_multiclass_development_2026-10-01"


def design(frame, fields):
    import numpy as np

    if set(frame.algorithm) - set(METHODS):
        raise ValueError("Unknown learner identity")
    raw = frame[list(fields)].to_numpy(dtype=float)
    learner = np.column_stack([(frame.algorithm == method).to_numpy(dtype=float)
                               for method in METHODS])
    return np.column_stack((raw, learner))


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
        raise RuntimeError("Unexpected dataset size")
    all_rows = []
    for panel in ("primary", "control"):
        tr = train[train.panel == panel]
        for method, fields in (("b2_pooled_multiclass", FEATURES_B2),
                               ("oeh_pooled_multiclass", OEH_FEATURES)):
            for cohort, frame in queries.items():
                qu = frame[frame.panel == panel]
                if len(tr) != (1260 if panel == "primary" else 540) or len(qu) != (630 if panel == "primary" else 270):
                    raise RuntimeError("Panel row count mismatch")
                probabilities = fit_multiclass(design(tr, fields),
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
        print(f"Pooled multiclass: {panel}", flush=True)
    detail = pd.DataFrame(all_rows)
    if len(detail) != 3600:
        raise RuntimeError("Incomplete pooled forecast panel")
    scores = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_60_mdp_cohorts_not_confirmation",
        "candidate": "learner_pooled_multiclass_same_tree_settings",
        "promoted": False,
        "feature_sets": {"b2_pooled_multiclass": FEATURES_B2,
                         "oeh_pooled_multiclass": OEH_FEATURES},
        "learner_encoding": {name: index for index, name in enumerate(METHODS)},
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
