"""Development-only matched three-way categorical forecasts for OEH and B2.

One global convex weight vector per feature set is learned from E1 MDP-grouped
out-of-fold log loss. Both opened 60-MDP sets are inspected only afterward.
No family-specific tuning, current-confirmation promotion, or E1 test occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS, sha256
from scripts.analysis.confirm_oeh_difficulty import OUTPUT as CONFIRMATION
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import (
    FEATURES as OEH_FEATURES, KEYS, PROBABILITY_COLUMNS,
)
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, E1, OPEN_LABELS, _load, fit_thresholds, score_probabilities,
)

OUTPUT = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01"
E1_FORECASTS = E1 / "predictions.csv"
FIRST_FORECASTS = OPEN_LABELS.parent / "predictions_preoutcome.csv"
SECOND_FORECASTS = CONFIRMATION / "predictions_preoutcome.csv"


def align(frame, path, method):
    import pandas as pd

    raw = pd.read_csv(path)
    raw = raw[raw.method == method][KEYS + PROBABILITY_COLUMNS]
    result = frame[KEYS].merge(raw, on=KEYS, validate="one_to_one")
    if len(result) != len(frame) or not result[KEYS].equals(frame[KEYS]):
        raise RuntimeError(f"Misaligned {method} forecasts at {path}")
    return result[PROBABILITY_COLUMNS].to_numpy(dtype=float)


def choose_weights(components, event):
    import numpy as np
    from scipy.optimize import minimize

    arrays = np.asarray(components, dtype=float)
    y = np.asarray(event, dtype=int)
    if (arrays.shape != (3, len(y), 7) or not np.isfinite(arrays).all()
            or (arrays < 0).any() or not np.allclose(arrays.sum(axis=2), 1.)):
        raise ValueError("Invalid OOF ensemble components")
    event_prob = arrays[:, np.arange(len(y)), y]

    def objective(weight):
        return float(-np.log(np.maximum(weight @ event_prob, 1e-12)).mean())

    fitted = minimize(objective, np.ones(3) / 3, method="SLSQP",
                      bounds=[(0., 1.)] * 3,
                      constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    candidates = [np.eye(3)[index] for index in range(3)]
    if fitted.success and np.isfinite(fitted.x).all():
        candidates.append(np.clip(fitted.x, 0, 1) / np.clip(fitted.x, 0, 1).sum())
    best = min(candidates, key=objective)
    return best, objective(best)


def main():
    import numpy as np
    import pandas as pd

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    train, first, paths = _load()
    original_manifest = json.loads((E1 / "manifest.json").read_text(encoding="utf-8"))
    first_seal = json.loads((FIRST_FORECASTS.parent / "preoutcome_seal.json").read_text(encoding="utf-8"))
    second_seal = json.loads((SECOND_FORECASTS.parent / "preoutcome_seal.json").read_text(encoding="utf-8"))
    second_result = json.loads((CONFIRMATION / "result_manifest.json").read_text(encoding="utf-8"))
    if (sha256(E1_FORECASTS) != original_manifest["output_hashes"]["predictions.csv"]
            or sha256(FIRST_FORECASTS) != first_seal["artifact_sha256"]["predictions_preoutcome.csv"]
            or sha256(SECOND_FORECASTS) != second_seal["artifact_sha256"]["predictions_preoutcome.csv"]
            or second_result["status"] != "complete"
            or sha256(CONFIRMATION / "labels.csv") != second_result["artifact_sha256"]["labels.csv"]):
        raise RuntimeError("An E1 or opened-cohort artifact changed")
    second = pd.read_csv(CONFIRMATION / "labels.csv").merge(
        pd.read_csv(CONFIRMATION / "pilot_features.csv"),
        on=["mdp_id", "policy_family", "algorithm"], validate="one_to_one")
    train = train.sort_values(KEYS).reset_index(drop=True)
    queries = {"first_open": first.sort_values(KEYS).reset_index(drop=True),
               "second_open": second.sort_values(KEYS).reset_index(drop=True)}
    if len(train) != 1800 or any(len(frame) != 900 for frame in queries.values()):
        raise RuntimeError("Incorrect E1/development row counts")
    forecasts = {name: {} for name in queries}
    weights = {}
    for method, fields, aft_name in (
        ("b2_three_way", FEATURES_B2, "xgboost_aft_b2"),
        ("oeh_three_way", OEH_FEATURES, "xgboost_aft_b2_oeh"),
    ):
        direct_oof = np.full((len(train), 7), np.nan)
        multi_oof = np.full((len(train), 7), np.nan)
        direct_query = {name: np.full((len(frame), 7), np.nan)
                        for name, frame in queries.items()}
        multi_query = {name: np.full((len(frame), 7), np.nan)
                       for name, frame in queries.items()}
        for panel in ("primary", "control"):
            for learner in METHODS:
                tr = np.flatnonzero((train.panel == panel) & (train.algorithm == learner))
                for fold in range(5):
                    fit = tr[train.iloc[tr].fold.to_numpy() != fold]
                    val = tr[train.iloc[tr].fold.to_numpy() == fold]
                    if (not len(fit) or not len(val)
                            or set(train.iloc[fit].mdp_id) & set(train.iloc[val].mdp_id)):
                        raise RuntimeError("MDP-grouped OOF fold mismatch")
                    x = train.iloc[fit][list(fields)].to_numpy(dtype=float)
                    y = train.iloc[fit].event.to_numpy(dtype=int)
                    q = train.iloc[val][list(fields)].to_numpy(dtype=float)
                    direct_oof[val] = fit_thresholds(x, y, q)
                    multi_oof[val] = fit_multiclass(x, y, q)
                x = train.iloc[tr][list(fields)].to_numpy(dtype=float)
                y = train.iloc[tr].event.to_numpy(dtype=int)
                for cohort, frame in queries.items():
                    positions = np.flatnonzero((frame.panel == panel)
                                               & (frame.algorithm == learner))
                    q = frame.iloc[positions][list(fields)].to_numpy(dtype=float)
                    direct_query[cohort][positions] = fit_thresholds(x, y, q)
                    multi_query[cohort][positions] = fit_multiclass(x, y, q)
                print(f"Three-way {method}: {panel}/{learner}", flush=True)
        if not (np.isfinite(direct_oof).all() and np.isfinite(multi_oof).all()):
            raise RuntimeError("OOF forecast incomplete")
        aft_oof = align(train, E1_FORECASTS, aft_name)
        choice, loss = choose_weights((aft_oof, direct_oof, multi_oof),
                                      train.event.to_numpy(dtype=int))
        weights[method] = {"aft_direct_multi": choice.tolist(), "e1_oof_logloss": loss}
        for cohort, frame in queries.items():
            fresh_aft = "b2_generic" if method == "b2_three_way" else "b2_plus_oeh"
            path = FIRST_FORECASTS if cohort == "first_open" else SECOND_FORECASTS
            aft_query = align(frame, path, fresh_aft)
            components = np.asarray((aft_query, direct_query[cohort], multi_query[cohort]))
            probabilities = np.einsum("i,ijk->jk", choice, components)
            brier, logloss = score_probabilities(probabilities,
                                                 frame.event.to_numpy(dtype=int))
            rows = frame[KEYS + ["event"]].copy()
            rows["cohort"] = cohort
            rows["method"] = method
            rows["brier"] = brier
            rows["logloss"] = logloss
            for index, column in enumerate(PROBABILITY_COLUMNS):
                rows[column] = probabilities[:, index]
            forecasts[cohort][method] = rows
    detail = pd.concat([frame for methods in forecasts.values()
                        for frame in methods.values()], ignore_index=True)
    if len(detail) != 3600:
        raise RuntimeError("Incomplete three-way forecast panel")
    scores = detail.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    detail.to_csv(OUTPUT / "predictions.csv", index=False)
    scores.to_csv(OUTPUT / "scores.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_60_mdp_cohorts_not_confirmation",
        "candidate": "aft_plus_direct_threshold_plus_joint_multiclass_oof_logscore_stack",
        "promoted": False, "weights": weights,
        "feature_sets": {"b2_three_way": FEATURES_B2,
                         "oeh_three_way": OEH_FEATURES},
        "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                         "e1_aft": sha256(E1_FORECASTS),
                         "first_aft": sha256(FIRST_FORECASTS),
                         "second_aft": sha256(SECOND_FORECASTS),
                         "second_labels": sha256(CONFIRMATION / "labels.csv")},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(OUTPUT / name)
                            for name in ("predictions.csv", "scores.csv")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(weights, indent=2), flush=True)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
