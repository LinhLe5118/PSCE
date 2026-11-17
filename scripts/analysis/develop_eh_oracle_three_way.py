"""Architecture-matched oracle EH transfer benchmark on opened development MDPs.

EH descriptors require the true MDP and are unavailable to an offline pilot.
This is a diagnostic transfer comparison for the offline difficulty endpoint,
not a reproduction of the original EH online sample-complexity guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
import xgboost as xgb

from scripts.analysis.compare_difficulty import GRID, METHODS, aft_interval
from scripts.analysis.develop_b2_eh_oracle_aft import (
    DEV_FEATURES, DEV_LABELS, DEV_ORACLE, E1_FEATURES, E1_LABELS,
    E1_ORACLE, EH, FIELDS, KEYS, P, ROOT, sha256,
)
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.develop_three_way_ensemble import choose_weights
from scripts.analysis.independent_difficulty import XGB_PARAMS


SECOND = ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01"
SECOND_ORACLE = ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/oracle_descriptors.csv"
OEH_THREE = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01"
IDENTITY = ["cohort", "mdp_id", "policy_family", "algorithm", "panel", "event"]


def numeric(frame: pd.DataFrame) -> np.ndarray:
    x = frame[list(FIELDS)].to_numpy(dtype=float)
    return np.nan_to_num(x, nan=0., posinf=1e6, neginf=-1e6)


def aft_probabilities(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    lower, upper = np.asarray([aft_interval(int(value)) for value in y], dtype=float).T
    matrix = xgb.DMatrix(x)
    matrix.set_float_info("label_lower_bound", lower)
    matrix.set_float_info("label_upper_bound", upper)
    model = xgb.train(XGB_PARAMS, matrix, num_boost_round=200)
    median = model.predict(xgb.DMatrix(q))
    if not np.isfinite(median).all() or (median <= 0).any():
        raise RuntimeError("Invalid oracle EH AFT forecast")
    cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
    p = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0.)
    return p / p.sum(axis=1, keepdims=True)


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    train = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                         validate="one_to_one")
    train = train.merge(pd.read_csv(E1_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    first = pd.read_csv(DEV_LABELS).merge(pd.read_csv(DEV_FEATURES), on=KEYS,
                                          validate="one_to_one")
    first = first.merge(pd.read_csv(DEV_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    second = pd.read_csv(SECOND / "labels.csv").merge(
        pd.read_csv(SECOND / "pilot_features.csv"), on=KEYS, validate="one_to_one")
    second = second.merge(pd.read_csv(SECOND_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                          validate="many_to_one")
    if len(train) != 1800 or len(first) != 900 or len(second) != 900:
        raise RuntimeError("Incomplete EH oracle data")
    return train.sort_values(KEYS).reset_index(drop=True), {
        "first_open": first.sort_values(KEYS).reset_index(drop=True),
        "second_open": second.sort_values(KEYS).reset_index(drop=True),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    train, queries = load()
    oof = {component: np.full((len(train), 7), np.nan)
           for component in ("aft", "direct", "multi")}
    full = {cohort: {component: np.full((len(frame), 7), np.nan)
                     for component in oof}
            for cohort, frame in queries.items()}
    for panel in ("primary", "control"):
        for learner in METHODS:
            tr = np.flatnonzero((train.panel == panel) & (train.algorithm == learner))
            for fold in range(5):
                fit = tr[train.iloc[tr].fold.to_numpy() != fold]
                val = tr[train.iloc[tr].fold.to_numpy() == fold]
                if set(train.iloc[fit].mdp_id) & set(train.iloc[val].mdp_id):
                    raise RuntimeError("MDP-disjoint fold violated")
                x, y, q = numeric(train.iloc[fit]), train.iloc[fit].event.to_numpy(dtype=int), numeric(train.iloc[val])
                oof["aft"][val] = aft_probabilities(x, y, q)
                oof["direct"][val] = fit_thresholds(x, y, q)
                oof["multi"][val] = fit_multiclass(x, y, q)
            x, y = numeric(train.iloc[tr]), train.iloc[tr].event.to_numpy(dtype=int)
            for cohort, frame in queries.items():
                pos = np.flatnonzero((frame.panel == panel) & (frame.algorithm == learner))
                q = numeric(frame.iloc[pos])
                full[cohort]["aft"][pos] = aft_probabilities(x, y, q)
                full[cohort]["direct"][pos] = fit_thresholds(x, y, q)
                full[cohort]["multi"][pos] = fit_multiclass(x, y, q)
            print(f"Matched EH stack fitted: {panel}/{learner}", flush=True)
    if any(not np.isfinite(array).all() for array in oof.values()):
        raise RuntimeError("Incomplete EH oracle OOF forecasts")
    weights, e1_loss = choose_weights(tuple(oof.values()), train.event.to_numpy(dtype=int))
    rows = []
    for cohort, frame in queries.items():
        components = np.asarray(tuple(full[cohort].values()))
        probs = np.einsum("i,ijk->jk", weights, components)
        brier, logloss = score_probabilities(probs, frame.event.to_numpy(dtype=int))
        scored = frame[KEYS + ["panel", "event"]].copy()
        scored["cohort"] = cohort
        scored["method"] = "b2_plus_eh_oracle_three_way"
        scored["brier"] = brier
        scored["logloss"] = logloss
        for i, col in enumerate(P):
            scored[col] = probs[:, i]
        rows.append(scored)
    eh_scored = pd.concat(rows, ignore_index=True)
    oeh_scored = pd.read_csv(OEH_THREE)
    oeh_scored = oeh_scored[oeh_scored.method == "oeh_three_way"]
    if (len(eh_scored) != 1800 or len(oeh_scored) != 1800
            or not eh_scored.sort_values(IDENTITY)[IDENTITY].reset_index(drop=True).equals(
                oeh_scored.sort_values(IDENTITY)[IDENTITY].reset_index(drop=True))):
        raise RuntimeError("EH/OEH forecast identities differ")
    scored = pd.concat([eh_scored, oeh_scored], ignore_index=True)
    summary = scored.groupby(["cohort", "panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "development_only_two_opened_cohorts_not_confirmation",
        "scope": "offline endpoint transfer; EH has true-MDP oracle features",
        "e1_oof_weights_aft_direct_multi": weights.tolist(),
        "e1_oof_logloss": e1_loss,
        "feature_set_eh": FIELDS,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (E1_LABELS, E1_FEATURES, E1_ORACLE,
                                       DEV_LABELS, DEV_FEATURES, DEV_ORACLE,
                                       SECOND / "labels.csv", SECOND / "pilot_features.csv",
                                       SECOND_ORACLE, OEH_THREE)},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"E1 OOF EH stack weights: {weights}, logloss={e1_loss:.6f}")


if __name__ == "__main__":
    main()
