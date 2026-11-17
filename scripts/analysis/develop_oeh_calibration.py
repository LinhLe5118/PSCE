"""Development-only calibration of direct OEH event probabilities.

The single smoothing coefficient is fitted on MDP-grouped E1 out-of-fold
predictions. Previously opened 60-MDP outcomes are scored only afterward.
This experiment cannot serve as independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_difficulty import METHODS, sha256
from scripts.analysis.develop_oeh_blend import FEATURES, KEYS, PROBABILITY_COLUMNS
from scripts.analysis.develop_oeh_difficulty import (
    ROOT, E1, OPEN_LABELS, _load, fit_thresholds, score_probabilities,
)

DEVELOPMENT = ROOT / "artifacts/oeh_difficulty_blend_development_2026-10-01"
OUTPUT = ROOT / "artifacts/oeh_difficulty_calibration_development_2026-10-01"


def smooth(probs, prior, coefficient):
    """Mix a forecast with its training-stratum event frequency."""
    import numpy as np

    p = np.asarray(probs, dtype=float)
    q = np.asarray(prior, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 7 or q.shape != p.shape
            or not 0 <= coefficient <= 1 or not np.isfinite(p).all()
            or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any()
            or not np.allclose(p.sum(axis=1), 1.)
            or not np.allclose(q.sum(axis=1), 1.)):
        raise ValueError("Invalid calibration probabilities")
    return (1. - coefficient) * p + coefficient * q


def choose_coefficient(probs, prior, event):
    """One globally fitted log-score coefficient, including both boundaries."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    y = np.asarray(event, dtype=int)
    p = np.asarray(probs, dtype=float)
    q = np.asarray(prior, dtype=float)
    if p.shape != q.shape or p.shape != (len(y), 7) or (y < 0).any() or (y > 6).any():
        raise ValueError("Invalid calibration training rows")

    def loss(value):
        z = smooth(p, q, value)[np.arange(len(y)), y]
        return float(-np.log(np.maximum(z, 1e-12)).mean())

    optimum = minimize_scalar(loss, bounds=(0., 1.), method="bounded",
                              options={"xatol": 1e-8})
    return min(((0., loss(0.)), (1., loss(1.)),
                (float(optimum.x), float(optimum.fun))), key=lambda item: item[1])


def main():
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    source, query, paths = _load()
    dev_manifest = json.loads((DEVELOPMENT / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DEVELOPMENT / "predictions.csv") != dev_manifest["artifacts"]["predictions.csv"]:
        raise RuntimeError("Development predictions changed")
    direct_query = pd.read_csv(DEVELOPMENT / "predictions.csv")
    direct_query = direct_query[direct_query.method == "oeh_direct"]
    direct_query = direct_query[KEYS + PROBABILITY_COLUMNS]
    source = source.sort_values(KEYS).reset_index(drop=True)
    query = query.sort_values(KEYS).reset_index(drop=True)
    aligned = query[KEYS].merge(direct_query, on=KEYS, validate="one_to_one")
    if len(aligned) != len(query) or not aligned[KEYS].equals(query[KEYS]):
        raise RuntimeError("Development forecast identity mismatch")

    oof = np.full((len(source), 7), np.nan)
    oof_prior = np.full((len(source), 7), np.nan)
    query_prior = np.full((len(query), 7), np.nan)
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            train = np.flatnonzero((source.panel == panel) & (source.algorithm == algorithm))
            test = np.flatnonzero((query.panel == panel) & (query.algorithm == algorithm))
            for fold in range(5):
                fit = train[source.iloc[train].fold.to_numpy() != fold]
                val = train[source.iloc[train].fold.to_numpy() == fold]
                if (not len(fit) or not len(val)
                        or set(source.iloc[fit].mdp_id) & set(source.iloc[val].mdp_id)):
                    raise RuntimeError("MDP-grouped fold mismatch")
                oof[val] = fit_thresholds(source.iloc[fit][list(FEATURES)].to_numpy(),
                                          source.iloc[fit].event.to_numpy(),
                                          source.iloc[val][list(FEATURES)].to_numpy())
                counts = np.bincount(source.iloc[fit].event.to_numpy(dtype=int), minlength=7)
                oof_prior[val] = counts / counts.sum()
            counts = np.bincount(source.iloc[train].event.to_numpy(dtype=int), minlength=7)
            query_prior[test] = counts / counts.sum()
            print(f"OOF calibration: {panel}/{algorithm}", flush=True)
    if (not np.isfinite(oof).all() or not np.isfinite(oof_prior).all()
            or not np.isfinite(query_prior).all()):
        raise RuntimeError("Incomplete grouped forecasts or priors")
    coefficient, oof_logloss = choose_coefficient(
        oof, oof_prior, source.event.to_numpy(dtype=int))
    raw = aligned[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    calibrated = smooth(raw, query_prior, coefficient)
    brier, logloss = score_probabilities(calibrated, query.event.to_numpy(dtype=int))
    detail = query[KEYS + ["event"]].copy()
    detail["method"] = "oeh_direct_calibrated"
    detail["brier"] = brier
    detail["logloss"] = logloss
    for index, name in enumerate(PROBABILITY_COLUMNS):
        detail[name] = calibrated[:, index]
    summary = detail.groupby("panel", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    args.output.mkdir(parents=True)
    detail.to_csv(args.output / "predictions.csv", index=False)
    summary.to_csv(args.output / "scores.csv", index=False)
    manifest = {
        "status": "development_only_open_60_mdp_not_independent_confirmation",
        "candidate": "direct_oeh_global_oof_prior_calibration",
        "promoted": False,
        "coefficient": coefficient,
        "oof_logloss": oof_logloss,
        "coefficient_training": "E1 MDP-grouped out-of-fold log-loss only",
        "input_sha256": {**{name: sha256(path) for name, path in paths.items()},
                         "direct_predictions": sha256(DEVELOPMENT / "predictions.csv")},
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": {name: sha256(args.output / name)
                            for name in ("predictions.csv", "scores.csv")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"E1 OOF coefficient: {coefficient:.8f}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
