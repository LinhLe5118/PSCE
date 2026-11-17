"""Development-only support-conditioned blend of matched B2/OEH forecasts.

The gate uses only pilot coverage, held-out support, and OEH pilot risk.
Coefficients are fit on the first opened stochastic cohort; the second
opened cohort supplies development feedback, not confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from scripts.analysis.develop_oeh_difficulty import score_probabilities


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"
FEATURES = {
    "first_open": ROOT / "artifacts/independent_validation_2026-09-30/pilot_features.csv",
    "second_open": ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01/pilot_features.csv",
}
OUTPUT = ROOT / "artifacts/oeh_difficulty_support_gate_development_2026-10-01"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
GATE_FEATURES = ("observed_sa_fraction", "lcoeh4_validation_support_mass",
                 "oeh3_risk_at_pilot")


def pair(raw: pd.DataFrame, cohort: str):
    selected = raw[(raw.cohort == cohort) & (raw.panel == "primary")]
    b2 = selected[selected.method == "b2_three_way"].sort_values(KEYS).reset_index(drop=True)
    oeh = selected[selected.method == "oeh_three_way"].sort_values(KEYS).reset_index(drop=True)
    if len(b2) != 630 or not b2[KEYS + ["event"]].equals(oeh[KEYS + ["event"]]):
        raise RuntimeError("Forecasts missing or misaligned")
    features = pd.read_csv(FEATURES[cohort])[KEYS + list(GATE_FEATURES)]
    joined = b2[KEYS].merge(features, on=KEYS, validate="one_to_one")
    if len(joined) != len(b2) or not joined[KEYS].equals(b2[KEYS]):
        raise RuntimeError("Pilot feature identity mismatch")
    x = joined[list(GATE_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError("Invalid pilot gate features")
    return b2, b2[P].to_numpy(dtype=float), oeh[P].to_numpy(dtype=float), x


def mixture(b2, oeh, x, beta):
    weight = expit(beta[0] + x @ beta[1:])[:, None]
    return (1. - weight) * b2 + weight * oeh


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {OUTPUT}")
    raw = pd.read_csv(SOURCE)
    if len(raw) != 3600 or raw.duplicated(KEYS + ["cohort", "method"]).any():
        raise RuntimeError("Invalid saved three-way forecasts")
    first, b2_first, oeh_first, x_first = pair(raw, "first_open")
    second, b2_second, oeh_second, x_second = pair(raw, "second_open")
    mean = x_first.mean(axis=0)
    scale = x_first.std(axis=0)
    scale[scale < 1e-8] = 1.
    x_first = (x_first - mean) / scale
    x_second = (x_second - mean) / scale
    y = first.event.to_numpy(dtype=int)

    def objective(beta):
        logloss = score_probabilities(mixture(b2_first, oeh_first, x_first, beta), y)[1]
        return float(logloss.mean() + .01 * np.sum(beta[1:] ** 2))

    fit = minimize(objective, x0=np.array([1., 0., 0., 0.]), method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 500})
    if not np.isfinite(fit.x).all():
        raise RuntimeError("Gate fit is nonfinite")
    rows = []
    for name, probabilities in (("b2_three_way", b2_second),
                                ("oeh_three_way", oeh_second),
                                ("oeh_support_gate", mixture(b2_second, oeh_second,
                                                             x_second, fit.x))):
        brier, logloss = score_probabilities(probabilities, second.event.to_numpy(dtype=int))
        frame = second[KEYS + ["event"]].copy()
        frame["method"] = name
        frame["brier"] = brier
        frame["logloss"] = logloss
        rows.append(frame)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("method", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    manifest = {
        "status": "development_only_two_opened_cohorts_not_confirmation",
        "features": GATE_FEATURES, "regularization": .01,
        "fit_cohort": "first_open_stochastic_primary",
        "score_cohort": "second_open_stochastic_primary",
        "beta": fit.x.tolist(), "optimizer_success": bool(fit.success),
        "fit_objective": objective(fit.x),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "feature_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                           for key, path in FEATURES.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"beta": manifest["beta"], "fit_objective": manifest["fit_objective"]}, indent=2))


if __name__ == "__main__":
    main()
