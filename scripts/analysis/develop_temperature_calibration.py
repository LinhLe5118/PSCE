"""Development-only, architecture-matched temperature check on opened cohorts.

Fit one scalar temperature for each forecast using first opened stochastic
MDPs, then score the second opened cohort. Neither cohort is confirmation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.analysis.develop_oeh_difficulty import score_probabilities


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_difficulty_temperature_development_2026-10-01"
PROBABILITY_COLUMNS = [f"p_{index}" for index in range(7)]
KEYS = ["cohort", "mdp_id", "policy_family", "algorithm", "panel", "event"]


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Temperature must be finite and positive")
    logp = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, event: np.ndarray) -> float:
    def loss(log_temperature: float) -> float:
        calibrated = temperature_scale(probabilities, float(np.exp(log_temperature)))
        return float(score_probabilities(calibrated, event)[1].mean())

    fit = minimize_scalar(loss, bounds=(np.log(0.5), np.log(3.0)), method="bounded",
                          options={"xatol": 1e-6})
    if not fit.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(fit.x))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact: {OUTPUT}")
    source = pd.read_csv(SOURCE)
    if len(source) != 3600 or source.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("Incomplete or duplicated source forecasts")
    if set(source.method) != {"oeh_three_way", "b2_three_way"}:
        raise RuntimeError("Unexpected source methods")
    rows = []
    temperatures = {}
    for method in ("b2_three_way", "oeh_three_way"):
        train = source[(source.cohort == "first_open") & (source.panel == "primary")
                       & (source.method == method)].sort_values(KEYS).reset_index(drop=True)
        query = source[(source.cohort == "second_open") & (source.panel == "primary")
                       & (source.method == method)].sort_values(KEYS).reset_index(drop=True)
        if len(train) != 630 or len(query) != 630 or train.mdp_id.nunique() != 42:
            raise RuntimeError("Expected 42 stochastic MDPs in each opened cohort")
        x_train = train[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        y_train = train.event.to_numpy(dtype=int)
        temperature = fit_temperature(x_train, y_train)
        temperatures[method] = temperature
        for calibrated in (False, True):
            probs = query[PROBABILITY_COLUMNS].to_numpy(dtype=float)
            if calibrated:
                probs = temperature_scale(probs, temperature)
            brier, logloss = score_probabilities(probs, query.event.to_numpy(dtype=int))
            frame = query[KEYS].copy()
            frame["method"] = method + ("_temperature" if calibrated else "_original")
            frame["brier"] = brier
            frame["logloss"] = logloss
            rows.append(frame)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("method", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    candidate = scored[scored.method == "oeh_three_way_temperature"]
    baseline = scored[scored.method == "b2_three_way_temperature"]
    joined = baseline[KEYS + ["brier", "logloss"]].merge(
        candidate[KEYS + ["brier", "logloss"]], on=KEYS,
        suffixes=("_b2", "_oeh"), validate="one_to_one")
    mdp = joined.groupby("mdp_id", as_index=False).agg(
        brier_b2=("brier_b2", "mean"), brier_oeh=("brier_oeh", "mean"),
        logloss_b2=("logloss_b2", "mean"), logloss_oeh=("logloss_oeh", "mean"))
    relative = {
        metric: float(1 - mdp[f"{metric}_oeh"].mean() / mdp[f"{metric}_b2"].mean())
        for metric in ("brier", "logloss")
    }
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    mdp.to_csv(OUTPUT / "mdp_scores.csv", index=False)
    manifest = {
        "status": "development_only_second_open_cohort_not_confirmation",
        "description": "Symmetric scalar temperature fit on first opened primary panel",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "temperatures": temperatures,
        "relative_oeh_gain_vs_matched_b2": relative,
        "scripts_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest["relative_oeh_gain_vs_matched_b2"], indent=2))


if __name__ == "__main__":
    main()
