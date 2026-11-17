"""Explore prior-smoothed OEH/B2 forecasts on two opened development cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.analysis.develop_b2_eh_oracle_aft import E1_LABELS, ROOT, sha256
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.develop_oeh_geometric_stack import SOURCE, get_pair, geometric


OUTPUT = ROOT / "artifacts/oeh_prior_calibration_development_2026-10-01"


def smooth(p: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return (1. - alpha) * p + alpha * prior[None, :]


def optimize(b2: np.ndarray, oeh: np.ndarray | None, y: np.ndarray,
             prior: np.ndarray) -> tuple[np.ndarray, float]:
    def predict(parameters: np.ndarray) -> np.ndarray:
        component = b2 if oeh is None else geometric(b2, oeh, float(parameters[0]))
        return smooth(component, prior, float(parameters[1]))

    def objective(parameters: np.ndarray) -> float:
        return float(score_probabilities(predict(parameters), y)[1].mean())

    fits = []
    for seed in ((0., 0.), (.5, .05), (1., .1), (1.5, .2)):
        fit = minimize(objective, np.array(seed), bounds=((0., 2.), (0., .4)),
                       method="L-BFGS-B")
        if fit.success:
            fits.append((fit.x, fit.fun))
    if not fits:
        raise RuntimeError("Calibration did not converge")
    return min(fits, key=lambda pair: pair[1])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    raw = pd.read_csv(SOURCE)
    first, b2_first, oeh_first = get_pair(raw, "first_open")
    second, b2_second, oeh_second = get_pair(raw, "second_open")
    e1 = pd.read_csv(E1_LABELS)
    prior = np.bincount(e1[e1.panel == "primary"].event.to_numpy(dtype=int),
                        minlength=7).astype(float)
    prior /= prior.sum()
    baseline_fit, baseline_loss = optimize(b2_first, None,
                                           first.event.to_numpy(dtype=int), prior)
    candidate_fit, candidate_loss = optimize(b2_first, oeh_first,
                                             first.event.to_numpy(dtype=int), prior)
    forecasts = {
        "b2_prior_calibrated": smooth(b2_second, prior, float(baseline_fit[1])),
        "oeh_geometric_prior_calibrated": smooth(
            geometric(b2_second, oeh_second, float(candidate_fit[0])),
            prior, float(candidate_fit[1])),
    }
    rows = []
    for method, probs in forecasts.items():
        brier, loss = score_probabilities(probs, second.event.to_numpy(dtype=int))
        frame = second.copy()
        frame["method"] = method
        frame["brier"] = brier
        frame["logloss"] = loss
        rows.append(frame)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("method", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    manifest = {
        "status": "development_only_first_fit_second_scored_not_confirmation",
        "first_open_fits": {
            "baseline": {"oeh_weight": float(baseline_fit[0]),
                         "prior_smoothing": float(baseline_fit[1]), "logloss": baseline_loss},
            "candidate": {"oeh_weight": float(candidate_fit[0]),
                          "prior_smoothing": float(candidate_fit[1]), "logloss": candidate_loss},
        },
        "prior": prior.tolist(), "source_sha256": sha256(SOURCE),
        "e1_labels_sha256": sha256(E1_LABELS), "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest["first_open_fits"], indent=2))


if __name__ == "__main__":
    main()
