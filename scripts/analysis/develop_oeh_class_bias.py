"""Exploratory class-bias calibration of OEH and B2 on opened MDPs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scripts.analysis.develop_b2_eh_oracle_aft import ROOT, sha256
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.develop_oeh_geometric_stack import SOURCE, get_pair, geometric


OUTPUT = ROOT / "artifacts/oeh_class_bias_development_2026-10-01"
RIDGE = .002


def calibrated(b2: np.ndarray, oeh: np.ndarray | None, theta: np.ndarray) -> np.ndarray:
    p = b2 if oeh is None else geometric(b2, oeh, float(theta[0]))
    logits = np.log(np.clip(p, 1e-12, 1.)) + np.r_[theta[1:], 0.][None, :]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def fit(b2: np.ndarray, oeh: np.ndarray | None, y: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(theta: np.ndarray) -> float:
        p = calibrated(b2, oeh, theta)
        loss = score_probabilities(p, y)[1].mean()
        return float(loss + RIDGE * np.sum(theta[1:] ** 2))

    starts = [np.r_[weight, np.zeros(6)] for weight in (0., .8, 1.)]
    results = [minimize(objective, start, method="L-BFGS-B",
                        bounds=[(0., 2.)] + [(-2., 2.)] * 6) for start in starts]
    good = [result for result in results if result.success]
    if not good:
        raise RuntimeError("Class-bias calibration failed")
    best = min(good, key=lambda result: result.fun)
    return best.x, float(score_probabilities(calibrated(b2, oeh, best.x), y)[1].mean())


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    raw = pd.read_csv(SOURCE)
    first, b2_first, oeh_first = get_pair(raw, "first_open")
    second, b2_second, oeh_second = get_pair(raw, "second_open")
    b2_fit, b2_loss = fit(b2_first, None, first.event.to_numpy(dtype=int))
    oeh_fit, oeh_loss = fit(b2_first, oeh_first, first.event.to_numpy(dtype=int))
    forecasts = {
        "b2_class_bias": calibrated(b2_second, None, b2_fit),
        "oeh_class_bias": calibrated(b2_second, oeh_second, oeh_fit),
    }
    rows = []
    for name, p in forecasts.items():
        brier, logloss = score_probabilities(p, second.event.to_numpy(dtype=int))
        scored = second.copy()
        scored["method"] = name
        scored["brier"] = brier
        scored["logloss"] = logloss
        rows.append(scored)
    scored = pd.concat(rows, ignore_index=True)
    summary = scored.groupby("method", as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "development_only_first_fit_second_scored_not_confirmation",
        "ridge": RIDGE,
        "first_open_fit": {
            "b2_theta": b2_fit.tolist(), "b2_logloss": b2_loss,
            "oeh_theta": oeh_fit.tolist(), "oeh_logloss": oeh_loss,
        },
        "source_sha256": sha256(SOURCE), "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"First-open logloss: B2={b2_loss:.6f}, OEH={oeh_loss:.6f}")


if __name__ == "__main__":
    main()
