"""Development-only OOF-free geometric forecast stack on two opened cohorts.

Fits a single global OEH-vs-matched-B2 weight on the first opened cohort;
the second opened cohort is development feedback, never confirmation.
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
OUTPUT = ROOT / "artifacts/oeh_difficulty_geometric_stack_development_2026-10-01"
KEYS = ["cohort", "panel", "mdp_id", "policy_family", "algorithm", "event"]
P = [f"p_{i}" for i in range(7)]


def geometric(b2: np.ndarray, oeh: np.ndarray, weight: float) -> np.ndarray:
    logp = (1. - weight) * np.log(np.clip(b2, 1e-12, 1.))
    logp += weight * np.log(np.clip(oeh, 1e-12, 1.))
    logp -= logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def get_pair(raw: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    subset = raw[(raw.cohort == cohort) & (raw.panel == "primary")]
    b2 = subset[subset.method == "b2_three_way"].sort_values(KEYS).reset_index(drop=True)
    oeh = subset[subset.method == "oeh_three_way"].sort_values(KEYS).reset_index(drop=True)
    if len(b2) != 630 or not b2[KEYS].equals(oeh[KEYS]):
        raise RuntimeError("Opened cohorts are incomplete or misaligned")
    return b2[KEYS], b2[P].to_numpy(dtype=float), oeh[P].to_numpy(dtype=float)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace artifact: {OUTPUT}")
    raw = pd.read_csv(SOURCE)
    if len(raw) != 3600 or raw.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("Source forecast count or identity mismatch")
    first, b2_first, oeh_first = get_pair(raw, "first_open")
    second, b2_second, oeh_second = get_pair(raw, "second_open")
    y_first = first.event.to_numpy(dtype=int)
    y_second = second.event.to_numpy(dtype=int)

    def loss(weight: float) -> float:
        return float(score_probabilities(geometric(b2_first, oeh_first, weight), y_first)[1].mean())

    fit = minimize_scalar(loss, bounds=(0., 2.), method="bounded",
                          options={"xatol": 1e-6})
    candidates = [0., 1., 2.] + ([float(fit.x)] if fit.success else [])
    weight = min(candidates, key=loss)
    rows = []
    for method, probs in (("b2_three_way", b2_second),
                          ("oeh_three_way", oeh_second),
                          ("geometric_stack", geometric(b2_second, oeh_second, weight))):
        brier, logloss = score_probabilities(probs, y_second)
        frame = second.copy()
        frame["method"] = method
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
        "fit_cohort": "first_open_stochastic_primary",
        "score_cohort": "second_open_stochastic_primary",
        "oeh_weight": weight,
        "first_open_logloss": loss(weight),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
