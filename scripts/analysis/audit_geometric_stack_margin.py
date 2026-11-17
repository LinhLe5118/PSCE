"""Exploratory paired MDP margin audit for the opened-cohort stack."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/oeh_difficulty_geometric_stack_development_2026-10-01"
OUT = ARTIFACT / "paired_margin_audit.json"
KEYS = ["mdp_id", "policy_family", "algorithm", "panel", "event"]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    raw = pd.read_csv(ARTIFACT / "scored_predictions.csv")
    b2 = raw[raw.method == "b2_three_way"].drop(columns="method")
    oeh = raw[raw.method == "geometric_stack"].drop(columns="method")
    paired = b2.merge(oeh, on=KEYS, suffixes=("_b2", "_oeh"), validate="one_to_one")
    if len(paired) != 630 or paired.mdp_id.nunique() != 42:
        raise RuntimeError("Incomplete paired stochastic cohort")
    mdp = paired.groupby("mdp_id", as_index=False).agg(
        brier_b2=("brier_b2", "mean"), brier_oeh=("brier_oeh", "mean"),
        logloss_b2=("logloss_b2", "mean"), logloss_oeh=("logloss_oeh", "mean"))
    effects = np.column_stack([
        0.95 * mdp.brier_b2.to_numpy() - mdp.brier_oeh.to_numpy(),
        0.95 * mdp.logloss_b2.to_numpy() - mdp.logloss_oeh.to_numpy(),
    ])
    center = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(len(mdp))
    if (se <= 0).any():
        raise RuntimeError("Degenerate MDP effects")
    rng = np.random.default_rng(20261001)
    draw = rng.integers(0, len(mdp), size=(2000, len(mdp)))
    sampled = effects[draw].mean(axis=1)
    critical = float(np.quantile(np.max((sampled - center) / se, axis=1), .95))
    lower = center - critical * se
    report = {
        "status": "exploratory_on_opened_development_cohort_not_confirmation",
        "mdps": len(mdp), "draws": 2000, "seed": 20261001,
        "margin": "candidate must beat matched B2 by at least 5% in both scores",
        "simultaneous_critical_value": critical,
        "metrics": {
            name: {"mean_margin_excess": float(center[i]),
                   "simultaneous_one_sided_lower_95": float(lower[i]),
                   "relative_gain": float(1 - mdp[f"{name}_oeh"].mean() /
                                          mdp[f"{name}_b2"].mean())}
            for i, name in enumerate(("brier", "logloss"))
        },
        "development_margin_gate": "GO" if (lower > 0).all() else "NO-GO",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
