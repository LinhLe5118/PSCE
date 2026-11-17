"""Paired exploratory EH-transfer margin audit across two opened development cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    ROOT / "artifacts/oeh_b2_eh_oracle_aft_development_2026-10-01/scored_predictions.csv",
    ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/scored_predictions.csv",
)
OUTPUT = ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/combined_opened_margin_audit.json"
KEYS = ["mdp_id", "policy_family", "algorithm", "panel", "event"]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    raw = pd.concat([pd.read_csv(path) for path in INPUTS], ignore_index=True)
    raw = raw[raw.panel == "primary"]
    baseline = raw[raw.method == "b2_plus_eh_oracle_aft"]
    candidate = raw[raw.method == "b2_plus_oeh_aft"]
    paired = baseline[KEYS + ["brier", "logloss"]].merge(
        candidate[KEYS + ["brier", "logloss"]], on=KEYS,
        suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(paired) != 1260 or paired.mdp_id.nunique() != 84:
        raise RuntimeError("Expected 84 distinct stochastic development MDPs")
    groups = paired.groupby("mdp_id", as_index=False).agg(**{
        f"{metric}_{method}": (f"{metric}_{method}", "mean")
        for metric in ("brier", "logloss") for method in ("baseline", "candidate")
    })
    effect = np.column_stack([
        .95 * groups[f"{metric}_baseline"].to_numpy()
        - groups[f"{metric}_candidate"].to_numpy()
        for metric in ("brier", "logloss")
    ])
    center = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(len(groups))
    rng = np.random.default_rng(20261001)
    draws = rng.integers(0, len(groups), size=(10000, len(groups)))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - center) / se,
                                        axis=1), .95))
    lower = center - critical * se
    result = {
        "status": "exploratory_combined_opened_development_not_confirmation",
        "comparison": "B2+OEH pilot-only AFT vs B2+oracle-EH AFT on offline difficulty",
        "stochastic_mdp_count": len(groups), "paired_forecasts": len(paired),
        "bootstrap_draws": len(draws), "required_relative_margin": .05,
        "simultaneous_critical_value": critical,
        "metrics": {metric: {
            "baseline_mean": float(groups[f"{metric}_baseline"].mean()),
            "candidate_mean": float(groups[f"{metric}_candidate"].mean()),
            "relative_gain": float(1 - groups[f"{metric}_candidate"].mean() /
                                   groups[f"{metric}_baseline"].mean()),
            "mean_excess_over_5pct": float(center[i]),
            "simultaneous_one_sided_lower_95": float(lower[i]),
        } for i, metric in enumerate(("brier", "logloss"))},
        "development_margin_gate": "GO" if (lower > 0).all() else "NO-GO",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
