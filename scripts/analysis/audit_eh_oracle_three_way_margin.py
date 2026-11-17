"""Exploratory paired 5% margin audit of architecture-matched OEH/EH stacks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01"


def main() -> None:
    output = ARTIFACT / "development_decision_audit.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    raw = pd.read_csv(ARTIFACT / "scored_predictions.csv")
    raw = raw[raw.panel == "primary"]
    key = ["cohort", "mdp_id", "policy_family", "algorithm", "panel", "event"]
    baseline = raw[raw.method == "b2_plus_eh_oracle_three_way"]
    candidate = raw[raw.method == "oeh_three_way"]
    paired = baseline[key + ["brier", "logloss"]].merge(
        candidate[key + ["brier", "logloss"]], on=key,
        suffixes=("_eh", "_oeh"), validate="one_to_one")
    if len(paired) != 1260 or paired.mdp_id.nunique() != 84:
        raise RuntimeError("Expected 84 stochastic MDPs and 1260 pairs")
    by_mdp = paired.groupby(["cohort", "mdp_id"], as_index=False).agg(**{
        f"{metric}_{method}": (f"{metric}_{method}", "mean")
        for metric in ("brier", "logloss") for method in ("eh", "oeh")
    })
    effect = np.column_stack([.95 * by_mdp[f"{m}_eh"].to_numpy()
                              - by_mdp[f"{m}_oeh"].to_numpy()
                              for m in ("brier", "logloss")])
    mean = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(len(by_mdp))
    draws = np.random.default_rng(20261001).integers(
        0, len(by_mdp), size=(10000, len(by_mdp)))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - mean) / se,
                                        axis=1), .95))
    lower = mean - critical * se
    improvement_mean = mean + .05 * np.array([
        by_mdp[f"{metric}_eh"].mean() for metric in ("brier", "logloss")])
    improvement_lower = lower + .05 * np.array([
        by_mdp[f"{metric}_eh"].mean() for metric in ("brier", "logloss")])
    gains = np.array([1 - by_mdp[f"{m}_oeh"].mean() / by_mdp[f"{m}_eh"].mean()
                      for m in ("brier", "logloss")])
    result = {
        "status": "exploratory_two_opened_cohorts_not_confirmation",
        "scope": "same three-component forecasting architecture; EH uses full true MDP",
        "mdps": len(by_mdp), "forecast_pairs": len(paired),
        "required_relative_margin": .05,
        "simultaneous_one_sided_confidence": .95,
        "metrics": {metric: {
            "eh_mean": float(by_mdp[f"{metric}_eh"].mean()),
            "oeh_mean": float(by_mdp[f"{metric}_oeh"].mean()),
            "relative_gain": float(1 - by_mdp[f"{metric}_oeh"].mean() /
                                   by_mdp[f"{metric}_eh"].mean()),
            "mean_excess_over_5pct": float(mean[i]),
            "simultaneous_lower": float(lower[i]),
            "mean_absolute_improvement": float(improvement_mean[i]),
            "simultaneous_absolute_improvement_lower": float(improvement_lower[i]),
        } for i, metric in enumerate(("brier", "logloss"))},
        "development_decision_rule": "Both point relative gains >=5% and both simultaneous 95% lower bounds for absolute improvement >0",
        "development_margin_gate": "GO" if (gains >= .05).all() and (improvement_lower > 0).all() else "NO-GO",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
