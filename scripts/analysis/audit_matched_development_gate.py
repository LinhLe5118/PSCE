"""Joint development gate against architecture-matched B2 and oracle EH."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EH = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/scored_predictions.csv"
B2 = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/joint_development_gate.json"
KEY = ["cohort", "mdp_id", "policy_family", "algorithm", "panel", "event"]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    eh = pd.read_csv(EH)
    b2 = pd.read_csv(B2)
    combined = pd.concat([eh, b2[b2.method == "b2_three_way"]], ignore_index=True)
    combined = combined[combined.panel == "primary"]
    methods = ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_three_way")
    if set(combined.method) != set(methods) or combined.duplicated(KEY + ["method"]).any():
        raise RuntimeError("Incomplete or duplicate development forecasts")
    mdp = combined.groupby(["cohort", "mdp_id", "method"], as_index=False).agg(
        brier=("brier", "mean"), logloss=("logloss", "mean"))
    if len(mdp) != 252 or mdp.mdp_id.nunique() != 84:
        raise RuntimeError("Expected 84 stochastic MDPs x three methods")
    effect_columns = []
    report = {}
    for baseline in methods[:2]:
        report[baseline] = {}
        b = mdp[mdp.method == baseline].sort_values("mdp_id").reset_index(drop=True)
        c = mdp[mdp.method == "oeh_three_way"].sort_values("mdp_id").reset_index(drop=True)
        if not b.mdp_id.equals(c.mdp_id):
            raise RuntimeError("MDP identity mismatch")
        for metric in ("brier", "logloss"):
            base = b[metric].to_numpy()
            cand = c[metric].to_numpy()
            effect_columns.append(base - cand)
            report[baseline][metric] = {
                "baseline_mean": float(base.mean()),
                "oeh_mean": float(cand.mean()),
                "relative_gain": float(1 - cand.mean() / base.mean()),
                "absolute_improvement": float((base - cand).mean()),
            }
    effect = np.column_stack(effect_columns)
    center = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(len(mdp) / 3)
    draws = np.random.default_rng(20261001).integers(0, 84, size=(10000, 84))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - center) / se,
                                        axis=1), .95))
    lower = center - critical * se
    for index, baseline in enumerate(methods[:2]):
        for j, metric in enumerate(("brier", "logloss")):
            report[baseline][metric]["simultaneous_lower_95"] = float(lower[2 * index + j])
    gains = np.array([report[b][m]["relative_gain"]
                      for b in methods[:2] for m in ("brier", "logloss")])
    result = {
        "status": "development_only_two_opened_cohorts_not_confirmation",
        "scope": "OEH pilot-only versus matched B2 and full-MDP oracle EH on offline difficulty",
        "stochastic_mdps": 84, "bootstrap_draws": len(draws),
        "joint_contrasts": 4, "simultaneous_critical_value": critical,
        "decision_rule": "Each point relative gain >=5% and each joint simultaneous absolute-improvement lower bound >0",
        "comparisons": report,
        "development_gate": "GO" if (gains >= .05).all() and (lower > 0).all() else "NO-GO",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
