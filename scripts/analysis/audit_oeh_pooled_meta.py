"""Exploratory paired audit of expanded OEH meta-training on opened MDPs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import ROOT


ARTIFACT = ROOT / "artifacts/oeh_pooled_meta_training_development_2026-10-01"
CURRENT = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/predictions.csv"


def main() -> None:
    output = ARTIFACT / "paired_margin_audit.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    pooled = pd.read_csv(ARTIFACT / "scored_predictions.csv")
    current = pd.read_csv(CURRENT)
    current = current[current.method == "oeh_three_way"]
    raw = pd.concat((pooled, current), ignore_index=True)
    raw = raw[raw.panel == "primary"]
    grouped = raw.groupby(["cohort", "mdp_id", "method"], as_index=False).agg(
        brier=("brier", "mean"), logloss=("logloss", "mean"))
    methods = ("b2_pooled_meta", "oeh_three_way")
    target = grouped[grouped.method == "oeh_pooled_meta"].sort_values("mdp_id")
    if len(target) != 84 or target.mdp_id.nunique() != 84:
        raise RuntimeError("Incomplete expanded OEH forecasts")
    effects, comparisons = [], {}
    for name in methods:
        other = grouped[grouped.method == name].sort_values("mdp_id")
        if not other.mdp_id.reset_index(drop=True).equals(target.mdp_id.reset_index(drop=True)):
            raise RuntimeError("MDP identity mismatch")
        comparisons[name] = {}
        for metric in ("brier", "logloss"):
            before, after = other[metric].to_numpy(), target[metric].to_numpy()
            effects.append(before - after)
            comparisons[name][metric] = {
                "baseline_mean": float(before.mean()),
                "candidate_mean": float(after.mean()),
                "relative_gain": float(1 - after.mean() / before.mean()),
                "mean_absolute_improvement": float((before - after).mean()),
            }
    effect = np.column_stack(effects)
    center = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(84)
    draws = np.random.default_rng(20261001).integers(0, 84, size=(10000, 84))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - center) / se,
                                        axis=1), .95))
    lower = center - critical * se
    for i, name in enumerate(methods):
        for j, metric in enumerate(("brier", "logloss")):
            comparisons[name][metric]["joint_simultaneous_lower_95"] = float(lower[2 * i + j])
    result = {
        "status": "exploratory_two_opened_cohorts_not_independent_confirmation",
        "stochastic_mdps": 84, "joint_contrasts": 4, "bootstrap_draws": 10000,
        "comparisons": comparisons,
        "matched_b2_development_gate": "GO" if all(
            comparisons["b2_pooled_meta"][metric]["relative_gain"] >= .05
            and comparisons["b2_pooled_meta"][metric]["joint_simultaneous_lower_95"] > 0
            for metric in ("brier", "logloss")) else "NO-GO",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
