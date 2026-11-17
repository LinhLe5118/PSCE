"""Joint exploratory development audit for expanded OEH vs B2 and oracle EH."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import ROOT, sha256


POOLED = ROOT / "artifacts/oeh_pooled_meta_training_development_2026-10-01/scored_predictions.csv"
EH = ROOT / "artifacts/oeh_eh_oracle_pooled_meta_development_2026-10-01/scored_predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_pooled_meta_training_development_2026-10-01/joint_b2_eh_margin_audit.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    raw = pd.concat((pd.read_csv(POOLED), pd.read_csv(EH)), ignore_index=True)
    raw = raw[raw.panel == "primary"]
    by_mdp = raw.groupby(["cohort", "mdp_id", "method"], as_index=False).agg(
        brier=("brier", "mean"), logloss=("logloss", "mean"))
    if len(by_mdp) != 252 or by_mdp.mdp_id.nunique() != 84:
        raise RuntimeError("Expected 84 stochastic MDPs x three methods")
    target = by_mdp[by_mdp.method == "oeh_pooled_meta"].sort_values("mdp_id")
    methods = ("b2_pooled_meta", "b2_plus_eh_oracle_pooled_meta")
    effects, report = [], {}
    for name in methods:
        baseline = by_mdp[by_mdp.method == name].sort_values("mdp_id")
        if not baseline.mdp_id.reset_index(drop=True).equals(target.mdp_id.reset_index(drop=True)):
            raise RuntimeError("MDP identity mismatch")
        report[name] = {}
        for metric in ("brier", "logloss"):
            before, after = baseline[metric].to_numpy(), target[metric].to_numpy()
            effects.append(before - after)
            report[name][metric] = {
                "baseline_mean": float(before.mean()),
                "oeh_mean": float(after.mean()),
                "relative_gain": float(1 - after.mean() / before.mean()),
                "mean_absolute_improvement": float((before - after).mean()),
            }
    effect = np.column_stack(effects)
    mean = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(84)
    draws = np.random.default_rng(20261001).integers(0, 84, size=(10000, 84))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - mean) / se,
                                        axis=1), .95))
    lower = mean - critical * se
    for i, name in enumerate(methods):
        for j, metric in enumerate(("brier", "logloss")):
            report[name][metric]["joint_simultaneous_lower_95"] = float(lower[2 * i + j])
    result = {
        "status": "exploratory_two_opened_cohorts_not_confirmation",
        "mdps": 84, "bootstrap_draws": 10000,
        "comparison_scope": "same expanded meta-training and three-way architecture; EH uses true-MDP oracle descriptors",
        "comparisons": report,
        "development_gate": "GO" if all(
            report[name][metric]["relative_gain"] >= .05
            and report[name][metric]["joint_simultaneous_lower_95"] > 0
            for name in methods for metric in ("brier", "logloss")) else "NO-GO",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (POOLED, EH)},
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
