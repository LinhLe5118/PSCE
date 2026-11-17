"""Exploratory OEH successor using only already-opened development cohorts.

The ongoing unseen 120-MDP evaluation belongs to a different, frozen OEH
candidate. This file never accesses its forecasts or learner outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.develop_oeh_geometric_stack import (
    SOURCE, get_pair, geometric,
)
from scripts.analysis.develop_b2_eh_oracle_aft import ROOT, sha256


WEIGHT_MANIFEST = ROOT / "artifacts/oeh_difficulty_geometric_stack_development_2026-10-01/manifest.json"
EH_FORECASTS = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/scored_predictions.csv"
OUTPUT = ROOT / "artifacts/oeh_geometric_successor_development_2026-10-01"
KEY = ["cohort", "mdp_id", "policy_family", "algorithm", "panel", "event"]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    weight = float(json.loads(WEIGHT_MANIFEST.read_text(encoding="utf-8"))["oeh_weight"])
    source = pd.read_csv(SOURCE)
    eh = pd.read_csv(EH_FORECASTS)
    rows = []
    for cohort in ("first_open", "second_open"):
        keys, b2, oeh = get_pair(source, cohort)
        probs = geometric(b2, oeh, weight)
        brier, logloss = score_probabilities(probs, keys.event.to_numpy(dtype=int))
        scored = keys.copy()
        scored["method"] = "oeh_geometric_successor"
        scored["brier"] = brier
        scored["logloss"] = logloss
        rows.append(scored)
    successor = pd.concat(rows, ignore_index=True)
    baselines = pd.concat([
        source[source.method.isin(("b2_three_way", "oeh_three_way"))],
        eh[eh.method == "b2_plus_eh_oracle_three_way"],
    ], ignore_index=True)
    baseline = baselines[baselines.panel == "primary"]
    if (len(successor) != 1260 or successor.mdp_id.nunique() != 84
            or baseline.duplicated(KEY + ["method"]).any()):
        raise RuntimeError("Incomplete or duplicate opened-cohort forecasts")
    scored = pd.concat([successor, baseline], ignore_index=True)
    summary = scored.groupby(["method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("brier", "mean"),
        seven_class_log_loss=("logloss", "mean"))
    grouped = scored.groupby(["mdp_id", "method"], as_index=False).agg(
        brier=("brier", "mean"), logloss=("logloss", "mean"))
    names = ("b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_three_way")
    effects, comparisons = [], {}
    target = grouped[grouped.method == "oeh_geometric_successor"].sort_values("mdp_id")
    if len(target) != 84:
        raise RuntimeError("Missing stochastic MDPs")
    for name in names:
        other = grouped[grouped.method == name].sort_values("mdp_id")
        if not other.mdp_id.reset_index(drop=True).equals(target.mdp_id.reset_index(drop=True)):
            raise RuntimeError("Paired MDP identity mismatch")
        comparisons[name] = {}
        for metric in ("brier", "logloss"):
            before = other[metric].to_numpy(dtype=float)
            after = target[metric].to_numpy(dtype=float)
            effects.append(before - after)
            comparisons[name][metric] = {
                "baseline_mean": float(before.mean()),
                "candidate_mean": float(after.mean()),
                "relative_gain": float(1 - after.mean() / before.mean()),
                "absolute_improvement": float((before - after).mean()),
            }
    effect = np.column_stack(effects)
    mean = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(84)
    draws = np.random.default_rng(20261001).integers(0, 84, size=(10000, 84))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - mean) / se,
                                        axis=1), .95))
    lower = mean - critical * se
    for i, name in enumerate(names):
        for j, metric in enumerate(("brier", "logloss")):
            comparisons[name][metric]["joint_simultaneous_lower_95"] = float(lower[2 * i + j])
    gate = all(comparisons[name][metric]["relative_gain"] >= .05
               and comparisons[name][metric]["joint_simultaneous_lower_95"] > 0
               for name in names[:2] for metric in ("brier", "logloss"))
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "scored_predictions.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_successor_two_opened_cohorts_not_independent_confirmation",
        "weight_fit_on": "first_open", "weight": weight,
        "development_gate_against_b2_and_oracle_eh": "GO" if gate else "NO-GO",
        "comparisons": comparisons,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in (SOURCE, WEIGHT_MANIFEST, EH_FORECASTS)},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Successor exploratory development gate: {'GO' if gate else 'NO-GO'}")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
