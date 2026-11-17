"""Development-only simultaneous matched gate for depth-3 cross-learner OEH."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import ROOT, sha256
from scripts.analysis.develop_tree_capacity import OUTPUT as CAPACITY
from scripts.analysis.develop_oracle_cross_capacity import OUTPUT as ORACLE


OUTPUT = ROOT / "artifacts/oeh_cross_capacity_gate_development_2026-10-01"
METRICS = ("integrated_brier", "seven_class_log_loss")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Cross-capacity gate artifact exists")
    first = pd.read_csv(CAPACITY / "scored.csv")
    first = first[first.variant == "depth3_200"].drop(columns="variant")
    second = pd.read_csv(ORACLE / "scored.csv")
    rows = pd.concat((first, second), ignore_index=True)
    methods = {"b2", "oeh_cross", "eh_oracle_depth3_200"}
    if (set(rows.method) != methods or rows.mdp_id.nunique() != 168
            or len(rows) != 168 * 15 * len(methods)
            or rows.duplicated(["mdp_id", "policy_family", "algorithm", "method"]).any()):
        raise RuntimeError("Incomplete matched development panel")
    mdp = rows.groupby(["mdp_id", "method"], as_index=False)[list(METRICS)].mean()
    ids = sorted(mdp.mdp_id.unique())
    indexed = mdp.set_index(["mdp_id", "method"])
    effects, contrasts = [], []
    for baseline in ("b2", "eh_oracle_depth3_200"):
        for metric in METRICS:
            before = np.asarray([indexed.loc[(mdp_id, baseline), metric] for mdp_id in ids])
            after = np.asarray([indexed.loc[(mdp_id, "oeh_cross"), metric] for mdp_id in ids])
            delta = before - after
            effects.append(delta)
            contrasts.append({"baseline": baseline, "metric": metric,
                              "baseline_mean": float(before.mean()),
                              "oeh_mean": float(after.mean()),
                              "relative_gain": float(1. - after.mean() / before.mean()),
                              "absolute_improvement": float(delta.mean())})
    effect = np.column_stack(effects)
    means = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(len(ids))
    draws = np.random.default_rng(20261003).integers(0, len(ids), size=(10000, len(ids)))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - means) / se,
                                        axis=1), .95))
    lower = means - critical * se
    for i, contrast in enumerate(contrasts):
        contrast["simultaneous_one_sided_lower_95"] = float(lower[i])
    gate = all(row["relative_gain"] >= .05 and row["simultaneous_one_sided_lower_95"] > 0
               for row in contrasts)
    OUTPUT.mkdir(parents=True)
    pd.DataFrame(contrasts).to_csv(OUTPUT / "contrasts.csv", index=False)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "status": "exploratory_development_gate_not_independent_confirmation",
        "mdps": 168, "bootstrap_draws": 10000,
        "simultaneous_critical": critical,
        "development_gate": "GO" if gate else "NO-GO",
        "candidate": "OEH cross-learner context with depth-3 multiclass component",
        "matched_comparators": ["B2 same training/architecture/capacity",
                                "B2+full-MDP-oracle-EH same training/architecture/capacity"],
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                         for path in (CAPACITY / "scored.csv", ORACLE / "scored.csv")},
        "script_sha256": sha256(Path(__file__)),
        "contrasts_sha256": sha256(OUTPUT / "contrasts.csv"),
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    print(f"Development-only gate: {'GO' if gate else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
