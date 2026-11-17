"""Development-only simultaneous matched gate for the early-budget hybrid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.develop_b2_eh_oracle_aft import ROOT, sha256
from scripts.analysis.develop_early_hybrid_four_opened import OUTPUT as DEVELOPMENT


OUTPUT = ROOT / "artifacts/oeh_early_hybrid_gate_development_2026-10-01"
METRICS = ("integrated_brier", "seven_class_log_loss")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Early-hybrid gate artifact exists")
    rows = pd.read_csv(DEVELOPMENT / "scored.csv")
    methods = {"b2_three_way", "b2_plus_eh_oracle_three_way", "oeh_cross_capacity",
               "early_b2_alpha_0.25", "early_b2_alpha_0.5",
               "early_b2_alpha_0.75", "early_b2_alpha_1"}
    if (set(rows.method) != methods or rows.mdp_id.nunique() != 252
            or len(rows) != 252 * 15 * len(methods)
            or rows.duplicated(["mdp_id", "policy_family", "algorithm", "method"]).any()):
        raise RuntimeError("Incomplete matched development panel")
    mdp = rows.groupby(["mdp_id", "method"], as_index=False)[list(METRICS)].mean()
    ids = sorted(mdp.mdp_id.unique())
    indexed = mdp.set_index(["mdp_id", "method"])
    effects, contrasts = [], []
    for baseline in ("b2_three_way", "b2_plus_eh_oracle_three_way"):
        for metric in METRICS:
            before = np.asarray([indexed.loc[(mdp_id, baseline), metric] for mdp_id in ids])
            after = np.asarray([indexed.loc[(mdp_id, "early_b2_alpha_0.5"), metric] for mdp_id in ids])
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
    draws = np.random.default_rng(20261005).integers(0, len(ids), size=(10000, len(ids)))
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
        "mdps": 252, "bootstrap_draws": 10000,
        "simultaneous_critical": critical,
        "development_gate": "GO" if gate else "NO-GO",
        "candidate": "OEH cross-learner context with fixed B2 early-class log-ratio alpha 0.5",
        "matched_comparators": ["B2 same training/architecture/capacity",
                                "B2+full-MDP-oracle-EH same training/architecture/capacity"],
        "input_sha256": {str((DEVELOPMENT / "scored.csv").relative_to(ROOT)).replace("\\", "/"):
                         sha256(DEVELOPMENT / "scored.csv")},
        "script_sha256": sha256(Path(__file__)),
        "contrasts_sha256": sha256(OUTPUT / "contrasts.csv"),
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    print(f"Development-only gate: {'GO' if gate else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
