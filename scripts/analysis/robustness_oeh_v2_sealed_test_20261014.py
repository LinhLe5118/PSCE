"""Descriptive robustness of the OEH v2 sealed test (not part of the registered gate).

* leave-one-external-panel-out: the pooled step-1 and step-2 external contrasts without each panel;
* per panel: v2 vs B2 + legacy and vs OEH current with 95% t intervals of the mean per-MDP gain;
* panel-stratified bootstrap of the pooled external contrasts (resampling MDPs within each panel);
* log-loss floor 0.01 instead of 1e-12;
* per learner and panel: v2 vs B2 + legacy and OEH (criterion 3 of the plan: no learner significantly worse).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14"
EXTERNAL = ("colosseum2029", "spibb2", "forest")
P = [f"p_{i}" for i in range(7)]


def per_mdp(scored: pd.DataFrame, by=("source_panel", "method", "mdp_id")) -> pd.DataFrame:
    return scored.groupby(list(by))[["brier", "log_loss", "log_loss_floor01"]].mean()


def interval(d: np.ndarray) -> tuple[float, float, float]:
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)


def main() -> None:
    s = pd.read_csv(OUT / "scored.csv.gz")
    s = s[s.method.isin(["v2", "oeh_final", "legacy", "b2_matched"])].copy()
    p = s[P].to_numpy(float)
    y = s.event.to_numpy(int)
    s["log_loss_floor01"] = -np.log(np.clip(p[np.arange(len(y)), y], 0.01, 1))
    m = per_mdp(s).reset_index()
    wide = m.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=["brier", "log_loss", "log_loss_floor01"])
    rows = []
    for base in ("legacy", "oeh_final", "b2_matched"):
        for metric in ("brier", "log_loss", "log_loss_floor01"):
            gain = (wide[(metric, base)] - wide[(metric, "v2")]).rename("gain").reset_index()
            for panel in ("internal", *EXTERNAL):
                mean, lo, hi = interval(gain[gain.source_panel == panel].gain.to_numpy())
                rows.append({"analysis": "per_panel", "panel": panel, "baseline": base, "metric": metric,
                             "mean_gain": mean, "low95": lo, "high95": hi})
            ext = gain[gain.source_panel.isin(EXTERNAL)]
            mean, lo, hi = interval(ext.gain.to_numpy())
            rows.append({"analysis": "pooled_external_t", "panel": "pooled", "baseline": base, "metric": metric,
                         "mean_gain": mean, "low95": lo, "high95": hi})
            for left in EXTERNAL:
                kept = ext[ext.source_panel != left].gain.to_numpy()
                mean, lo, hi = interval(kept)
                rows.append({"analysis": "leave_one_panel_out", "panel": f"without {left}", "baseline": base, "metric": metric,
                             "mean_gain": mean, "low95": lo, "high95": hi})
            rng = np.random.default_rng(20261102)
            groups = [ext[ext.source_panel == q].gain.to_numpy() for q in EXTERNAL]
            draws = np.empty(10_000)
            n = sum(len(g) for g in groups)
            for b in range(len(draws)):
                draws[b] = sum(g[rng.integers(0, len(g), len(g))].sum() for g in groups) / n
            rows.append({"analysis": "pooled_external_stratified_bootstrap", "panel": "pooled", "baseline": base, "metric": metric,
                         "mean_gain": float(ext.gain.mean()), "low95": float(np.quantile(draws, 0.025)),
                         "high95": float(np.quantile(draws, 0.975))})
    learner = per_mdp(s, ("source_panel", "algorithm", "method", "mdp_id")).reset_index()
    for (panel, algo), g in learner.groupby(["source_panel", "algorithm"]):
        w = g.pivot(index="mdp_id", columns="method", values=["brier", "log_loss"])
        for base in ("legacy", "oeh_final"):
            for metric in ("brier", "log_loss"):
                mean, lo, hi = interval((w[(metric, base)] - w[(metric, "v2")]).to_numpy())
                rows.append({"analysis": "per_learner", "panel": f"{panel}/{algo}", "baseline": base, "metric": metric,
                             "mean_gain": mean, "low95": lo, "high95": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "robustness.csv", index=False)
    pd.set_option("display.width", 200)
    show = out[out.analysis != "per_learner"]
    print(show[show.baseline != "b2_matched"].round(4).to_string(index=False))
    worse = out[(out.analysis == "per_learner") & (out.high95 < 0)]
    print("\nlearner cells where v2 is significantly worse (95% t):")
    print(worse.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
