"""Threshold sensitivity of the label definition (task card T6; post hoc, descriptive).

Relabels every opened panel from its seed-level outcomes for success thresholds
tau in {0.90, 0.95, 0.99} and required seeds k in {7, 8, 9} of 10 (sustained-suffix rule
unchanged), after reproducing the official labels at (0.95, 8) exactly, and rescores the
sealed forecasts. The sealed forecasts target (0.95, 8/10): other cells measure whether the
*comparison* between forecasters is robust to the label definition, not forecasts built for
those targets.

Usage: ``python -m scripts.analysis.threshold_sensitivity [--output DIR]``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis import extrapolation_endpoint as X

ROOT = X.ROOT
A = X.A
OUTPUT = A / "week8/threshold_sensitivity_2026-10-15"
KEYS, P = X.KEYS, X.P
BUDGETS = (1, 2, 5, 10, 20, 50)
TAUS = (0.90, 0.95, 0.99)
SEEDS = (7, 8, 9)
OFFICIAL = (0.95, 8)
OUTCOMES = {
    "cohort7": A / "oeh_confirmatory_unseen_120_2026-10-04/all_core_budget_seed_outcomes.csv",
    "cohort8": A / "oeh_eighth_cohort_2026-10-14/internal/outcomes.csv.gz",
    "colosseum2027": A / "week8/colosseum_confirm_2026-10-14/colosseum_outcomes.csv",
    "colosseum2028": A / "oeh_eighth_cohort_2026-10-14/colosseum/outcomes.csv.gz",
    "spibb": A / "oeh_spibb_panel_2026-10-14/spibb/outcomes.csv.gz",
    **{p: A / f"oeh_v2_sealed_test_2026-10-14/{p}/outcomes.csv.gz" for p in ("internal", "colosseum2029", "spibb2", "forest")},
    **{p: A / f"oeh_fifth_panel_2026-10-15/{p}/outcomes.csv.gz" for p in ("frozenlake", "gambler")},
}
GROUPS = {
    "final internal (OEH)": (("internal",), "oeh"),
    "final external pooled (OEH)": (("colosseum2029", "spibb2", "forest"), "oeh"),
    "panel 5 pooled (OEH)": (("frozenlake", "gambler"), "oeh"),
    "development internal (OEH-F)": (("cohort7", "cohort8"), "oeh_f"),
    "development external (OEH-F)": (("colosseum2027", "colosseum2028", "spibb"), "oeh_f"),
}


def suffix_events(counts: np.ndarray, k: int) -> np.ndarray:
    """First budget index from which every later count is >= k; 6 if none (vectorized suffix rule)."""
    ok = counts >= k
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok, 1), 1), 1)
    return np.where(suffix.any(1), suffix.argmax(1), counts.shape[1])


def counts(outcomes: pd.DataFrame, tau: float) -> pd.DataFrame:
    o = outcomes[outcomes.budget.isin(BUDGETS)]
    c = (o.assign(s=(o.normalized_return >= tau).astype(int)).groupby(KEYS + ["budget"]).s.sum().unstack("budget")
         .reindex(columns=list(BUDGETS)))
    if c.isna().any().any():
        raise RuntimeError("incomplete seed grid")
    return c


def relabel(panel: str) -> tuple[pd.DataFrame, dict]:
    o = pd.read_csv(OUTCOMES[panel], usecols=KEYS + ["budget", "training_seed", "normalized_return", "success"])
    n = o.groupby(KEYS + ["budget"]).size()
    if (n != 10).any():
        raise RuntimeError(f"{panel}: ten seeds required")
    stored = o.success.astype(bool).to_numpy()
    recomputed = (o.normalized_return >= OFFICIAL[0]).to_numpy()
    out = {}
    for tau in TAUS:
        c = counts(o, tau)
        for k in SEEDS:
            out[(tau, k)] = pd.Series(suffix_events(c.to_numpy(int), k), index=c.index)
    labels = pd.read_csv(X.PANELS[panel]["labels"])
    if "panel" in labels:
        labels = labels[labels.panel == "primary"]
    official = labels.set_index(KEYS).event
    mine = out[OFFICIAL].reindex(official.index)
    check = {"rows": int(len(official)), "label_mismatches": int((mine.to_numpy() != official.to_numpy()).sum()),
             "success_flag_mismatches": int((stored != recomputed).sum())}
    frame = pd.DataFrame({f"e_{t:.2f}_{k}": s for (t, k), s in out.items()}).reset_index()
    return frame, check


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    checks, rows, census = {}, [], []
    per_cell = {}
    for panel in OUTCOMES:
        frame, check = relabel(panel)
        checks[panel] = check
        if check["label_mismatches"] or check["success_flag_mismatches"]:
            raise RuntimeError(f"{panel}: official labels not reproduced {check}")
        pred = X.load(panel).drop(columns=["event"])
        m = pred.merge(frame, on=KEYS, validate="many_to_one")
        p = m[P].to_numpy(float)
        for tau in TAUS:
            for k in SEEDS:
                y = m[f"e_{tau:.2f}_{k}"].to_numpy(int)
                cell = m[["source_panel", "canonical", "mdp_id"]].copy()
                cell["ibs"], cell["ll"] = X.ibs(p, y), X.log_loss(p, y)
                per_cell.setdefault((tau, k), []).append(cell.groupby(["source_panel", "canonical", "mdp_id"])[["ibs", "ll"]].mean().reset_index())
                lab = frame[f"e_{tau:.2f}_{k}"]
                census.append({"panel": panel, "tau": tau, "k": k, "censored_share": float((lab == 6).mean())})
    for (tau, k), parts in per_cell.items():
        pm = pd.concat(parts, ignore_index=True)
        for group, (panels, cand) in GROUPS.items():
            for base in ("b2", "robust"):
                w = pm[pm.source_panel.isin(panels)].pivot_table(index=["source_panel", "mdp_id"], columns="canonical", values=["ibs", "ll"])
                row = {"group": group, "candidate": cand, "baseline": base, "tau": tau, "k": k, "mdps": int(len(w))}
                for metric in ("ibs", "ll"):
                    d = (w[(metric, base)] - w[(metric, cand)]).to_numpy()
                    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                    row.update({f"{metric}_gain": float(d.mean()), f"{metric}_low95": float(d.mean() - half),
                                f"{metric}_high95": float(d.mean() + half), f"{metric}_rel": float(d.mean() / w[(metric, base)].mean())})
                rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "sensitivity.csv", index=False)
    pd.DataFrame(census).to_csv(out / "censoring_by_cell.csv", index=False)

    # heatmap: relative gain over B2 (rows: tau, columns: k), one panel per group and metric
    cmap = LinearSegmentedColormap.from_list("rg", ["#e34948", "#f0efec", "#2a78d6"])  # red = worse, gray = 0, blue = better
    sub = sens[sens.baseline == "b2"]
    groups = list(GROUPS)
    fig, axes = plt.subplots(2, len(groups), figsize=(2.15 * len(groups), 4.6))
    vmax = float(np.ceil(100 * sub[["ibs_rel", "ll_rel"]].abs().to_numpy().max() / 10) * 10)
    for col, group in enumerate(groups):
        for row, (metric, name) in enumerate((("ibs_rel", "IBS"), ("ll_rel", "LL"))):
            ax = axes[row, col]
            grid = sub[sub.group == group].pivot(index="tau", columns="k", values=metric).reindex(index=list(TAUS), columns=list(SEEDS)) * 100
            low = sub[sub.group == group].pivot(index="tau", columns="k", values=metric.replace("_rel", "_low95")).reindex(index=list(TAUS), columns=list(SEEDS))
            high = sub[sub.group == group].pivot(index="tau", columns="k", values=metric.replace("_rel", "_high95")).reindex(index=list(TAUS), columns=list(SEEDS))
            ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax)
            for i in range(len(TAUS)):
                for j in range(len(SEEDS)):
                    star = "°" if low.iloc[i, j] <= 0 <= high.iloc[i, j] else ""  # interval includes 0
                    edge = (TAUS[i], SEEDS[j]) == OFFICIAL
                    ax.text(j, i, f"{grid.iloc[i, j]:.0f}{star}", ha="center", va="center", fontsize=7,
                            color="#0b0b0b", fontweight="bold" if edge else "normal")
            ax.set_xticks(range(len(SEEDS)), [f"{k}/10" for k in SEEDS], fontsize=6)
            ax.set_yticks(range(len(TAUS)), [f"{t:.2f}" for t in TAUS], fontsize=6)
            if col == 0:
                ax.set_ylabel(f"{name}: threshold", fontsize=7)
            if row == 0:
                ax.set_title(group, fontsize=6.5)
    fig.suptitle("Gain over B2 (%) under alternative label definitions; bold = registered (0.95, 8/10); ° = 95% interval includes 0",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for target in (out / "fig_threshold_sensitivity", ROOT / "paper/figures/fig_threshold_sensitivity"):
        fig.savefig(target.with_suffix(".pdf"))
        fig.savefig(target.with_suffix(".png"), dpi=200)

    inputs = sorted(set(OUTCOMES.values()) | {X.PANELS[p]["labels"] for p in OUTCOMES} | {f for p in OUTCOMES for f in X.PANELS[p]["pred"]})
    manifest = {
        "status": "posthoc_descriptive", "task": "T6 threshold sensitivity", "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {"tau": list(TAUS), "seeds_required": list(SEEDS), "official": list(OFFICIAL)},
        "note": "sealed forecasts target (0.95, 8/10); other cells test the robustness of the comparison only",
        "official_label_reproduction": checks,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): X.sha(p) for p in inputs},
        "outputs_sha256": {p.name: X.sha(p) for p in (out / "sensitivity.csv", out / "censoring_by_cell.csv", out / "fig_threshold_sensitivity.pdf")},
        "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): X.sha(Path(__file__))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("official labels reproduced:", all(not v["label_mismatches"] for v in checks.values()), sum(v["rows"] for v in checks.values()), "rows")
    show = sub[["group", "tau", "k", "ibs_rel", "ibs_low95", "ll_rel", "ll_low95"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
