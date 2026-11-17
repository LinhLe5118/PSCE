"""Cross-panel forest plot: relative gain over B2 (integrated Brier and log loss) on every sealed panel.

Development panels come from ``cross_panel_reanalysis_2026-10-14/paired.csv`` (the full head vs B2). The
final sealed tests (four-panel test and fifth panel), when scored, add their panels with both the
full head alone and OEH (the final method with the shift gate). Colors: reference palette slots 1 and 3
(validated, as in the gain-by-distance figure).
Intervals are 95% per-panel t intervals of the mean per-MDP gain, divided by B2's mean (descriptive).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "artifacts/week8/cross_panel_reanalysis_2026-10-14/paired.csv"
V2 = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14/scored.csv.gz"
FIFTH = ROOT / "artifacts/oeh_fifth_panel_2026-10-15/scored.csv.gz"
OUT = ROOT / "paper/figures/fig_cross_panel"
LABELS = {  # development panels first, then the final sealed tests
    "cohort4": "Cohort 4", "cohort7": "Cohort 7", "cohort8": "Cohort 8", "colosseum2027": "Colosseum 2027",
    "colosseum2028": "Colosseum 2028", "spibb": "SPIBB", "bridge_atari5": "BRIDGE Atari-5", "bridge_expanded5": "BRIDGE Expanded-5",
    "internal": "Final: internal", "colosseum2029": "Final: Colosseum 2029", "spibb2": "Final: SPIBB-2",
    "forest": "Final: forest (MDPToolbox)", "frozenlake": "Fifth: FrozenLake", "gambler": "Fifth: gambler"}


def rows() -> pd.DataFrame:
    p = pd.read_csv(PAIRED)
    p = p[p.comparator == "b2_matched"]
    out = [{"panel": r.panel, "kind": r.kind, "method": "Full head", "metric": r.metric, "mdps": int(r.mdps),
            "gain": r.relative_gain, "low": r.ci95_low / r.comparator_mean, "high": r.ci95_high / r.comparator_mean}
           for r in p.itertuples()]
    sources = []
    if V2.exists():
        s = pd.read_csv(V2, usecols=["source_panel", "method", "mdp_id", "brier", "log_loss"])
        sources.append(s[s.method.isin(["b2_matched", "oeh_final", "v2"])])
    if FIFTH.exists():
        s = pd.read_csv(FIFTH, usecols=["source_panel", "method", "mdp_id", "brier", "log_loss"])
        s = s[s.method.isin(["b2_matched", "oeh_full_head", "oeh"])]
        sources.append(s.replace({"method": {"oeh_full_head": "oeh_final", "oeh": "v2"}}))
    if sources:
        per = pd.concat(sources).groupby(["source_panel", "method", "mdp_id"])[["brier", "log_loss"]].mean()
        for panel, g in per.groupby(level="source_panel"):
            wide = g.droplevel(0).unstack("method")
            for metric, short in (("brier", "ibs"), ("log_loss", "ll")):
                base = wide[(metric, "b2_matched")]
                for method, label in (("oeh_final", "Full head"), ("v2", "OEH")):
                    gain = base - wide[(metric, method)]
                    half = 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))
                    out.append({"panel": panel, "kind": "internal" if panel == "internal" else "external", "method": label,
                                "metric": short, "mdps": len(gain), "gain": gain.mean() / base.mean(),
                                "low": (gain.mean() - half) / base.mean(), "high": (gain.mean() + half) / base.mean()})
    return pd.DataFrame(out)


def plot(frame: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [p for p in LABELS if p in set(frame.panel)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(order) + 1.0), sharey=True)
    for ax, metric, title in zip(axes, ("ibs", "ll"), ("Integrated Brier", "Log loss")):
        for i, panel in enumerate(order):
            y = len(order) - 1 - i
            sub = frame[(frame.panel == panel) & (frame.metric == metric)]
            both = (sub.method == "OEH").any()
            for method, dy, marker, color in (("Full head", 0.13 if both else 0.0, "s", "#1baf7a"),
                                              ("OEH", -0.13, "o", "#2a78d6")):
                r = sub[sub.method == method]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([100 * r.low, 100 * r.high], [y + dy] * 2, color=color, lw=1.6)
                ax.plot(100 * r.gain, y + dy, marker=marker, color=color, ms=4.5, label=method)
        ax.axvline(0, color="0.55", lw=0.8, ls="--")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("gain over B2 (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{LABELS[p]} ({int(frame[frame.panel == p].mdps.iloc[0])})" for p in reversed(order)], fontsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), fontsize=7, loc="upper center", ncol=len(unique), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)


def main() -> None:
    frame = rows()
    frame.to_csv(ROOT / "artifacts/week8/cross_panel_reanalysis_2026-10-14/forest_plot_rows.csv", index=False)
    plot(frame)
    print(frame.round(4).to_string())


if __name__ == "__main__":
    main()
