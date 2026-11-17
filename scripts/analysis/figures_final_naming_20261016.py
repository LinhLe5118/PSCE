"""Redraw five figures with the locked final naming (16/10/2026); data unchanged.

The figures were first drawn with development names ("Full head alone", "B2 + diagnostics head", "legacy",
"full OEH", "Final:", "Fifth:", "panel 5", "Cohort 4"). This script reads only the committed data that the
original scripts wrote and redraws the same plots with the names fixed for the thesis and the paper:

* OEH-F (full head alone), robust head (B2 + D), pilot diagnostics D;
* Test 1 (four panels), Test 2 (FrozenLake, gambler's problem), Test 3 (MDP Playground, labmaze);
* cohorts numbered chronologically: E1, cohorts 1-2 (60-MDP), 3-6 (120-MDP), 7, 8. The panel the earlier
  re-analysis called "cohort 4" is the fourth 120-MDP test, cohort 6.

Nothing is refitted or rescored. The original artifact directories are not modified (some of their figure
files are hashed in manifests); outputs go to a new directory with its own manifest, and the copies used by
the paper are refreshed under paper/figures/.

    python -m scripts.analysis.figures_final_naming_20261016
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts"
OUT = A / "week8/figures_final_naming_2026-10-16"
PAPER = ROOT / "paper/figures"
SOURCES = {
    "cross_panel": A / "week8/cross_panel_reanalysis_2026-10-14/forest_plot_rows.csv",
    "gain_by_distance": A / "week8/oeh_final_posthoc_2026-10-15/gain_by_distance.csv",
    "calibration": A / "week8/manuscript_tables_2026-10-13/calibration_by_budget.csv",
    "ablation": A / "week8/m1_matched_ablation_2026-10-13/paired_mdp_delta.csv",
    "sensitivity": A / "week8/threshold_sensitivity_2026-10-15/sensitivity.csv",
}
# Locked names (writing kit TERMINOLOGY.md, version 2026-10-16b)
PANELS = {  # development panels first, then the final tests
    "cohort4": "Cohort 6", "cohort7": "Cohort 7", "cohort8": "Cohort 8", "colosseum2027": "Colosseum 2027",
    "colosseum2028": "Colosseum 2028", "spibb": "SPIBB", "bridge_atari5": "BRIDGE Atari-5",
    "bridge_expanded5": "BRIDGE Expanded-5", "internal": "Test 1: internal", "colosseum2029": "Test 1: Colosseum 2029",
    "spibb2": "Test 1: SPIBB-2", "forest": "Test 1: forest", "frozenlake": "Test 2: FrozenLake",
    "gambler": "Test 2: gambler's problem"}
METHODS = {"Full head": "OEH-F", "OEH": "OEH"}
DISTANCE = {"v2": "OEH", "oeh_final": "OEH-F", "legacy": "Robust head (B2 + D)"}
SENSITIVITY = {
    "final internal (OEH)": "Test 1, internal (OEH)",
    "final external pooled (OEH)": "Test 1, external pooled (OEH)",
    "panel 5 pooled (OEH)": "Test 2, pooled (OEH)",
    "development internal (OEH-F)": "Development, internal (OEH-F)",
    "development external (OEH-F)": "Development, external (OEH-F)"}
ABLATION = {"b2_matched": "B2 matched", "oeh_without_direct_oehb": "Without direct OEH-B",
            "oeh_without_oehh_risk": "Without OEH-H", "oeh_without_legacy_diagnostics": "Without D and context"}
LEARNERS = {"behavior_cloning_tabular": "BC", "cfqi_tabular": "CFQI", "iql_tabular": "IQL"}
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
PDF_META = {"CreationDate": None, "ModDate": None}
PNG_META = {"Software": None}


def save(fig, name: str, paper: bool) -> list[Path]:
    paths = []
    for target in [OUT / name] + ([PAPER / name] if paper else []):
        fig.savefig(target.with_suffix(".pdf"), metadata=PDF_META, bbox_inches="tight")
        paths.append(target.with_suffix(".pdf"))
        # the paper builds from PDFs; refresh a PNG copy there only if one is already tracked
        if target.parent == OUT or target.with_suffix(".png").exists():
            fig.savefig(target.with_suffix(".png"), dpi=200, metadata=PNG_META, bbox_inches="tight")
            paths.append(target.with_suffix(".png"))
    return paths


def cross_panel(plt) -> list[Path]:
    frame = pd.read_csv(SOURCES["cross_panel"]).replace({"method": METHODS})
    order = [p for p in PANELS if p in set(frame.panel)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(order) + 1.0), sharey=True)
    for ax, metric, title in zip(axes, ("ibs", "ll"), ("Integrated Brier", "Log loss")):
        for i, panel in enumerate(order):
            y = len(order) - 1 - i
            sub = frame[(frame.panel == panel) & (frame.metric == metric)]
            both = (sub.method == "OEH").any()
            for method, dy, marker, color in (("OEH-F", 0.13 if both else 0.0, "s", AQUA), ("OEH", -0.13, "o", BLUE)):
                r = sub[sub.method == method]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([100 * r.low, 100 * r.high], [y + dy] * 2, color=color, lw=1.6)
                ax.plot(100 * r.gain, y + dy, marker=marker, color=color, ms=4.5, label=method)
        ax.axvline(0, color="0.55", lw=0.8, ls="--")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("gain over B2 matched (%)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{PANELS[p]} ({int(frame[frame.panel == p].mdps.iloc[0])})" for p in reversed(order)],
                            fontsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), fontsize=7, loc="upper center", ncol=len(unique), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save(fig, "fig_cross_panel", paper=True)
    plt.close(fig)
    return out


def gain_by_distance(plt) -> list[Path]:
    import matplotlib.ticker

    gain = pd.read_csv(SOURCES["gain_by_distance"])
    colors = {"v2": BLUE, "oeh_final": AQUA, "legacy": ORANGE}
    markers = {"v2": "o", "oeh_final": "s", "legacy": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = gain.d_median.to_numpy()
    for ax, metric, title in zip(axes, ("brier", "log_loss"), ("Integrated Brier", "Log loss")):
        for m, label in DISTANCE.items():
            y, se = gain[f"{metric}_gain_{m}"].to_numpy(), gain[f"{metric}_se_{m}"].to_numpy()
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=colors[m], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=colors[m], lw=2, marker=markers[m], ms=4.5, label=label)
        ax.axhline(0, color="0.55", lw=0.8)
        ax.axvline(3.0, color="0.55", lw=0.8, ls="--")
        ax.text(3.05, ax.get_ylim()[1], " $d_0$", va="top", fontsize=7, color="0.35")
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks([0.75, 1, 1.5, 2, 3, 5])
        ax.set_xticklabels(["0.75", "1", "1.5", "2", "3", "5"])
        ax.set_xlabel("shift distance $d$ (bin median)", fontsize=8)
        ax.set_ylabel(f"{title} gain over B2 matched", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = save(fig, "fig_gain_by_distance", paper=True)
    plt.close(fig)
    return out


def _serif(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.5})


def threshold_brier(plt) -> list[Path]:
    cal = pd.read_csv(SOURCES["calibration"])
    with plt.rc_context():
        _serif(plt)
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
        budgets = [1, 2, 5, 10, 20, 50]
        for ax, name in zip(axes, LEARNERS.values()):
            g = cal[cal.learner == name]
            for method, color, label, marker in (("b2_matched", ORANGE, "B2 matched", "^"),
                                                 ("oeh_final", BLUE, "OEH-F", "s")):
                ax.plot(range(6), g[g.method == method].threshold_brier, color=color, lw=1.8, marker=marker, ms=4,
                        label=label)
            ax.set_xticks(range(6), [f"x{b}" for b in budgets], fontsize=7.5)
            ax.set_title(name, fontsize=8.5, color=INK)
            ax.set_ylim(bottom=0)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(r"Brier of $P(E_l \leq x_j)$")
        axes[0].legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        out = save(fig, "fig_threshold_brier", paper=True)
        plt.close(fig)
    return out


def ablation(plt) -> list[Path]:
    delta = pd.read_csv(SOURCES["ablation"])
    delta = delta[(delta.panel == "primary") & (delta.metric == "brier") & delta.comparison.str.endswith("minus oeh_full_108")]
    arms = list(ABLATION)
    with plt.rc_context():
        _serif(plt)
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
        for ax, learner in zip(axes, ["BC", "CFQI", "IQL", "pooled"]):
            key = {v: k for k, v in LEARNERS.items()}.get(learner, "pooled")
            g = delta[delta.algorithm == key].set_index("comparison")
            for y, arm in enumerate(arms):
                r = g.loc[f"{arm} minus oeh_full_108"]
                color = ORANGE if r.ci95_low > 0 else (BLUE if r.ci95_high < 0 else GRAY)
                ax.plot([r.ci95_low, r.ci95_high], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot(r.mean_delta, y, "o", color=color, ms=5, markeredgecolor="white", markeredgewidth=1)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_title(learner if learner != "pooled" else "All learners", fontsize=8.5, color=INK)
            ax.xaxis.grid(True)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=6.5)
        axes[0].set_yticks(range(len(arms)), [ABLATION[a] for a in arms], fontsize=7.5)
        axes[0].invert_yaxis()
        fig.supxlabel("IBS change vs OEH-F (arm − OEH-F; positive = worse); cohort 7, exploratory", fontsize=7.5,
                      color=INK2)
        fig.tight_layout()
        out = save(fig, "fig_ablation", paper=False)
        plt.close(fig)
    return out


def sensitivity(plt) -> list[Path]:
    from matplotlib.colors import LinearSegmentedColormap

    taus, seeds, official = (0.90, 0.95, 0.99), (7, 8, 9), (0.95, 8)
    sens = pd.read_csv(SOURCES["sensitivity"])
    sub = sens[sens.baseline == "b2"]
    groups = list(SENSITIVITY)
    cmap = LinearSegmentedColormap.from_list("rg", ["#e34948", "#f0efec", BLUE])
    fig, axes = plt.subplots(2, len(groups), figsize=(2.15 * len(groups), 4.6))
    vmax = float(np.ceil(100 * sub[["ibs_rel", "ll_rel"]].abs().to_numpy().max() / 10) * 10)
    for col, group in enumerate(groups):
        for row, (metric, name) in enumerate((("ibs_rel", "IBS"), ("ll_rel", "LL"))):
            ax = axes[row, col]
            g = sub[sub.group == group]
            grid = g.pivot(index="tau", columns="k", values=metric).reindex(index=list(taus), columns=list(seeds)) * 100
            low = g.pivot(index="tau", columns="k", values=metric.replace("_rel", "_low95")).reindex(
                index=list(taus), columns=list(seeds))
            high = g.pivot(index="tau", columns="k", values=metric.replace("_rel", "_high95")).reindex(
                index=list(taus), columns=list(seeds))
            ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax)
            for i in range(len(taus)):
                for j in range(len(seeds)):
                    star = "°" if low.iloc[i, j] <= 0 <= high.iloc[i, j] else ""
                    edge = (taus[i], seeds[j]) == official
                    ax.text(j, i, f"{grid.iloc[i, j]:.0f}{star}", ha="center", va="center", fontsize=7,
                            color="#0b0b0b", fontweight="bold" if edge else "normal")
            ax.set_xticks(range(len(seeds)), [f"{k}/10" for k in seeds], fontsize=6)
            ax.set_yticks(range(len(taus)), [f"{t:.2f}" for t in taus], fontsize=6)
            if col == 0:
                ax.set_ylabel(f"{name}: threshold", fontsize=7)
            if row == 0:
                ax.set_title(SENSITIVITY[group], fontsize=6.5)
    fig.suptitle("Relative improvement over B2 matched (%) under alternative label definitions; "
                 "bold = registered (0.95, 8/10); ° = 95% interval includes 0", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = save(fig, "fig_threshold_sensitivity", paper=True)
    plt.close(fig)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    outputs = []
    for draw in (cross_panel, gain_by_distance, threshold_brier, ablation, sensitivity):
        outputs += draw(plt)
    manifest = {
        "status": "descriptive; figures redrawn from committed data with the locked final names; nothing refitted",
        "names": {"panels": PANELS, "methods": METHODS, "distance": DISTANCE, "sensitivity": SENSITIVITY,
                  "ablation": ABLATION},
        "script": {rel(Path(__file__)): sha256(Path(__file__))},
        "sources": {rel(p): sha256(p) for p in SOURCES.values()},
        "outputs": {rel(p): sha256(p) for p in outputs},
    }
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {len(outputs)} files; manifest {rel(OUT / 'manifest.json')}")


if __name__ == "__main__":
    main()
