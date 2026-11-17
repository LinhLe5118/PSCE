"""Paper figures for the fourth-cohort OEH test, rendered from sealed result CSVs.

Every number is read from the analysis outputs; nothing is recomputed. Colors
follow the validated categorical order (blue, orange, aqua) with at most three
colored series per figure; comparators are drawn in recessive gray.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e2dd"
LABELS = {
    "oeh_final": "OEH (final)", "oeh_b_raw": "OEH-B, unfitted", "prior_oeh_matched": "previous OEH (saturated)",
    "b2_matched": "B2 (matched)", "b2_plus_eh_oracle_matched": "B2 + oracle EH [privileged]",
    "b2_plus_tq_saco_original": "B2 + TQ/SACo (author code) [privileged]",
    "published_metrics_only": "TQ, SACo, ERI, BWD only [privileged]",
    "b2_plus_bwd": "B2 + BWD (ICLR 2026)", "b2_plus_eri": "B2 + ERI",
    "b2_generic": "B2 AFT", "train_frequency": "fold frequency",
    "coverage_adaptation": "coverage adapt.", "quality_adaptation": "quality adapt.",
    "coverage_quality_adaptation": "coverage+quality",
}


def style(ax, *, xgrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=8)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def difficulty(out: Path) -> None:
    summary = pd.read_csv(SOURCE / "summary.csv")
    summary = summary[summary.panel == "primary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    for ax, metric, title in zip(axes, ("integrated_brier", "seven_class_log_loss"),
                                 ("Integrated Brier (lower is better)", "Seven-class log loss")):
        frame = summary.sort_values(metric, ascending=False)
        y = np.arange(len(frame))
        colors = [BLUE if m == "oeh_final" else AQUA if m == "oeh_b_raw" else MUTED for m in frame.method]
        ax.hlines(y, 0, frame[metric], color=GRID, linewidth=2)
        ax.scatter(frame[metric], y, s=36, color=colors, zorder=3, edgecolor="white", linewidth=1.5)
        ax.set_yticks(y, [LABELS.get(m, m) for m in frame.method], color=INK, fontsize=8)
        for value, row in zip(frame[metric], y):
            ax.text(value, row + 0.32, f"{value:.4f}", fontsize=6.5, color=SECONDARY, ha="center")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    fig.suptitle("Requirement forecasts, 84 new stochastic MDPs", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "oeh_final_difficulty")


def contrasts(out: Path) -> None:
    """Relative improvement of OEH over each comparator with the 5% gate line."""
    table = pd.read_csv(SOURCE / "contrasts.csv")
    table["gain_pct"] = 100 * table.relative_gain
    table["lower_pct"] = 100 * table.simultaneous_one_sided_lower_95 / table.baseline_mean
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=False)
    for ax, metric, title in zip(axes, ("integrated_brier", "seven_class_log_loss"),
                                 ("Integrated Brier", "Seven-class log loss")):
        frame = table[table.metric == metric].sort_values("gain_pct")
        y = np.arange(len(frame))
        privileged = frame.baseline.str.contains("oracle|tq_saco|published")
        color = np.where(privileged, ORANGE, BLUE)
        ax.axvline(0, color=SECONDARY, linewidth=0.8)
        ax.axvline(5, color=MUTED, linewidth=1, linestyle="--")
        ax.text(5.3, len(frame) - 0.5, "5% gate", fontsize=7, color=SECONDARY, va="center")
        ax.hlines(y, frame.lower_pct, frame.gain_pct, color=color, linewidth=2)
        ax.scatter(frame.gain_pct, y, s=34, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        for value, row in zip(frame.gain_pct, y):
            ax.text(value + 1.0, row, f"{value:.1f}%", fontsize=6.5, color=SECONDARY, va="center")
        ax.set_yticks(y, [LABELS.get(m, m) for m in frame.baseline], color=INK, fontsize=7.5)
        ax.set_title(f"{title}: relative improvement (point) and simultaneous 95% lower bound (bar end)",
                     fontsize=8, color=INK, loc="left")
        ax.set_xlabel("improvement of OEH over comparator (%)", fontsize=7.5, color=SECONDARY)
        style(ax)
    axes[0].scatter([], [], color=BLUE, label="same information as OEH")
    axes[0].scatter([], [], color=ORANGE, label="privileged information")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, out, "oeh_final_contrasts")


def reliability(out: Path) -> None:
    scored = pd.read_csv(SOURCE / "scored_predictions.csv")
    scored = scored[scored.panel == "primary"]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    for method, color in (("oeh_final", BLUE), ("b2_matched", ORANGE), ("oeh_b_raw", AQUA)):
        frame = scored[scored.method == method]
        bins = np.clip((frame.p_6 * 10).astype(int), 0, 9)
        grouped = frame.assign(bin=bins, censored=(frame.event == 6)).groupby("bin")
        x, y = grouped.p_6.mean(), grouped.censored.mean()
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=LABELS[method])
    ax.set_xlabel("Forecast P(requirement > x50)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("Observed censored fraction", fontsize=8, color=SECONDARY)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Censoring calibration", fontsize=9, color=INK, loc="left")
    style(ax, xgrid=False)
    ax.grid(color=GRID, linewidth=0.8)
    save(fig, out, "oeh_final_reliability")


def horizon(out: Path) -> None:
    table = pd.read_csv(SOURCE / "horizon_comparison.csv")
    table = table[~table.metric.str.contains("minus")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, target, title in zip(axes, ("event", "min_event"),
                                 ("CFQI requirement", "Best core learner requirement")):
        frame = table[table.target == target]
        y = np.arange(len(frame))
        colors = [BLUE if "OEH-H" in m else MUTED for m in frame.metric]
        ax.barh(y, frame.spearman, color=colors, height=0.6)
        ax.axvline(0, color=SECONDARY, linewidth=0.8)
        for value, row in zip(frame.spearman, y):
            ax.text(value + (0.01 if value >= 0 else -0.01), row, f"{value:.3f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=7, color=SECONDARY)
        ax.set_yticks(y, frame.metric, fontsize=8, color=INK)
        ax.set_title(f"Spearman with {title}", fontsize=8.5, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    save(fig, out, "oeh_final_horizon")


def selection(out: Path) -> None:
    summary = pd.read_csv(SOURCE / "selection_summary.csv")
    names = {"oeh_final_selection": "OEH selection", "oeh_v5_router": "OEH-v5 router",
             "oracle_group": "per-group oracle (bound)", "random": "random learner"}
    summary["label"] = summary.rule.map(lambda r: names.get(r, r.replace("fixed:", "always ")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, metric, title in zip(axes, ("mean_normalized_return", "seed_success_rate"),
                                 ("Mean normalized return at x10", "Seed success rate at x10")):
        frame = summary.sort_values(metric)
        y = np.arange(len(frame))
        colors = [BLUE if r == "oeh_final_selection" else ORANGE if r == "oeh_v5_router"
                  else MUTED for r in frame.rule]
        ax.hlines(y, frame[metric].min() * 0.95, frame[metric], color=GRID, linewidth=2)
        ax.scatter(frame[metric], y, s=36, color=colors, zorder=3, edgecolor="white", linewidth=1.5)
        ax.set_yticks(y, frame.label, fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    save(fig, out, "oeh_final_selection")


def decomposition(out: Path) -> None:
    panel = pd.read_csv(SOURCE / "horizon_panel.csv")
    panel = panel[(panel.panel == "primary") & panel.oracle_oeh_h.notna()]
    grouped = panel.groupby("policy_family")[["oracle_oeh_h_statistical", "oracle_oeh_h_coverage"]].median()
    trap = panel.groupby("policy_family").oracle_oeh_h.size()
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped.oracle_oeh_h_statistical, width=0.34, color=BLUE, label="statistical term")
    ax.bar(x + 0.18, grouped.oracle_oeh_h_coverage, width=0.34, color=ORANGE, label="coverage term")
    ax.set_xticks(x, [f"{name}\n(n={trap[name]})" for name in grouped.index], fontsize=7, color=INK)
    ax.set_ylabel("median, log_A transitions", fontsize=8, color=SECONDARY)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Oracle OEH-H decomposition by behavior family", fontsize=9, color=INK, loc="left")
    style(ax, xgrid=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    save(fig, out, "oeh_final_decomposition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    for render in (difficulty, contrasts, reliability, horizon, selection, decomposition):
        render(args.output)
        print(f"rendered {render.__name__}", flush=True)


if __name__ == "__main__":
    main()
