"""Figure of the sixth sealed panel for the paper (descriptive view of committed results).

Reads the committed panel-6 summaries, computes the relative improvement of OEH over each comparator on the
registered extrapolation endpoint (pooled 192 MDPs) and draws one column-width figure. The 95% bars are the
MDP-clustered intervals of the absolute difference divided by the comparator's mean score. Nothing is refitted.

    python -m scripts.analysis.paper_fig_panel6
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts/oeh_panel6_2026-10-15"
SOURCES = (PANEL / "summary.csv", PANEL / "tabpfn_descriptive.csv")
OUT = ROOT / "artifacts/week8/paper_fig_panel6_2026-10-16"
PAPER = ROOT / "paper/figures/fig_panel6"
ROWS = (
    ("b2_plus_prefix_fqe", "B2 + prefix/FQE (F1)"),
    ("b2_matched", "B2 matched (F2)"),
    ("b2_plus_lce", "B2 + LCE"),
    ("lce_raw", "LCE raw"),
    ("tabpfn_v2", "TabPFN v2 on B2"),
    ("robust", "Robust head (B2 + D)"),
    ("s4", "S4 (simplified)"),
    ("oeh_without_horizon", "OEH without OEH-H"),
    ("oeh_full_head", "OEH-F (no gate)"),
)
METRICS = (("ibs_ge", r"IBS$_{\geq}$", "#2a78d6", "o"), ("ll_ext", r"LL$_{\rm ext}$", "#eb6834", "s"))


def relative_gains() -> pd.DataFrame:
    """Relative improvement (%) of OEH over each comparator, pooled, with scaled 95% intervals."""
    summary = pd.read_csv(SOURCES[0])
    tabpfn = pd.read_csv(SOURCES[1])
    frame = pd.concat([summary, tabpfn[tabpfn.method == "tabpfn_v2"]], ignore_index=True)
    pooled = frame[frame.scope == "pooled"].set_index("method")
    rows = []
    for method, label in ROWS:
        r = pooled.loc[method]
        row = {"method": method, "label": label}
        for metric, *_ in METRICS:
            mean = float(r[metric])
            for key, column in (("gain", f"{metric}_gain_of_oeh"), ("low", f"{metric}_low95"),
                                ("high", f"{metric}_high95")):
                row[f"{metric}_{key}_pct"] = 100.0 * float(r[column]) / mean
        rows.append(row)
    return pd.DataFrame(rows)


def figure(gains: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    y = list(range(len(gains)))[::-1]
    for offset, (metric, name, color, marker) in zip((0.16, -0.16), METRICS):
        for yi, (_, r) in zip(y, gains.iterrows()):
            ax.plot([r[f"{metric}_low_pct"], r[f"{metric}_high_pct"]], [yi + offset] * 2, color=color, lw=1.4)
        ax.plot(gains[f"{metric}_gain_pct"], [yi + offset for yi in y], ls="none", marker=marker, ms=4.5,
                color=color, mec="white", mew=0.6, label=name)
    ax.axvline(0, color="0.55", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(gains.label, fontsize=6.8)
    ax.set_xlabel("relative improvement of OEH (%)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.grid(axis="x", color="0.92", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False)
    fig.tight_layout()
    for target in (OUT / "fig_panel6", PAPER):
        fig.savefig(target.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(target.with_suffix(".png"), dpi=200, metadata={"Software": None})
    plt.close(fig)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gains = relative_gains()
    gains.to_csv(OUT / "relative_gains.csv", index=False, lineterminator="\n", float_format="%.6f")
    figure(gains)
    outputs = [OUT / "relative_gains.csv", OUT / "fig_panel6.pdf", PAPER.with_suffix(".pdf")]
    manifest = {
        "status": "descriptive view of the committed sixth-panel results; nothing refitted",
        "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in SOURCES},
        "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in outputs},
    }
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(gains.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
