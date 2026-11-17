"""Build the manuscript pipeline figure from the frozen study design.

The figure is explanatory only: it reads no evaluation labels or predictions.
Run from the repository root with `python scripts/figures/build_oeh_pipeline.py`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, wh, title, detail, face, edge="#233454"):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=face, edgecolor=edge, linewidth=1.1,
        )
    )
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="#172748")
    ax.text(x + w / 2, y + h * 0.31, detail, ha="center", va="center",
            fontsize=7.15, color="#26354c", linespacing=1.2)


def arrow(ax, start, end, color="#364765", style="-|>", lw=1.25):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=10, linewidth=lw,
                                 color=color, connectionstyle="arc3"))


fig, ax = plt.subplots(figsize=(10.8, 2.45))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

source = "#f1f4f9"
pilot = "#e2f5f3"
label = "#fff1db"
forecast = "#e8e7fa"

box(ax, (.015, .385), (.14, .27), "MDP + behavior", "Synthetic kernels or\npublished BRIDGE MDPs", source)
box(ax, (.205, .665), (.17, .25), "Pilot log $D_0$", "Independent stream;\nfeatures use pilot only", pilot)
box(ax, (.205, .12), (.17, .25), "Training streams", "Six nested budgets;\n10 seeds per learner", label)
box(ax, (.43, .665), (.17, .25), "OEH descriptors", "Posterior, planning risk,\ncoverage and context", pilot)
box(ax, (.655, .665), (.15, .25), "Frozen forecaster", "Fit on opened\nmeta-training MDPs", forecast)
box(ax, (.845, .665), (.14, .25), "Forecast", "$P(E_l=c\\mid D_0,l)$\nfor seven classes", forecast)
box(ax, (.43, .12), (.17, .25), "BC / CFQI / IQL", "Train offline; score\nnormalized return", label)
box(ax, (.655, .12), (.15, .25), "Measured label", "First stable $x1,\\ldots,x50$\nor right-censored $>x50$", label)
box(ax, (.845, .12), (.14, .25), "Evaluation", "Brier, log loss;\npaired MDP analysis", source)

arrow(ax, (.155, .56), (.205, .79))
arrow(ax, (.155, .47), (.205, .245))
arrow(ax, (.375, .79), (.43, .79))
arrow(ax, (.60, .79), (.655, .79))
arrow(ax, (.805, .79), (.845, .79))
arrow(ax, (.375, .245), (.43, .245))
arrow(ax, (.60, .245), (.655, .245))
arrow(ax, (.805, .245), (.845, .245))
arrow(ax, (.915, .665), (.915, .37), color="#57708e")
ax.text(.50, .505, "No test-label information enters the forecast path",
        ha="center", va="center", fontsize=8, style="italic", color="#57677d")
ax.text(.292, .965, "Forecast path", ha="center", fontsize=8.2,
        fontweight="bold", color="#217672")
ax.text(.292, .045, "Label path", ha="center", fontsize=8.2,
        fontweight="bold", color="#a56925")

fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
for extension in ("pdf", "png"):
    fig.savefig(OUT / f"oeh_pipeline.{extension}", dpi=220, facecolor="white")
plt.close(fig)
