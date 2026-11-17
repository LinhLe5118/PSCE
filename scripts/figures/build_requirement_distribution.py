"""Plot the sealed 120-MDP primary-panel requirement labels for the thesis.

Only the published post-outcome diagnostic CSV is read. This plot is descriptive;
the counts are not used to fit or alter any forecaster.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "week8" / "difficulty_diagnostics_exploratory_2026-10-07" / "difficulty_distribution.csv"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

learners = {
    "behavior_cloning_tabular": "BC",
    "cfqi_tabular": "CFQI",
    "iql_tabular": "IQL",
}
budgets = ["x1", "x2", "x5", "x10", "x20", "x50", ">x50"]
colors = ["#285c86", "#397ba8", "#53a0ba", "#6ebdb2", "#a3cfa4", "#e2b76e", "#8d99ab"]
counts = defaultdict(lambda: np.zeros(7, dtype=int))

with SOURCE.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["panel"] != "primary":
            continue
        if row["algorithm"] not in learners:
            raise ValueError(f"Unexpected learner: {row['algorithm']}")
        event = int(row["event"])
        if not 0 <= event <= 6:
            raise ValueError(f"Unexpected class: {event}")
        counts[row["algorithm"]][event] += int(row["rows"])

for learner in learners:
    if int(counts[learner].sum()) != 420:
        raise ValueError(f"Primary panel does not have 420 rows for {learner}")

fig, ax = plt.subplots(figsize=(8.4, 3.05))
left = np.zeros(3, dtype=float)
for event, (budget, color) in enumerate(zip(budgets, colors)):
    values = np.array([counts[key][event] / 420 for key in learners])
    ax.barh(list(learners.values()), values, left=left, height=.57,
            label=budget, color=color, edgecolor="white", linewidth=.6)
    left += values

ax.set_xlim(0, 1)
ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
ax.set_xlabel("Share of MDP–behavior rows (%)")
ax.invert_yaxis()
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=.2)
ax.set_axisbelow(True)
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(.5, 1.27), frameon=False)
for i, learner in enumerate(learners):
    censored = counts[learner][6]
    ax.text(.99, i, f"{censored}/420 censored", ha="right", va="center",
            fontsize=8, color="#192640", fontweight="bold")
fig.subplots_adjust(left=.10, right=.985, bottom=.22, top=.81)
for extension in ("pdf", "png"):
    fig.savefig(OUT / f"requirement_distribution.{extension}", dpi=220,
                facecolor="white")
plt.close(fig)
