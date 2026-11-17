"""Tables, numbers and figures for thesis Chapter 3 (Dataset and Data Generation).

Reads only opened artifacts of the Devoted repository (labels, pilot features,
sealed predictions, seed outcomes and protocols of panels whose analysis is
complete) and writes everything into the chapter folder (figures/, tables/, numbers.json, manifest.json).
Panel 6 is described from its protocol only: no outcome file of it is read.

Usage (repository root):  python docs/thesis/chapter3_dataset_2026-10-16/scripts/build_ch3_assets.py
Needs gymnasium (1.0.0) only for the FrozenLake map of Figure 3.2.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]  # repository root
A = REPO / "artifacts"
OUT = Path(__file__).resolve().parents[1]
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

KEYS = ["mdp_id", "policy_family", "algorithm"]
B2 = ["log_pilot_transitions", "log_state_action_count", "observed_sa_fraction", "observed_state_fraction",
      "normalized_visitation_entropy", "log1p_min_positive_sa_count", "trajectory_return_mean",
      "trajectory_return_std", "trajectory_return_iqr", "terminal_trajectory_fraction",
      "weighted_transition_entropy", "observed_cell_transition_entropy", "low_count_cell_fraction",
      "empirical_stochastic_cell_fraction", "low_coverage_entropy", "high_coverage_entropy",
      "bootstrap_return_mean_std", "bootstrap_rare_exposure_mean_std"]
CLASSES = ["x1", "x2", "x5", "x10", "x20", "x50", ">x50"]
LEARNER = {"behavior_cloning_tabular": "BC", "cfqi_tabular": "CFQI", "iql_tabular": "IQL"}
LCOLOR = {"BC": "#4C72B0", "CFQI": "#DD8452", "IQL": "#55A868"}
BEHAVIOR = ["q_star_softmax", "perturbed_reward", "shifted_dynamics", "structured_random", "two_policy_mixture"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})

# --------------------------------------------------------------------------- #
# Registry of opened panels
# --------------------------------------------------------------------------- #
V2 = A / "oeh_v2_sealed_test_2026-10-14"
P5 = A / "oeh_fifth_panel_2026-10-15"
P6 = A / "oeh_panel6_2026-10-15"
C8 = A / "oeh_eighth_cohort_2026-10-14"
PANELS = {
    # name: (group, role, labels, features, outcomes, source)
    "E1": ("meta", "Meta-training", A / "difficulty_e1_posthoc_2026-09-30/labels.csv",
           A / "week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv", None, "Project generator"),
    "Validation-60 (1st)": ("meta", "Meta-training", A / "independent_difficulty_2026-09-30/labels.csv",
                            A / "independent_validation_2026-09-30/pilot_features.csv", None, "Project generator"),
    "Validation-60 (2nd)": ("meta", "Meta-training", A / "oeh_difficulty_confirmation_2026-10-01/labels.csv",
                            A / "oeh_difficulty_confirmation_2026-10-01/pilot_features.csv", None, "Project generator"),
    "Cohort 1": ("meta", "Meta-training", A / "oeh_difficulty_unseen_120_2026-10-01/labels.csv",
                 A / "oeh_difficulty_unseen_120_2026-10-01/pilot_features.csv", None, "Project generator"),
    "Cohort 2": ("meta", "Meta-training", A / "oeh_pooled_successor_unseen_120_2026-10-01/labels.csv",
                 A / "oeh_pooled_successor_unseen_120_2026-10-01/pilot_features.csv", None, "Project generator"),
    "Cohort 3": ("meta", "Meta-training", A / "oeh_cross_capacity_unseen_120_2026-10-01/labels.csv",
                 A / "oeh_cross_capacity_unseen_120_2026-10-01/pilot_features.csv", None, "Project generator"),
    "Cohort 4": ("meta", "Meta-training", A / "oeh_final_unseen_120_2026-10-02/labels.csv",
                 A / "oeh_final_unseen_120_2026-10-02/pilot_features.csv",
                 A / "oeh_final_unseen_120_2026-10-02/all_core_budget_seed_outcomes.csv", "Project generator"),
    "Cohort 7": ("dev", "Development / selection", A / "oeh_confirmatory_unseen_120_2026-10-04/labels.csv",
                 A / "oeh_confirmatory_unseen_120_2026-10-04/pilot_features.csv",
                 A / "oeh_confirmatory_unseen_120_2026-10-04/all_core_budget_seed_outcomes.csv", "Project generator"),
    "Cohort 8": ("dev", "Development / selection", C8 / "internal/labels.csv", C8 / "internal/query_preoutcome.csv.gz",
                 C8 / "internal/outcomes.csv.gz", "Project generator"),
    "Colosseum 2027": ("dev", "Development / selection", A / "week8/colosseum_confirm_2026-10-14/colosseum_labels.csv",
                       A / "week8/colosseum_confirm_2026-10-14/colosseum_pilot_features.csv",
                       A / "week8/colosseum_confirm_2026-10-14/colosseum_outcomes.csv", "Colosseum"),
    "Colosseum 2028": ("dev", "Development / selection", C8 / "colosseum/labels.csv", C8 / "colosseum/query_preoutcome.csv.gz",
                       C8 / "colosseum/outcomes.csv.gz", "Colosseum"),
    "SPIBB": ("dev", "Development / selection", A / "oeh_spibb_panel_2026-10-14/spibb/labels.csv",
              A / "oeh_spibb_panel_2026-10-14/spibb/query_preoutcome.csv.gz",
              A / "oeh_spibb_panel_2026-10-14/spibb/outcomes.csv.gz", "SPIBB author code"),
    "Behavior cohort (M3)": ("dev", "Development (rejected candidate)", A / "oeh_v2_behavior_cohort_2026-10-14/labels.csv",
                             None, A / "oeh_v2_behavior_cohort_2026-10-14/outcomes.csv.gz", "Project generator"),
    "Final internal": ("final", "Final test 1", V2 / "internal/labels.csv", V2 / "internal/pilot_features.csv",
                       V2 / "internal/outcomes.csv.gz", "Project generator"),
    "Colosseum 2029": ("final", "Final test 1", V2 / "colosseum2029/labels.csv", V2 / "colosseum2029/pilot_features.csv",
                       V2 / "colosseum2029/outcomes.csv.gz", "Colosseum"),
    "SPIBB-2": ("final", "Final test 1", V2 / "spibb2/labels.csv", V2 / "spibb2/pilot_features.csv",
                V2 / "spibb2/outcomes.csv.gz", "SPIBB author code"),
    "Forest": ("final", "Final test 1", V2 / "forest/labels.csv", V2 / "forest/pilot_features.csv",
               V2 / "forest/outcomes.csv.gz", "MDPToolbox"),
    "FrozenLake": ("p5", "Final test 2", P5 / "frozenlake/labels.csv", P5 / "frozenlake/pilot_features.csv",
                   P5 / "frozenlake/outcomes.csv.gz", "Gymnasium"),
    "Gambler": ("p5", "Final test 2", P5 / "gambler/labels.csv", P5 / "gambler/pilot_features.csv",
                P5 / "gambler/outcomes.csv.gz", "Sutton & Barto"),
    "MDP Playground": ("p6", "Final test 3", P6 / "mdpp/labels.csv", P6 / "mdpp/pilot_features.csv",
                       P6 / "mdpp/outcomes.csv.gz", "MDP Playground"),
    "labmaze": ("p6", "Final test 3", P6 / "labmaze/labels.csv", P6 / "labmaze/pilot_features.csv",
                P6 / "labmaze/outcomes.csv.gz", "labmaze"),
}
FINAL = ["Final internal", "Colosseum 2029", "SPIBB-2", "Forest"]
PROFILE = ["Meta-training (720)", *FINAL, "FrozenLake", "Gambler", "MDP Playground", "labmaze"]
PCOLOR = {"Meta-training (720)": "#9E9E9E", "Final internal": "#4C72B0", "Colosseum 2029": "#C44E52",
          "SPIBB-2": "#8172B2", "Forest": "#55A868", "FrozenLake": "#64B5CD", "Gambler": "#CCB974",
          "MDP Playground": "#DA8BC3", "labmaze": "#8C613C"}
INPUTS: dict[str, str] = {}


def sha(path: Path) -> str:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    INPUTS[str(Path(path).relative_to(REPO)).replace("\\", "/")] = h
    return h


def read(path: Path) -> pd.DataFrame:
    sha(path)
    return pd.read_csv(path)


def family(mdp_id: str) -> str:
    s = mdp_id.split("/")[-1]
    for f in ("uniform_garnet", "forward_layered", "bottleneck", "lock_structured", "riverswim", "deepsea",
              "frozenlake", "simplegrid", "forest", "gambler", "spibb"):
        if f in s:
            return f
    return "other"


labels, feats = {}, {}
for name, (group, role, lab, fea, _, _) in PANELS.items():
    d = read(lab)
    if "panel" in d:  # generator cohorts carry the deterministic-control rows separately
        d["primary"] = d["panel"].eq("primary")
    else:
        d["primary"] = True
    d["family"] = d.mdp_id.map(family)
    d["learner"] = d.algorithm.map(LEARNER)
    labels[name] = d
    if fea is not None:
        feats[name] = read(fea)

# --------------------------------------------------------------------------- #
# Registry table
# --------------------------------------------------------------------------- #
GAMMA_H = "0.95 / 100"
reg = []
for name, (group, role, lab, fea, out, src) in PANELS.items():
    d = labels[name]
    nb = 8 if "successes_x200" in d else 6
    sa = np.exp(feats[name].drop_duplicates(["mdp_id"]).log_state_action_count) if name in feats else None
    prim = d[d.primary]
    reg.append(dict(
        panel=name, group=group, role=role, source=src, mdps=d.mdp_id.nunique(),
        primary_mdps=prim.mdp_id.nunique(), behaviors=d.policy_family.nunique(),
        groups=d[["mdp_id", "policy_family"]].drop_duplicates().shape[0], rows=len(d),
        budgets=nb, jobs=len(d) * nb * 10,
        sa_min=None if sa is None else int(round(sa.min())), sa_max=None if sa is None else int(round(sa.max())),
        censored_pct=100 * prim.event.eq(6).mean(), gamma_horizon=GAMMA_H))
reg = pd.DataFrame(reg)
reg.to_csv(TAB / "registry.csv", index=False)

meta = reg[reg.group == "meta"]
numbers = {
    "meta_mdps": int(meta.mdps.sum()), "meta_rows": int(meta.rows.sum()), "meta_jobs": int(meta.jobs.sum()),
    "final_mdps": int(reg[reg.group == "final"].mdps.sum()), "final_rows": int(reg[reg.group == "final"].rows.sum()),
    "final_jobs": int(reg[reg.group == "final"].jobs.sum()),
    "final_external_mdps": int(reg[reg.panel.isin(FINAL[1:])].mdps.sum()),
    "all_jobs_opened": int(reg.jobs.sum()), "all_rows_opened": int(reg.rows.sum()),
    "all_mdps_opened": int(reg.mdps.sum()),
}

# --------------------------------------------------------------------------- #
# Label distributions
# --------------------------------------------------------------------------- #
dist = []
for name in labels:
    d = labels[name][labels[name].primary]
    for lr, g in d.groupby("learner"):
        c = g.event.value_counts().reindex(range(7), fill_value=0)
        dist.append(dict(panel=name, learner=lr, n=len(g), **{CLASSES[k]: int(c[k]) for k in range(7)},
                         censored_pct=100 * c[6] / len(g)))
dist = pd.DataFrame(dist)
dist.to_csv(TAB / "label_distribution.csv", index=False)

beh = []
for name in labels:
    d = labels[name][labels[name].primary]
    for b, g in d.groupby("policy_family"):
        beh.append(dict(panel=name, behavior=b, n=len(g), censored_pct=100 * g.event.eq(6).mean()))
beh = pd.DataFrame(beh)
beh.to_csv(TAB / "censoring_by_behavior.csv", index=False)

fam = []
for name in labels:
    d = labels[name][labels[name].primary]
    for f, g in d.groupby("family"):
        fam.append(dict(panel=name, family=f, mdps=g.mdp_id.nunique(), n=len(g), censored_pct=100 * g.event.eq(6).mean(),
                        **{f"{lr}_censored_pct": 100 * g[g.learner == lr].event.eq(6).mean() for lr in LCOLOR}))
fam = pd.DataFrame(fam)
fam.to_csv(TAB / "censoring_by_family.csv", index=False)

# Label noise: non-monotone success counts and closeness to the 8/10 threshold
noise = []
for name in labels:
    d = labels[name][labels[name].primary]
    cols = [f"successes_x{b}" for b in (1, 2, 5, 10, 20, 50)]
    if not all(c in d for c in cols):
        continue
    s = d[cols].to_numpy()
    dips = (np.diff(s, axis=1) < 0).any(1)
    cross = ((s[:, :-1] >= 8) & (s[:, 1:] < 8)).any(1)  # reached the threshold and fell back below it
    near = (np.abs(s - 8) <= 1).any(1)
    row = dict(panel=name, rows=len(d), nonmonotone_pct=100 * dips.mean(), fallback_pct=100 * cross.mean(),
               near_threshold_pct=100 * near.mean())
    if "event_extended" in d:
        c = d[d.event == 6]
        row.update(censored=len(c), resolved_x100=int((c.event_extended == 6).sum()),
                   resolved_x200=int((c.event_extended == 7).sum()), still_censored=int((c.event_extended == 8).sum()))
    noise.append(row)
noise = pd.DataFrame(noise)
noise.to_csv(TAB / "label_noise.csv", index=False)

# --------------------------------------------------------------------------- #
# Pilot profiles (one row per MDP-behavior group; features are learner-invariant)
# --------------------------------------------------------------------------- #
def groups_of(name):
    f = feats[name].drop_duplicates(["mdp_id", "policy_family"]).copy()
    if name in labels and "panel" in labels[name]:
        prim = set(labels[name].loc[labels[name].primary, "mdp_id"])
        f = f[f.mdp_id.isin(prim)]
    return f


meta_f = pd.concat([groups_of(n) for n in PANELS if PANELS[n][0] == "meta"], ignore_index=True)
prof = {"Meta-training (720)": meta_f, **{n: groups_of(n) for n in PROFILE[1:]}}
SHOW = {"observed_sa_fraction": "Observed state–action fraction", "normalized_visitation_entropy": "Normalized visitation entropy",
        "empirical_stochastic_cell_fraction": "Stochastic-cell fraction", "low_count_cell_fraction": "Low-count cell fraction",
        "weighted_transition_entropy": "Weighted transition entropy", "trajectory_return_mean": "Mean trajectory return"}
summ = []
for name, f in prof.items():
    row = dict(panel=name, groups=len(f), sa_median=float(np.exp(f.log_state_action_count).median()))
    for c in SHOW:
        q = f[c].quantile([.25, .5, .75]).to_numpy()
        row[c] = f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"
    summ.append(row)
summ = pd.DataFrame(summ)
summ.to_csv(TAB / "pilot_profiles.csv", index=False)

# Shift distance d from sealed predictions (same value for every method of a row)
dist_d = {}
for name, path in (("Final internal", V2 / "internal/predictions_preoutcome.csv"),
                   ("Colosseum 2029", V2 / "colosseum2029/predictions_preoutcome.csv"),
                   ("SPIBB-2", V2 / "spibb2/predictions_preoutcome.csv"), ("Forest", V2 / "forest/predictions_preoutcome.csv"),
                   ("FrozenLake", P5 / "frozenlake/predictions_preoutcome.csv"), ("Gambler", P5 / "gambler/predictions_preoutcome.csv"),
                   ("MDP Playground", P6 / "mdpp/predictions_preoutcome.csv"), ("labmaze", P6 / "labmaze/predictions_preoutcome.csv")):
    p = read(path)
    if "panel" in p:
        p = p[p.panel == "primary"]
    dist_d[name] = p.drop_duplicates(["mdp_id", "policy_family"])["distance"].to_numpy()
shift = pd.DataFrame([dict(panel=n, groups=len(v), median=np.median(v), q25=np.quantile(v, .25), q75=np.quantile(v, .75),
                           share_ge3=100 * np.mean(v >= 3)) for n, v in dist_d.items()])
shift.to_csv(TAB / "shift_distance.csv", index=False)

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def box(ax, x, y, w, h, text, fc, ec="#333333", fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else None, wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#333333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=color, lw=1.0, ls=ls))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, 0.1, 1.9, 1.45, 0.9, "MDP $M$\n(kernel $P$, reward $R$,\n$\\rho$, $\\gamma$)", "#E8EEF7", bold=False)
    box(ax, 0.1, 0.55, 1.45, 0.9, "Behavior\npolicy $\\beta$", "#E8EEF7")
    box(ax, 2.0, 3.25, 2.0, 0.9, "Pilot stream\n(seed namespace A)\nprefixes x2, x5, x10", "#FFF4D6")
    box(ax, 2.0, 0.55, 2.0, 1.55, "10 training streams\n(seed namespace B)\nnested prefixes\nx1, x2, x5, x10, x20, x50\n(+ x100, x200)", "#FFF4D6")
    box(ax, 4.5, 3.25, 1.8, 0.9, "Pilot $D_0$ (x10)\n= $10\\,|S||A|$\ntransitions", "#FDE2C8")
    box(ax, 4.5, 0.55, 1.8, 1.55, "Train learner\nBC / CFQI / IQL\non every prefix\nand seed", "#DCEFD9")
    box(ax, 6.8, 0.55, 1.6, 1.55, "Exact policy\nevaluation in $M$\n→ normalized\nreturn", "#DCEFD9")
    box(ax, 8.75, 0.55, 1.35, 1.55, "Success counts\n(of 10 seeds)\n→ label $E$\n(7 classes)", "#D8E4F2", bold=True)
    box(ax, 6.8, 3.25, 3.3, 0.9, "Forecaster input: pilot + public metadata\n→ 7-class forecast (sealed before labels)", "#F3E0EC")
    arrow(ax, 1.55, 2.35, 2.0, 3.6)
    arrow(ax, 1.55, 2.35, 2.0, 1.4)
    arrow(ax, 1.55, 1.0, 2.0, 3.45)
    arrow(ax, 1.55, 1.0, 2.0, 1.0)
    arrow(ax, 4.0, 3.7, 4.5, 3.7)
    arrow(ax, 4.0, 1.32, 4.5, 1.32)
    arrow(ax, 6.3, 1.32, 6.8, 1.32)
    arrow(ax, 8.4, 1.32, 8.75, 1.32)
    arrow(ax, 6.3, 3.7, 6.8, 3.7)
    arrow(ax, 9.42, 2.1, 9.42, 3.25, color="#9C2B2B", style="<|-|>", ls="--")
    ax.text(9.5, 2.65, "scored only\nafter seal", fontsize=7.5, color="#9C2B2B", va="center")
    ax.text(5.4, 2.62, "no transition is shared between the two branches", ha="center", fontsize=7.8,
            style="italic", color="#555555")
    ax.plot([2.0, 6.3], [2.45, 2.45], color="#999999", lw=0.8, ls=":")
    ax.text(0.1, 4.35, "Forecast branch", fontsize=8.5, weight="bold", color="#7A4A00")
    ax.text(0.1, 0.18, "Label branch", fontsize=8.5, weight="bold", color="#2E6B2E")
    fig.savefig(FIG / "fig3_1_pipeline.png")
    plt.close(fig)


def fig_gallery():
    """Schematic examples of every environment source (not drawn to scale)."""
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    proto = json.loads((P5 / "protocol.json").read_text())
    seal = json.loads((P5 / "frozenlake/dataset_seal.json").read_text())
    sha(P5 / "protocol.json")
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def node(ax, x, y, c="#FFFFFF", r=0.16, ec="#333333", text=None):
        ax.add_patch(plt.Circle((x, y), r, fc=c, ec=ec, lw=0.9, zorder=3))
        if text:
            ax.text(x, y, text, ha="center", va="center", fontsize=6.5, zorder=4)

    def edge(ax, x0, y0, x1, y1, c="#888888", lw=0.8, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=c, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=5, shrinkB=5, zorder=2))

    # (a) uniform Garnet: random successors
    ax = axes[0, 0]
    pts = np.c_[np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]), np.sin(np.linspace(0, 2 * np.pi, 9)[:-1])]
    for i, (x, y) in enumerate(pts):
        for j in rng.choice([k for k in range(8) if k != i], 2, replace=False):
            edge(ax, x, y, *pts[j], rad=0.15)
    for i, (x, y) in enumerate(pts):
        node(ax, x, y, "#FFD54F" if i == 3 else "#FFFFFF")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("(a) Uniform Garnet\nrandom branching, sparse reward")
    # (b) forward-layered
    ax = axes[0, 1]
    L, W = 5, 3
    for l in range(L - 1):
        for w in range(W):
            edge(ax, l, w, l + 1, w, c="#333333", lw=1.1)
            edge(ax, l, w, l + 1, (w + 1) % W, c="#BBBBBB")
    for l in range(L):
        for w in range(W):
            node(ax, l, w, "#FFD54F" if l == L - 1 and w == 1 else "#FFFFFF")
    ax.set_xlim(-0.6, L - 0.4)
    ax.set_ylim(-0.8, W - 0.2)
    ax.set_title("(b) Forward-layered\nbackbone + branching, late reward")
    # (c) bottleneck
    ax = axes[0, 2]
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    left = np.c_[-1.35 + 0.6 * np.cos(t), 0.6 * np.sin(t)]
    right = np.c_[1.35 + 0.6 * np.cos(t + np.pi), 0.6 * np.sin(t + np.pi)]
    for P_ in (left, right):
        for i in range(6):
            edge(ax, *P_[i], *P_[(i + 1) % 6], rad=0.25)
            edge(ax, *P_[i], *P_[(i + 3) % 6], c="#CCCCCC")
    edge(ax, *left[0], *right[0], c="#C44E52", lw=1.6)
    ax.text(0, 0.18, "bridge", ha="center", fontsize=7, color="#C44E52")
    for i, (x, y) in enumerate(left):
        node(ax, x, y, "#F8BBD0" if i == 0 else "#FFFFFF")
    for i, (x, y) in enumerate(right):
        node(ax, x, y, "#F8BBD0" if i == 0 else ("#FFD54F" if i == 3 else "#FFFFFF"))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("(c) Bottleneck\ntwo communities, rare bridge")
    # (d) combination lock
    ax = axes[0, 3]
    n = 6
    for i in range(n - 1):
        edge(ax, i, 0, i + 1, 0, c="#333333", lw=1.1)
        edge(ax, i + 1, 0, 0, 0, c="#C44E52", rad=-0.45 if i % 2 else 0.45, lw=0.6)
    for i in range(n):
        node(ax, i, 0, "#FFD54F" if i == n - 1 else "#FFFFFF", text=str(i))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-2.2, 2.2)
    ax.set_title("(d) Combination lock (control)\ndeterministic; wrong action resets")
    # (e) Colosseum RiverSwim
    ax = axes[1, 0]
    n = 6
    for i in range(n - 1):
        edge(ax, i, 0, i + 1, 0, c="#333333", rad=0.35)
        edge(ax, i + 1, 0, i, 0, c="#BBBBBB", rad=0.35)
    for i in range(n):
        node(ax, i, 0, "#FFD54F" if i == n - 1 else ("#E3F2FD" if i == 0 else "#FFFFFF"))
    ax.text(0, -0.85, "small\nreward", ha="center", fontsize=6.5)
    ax.text(n - 1, -0.85, "large\nreward", ha="center", fontsize=6.5)
    ax.text((n - 1) / 2, 1.1, "Colosseum families: RiverSwim, DeepSea,\nFrozenLake, SimpleGrid (authors' sampler)",
            ha="center", fontsize=6.6, color="#555555")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-1.3, 1.6)
    ax.set_title("(e) Colosseum (e.g. RiverSwim)\nstochastic current against the agent")
    # (f) FrozenLake: a registered fifth-panel map
    ax = axes[1, 1]
    ids = set(labels["FrozenLake"].mdp_id)
    kept = [r for r in seal["selection"] if r["passes_filter"] and r["mdp_id"] in ids]
    holes = [sum(row.count("H") for row in generate_random_map(size=r["size"], p=r["p"], seed=r["map_seed"])) for r in kept]
    c = kept[int(np.argmax(holes))]  # registered map with the most holes, for illustration
    desc = generate_random_map(size=c["size"], p=c["p"], seed=c["map_seed"])
    col = {"S": "#90CAF9", "F": "#E3F2FD", "H": "#37474F", "G": "#FFD54F"}
    for r, rowtxt in enumerate(desc):
        for q, ch in enumerate(rowtxt):
            ax.add_patch(plt.Rectangle((q, -r), 1, 1, fc=col[ch], ec="white"))
            if ch in "SG":
                ax.text(q + .5, -r + .5, ch, ha="center", va="center", fontsize=8, weight="bold")
    ax.set_xlim(0, len(desc[0]))
    ax.set_ylim(-len(desc) + 1, 1)
    ax.set_aspect("equal")
    ax.set_title(f"(f) FrozenLake {c['size']}×{c['size']} (slippery)\nregistered map, frozen p = {c['p']}")
    # (g) forest management
    ax = axes[1, 2]
    n = 5
    for i in range(n - 1):
        edge(ax, i, 0, i + 1, 0, c="#2E7D32", lw=1.1)
    for i in range(1, n):
        edge(ax, i, 0, 0, 0, c="#C44E52", rad=-0.5, lw=0.7)
    for i in range(n):
        node(ax, i, 0, "#C8E6C9" if i < n - 1 else "#FFD54F", text=str(i))
    ax.text(2, 0.45, "Wait: age + 1 (oldest state keeps its age)", ha="center", fontsize=6.5, color="#2E7D32")
    ax.text(2, -1.55, "Cut (reward), or fire with prob. p while waiting: age → 0", ha="center", fontsize=6.5, color="#C44E52")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-1.8, 1.2)
    ax.set_title("(g) Forest management (MDPToolbox)\nstate = forest age; Wait / Cut")
    # (h) gambler
    ax = axes[1, 3]
    N = 8
    for s in range(1, N):
        edge(ax, s, 0, min(N, s + 2), 0, c="#2E7D32", rad=-0.5, lw=0.7)
        edge(ax, s, 0, max(0, s - 2), 0, c="#C44E52", rad=-0.5, lw=0.7)
    for s in range(N + 1):
        node(ax, s, 0, "#FFD54F" if s == N else ("#37474F" if s == 0 else "#FFFFFF"), r=0.22, text=str(s))
    ax.text(N / 2, 1.25, "win stake with $p_h$", ha="center", fontsize=6.5, color="#2E7D32")
    ax.text(N / 2, -1.35, "lose stake with $1-p_h$", ha="center", fontsize=6.5, color="#C44E52")
    ax.set_xlim(-0.6, N + 0.6)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title("(h) Gambler's problem\ncapital 0..N; reward on reaching N")
    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="datalim")
    fig.text(0.5, 0.005, "Yellow: rewarding state/cell. Schematics illustrate structure only; sizes and probabilities are sampled per MDP.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "fig3_2_environment_gallery.png")
    plt.close(fig)


def fig_label_example():
    """Nested budgets and the sustained-success rule on one final-internal group."""
    d = labels["Final internal"]
    cand = d.pivot_table(index=["mdp_id", "policy_family"], columns="learner", values="event")
    cand = cand[(cand.nunique(axis=1) == 3) & (cand.max(axis=1) <= 6)]
    # prefer a group with two finite labels and one censored label
    pick = cand[(cand == 6).sum(1) == 1]
    pick = pick if len(pick) else cand
    key = sorted(pick.index)[0]
    out = read(V2 / "internal/outcomes.csv.gz")
    o = out[(out.mdp_id == key[0]) & (out.policy_family == key[1])].copy()
    o["learner"] = o.algorithm.map(LEARNER)
    budgets = [1, 2, 5, 10, 20, 50, 100, 200]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(budgets))
    for i, lr in enumerate(["BC", "CFQI", "IQL"]):
        g = o[o.learner == lr]
        for k, b in enumerate(budgets):
            v = g[g.budget == b].normalized_return.to_numpy()
            ax.scatter(np.full(len(v), k + (i - 1) * 0.22), np.clip(v, -0.3, 1.05), s=9, color=LCOLOR[lr], alpha=0.75,
                       label=lr if k == 0 else None)
    ax.axhline(0.95, color="#9C2B2B", ls="--", lw=0.9, label="success: NR ≥ 0.95")
    ax.axvspan(5.5, 7.5, color="#EEEEEE", zorder=0)
    ax.text(6.5, 0.03, "secondary\nbudgets", ha="center", fontsize=7, color="#666666",
            transform=ax.get_xaxis_transform())
    ax.set_xticks(x, [f"x{b}" for b in budgets])
    ax.set_ylabel("Normalized return of each seed")
    ax.set_xlabel("Training budget (multiple of |S||A| transitions)")
    ax.set_title("(a) Ten seeds per budget, three learners")
    ax.legend(frameon=True, framealpha=0.9, ncol=2, loc="lower left", fontsize=7.5)
    ax = axes[1]
    row = d[(d.mdp_id == key[0]) & (d.policy_family == key[1])].set_index("learner")
    for lr in ["BC", "CFQI", "IQL"]:
        s = row.loc[lr, [f"successes_x{b}" for b in budgets]].to_numpy(int)
        ax.plot(x, s, marker="o", color=LCOLOR[lr], label=f"{lr}: E = {CLASSES[int(row.loc[lr, 'event'])]}")
    ax.axhline(8, color="#9C2B2B", ls="--", lw=0.9)
    ax.text(0, 8.2, "8/10 seeds", color="#9C2B2B", fontsize=7.5)
    ax.axvspan(5.5, 7.5, color="#EEEEEE", zorder=0)
    ax.set_xticks(x, [f"x{b}" for b in budgets])
    ax.set_ylim(-0.4, 10.6)
    ax.set_ylabel("Seeds with success (of 10)")
    ax.set_xlabel("Training budget")
    ax.set_title("(b) Label = first budget of a sustained ≥ 8/10 suffix")
    ax.legend(frameon=False, fontsize=7.5, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_3_label_example.png")
    plt.close(fig)
    return dict(example_group=f"{key[0]} / {key[1]}",
                example_labels={lr: CLASSES[int(row.loc[lr, "event"])] for lr in ["BC", "CFQI", "IQL"]},
                example_counts={lr: row.loc[lr, [f"successes_x{b}" for b in budgets]].astype(int).tolist()
                                for lr in ["BC", "CFQI", "IQL"]})


def fig_registry():
    order = list(PANELS)
    gcol = {"meta": "#9E9E9E", "dev": "#F2B447", "final": "#4C72B0", "p5": "#55A868", "p6": "#8E44AD"}
    gname = {"meta": "Meta-training (fit)", "dev": "Development / selection (opened)",
             "final": "Final test 1 (4 panels)", "p5": "Final test 2 (fifth panel)", "p6": "Final test 3 (sixth panel)"}
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    y = np.arange(len(order))[::-1]
    r = reg.set_index("panel").loc[order]
    ax.barh(y, r.mdps, color=[gcol[g] for g in r.group], edgecolor="white")
    for yi, (n, m, rows, cen) in zip(y, zip(order, r.mdps, r.rows, r.censored_pct)):
        ax.text(m + 2, yi, f"{m} MDPs · {rows:,} rows · {cen:.0f}% >x50", va="center", fontsize=7.4)
    ax.set_yticks(y, [f"{n}  [{r.loc[n, 'source']}]" for n in order], fontsize=7.6)
    ax.set_xlabel("Number of MDPs")
    ax.set_xlim(0, 260)
    ax.legend(handles=[mpatches.Patch(color=gcol[g], label=gname[g]) for g in gcol], frameon=False, loc="lower right",
              fontsize=7.5)
    ax.set_title("Opened panels by role (chronological within each role)")
    fig.savefig(FIG / "fig3_4_registry.png")
    plt.close(fig)


def fig_profiles():
    cols = list(SHOW)
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6))
    names = list(prof)
    for ax, c in zip(axes.flat, cols):
        data = [prof[n][c].dropna().to_numpy() for n in names]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for b, n in zip(parts["bodies"], names):
            b.set_facecolor(PCOLOR[n])
            b.set_alpha(0.75)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color("#333333")
            parts[k].set_linewidth(0.7)
        ax.set_xticks(range(1, len(names) + 1), [n.replace(" (720)", "").replace("Final internal", "Final int.")
                                                for n in names], rotation=35, ha="right", fontsize=7.2)
        ax.set_title(SHOW[c])
    fig.tight_layout()
    fig.savefig(FIG / "fig3_5_pilot_profiles.png")
    plt.close(fig)


def fig_pca():
    X = meta_f[B2].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    W = vt[:2].T
    expl = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    expl = expl / expl.sum()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for n in prof:
        P = ((prof[n][B2].to_numpy(float) - mu) / sd - Z.mean(0)) @ W
        ax.scatter(P[:, 0], P[:, 1], s=6 if n.startswith("Meta") else 9, alpha=0.35 if n.startswith("Meta") else 0.75,
                   color=PCOLOR[n], label=n, lw=0)
    ax.set_xlabel(f"PC1 ({100 * expl[0]:.0f}% of meta-training variance)")
    ax.set_ylabel(f"PC2 ({100 * expl[1]:.0f}%)")
    ax.legend(frameon=False, fontsize=7.5, markerscale=2.2, loc="best")
    ax.set_title("Pilot statistics (18 B2 fields), standardized and projected on meta-training PCs")
    fig.savefig(FIG / "fig3_6_pca_shift.png")
    plt.close(fig)
    return dict(pca_explained=[float(expl[0]), float(expl[1])])


def fig_labels():
    panels = ["Meta-training (720)", "Cohort 7", "Cohort 8", "Colosseum 2027", "Colosseum 2028", "SPIBB", *FINAL,
              "FrozenLake", "Gambler", "MDP Playground", "labmaze"]
    metad = pd.concat([labels[n][labels[n].primary] for n in PANELS if PANELS[n][0] == "meta"])
    cmap = plt.get_cmap("viridis")
    ccol = [cmap(i / 5.5) for i in range(6)] + ["#BDBDBD"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.4), sharey=True)
    for ax, lr in zip(axes, ["BC", "CFQI", "IQL"]):
        left = np.zeros(len(panels))
        for k in range(7):
            vals = []
            for n in panels:
                d = metad if n.startswith("Meta") else labels[n][labels[n].primary]
                g = d[d.learner == lr]
                vals.append(100 * (g.event == k).mean())
            ax.barh(range(len(panels))[::-1], vals, left=left, color=ccol[k], edgecolor="white", lw=0.4,
                    hatch="///" if k == 6 else None, label=CLASSES[k])
            left += np.array(vals)
        ax.set_title(lr)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of rows")
    axes[0].set_yticks(range(len(panels))[::-1], panels, fontsize=7.6)
    axes[2].legend(frameon=False, fontsize=7.2, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Label E")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_7_label_distribution.png")
    plt.close(fig)


def fig_censor_heatmap():
    panels = ["Cohort 7", "Cohort 8", "Colosseum 2027", "Colosseum 2028", "Final internal", "Colosseum 2029", "Forest",
              "FrozenLake", "Gambler", "MDP Playground", "labmaze"]
    M = np.full((len(BEHAVIOR), len(panels)), np.nan)
    for j, n in enumerate(panels):
        b = beh[beh.panel == n].set_index("behavior").censored_pct
        for i, k in enumerate(BEHAVIOR):
            if k in b:
                M[i, j] = b[k]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 65 else "black")
    ax.set_xticks(range(len(panels)), panels, rotation=30, ha="right", fontsize=7.6)
    ax.set_yticks(range(len(BEHAVIOR)), BEHAVIOR, fontsize=7.6)
    fig.colorbar(im, ax=ax, label="% rows labeled >x50", shrink=0.9)
    ax.set_title("Censoring by behavior and panel (all three learners pooled)")
    fig.savefig(FIG / "fig3_8_censoring_heatmap.png")
    plt.close(fig)


def fig_shift():
    names = list(dist_d)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, n in enumerate(names):
        v = dist_d[n]
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=5, alpha=0.45, color=PCOLOR[n], lw=0)
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="black"))
    ax.axhline(3, color="#9C2B2B", ls="--", lw=0.9)
    ax.text(len(names) - 0.5, 3.1, "d = 3 (gate saturates)", color="#9C2B2B", fontsize=7.5, ha="right")
    ax.set_xticks(range(len(names)), names, fontsize=7.8)
    ax.set_yscale("log")
    ax.set_ylabel("Distance d to meta-training (log scale)")
    ax.set_title("Pilot-statistic distance of each MDP–behavior group to the 720-MDP meta-training panel")
    fig.savefig(FIG / "fig3_9_shift_distance.png")
    plt.close(fig)


def fig_curves():
    budgets = [1, 2, 5, 10, 20, 50, 100, 200]
    names = FINAL + ["FrozenLake", "Gambler", "MDP Playground", "labmaze"]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.8), sharey=True)
    axes = axes.ravel()
    res = {}
    for ax, n in zip(axes, names):
        o = read(PANELS[n][4])
        o["learner"] = o.algorithm.map(LEARNER)
        g = o.groupby(["learner", "budget"]).success.mean().unstack(0) * 100
        res[n] = g
        for lr in ["BC", "CFQI", "IQL"]:
            ax.plot(range(len(budgets)), g.loc[budgets, lr].to_numpy(), marker="o", ms=3, color=LCOLOR[lr], label=lr)
        ax.axvspan(5.5, 7.5, color="#EEEEEE", zorder=0)
        ax.set_xticks(range(len(budgets)), [f"x{b}" for b in budgets], rotation=60, fontsize=6.8)
        ax.set_title(n)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of seed runs with NR ≥ 0.95")
    axes[4].set_ylabel("% of seed runs with NR ≥ 0.95")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_10_success_curves.png")
    plt.close(fig)
    return res


fig_pipeline()
fig_gallery()
ex = fig_label_example()
fig_registry()
fig_profiles()
numbers.update(fig_pca())
fig_labels()
fig_censor_heatmap()
fig_shift()
curves = fig_curves()
pd.concat({k: v for k, v in curves.items()}).to_csv(TAB / "success_curves.csv")
numbers.update(ex)

# A few derived numbers quoted in the text
fin = pd.concat([labels[n] for n in FINAL])
numbers["final_censored_pct"] = 100 * fin.event.eq(6).mean()
numbers["final_censored_by_learner"] = {lr: 100 * fin[fin.learner == lr].event.eq(6).mean() for lr in LCOLOR}
metad = pd.concat([labels[n] for n in PANELS if PANELS[n][0] == "meta"])
numbers["meta_censored_pct_primary"] = 100 * metad[metad.primary].event.eq(6).mean()
numbers["meta_primary_rows"] = int(metad.primary.sum())
(OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
manifest = dict(status="posthoc_descriptive", script=str(Path(__file__).name),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                repo_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(), inputs_sha256=INPUTS,
                outputs_sha256={str(p.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted([*FIG.glob("*.png"), *TAB.glob("*.csv"), OUT / "numbers.json"])})
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(numbers, indent=1, default=float))
print(reg.to_string())
