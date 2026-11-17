"""Manuscript tables and figures for the 2026-10-13 paper/thesis revision.

Reads only sealed or completed artifacts and writes a new namespace. Every
subgroup, calibration and transfer number in the paper/thesis is produced here
so the text never relies on hand-copied values. Analyses of the opened seventh
cohort beyond its registered gate are descriptive (post-outcome).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
C7 = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
C4 = ROOT / "artifacts/oeh_final_unseen_120_2026-10-02"
M1 = ROOT / "artifacts/week8/m1_matched_ablation_2026-10-13"
BRIDGE = {
    "MiniGrid-14 (reduced OEH)": ROOT / "artifacts/week8/bridge_reduced_test_analysis_2026-10-07",
    "Atari-5 (full OEH)": ROOT / "artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08",
    "Expanded-5 (full OEH)": ROOT / "artifacts/week8/bridge_expanded_full_test_analysis_2026-10-08",
}
FOURROOMS = ROOT / "artifacts/published_fourrooms_calibration_2026-10-04/result.json"
OUTPUT = ROOT / "artifacts/week8/manuscript_tables_2026-10-13"
LEARNERS = {"behavior_cloning_tabular": "BC", "cfqi_tabular": "CFQI", "iql_tabular": "IQL"}
FAMILIES = {"uniform_garnet": "Garnet", "forward_layered": "Forward-layered", "bottleneck": "Bottleneck"}
CLASSES = ["x1", "x2", "x5", "x10", "x20", "x50", ">x50"]
P = [f"p_{i}" for i in range(7)]
SEED = 20261013
MATCHED = {
    "b2_matched": ("B2 matched", "Tier 1 matched", "pilot"),
    "b2_plus_eri": ("B2 + ERI adaptation", "Tier 1 matched", "pilot"),
    "b2_plus_bwd": ("B2 + BWD adaptation", "Tier 1 matched", "pilot"),
    "b2_generic": ("B2 generic (AFT only)", "Tier 1 other head", "pilot"),
    "coverage_quality_adaptation": ("Coverage + quality (AFT only)", "Tier 1 other head", "pilot"),
    "train_frequency": ("Train frequency", "Tier 1 no head", "labels only"),
    "b2_plus_eh_oracle_matched": ("B2 + oracle EH", "Tier 2 privileged", "true MDP"),
    "b2_plus_tq_saco_original": ("B2 + TQ/SACo (author code)", "Tier 2 privileged", "true-MDP reference data"),
    "published_metrics_only": ("Published metrics only", "Tier 2 privileged", "true-MDP reference data"),
    "oeh_b_raw": ("Raw OEH-B (unfitted)", "reported, no gate", "pilot"),
    "prior_oeh_matched": ("Earlier OEH features", "reported, no gate", "pilot"),
    "oeh_final": ("OEH (108 columns)", "candidate", "pilot"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_ci(delta: np.ndarray, rng, draws=2000) -> tuple[float, float]:
    idx = rng.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[idx].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def per_mdp(frame: pd.DataFrame, keys) -> pd.DataFrame:
    return frame.groupby([*keys, "mdp_id"], observed=True)[["integrated_brier", "seven_class_log_loss"]].mean().reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    inputs = [C7 / "scored_predictions.csv", C7 / "summary.csv", C7 / "contrasts.csv", C7 / "labels.csv",
              C7 / "selection_summary.csv", C7 / "selection_contrasts.csv", C7 / "confirmatory_verdict.json",
              C7 / "protocol.json", C4 / "summary.csv", FOURROOMS]
    result_manifest = json.loads((C7 / "result_manifest.json").read_text(encoding="utf-8"))
    for name in ("scored_predictions.csv", "summary.csv", "contrasts.csv", "labels.csv"):
        if sha256(C7 / name) != result_manifest["artifact_sha256"][name]:
            raise RuntimeError(f"Sealed cohort-7 artifact changed: {name}")
    protocol = json.loads((C7 / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "datasets/oeh_confirmatory_unseen_120_2026-10-04/manifest.json").read_text(encoding="utf-8"))
    family_of = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    sizes = {item["instance_id"]: (item["num_states"], item["num_actions"]) for item in manifest["instances"]}

    scored = pd.read_csv(C7 / "scored_predictions.csv")
    scored["family"] = scored.mdp_id.map(family_of)
    primary = scored[scored.panel == "primary"].copy()

    # 1. Main forecast table with registered contrasts (sealed numbers).
    summary = pd.read_csv(C7 / "summary.csv")
    contrasts = pd.read_csv(C7 / "contrasts.csv")
    rows = []
    for method, (label, tier, access) in MATCHED.items():
        s = summary[(summary.panel == "primary") & (summary.method == method)].iloc[0]
        own = per_mdp(primary[primary.method == method], [])
        if not np.isclose(own.integrated_brier.mean(), s.integrated_brier, atol=1e-12):
            raise RuntimeError(f"Recomputed IBS differs for {method}")
        row = {"method": method, "label": label, "tier": tier, "access": access, "mdps": int(s.mdps),
               "ibs": s.integrated_brier, "log_loss": s.seven_class_log_loss}
        for metric, key in (("integrated_brier", "ibs"), ("seven_class_log_loss", "ll")):
            c = contrasts[(contrasts.baseline == method) & (contrasts.metric == metric)]
            if len(c):
                row[f"{key}_relative_gain"] = float(c.relative_gain.iloc[0])
                row[f"{key}_abs_gain"] = float(c.mean_absolute_improvement.iloc[0])
                row[f"{key}_simultaneous_lower"] = float(c.simultaneous_one_sided_lower_95.iloc[0])
        control = summary[(summary.panel == "control") & (summary.method == method)]
        row["control_ibs"] = float(control.integrated_brier.iloc[0]) if len(control) else np.nan
        row["control_log_loss"] = float(control.seven_class_log_loss.iloc[0]) if len(control) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "forecast_main.csv", index=False)

    # 2. Label distribution and censoring.
    labels = pd.read_csv(C7 / "labels.csv")
    labels["family"] = labels.mdp_id.map(family_of)
    dist = (labels.groupby(["panel", "algorithm", "event"]).size().unstack(fill_value=0)
            .reindex(columns=range(7), fill_value=0))
    dist.columns = CLASSES
    dist = dist.reset_index()
    dist["rows"] = dist[CLASSES].sum(1)
    dist["censored_fraction"] = dist[">x50"] / dist["rows"]
    dist.to_csv(out / "class_distribution.csv", index=False)
    lab = labels[labels.panel == "primary"].copy()
    counts = lab[[f"successes_x{b}" for b in (1, 2, 5, 10, 20, 50)]].to_numpy()
    nonmonotone = (np.diff(counts, axis=1) < 0).any(axis=1)
    label_facts = {
        "primary_rows": int(len(lab)), "primary_censored": int((lab.event == 6).sum()),
        "primary_censored_fraction": float((lab.event == 6).mean()),
        "nonmonotone_success_count_fraction": float(nonmonotone.mean()),
        "censored_by_learner": {LEARNERS[k]: float(v) for k, v in lab.groupby("algorithm").event.apply(lambda e: (e == 6).mean()).items()},
        "censored_by_family": {FAMILIES[k]: float(v) for k, v in lab.groupby("family").event.apply(lambda e: (e == 6).mean()).items()},
        "censored_by_behavior": {k: float(v) for k, v in lab.groupby("policy_family").event.apply(lambda e: (e == 6).mean()).items()},
        "control_censored_fraction": float((labels[labels.panel == "control"].event == 6).mean()),
        "state_action_sizes_primary": {"min": int(min(s * a for m, (s, a) in sizes.items() if family_of[m] != "lock_structured")),
                                       "max": int(max(s * a for m, (s, a) in sizes.items() if family_of[m] != "lock_structured"))},
        "sum_SA_all": int(sum(s * a for s, a in sizes.values())),
    }
    example = labels[(labels.mdp_id == "oehfinal120_forward_layered_004") & (labels.policy_family == "q_star_softmax")]
    label_facts["example_group"] = {LEARNERS[r.algorithm]: {"counts": [int(getattr(r, f"successes_x{b}")) for b in (1, 2, 5, 10, 20, 50)],
                                                             "event": CLASSES[int(r.event)]} for r in example.itertuples()}

    # 3. Subgroups: learner, family, behavior, finite vs censored (OEH vs B2 matched).
    pair = primary[primary.method.isin(["oeh_final", "b2_matched"])]
    sub_rows = []
    for dim, mapping in (("algorithm", LEARNERS), ("family", FAMILIES), ("policy_family", None)):
        for value, group in pair.groupby(dim):
            m = per_mdp(group, ["method"])
            for metric in ("integrated_brier", "seven_class_log_loss"):
                pv = m.pivot(index="mdp_id", columns="method", values=metric)
                delta = (pv.oeh_final - pv.b2_matched).to_numpy()
                low, high = boot_ci(delta, rng)
                sub_rows.append({"dimension": dim, "group": mapping[value] if mapping else value,
                                 "metric": metric, "mdps": len(pv), "rows_per_method": int(len(group) / 2),
                                 "b2_matched": pv.b2_matched.mean(), "oeh": pv.oeh_final.mean(),
                                 "oeh_minus_b2": delta.mean(), "relative_change": delta.mean() / pv.b2_matched.mean(),
                                 "ci95_low": low, "ci95_high": high,
                                 "censored_fraction": float((group[group.method == "oeh_final"].event == 6).mean())})
    pd.DataFrame(sub_rows).to_csv(out / "subgroups_oeh_vs_b2.csv", index=False)
    oeh = primary[primary.method == "oeh_final"]
    finite = oeh.event < 6
    error_share = {"finite_row_fraction": float(finite.mean()),
                   "finite_share_of_total_ibs": float(oeh.integrated_brier[finite].sum() / oeh.integrated_brier.sum()),
                   "finite_share_of_total_ll": float(oeh.seven_class_log_loss[finite].sum() / oeh.seven_class_log_loss.sum()),
                   "cfqi_share_of_total_ibs": float(oeh.integrated_brier[oeh.algorithm == "cfqi_tabular"].sum() / oeh.integrated_brier.sum()),
                   "cfqi_share_of_total_ll": float(oeh.seven_class_log_loss[oeh.algorithm == "cfqi_tabular"].sum() / oeh.seven_class_log_loss.sum())}
    for method in ("oeh_final", "b2_matched"):
        g = primary[primary.method == method]
        for name, mask in (("finite", g.event < 6), ("censored", g.event == 6)):
            error_share[f"{method}_{name}_mean_ibs"] = float(g.integrated_brier[mask].mean())
            error_share[f"{method}_{name}_mean_ll"] = float(g.seven_class_log_loss[mask].mean())

    # 4. Threshold calibration by budget (descriptive).
    cal = []
    for (method, learner), g in primary[primary.method.isin(["oeh_final", "b2_matched"])].groupby(["method", "algorithm"]):
        cdf = np.cumsum(g[P].to_numpy(float), axis=1)[:, :6]
        for j, b in enumerate((1, 2, 5, 10, 20, 50)):
            cal.append({"method": method, "learner": LEARNERS[learner], "budget": b,
                        "mean_forecast_cdf": cdf[:, j].mean(), "observed_reach": float((g.event <= j).mean()),
                        "threshold_brier": float(((cdf[:, j] - (g.event.to_numpy() <= j)) ** 2).mean())})
    cal = pd.DataFrame(cal)
    cal.to_csv(out / "calibration_by_budget.csv", index=False)

    # 5. Selection.
    selection = pd.read_csv(C7 / "selection_summary.csv")
    selection.to_csv(out / "selection_summary.csv", index=False)
    verdict = json.loads((C7 / "confirmatory_verdict.json").read_text(encoding="utf-8"))

    # 6. External transfer panels.
    bridge = []
    for panel, folder in BRIDGE.items():
        inputs += [folder / "summary.csv", folder / "paired_task_bootstrap.csv", folder / "manifest.json"]
        s = pd.read_csv(folder / "summary.csv")
        s.insert(0, "panel", panel)
        bridge.append(s.rename(columns={"mdps": "tasks"}))
    bridge = pd.concat(bridge, ignore_index=True)
    bridge["learner"] = bridge.algorithm.map(LEARNERS)
    bridge.to_csv(out / "bridge_summary.csv", index=False)
    paired = []
    for panel, folder in BRIDGE.items():
        p = pd.read_csv(folder / "paired_task_bootstrap.csv")
        p.insert(0, "panel", panel)
        paired.append(p)
    pd.concat(paired, ignore_index=True).to_csv(out / "bridge_paired.csv", index=False)
    fourrooms = json.loads(FOURROOMS.read_text(encoding="utf-8"))

    # 7. Historical cohort sequence (registered verdicts from their manifests/docs).
    history = []
    for name, folder, oeh_m, b2_m, verdict_text in (
            ("1st new 120", "oeh_difficulty_unseen_120_2026-10-01", "oeh_three_way", "b2_three_way", "NO-GO"),
            ("2nd new 120", "oeh_pooled_successor_unseen_120_2026-10-01", "oeh_three_way", "b2_three_way", "NO-GO"),
            ("3rd new 120", "oeh_cross_capacity_unseen_120_2026-10-01", "oeh_cross_capacity", "b2_three_way", "NO-GO"),
            ("4th new 120", "oeh_final_unseen_120_2026-10-02", "oeh_final", "b2_matched",
             "Tier 1 GO; Tier 2, combined and selection NO-GO"),
            ("7th new 120", "oeh_confirmatory_unseen_120_2026-10-04", "oeh_final", "b2_matched",
             "Tier 1, Tier 2, combined GO; selection vs fixed CFQI GO; vs v5 not superior")):
        s = pd.read_csv(ROOT / "artifacts" / folder / "summary.csv")
        s = s[s.panel == "primary"].set_index("method")
        inputs.append(ROOT / "artifacts" / folder / "summary.csv")
        history.append({"cohort": name, "artifact": folder, "oeh_ibs": s.loc[oeh_m, "integrated_brier"],
                        "oeh_ll": s.loc[oeh_m, "seven_class_log_loss"], "b2_ibs": s.loc[b2_m, "integrated_brier"],
                        "b2_ll": s.loc[b2_m, "seven_class_log_loss"],
                        "ibs_rel_gain": 1 - s.loc[oeh_m, "integrated_brier"] / s.loc[b2_m, "integrated_brier"],
                        "ll_rel_gain": 1 - s.loc[oeh_m, "seven_class_log_loss"] / s.loc[b2_m, "seven_class_log_loss"],
                        "registered_verdict": verdict_text})
    pd.DataFrame(history).to_csv(out / "cohort_history.csv", index=False)

    # 8. M1 ablation (if scored).
    ablation = None
    if (M1 / "manifest.json").exists():
        inputs += [M1 / "summary.csv", M1 / "paired_mdp_delta.csv", M1 / "manifest.json", M1 / "fit_runtime.csv",
                   M1 / "arm_feature_cost.csv", M1 / "cost_summary.csv"]
        ablation = pd.read_csv(M1 / "summary.csv")
    facts = {"labels": label_facts, "error_share": error_share, "selection_verdict": verdict,
             "fourrooms": {k: fourrooms[k] for k in fourrooms if k in ("verdict", "metrics", "contrasts", "splits", "status")}}
    (out / "facts.json").write_text(json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8")

    figures(out, dist, cal, ablation)
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete", "script_sha256": sha256(Path(__file__)),
        "note": "Descriptive post-outcome tables except the copied registered gate values; bootstrap seed 20261013",
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs if p.exists()},
        "outputs": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"},
    }, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out.as_posix().encode("ascii", "replace").decode())


# Validated reference palette (light print surface): slots 1-3 and inks.
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"


def _style(plt):
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "grid.color": "#e4e3df",
                         "grid.linewidth": 0.5, "savefig.dpi": 300})


def figures(out: Path, dist: pd.DataFrame, cal: pd.DataFrame, ablation) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    # Requirement class distribution, one panel per learner (stochastic primary).
    d = dist[dist.panel == "primary"].set_index("algorithm")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, (key, name) in zip(axes, LEARNERS.items()):
        values = d.loc[key, CLASSES].to_numpy(float)
        colors = [BLUE] * 6 + [GRAY]
        bars = ax.bar(range(7), values, color=colors, width=0.72, edgecolor="white", linewidth=1.0)
        bars[-1].set_hatch("////")
        ax.set_xticks(range(7), CLASSES, fontsize=7.5)
        ax.set_title(f"{name}  ({d.loc[key, 'censored_fraction']:.1%} censored)", fontsize=8.5, color=INK)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        for x, v in enumerate(values):
            ax.text(x, v + 4, f"{int(v)}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    axes[0].set_ylabel("MDP-behavior rows (of 420)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_class_distribution.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Threshold Brier by budget: where the integrated score difference arises.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=False)
    budgets = [1, 2, 5, 10, 20, 50]
    for ax, name in zip(axes, LEARNERS.values()):
        g = cal[cal.learner == name]
        for method, color, label, marker in (("b2_matched", ORANGE, "B2 matched", "^"), ("oeh_final", BLUE, "OEH", "s")):
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
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_threshold_brier.{ext}", bbox_inches="tight")
    plt.close(fig)

    if ablation is None:
        return
    delta = pd.read_csv(M1 / "paired_mdp_delta.csv")
    delta = delta[(delta.panel == "primary") & (delta.metric == "brier") & delta.comparison.str.endswith("minus oeh_full_108")]
    arms = ["b2_matched", "oeh_without_direct_oehb", "oeh_without_oehh_risk", "oeh_without_legacy_diagnostics"]
    names = ["B2 only", "No direct OEH-B", "No OEH-H/risk", "No legacy+context"]
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
    axes[0].set_yticks(range(len(arms)), names, fontsize=7.5)
    axes[0].invert_yaxis()
    fig.supxlabel("IBS change vs full OEH (arm − full; positive = worse)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
