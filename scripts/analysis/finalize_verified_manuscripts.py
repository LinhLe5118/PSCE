"""Generate manuscript-ready BRIDGE results only after the strict four-panel audit.

This script does not recompute or change experimental scores. Its inputs are the
hash-checked outputs of verify_common_difficulty_metrics.py. A preview writes
only to the preview artifact directory and never changes manuscript sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
WEEK8 = REPO / "artifacts/week8"
PANELS = (
    "bridge_expanded_prospective_5",
    "bridge_opened_minigrid_14_exploratory",
)
LEARNERS = {
    "behavior_cloning_tabular": "BC",
    "cfqi_tabular": "CFQI",
    "iql_tabular": "IQL",
}
METHODS = {
    "b2_matched_internal_diagnostic": "B2 internal",
    "published_xgboost_cf_2026_adapted_b2": "XGBoost-CF/18 adapted",
    "xgboost_cf_2026_full108_architecture_control": "XGBoost-CF/108 control",
    "oeh_full_108": "Full OEH",
    "b2_matched": "B2 internal",
    "oeh_without_direct_oehb": "Reduced OEH",
    "oeh_full_108_post_outcome_exploratory": "Full OEH (exploratory)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{value:+.5f}" if value < 0 else f"+{value:.5f}"


def load_verified(preview: bool) -> tuple[Path, dict, pd.DataFrame, pd.DataFrame]:
    folder = WEEK8 / ("difficulty_common_metric_preview_2026-10-09" if preview else
                      "difficulty_common_metric_final_2026-10-09")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = "PARTIAL_NOT_FOR_MANUSCRIPT" if preview else "ALL_PANELS_COMMON_METRIC_VERIFIED"
    if manifest["status"] != expected:
        raise RuntimeError(f"Common-metric gate has status {manifest['status']}, expected {expected}")
    if not preview and (manifest["missing"] or
                        {s["panel"] for s in manifest["source"]} != {
                            "internal120_stochastic_primary", "bridge_fresh_atari_5", *PANELS}):
        raise RuntimeError("The four required source panels are incomplete")
    for key, filename in (("summary_sha256", "summary.csv"), ("paired_sha256", "paired.csv")):
        if sha256(folder / filename) != manifest["output"][key]:
            raise RuntimeError(f"Verified score file changed after the audit: {filename}")
    summary = pd.read_csv(folder / "summary.csv")
    paired = pd.read_csv(folder / "paired.csv")
    if not preview:
        for panel, n in ((PANELS[0], 5), (PANELS[1], 14)):
            rows = summary[summary.panel == panel]
            if set(rows.learner) != set(LEARNERS) or set(rows.mdps) != {n}:
                raise RuntimeError(f"Incomplete learner or MDP rows for {panel}")
            if panel == PANELS[1] and set(rows.freshness) != {"post_outcome_exploratory"}:
                raise RuntimeError("Opened MiniGrid must remain exploratory")
    return folder, manifest, summary, paired


def load_posterior_coverage(preview: bool) -> list[dict]:
    """Audit prelabel OEH-B draw coverage against its sealed group hashes."""
    if preview:
        return []
    folder = WEEK8 / "bridge_expanded_full_oehb_preoutcome_2026-10-08"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("group_sha256", {})
    if (manifest.get("status") != "SEALED_FRESH_FULL_OEHB_PREOUTCOME_NO_TEST_LABELS" or
            manifest.get("test_labels_read") or len(hashes) != 25):
        raise RuntimeError("Incomplete or unsealed prospective OEH-B coverage manifest")
    coverage = []
    for filename, digest in sorted(hashes.items()):
        path = folder / "groups" / filename
        if sha256(path) != digest:
            raise RuntimeError(f"Prospective OEH-B group hash changed: {filename}")
        group = json.loads(path.read_text(encoding="utf-8"))
        sampling = group["sampling"]
        requested = int(sampling["draws_requested"])
        returned = int(sampling["draws_returned"])
        if (group["status"] != "FRESH_FULL_OEHB_PREOUTCOME_ONLY" or
                group["test_labels_read"] or requested != 64 or not 0 <= returned <= requested):
            raise RuntimeError(f"Invalid prospective OEH-B sampling record: {filename}")
        coverage.append({"mdp": group["mdp_id"], "behavior": group["policy_family"],
                         "requested": requested, "returned": returned})
    return coverage


def make_markdown(summary: pd.DataFrame, paired: pd.DataFrame, preview: bool,
                  coverage: list[dict]) -> str:
    out = ["## 5.9 Verified BRIDGE follow-up results", "",
           "This section is generated from the strict common seven-class metric audit. "
           "Within each panel, methods use identical MDP–behavior–learner rows. "
           "Integrated Brier and seven-class log loss are lower when better. "
           "Paired differences are full OEH minus comparator; negative favors OEH. "
           "Intervals are descriptive 95% MDP-cluster bootstrap intervals, unadjusted "
           "for multiple comparisons. Do not pool panels or treat intervals as a new superiority gate.", ""]
    if preview:
        out.insert(2, "**INCOMPLETE PREVIEW — not for the manuscript.**")
    for panel in PANELS:
        s = summary[summary.panel == panel]
        if s.empty:
            continue
        title = ("Prospective five-MDP BRIDGE follow-up" if panel == PANELS[0]
                 else "Previously opened 14-MDP BRIDGE MiniGrid analysis")
        status = ("sealed before labels" if panel == PANELS[0]
                  else "exploratory; outcomes already opened to the research program")
        out += [f"### {title}", "", f"Status: **{status}**. The published BRIDGE tasks supply MDPs; "
                "the team generated the offline pilot logs and learner-requirement labels.", "",
                "| Learner | Forecast method | MDPs | Rows | Censored >x50 | Integrated Brier | Seven-class log loss |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for _, row in s.iterrows():
            out.append(f"| {LEARNERS.get(row.learner, row.learner)} | "
                       f"{METHODS.get(row.method, row.method)} | {int(row.mdps)} | "
                       f"{int(row.rows)} | {row.censored_fraction:.1%} | "
                       f"{row.integrated_brier:.6f} | {row.seven_class_log_loss:.6f} |")
        out += ["", "| Learner | Metric | Full OEH minus comparator | Paired MDP mean | Descriptive 95% interval |",
                "|---|---|---|---:|---|"]
        p = paired[paired.panel == panel]
        for _, row in p.iterrows():
            comparison = str(row.comparison)
            comparator = comparison.split(" minus ", 1)[1]
            metric = "Brier" if row.metric == "verified_brier" else "Log loss"
            out.append(f"| {LEARNERS.get(row.learner, row.learner)} | {metric} | "
                       f"{METHODS.get(comparator, comparator)} | {fmt(row.paired_mdp_delta)} | "
                       f"[{fmt(row.ci95_low_descriptive)}, {fmt(row.ci95_high_descriptive)}] |")
        out += ["", "The comparisons against B2 are internal feature controls. "
                "XGBoost-CF/18 adapts a published general survival method to this seven-class target; "
                "XGBoost-CF/108 also receives OEH descriptors. Original ERI, TQ, SACo and EH scores "
                "are scalar ranking diagnostics with different endpoints or access and are not assigned "
                "Brier/log-loss values in this table.", ""]
        comparator = ("published_xgboost_cf_2026_adapted_b2" if panel == PANELS[0]
                      else "b2_matched")
        own = ("oeh_full_108" if panel == PANELS[0]
               else "oeh_full_108_post_outcome_exploratory")
        selected = p[(p.metric == "verified_brier") &
                     (p.comparison == f"{own} minus {comparator}")]
        gains = int((selected.paired_mdp_delta < 0).sum())
        intervals = int((selected.ci95_high_descriptive < 0).sum())
        harms = int((selected.ci95_low_descriptive > 0).sum())
        out += [f"Brier versus {METHODS[comparator]}: OEH has a lower point estimate for "
                f"{gains}/3 learners; {intervals}/3 descriptive intervals lie entirely below zero "
                f"and {harms}/3 lie entirely above zero. These counts describe this panel only.", ""]
    if not preview:
        short = [row for row in coverage if row["returned"] < row["requested"]]
        zero = [row for row in short if row["returned"] == 0]
        out += ["### Posterior draw coverage on the prospective panel", "",
                f"The prelabel OEH-B audit returned fewer than 64 posterior draws in "
                f"**{len(short)}/25** MDP–behavior groups; **{len(zero)}/25** returned none. "
                "For a zero-draw group, the frozen requirement forecast treats all 64 requested "
                "draws as censored and applies Krichevsky–Trofimov smoothing. This is a "
                "registered fallback, not an empirical 64-draw posterior estimate. "
                "The score tables retain these groups and must be read with this coverage limit.", ""]
        if short:
            out += ["| MDP | Behavior | Posterior draws returned / requested |",
                    "|---|---|---:|"]
            out += [f"| {row['mdp']} | {row['behavior']} | "
                    f"{row['returned']}/{row['requested']} |" for row in short]
            out.append("")
    return "\n".join(out) + "\n"


def make_latex(summary: pd.DataFrame, paired: pd.DataFrame,
               coverage: list[dict]) -> tuple[str, str]:
    s = summary[summary.panel == PANELS[0]]
    p = paired[paired.panel == PANELS[0]]
    lines = [r"\subsection{Prospective BRIDGE follow-up}",
             r"The additional five published BRIDGE MDPs were selected and all four forecasters sealed before learner outcomes. The group generated the offline logs and labels. Table~\ref{tab:bridge-followup} uses the same seven-class scores as the internal and first BRIDGE panels; the panels remain separate.",
             r"\begin{table}[t]", r"\centering\small",
             r"\caption{Prospective five-MDP follow-up; 25 MDP--behavior rows per learner. CF/18 adapts a published survival forecaster to this endpoint. Both seven-class scores are lower when better.}",
             r"\label{tab:bridge-followup}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Brier} & \multicolumn{2}{c}{Log loss}\\",
             r"Learner & CF/18 & OEH & CF/18 & OEH\\", r"\midrule"]
    order = ("b2_matched_internal_diagnostic", "published_xgboost_cf_2026_adapted_b2",
             "xgboost_cf_2026_full108_architecture_control", "oeh_full_108")
    for learner, label in LEARNERS.items():
        learner_rows = s[s.learner == learner]
        if len(learner_rows) != len(order) or set(learner_rows.method) != set(order):
            raise RuntimeError(f"Incomplete follow-up methods for {learner}")
        adapted = learner_rows[learner_rows.method == order[1]].iloc[0]
        oeh = learner_rows[learner_rows.method == order[3]].iloc[0]
        lines.append(f"{label} & {adapted.integrated_brier:.5f} & "
                     f"{oeh.integrated_brier:.5f} & "
                     f"{adapted.seven_class_log_loss:.5f} & "
                     f"{oeh.seven_class_log_loss:.5f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    statements = []
    for learner, learner_label in LEARNERS.items():
        row = p[(p.learner == learner) & (p.metric == "verified_brier") &
                (p.comparison == f"oeh_full_108 minus {order[1]}")]
        if len(row) != 1:
            raise RuntimeError(f"Missing paired difference {learner}/{order[1]}")
        row = row.iloc[0]
        statements.append(f"{learner_label} ${fmt(row.paired_mdp_delta)}$ "
                          f"[${fmt(row.ci95_low_descriptive)},{fmt(row.ci95_high_descriptive)}$]")
    lines.append("Full OEH minus adapted CF/18 Brier by learner is " + "; ".join(statements) +
                 ". Bracketed intervals resample five MDPs and are descriptive, not a simultaneous gate. "
                 "The internal B2 and 108-input architecture control, censoring and all paired scores are in the supplement.")
    opened = summary[summary.panel == PANELS[1]]
    op = paired[(paired.panel == PANELS[1]) & (paired.metric == "verified_brier") &
                (paired.comparison == "oeh_full_108_post_outcome_exploratory minus b2_matched")]
    mini = []
    for learner, learner_label in LEARNERS.items():
        row = op[op.learner == learner]
        if len(row) != 1:
            raise RuntimeError(f"Missing opened MiniGrid paired row {learner}")
        row = row.iloc[0]
        mini.append(f"{learner_label} ${fmt(row.paired_mdp_delta)}$ "
                    f"[${fmt(row.ci95_low_descriptive)},{fmt(row.ci95_high_descriptive)}$]")
    if len(opened) != 9:
        raise RuntimeError("Opened MiniGrid lacks the three complete methods")
    lines.append("On the 14 previously opened MiniGrid MDPs, full OEH minus internal B2 Brier is " +
                 "; ".join(mini) + ". These post-outcome intervals are exploratory. "
                 "All methods, both metrics, censoring and remaining paired contrasts are in the supplement.")
    supplement = [r"\section{Completed BRIDGE follow-up scores}",
                  r"Every row below uses the common seven-class scoring audit. Published BRIDGE environments do not imply published offline logs or labels. MiniGrid-14 was opened before the full run and is exploratory."]
    for panel in PANELS:
        rows = summary[summary.panel == panel]
        caption = "Prospective BRIDGE five" if panel == PANELS[0] else "Opened BRIDGE MiniGrid fourteen"
        supplement += [r"\begin{table*}[t]", r"\centering\small",
                       rf"\caption{{{caption}: all seven-class forecast scores; lower is better.}}",
                       r"\begin{tabular}{llrrccc}", r"\toprule",
                       r"Learner & Method & MDPs & Rows & $>x50$ & Brier & Log loss\\", r"\midrule"]
        for _, row in rows.iterrows():
            method = METHODS.get(row.method, row.method).replace("_", r"\_")
            supplement.append(f"{LEARNERS.get(row.learner, row.learner)} & {method} & "
                              f"{int(row.mdps)} & {int(row.rows)} & {100 * row.censored_fraction:.1f}\\% & "
                              f"{row.integrated_brier:.5f} & {row.seven_class_log_loss:.5f}" + r"\\")
        supplement += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    supplement += ["Paired MDP bootstrap intervals for every method and metric are given in the "
                   r"machine-readable \texttt{paired.csv} of the hash-checked audit. "
                   "They are descriptive and unadjusted for multiplicity."]
    short = [row for row in coverage if row["returned"] < row["requested"]]
    zero = [row for row in short if row["returned"] == 0]
    lines.append(
        f"The sealed OEH-B sampler returned fewer than 64 posterior draws in {len(short)}/25 "
        f"MDP--behavior groups; {len(zero)}/25 returned none. Zero-draw groups use the frozen "
        "all-censored, Krichevsky--Trofimov-smoothed fallback rather than simulated "
        "64-draw posterior evidence. All groups remain in the scores; details are in the supplement."
    )
    supplement.append(r"\paragraph{Prospective posterior draw coverage.}")
    supplement.append(
        f"Fewer than 64 draws were returned in {len(short)}/25 groups; "
        f"{len(zero)}/25 returned zero and used the frozen censored fallback."
    )
    for row in short:
        mdp = row["mdp"].replace("_", r"\_")
        behavior = row["behavior"].replace("_", r"\_")
        supplement.append(f"{mdp}, {behavior}: {row['returned']}/{row['requested']} draws. ")
    return "\n\n".join(lines) + "\n", "\n".join(supplement) + "\n"


def discussion_counts(paired: pd.DataFrame, panel: str, own: str,
                      comparator: str) -> tuple[int, int, int]:
    rows = paired[(paired.panel == panel) & (paired.metric == "verified_brier") &
                  (paired.comparison == f"{own} minus {comparator}")]
    if len(rows) != 3 or set(rows.learner) != set(LEARNERS):
        raise RuntimeError(f"Incomplete discussion contrasts: {panel}/{comparator}")
    return (int((rows.paired_mdp_delta < 0).sum()),
            int((rows.ci95_high_descriptive < 0).sum()),
            int((rows.ci95_low_descriptive > 0).sum()))


def make_discussion(paired: pd.DataFrame) -> tuple[str, str]:
    prospect = discussion_counts(paired, PANELS[0], "oeh_full_108",
                                "published_xgboost_cf_2026_adapted_b2")
    opened = discussion_counts(paired, PANELS[1],
                               "oeh_full_108_post_outcome_exploratory", "b2_matched")
    tex = (r"\paragraph{Completed BRIDGE follow-up.} On the separately sealed five-task "
           "panel, OEH has a lower Brier point estimate than adapted XGBoost-CF/18 for "
           f"{prospect[0]} of three learners; {prospect[1]} descriptive MDP intervals "
           f"exclude zero in OEH's favor and {prospect[2]} in the other direction. "
           "On the 14 previously opened MiniGrid tasks, the corresponding counts versus "
           f"internal B2 are {opened[0]}, {opened[1]}, and {opened[2]}; that analysis is "
           "post-outcome exploratory. The panels differ in environment family and analysis "
           "status, so their scores are not pooled into a literature-wide superiority claim. "
           "The team generated all BRIDGE offline logs and learner labels.\n")
    md = ("## 6.6 Completed BRIDGE transfer interpretation\n\n"
          "On the separately sealed five-task panel, OEH has a lower Brier point estimate "
          f"than adapted XGBoost-CF/18 for {prospect[0]} of three learners; "
          f"{prospect[1]} descriptive MDP intervals exclude zero in OEH's favor and "
          f"{prospect[2]} in the other direction. On the 14 previously opened MiniGrid tasks, "
          f"the corresponding counts against internal B2 are {opened[0]}, {opened[1]}, "
          f"and {opened[2]}. The latter analysis is post-outcome exploratory. The panels "
          "differ in environment family and analysis status; their scores are not pooled "
          "into a literature-wide superiority claim. The published BRIDGE MDPs are external, "
          "whereas the offline logs and learner labels were generated by this group.\n")
    return tex, md


def make_closing_text(paired: pd.DataFrame) -> tuple[str, str, str]:
    """Extend the abstracts and thesis conclusion without changing the frozen claims."""
    prospect = discussion_counts(paired, PANELS[0], "oeh_full_108",
                                "published_xgboost_cf_2026_adapted_b2")
    opened = discussion_counts(paired, PANELS[1],
                               "oeh_full_108_post_outcome_exploratory", "b2_matched")
    abstract = (
        "A separate prelabel-sealed five-task BRIDGE follow-up gives lower OEH Brier point "
        f"estimates than an adapted published survival forecaster for {prospect[0]}/3 learners; "
        f"{prospect[1]}/3 descriptive task-cluster intervals exclude zero in OEH's favor. "
        "A 14-task MiniGrid reanalysis uses previously opened outcomes and remains exploratory."
    )
    paper = abstract.replace("OEH's", r"\OEH{}'s") + "\n"
    thesis = (
        "## 7.4 Verified external follow-up conclusion\n\n"
        "The strict common-metric audit verified the internal stochastic panel, the first "
        "five-task BRIDGE Atari panel, the separately selected prospective five-task BRIDGE "
        "follow-up, and the 14-task previously opened MiniGrid reanalysis. On the prospective "
        "follow-up, full OEH has a lower Brier point estimate than the adapted published "
        f"XGBoost-CF/18 survival forecaster for {prospect[0]} of three learners. Of the three "
        f"descriptive MDP-cluster intervals, {prospect[1]} exclude zero in OEH's favor and "
        f"{prospect[2]} in the opposite direction; the rest include zero. On the opened "
        f"MiniGrid tasks, the corresponding counts versus internal B2 are {opened[0]}, "
        f"{opened[1]} and {opened[2]}. The latter scores are exploratory even if favorable. "
        "These panels share the seven-class metric and matched rows within each panel, but "
        "their environments, sample sizes and inferential status differ. The published "
        "BRIDGE artifacts are MDP definitions; the offline logs and learner labels were "
        "generated by this group. Thus the completed extension does not establish "
        "literature-wide superiority, uniform transfer across learners, or a net saving in "
        "data and compute. Task-level explanations, cost accounting and a newly untouched "
        "component ablation remain the next research steps.\n"
    )
    return paper, abstract + "\n", thesis


def write_once(path: Path, content: str, *, preview: bool) -> None:
    if not preview and path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise RuntimeError(f"Manuscript output was edited; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    folder, manifest, summary, paired = load_verified(args.preview)
    coverage = load_posterior_coverage(args.preview)
    markdown = make_markdown(summary, paired, args.preview, coverage)
    packet = folder / "MANUSCRIPT_RESULTS.md"
    write_once(packet, markdown, preview=args.preview)
    if not args.preview:
        main_tex, supplement_tex = make_latex(summary, paired, coverage)
        discussion_tex, discussion_md = make_discussion(paired)
        abstract_tex, abstract_md, conclusion_md = make_closing_text(paired)
        write_once(REPO / "paper/sections/bridge_verified_results.tex", main_tex, preview=False)
        write_once(REPO / "paper/sections/supplement_bridge_verified.tex", supplement_tex, preview=False)
        write_once(REPO / "paper/sections/bridge_verified_discussion.tex", discussion_tex, preview=False)
        write_once(REPO / "paper/sections/bridge_verified_abstract.tex", abstract_tex, preview=False)
        write_once(WORKSPACE / "Drive/Thesis/05a_Bridge_Verified_Results_2026-10-09.md",
                   markdown, preview=False)
        write_once(WORKSPACE / "Drive/Thesis/06a_Bridge_Verified_Discussion_2026-10-09.md",
                   discussion_md, preview=False)
        write_once(WORKSPACE / "Drive/Thesis/07a_Bridge_Verified_Conclusion_2026-10-09.md",
                   conclusion_md, preview=False)
        write_once(WORKSPACE / "Drive/Thesis/bridge_verified_abstract.txt",
                   abstract_md, preview=False)
    output = {"status": "PREVIEW_ONLY" if args.preview else "MANUSCRIPT_SOURCES_GENERATED",
              "common_metric_manifest_sha256": sha256(folder / "manifest.json"),
              "results_packet_sha256": sha256(packet),
              "panels": list(PANELS)}
    if not args.preview:
        output["prospective_oehb_manifest_sha256"] = sha256(
            WEEK8 / "bridge_expanded_full_oehb_preoutcome_2026-10-08/manifest.json")
        output["posterior_draw_coverage"] = {
            "groups": len(coverage),
            "shortfall_groups": sum(row["returned"] < row["requested"] for row in coverage),
            "zero_draw_groups": sum(row["returned"] == 0 for row in coverage),
        }
        output["generated_source_sha256"] = {
            "paper_results": sha256(REPO / "paper/sections/bridge_verified_results.tex"),
            "paper_supplement": sha256(REPO / "paper/sections/supplement_bridge_verified.tex"),
            "paper_discussion": sha256(REPO / "paper/sections/bridge_verified_discussion.tex"),
            "paper_abstract": sha256(REPO / "paper/sections/bridge_verified_abstract.tex"),
            "thesis_results": sha256(WORKSPACE / "Drive/Thesis/05a_Bridge_Verified_Results_2026-10-09.md"),
            "thesis_discussion": sha256(WORKSPACE / "Drive/Thesis/06a_Bridge_Verified_Discussion_2026-10-09.md"),
            "thesis_conclusion": sha256(WORKSPACE / "Drive/Thesis/07a_Bridge_Verified_Conclusion_2026-10-09.md"),
            "thesis_abstract": sha256(WORKSPACE / "Drive/Thesis/bridge_verified_abstract.txt"),
        }
    (folder / "manuscript_source_manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
