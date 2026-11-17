"""Apply the final OEH profile contract to an existing EB-OEH audit artifact."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import (read_csv_rows, reserve_output_dir, sha256_file,
                           write_csv_rows, write_json)
from oeh_final_profile import build_final_proxy_profile


def _float(value):
    return None if value in (None, "") else float(value)


def _summary(rows):
    def mean(key):
        values = [float(row[key]) for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    def cluster_ci(key):
        groups = defaultdict(list)
        for row in rows:
            if row[key] is not None:
                groups[row["mdp_id"]].append(float(row[key]))
        clusters = list(groups.values())
        if not clusters:
            return None
        rng = np.random.default_rng(20260919)
        draws = np.empty(10000)
        for index in range(len(draws)):
            chosen = rng.integers(0, len(clusters), len(clusters))
            draws[index] = np.mean([value for j in chosen for value in clusters[j]])
        return [float(value) for value in np.quantile(draws, [.025, .975])]

    return {
        "cases": len(rows),
        "data_selector_available": sum(row["data_selected_depth"] is not None for row in rows),
        "eh_reference_available": sum(row["eh_selected_depth"] is not None for row in rows),
        "both_available": sum(row["data_selected_depth"] is not None
                              and row["eh_selected_depth"] is not None for row in rows),
        "selectors_differ": sum(row["data_selected_depth"] is not None
                                and row["eh_selected_depth"] is not None
                                and row["data_selected_depth"] != row["eh_selected_depth"]
                                for row in rows),
        "data_fallback_success": sum(row["data_fallback_regret"] <= .05+1e-12 for row in rows),
        "eh_fallback_success": sum(row["eh_fallback_regret"] <= .05+1e-12 for row in rows),
        "fixed100_success": sum(row["fixed100_regret"] <= .05+1e-12 for row in rows),
        "mean_data_fallback_regret": mean("data_fallback_regret"),
        "mean_eh_fallback_regret": mean("eh_fallback_regret"),
        "mean_fixed100_regret": mean("fixed100_regret"),
        "mean_data_gain_vs_fixed100": mean("data_gain_vs_fixed100"),
        "data_gain_ci95_cluster_mdp": cluster_ci("data_gain_vs_fixed100"),
        "mean_eh_gain_vs_fixed100": mean("eh_gain_vs_fixed100"),
        "mean_data_minus_eh_regret": mean("data_minus_eh_regret"),
        "data_minus_eh_ci95_cluster_mdp": cluster_ci("data_minus_eh_regret"),
        "data_better_worse_tied_vs_eh": [
            sum(row["data_minus_eh_regret"] < -1e-12 for row in rows),
            sum(row["data_minus_eh_regret"] > 1e-12 for row in rows),
            sum(abs(row["data_minus_eh_regret"]) <= 1e-12 for row in rows),
        ],
    }


def _report(payload):
    lines = [
        "# Final OEH profile application audit (2026-09-19)", "",
        "Status: **development proxy applied; research objective not completed**.", "",
        "This artifact applies the final output contract `(k, N_hat_k, status)` to the "
        "existing 22,725-point EB-OEH development audit. It does not recompute the "
        "pilot estimator and does not access the sealed evaluation split.", "",
        "The primary selector minimizes `N_hat_k^EB` under the QVI compute budget. "
        "The legacy `k + log_A(N_hat_k^EB)` selector is retained only as the "
        "EH-reference comparator. `unresolved` and `not_epsilon_solvable` depths are "
        "right-censored rather than assigned a finite number.", "",
        "## Results", "",
    ]
    for name in ("all", "stochastic"):
        item = payload[name]
        lines += [
            f"### {name.capitalize()} ({item['cases']} pilots)", "",
            f"- Selectable profiles: data objective {item['data_selector_available']}; "
            f"EH-reference {item['eh_reference_available']}; both {item['both_available']}; "
            f"different selected depths {item['selectors_differ']}.",
            f"- NR >= 0.95 with registered depth-100 fallback: data objective "
            f"{item['data_fallback_success']}; EH-reference {item['eh_fallback_success']}; "
            f"fixed depth 100 {item['fixed100_success']}.",
            f"- Mean normalized regret with fallback: data objective "
            f"{item['mean_data_fallback_regret']:.6f}; EH-reference "
            f"{item['mean_eh_fallback_regret']:.6f}; fixed depth 100 "
            f"{item['mean_fixed100_regret']:.6f}.",
            f"- Mean regret gain over fixed depth 100: data objective "
            f"{item['mean_data_gain_vs_fixed100']:.6f}, cluster CI "
            f"{item['data_gain_ci95_cluster_mdp']}; EH-reference "
            f"{item['mean_eh_gain_vs_fixed100']:.6f}.", "",
            f"- Data objective minus EH-reference regret: "
            f"{item['mean_data_minus_eh_regret']:.6f}, cluster CI "
            f"{item['data_minus_eh_ci95_cluster_mdp']}; better/worse/tied "
            f"{item['data_better_worse_tied_vs_eh']}.", "",
        ]
    lines += [
        "## Claim boundary", "",
        "This run validates system wiring and exposes the consequence of using the "
        "offline data objective. It is not evidence that the EB plug-in estimates the "
        "KL target: the current estimator reuses data for policy proposal and evidence, "
        "uses an occupancy threshold, and has no finite-sample proof. B1/B2/B3 grouped "
        "out-of-fold prediction and learner-selection utility have not been run. Therefore "
        "the final research objective remains incomplete.", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-budget-qvi", type=int, default=101)
    args = parser.parse_args(argv)
    curves_path = args.input_artifact / "curves.csv"
    report_path = args.input_artifact / "report.json"
    if not curves_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Input artifact must contain curves.csv and report.json")
    reserve_output_dir(args.output_dir)
    rows = read_csv_rows(curves_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset_id"]].append(row)

    profile_rows, summaries = [], []
    for dataset_id, group in sorted(grouped.items()):
        records = [{
            "depth": int(row["est_depth"]),
            "qvi_index": int(row["est_qvi_index"]),
            "admissible": row["est_admissible"],
            "identified": row["est_identified"],
            "eb_sample_requirement": row["est_eb_sample_requirement"],
            "eb_evidence": row["est_eb_evidence"],
        } for row in group]
        final = build_final_proxy_profile(
            records, compute_budget=args.compute_budget_qvi)
        by_depth = {int(row["est_depth"]): row for row in group}
        data_depth = final["primary_data_selection"]["depth"]
        eh_depth = final["eh_reference_selection"]["depth"]
        fixed_depth = max(by_depth)
        fixed_regret = float(by_depth[fixed_depth]["true_candidate_regret"])
        data_regret = (fixed_regret if data_depth is None else
                       float(by_depth[data_depth]["true_candidate_regret"]))
        eh_regret = (fixed_regret if eh_depth is None else
                     float(by_depth[eh_depth]["true_candidate_regret"]))
        identity = {key: group[0][key] for key in (
            "dataset_id", "mdp_id", "policy_family", "pilot_multiplier",
            "num_transitions", "structural_family")}
        summaries.append({
            **identity,
            "identified_admissible_depths": sum(
                item["status"] == "identified" for item in final["profile"]),
            "data_selected_depth": data_depth,
            "data_selected_qvi_index": final["primary_data_selection"]["qvi_index"],
            "data_selected_n_hat_eb": final["primary_data_selection"]["n_hat_eb"],
            "eh_selected_depth": eh_depth,
            "eh_selected_qvi_index": final["eh_reference_selection"]["qvi_index"],
            "fixed100_regret": fixed_regret,
            "data_fallback_regret": data_regret,
            "eh_fallback_regret": eh_regret,
            "data_gain_vs_fixed100": fixed_regret-data_regret,
            "eh_gain_vs_fixed100": fixed_regret-eh_regret,
            "data_minus_eh_regret": data_regret-eh_regret,
        })
        for item in final["profile"]:
            profile_rows.append({**identity, **item})

    all_summary = _summary(summaries)
    stochastic_summary = _summary([
        row for row in summaries if row["structural_family"] != "lock_structured"])
    payload = {
        "status": "complete_development_proxy_research_incomplete",
        "formula_version": "oeh_identification_profile_v1",
        "input_artifact": str(args.input_artifact.resolve()),
        "input_curves_sha256": sha256_file(curves_path),
        "input_report_sha256": sha256_file(report_path),
        "source_sha256": {
            "code/oeh_final_profile.py": sha256_file(ROOT / "code/oeh_final_profile.py"),
            "scripts/analysis/apply_final_oeh_profile.py": sha256_file(Path(__file__)),
        },
        "compute_budget_qvi": args.compute_budget_qvi,
        "sealed_evaluation_accessed": False,
        "limitations": [
            "Reuses development EB-OEH curve outputs; no estimator recomputation.",
            "Current EB plug-in is not cross-fitted and is not the exact KL optimizer.",
            "No B1/B2/B3 grouped out-of-fold learner-requirement evaluation.",
            "Pilot cases sharing an MDP are not independent units.",
        ],
        "all": all_summary,
        "stochastic": stochastic_summary,
    }
    write_csv_rows(args.output_dir / "profile.csv", profile_rows, list(profile_rows[0]))
    write_csv_rows(args.output_dir / "summary.csv", summaries, list(summaries[0]))
    write_json(args.output_dir / "report.json", payload)
    (args.output_dir / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    write_json(args.output_dir / "audit.json", {
        "profile_rows": len(profile_rows), "cases": len(summaries),
        "expected_profile_rows": len(summaries)*(args.compute_budget_qvi),
        "row_count_ok": len(profile_rows) == len(summaries)*args.compute_budget_qvi,
        "unique_dataset_ids": len({row["dataset_id"] for row in summaries}),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
