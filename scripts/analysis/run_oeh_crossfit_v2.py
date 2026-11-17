#!/usr/bin/env python
"""Run target-aligned cross-fitted OEH v2 on development pilots."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import (reserve_output_dir, sha256_file, verified_dataset_path,
                           write_csv_rows, write_json)
from oeh_crossfit_v2 import estimate_crossfit_oeh_v2
from scripts.analysis.run_oeh_development_smoke import development_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-mdps", type=int)
    parser.add_argument("--k-max", type=int, default=100)
    args = parser.parse_args(argv)
    manifest, records, labels, index = development_inputs(args.dataset_dir, args.limit_mdps)
    reserve_output_dir(args.output_dir)
    plan = {
        "status": "started", "method": "target_aligned_crossfit_oeh_v2",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
        "index_sha256": sha256_file(index),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "code/oeh_crossfit_v2.py", "code/oeh_crossfit.py",
            "code/oeh_final_profile.py", "code/oeh_eh_aligned.py",
            "scripts/analysis/run_oeh_crossfit_v2.py",
        )},
        "design": {
            "trajectory_folds": 2, "seed": 20260920, "gamma": .95,
            "epsilon": .05, "delta": .05, "k_max": args.k_max,
            "support": "critical top/runner pair only",
            "aggregation": "return-impact trimming within epsilon minus planning regret",
            "requirement": "future two-sample empirical-Bernstein planning scale",
            "current_budget_sufficiency": "reported separately",
        },
        "sealed_evaluation_accessed": False,
    }
    write_json(args.output_dir / "plan.json", plan)
    started = time.perf_counter()
    summaries, curves, errors = [], [], []
    for position, record in enumerate(records, 1):
        label = labels[record["dataset_id"]]
        identity = {
            "dataset_id": record["dataset_id"], "mdp_id": label["instance_id"],
            "policy_family": label["policy_family"],
            "pilot_multiplier": record["size_multiplier"],
            "num_transitions": record["size"],
        }
        try:
            with np.load(verified_dataset_path(args.dataset_dir, record), allow_pickle=False) as loaded:
                arrays = {name: loaded[name][:record["size"]] for name in loaded.files}
            result = estimate_crossfit_oeh_v2(
                arrays, record["initial_state_distribution"],
                num_states=record["num_states"], num_actions=record["num_actions"],
                k_max=args.k_max)
            statuses = Counter(item["status"] for item in result["profile"])
            data = result["primary_data_selection"]
            reference = result["eh_reference_selection"]
            current_sufficient = sum(
                row["current_budget_sufficient_both"] for row in result["records"])
            summaries.append({
                **identity,
                "fold_trajectories_first": result["fold_trajectory_counts"][0],
                "fold_trajectories_second": result["fold_trajectory_counts"][1],
                "identified_depths": statuses["identified"],
                "epsilon_trivial_depths": statuses["epsilon_trivial"],
                "unsupported_depths": statuses["unsupported"],
                "unresolved_depths": statuses["unresolved"],
                "not_epsilon_solvable_depths": statuses["not_epsilon_solvable"],
                "current_budget_sufficient_depths": current_sufficient,
                "data_selected_depth": data["depth"],
                "data_selected_qvi_index": data["qvi_index"],
                "data_selected_n_hat_eb": data["n_hat_eb"],
                "eh_selected_depth": reference["depth"],
                "eh_selected_qvi_index": reference["qvi_index"],
            })
            by_depth = {row["depth"]: row for row in result["records"]}
            for item in result["profile"]:
                details = by_depth[item["depth"]]
                curves.append({
                    **identity, **item,
                    "planning_regret_max": details["planning_regret_max"],
                    "relevant_states_max": details["relevant_states_max"],
                    "retained_states_max": details["retained_states_max"],
                    "ignored_states_max": details["ignored_states_max"],
                    "ignored_impact_max": details["ignored_impact_max"],
                    "unsupported_states_max": details["unsupported_states_max"],
                    "unresolved_states_max": details["unresolved_states_max"],
                    "minimum_empirical_gap": details["minimum_empirical_gap"],
                    "current_budget_sufficient_both": details["current_budget_sufficient_both"],
                })
            print(f"[{position}/{len(records)}] {identity['mdp_id']} "
                  f"{identity['policy_family']} x{identity['pilot_multiplier']}: "
                  f"{dict(statuses)}", flush=True)
        except Exception as exc:
            errors.append({**identity, "error": repr(exc)})
            print(f"[{position}/{len(records)}] ERROR {exc!r}", flush=True)
        write_json(args.output_dir / "progress.json", {
            "processed": position, "total": len(records), "errors": errors})
    if summaries:
        write_csv_rows(args.output_dir / "summary.csv", summaries, list(summaries[0]))
    if curves:
        write_csv_rows(args.output_dir / "profile.csv", curves, list(curves[0]))
    status_counts = Counter()
    for row in summaries:
        if row["identified_depths"]:
            status_counts["selectable"] += 1
        elif row["epsilon_trivial_depths"]:
            status_counts["epsilon_trivial"] += 1
        elif row["unsupported_depths"]:
            status_counts["unsupported"] += 1
        elif row["unresolved_depths"]:
            status_counts["unresolved"] += 1
        else:
            status_counts["not_epsilon_solvable"] += 1
    report = {
        **plan, "status": "complete" if not errors else "completed_with_errors",
        "cases": len(summaries), "profile_rows": len(curves),
        "case_status_counts": dict(status_counts), "errors": errors,
        "current_budget_sufficient_cases": sum(
            row["current_budget_sufficient_depths"] > 0 for row in summaries),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "REPORT.md").write_text(
        "# Target-aligned cross-fitted OEH v2\n\n"
        f"Status: **{report['status']}**. Cases: {len(summaries)}; profile rows: "
        f"{len(curves)}; errors: {len(errors)}.\n\n"
        f"Case statuses: `{dict(status_counts)}`. Current-budget sufficient cases: "
        f"`{report['current_budget_sufficient_cases']}`.\n\n"
        "This version estimates future required data from independent folds; it "
        "does not require the current pilot to have already reached that scale. "
        "It remains a local approximation rather than the exact KL optimizer.\n",
        encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
