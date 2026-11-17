#!/usr/bin/env python
"""Run the frozen pilot-only target-return OEH-v3 profile."""
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
from oeh_risk_v3 import estimate_oeh_risk_v3
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
        "status": "started", "method": "oeh_target_return_risk_v3",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
        "index_sha256": sha256_file(index),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "code/oeh_risk_v3.py", "code/oeh_crossfit.py",
            "code/oeh_final_profile.py", "code/oeh_eh_aligned.py",
            "code/stochastic_features.py", "scripts/analysis/run_oeh_risk_v3.py",
        )},
        "design": {
            "seed": 20260920, "gamma": .95, "epsilon": .05,
            "k_max": args.k_max,
            "estimand": "pilot-observed action support",
            "requirement": "return-weighted empirical-Bernstein risk inversion",
            "trajectory_crossfit_role": "policy instability diagnostic only",
            "single_supported_action": "zero action-identification cost",
            "feature_block": [
                "log1p_requirement", "selected_qvi_fraction",
                "risk_at_pilot", "policy_instability",
            ],
        },
        "outcome_labels_accessed": False,
        "training_buffers_accessed": False,
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
            result = estimate_oeh_risk_v3(
                arrays, record["initial_state_distribution"],
                num_states=record["num_states"], num_actions=record["num_actions"],
                k_max=args.k_max)
            statuses = Counter(item["status"] for item in result["profile"])
            choice = result["primary_data_selection"]
            summaries.append({
                **identity,
                "identified_depths": statuses["identified"],
                "epsilon_trivial_depths": statuses["epsilon_trivial"],
                "unresolved_depths": statuses["unresolved"],
                "not_epsilon_solvable_depths": statuses["not_epsilon_solvable"],
                "selected_depth": choice["depth"],
                "selected_qvi_index": choice["qvi_index"],
                "selected_n_hat_eb": choice["n_hat_eb"],
                "current_budget_sufficient_depths": sum(
                    row["current_budget_sufficient"] for row in result["records"]),
            })
            by_depth = {row["depth"]: row for row in result["records"]}
            for item in result["profile"]:
                detail = by_depth[item["depth"]]
                curves.append({
                    **identity, **item,
                    "planning_regret": detail["planning_regret"],
                    "decision_risk_at_zero": detail["decision_risk_at_zero"],
                    "risk_at_pilot": detail["risk_at_pilot"],
                    "relevant_states": detail["relevant_states"],
                    "comparison_states": detail["comparison_states"],
                    "single_action_states": detail["single_action_states"],
                    "tied_states": detail["tied_states"],
                    "minimum_gap": detail["minimum_gap"],
                    "maximum_impact": detail["maximum_impact"],
                    "policy_instability": detail["policy_instability"],
                    "current_budget_sufficient": detail["current_budget_sufficient"],
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
    case_status = Counter()
    for row in summaries:
        if row["identified_depths"]:
            case_status["selectable"] += 1
        elif row["epsilon_trivial_depths"]:
            case_status["epsilon_trivial"] += 1
        elif row["unresolved_depths"]:
            case_status["unresolved"] += 1
        else:
            case_status["not_epsilon_solvable"] += 1
    report = {
        **plan, "status": "complete" if not errors else "completed_with_errors",
        "cases": len(summaries), "profile_rows": len(curves),
        "case_status_counts": dict(case_status), "errors": errors,
        "current_budget_sufficient_cases": sum(
            row["current_budget_sufficient_depths"] > 0 for row in summaries),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "REPORT.md").write_text(
        "# OEH-v3 target-return risk profile\n\n"
        f"Status: **{report['status']}**. Cases: {len(summaries)}; profile rows: "
        f"{len(curves)}; errors: {len(errors)}.\n\n"
        f"Case statuses: `{dict(case_status)}`. Current-budget sufficient cases: "
        f"`{report['current_budget_sufficient_cases']}`.\n\n"
        "The estimator used pilot transitions only. It did not access learner "
        "outcomes, training buffers, or the sealed evaluation split. It is a "
        "support-constrained forecasting proxy, not a confidence certificate.\n",
        encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
