#!/usr/bin/env python
"""Audit the frozen OEH-v2 development and controlled-factorial artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STATUSES = {
    "identified", "epsilon_trivial", "unsupported", "unresolved",
    "not_epsilon_solvable",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--factorial-dir", type=Path, required=True)
    parser.add_argument("--legacy-factorial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "This artifact audit requires pandas; install the analysis environment."
        ) from error
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    profile_report = json.loads((args.profile_dir / "report.json").read_text(encoding="utf-8"))
    profile = pd.read_csv(args.profile_dir / "profile.csv")
    factorial_report = json.loads((args.factorial_dir / "report.json").read_text(encoding="utf-8"))
    factorial = pd.read_csv(args.factorial_dir / "factorial_rows.csv")
    legacy_report = json.loads((args.legacy_factorial_dir / "report.json").read_text(encoding="utf-8"))

    checks = {}
    checks["profile_completed_without_errors"] = (
        profile_report["status"] == "complete" and not profile_report["errors"]
        and len(profile) == 22_725)
    checks["statuses_are_explicit"] = set(profile.status.unique()) <= STATUSES
    identified = profile[profile.status.eq("identified")]
    trivial = profile[profile.status.eq("epsilon_trivial")]
    censored = profile[~profile.status.isin(["identified", "epsilon_trivial"])]
    checks["identified_requirements_are_finite_positive"] = (
        len(identified) > 0
        and identified.n_hat_eb.map(lambda x: math.isfinite(x) and x > 0).all())
    checks["no_artificial_one_sample_floor"] = not identified.n_hat_eb.eq(1.0).any()
    checks["epsilon_trivial_is_zero_and_labeled"] = (
        len(trivial) > 0 and trivial.n_hat_eb.eq(0.0).all())
    checks["censored_rows_have_no_finite_requirement"] = censored.n_hat_eb.isna().all()
    checks["full_profile_has_nonconstant_finite_signal"] = identified.n_hat_eb.nunique() > 1

    factorial_identified = factorial[factorial.oeh_status.eq("identified")]
    checks["factorial_completed"] = (
        factorial_report["status"] == "complete" and len(factorial) == 810)
    checks["factorial_has_finite_identification"] = len(factorial_identified) > 0
    checks["factorial_requirements_are_nonconstant"] = (
        factorial_identified.oeh_requirement.nunique() > 1)
    checks["factorial_has_no_one_sample_floor"] = (
        not factorial_identified.oeh_requirement.eq(1.0).any())
    rates = factorial.groupby("budget_multiplier").oeh_resolved.mean().to_dict()
    checks["aggregate_resolution_rises_with_budget"] = (
        rates[2] < rates[10] < rates[50])
    legacy_counts = legacy_report["oeh_status_counts"]
    checks["legacy_failure_preserved_for_before_after"] = (
        legacy_counts.get("identified", 0) == 0
        and legacy_counts.get("unsupported", 0) + legacy_counts.get("unresolved", 0) == 810)
    checks = {name: bool(value) for name, value in checks.items()}

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "profile": {
            "rows": len(profile), "identified_rows": len(identified),
            "epsilon_trivial_rows": len(trivial),
            "identified_unique_requirements": int(identified.n_hat_eb.nunique()),
            "identified_requirement_min": float(identified.n_hat_eb.min()),
            "identified_requirement_max": float(identified.n_hat_eb.max()),
        },
        "factorial": {
            "rows": len(factorial), "identified_cells": len(factorial_identified),
            "identified_unique_requirements": int(factorial_identified.oeh_requirement.nunique()),
            "resolved_rate_by_budget": {str(k): float(v) for k, v in rates.items()},
            "status_counts": factorial_report["oeh_status_counts"],
        },
        "legacy_factorial_status_counts": legacy_counts,
        "scope": (
            "Mechanical and implementation audit on development data; not an exact-KL "
            "theorem, held-out validation, or evidence of predictive utility."),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OEH v2 final development audit", "",
        f"Status: **{report['status'].upper()}**.", "",
        "| Check | Pass |", "|---|---:|",
        *[f"| {name} | {value} |" for name, value in checks.items()], "",
        (f"The 22,725-row profile contains {len(identified)} nontrivial identified "
         f"rows with {identified.n_hat_eb.nunique()} distinct requirements and "
         f"{len(trivial)} explicitly labeled epsilon-trivial rows."),
        (f"The controlled factorial identifies {len(factorial_identified)}/810 cells; "
         f"all {factorial_identified.oeh_requirement.nunique()} finite requirements "
         "are distinct in this run and none equals the artificial floor 1."),
        f"Aggregate resolved rates by budget: `{report['factorial']['resolved_rate_by_budget']}`.", "",
        report["scope"], "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
