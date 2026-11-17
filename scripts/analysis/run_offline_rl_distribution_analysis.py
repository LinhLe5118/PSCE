#!/usr/bin/env python
"""Week 4: report the full IQL/CFQI empirical-requirement distribution.

This is a read-only re-analysis of the closed 2026-08-22 development run.  It
does not fit either learner again.  The purpose is to replace the misleading
finite-case median with the complete distribution, including right censoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json


ALGORITHMS = ("cfqi_tabular", "iql_tabular")
MULTIPLIERS = (1, 2, 5, 10, 20, 50)
EARLY_MAX_MULTIPLIER = 5

GROUP_FIELDS = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "n_emp_multiplier",
    "n_emp_transitions",
    "n_emp_censored",
    "distribution_bin",
    "success_counts",
    "max_success_count",
    "final_success_count",
    "success_count_non_monotone",
)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q10": None, "q25": None,
                "median": None, "q75": None, "q90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    points = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(array.size),
        **dict(zip(("min", "q10", "q25", "median", "q75", "q90", "max"),
                   (float(value) for value in points))),
    }


def classify_requirement(multiplier: int | None, censored: bool) -> str:
    """Pre-registered bins: <=5, 10..50, and censored at 50."""

    if censored:
        if multiplier is not None:
            raise ValueError("A right-censored N_emp row cannot carry a finite multiplier.")
        return "right_censored_at_50"
    if multiplier is None or multiplier not in MULTIPLIERS:
        raise ValueError(f"Finite N_emp multiplier must belong to {MULTIPLIERS}.")
    return "early_le_5" if multiplier <= EARLY_MAX_MULTIPLIER else "late_10_to_50"


def analyse(
    rl_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    n_emp_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_run_rows = 75 * len(ALGORITHMS) * len(MULTIPLIERS) * 10
    if len(rl_rows) != expected_run_rows:
        raise ValueError(f"Expected {expected_run_rows} registered runs, found {len(rl_rows)}.")

    grouped_runs: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    returns: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rl_rows:
        algorithm = str(row["algorithm"])
        multiplier = int(row["training_size_multiplier"])
        if algorithm not in ALGORITHMS or multiplier not in MULTIPLIERS:
            raise ValueError(f"Unexpected run cell {algorithm}/{multiplier}.")
        if str(row["run_status"]) != "observed":
            raise ValueError("The closed run is expected to contain no invalid registered rows.")
        key = (str(row["mdp_id"]), str(row["policy_family"]), algorithm, multiplier)
        grouped_runs[key].append(row)
        returns[(algorithm, multiplier)].append(float(row["normalized_return"]))
    if any(len(rows) != 10 for rows in grouped_runs.values()):
        raise ValueError("Every MDP-policy-algorithm-size cell must contain ten seeds.")

    group_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in n_emp_rows:
        key3 = (str(source["mdp_id"]), str(source["policy_family"]), str(source["algorithm"]))
        if key3 in seen:
            raise ValueError(f"Duplicate N_emp row: {key3}.")
        seen.add(key3)
        if key3[2] not in ALGORITHMS:
            raise ValueError(f"Unexpected N_emp algorithm {key3[2]}.")
        censored = _as_bool(source["n_emp_censored"])
        multiplier = None if str(source["n_emp_multiplier"]).strip() == "" else int(source["n_emp_multiplier"])
        transitions = None if str(source["n_emp_transitions"]).strip() == "" else int(source["n_emp_transitions"])
        observed_counts = [
            sum(_as_bool(row["success"]) for row in grouped_runs[(*key3, size)])
            for size in MULTIPLIERS
        ]
        recorded_counts = [int(value) for value in json.loads(str(source["success_counts"]))]
        if observed_counts != recorded_counts:
            raise ValueError(f"success_counts disagree with rl_run_results for {key3}.")
        # The registered label is the first budget whose *entire suffix* stays
        # at 8/10 or above, not merely the first isolated threshold crossing.
        # This distinction matters because several observed curves regress at
        # a later size and are therefore correctly right-censored.
        stable_first = next(
            (
                size
                for index, size in enumerate(MULTIPLIERS)
                if all(count >= 8 for count in observed_counts[index:])
            ),
            None,
        )
        if stable_first != multiplier:
            raise ValueError(f"N_emp is not the first stable 8/10-success budget for {key3}.")
        non_monotone = any(right < left for left, right in zip(observed_counts, observed_counts[1:]))
        group_rows.append({
            "mdp_id": key3[0],
            "policy_family": key3[1],
            "algorithm": key3[2],
            "n_emp_multiplier": "" if multiplier is None else multiplier,
            "n_emp_transitions": "" if transitions is None else transitions,
            "n_emp_censored": censored,
            "distribution_bin": classify_requirement(multiplier, censored),
            "success_counts": json.dumps(observed_counts, separators=(",", ":")),
            "max_success_count": max(observed_counts),
            "final_success_count": observed_counts[-1],
            "success_count_non_monotone": non_monotone,
        })
    if len(seen) != 75 * len(ALGORITHMS):
        raise ValueError(f"Expected 150 N_emp groups, found {len(seen)}.")

    curve_lookup = {
        (str(row["algorithm"]), int(row["training_size_multiplier"])): row
        for row in curve_rows if str(row["algorithm"]) in ALGORITHMS
    }
    if set(curve_lookup) != {(algorithm, size) for algorithm in ALGORITHMS for size in MULTIPLIERS}:
        raise ValueError("learning_curve_summary is missing a registered algorithm-size cell.")

    algorithm_report: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        subset = [row for row in group_rows if row["algorithm"] == algorithm]
        multiplier_counts = Counter(
            "censored" if row["n_emp_multiplier"] == "" else str(row["n_emp_multiplier"])
            for row in subset
        )
        bin_counts = Counter(str(row["distribution_bin"]) for row in subset)
        finite_transitions = [int(row["n_emp_transitions"]) for row in subset if row["n_emp_transitions"] != ""]
        family_counts: dict[str, dict[str, int]] = {}
        for family in sorted({str(row["policy_family"]) for row in subset}):
            family_counts[family] = dict(Counter(
                str(row["distribution_bin"]) for row in subset if row["policy_family"] == family
            ))
        curves: list[dict[str, Any]] = []
        for size in MULTIPLIERS:
            source = curve_lookup[(algorithm, size)]
            quantiles = _quantiles(returns[(algorithm, size)])
            if int(source["valid_runs"]) != quantiles["count"]:
                raise ValueError(f"Curve count disagrees for {algorithm}/{size}.")
            if not np.isclose(float(source["median_normalized_return"]), float(quantiles["median"])):
                raise ValueError(f"Curve median disagrees for {algorithm}/{size}.")
            curves.append({
                "multiplier": size,
                "success_rate": float(source["success_rate"]),
                "normalized_return": quantiles,
            })
        extreme = bin_counts["early_le_5"] + bin_counts["right_censored_at_50"]
        algorithm_report[algorithm] = {
            "groups": len(subset),
            "finite_groups": len(finite_transitions),
            "right_censored_groups": bin_counts["right_censored_at_50"],
            "n_emp_multiplier_distribution": {
                str(size): multiplier_counts[str(size)] for size in MULTIPLIERS
            } | {"right_censored": multiplier_counts["censored"]},
            "pre_registered_bins": {
                name: bin_counts[name]
                for name in ("early_le_5", "late_10_to_50", "right_censored_at_50")
            },
            "early_or_censored_share": extreme / len(subset),
            "finite_transition_quantiles": _quantiles(finite_transitions),
            "groups_with_non_monotone_success_counts": sum(
                bool(row["success_count_non_monotone"]) for row in subset
            ),
            "groups_with_zero_successes_at_all_sizes": sum(
                int(row["max_success_count"]) == 0 for row in subset
            ),
            "by_policy_family": family_counts,
            "learning_curves": curves,
        }

    iql = algorithm_report["iql_tabular"]
    cfqi = algorithm_report["cfqi_tabular"]
    report = {
        "scope": "closed development-only 2026-08-22 run; no retraining",
        "hypothesis": (
            "IQL's registered expectile-0.7 configuration is associated with an early-or-"
            "right-censored N_emp distribution rather than merely a right-skewed finite distribution."
        ),
        "bin_definition": {
            "early_le_5": "finite N_emp multiplier in {1,2,5}",
            "late_10_to_50": "finite N_emp multiplier in {10,20,50}",
            "right_censored_at_50": "fewer than 8/10 successes through multiplier 50",
        },
        "algorithms": algorithm_report,
        "contrast": {
            "iql_minus_cfqi_early_or_censored_share": (
                float(iql["early_or_censored_share"]) - float(cfqi["early_or_censored_share"])
            ),
            "iql_minus_cfqi_censoring_rate": (
                int(iql["right_censored_groups"]) - int(cfqi["right_censored_groups"])
            ) / 75,
        },
        "causal_scope": (
            "Descriptive compatibility only: these files do not isolate expectile from the other "
            "algorithmic differences between tabular IQL and CFQI."
        ),
    }
    return group_rows, report


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    source_paths = {
        name: input_dir / name
        for name in ("rl_run_results.csv", "learning_curve_summary.csv", "empirical_sample_complexity.csv")
    }
    rows, report = analyse(
        read_csv_rows(source_paths["rl_run_results.csv"]),
        read_csv_rows(source_paths["learning_curve_summary.csv"]),
        read_csv_rows(source_paths["empirical_sample_complexity.csv"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "n_emp_distribution.csv", rows, GROUP_FIELDS)
    report["source_sha256"] = {name: sha256_file(path) for name, path in source_paths.items()}
    report["artifact_sha256"] = sha256_file(output_dir / "n_emp_distribution.csv")
    write_json(output_dir / "offline_rl_distribution_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir)
    for algorithm, values in report["algorithms"].items():
        print(algorithm, values["n_emp_multiplier_distribution"])
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
