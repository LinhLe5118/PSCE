#!/usr/bin/env python
"""Paired development-only ablation of IQL's expectile parameter.

Every expectile is evaluated on the same MDP, behavior policy, transition
prefix and training seed.  The registered 0.7 cell must reproduce the closed
2026-08-22 IQL results exactly; otherwise the run aborts before handoff.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from empirical_requirement import iql_tabular  # noqa: E402
from evaluation.io import reserve_output_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from scripts.training.run_empirical_requirement import _load_run, _score_policy, _training_seed_index  # noqa: E402


REGISTERED_EXPECTILE = 0.7
EXPECTILE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
EARLY_MAX_MULTIPLIER = 5

RUN_FIELDS = (
    "expectile", "mdp_id", "policy_family", "training_size_multiplier",
    "training_seed", "normalized_return", "success", "run_status",
)
N_EMP_FIELDS = (
    "expectile", "mdp_id", "policy_family", "n_emp_multiplier",
    "n_emp_transitions", "n_emp_censored", "category", "success_counts",
    "success_rates",
)
CURVE_FIELDS = (
    "expectile", "training_size_multiplier", "valid_runs", "invalid_runs",
    "mean_normalized_return", "median_normalized_return", "q10_normalized_return",
    "q25_normalized_return", "q75_normalized_return", "q90_normalized_return",
    "success_rate",
)
PAIR_FIELDS = (
    "comparison_expectile", "registered_category", "comparison_category", "groups",
)


def validate_expectiles(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if result != EXPECTILE_GRID:
        raise ValueError(
            f"Expectile grid must equal the pre-registered grid {EXPECTILE_GRID}; got {result}."
        )
    return result


def requirement_category(multiplier: int | None) -> str:
    if multiplier is None:
        return "right_censored"
    return "early" if int(multiplier) <= EARLY_MAX_MULTIPLIER else "late"


def derive_requirement(successes: Sequence[bool], multipliers: Sequence[int], runs: int) -> tuple[int | None, tuple[int, ...]]:
    if len(successes) != len(multipliers) * runs:
        raise ValueError("Success vector does not cover the registered size/seed grid.")
    counts = tuple(sum(successes[index * runs:(index + 1) * runs]) for index in range(len(multipliers)))
    required = int(PROTOCOL_CONFIG["learning"]["success_runs_required"])
    for index, multiplier in enumerate(multipliers):
        if all(count >= required for count in counts[index:]):
            return int(multiplier), counts
    return None, counts


def distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["category"]) for row in rows)
    histogram = Counter(
        "censored" if row["n_emp_multiplier"] in (None, "") else str(int(row["n_emp_multiplier"]))
        for row in rows
    )
    finite = [int(row["n_emp_multiplier"]) for row in rows if row["n_emp_multiplier"] not in (None, "")]
    total = len(rows)
    return {
        "groups": total,
        "early": counts["early"],
        "late": counts["late"],
        "right_censored": counts["right_censored"],
        "polar_mass_count": counts["early"] + counts["right_censored"],
        "polar_mass_fraction": (counts["early"] + counts["right_censored"]) / total,
        "finite_median_multiplier": None if not finite else statistics.median(finite),
        "histogram": {key: histogram.get(key, 0) for key in ("1", "2", "5", "10", "20", "50", "censored")},
    }


def paired_transition_rows(n_emp_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_expectile: dict[float, dict[tuple[str, str], str]] = defaultdict(dict)
    for row in n_emp_rows:
        by_expectile[float(row["expectile"])][(str(row["mdp_id"]), str(row["policy_family"]))] = str(row["category"])
    registered = by_expectile[REGISTERED_EXPECTILE]
    categories = ("early", "late", "right_censored")
    output: list[dict[str, Any]] = []
    for expectile in EXPECTILE_GRID:
        if expectile == REGISTERED_EXPECTILE:
            continue
        comparison = by_expectile[expectile]
        if set(comparison) != set(registered):
            raise ValueError(f"Paired groups differ for expectile {expectile}.")
        counts = Counter((registered[key], comparison[key]) for key in registered)
        for registered_category in categories:
            for comparison_category in categories:
                output.append({
                    "comparison_expectile": expectile,
                    "registered_category": registered_category,
                    "comparison_category": comparison_category,
                    "groups": counts[(registered_category, comparison_category)],
                })
    return output


def validate_registered_control(run_rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {
        (str(row["mdp_id"]), str(row["policy_family"]), int(row["training_size_multiplier"]), int(row["training_seed"])): row
        for row in run_rows if float(row["expectile"]) == REGISTERED_EXPECTILE
    }
    reference = {
        (str(row["mdp_id"]), str(row["policy_family"]), int(row["training_size_multiplier"]), int(row["training_seed"])): row
        for row in reference_rows if str(row["algorithm"]) == "iql_tabular"
    }
    if set(observed) != set(reference) or len(observed) != 4500:
        raise ValueError(f"Registered control coverage mismatch: observed={len(observed)}, reference={len(reference)}.")
    mismatches: list[str] = []
    for key in sorted(reference):
        left, right = observed[key], reference[key]
        same = (
            float(left["normalized_return"]) == float(right["normalized_return"])
            and str(left["success"]).lower() == str(right["success"]).lower()
            and str(left["run_status"]) == str(right["run_status"])
        )
        if not same:
            mismatches.append("/".join(map(str, key)))
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise ValueError(f"Registered expectile 0.7 failed exact reproduction at {mismatches}.")
    return {"rows": len(observed), "exact_match": True, "mismatches": 0}


def _curve_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["expectile"]), int(row["training_size_multiplier"]))].append(row)
    output: list[dict[str, Any]] = []
    for (expectile, multiplier), group in sorted(grouped.items()):
        valid = np.asarray([float(row["normalized_return"]) for row in group if row["run_status"] == "observed"], dtype=np.float64)
        output.append({
            "expectile": expectile,
            "training_size_multiplier": multiplier,
            "valid_runs": len(valid),
            "invalid_runs": len(group) - len(valid),
            "mean_normalized_return": float(np.mean(valid)),
            "median_normalized_return": float(np.median(valid)),
            "q10_normalized_return": float(np.quantile(valid, 0.10)),
            "q25_normalized_return": float(np.quantile(valid, 0.25)),
            "q75_normalized_return": float(np.quantile(valid, 0.75)),
            "q90_normalized_return": float(np.quantile(valid, 0.90)),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
        })
    return output


def run(dataset_dir: Path, reference_dir: Path, expectiles: Sequence[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expectiles = validate_expectiles(expectiles)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_mode") != "review":
        raise ValueError("Expectile ablation requires a review-mode dataset bundle.")
    records = [item for item in manifest["datasets"] if item["stream"] == "training"]
    if not records or any(item["benchmark_split"] != "development" for item in records):
        raise ValueError("Expectile ablation accepts development training records only.")

    learning = PROTOCOL_CONFIG["learning"]
    collection = PROTOCOL_CONFIG["collection"]
    multipliers = tuple(int(value) for value in collection["size_multipliers"])
    training_seeds = tuple(int(value) for value in collection["training_seed_indices"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["instance_id"]), str(record["policy_family"]))].append(record)
    if len(grouped) != 75:
        raise ValueError(f"Expected all 75 development MDP-policy groups, found {len(grouped)}.")

    instances = {str(item["instance_id"]): item for item in manifest["instances"]}
    run_rows: list[dict[str, Any]] = []
    n_emp_rows: list[dict[str, Any]] = []
    total = len(grouped) * len(expectiles) * len(multipliers) * len(training_seeds)
    completed = 0
    for (instance_id, family), group in sorted(grouped.items()):
        instance = instances[instance_id]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        random_return = float(instance["random_return"])
        dynamic_range = float(instance["dynamic_range"])
        indexed: dict[tuple[int, int], dict[str, Any]] = {}
        for record in group:
            multiplier = record.get("size_multiplier")
            if multiplier is not None:
                indexed[(int(multiplier), _training_seed_index(record))] = record
        expected = {(m, seed) for m in multipliers for seed in training_seeds}
        if set(indexed) != expected:
            raise ValueError(f"Training grid mismatch for {instance_id}/{family}.")
        cache: dict[str, dict[str, np.ndarray]] = {}
        scores: dict[float, list[bool]] = {expectile: [] for expectile in expectiles}
        for multiplier in multipliers:
            for seed in training_seeds:
                transitions = _load_run(dataset_dir, indexed[(multiplier, seed)], cache)
                for expectile in expectiles:
                    policy = iql_tabular(
                        transitions,
                        num_states=mdp.num_states,
                        num_actions=mdp.num_actions,
                        gamma=mdp.gamma,
                        expectile=expectile,
                        inverse_temperature=float(learning["iql_inverse_temperature"]),
                        iterations=int(learning["iterations"]),
                        tolerance=float(learning["tolerance"]),
                    )
                    normalized_return, status = _score_policy(
                        mdp, policy, random_return=random_return, dynamic_range=dynamic_range
                    )
                    success = status == "observed" and normalized_return >= float(learning["success_normalized_return"])
                    scores[expectile].append(success)
                    run_rows.append({
                        "expectile": expectile,
                        "mdp_id": instance_id,
                        "policy_family": family,
                        "training_size_multiplier": multiplier,
                        "training_seed": seed,
                        "normalized_return": normalized_return,
                        "success": success,
                        "run_status": status,
                    })
                    completed += 1
        print(f"[{completed}/{total}] {instance_id} {family}", flush=True)
        sa = int(mdp.num_states * mdp.num_actions)
        for expectile in expectiles:
            requirement, counts = derive_requirement(scores[expectile], multipliers, len(training_seeds))
            n_emp_rows.append({
                "expectile": expectile,
                "mdp_id": instance_id,
                "policy_family": family,
                "n_emp_multiplier": requirement,
                "n_emp_transitions": None if requirement is None else requirement * sa,
                "n_emp_censored": requirement is None,
                "category": requirement_category(requirement),
                "success_counts": json.dumps(counts, separators=(",", ":")),
                "success_rates": json.dumps([count / len(training_seeds) for count in counts], separators=(",", ":")),
            })

    if len(run_rows) != 22500 or len(n_emp_rows) != 375:
        raise ValueError(f"Incomplete ablation: {len(run_rows)} runs, {len(n_emp_rows)} groups.")
    reference_path = reference_dir / "rl_run_results.csv"
    control = validate_registered_control(run_rows, read_csv_rows(reference_path))
    curves = _curve_rows(run_rows)
    transitions = paired_transition_rows(n_emp_rows)
    distributions: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    for expectile in expectiles:
        subset = [row for row in n_emp_rows if float(row["expectile"]) == expectile]
        distributions[str(expectile)] = distribution_summary(subset)
        by_family[str(expectile)] = {
            family: distribution_summary([row for row in subset if row["policy_family"] == family])
            for family in sorted({str(row["policy_family"]) for row in subset})
        }
    report = {
        "scope": "development-only paired IQL expectile intervention",
        "expectile_grid": list(expectiles),
        "registered_expectile": REGISTERED_EXPECTILE,
        "fixed_parameters": {
            "inverse_temperature": learning["iql_inverse_temperature"],
            "iterations": learning["iterations"],
            "tolerance": learning["tolerance"],
            "success_normalized_return": learning["success_normalized_return"],
            "success_runs_required": learning["success_runs_required"],
            "training_size_multipliers": list(multipliers),
            "training_seed_indices": list(training_seeds),
        },
        "counts": {"mdp_policy_groups_per_expectile": 75, "runs": len(run_rows), "n_emp_rows": len(n_emp_rows)},
        "registered_control": control,
        "distribution": distributions,
        "distribution_by_policy_family": by_family,
        "input_sha256": {
            "dataset_manifest": sha256_file(manifest_path),
            "reference_rl_run_results": sha256_file(reference_path),
        },
    }
    return run_rows, n_emp_rows, curves, transitions, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expectiles", default=",".join(map(str, EXPECTILE_GRID)))
    args = parser.parse_args()
    reserve_output_dir(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    expectiles = [float(value) for value in args.expectiles.split(",") if value.strip()]
    run_rows, n_emp_rows, curves, transitions, report = run(args.dataset_dir, args.reference_dir, expectiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "expectile_run_results.csv", run_rows, RUN_FIELDS)
    write_csv_rows(args.output_dir / "expectile_n_emp_distribution.csv", n_emp_rows, N_EMP_FIELDS)
    write_csv_rows(args.output_dir / "expectile_learning_curves.csv", curves, CURVE_FIELDS)
    write_csv_rows(args.output_dir / "expectile_paired_transitions.csv", transitions, PAIR_FIELDS)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    report["git_commit"] = commit
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name)
        for name in (
            "expectile_run_results.csv", "expectile_n_emp_distribution.csv",
            "expectile_learning_curves.csv", "expectile_paired_transitions.csv",
        )
    }
    write_json(args.output_dir / "expectile_ablation_report.json", report)
    for expectile in EXPECTILE_GRID:
        print(expectile, report["distribution"][str(expectile)])
    print(f"Registered 0.7 exact reproduction: {report['registered_control']['exact_match']}")


if __name__ == "__main__":
    main()
