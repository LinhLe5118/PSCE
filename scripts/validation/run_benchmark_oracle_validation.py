"""Development-only benchmark and exact-oracle validation.

This is the Workstream 1 handoff defined by
``docs/week2/WEEK2_ARTIFACT_CONTRACT.md``.  It materializes only the 15
development MDPs, computes exact policy/value labels, validates the measured
``N_emp`` handoff, and adopts a precomputed support-restricted exact evidence
curve when one is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from environments.behavior_policies import behavior_policy_suite  # noqa: E402
from environments.benchmark_environments import (  # noqa: E402
    BENCHMARK_GAMMA,
    DEVELOPMENT_COUNT,
    benchmark_specs,
    resolve_dynamic_range,
)
from evaluation.validation.leakage import assert_no_oracle_leakage  # noqa: E402
from evaluation.validation.schema import validate_artifact  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG_PATH  # noqa: E402
from exact_solver import (  # noqa: E402
    dynamic_range,
    evaluate_policy,
    normalized_return,
    random_policy,
    value_iteration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _branching_factor(mdp: Any) -> float:
    successors = np.count_nonzero(np.asarray(mdp.P) > 0.0, axis=2)
    return float(np.mean(successors))


def _policy_residual(mdp: Any, policy: np.ndarray, values: np.ndarray) -> float:
    p = np.asarray(mdp.P, dtype=np.float64)
    rewards = np.asarray(mdp.R, dtype=np.float64)
    expected = rewards if rewards.ndim == 2 else np.sum(p * rewards, axis=2)
    q_values = expected + float(mdp.gamma) * np.einsum("san,n->sa", p, values)
    return float(np.max(np.abs(values - np.sum(policy * q_values, axis=1))))


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    specs = [spec for spec in benchmark_specs() if spec.split == "development"]
    if len(specs) != DEVELOPMENT_COUNT:
        raise AssertionError(f"Expected {DEVELOPMENT_COUNT} development MDPs, found {len(specs)}")

    config_path = PROTOCOL_CONFIG_PATH
    config_hash = _sha256(config_path)
    git_commit = _git_commit()
    benchmark_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    mdp_fingerprints: dict[str, str] = {}
    max_optimal_residual = 0.0
    max_policy_residual = 0.0
    regeneration_counts: list[int] = []

    for original_spec in specs:
        spec, mdp, rejected = resolve_dynamic_range(original_spec)
        regeneration_counts.append(rejected)
        solution = value_iteration(mdp)
        range_value, optimal_return, random_return = dynamic_range(mdp, solution=solution)
        max_optimal_residual = max(max_optimal_residual, float(solution.bellman_residual))
        mdp_fingerprints[spec.instance_id] = hashlib.sha256(
            np.asarray(mdp.P, dtype=np.float64).tobytes()
            + np.asarray(mdp.R, dtype=np.float64).tobytes()
        ).hexdigest()
        benchmark_rows.append({
            "mdp_id": spec.instance_id,
            "family": spec.family,
            "benchmark_split": spec.split,
            "num_states": mdp.num_states,
            "num_actions": mdp.num_actions,
            "gamma": mdp.gamma,
            "branching_factor": _branching_factor(mdp),
            "mixing_time_proxy": "",
            "dynamic_range": range_value,
            "excluded_dynamic_range": rejected > 0,
            "git_commit": git_commit,
            "config_hash": config_hash,
        })

        policies = behavior_policy_suite(
            mdp, seed=int(spec.seed) + 3_000_000, variant_index=int(spec.ordinal)
        )
        if len(policies) != 5 or sum(policy.uses_q_star for policy in policies) != 1:
            raise AssertionError("Each MDP must have five policies and exactly one Q-star user")
        for policy in policies:
            evaluation = evaluate_policy(mdp, policy.probabilities)
            max_policy_residual = max(
                max_policy_residual,
                _policy_residual(mdp, policy.probabilities, evaluation.values),
            )
            nr = normalized_return(mdp, policy.probabilities, solution=solution)
            oracle_rows.append({
                "mdp_id": spec.instance_id,
                "policy_family": policy.family,
                "behavior_return": evaluation.expected_return,
                "behavior_normalized_return": nr,
                "behavior_optimality_gap": 1.0 - nr,
                "optimal_return": optimal_return,
                "random_return": random_return,
                "uses_q_star": policy.uses_q_star,
            })

    # Regenerate independently and compare model fingerprints.  This tests the
    # benchmark recipes rather than merely checking deterministic CSV output.
    repeated = {}
    for original_spec in specs:
        spec, mdp, _ = resolve_dynamic_range(original_spec)
        repeated[spec.instance_id] = hashlib.sha256(
            np.asarray(mdp.P, dtype=np.float64).tobytes()
            + np.asarray(mdp.R, dtype=np.float64).tobytes()
        ).hexdigest()
    reproducible = repeated == mdp_fingerprints
    if not reproducible:
        raise AssertionError("Development benchmark regeneration is not reproducible")

    summary = {
        "scope": "development-only",
        "mdp_count": len(benchmark_rows),
        "oracle_row_count": len(oracle_rows),
        "family_counts": dict(Counter(row["family"] for row in benchmark_rows)),
        "policy_family_counts": dict(Counter(row["policy_family"] for row in oracle_rows)),
        "gamma": BENCHMARK_GAMMA,
        "dynamic_range_min": min(row["dynamic_range"] for row in benchmark_rows),
        "dynamic_range_regenerations": sum(regeneration_counts),
        "max_optimal_bellman_residual": max_optimal_residual,
        "max_policy_bellman_residual": max_policy_residual,
        "benchmark_reproducible": reproducible,
        "git_commit": git_commit,
        "config_hash": config_hash,
    }
    return benchmark_rows, oracle_rows, summary


def run(output_dir: Path, *, exact_k_curve_source: Path, n_emp_source: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows, oracle_rows, summary = build_rows()
    benchmark_path = output_dir / "benchmark_metadata.csv"
    oracle_path = output_dir / "exact_oracle_results.csv"
    exact_curve_path = output_dir / "exact_k_curve.csv"
    solution_path = output_dir / "exact_solution_summary.csv"
    _write_csv(benchmark_path, list(benchmark_rows[0]), benchmark_rows)
    _write_csv(oracle_path, list(oracle_rows[0]), oracle_rows)

    solution_rows = []
    for original_spec in (spec for spec in benchmark_specs() if spec.split == "development"):
        spec, mdp, _ = resolve_dynamic_range(original_spec)
        solution = value_iteration(mdp)
        solution_rows.append({
            "mdp_id": spec.instance_id,
            "v_star": json.dumps(solution.values.tolist(), separators=(",", ":")),
            "pi_star_action_mask": json.dumps(
                solution.optimal_action_mask.astype(int).tolist(), separators=(",", ":")
            ),
            "bellman_residual": solution.bellman_residual,
            "iterations": solution.iterations,
        })
    _write_csv(solution_path, list(solution_rows[0]), solution_rows)

    # The curve is support-restricted by data-only OEH output. Workstream 3
    # produced it before Workstream 1 existed; adopt it after strict validation.
    validate_artifact(exact_k_curve_source, "exact_k_curve")
    shutil.copyfile(exact_k_curve_source, exact_curve_path)
    validate_artifact(n_emp_source, "empirical_sample_complexity")
    validate_artifact(benchmark_path, "benchmark_metadata")
    validate_artifact(oracle_path, "exact_oracle_results")
    validate_artifact(exact_curve_path, "exact_k_curve")

    with exact_curve_path.open(encoding="utf-8", newline="") as handle:
        exact_rows = list(csv.DictReader(handle))
    with n_emp_source.open(encoding="utf-8", newline="") as handle:
        n_emp_rows = list(csv.DictReader(handle))
    development_ids = {row["mdp_id"] for row in benchmark_rows}
    if not {row["mdp_id"] for row in exact_rows}.issubset(development_ids):
        raise ValueError("exact_k_curve contains an MDP outside the development split")
    if {row["mdp_id"] for row in n_emp_rows} != development_ids:
        raise ValueError(f"N_emp handoff does not cover exactly the {DEVELOPMENT_COUNT} development MDPs")

    # OEH/calibration inputs are data-only. Exact and N_emp columns must remain
    # labels used after feature construction, never estimator features.
    assert_no_oracle_leakage(
        ["observed_reachable_sa_fraction", "normalized_visitation_entropy"],
        context="Workstream 1 data-only feature audit",
    )
    summary.update({
        "exact_k_curve_rows": len(exact_rows),
        "exact_k_curve_pairs": len({(row["mdp_id"], row["policy_family"]) for row in exact_rows}),
        "exact_k_dagger_distribution": dict(Counter(row["k_dagger_exact"] for row in exact_rows if row["depth"] == "0")),
        "n_emp_rows": len(n_emp_rows),
        "n_emp_censored": sum(row["n_emp_censored"].lower() == "true" for row in n_emp_rows),
        "exact_solution_rows": len(solution_rows),
        "leakage_check": "pass",
    })
    summary["artifact_sha256"] = {
        path.name: _sha256(path)
        for path in (benchmark_path, oracle_path, exact_curve_path, solution_path)
    }
    (output_dir / "benchmark_oracle_validation_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-k-curve-source", type=Path, required=True)
    parser.add_argument("--n-emp-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, exact_k_curve_source=args.exact_k_curve_source,
                         n_emp_source=args.n_emp_source), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
