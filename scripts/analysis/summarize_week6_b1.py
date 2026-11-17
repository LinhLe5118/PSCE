"""Summarize the Week-6 B1 stochastic dataset evidence without new data.

Only committed development/probe artifacts and the registered configuration are
read.  The sealed evaluation manifest, buffers, and outcomes are out of scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/features.csv"
FEATURE_REPORT = ROOT / "artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/report.json"
PREFIXES = ROOT / "artifacts/research/oeh_v3_post_freeze_holdout_2026-09-20/summary.csv"
KERNELS = ROOT / "artifacts/week6/stochastic_benchmark_2026-09-21/development60.json"
KERNEL_CHECKS = ROOT / "artifacts/week6/stochastic_benchmark_2026-09-21/checks.json"
POLICY_METADATA = ROOT / "artifacts/research/final_local_methods_development_v2_2026-09-20/behavior_policy_metadata.csv"
POLICY_MANIFEST = ROOT / "artifacts/research/final_local_methods_development_v2_2026-09-20/artifact_manifest.json"
FACTORIAL_EFFECTS = ROOT / "artifacts/research/controlled_stochastic_factorial_v2_development_2026-09-20/matched_main_effects.csv"
FACTORIAL_REPORT = ROOT / "artifacts/research/controlled_stochastic_factorial_v2_development_2026-09-20/report.json"
CONFIG = ROOT / "protocol_config.json"

FAMILIES = ("uniform_garnet", "forward_layered", "lock_structured", "bottleneck")
POLICIES = ("q_star_softmax", "perturbed_reward", "structured_random", "two_policy_mixture", "shifted_dynamics")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_matches(path: Path, expected: str) -> str | None:
    """Return the exact/line-ending verification mode for tracked text files."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return "byte_exact"
    lf = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf).hexdigest() == expected:
        return "lf_equivalent"
    crlf = lf.replace(b"\n", b"\r\n")
    if hashlib.sha256(crlf).hexdigest() == expected:
        return "crlf_equivalent"
    return None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "Cannot summarize an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(rows: list[dict[str, str]], field: str) -> dict[str, float | int]:
    values = [float(row[field]) for row in rows]
    return {
        "n": len(values), "min": min(values), "q25": quantile(values, .25),
        "median": statistics.median(values), "mean": statistics.fmean(values),
        "q75": quantile(values, .75), "max": max(values),
    }


def family_from_id(mdp_id: str) -> str:
    for family in FAMILIES:
        if f"_{family}_" in mdp_id:
            return family
    raise ValueError(f"Unknown structural family in {mdp_id}")


def grouped_feature_summary(rows: list[dict[str, str]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        name: {
            "groups": len(items),
            "observed_sa_fraction": numeric_summary(items, "observed_sa_fraction"),
            "normalized_visitation_entropy": numeric_summary(items, "normalized_visitation_entropy"),
            "trajectory_return_mean": numeric_summary(items, "trajectory_return_mean"),
            "trajectory_return_std": numeric_summary(items, "trajectory_return_std"),
            "terminal_trajectory_fraction": numeric_summary(items, "terminal_trajectory_fraction"),
            "weighted_transition_entropy": numeric_summary(items, "weighted_transition_entropy"),
            "empirical_stochastic_cell_fraction": numeric_summary(items, "empirical_stochastic_cell_fraction"),
        }
        for name, items in sorted(groups.items())
    }


def build_report() -> dict[str, Any]:
    features = read_csv(FEATURES)
    feature_report = read_json(FEATURE_REPORT)
    prefixes = read_csv(PREFIXES)
    kernels = read_json(KERNELS)
    kernel_checks = read_json(KERNEL_CHECKS)
    policy_rows = read_csv(POLICY_METADATA)
    policy_manifest = read_json(POLICY_MANIFEST)
    effects = read_csv(FACTORIAL_EFFECTS)
    factorial_report = read_json(FACTORIAL_REPORT)
    config = read_json(CONFIG)

    feature_verification = digest_matches(FEATURES, feature_report["artifact_sha256"]["features.csv"])
    kernel_verification = digest_matches(KERNELS, kernel_checks["files_sha256"]["development60.json"])
    policy_verification = digest_matches(
        POLICY_METADATA, policy_manifest["outputs"]["behavior_policy_metadata.csv"]["sha256"])
    require(feature_verification is not None,
            "Feature artifact digest differs from its report")
    require(kernel_verification is not None,
            "Kernel audit digest differs from checks.json")
    require(policy_verification is not None,
            "Policy metadata digest differs from its manifest")
    require(len(features) == 300 and len({row["mdp_id"] for row in features}) == 60,
            "Expected the immutable 60-MDP/300-group development evidence")
    require(len(policy_rows) == 75, "Expected 75 registered development policy rows")
    require(len(prefixes) == 900, "Expected 300 groups at three pilot prefixes")
    require(kernels["evaluation_manifest_or_buffers_read"] is False, "Kernel audit crossed the evaluation boundary")
    require(feature_report["sealed_evaluation_accessed"] is False, "Feature artifact crossed the evaluation boundary")
    require(factorial_report["sealed_evaluation_accessed"] is False, "Factorial crossed the evaluation boundary")

    for row in features:
        row["structural_family"] = family_from_id(row["mdp_id"])
    feature_by_family = grouped_feature_summary(features, "structural_family")
    feature_by_policy = grouped_feature_summary(features, "policy_family")

    kernel_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kernels["rows"]:
        kernel_by_family[row["family"]].append(row)
    dynamics = {}
    for family, rows in sorted(kernel_by_family.items()):
        eligible = sum(int(row["eligible_state_action_rows"]) for row in rows)
        stochastic = sum(int(row["stochastic_state_action_rows"]) for row in rows)
        dynamics[family] = {
            "base_mdps": len(rows), "panel": rows[0]["panel"],
            "mdps_with_stochastic_dynamics": sum(bool(row["has_stochastic_dynamics"]) for row in rows),
            "eligible_state_action_rows": eligible, "stochastic_state_action_rows": stochastic,
            "stochastic_row_fraction": stochastic / eligible,
            "empirical_pilot_metrics": {
                "weighted_transition_entropy": feature_by_family[family]["weighted_transition_entropy"],
                "empirical_stochastic_cell_fraction": feature_by_family[family]["empirical_stochastic_cell_fraction"],
            },
        }

    thresholds = [float(value) for value in config["coverage"]["thresholds"]]
    coverage_counts = Counter()
    for row in policy_rows:
        value = float(row["observed_reachable_sa_fraction"])
        label = "low" if value < thresholds[0] else "medium" if value < thresholds[1] else "high" if value < thresholds[2] else "very_high"
        coverage_counts[label] += 1

    feature_lookup = {(row["mdp_id"], row["policy_family"]): row for row in features}
    prefix_counts = Counter(int(row["pilot_multiplier"]) for row in prefixes)
    prefix_formula_matches = 0
    for row in prefixes:
        feature = feature_lookup[(row["mdp_id"], row["policy_family"])]
        expected = int(row["pilot_multiplier"]) * int(feature["num_states"]) * int(feature["num_actions"])
        prefix_formula_matches += int(int(row["num_transitions"]) == expected)

    selected_effects = {}
    for row in effects:
        key = (row["factor"], row["outcome"])
        if key in {
            ("slip", "weighted_transition_entropy"), ("slip", "cfqi_normalized_return"),
            ("coverage", "observed_sa_fraction"), ("coverage", "cfqi_normalized_return"),
            ("budget", "observed_sa_fraction"), ("budget", "cfqi_normalized_return"),
            ("budget", "oeh_resolved"),
        }:
            selected_effects[f"{key[0]}::{key[1]}"] = {
                "contrast": row["contrast"], "matched_pairs": int(row["matched_pairs"]),
                "mean_difference": float(row["mean_difference"]),
            }

    training_grid = [int(value) for value in config["collection"]["size_multipliers"]]
    training_seeds = [int(value) for value in config["collection"]["training_seed_indices"]]
    evaluation_counts = config["benchmark"]["evaluation_family_counts"]
    primary_mdps = sum(int(evaluation_counts[name]) for name in FAMILIES if name != "lock_structured")
    controls = int(evaluation_counts["lock_structured"])
    core_learners = 3  # CFQI, IQL and tabular BC in T1 v1.1.

    sources = [FEATURES, FEATURE_REPORT, PREFIXES, KERNELS, KERNEL_CHECKS,
               POLICY_METADATA, POLICY_MANIFEST, FACTORIAL_EFFECTS, FACTORIAL_REPORT, CONFIG]
    return {
        "schema_version": "week6-b1-stochastic-dataset-analysis-v1",
        "status": "PASS_WITH_SCOPED_LIMITATIONS",
        "owner": "Việt Trần",
        "scope": {
            "development_only": True, "new_development_dataset_created": False,
            "evaluation_manifest_buffers_or_outcomes_read": False,
            "unit_of_inference": "base MDP, not behavior/group/seed",
        },
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sources},
        "recorded_digest_verification": {
            str(FEATURES.relative_to(ROOT)).replace("\\", "/"): feature_verification,
            str(KERNELS.relative_to(ROOT)).replace("\\", "/"): kernel_verification,
            str(POLICY_METADATA.relative_to(ROOT)).replace("\\", "/"): policy_verification,
        },
        "cohort": {
            "development_base_mdps": 60, "development_behavior_groups": 300,
            "family_group_counts": dict(Counter(row["structural_family"] for row in features)),
            "kernel_panels": kernels["development_counts"],
        },
        "transition_stochasticity": {
            "definition": "At least two successors with probability >1e-12 in a reachable nonterminal state-action row; family assignment is verified against reconstructed development kernels.",
            "by_family": dynamics,
        },
        "behavior_stochasticity": {
            "definition": "A separate randomized policy-generation axis; policy family is design metadata, while visitation entropy is an observed joint consequence of policy, dynamics and finite sampling.",
            "registered_families": list(POLICIES),
            "parameter_grids": {
                "q_star_softmax_temperature": config["behavior_policies"]["temperature_grid"],
                "epsilon": config["behavior_policies"]["epsilon_grid"],
                "two_policy_mixture_weight": config["behavior_policies"]["mixture_weight_grid"],
                "shifted_dynamics": config["behavior_policies"]["dynamics_shift_grid"],
            },
            "observed_by_policy": feature_by_policy,
        },
        "coverage": {
            "registered_metric": "observed_reachable_sa_fraction",
            "registered_15_mdp_development_rows": len(policy_rows),
            "registered_thresholds": thresholds,
            "registered_stratum_counts": {name: coverage_counts[name] for name in config["coverage"]["labels"]},
            "registered_metric_summary": numeric_summary(policy_rows, "observed_reachable_sa_fraction"),
            "holdout_60_mdp_all_sa_metric_summary": numeric_summary(features, "observed_sa_fraction"),
            "warning": "observed_sa_fraction uses all SxA cells and is not interchangeable with the registered reachable-SxA metric.",
        },
        "trajectory_quality": {
            "observed_by_policy": {
                name: {field: values[field] for field in (
                    "trajectory_return_mean", "trajectory_return_std", "terminal_trajectory_fraction")}
                for name, values in feature_by_policy.items()
            },
            "interpretation": "Return moments and terminal fraction describe collected trajectories; they are not pure behavior-quality effects because reward and dynamics also vary.",
        },
        "data_budget": {
            "pilot_multipliers": config["collection"]["pilot_size_multipliers"],
            "pilot_prefix_rows": {str(key): prefix_counts[key] for key in sorted(prefix_counts)},
            "pilot_prefix_formula_matches": prefix_formula_matches,
            "pilot_prefix_rows_checked": len(prefixes),
            "x10_transition_count_summary": numeric_summary(features, "num_transitions"),
            "training_multipliers": training_grid, "training_seed_indices": training_seeds,
            "planned_core_fits": {
                "stochastic_primary": primary_mdps * 5 * len(training_grid) * len(training_seeds) * core_learners,
                "deterministic_control": controls * 5 * len(training_grid) * len(training_seeds) * core_learners,
                "all_120_mdps": (primary_mdps + controls) * 5 * len(training_grid) * len(training_seeds) * core_learners,
            },
            "warning": "Nested prefixes, policies and learner seeds increase measurements, not the number of independent base-MDP clusters.",
        },
        "controlled_factorial": {
            "rows": factorial_report["rows"], "matched_topology_reward_blocks": 6,
            "selected_matched_effects": selected_effects,
            "interpretation": "The matched probe separates slip, behavior-decoy probability and budget mechanically, but six blocks support descriptive mechanism evidence only.",
        },
        "inference_limits": [
            "Development artifacts are consumed method-development evidence, not independent confirmation.",
            "Observed transition entropy is count-biased at low support; a one-observation zero is not evidence of deterministic dynamics.",
            "Behavior policy family, environment dynamics, coverage, trajectory return and budget must not be collapsed into one difficulty variable.",
            "The fixed 84-MDP stochastic evaluation mixture supports claims only for the three registered tabular families and fixed learner portfolio/budgets.",
            "Controlled-factorial contrasts are descriptive because there are only six topology/reward blocks.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists():
        parser.error("Output exists; preserve previous evidence")
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
