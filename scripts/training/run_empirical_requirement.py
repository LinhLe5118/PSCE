"""Run the development-only Offline RL and data-generation workstream.

The registered CFQI/IQL learners produce the two Week 2 contract tables used
to resolve ``N_emp``. Tabular behavior cloning is evaluated alongside them as
a sanity check, but is deliberately kept out of the registered ``N_emp``
artifact. Exact MDP quantities are used only after fitting, for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from empirical_requirement import (  # noqa: E402
    behavior_cloning_tabular,
    cfqi_tabular,
    empirical_requirement,
    iql_tabular,
    _model,
    _greedy_policy,
)
from exact_solver import evaluate_policy  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from evaluation.io import verified_dataset_path, reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from evaluation.validation.schema import validate_artifact  # noqa: E402
from evaluation.training_protocol import validate_training_manifest, verify_training_buffers  # noqa: E402


RL_RUN_FIELDS = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "training_size_multiplier",
    "training_seed",
    "normalized_return",
    "success",
    "run_status",
)
BEHAVIOR_METADATA_FIELDS = (
    "mdp_id",
    "policy_family",
    "policy_parameters",
    "observed_reachable_sa_fraction",
    "observed_sa_fraction_all",
    "observed_policy_support_fraction",
    "fraction_policy_support_at_least_n_min",
    "minimum_policy_support_count",
    "normalized_visitation_entropy",
    "num_transitions",
    "num_trajectories_started",
)
N_EMP_FIELDS = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "n_emp_multiplier",
    "n_emp_transitions",
    "n_emp_censored",
    "success_counts",
    "success_rates",
)
CURVE_FIELDS = (
    "algorithm",
    "training_size_multiplier",
    "valid_runs",
    "invalid_runs",
    "mean_normalized_return",
    "median_normalized_return",
    "success_rate",
)
SUCCESS_INTERVAL_FIELDS = (
    "mdp_id",
    "policy_family",
    "algorithm",
    "training_size_multiplier",
    "success_count",
    "total_runs",
    "success_rate",
    "wilson_lower_95",
    "wilson_upper_95",
)
SOURCE_PROVENANCE_FILES = (
    'code/tabular_acceleration.py',
    "scripts/training/run_empirical_requirement.py",
    "evaluation/training_protocol.py",
    "evaluation/io.py",
    "config/protocol_config.py",
    "environments/benchmark_environments.py",
    "environments/tabular_environments.py",
    "code/adaptive_pilot.py",
    "code/exact_solver.py",
    "code/empirical_requirement.py",
    "protocol_config.json",
    "requirements.txt",
)


@dataclass(frozen=True)
class OfflineRLArtifacts:
    summary: dict[str, Any]
    rl_run_results: tuple[dict[str, Any], ...]
    behavior_policy_metadata: tuple[dict[str, Any], ...]
    empirical_sample_complexity: tuple[dict[str, Any], ...]
    bc_sanity_results: tuple[dict[str, Any], ...]
    learning_curve_summary: tuple[dict[str, Any], ...]


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a binomial success rate."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0.")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half_width, center + half_width


def _success_rate_intervals(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["mdp_id"]),
            str(row["policy_family"]),
            str(row["algorithm"]),
            int(row["training_size_multiplier"]),
        )
        grouped[key].append(row)
    intervals: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        successes = sum(bool(row["success"]) for row in group)
        lower, upper = _wilson_interval(successes, len(group))
        intervals.append(
            {
                "mdp_id": key[0],
                "policy_family": key[1],
                "algorithm": key[2],
                "training_size_multiplier": key[3],
                "success_count": successes,
                "total_runs": len(group),
                "success_rate": successes / len(group),
                "wilson_lower_95": lower,
                "wilson_upper_95": upper,
            }
        )
    return intervals


def _source_provenance() -> dict[str, Any]:
    source_hashes = {
        name: sha256_file(ROOT / name) for name in SOURCE_PROVENANCE_FILES
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *SOURCE_PROVENANCE_FILES],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_sources_dirty: bool | None = bool(status)
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, tracked_sources_dirty = None, None
    return {
        "git_commit": commit,
        "tracked_sources_dirty": tracked_sources_dirty,
        "source_sha256": source_hashes,
    }


def _training_seed_index(record: dict[str, Any]) -> int:
    collection = PROTOCOL_CONFIG["collection"]
    modulus = 2**32 - 1
    expected = {
        int(
            (
                int(record["mdp_seed"])
                + int(collection["training_seed_offset"])
                + index
            )
            % modulus
        ): int(index)
        for index in collection["training_seed_indices"]
    }
    stream_seed = int(record["stream_seed"])
    if stream_seed not in expected:
        raise ValueError(
            f"Unregistered training stream seed {stream_seed} "
            f"for {record['instance_id']}."
        )
    return expected[stream_seed]


def _load_run(
    dataset_dir: Path,
    record: dict[str, Any],
    cache: dict[str, dict[str, np.ndarray]],
) -> Sequence[Any]:
    file_name = str(record["file"])
    if file_name not in cache:
        path = verified_dataset_path(dataset_dir, record)
        with np.load(path, allow_pickle=False) as loaded:
            cache[file_name] = {name: loaded[name] for name in loaded.files}
    size = int(record["size"])
    arrays = {name: values[:size] for name, values in cache[file_name].items()}
    return transitions_from_arrays(arrays)


def _score_policy(
    mdp: Any,
    policy: np.ndarray,
    *,
    random_return: float,
    dynamic_range: float,
) -> tuple[float, str]:
    if policy.shape != (mdp.num_states, mdp.num_actions) or not np.all(
        np.isfinite(policy)
    ):
        return math.nan, "invalid"
    if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-10) or np.any(
        policy < 0.0
    ):
        return math.nan, "invalid"
    normalized_return = (
        evaluate_policy(mdp, policy).expected_return - random_return
    ) / dynamic_range
    if not math.isfinite(normalized_return):
        return math.nan, "invalid"
    return float(normalized_return), "observed"


def _curve_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["algorithm"]), int(row["training_size_multiplier"]))
        ].append(row)
    result: list[dict[str, Any]] = []
    for (algorithm, multiplier), group in sorted(grouped.items()):
        valid = [
            float(row["normalized_return"])
            for row in group
            if row["run_status"] == "observed"
        ]
        result.append(
            {
                "algorithm": algorithm,
                "training_size_multiplier": multiplier,
                "valid_runs": len(valid),
                "invalid_runs": len(group) - len(valid),
                "mean_normalized_return": (
                    float(np.mean(valid)) if valid else math.nan
                ),
                "median_normalized_return": (
                    float(np.median(valid)) if valid else math.nan
                ),
                "success_rate": sum(bool(row["success"]) for row in group)
                / len(group),
            }
        )
    return result


def _monotonic_diagnostics(
    curves: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for algorithm in sorted({str(row["algorithm"]) for row in curves}):
        rows = sorted(
            (row for row in curves if row["algorithm"] == algorithm),
            key=lambda row: int(row["training_size_multiplier"]),
        )
        medians = [float(row["median_normalized_return"]) for row in rows]
        success_rates = [float(row["success_rate"]) for row in rows]
        diagnostics[algorithm] = {
            "multipliers": [
                int(row["training_size_multiplier"]) for row in rows
            ],
            "median_normalized_returns": medians,
            "success_rates": success_rates,
            "median_curve_non_decreasing": all(
                b + 1e-12 >= a for a, b in zip(medians, medians[1:])
            ),
            "success_curve_non_decreasing": all(
                b + 1e-12 >= a
                for a, b in zip(success_rates, success_rates[1:])
            ),
        }
    return diagnostics


def prepare_training_plan(dataset_dir: Path) -> dict[str, Any]:
    """Validate the full bundle and record the run contract without training."""
    import scipy

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = validate_training_manifest(manifest)
    buffers = verify_training_buffers(dataset_dir, manifest)
    return {
        "status": "preflight_passed_not_trained",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "Linh",
        "scope": "development reproduction; evaluation sealed",
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "verified_buffers": buffers,
        "expected_counts": counts,
        "protocol": PROTOCOL_CONFIG,
        "provenance": _source_provenance(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(), "device": "CPU"},
        "stop_rule": "Run the complete fixed grid, regardless of feasibility or early success; exceptions abort the run.",
        "retry_rule": "No checkpoint resume. Preserve failed output/log; rerun unchanged config/data in a new directory. Never pool reruns as seeds.",
        "convergence_rule": "At most 100 updates or update norm <= 1e-10; reaching the cap is not itself an invalid policy.",
        "label_rule": "N_emp is the first tested prefix for which that prefix and every larger tested prefix achieve normalized return >= 0.95 in at least 8/10 seeds; otherwise right-censored at the largest prefix. BC is sanity-only.",
    }


def run_with_artifacts(dataset_dir: Path) -> OfflineRLArtifacts:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["run_mode"] != "review":
        raise ValueError(
            "Offline RL development validation requires a review bundle."
        )
    records = [
        item for item in manifest["datasets"] if item["stream"] == "training"
    ]
    if any(item["benchmark_split"] != "development" for item in records):
        raise ValueError(
            "Offline RL development validation cannot inspect evaluation records."
        )

    learning = PROTOCOL_CONFIG["learning"]
    collection = PROTOCOL_CONFIG["collection"]
    multipliers = tuple(int(value) for value in collection["size_multipliers"])
    training_seeds = tuple(
        int(value) for value in collection["training_seed_indices"]
    )
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[
            (str(record["instance_id"]), str(record["policy_family"]))
        ].append(record)
    fresh_holdout = manifest.get("review_cohort", {}).get("role") == (
        "oeh_v3_post_freeze_holdout"
    )
    expected_instance_count = (
        len(manifest.get("instances", ())) if fresh_holdout
        else int(PROTOCOL_CONFIG["benchmark"]["development_count"])
    )
    expected_groups = expected_instance_count * len(
        PROTOCOL_CONFIG['behavior_policies']['families']
    )
    if len(by_pair) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} development instance-policy groups, found {len(by_pair)}."
        )

    instances = {
        str(item["instance_id"]): item for item in manifest["instances"]
    }
    rl_rows: list[dict[str, Any]] = []
    bc_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    n_emp_rows: list[dict[str, Any]] = []

    for (instance_id, family), group in sorted(by_pair.items()):
        instance = instances[instance_id]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        random_return = float(instance["random_return"])
        dynamic_range = float(instance["dynamic_range"])
        if dynamic_range <= 0.0:
            raise ValueError(f"Non-positive dynamic range for {instance_id}.")

        indexed: dict[tuple[int, int], dict[str, Any]] = {}
        for record in group:
            multiplier = record.get("size_multiplier")
            if multiplier is None:
                continue
            key = (int(multiplier), _training_seed_index(record))
            if key in indexed:
                raise ValueError(
                    f"Duplicate training prefix for {instance_id}/{family}/{key}."
                )
            indexed[key] = record
        expected = {
            (multiplier, seed)
            for multiplier in multipliers
            for seed in training_seeds
        }
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            extra = sorted(set(indexed) - expected)
            raise ValueError(
                f"{instance_id}/{family} training grid mismatch; "
                f"missing={missing}, extra={extra}."
            )

        largest_record = indexed[(multipliers[-1], training_seeds[0])]
        coverage = largest_record["coverage"]
        behavior_rows.append(
            {
                "mdp_id": instance_id,
                "policy_family": family,
                "policy_parameters": json.dumps(
                    largest_record["policy_parameters"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **{
                    name: coverage[name]
                    for name in BEHAVIOR_METADATA_FIELDS[3:]
                },
            }
        )

        learners: dict[
            str, Callable[[Sequence[Any]], np.ndarray]
        ] = {
            "cfqi_tabular": lambda transitions: cfqi_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
                gamma=mdp.gamma,
                iterations=int(learning["iterations"]),
                tolerance=float(learning["tolerance"]),
            ),
            "iql_tabular": lambda transitions: iql_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
                gamma=mdp.gamma,
                expectile=float(learning["iql_expectile"]),
                inverse_temperature=float(
                    learning["iql_inverse_temperature"]
                ),
                iterations=int(learning["iterations"]),
                tolerance=float(learning["tolerance"]),
            ),
            "behavior_cloning_tabular": lambda transitions: behavior_cloning_tabular(
                transitions,
                num_states=mdp.num_states,
                num_actions=mdp.num_actions,
            ),
        }
        qualities: dict[str, list[list[float]]] = {
            name: [[] for _ in multipliers] for name in learners
        }
        cache: dict[str, dict[str, np.ndarray]] = {}
        for multiplier_index, multiplier in enumerate(multipliers):
            accelerated_policies = {}
            accelerated_iql = {}
            backend = os.environ.get('DAORL_CFQI_BACKEND', 'reference')
            iql_backend = os.environ.get('DAORL_IQL_BACKEND', 'reference')
            batch_size = int(os.environ.get('DAORL_CFQI_BATCH_SIZE', '8'))
            if backend not in {'reference', 'numpy', 'cuda','hybrid'} or iql_backend not in {'reference','numpy','cuda','hybrid'} or batch_size <= 0:
                raise ValueError('Invalid CFQI backend/batch size')
            if backend != 'reference' or iql_backend != 'reference':
                from tabular_acceleration import cfqi_models, iql_models
                for start in range(0, len(training_seeds), batch_size):
                    seeds = training_seeds[start:start+batch_size]
                    models = [_model(_load_run(dataset_dir, indexed[(multiplier, seed)], cache),
                        mdp.num_states, mdp.num_actions) for seed in seeds]
                    if backend != 'reference':
                        qs, _, _ = cfqi_models(models, mdp.gamma,
                            iterations=int(learning['iterations']),
                            tolerance=float(learning['tolerance']), backend=backend)
                        accelerated_policies.update({seed: _greedy_policy(q, model[2], model[3])
                            for seed, q, model in zip(seeds, qs, models)})
                    if iql_backend != 'reference':
                        policies, _, _ = iql_models(models, mdp.gamma,
                            iterations=int(learning['iterations']),tolerance=float(learning['tolerance']),
                            expectile=float(learning['iql_expectile']),
                            inverse_temperature=float(learning['iql_inverse_temperature']),backend=iql_backend)
                        accelerated_iql.update(dict(zip(seeds,policies)))
            for training_seed in training_seeds:
                transitions = _load_run(
                    dataset_dir,
                    indexed[(multiplier, training_seed)],
                    cache,
                )
                for algorithm, learner in learners.items():
                    if algorithm == 'cfqi_tabular' and accelerated_policies:
                        policy = accelerated_policies[training_seed]
                    elif algorithm == 'iql_tabular' and accelerated_iql:
                        policy = accelerated_iql[training_seed]
                    else:
                        policy = learner(transitions)
                    normalized_return, status = _score_policy(
                        mdp,
                        policy,
                        random_return=random_return,
                        dynamic_range=dynamic_range,
                    )
                    success = (
                        status == "observed"
                        and normalized_return
                        >= float(learning["success_normalized_return"])
                    )
                    row = {
                        "mdp_id": instance_id,
                        "policy_family": family,
                        "algorithm": algorithm,
                        "training_size_multiplier": multiplier,
                        "training_seed": training_seed,
                        "normalized_return": normalized_return,
                        "success": success,
                        "run_status": status,
                    }
                    qualities[algorithm][multiplier_index].append(
                        normalized_return
                    )
                    if algorithm == "behavior_cloning_tabular":
                        bc_rows.append(row)
                    else:
                        rl_rows.append(row)

        state_action_count = int(mdp.num_states * mdp.num_actions)
        for algorithm in learning["algorithms"]:
            prefix_scores = qualities[str(algorithm)]
            requirement = empirical_requirement(
                [
                    (multiplier * state_action_count, tuple(scores))
                    for multiplier, scores in zip(
                        multipliers, prefix_scores
                    )
                ],
                learner=lambda score: float(score),
                evaluate=lambda score: float(score),
                success_threshold=float(
                    learning["success_normalized_return"]
                ),
                required_successes=int(learning["success_runs_required"]),
            )
            multiplier_requirement = (
                None
                if requirement.requirement is None
                else requirement.requirement // state_action_count
            )
            n_emp_rows.append(
                {
                    "mdp_id": instance_id,
                    "policy_family": family,
                    "algorithm": str(algorithm),
                    "n_emp_multiplier": multiplier_requirement,
                    "n_emp_transitions": requirement.requirement,
                    "n_emp_censored": requirement.right_censored,
                    "success_counts": json.dumps(
                        list(requirement.success_counts),
                        separators=(",", ":"),
                    ),
                    "success_rates": json.dumps(
                        list(requirement.success_rates),
                        separators=(",", ":"),
                    ),
                }
            )

    curves = _curve_summary([*rl_rows, *bc_rows])
    summary: dict[str, Any] = {
        "scope": "development-only independent training streams",
        "numerical_execution": {
            "cfqi_backend": os.environ.get('DAORL_CFQI_BACKEND', 'reference'),
            "cfqi_batch_size": int(os.environ.get('DAORL_CFQI_BATCH_SIZE', '8')),
            "iql_backend": os.environ.get('DAORL_IQL_BACKEND','reference'), "policy_scoring": "reference_cpu",
            "precision": "float64", "gpu_acceleration_validated_for_scientific_use": False,
        },
        "protocol_status_at_run": manifest["protocol_status"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "registered_algorithms": list(learning["algorithms"]),
        "sanity_algorithms": ["behavior_cloning_tabular"],
        "instance_policy_groups": len(by_pair),
        "training_sizes": list(multipliers),
        "training_seeds": list(training_seeds),
        "registered_run_count": len(rl_rows),
        "bc_sanity_run_count": len(bc_rows),
        "invalid_registered_runs": sum(
            row["run_status"] == "invalid" for row in rl_rows
        ),
        "invalid_bc_runs": sum(
            row["run_status"] == "invalid" for row in bc_rows
        ),
        "learning_curve_diagnostics": _monotonic_diagnostics(curves),
    }
    for algorithm in learning["algorithms"]:
        values = [
            row for row in n_emp_rows if row["algorithm"] == algorithm
        ]
        finite = [
            int(row["n_emp_transitions"])
            for row in values
            if row["n_emp_transitions"] is not None
        ]
        summary[str(algorithm)] = {
            "count": len(values),
            "right_censored": sum(
                bool(row["n_emp_censored"]) for row in values
            ),
            "finite_count": len(finite),
            "n_emp_min": min(finite) if finite else None,
            "n_emp_median": (
                float(np.median(finite)) if finite else None
            ),
            "n_emp_max": max(finite) if finite else None,
            "per_family_censored": dict(
                Counter(
                    row["policy_family"]
                    for row in values
                    if row["n_emp_censored"]
                )
            ),
        }

    return OfflineRLArtifacts(
        summary=summary,
        rl_run_results=tuple(rl_rows),
        behavior_policy_metadata=tuple(behavior_rows),
        empirical_sample_complexity=tuple(n_emp_rows),
        bc_sanity_results=tuple(bc_rows),
        learning_curve_summary=tuple(curves),
    )


def run(dataset_dir: Path) -> dict[str, Any]:
    """Compatibility wrapper returning the JSON summary used by Week 1."""
    return run_with_artifacts(dataset_dir).summary


def write_artifacts(
    output_dir: Path, artifacts: OfflineRLArtifacts
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Artifact output directory is not empty: {output_dir}. "
            "Choose a new directory to avoid mixing or overwriting runs."
        )
    provenance = _source_provenance()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "rl_run_results.csv",
        artifacts.rl_run_results,
        RL_RUN_FIELDS,
    )
    write_csv_rows(
        output_dir / "behavior_policy_metadata.csv",
        artifacts.behavior_policy_metadata,
        BEHAVIOR_METADATA_FIELDS,
    )
    write_csv_rows(
        output_dir / "empirical_sample_complexity.csv",
        artifacts.empirical_sample_complexity,
        N_EMP_FIELDS,
    )
    write_csv_rows(
        output_dir / "bc_sanity_results.csv",
        artifacts.bc_sanity_results,
        RL_RUN_FIELDS,
    )
    write_csv_rows(
        output_dir / "learning_curve_summary.csv",
        artifacts.learning_curve_summary,
        CURVE_FIELDS,
    )
    interval_rows = _success_rate_intervals(artifacts.rl_run_results)
    write_csv_rows(
        output_dir / "success_rate_intervals.csv",
        interval_rows,
        SUCCESS_INTERVAL_FIELDS,
    )
    write_json(output_dir / "offline_rl_summary.json", artifacts.summary)

    # Fail before handoff if a contracted table is not consumable by the
    # shared Week 2 evaluator. BC/curve files are explicitly diagnostic.
    validate_artifact(
        output_dir / "rl_run_results.csv", "rl_run_results"
    )
    validate_artifact(
        output_dir / "behavior_policy_metadata.csv",
        "behavior_policy_metadata",
    )
    validate_artifact(
        output_dir / "empirical_sample_complexity.csv",
        "empirical_sample_complexity",
    )

    output_rows = {
        "rl_run_results.csv": len(artifacts.rl_run_results),
        "behavior_policy_metadata.csv": len(artifacts.behavior_policy_metadata),
        "empirical_sample_complexity.csv": len(artifacts.empirical_sample_complexity),
        "bc_sanity_results.csv": len(artifacts.bc_sanity_results),
        "learning_curve_summary.csv": len(artifacts.learning_curve_summary),
        "success_rate_intervals.csv": len(interval_rows),
        "offline_rl_summary.json": None,
    }
    write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "offline-rl-artifacts-1",
            "scope": artifacts.summary["scope"],
            "protocol_status": artifacts.summary["protocol_status_at_run"],
            "input_manifest_sha256": artifacts.summary["dataset_manifest_sha256"],
            "provenance": provenance,
            "outputs": {
                name: {
                    "rows": rows,
                    "sha256": sha256_file(output_dir / name),
                }
                for name, rows in output_rows.items()
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfqi-backend', choices=['reference','numpy','cuda','hybrid'], default='reference')
    parser.add_argument('--iql-backend', choices=['reference','numpy','cuda','hybrid'], default='reference')
    parser.add_argument('--cfqi-batch-size', type=int, default=8)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path,
                        help="Validate the full bundle and write a run plan without fitting.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional compatibility JSON summary path.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory for Week 2 CSV artifacts and diagnostics.",
    )
    args = parser.parse_args()
    if args.cfqi_batch_size <= 0:
        parser.error('--cfqi-batch-size must be positive')
    os.environ['DAORL_CFQI_BACKEND'] = args.cfqi_backend
    os.environ['DAORL_IQL_BACKEND'] = args.iql_backend
    os.environ['DAORL_CFQI_BATCH_SIZE'] = str(args.cfqi_batch_size)
    if args.preflight_output is not None:
        if args.output is not None or args.artifact_dir is not None:
            parser.error("--preflight-output cannot be combined with training outputs.")
        if args.preflight_output.exists():
            raise FileExistsError(f"Output already exists: {args.preflight_output}")
        plan = prepare_training_plan(args.dataset_dir)
        write_json(args.preflight_output, plan)
        print(json.dumps(plan, indent=2))
        return
    if args.output is None and args.artifact_dir is None:
        parser.error("At least one of --output or --artifact-dir is required.")

    started = time.perf_counter()
    plan = prepare_training_plan(args.dataset_dir)
    if args.artifact_dir is not None:
        reserve_output_dir(args.artifact_dir)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    artifacts = run_with_artifacts(args.dataset_dir)
    artifacts.summary["wall_seconds_including_preflight"] = time.perf_counter() - started
    artifacts.summary["training_plan"] = plan
    if args.output is not None:
        write_json(args.output, artifacts.summary)
    if args.artifact_dir is not None:
        write_artifacts(args.artifact_dir, artifacts)
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
