"""Workstream 3 runner — OEH scientific validation and Week 2 artifacts.

Runs the registered data-only pipeline over a development bundle and emits both
halves of this workstream's deliverable:

* the scientific checks — depth-zero reduction, trajectory-bootstrap
  dispersion, the evidence curve, ``k†``/``N_hat_req``, the comparison against
  the exact reference and against the measured ``N_emp``, and the
  failure/abstention breakdown;
* the artifact tables in ``docs/week2/WEEK2_ARTIFACT_CONTRACT.md``, each
  validated against ``evaluation/validation/schema.py`` before it is written.

Ordering rule, inherited from ``scripts/validation/run_week1_validation.py``: every exact quantity
is read only after a data-only result already exists for that pilot.  Nothing
computed here selects a threshold, a seed, or a variant.

``exact_k_curve.csv`` is Workstream 1's contract artifact.  It is produced here
because the ``k†`` comparison needs it and no Workstream 1 run has emitted it
yet; the run report says so explicitly so ownership can be settled rather than
silently assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from environments.behavior_policies import behavior_policy_suite  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from evaluation.io import sha256_file, verified_dataset_path, write_csv_rows, write_json  # noqa: E402
from evaluation.validation.schema import validate_artifact  # noqa: E402
from exact_evidence_curve import exact_evidence_curve  # noqa: E402
from oeh_estimator import OEHConfig  # noqa: E402
from oeh_validation import (  # noqa: E402
    abstention_rows,
    bootstrap_dispersion,
    bootstrap_results_rows,
    check_depth_zero,
    compare_k_dagger,
    compare_with_empirical_requirement,
    evidence_curve_rows,
    exact_k_curve_rows,
    failure_analysis,
    oeh_results_rows,
    run_pilot,
    selected_subset_rows,
)
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402

CONTRACT_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "oeh_results": (
        "oeh_results.csv",
        (
            "mdp_id",
            "policy_family",
            "pilot_multiplier",
            "selected",
            "oeh_status",
            "abstain_reason",
            "k_dagger_oeh",
            "effective_horizon",
            "n_req_hat",
            "valid_bootstrap_fraction",
            "gap_occupancy_at_k_dagger",
        ),
    ),
    "abstention_results": (
        "abstention_results.csv",
        ("mdp_id", "policy_family", "pilot_multiplier", "oeh_status", "reason"),
    ),
    "exact_k_curve": (
        "exact_k_curve.csv",
        (
            "mdp_id",
            "policy_family",
            "depth",
            "delta_hat",
            "sigma_hat",
            "multiplier",
            "k_dagger_exact",
            "valid_sample_fraction",
        ),
    ),
}

UNVALIDATED_FILES: dict[str, tuple[str, ...]] = {
    "evidence_curves.csv": (
        "mdp_id",
        "policy_family",
        "pilot_multiplier",
        "depth",
        "multiplier",
        "loss_at_one",
        "gap_occupancy",
    ),
    "bootstrap_results.csv": (
        "mdp_id",
        "policy_family",
        "pilot_multiplier",
        "num_bootstrap",
        "valid_bootstrap_fraction",
        "bootstrap_seed",
    ),
    "k_dagger_estimates.csv": (
        "mdp_id",
        "policy_family",
        "pilot_multiplier",
        "k_dagger_oeh",
        "oeh_status",
    ),
    "n_required_estimates.csv": (
        "mdp_id",
        "policy_family",
        "pilot_multiplier",
        "n_req_hat",
        "oeh_status",
    ),
    "scientific_checks.csv": (
        "mdp_id",
        "policy_family",
        "selection_status",
        "n0_effective_multiplier",
        "oeh_status",
        "k_dagger_oeh",
        "k_dagger_exact",
        "n_req_hat",
        "effective_horizon",
        "depth_zero_status",
        "depth_zero_gap_error",
        "depth_zero_bellman_residual",
        "bootstrap_gap_sigma_ratio",
        "bootstrap_value_correlation",
        "observed_reachable_sa_fraction",
        "policy_normalized_return",
        "oracle_status",
        "oracle_error",
    ),
}


ORACLE_UNDEFINED_MESSAGE = (
    "Fewer than two support-valid oracle datasets; curve undefined."
)


def _oracle_error_status(error: Exception) -> str:
    """Classify the registered undefined-oracle outcome separately from bugs."""

    if isinstance(error, RuntimeError) and str(error) == ORACLE_UNDEFINED_MESSAGE:
        return "oracle_undefined"
    return "oracle_error"


def _cell(value: Any) -> Any:
    """Undefined diagnostics are empty cells, never NaN or a stand-in number."""

    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _support_mask(result: Any) -> np.ndarray:
    mask = np.zeros((result.num_states, result.num_actions), dtype=np.bool_)
    for state, actions in enumerate(result.supported_actions):
        mask[state, list(actions)] = True
    return mask


def _pilot_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        item
        for item in manifest["datasets"]
        if item["stream"] == "pilot"
        and item["budget_role"] == "adaptive_pilot_candidate"
        and item["size_multiplier"] == 10
    ]
    if any(item["benchmark_split"] != "development" for item in records):
        raise ValueError("This runner refuses a bundle containing evaluation records.")
    return records


def _read_n_emp(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    from evaluation.io import read_csv_rows

    return read_csv_rows(path)


def run(
    dataset_dir: Path,
    *,
    bootstrap_seed: int,
    oracle_seed: int,
    n_emp_path: Path | None,
    limit: int | None,
    skip_oracle: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_mode"] != "review":
        raise ValueError("Workstream 3 validation requires a development-only review run.")
    records = _pilot_records(manifest)
    if limit is not None:
        records = records[:limit]
    config = OEHConfig.load()

    tables: dict[str, list[dict[str, Any]]] = {
        "oeh_results": [],
        "evidence_curves": [],
        "bootstrap_results": [],
        "abstention_results": [],
        "exact_k_curve": [],
    }
    depth_zero_failures: list[dict[str, Any]] = []
    depth_zero_pass = 0
    depth_zero_not_applicable = 0
    per_pilot: list[dict[str, Any]] = []
    gap_errors: list[float] = []
    bellman_residuals: list[float] = []
    dispersion_ratios: list[float] = []
    dispersion_correlations: list[float] = []
    exact_by_key: dict[tuple[str, str], int] = {}
    oracle_exclusions: Counter[str] = Counter()
    oracle_failures: Counter[str] = Counter()
    oracle_exclusion_cases: list[dict[str, str]] = []
    oracle_failure_cases: list[dict[str, str]] = []
    pilot_sizes: dict[tuple[str, str], int] = {}
    coverage_by_key: dict[tuple[str, str], float] = {}
    selection_statuses: Counter[str] = Counter()
    started = time.time()

    for index, record in enumerate(records, start=1):
        mdp_id = str(record["instance_id"])
        family = str(record["policy_family"])
        key = (mdp_id, family)
        with np.load(verified_dataset_path(dataset_dir, record), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        pilot = run_pilot(
            arrays,
            np.asarray(record["initial_state_distribution"], dtype=np.float64),
            num_states=int(record["num_states"]),
            num_actions=int(record["num_actions"]),
            bootstrap_seed=bootstrap_seed,
        )
        result = pilot.selected_result
        selection_statuses[pilot.selection.status] += 1
        pilot_sizes[key] = int(pilot.selection.n0_effective_size)
        coverage_by_key[key] = float(
            record["coverage"]["observed_reachable_sa_fraction"]
        )

        rows = oeh_results_rows(mdp_id, family, pilot)
        tables["oeh_results"].extend(rows)
        tables["evidence_curves"].extend(evidence_curve_rows(mdp_id, family, pilot))
        tables["bootstrap_results"].extend(
            bootstrap_results_rows(
                mdp_id,
                family,
                pilot,
                bootstrap_seed=bootstrap_seed,
                num_bootstrap=config.num_bootstrap,
            )
        )
        tables["abstention_results"].extend(abstention_rows(rows))

        # Check 1 -- depth zero, on the prefix the selector actually chose.
        check = check_depth_zero(
            pilot,
            result,
            gamma=config.gamma,
            prefix_length=int(pilot.selection.n0_effective_size),
        )
        # An abstained result carries no depth-zero quantities by design, so it
        # is out of scope for this check rather than a failure of it.
        not_applicable = {"no depth-zero deltas", "no state had two supported actions"}
        if check.passed:
            depth_zero_status = "passed"
            depth_zero_pass += 1
            gap_errors.append(check.max_absolute_gap_error)
            bellman_residuals.append(check.max_bellman_residual)
        elif check.reason in not_applicable:
            depth_zero_status = "not_applicable"
            depth_zero_not_applicable += 1
        else:
            depth_zero_status = "failed"
            depth_zero_failures.append(
                {"mdp_id": mdp_id, "policy_family": family, **check.__dict__}
            )

        # Check 2 -- does bootstrapping the difference actually help?
        dispersion = None
        if not result.abstained:
            dispersion = bootstrap_dispersion(
                pilot,
                result,
                config=config,
                bootstrap_seed=bootstrap_seed,
                prefix_length=int(pilot.selection.n0_effective_size),
            )
            if np.isfinite(dispersion.median_ratio):
                dispersion_ratios.append(dispersion.median_ratio)
            if np.isfinite(dispersion.median_correlation):
                dispersion_correlations.append(dispersion.median_correlation)

        # Exact reference, read only now that a data-only result exists.
        oracle_status = "not_applicable" if result.abstained else "skipped"
        oracle_error = None
        if not skip_oracle and not result.abstained:
            try:
                spec = next(
                    item
                    for item in manifest["instances"]
                    if item["instance_id"] == mdp_id
                )
                mdp = make_benchmark_mdp(type("Spec", (), spec)())
                policy = next(
                    item
                    for item in behavior_policy_suite(
                        mdp,
                        seed=int(spec["seed"]) + 3_000_000,
                        variant_index=int(spec["ordinal"]),
                    )
                    if item.family == family
                )
                curve = exact_evidence_curve(
                    mdp,
                    policy.probabilities,
                    action_mask=_support_mask(result),
                    config=config,
                    rng=np.random.default_rng([oracle_seed, int(record["mdp_seed"])]),
                )
                tables["exact_k_curve"].extend(
                    exact_k_curve_rows(mdp_id, family, curve)
                )
                exact_by_key[key] = int(curve.k_dagger)
                oracle_status = "compared"
            except (RuntimeError, ValueError, StopIteration) as error:
                reason = f"{type(error).__name__}: {error}"
                oracle_status = _oracle_error_status(error)
                oracle_error = reason
                case = {
                    "mdp_id": mdp_id,
                    "policy_family": family,
                    "reason": reason,
                }
                if oracle_status == "oracle_undefined":
                    # Metric contract: this case is excluded and logged.  It is
                    # not an implementation or data-integrity failure.
                    oracle_exclusions[reason] += 1
                    oracle_exclusion_cases.append(case)
                else:
                    oracle_failures[reason] += 1
                    oracle_failure_cases.append(case)

        per_pilot.append(
            {
                "mdp_id": mdp_id,
                "policy_family": family,
                "selection_status": pilot.selection.status,
                "n0_effective_multiplier": pilot.selection.n0_effective_multiplier,
                "oeh_status": rows[-1]["oeh_status"],
                "k_dagger_oeh": _cell(result.k_dagger),
                "k_dagger_exact": _cell(exact_by_key.get(key)),
                "n_req_hat": _cell(result.n_req_hat),
                "effective_horizon": _cell(result.effective_horizon),
                "depth_zero_status": depth_zero_status,
                "depth_zero_gap_error": _cell(check.max_absolute_gap_error),
                "depth_zero_bellman_residual": _cell(check.max_bellman_residual),
                "bootstrap_gap_sigma_ratio": _cell(
                    dispersion.median_ratio if dispersion else None
                ),
                "bootstrap_value_correlation": _cell(
                    dispersion.median_correlation if dispersion else None
                ),
                "observed_reachable_sa_fraction": coverage_by_key[key],
                "policy_normalized_return": record["policy_normalized_return"],
                "oracle_status": oracle_status,
                "oracle_error": _cell(oracle_error),
            }
        )

        print(
            f"[{index}/{len(records)}] {mdp_id} {family}: "
            f"{pilot.selection.status} k_dagger={result.k_dagger} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    tables["scientific_checks"] = per_pilot
    selected_rows = [row for row in tables["oeh_results"] if row["selected"]]
    n_emp_rows = _read_n_emp(n_emp_path)
    report = {
        "scope": f"development-only: {len(records)} pilot records at 10|S||A|",
        "protocol_status_at_run": manifest["protocol_status"],
        "protocol_schema_version": PROTOCOL_CONFIG["schema_version"],
        "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "bootstrap_seed": bootstrap_seed,
        "oracle_seed": oracle_seed,
        "runtime_seconds": round(time.time() - started, 1),
        "selection_status_counts": dict(sorted(selection_statuses.items())),
        "check_1_depth_zero": {
            "description": (
                "Q_hat^(0) re-derived from raw transitions by an independent "
                "linear solve and compared with the estimator's depth-0 gaps."
            ),
            "pilots_passed": depth_zero_pass,
            "pilots_not_applicable": depth_zero_not_applicable,
            "pilots_not_applicable_note": (
                "an abstained result carries no depth-zero quantities by "
                "design; these are out of scope, not failures"
            ),
            "pilots_failed": len(depth_zero_failures),
            "failures": depth_zero_failures[:10],
            "max_absolute_gap_error": _stats(gap_errors),
            "max_bellman_residual": _stats(bellman_residuals),
        },
        "check_2_bootstrap_dispersion": {
            "description": (
                "PLAN sec 9 numeric check. Ratio = sigma(Q1-Q2) / "
                "sqrt(sigma(Q1)^2 + sigma(Q2)^2): realised gap dispersion over "
                "the dispersion implied by treating the two action values as "
                "independent, which is the inflation sec 3.3 warns about. Well "
                "below 1 is what sec 3.3 assumes; near 1 would invalidate it."
            ),
            "median_ratio_over_pilots": _stats(dispersion_ratios),
            "median_correlation_over_pilots": _stats(dispersion_correlations),
        },
        "check_3_evidence_curve": _evidence_curve_summary(tables["evidence_curves"]),
        "check_4_k_dagger_vs_exact": {
            **compare_k_dagger(selected_rows, exact_by_key),
            "oracle_undefined": {
                "count": len(oracle_exclusion_cases),
                "reasons": dict(oracle_exclusions),
                "cases": oracle_exclusion_cases,
                "handling": "excluded from the comparison denominator and logged",
            },
            "oracle_failures": dict(oracle_failures),
            "oracle_failure_cases": oracle_failure_cases,
            "exact_k_curve_provenance": (
                "Ownership is content-addressed per WEEK2_ARTIFACT_CONTRACT.md "
                "'Ownership is content-addressed, not name-addressed': this file's "
                "authoritative identity is its artifact_sha256 entry above, not "
                "which workstream's driver emitted it. Workstream 1 currently has "
                "no independent generator for it and ingests this run's output via "
                "scripts/validation/run_benchmark_oracle_validation.py --exact-k-curve-source."
            ),
            "delta_sigma_aggregate": "unweighted nanmean over states",
        },
        "check_5_vs_empirical_requirement": (
            compare_with_empirical_requirement(selected_rows, n_emp_rows, pilot_sizes)
            if n_emp_rows
            else {"skipped": "no empirical_sample_complexity.csv supplied"}
        ),
        "check_6_failure_and_abstention": failure_analysis(
            tables["oeh_results"], coverage_by_key
        ),
    }
    return tables, report


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _evidence_curve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Is ``m_k`` flat, monotone, or does it have an interior minimum?

    This is PLAN section 13's risk #1 stated as a measurement: an interior
    minimum is what the depth claim needs; a curve pinned at the ``mu >= 1``
    floor at every depth carries no depth information at all.
    """

    curves: dict[tuple[Any, Any, Any], list[tuple[int, float]]] = {}
    for row in rows:
        if row["multiplier"] == "":
            continue
        key = (row["mdp_id"], row["policy_family"], row["pilot_multiplier"])
        curves.setdefault(key, []).append((int(row["depth"]), float(row["multiplier"])))
    shapes: Counter[str] = Counter()
    interior_depths: list[int] = []
    for values in curves.values():
        series = np.asarray([value for _, value in sorted(values)])
        if np.allclose(series, series[0]):
            shapes["flat"] += 1
            continue
        argmin = int(np.argmin(series))
        if argmin == 0:
            shapes["minimum_at_depth_0"] += 1
        elif argmin == len(series) - 1:
            shapes["minimum_at_k_max"] += 1
        else:
            shapes["interior_minimum"] += 1
            interior_depths.append(argmin)
    return {
        "curves": len(curves),
        "shape_counts": dict(sorted(shapes.items())),
        "interior_minimum_depths": dict(sorted(Counter(interior_depths).items())),
        "all_curves_flat": shapes.get("flat", 0) == len(curves) and bool(curves),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--oracle-seed", type=int, default=20260817)
    parser.add_argument(
        "--n-emp",
        type=Path,
        help="empirical_sample_complexity.csv from Workstream 2, for the N_emp comparison.",
    )
    parser.add_argument("--limit", type=int, help="Only the first N pilots; smoke use only.")
    parser.add_argument("--skip-oracle", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )

    tables, report = run(
        args.dataset_dir,
        bootstrap_seed=args.bootstrap_seed,
        oracle_seed=args.oracle_seed,
        n_emp_path=args.n_emp,
        limit=args.limit,
        skip_oracle=args.skip_oracle,
    )

    tables["k_dagger_estimates"] = selected_subset_rows(
        tables["oeh_results"], UNVALIDATED_FILES["k_dagger_estimates.csv"]
    )
    tables["n_required_estimates"] = selected_subset_rows(
        tables["oeh_results"], UNVALIDATED_FILES["n_required_estimates.csv"]
    )

    written: dict[str, str] = {}
    for name, (filename, fields) in CONTRACT_FILES.items():
        path = args.output_dir / filename
        write_csv_rows(path, tables[name], fields)
        validate_artifact(path, name)
        written[filename] = sha256_file(path)
    for filename, fields in UNVALIDATED_FILES.items():
        path = args.output_dir / filename
        write_csv_rows(path, tables[filename.removesuffix(".csv")], fields)
        written[filename] = sha256_file(path)

    report["artifact_sha256"] = written
    report["schema_validated"] = sorted(CONTRACT_FILES)
    report["schema_unvalidated"] = sorted(
        name for name in UNVALIDATED_FILES if True
    )
    write_json(args.output_dir / "oeh_validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
