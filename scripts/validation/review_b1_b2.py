"""Create a reproducible B1/B2 acceptance review from immutable artifacts.

The reviewer never modifies its inputs.  B1 re-hashes the public evidence
artifacts and records the oracle/undefined contract.  B2 reads the sealed
Week-5 source bundle directly, verifies every declared file digest, then
recomputes job-grid, N_emp-label, and learning-curve checks from job rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = "artifacts/week5/full_development_pipeline_2026-09-14_181522/"
MULTIPLIERS = (1, 2, 5, 10, 20, 50)
SEEDS = tuple(range(10))
REGISTERED = ("cfqi_tabular", "iql_tabular")
BC = "behavior_cloning_tabular"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_text(value: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(value.decode("utf-8"))))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_hashes(items: Iterable[tuple[Path, str]]) -> list[dict[str, Any]]:
    checks = []
    for path, expected in items:
        raw = path.read_bytes()
        actual = digest_bytes(raw)
        crlf_actual = digest_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        byte_exact = actual == expected
        line_ending_only = not byte_exact and crlf_actual == expected
        checks.append({"path": str(path.relative_to(ROOT)), "expected": expected,
                       "actual": actual, "match": byte_exact or line_ending_only,
                       "verification": "byte_exact" if byte_exact else "crlf_equivalent" if line_ending_only else "mismatch"})
    require(all(item["match"] for item in checks), "A B1 artifact digest did not match, including CRLF equivalence")
    return checks


def review_b1() -> dict[str, Any]:
    figure = read_json(ROOT / "paper/figures/oeh_mechanism_report.json")
    positive = read_json(ROOT / "artifacts/week2/benchmark_positive_control_validation_2026-09-03/positive_control_report.json")
    exact = read_json(ROOT / "artifacts/week2/exact_corridor_curve_2026-09-07/exact_corridor_report.json")
    oracle = read_json(ROOT / "artifacts/week2/benchmark_oracle_handoff_2026-08-23/benchmark_oracle_validation_report.json")
    pipeline = read_json(ROOT / "artifacts/week5/pipeline_validation_2026-09-15/pipeline_validation.json")

    checks: list[tuple[Path, str]] = []
    for name, expected in figure["output_sha256"].items():
        checks.append((ROOT / "paper/figures" / name, expected))
    checks.append((ROOT / "artifacts/week2/benchmark_positive_control_validation_2026-09-03/positive_control.csv",
                   positive["artifact_sha256"]["positive_control.csv"]))
    for name, expected in exact["artifact_sha256"].items():
        checks.append((ROOT / "artifacts/week2/exact_corridor_curve_2026-09-07" / name, expected))
    for name, expected in oracle["artifact_sha256"].items():
        checks.append((ROOT / "artifacts/week2/benchmark_oracle_handoff_2026-08-23" / name, expected))

    pipeline_oracle = pipeline["checks"]["oracle"]
    require(pipeline["integrity_passed"], "Pipeline integrity did not pass")
    require(pipeline_oracle["unexpected_failures"] == {}, "Unexpected oracle failures remain")
    require(exact["undefined_k_dagger_exact"] == exact["undefined_no_two_action_state"],
            "Exact undefined rows are not explained by the support contract")

    return {
        "workstream": "B1 — Benchmark & Oracle Evidence",
        "status": "pass_with_scoped_limitations",
        "review_method": "Re-hash committed evidence artifacts and reconcile their recorded oracle contracts.",
        "artifact_hash_checks": check_hashes(checks),
        "figure": {
            "paths": ["paper/figures/oeh_mechanism.csv", "paper/figures/oeh_mechanism.pdf",
                      "paper/figures/oeh_mechanism.png", "paper/figures/oeh_mechanism.svg"],
            "generator": "scripts/analysis/plot_oeh_mechanism.py",
            "caption_source": "paper/sections/06_results.tex",
            "selected_source_row": figure["selected_source_row"],
            "scope": figure["scope"],
            "units": {
                "depth": "Bellman-backup count",
                "weighted_gap": "discounted return units (occupancy-weighted action-value gap)",
                "weighted_sigma": "discounted return units",
                "loss_at_one_and_theorem_ceiling": "discounted return units",
                "epsilon": "discounted return units",
                "multiplier": "dimensionless data-size multiplier",
            },
        },
        "positive_controls": {
            "seed": positive["seed"], "criterion": positive["criterion"],
            "passed": [item["label"] for item in positive["verdicts"] if item["passed"]],
            "limitation": "Controls exclude a universal hard-coded floor; they do not validate every model assumption.",
        },
        "oracle_reconciliation": {
            "development_oracle_rows": oracle["oracle_row_count"],
            "development_exact_curve_pairs": oracle["exact_k_curve_pairs"],
            "development_exact_k_dagger_distribution": oracle["exact_k_dagger_distribution"],
            "corridor_exact_undefined": exact["undefined_k_dagger_exact"],
            "corridor_undefined_reason": "fewer than two support-valid actions",
            "pipeline_pairs_compared": pipeline_oracle["pairs_compared"],
            "pipeline_expected_undefined": pipeline_oracle["undefined_count"],
            "pipeline_unexpected_failures": len(pipeline_oracle["unexpected_failures"]),
            "estimator_limit": "Undefined oracle rows are excluded from the oracle denominator; they are never recoded as depth zero.",
        },
        "deviations": [
            {"id": "B1-CSV-LINE-ENDINGS", "status": "resolved_by_representation_check",
             "finding": "Three historical oracle CSVs have LF bytes in this checkout while their generation-time hashes encode CRLF bytes.",
             "handling": "The reviewer records both hashes and accepts only exact CRLF-normalized equivalence; no CSV content was changed.",
             "affected_files": [item["path"] for item in check_hashes(checks) if item["verification"] == "crlf_equivalent"]},
        ],
        "limitations": [
            "All benchmark evidence is development/probe evidence; the sealed 120-MDP evaluation split was not opened.",
            "The corridor figure is a post-hoc mechanism visualization, not a confirmatory estimate.",
            "Flat exact curves validate computation on these artifacts but do not establish a positive-depth claim.",
        ],
    }


def zip_entry(bundle: zipfile.ZipFile, relative_path: str) -> bytes:
    try:
        return bundle.read(RUN_ROOT + relative_path)
    except KeyError as error:
        raise ValueError(f"Missing source-bundle entry: {relative_path}") from error


def run_key(row: dict[str, str]) -> tuple[str, str, str, int, int]:
    return (row["mdp_id"], row["policy_family"], row["algorithm"],
            int(row["training_size_multiplier"]), int(row["training_seed"]))


def label_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["mdp_id"], row["policy_family"], row["algorithm"]


def recompute_label(rows: list[dict[str, str]], label: dict[str, str]) -> dict[str, Any]:
    key = label_key(label)
    selected = [row for row in rows if row["mdp_id"] == key[0] and row["policy_family"] == key[1]
                and row["algorithm"] == key[2]]
    counts = []
    for multiplier in MULTIPLIERS:
        cell = [row for row in selected if int(row["training_size_multiplier"]) == multiplier]
        require(len(cell) == len(SEEDS), f"Representative {key} has incomplete seed cell")
        counts.append(sum(as_bool(row["success"]) for row in cell))
    expected_multiplier = next((multiplier for index, multiplier in enumerate(MULTIPLIERS)
                                if all(value >= 8 for value in counts[index:])), None)
    expected_censored = expected_multiplier is None
    recorded_counts = json.loads(label["success_counts"])
    recorded_multiplier = int(label["n_emp_multiplier"]) if label["n_emp_multiplier"] else None
    return {
        "mdp_id": key[0], "policy_family": key[1], "algorithm": key[2],
        "recorded_success_counts": recorded_counts, "recomputed_success_counts": counts,
        "recorded_n_emp_multiplier": recorded_multiplier,
        "recomputed_n_emp_multiplier": expected_multiplier,
        "recorded_censored": as_bool(label["n_emp_censored"]),
        "recomputed_censored": expected_censored,
        "match": (recorded_counts == counts and recorded_multiplier == expected_multiplier
                  and as_bool(label["n_emp_censored"]) == expected_censored),
    }


def curve_check(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> dict[str, Any]:
    max_mean_error = max_median_error = max_success_error = 0.0
    for item in summary:
        algorithm, multiplier = item["algorithm"], int(item["training_size_multiplier"])
        cell = [float(row["normalized_return"]) for row in rows if row["algorithm"] == algorithm
                and int(row["training_size_multiplier"]) == multiplier]
        successes = [as_bool(row["success"]) for row in rows if row["algorithm"] == algorithm
                     and int(row["training_size_multiplier"]) == multiplier]
        require(len(cell) == 2400 and len(successes) == 2400, "Learning-curve cell has wrong size")
        max_mean_error = max(max_mean_error, abs(statistics.fmean(cell) - float(item["mean_normalized_return"])))
        max_median_error = max(max_median_error, abs(statistics.median(cell) - float(item["median_normalized_return"])))
        max_success_error = max(max_success_error, abs(statistics.fmean(successes) - float(item["success_rate"])))
    return {"cells_checked": len(summary), "max_mean_error": max_mean_error,
            "max_median_error": max_median_error, "max_success_rate_error": max_success_error,
            "match_atol_1e-12": max(max_mean_error, max_median_error, max_success_error) <= 1e-12}


def review_b2() -> dict[str, Any]:
    bundle_path = ROOT / "artifacts/week5/source_bundles/full_pipeline_results_2026-09-14.zip"
    expected_archive = (ROOT / "artifacts/week5/source_bundles/full_pipeline_results_2026-09-14.sha256").read_text().split()[0]
    pipeline = read_json(ROOT / "artifacts/week5/pipeline_validation_2026-09-15/pipeline_validation.json")
    provenance = read_json(ROOT / "artifacts/week5/d2_learner_analysis_2026-09-15/provenance.json")
    analysis = read_json(ROOT / "artifacts/week5/d2_learner_analysis_2026-09-15/analysis.json")

    archive_digest = digest_file(bundle_path)
    require(archive_digest == expected_archive, "Source bundle digest does not match its detached SHA-256")
    with zipfile.ZipFile(bundle_path) as bundle:
        source_checks = []
        for name, expected in pipeline["source_file_sha256"].items():
            actual = digest_bytes(zip_entry(bundle, name))
            source_checks.append({"path": name, "match": actual == expected})
        require(all(item["match"] for item in source_checks), "A declared pipeline source hash did not match")

        runs = read_csv_text(zip_entry(bundle, "training/rl_run_results.csv"))
        bc_runs = read_csv_text(zip_entry(bundle, "training/bc_sanity_results.csv"))
        labels = read_csv_text(zip_entry(bundle, "training/empirical_sample_complexity.csv"))
        metadata = read_csv_text(zip_entry(bundle, "training/behavior_policy_metadata.csv"))
        curves = read_csv_text(zip_entry(bundle, "training/learning_curve_summary.csv"))
        manifest = json.loads(zip_entry(bundle, "training/artifact_manifest.json"))

    require(set(runs[0]) == {"mdp_id", "policy_family", "algorithm", "training_size_multiplier",
                             "training_seed", "normalized_return", "success", "run_status"}, "Run schema changed")
    require(len(metadata) == 240 and len(labels) == 480, "Metadata or label row count differs")
    keys = [run_key(row) for row in runs]
    bc_keys = [run_key(row) for row in bc_runs]
    expected_registered = 48 * 5 * len(REGISTERED) * len(MULTIPLIERS) * len(SEEDS)
    expected_bc = 48 * 5 * len(MULTIPLIERS) * len(SEEDS)
    require(len(runs) == expected_registered and len(set(keys)) == expected_registered, "Registered run grid differs")
    require(len(bc_runs) == expected_bc and len(set(bc_keys)) == expected_bc, "BC run grid differs")
    require(not any(row["run_status"] == "invalid" for row in runs + bc_runs), "Invalid run found")

    labels_by_key = {label_key(row): row for row in labels}
    require(len(labels_by_key) == 480, "Duplicate or missing label keys")
    representatives = []
    for algorithm in REGISTERED:
        for censored in (False, True):
            candidates = sorted((row for row in labels if row["algorithm"] == algorithm
                                 and as_bool(row["n_emp_censored"]) == censored), key=label_key)
            require(candidates, f"No representative label for {algorithm}, censored={censored}")
            representatives.append(recompute_label(runs, candidates[0]))
    require(all(item["match"] for item in representatives), "Representative N_emp recomputation differs")

    curves_checked = curve_check(runs + bc_runs, curves)
    require(curves_checked["match_atol_1e-12"], "Learning-curve summary differs from job rows")

    observed_job_audit = {item["algorithm"]: item for item in analysis["job_audit"]}
    require(all(observed_job_audit[name]["status"] == "pass" for name in (*REGISTERED, BC)),
            "Published D2 job audit is not pass")
    pipeline_training = pipeline["checks"]["training_validation"]
    require(pipeline_training["status"] == "passed" and pipeline_training["unique_run_keys_verified"],
            "Pipeline training validator is not pass")

    return {
        "workstream": "B2 — Independent Training Review",
        "status": "pass_with_documented_oracle_exclusions",
        "review_method": "Independent recomputation from the immutable source ZIP; no learner result, source CSV, or historical artifact was modified.",
        "source_bundle": {"path": str(bundle_path.relative_to(ROOT)), "sha256": archive_digest,
                          "detached_sha256_match": True, "declared_source_files_checked": len(source_checks),
                          "declared_source_files_all_match": True},
        "configuration_and_schema": {
            "profile": "config/profiles/expanded_development_v2.json",
            "dataset_manifest_sha256": manifest["input_manifest_sha256"],
            "algorithms": list(REGISTERED), "bc_role": "sanity baseline",
            "multipliers": list(MULTIPLIERS), "seeds": list(SEEDS),
            "run_schema": "exact header match", "metadata_rows": len(metadata), "label_rows": len(labels),
        },
        "job_grid": {
            "registered_rows": len(runs), "expected_registered_rows": expected_registered,
            "bc_rows": len(bc_runs), "expected_bc_rows": expected_bc,
            "registered_duplicate_keys": len(keys) - len(set(keys)),
            "bc_duplicate_keys": len(bc_keys) - len(set(bc_keys)),
            "invalid_rows": sum(row["run_status"] == "invalid" for row in runs + bc_runs),
        },
        "representative_n_emp_rechecks": representatives,
        "learning_curve_recomputation": curves_checked,
        "consistency": {
            "published_d2_job_audit_pass": True,
            "pipeline_training_validation": pipeline_training,
            "provenance_inputs": provenance["inputs"],
        },
        "deviations": [
            {"id": "ORACLE-UNDEFINED-59", "status": "resolved_and_excluded",
             "finding": "The original pipeline validator treated 59 cases with fewer than two support-valid oracle datasets as failures.",
             "handling": "The corrected contract records them as oracle_undefined, excludes them from the oracle denominator, and preserves their observed/censored learner status.",
             "evidence": "artifacts/week5/pipeline_validation_2026-09-15/oracle_undefined_cases.csv; 91 compared, 59 expected undefined, 0 unexpected oracle failures."},
            {"id": "TRAINING-REVIEW-RESULT", "status": "no_new_discrepancy",
             "finding": "All source hashes, schemas, grid keys, four representative N_emp labels, and 18 learning-curve cells matched independent recomputation.",
             "handling": "No result was changed; the review report is additive and historical artifacts remain untouched.",
             "evidence": "This report's source-bundle, representative_n_emp_rechecks, and learning_curve_recomputation fields."},
        ],
        "limitations": [
            "This is an independent artifact review and deterministic recomputation, not a new stochastic training run.",
            "The review is restricted to the 48-MDP expanded development profile; it is not held-out evaluation evidence.",
            "Oracle-undefined cases remain non-comparable by contract and must not be read as zero depth or training failure.",
        ],
    }


def markdown_summary(b1: dict[str, Any], b2: dict[str, Any]) -> str:
    oracle = b1["oracle_reconciliation"]
    b2_grid = b2["job_grid"]
    return f"""# B1/B2 acceptance review — 2026-09-17

This additive review reads immutable committed artifacts and the Week-5 source
bundle. It does not overwrite a historical artifact or execute a new learner
training run.

## B1 — Benchmark & Oracle Evidence: PASS WITH SCOPED LIMITATIONS

- Re-hashed {len(b1['artifact_hash_checks'])} evidence files; every digest matches.
- The mechanism figure has source data, generator, caption source and units in
  `b1_benchmark_oracle_review.json`.
- Positive controls pass the unchanged registered criterion.
- Oracle reconciliation: {oracle['pipeline_pairs_compared']} comparable pairs,
  {oracle['pipeline_expected_undefined']} expected undefined pairs, and
  {oracle['pipeline_unexpected_failures']} unexpected failures. Undefined cases
  are excluded rather than recoded as depth zero.

## B2 — Independent Training Review: PASS WITH DOCUMENTED ORACLE EXCLUSIONS

- Verified the detached SHA-256 of the source bundle and all
  {b2['source_bundle']['declared_source_files_checked']} declared source-file hashes.
- Recomputed the {b2_grid['registered_rows']:,}-row registered grid and
  {b2_grid['bc_rows']:,}-row BC grid: no duplicate keys and no invalid rows.
- Recomputed four deterministic representative `N_emp` labels (finite and
  right-censored for each registered learner) and all 18 learning-curve cells.
  All match the source job rows.
- The resolved deviation and its evidence are listed in
  `b2_independent_training_review.json`.

## Boundaries

Both reviews concern development/probe artifacts. The 120-MDP evaluation split
remains sealed; no result here is held-out validation or a causal claim about a
learner.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/week5/b1_b2_acceptance_review_2026-09-17")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    b1, b2 = review_b1(), review_b2()
    (args.output_dir / "b1_benchmark_oracle_review.json").write_text(json.dumps(b1, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "b2_independent_training_review.json").write_text(json.dumps(b2, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(markdown_summary(b1, b2), encoding="utf-8")


if __name__ == "__main__":
    main()
