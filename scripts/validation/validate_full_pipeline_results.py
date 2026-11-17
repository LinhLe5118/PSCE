"""Validate an existing full development-pipeline result directory.

The legacy one-off pipeline validator treated every missing exact-oracle curve
as an execution failure.  That contradicts the registered metric contract:
the specific "fewer than two support-valid oracle datasets" outcome is
undefined, excluded from the comparison denominator, and logged.  This module
keeps that expected exclusion separate from unexpected oracle errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.validation.run_oeh_validation import ORACLE_UNDEFINED_MESSAGE


LEGACY_UNDEFINED_REASON = f"RuntimeError: {ORACLE_UNDEFINED_MESSAGE}"
ELIGIBLE_OEH_STATUSES = {"observed", "censored"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _embedded_hash_failures(run_dir: Path) -> list[dict[str, str]]:
    """Verify output hashes embedded in artifact manifests and reports."""

    failures: list[dict[str, str]] = []
    for report_path in sorted(run_dir.rglob("*.json")):
        try:
            payload = _read_json(report_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        declared: Mapping[str, Any] | None = None
        if isinstance(payload.get("artifact_sha256"), dict):
            declared = payload["artifact_sha256"]
        elif isinstance(payload.get("outputs"), dict):
            declared = payload["outputs"]
        if declared is None:
            continue
        for filename, value in declared.items():
            expected = value.get("sha256") if isinstance(value, dict) else value
            if not isinstance(expected, str):
                continue
            artifact = report_path.parent / filename
            if not artifact.is_file():
                failures.append(
                    {
                        "manifest": report_path.relative_to(run_dir).as_posix(),
                        "artifact": artifact.relative_to(run_dir).as_posix(),
                        "reason": "missing",
                    }
                )
            elif _sha256(artifact) != expected:
                failures.append(
                    {
                        "manifest": report_path.relative_to(run_dir).as_posix(),
                        "artifact": artifact.relative_to(run_dir).as_posix(),
                        "reason": "sha256_mismatch",
                    }
                )
    return failures


def classify_oracle_outcomes(
    check: Mapping[str, Any], scientific_rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Normalize legacy and current OEH reports and verify their denominator."""

    unexpected = dict(check.get("oracle_failures", {}))
    registered = check.get("oracle_undefined")
    if isinstance(registered, dict):
        undefined_count = int(registered.get("count", 0))
        undefined_reasons = dict(registered.get("reasons", {}))
    else:
        undefined_count = int(unexpected.pop(LEGACY_UNDEFINED_REASON, 0))
        undefined_reasons = (
            {LEGACY_UNDEFINED_REASON: undefined_count} if undefined_count else {}
        )

    eligible = [
        row for row in scientific_rows if row.get("oeh_status") in ELIGIBLE_OEH_STATUSES
    ]
    undefined_cases = [
        {
            "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"],
            "oeh_status": row["oeh_status"],
            "reason": row.get("oracle_error") or LEGACY_UNDEFINED_REASON,
        }
        for row in eligible
        if not row.get("k_dagger_exact")
    ]
    compared = int(check.get("pairs_compared", 0))
    denominator_consistent = (
        not unexpected
        and len(undefined_cases) == undefined_count
        and len(eligible) == compared + undefined_count
    )
    return {
        "eligible_pairs": len(eligible),
        "pairs_compared": compared,
        "undefined_count": undefined_count,
        "undefined_reasons": undefined_reasons,
        "undefined_cases": undefined_cases,
        "unexpected_failures": unexpected,
        "denominator_consistent": denominator_consistent,
    }


def validate(run_dir: Path) -> dict[str, Any]:
    training = _read_json(run_dir / "validation.json")
    status = _read_json(run_dir / "status.json")
    oeh = _read_json(run_dir / "oeh" / "oeh_validation_report.json")
    scientific_rows = _read_csv(run_dir / "oeh" / "scientific_checks.csv")
    check_4 = classify_oracle_outcomes(
        oeh["check_4_k_dagger_vs_exact"], scientific_rows
    )
    hash_failures = _embedded_hash_failures(run_dir)
    prior_stage_failures = [
        item
        for item in status.get("completed", [])
        if item.get("name") != "validate_pipeline" and item.get("exit_code") != 0
    ]
    depth_zero_failures = int(oeh["check_1_depth_zero"]["pilots_failed"])

    integrity_passed = all(
        (
            training.get("status") == "passed",
            training.get("artifact_hashes_verified") is True,
            training.get("unique_run_keys_verified") is True,
            not prior_stage_failures,
            not hash_failures,
            depth_zero_failures == 0,
            not check_4["unexpected_failures"],
            check_4["denominator_consistent"],
        )
    )
    file_hashes = {
        path.relative_to(run_dir).as_posix(): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    return {
        "scope": "expanded development pipeline; sealed evaluation excluded",
        "pipeline_integrity_status": (
            "passed_with_expected_oracle_exclusions" if integrity_passed else "failed"
        ),
        "integrity_passed": integrity_passed,
        "original_validator": {
            "all_stages_passed": status.get("all_stages_passed"),
            "validate_pipeline_exit_code": next(
                (
                    item.get("exit_code")
                    for item in status.get("completed", [])
                    if item.get("name") == "validate_pipeline"
                ),
                None,
            ),
            "misclassification": (
                "The legacy validator treated registered oracle exclusions as failures."
            ),
        },
        "checks": {
            "training_validation": training,
            "prior_stage_failures": prior_stage_failures,
            "embedded_hash_failures": hash_failures,
            "depth_zero_failures": depth_zero_failures,
            "oracle": check_4,
        },
        "scientific_result": {
            "depth_claim_supported": not bool(
                oeh["check_3_evidence_curve"].get("all_curves_flat")
            ),
            "evidence_curve": oeh["check_3_evidence_curve"],
            "interpretation": (
                "Execution and artifact integrity pass, but the observed flat curves "
                "are a negative scientific result for the intended depth claim."
            ),
        },
        "source_file_sha256": file_hashes,
    }


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = ("mdp_id", "policy_family", "oeh_status", "reason")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty; refusing to overwrite it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = validate(args.run_dir)
    (args.output_dir / "pipeline_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_cases(
        args.output_dir / "oracle_undefined_cases.csv",
        report["checks"]["oracle"]["undefined_cases"],
    )
    print(json.dumps({
        "pipeline_integrity_status": report["pipeline_integrity_status"],
        "oracle_undefined": report["checks"]["oracle"]["undefined_count"],
        "unexpected_oracle_failures": report["checks"]["oracle"]["unexpected_failures"],
    }, indent=2))
    if not report["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
