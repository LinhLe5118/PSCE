"""Re-review Week-6 B2 after the D1 confirmatory preflight landed.

The review is deliberately fixture-only.  It never discovers, opens, or
materializes the sealed evaluation bundle.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

from scripts.validation.run_confirmatory_preflight import (
    _fixture_bootstrap_callback,
    _identity_job_runner,
    _source_hashes,
    run_preflight,
)
from evaluation.confirmatory import CONFIRMATORY_OUTPUT_DIR_NAME, CONFIRMATORY_RUN_ID
from scripts.validation.verify_week6_h1_freeze import verify as verify_h1
from scripts.validation.verify_week6_t1_freeze import verify as verify_t1


ROOT = Path(__file__).resolve().parents[2]
T1_MANIFEST = ROOT / "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
AUTHORITATIVE_T1_MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
SUPERSEDED_T1_MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0.sha256.json"


def review(*, draws: int = 2_000) -> dict[str, Any]:
    """Run an independent fixture review and return machine-readable evidence."""

    t1 = verify_t1(ROOT, T1_MANIFEST)
    h1 = verify_h1(ROOT)

    with tempfile.TemporaryDirectory(prefix="week6-b2-v2-") as directory:
        output = Path(directory) / CONFIRMATORY_OUTPUT_DIR_NAME
        first = run_preflight(output_dir=output, fixture=True, draws=draws)
        resumed = run_preflight(
            output_dir=output,
            fixture=True,
            draws=draws,
            resume=True,
        )

    source_hashes = _source_hashes()
    identity_source = inspect.getsource(_identity_job_runner)
    bootstrap_source = inspect.getsource(_fixture_bootstrap_callback)

    full_plan_pass = (
        first["plan"]["training_seeds"] == list(range(10))
        and first["plan"]["size_multipliers"] == [1, 2, 5, 10, 20, 50]
        and first["plan"]["counts"]["jobs"] == 360
        and first["plan"]["same_buffer_across_methods"] is True
        and first["plan"]["same_budget_across_methods"] is True
    )
    resume_pass = (
        resumed["job_execution"]["status"] == "complete"
        and resumed["job_execution"]["missing_jobs"] == 0
        and resumed["paired_refit_bootstrap"]["draws_completed"] == draws
    )
    uses_identity_jobs = (
        "fixture_contract_passed" in identity_source
        and "intentionally does not score" in identity_source
    )
    uses_weighted_mean_fixture = (
        "sum(values[0] * count" in bootstrap_source
        and "sum(values[1] * count" in bootstrap_source
    )
    pins_authoritative_t1 = AUTHORITATIVE_T1_MANIFEST in source_hashes
    pins_superseded_t1 = SUPERSEDED_T1_MANIFEST in source_hashes

    blockers = [
        {
            "id": "B2-NG-01-ACTUAL-CONFIRMATORY-CALLBACK",
            "finding": (
                "The new runner enforces the execution contract, but its job callback "
                "is still _identity_job_runner and explicitly does not train or score."
            ),
            "evidence": {
                "callback": "_identity_job_runner",
                "fixture_status": "fixture_contract_passed",
                "actual_learner_fit_or_score": False,
            },
            "owner": "Nguyễn Quốc Việt (D1/E1 implementation)",
            "required_resolution": (
                "Connect the frozen CFQI/IQL/BC and OEH-v4E analysis callback on "
                "development data, then preserve the same stop/resume contract."
            ),
        },
        {
            "id": "B2-NG-02-ACTUAL-REFIT-BOOTSTRAP",
            "finding": (
                "The 2,000-draw harness runs, but the supplied callback only averages "
                "fixture values; it does not refit standardization, inner ridge, "
                "meta-models, or fold-best-fixed in each draw."
            ),
            "evidence": {
                "draws_completed": first["paired_refit_bootstrap"]["draws_completed"],
                "callback": "_fixture_bootstrap_callback",
                "actual_meta_refit_per_draw": False,
                "measured_seconds": first["runtime_memory"]["bootstrap_runtime_seconds"],
                "runtime_represents_actual_refit": False,
            },
            "owner": "Nguyễn Quốc Việt (D1/E1 implementation)",
            "required_resolution": (
                "Dry-run the complete refit callback on development evidence and record "
                "runtime/memory plus failure/resume evidence for the real computation."
            ),
        },
        {
            "id": "B2-NG-03-T1-R1-SOURCE-PIN",
            "finding": (
                "The runner hashes the superseded T1 manifest instead of the "
                "authoritative v1.1.0-r1 manifest named by WEEK6_T1_CURRENT.json."
            ),
            "evidence": {
                "authoritative_manifest_pinned": pins_authoritative_t1,
                "superseded_manifest_pinned": pins_superseded_t1,
            },
            "owner": "Nguyễn Quốc Việt (D1/E1 implementation)",
            "required_resolution": (
                "Replace the superseded manifest pin with the r1 manifest (and its "
                "detached integrity file if required), then regenerate preflight evidence."
            ),
        },
    ]

    if not full_plan_pass:
        raise ValueError("D1 fixture no longer proves the complete frozen job plan")
    if not resume_pass:
        raise ValueError("D1 fixture resume/checkpoint evidence did not pass")
    if not uses_identity_jobs or not uses_weighted_mean_fixture:
        raise ValueError("D1 callbacks changed; this review must be updated before use")

    return {
        "schema_version": "week6-b2-independent-preflight-review-v2",
        "status": "NO-GO",
        "reviewer": "Việt Trần",
        "decision": "Do not open or materialize the sealed evaluation cohort.",
        "evaluation_manifest_buffers_or_outcomes_read": False,
        "review_scope": "T1/H1 integrity plus D1 fixture-only confirmatory preflight",
        "package_integrity": {"T1": t1, "H1": h1},
        "fixture_evidence": {
            "run_id": first["run_id"],
            "run_id_matches_frozen_contract": first["run_id"] == CONFIRMATORY_RUN_ID,
            "evaluation_opened": first["evaluation_opened"],
            "oracle_imported": first["oracle_imported"],
            "methods": first["plan"]["methods"],
            "training_seeds": first["plan"]["training_seeds"],
            "size_multipliers": first["plan"]["size_multipliers"],
            "job_count": first["plan"]["counts"]["jobs"],
            "bootstrap_draws": first["paired_refit_bootstrap"]["draws_completed"],
            "resume_completed_jobs": resumed["job_execution"]["completed_jobs"],
            "resume_missing_jobs": resumed["job_execution"]["missing_jobs"],
        },
        "acceptance_matrix": {
            "t1_integrity": "PASS",
            "h1_integrity_and_formula_contract": "PASS",
            "evaluation_boundary": "PASS_NOT_OPENED",
            "full_core_seed_job_plan": "PASS_FIXTURE",
            "verified_resume_checkpoint_policy": "PASS_FIXTURE",
            "actual_confirmatory_training_scoring_callback": "FAIL",
            "actual_paired_refit_bootstrap_development_dry_run": "FAIL",
            "authoritative_t1_r1_source_pin": "FAIL",
        },
        "closed_previous_findings": [
            "B2-NG-03-SEED-SCHEDULE",
            "B2-NG-04-RESUME-CHECKPOINT",
        ],
        "blockers": blockers,
        "handover": {
            "quoc_viet": "Close the three remaining D1/E1 implementation blockers.",
            "hung": "Review the real callback against the frozen H1 formula and fallback.",
            "linh_p1": "May continue and finalize P1 using frozen/development evidence.",
            "linh_e1": "Must not start until fixes land and Việt Trần issues B2 GO.",
        },
        "reviewer_interpretation": (
            "PR #40 materially improves the execution harness and closes the full seed-plan "
            "and verified-resume findings on fixtures. It does not yet demonstrate the actual "
            "training/scoring or full refit-bootstrap computation, and it pins the wrong T1 "
            "manifest revision; therefore B2 remains NO-GO."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists():
        parser.error("Output exists; preserve previous review evidence")
    report = review(draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
