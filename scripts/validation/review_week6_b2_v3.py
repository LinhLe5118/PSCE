"""Final independent Week-6 B2 review after D1 development preflight fixes."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from evaluation.confirmatory import CONFIRMATORY_OUTPUT_DIR_NAME, CONFIRMATORY_RUN_ID
from evaluation.development_confirmatory import (
    build_development_train_score_callback,
    load_development_refit_panel,
)
from scripts.validation.run_confirmatory_preflight import (
    T1_MANIFEST_REL,
    _source_hashes,
    run_preflight,
)
from scripts.validation.verify_week6_h1_freeze import verify as verify_h1
from scripts.validation.verify_week6_t1_freeze import verify as verify_t1


ROOT = Path(__file__).resolve().parents[2]
T1_MANIFEST = ROOT / T1_MANIFEST_REL
D1_EVIDENCE = ROOT / "docs/week6/D1_REAL_DEVELOPMENT_PREFLIGHT_2026-09-24.md"
SUPERSEDED_T1_MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0.sha256.json"


def _extract_number(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Missing D1 evidence matching {pattern!r}")
    return float(match.group(1).replace(",", ""))


def review(*, fixture_draws: int = 2_000) -> dict[str, Any]:
    """Return the final fixture/development-only B2 gate decision."""

    t1 = verify_t1(ROOT, T1_MANIFEST)
    h1 = verify_h1(ROOT)
    source_hashes = _source_hashes()
    evidence = D1_EVIDENCE.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="week6-b2-v3-") as directory:
        output = Path(directory) / CONFIRMATORY_OUTPUT_DIR_NAME
        first = run_preflight(
            output_dir=output,
            fixture=True,
            draws=fixture_draws,
        )
        resumed = run_preflight(
            output_dir=output,
            fixture=True,
            draws=fixture_draws,
            resume=True,
        )

    panel = load_development_refit_panel(root=ROOT)
    train_score_source = inspect.getsource(build_development_train_score_callback)
    real_train_score_symbols = {
        "cfqi_tabular": "cfqi_tabular(" in train_score_source,
        "iql_tabular": "iql_tabular(" in train_score_source,
        "behavior_cloning_tabular": "behavior_cloning_tabular(" in train_score_source,
        "exact_policy_scorer": "_score_policy(" in train_score_source,
    }

    evidence_draws = int(_extract_number(evidence, r"Draws completed\s*\|\s*([0-9,]+)\s*/"))
    evidence_runtime = _extract_number(evidence, r"Wall runtime\s*\|\s*`?([0-9.]+)")
    evidence_memory = int(_extract_number(evidence, r"Peak Python heap\s*\|\s*([0-9,]+)"))
    evidence_source_entries = int(
        _extract_number(evidence, r"source map contains\s+([0-9,]+)\s+entries")
    )

    checks = {
        "t1_integrity": t1["status"] == "PASS",
        "h1_integrity": h1["status"] == "PASS",
        "authoritative_t1_r1_pinned": (
            T1_MANIFEST_REL in source_hashes
            and SUPERSEDED_T1_MANIFEST not in source_hashes
        ),
        "real_train_score_callback_wired": all(real_train_score_symbols.values()),
        "development_refit_panel_complete": (
            len(panel.rows) == 225
            and len(panel.cluster_strata) == 15
            and set(panel.observed) == {"v4e", "fold_best_fixed"}
            and panel.observed["v4e"] != panel.observed["fold_best_fixed"]
        ),
        "development_2000_draw_evidence": (
            evidence_draws == 2_000
            and evidence_runtime > 0
            and evidence_memory > 0
            and re.search(r"inner\s+4-fold\s+ridge\s+selection", evidence) is not None
            and re.search(r"outer\s+5-fold\s+learner\s+selection", evidence) is not None
            and "passed hash verification without re-fitting" in evidence
        ),
        "fixture_full_job_plan": (
            first["plan"]["training_seeds"] == list(range(10))
            and first["plan"]["size_multipliers"] == [1, 2, 5, 10, 20, 50]
            and first["plan"]["counts"]["jobs"] == 360
            and first["plan"]["same_buffer_across_methods"] is True
            and first["plan"]["same_budget_across_methods"] is True
        ),
        "verified_resume_checkpoint": (
            resumed["job_execution"]["status"] == "complete"
            and resumed["job_execution"]["missing_jobs"] == 0
            and resumed["paired_refit_bootstrap"]["draws_completed"] == fixture_draws
        ),
        "evaluation_boundary_preserved": (
            first["evaluation_opened"] is False
            and first["oracle_imported"] is False
            and "did not open the sealed evaluation" in evidence
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"B2 final review checks failed: {failed}")

    return {
        "schema_version": "week6-b2-independent-preflight-review-v3",
        "status": "GO",
        "reviewer": "Việt Trần",
        "decision": (
            "B2 execution preflight gate passed. This does not itself authorize "
            "opening sealed evaluation; E1 still requires explicit authorization "
            "and first-access seal/schema/access-log checks."
        ),
        "evaluation_manifest_buffers_or_outcomes_read": False,
        "package_integrity": {"T1": t1, "H1": h1},
        "checks": checks,
        "closed_findings": [
            "B2-NG-01-ACTUAL-CONFIRMATORY-CALLBACK",
            "B2-NG-02-ACTUAL-REFIT-BOOTSTRAP",
            "B2-NG-03-T1-R1-SOURCE-PIN",
        ],
        "fixture_evidence": {
            "run_id": first["run_id"],
            "run_id_matches_frozen_contract": first["run_id"] == CONFIRMATORY_RUN_ID,
            "jobs": first["plan"]["counts"]["jobs"],
            "draws": first["paired_refit_bootstrap"]["draws_completed"],
            "resume_missing_jobs": resumed["job_execution"]["missing_jobs"],
            "evaluation_opened": first["evaluation_opened"],
        },
        "development_evidence": {
            "train_score_symbols": real_train_score_symbols,
            "refit_rows": len(panel.rows),
            "refit_base_mdp_clusters": len(panel.cluster_strata),
            "observed_endpoints": dict(panel.observed),
            "reported_draws": evidence_draws,
            "reported_runtime_seconds": evidence_runtime,
            "reported_peak_python_heap_bytes": evidence_memory,
            "reported_source_hash_entries": evidence_source_entries,
            "d1_evidence_path": str(D1_EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
        },
        "non_blocking_limitations": [
            (
                "The full D1 dry-run checkpoint/job bundle remains in the D1 owner's "
                "ignored local tmp directory; the committed handover retains selected "
                "artifact hashes, runtime/memory, inputs, and reproduction commands."
            ),
            (
                "B2 independently exercised the tracked development refit callback and "
                "fixture execution contract, but did not recreate the external development "
                "training bundle or generate a new development dataset."
            ),
        ],
        "e1_first_access_requirements": [
            "Obtain explicit authorization to open E1.",
            "Verify sealed bundle checksum, schema, quota, split isolation, and base-MDP aliases before buffer access.",
            "Write the first-access log and stop on any mismatch.",
            "Retain raw outputs, logs, runtime, failures, and all negative results without retuning.",
        ],
        "handover": {
            "linh_p1": "May finalize P1.",
            "linh_e1": "May start only after explicit E1 authorization and the first-access checks above.",
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-draws", type=int, default=2_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists():
        parser.error("Output exists; preserve previous review evidence")
    report = review(fixture_draws=args.fixture_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
