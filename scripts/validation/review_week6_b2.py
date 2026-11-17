"""Independent Week-6 B2 review without opening sealed evaluation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.validation.run_baseline_preflight import (
    CLI_PROBES, METHODS, _probe_cli, _sealed_evaluation_guard,
)
from scripts.validation.verify_week6_h1_freeze import verify as verify_h1
from scripts.validation.verify_week6_t1_freeze import verify as verify_t1


ROOT = Path(__file__).resolve().parents[2]
T1_MANIFEST = ROOT / "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
T1_CONTRACT = ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json"
H1_CONTRACT = ROOT / "docs/protocol/WEEK6_H1_FORMULA_v1.0.0.json"
D1_DIR = ROOT / "artifacts/week6/d1_unified_preflight_2026-09-21_final"
D1_REPORT = D1_DIR / "preflight_report.json"
D1_PROVENANCE = D1_DIR / "baseline_provenance.csv"
DEVELOPMENT_KERNEL_AUDIT = ROOT / "artifacts/week6/stochastic_benchmark_2026-09-21/development60.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def review(run_cli_probes: bool = True) -> dict[str, Any]:
    t1 = verify_t1(ROOT, T1_MANIFEST)
    h1 = verify_h1(ROOT)
    t1_contract = read_json(T1_CONTRACT)
    h1_contract = read_json(H1_CONTRACT)
    d1 = read_json(D1_REPORT)
    kernels = read_json(DEVELOPMENT_KERNEL_AUDIT)

    require(d1["status"] == "pass" and d1["dry_run"] is True and d1["fit_called"] is False,
            "D1 is not a passing no-fit dry-run")
    require(d1["sealed_evaluation_accessed"] is False, "D1 reports evaluation access")
    require(d1["checks"]["dataset"]["sealed_evaluation_guard"]["evaluation_buffers_opened"] == 0,
            "D1 opened an evaluation buffer")

    source_hash_checks = []
    for name, expected in d1["provenance"]["source_sha256"].items():
        actual = sha256(ROOT / name)
        source_hash_checks.append({"path": name, "expected": expected, "actual": actual, "match": actual == expected})
    require(all(row["match"] for row in source_hash_checks), "A D1 source hash changed")

    with D1_PROVENANCE.open("r", encoding="utf-8-sig", newline="") as handle:
        provenance_rows = list(csv.DictReader(handle))
    report_methods = {row["method"] for row in d1["provenance"]["methods"]}
    require({row["method"] for row in provenance_rows} == report_methods == {row["method"] for row in METHODS},
            "D1 provenance methods disagree")

    fixture_development = _sealed_evaluation_guard({
        "instances": [{"instance_id": "fixture-dev", "split": "development"}],
        "datasets": [{"dataset_id": "fixture-dev-pilot", "benchmark_split": "development"}],
    })
    fixture_rejects_evaluation = False
    try:
        _sealed_evaluation_guard({
            "instances": [{"instance_id": "fixture-eval", "split": "evaluation"}],
            "datasets": [],
        })
    except ValueError:
        fixture_rejects_evaluation = True
    require(fixture_rejects_evaluation, "Fixture guard failed to reject evaluation metadata")

    cli = ([_probe_cli(ROOT / item["path"]) for item in CLI_PROBES]
           if run_cli_probes else d1["checks"]["runner_cli_help"])
    require(all(row["status"] == "pass" for row in cli), "A runner CLI smoke probe failed")

    collection = read_json(ROOT / "protocol_config.json")["collection"]
    pilot_seeds = {int(collection["pilot_seed_offset"]) + int(index) for index in collection["pilot_seed_indices"]}
    training_seeds = {int(collection["training_seed_offset"]) + int(index) for index in collection["training_seed_indices"]}
    seed_namespace_disjoint = pilot_seeds.isdisjoint(training_seeds)
    require(seed_namespace_disjoint, "Pilot/training seed namespaces overlap")

    scripts_with_run_id = []
    run_id = t1_contract["execution"]["run_id"]
    for path in (ROOT / "scripts").rglob("*.py"):
        if run_id in path.read_text(encoding="utf-8"):
            scripts_with_run_id.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    required_training_seed_indices = list(collection["training_seed_indices"])
    d1_seed_indices = d1["requested_contract"]["training_seeds"]
    expected_output = ROOT / t1_contract["execution"]["output"]
    blockers = [
        {
            "id": "B2-NG-01-CONFIRMATORY-RUNNER",
            "finding": "No executable under scripts/ contains the frozen run_id; D1 probes development runners only.",
            "evidence": {"run_id": run_id, "scripts_with_run_id": scripts_with_run_id},
            "owner": "D1/E1 implementation owner",
            "required_resolution": "Implement and fixture-test a separate sealed confirmatory runner without weakening the global evaluation guard.",
        },
        {
            "id": "B2-NG-02-REFIT-BOOTSTRAP",
            "finding": "D1 does not dry-run the 2,000-draw paired base-MDP refit bootstrap or provide a runtime estimate.",
            "evidence": {"d1_fit_called": d1["fit_called"], "d1_schema": d1["schema_version"], "t1_draws": t1_contract["bootstrap"]["draws"]},
            "owner": "D1 implementation owner",
            "required_resolution": "Add a fixture/development refit-bootstrap dry-run, failure injection, and measured runtime/memory estimate.",
        },
        {
            "id": "B2-NG-03-SEED-SCHEDULE",
            "finding": "D1 exercised training seed index 0 only, not the frozen 0..9 schedule.",
            "evidence": {"d1_seed_indices": d1_seed_indices, "required_seed_indices": required_training_seed_indices},
            "owner": "D1 implementation owner",
            "required_resolution": "Preflight all job keys for ten core seeds and demonstrate identical buffers/budgets across the core learners.",
        },
        {
            "id": "B2-NG-04-RESUME-CHECKPOINT",
            "finding": "The named confirmatory run has no implementation of verified missing-job resume, completed-output hash checks, or checkpoint identity.",
            "evidence": {"policy": t1_contract["execution"]["resume"], "confirmatory_runner_present": bool(scripts_with_run_id)},
            "owner": "D1/E1 implementation owner",
            "required_resolution": "Fixture-test mismatch refusal and missing-jobs-only resume for source/config/input/job/output identities.",
        },
    ]

    return {
        "schema_version": "week6-b2-independent-preflight-review-v1",
        "status": "NO-GO",
        "owner": "Việt Trần",
        "decision": "Do not open or materialize the sealed evaluation cohort.",
        "evaluation_manifest_buffers_or_outcomes_read": False,
        "package_integrity": {
            "T1": t1, "H1": h1,
            "D1": {
                "status": d1["status"], "scope": "development-only no-fit baseline/CLI preflight",
                "report_sha256": sha256(D1_REPORT), "provenance_sha256": sha256(D1_PROVENANCE),
                "source_hashes_checked": len(source_hash_checks),
                "source_hashes_all_match": all(row["match"] for row in source_hash_checks),
                "candidate_rows": len(provenance_rows),
            },
        },
        "schema_and_formula": {
            "t1_version": t1_contract["version"], "h1_version": h1_contract["version"],
            "h1_feature_dimensions": h1_contract["feature_dimensions"],
            "d1_schema": d1["schema_version"], "d1_methods": sorted(report_methods),
        },
        "seed_namespace": {
            "rng": collection["rng"], "pilot_seed_values": sorted(pilot_seeds),
            "training_seed_values": sorted(training_seeds), "disjoint": seed_namespace_disjoint,
            "d1_exercised_indices": d1_seed_indices, "required_training_indices": required_training_seed_indices,
        },
        "paths_and_outputs": {
            "frozen_run_id": run_id, "frozen_output_directory": t1_contract["execution"]["output"],
            "frozen_output_directory_exists": expected_output.exists(),
            "confirmatory_scripts_with_run_id": scripts_with_run_id,
        },
        "fixture_and_development_checks": {
            "development_fixture_guard": fixture_development,
            "evaluation_fixture_rejected_before_buffer_access": fixture_rejects_evaluation,
            "runner_cli_probes": cli,
            "development_kernel_audit_status": kernels["status"],
            "development_kernel_panels": kernels["development_counts"],
            "new_development_dataset_created": False,
        },
        "acceptance_matrix": {
            "t1_integrity": "PASS", "h1_integrity_and_formula_contract": "PASS",
            "d1_baseline_provenance_and_no_fit_cli_scope": "PASS",
            "sealed_confirmatory_runner": "FAIL", "paired_refit_bootstrap_dry_run": "FAIL",
            "full_core_seed_job_plan": "FAIL", "verified_resume_checkpoint_policy": "FAIL",
            "evaluation_boundary": "PASS_NOT_OPENED",
        },
        "blockers": blockers,
        "deferred_e1_checks": [
            "Verify the sealed bundle checksum and record first access in the E1 access log.",
            "Validate actual evaluation schema/quota/unique base-MDP aliases before any buffer is opened.",
            "Stop on any mismatch; do not regenerate, retune, drop a case, or inspect outcomes to repair the gate.",
        ],
        "reviewer_interpretation": "T1 and H1 are intact and D1 passed its narrower baseline preflight. Those facts do not satisfy T1's operational execution gate; the four implementation gaps make NO-GO mandatory.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-cli-probes", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists():
        parser.error("Output exists; preserve previous review evidence")
    report = review(run_cli_probes=not args.skip_cli_probes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
