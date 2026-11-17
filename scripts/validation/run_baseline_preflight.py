#!/usr/bin/env python
"""Run the Week 6 D1 baseline provenance and unified-runner preflight.

This command is intentionally a dry-run.  It validates the development data
contract, the common scoring/configuration contract, every project runner's
CLI, and any supplied external checkouts.  It never fits a learner and it
rejects a manifest containing evaluation records before opening a data buffer.

The output is an auditable bundle containing a machine-readable report and a
CSV provenance matrix.  The report separates checks that were run locally
from provenance that is recorded in the committed baseline audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH  # noqa: E402
from evaluation.io import reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from evaluation.training_protocol import (  # noqa: E402
    validate_training_manifest,
    verify_training_buffers,
)
from scripts.training.run_empirical_requirement import _training_seed_index  # noqa: E402


SCHEMA_VERSION = "w6-d1-baseline-preflight-v1"
DATASET_ID = "offline_rl_development_2026-08-22"
DEFAULT_BUDGET = 10
DEFAULT_TRAINING_SEEDS = (0,)

CLI_PROBES = (
    {
        "name": "tabular_unified_runner",
        "path": "scripts/training/run_empirical_requirement.py",
        "role": "registered CFQI/IQL/BC development runner",
    },
    {
        "name": "spibb_external_runner",
        "path": "scripts/external/run_spibb_development.py",
        "role": "Pi_b-SPIBB author-code adapter",
    },
    {
        "name": "vi_lcb_external_runner",
        "path": "scripts/external/run_vi_lcb_development.py",
        "role": "discounted VI-LCB paper-spec adapter",
    },
    {
        "name": "neural_external_runner",
        "path": "scripts/external/run_neural_discrete_development.py",
        "role": "author BCQ and pinned d3rlpy CQL adapter",
    },
)

METHODS = (
    {
        "method": "cfqi_tabular",
        "family": "registered_tabular",
        "source_type": "project_reference_implementation",
        "source_repository": "local project",
        "source_commit": "project HEAD",
        "runner": "scripts/training/run_empirical_requirement.py",
        "configuration": "100 iterations; tolerance 1e-10",
        "comparison_role": "primary same-data comparison",
    },
    {
        "method": "iql_tabular",
        "family": "registered_tabular",
        "source_type": "project_reference_implementation",
        "source_repository": "local project",
        "source_commit": "project HEAD",
        "runner": "scripts/training/run_empirical_requirement.py",
        "configuration": "expectile 0.7; inverse temperature 3.0; 100 iterations",
        "comparison_role": "primary same-data comparison",
    },
    {
        "method": "behavior_cloning_tabular",
        "family": "sanity_baseline",
        "source_type": "project_reference_implementation",
        "source_repository": "local project",
        "source_commit": "project HEAD",
        "runner": "scripts/training/run_empirical_requirement.py",
        "configuration": "logged-action frequency policy; not part of N_emp",
        "comparison_role": "sanity baseline",
    },
    {
        "method": "pib_spibb_n10_author",
        "family": "external_author",
        "source_type": "author_repository_unmodified_core",
        "source_repository": "https://github.com/RomainLaroche/SPIBB",
        "source_commit": "833f32b642759111907d2c4d086d1ae1ce9d84cb",
        "runner": "scripts/external/run_spibb_development.py",
        "configuration": "N_wedge=10; known behavior policy; exact project scorer",
        "comparison_role": "external tabular comparator",
    },
    {
        "method": "vi_lcb_cb005_author_empirical",
        "family": "external_paper_spec_port",
        "source_type": "author_repository_pinned_port",
        "source_repository": "https://github.com/Laixishi/Model-based-VI-LCB",
        "source_commit": "37858d392cf11a01ff5af4ef7815672ec0a36ca9",
        "runner": "scripts/external/run_vi_lcb_development.py",
        "configuration": "discounted Algorithm 1/equation (28); c_b=0.05",
        "comparison_role": "external tabular comparator",
    },
    {
        "method": "vi_lcb_cb144_theory",
        "family": "external_paper_spec_port",
        "source_type": "author_repository_pinned_port",
        "source_repository": "https://github.com/Laixishi/Model-based-VI-LCB",
        "source_commit": "37858d392cf11a01ff5af4ef7815672ec0a36ca9",
        "runner": "scripts/external/run_vi_lcb_development.py",
        "configuration": "discounted Algorithm 1/equation (28); c_b=144",
        "comparison_role": "external tabular comparator",
    },
    {
        "method": "discrete_bcq_author",
        "family": "external_neural",
        "source_type": "author_repository_unmodified_learner",
        "source_repository": "https://github.com/sfujim/BCQ",
        "source_commit": "4876f7e5afa9eb2981feec5daf67202514477518",
        "runner": "scripts/external/run_neural_discrete_development.py",
        "configuration": "one-hot; 2x256; batch 64; 500 updates; threshold 0.3",
        "comparison_role": "fixed-budget robustness panel",
    },
    {
        "method": "discrete_cql_d3rlpy",
        "family": "external_neural",
        "source_type": "pinned_d3rlpy_implementation",
        "source_repository": "https://github.com/takuseno/d3rlpy",
        "source_commit": "38c34b6f99fde82875e2ad81d1ba1f7a7da34137",
        "runner": "scripts/external/run_neural_discrete_development.py",
        "configuration": "one-hot; 2x256; batch 64; 500 updates; alpha 1.0",
        "comparison_role": "fixed-budget robustness panel",
    },
)

PROVENANCE_FIELDS = (
    "method",
    "family",
    "source_type",
    "source_repository",
    "source_commit",
    "runner",
    "configuration",
    "comparison_role",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _git_output(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_repo_state(path: Path | None, expected_commit: str) -> dict[str, Any]:
    if path is None:
        return {
            "status": "recorded_only",
            "path": None,
            "expected_commit": expected_commit,
            "observed_commit": None,
            "tracked_files_dirty": None,
            "note": "No local external checkout was supplied; committed provenance is retained.",
        }
    path = path.resolve()
    result: dict[str, Any] = {
        "status": "fail",
        "path": _display_path(path),
        "expected_commit": expected_commit,
        "observed_commit": None,
        "tracked_files_dirty": None,
    }
    if not path.is_dir():
        result["note"] = "Checkout directory does not exist."
        return result
    try:
        observed = _git_output(path, "rev-parse", "HEAD")
        dirty = bool(_git_output(path, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError) as exc:
        result["note"] = f"Git inspection failed: {exc}"
        return result
    result["observed_commit"] = observed
    result["tracked_files_dirty"] = dirty
    result["status"] = "pass" if observed == expected_commit and not dirty else "fail"
    if result["status"] == "fail":
        result["note"] = "Commit or tracked-file cleanliness does not match the pinned source."
    return result


def _probe_cli(path: Path) -> dict[str, Any]:
    command = [sys.executable, str(path), "--help"]
    result: dict[str, Any] = {
        "runner": str(path.relative_to(ROOT)),
        "command": command,
        "status": "fail",
        "returncode": None,
        "stdout_preview": "",
        "stderr_preview": "",
    }
    if not path.is_file():
        result["stderr_preview"] = "runner file does not exist"
        return result
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["stderr_preview"] = str(exc)
        return result
    result["returncode"] = completed.returncode
    result["stdout_preview"] = completed.stdout[:500]
    result["stderr_preview"] = completed.stderr[:500]
    result["status"] = "pass" if completed.returncode == 0 else "fail"
    return result


def _sealed_evaluation_guard(manifest: dict[str, Any]) -> dict[str, Any]:
    """Inspect metadata only and reject evaluation before opening buffers."""
    instances = manifest.get("instances", [])
    records = manifest.get("datasets", [])
    if not isinstance(instances, list) or not isinstance(records, list):
        raise ValueError("Manifest must contain instances and datasets lists.")
    non_development_instances = [
        str(item.get("instance_id", "<missing>"))
        for item in instances
        if item.get("split") != "development"
    ]
    non_development_records = [
        str(item.get("dataset_id", "<missing>"))
        for item in records
        if item.get("benchmark_split") != "development"
    ]
    if non_development_instances or non_development_records:
        raise ValueError(
            "Sealed-evaluation guard rejected non-development metadata before buffer access: "
            f"instances={non_development_instances[:3]}, records={non_development_records[:3]}"
        )
    return {
        "status": "pass",
        "evaluation_metadata_records": 0,
        "evaluation_buffers_opened": 0,
        "development_instances": len(instances),
        "development_records": len(records),
        "note": "Metadata-only guard passed; no evaluation path or buffer was opened.",
    }


def _dataset_preflight(
    dataset_dir: Path,
    *,
    budget: int,
    training_seeds: tuple[int, ...],
    limit_groups: int | None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = _read_json(manifest_path)

    # This must be first: it establishes the no-evaluation-access invariant
    # before verify_training_buffers hashes any physical buffer.
    guard = _sealed_evaluation_guard(manifest)
    contract_counts = validate_training_manifest(manifest)

    allowed_budgets = tuple(int(value) for value in PROTOCOL_CONFIG["collection"]["size_multipliers"])
    registered_seeds = tuple(int(value) for value in PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    if budget not in allowed_budgets:
        raise ValueError(f"Budget {budget} is not registered: {allowed_budgets}")
    if not set(training_seeds).issubset(set(registered_seeds)):
        raise ValueError(f"Training seeds {training_seeds} are not registered: {registered_seeds}")

    # Only after the metadata guard and full manifest contract pass do we hash
    # the development buffers.  No learner or fitting code is called here.
    verified_buffers = verify_training_buffers(dataset_dir, manifest)
    selected = [
        record
        for record in manifest["datasets"]
        if record.get("stream") == "training"
        and int(record.get("size_multiplier", -1)) == budget
        and _training_seed_index(record) in training_seeds
    ]
    groups = {(str(record["instance_id"]), str(record["policy_family"])) for record in selected}
    counts = Counter((str(record["instance_id"]), str(record["policy_family"])) for record in selected)
    expected_groups = int(contract_counts["instance_policy_groups"])
    if limit_groups is not None:
        if limit_groups <= 0:
            raise ValueError("--limit-groups must be positive when supplied")
        if len(groups) > limit_groups:
            groups = set(sorted(groups)[:limit_groups])
            selected = [
                record
                for record in selected
                if (str(record["instance_id"]), str(record["policy_family"])) in groups
            ]
            counts = Counter((str(record["instance_id"]), str(record["policy_family"])) for record in selected)
    expected_selected = len(groups) * len(training_seeds)
    if len(selected) != expected_selected or any(value != len(training_seeds) for value in counts.values()):
        raise ValueError(
            "Selected training grid is not exactly one prefix per group and seed: "
            f"records={len(selected)}, groups={len(groups)}, seeds={training_seeds}"
        )
    if limit_groups is None and len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} groups, found {len(groups)}")

    return {
        "status": "pass",
        "dataset_dir": _display_path(dataset_dir),
        "dataset_id": manifest.get("dataset_id", DATASET_ID),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_path": _display_path(manifest_path),
        "protocol_schema_version": manifest.get("protocol_schema_version"),
        "run_mode": manifest.get("run_mode"),
        "verified_physical_buffers": verified_buffers,
        "manifest_record_count": len(manifest["datasets"]),
        "selected_budget": budget,
        "selected_training_seeds": list(training_seeds),
        "selected_groups": len(groups),
        "selected_training_prefixes": len(selected),
        "selected_registered_tabular_runs": len(selected) * len(PROTOCOL_CONFIG["learning"]["algorithms"]),
        "selected_behavior_cloning_sanity_runs": len(selected),
        "contract_counts": contract_counts,
        "sealed_evaluation_guard": guard,
        "limit_groups": limit_groups,
    }


def _source_hashes() -> dict[str, str]:
    files = (
        "protocol_config.json",
        "config/protocol_config.py",
        "scripts/training/run_empirical_requirement.py",
        "scripts/external/run_spibb_development.py",
        "scripts/external/run_vi_lcb_development.py",
        "scripts/external/run_neural_discrete_development.py",
        "evaluation/training_protocol.py",
        "evaluation/io.py",
        "evaluation/metrics/contract.py",
        "docs/research/EXTERNAL_BASELINE_AUDIT_2026-09-19.md",
        "artifacts/research/final_all_methods_comparison_2026-09-20/REPORT.md",
    )
    return {name: sha256_file(ROOT / name) for name in files}


def _project_commit() -> str | None:
    try:
        return _git_output(ROOT, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None


def _display_path(path: Path) -> str:
    """Keep committed reports portable without hiding the evidence location."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve_external_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    root = args.external_root.resolve() if args.external_root else None
    values: dict[str, Path | None] = {}
    for name in ("spibb", "vi_lcb", "bcq", "d3rlpy"):
        explicit = getattr(args, f"{name}_dir")
        values[name] = explicit.resolve() if explicit else (root / name if root else None)
    return values


def _build_report(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    dataset = _dataset_preflight(
        args.dataset_dir,
        budget=args.budget,
        training_seeds=tuple(args.training_seeds),
        limit_groups=args.limit_groups,
    )
    cli = [_probe_cli(ROOT / item["path"]) for item in CLI_PROBES]
    external = _resolve_external_paths(args)
    checkout_checks = {
        "spibb": _git_repo_state(external["spibb"], METHODS[3]["source_commit"]),
        "vi_lcb": _git_repo_state(external["vi_lcb"], METHODS[4]["source_commit"]),
        "bcq": _git_repo_state(external["bcq"], METHODS[6]["source_commit"]),
        "d3rlpy": _git_repo_state(external["d3rlpy"], METHODS[7]["source_commit"]),
    }
    checkout_failures = [name for name, result in checkout_checks.items() if result["status"] == "fail"]
    cli_failures = [result["runner"] for result in cli if result["status"] != "pass"]
    recorded_only = [name for name, result in checkout_checks.items() if result["status"] == "recorded_only"]
    if cli_failures or checkout_failures:
        status = "fail"
    elif recorded_only:
        status = "pass_with_recorded_external_provenance"
    else:
        status = "pass"

    project_commit = _project_commit()
    methods = []
    for method in METHODS:
        row = dict(method)
        if row["source_commit"] == "project HEAD":
            row["source_commit"] = project_commit or "unavailable"
        methods.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "status": status,
        "dry_run": True,
        "fit_called": False,
        "sealed_evaluation_accessed": False,
        "scope": {
            "week": 6,
            "task": "D1 Baseline Provenance & Unified Runner Preflight",
            "owner": "Nguyễn Quốc Việt",
            "development_only": True,
            "evaluation_status": "sealed; not opened by this command",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "project_root": str(ROOT),
            "project_commit": project_commit,
            "protocol_config": _display_path(PROTOCOL_CONFIG_PATH),
        },
        "requested_contract": {
            "budget": args.budget,
            "training_seeds": list(args.training_seeds),
            "limit_groups": args.limit_groups,
            "normalization": "exact MDP normalized return; success threshold from protocol_config.json",
            "success_threshold": PROTOCOL_CONFIG["learning"]["success_normalized_return"],
            "registered_tabular_algorithms": list(PROTOCOL_CONFIG["learning"]["algorithms"]),
        },
        "checks": {
            "dataset": dataset,
            "runner_cli_help": cli,
            "external_checkout_validation": checkout_checks,
            "external_checkout_failures": checkout_failures,
            "external_checkout_recorded_only": recorded_only,
        },
        "provenance": {
            "methods": methods,
            "source_sha256": _source_hashes(),
            "primary_audit": "docs/research/EXTERNAL_BASELINE_AUDIT_2026-09-19.md",
            "primary_comparison_report": "artifacts/research/final_all_methods_comparison_2026-09-20/REPORT.md",
        },
        "interpretation": {
            "pass_means": "All local dry-run checks passed; no learner was fitted.",
            "recorded_only_means": "The pinned commit is documented, but that external checkout was not supplied to this invocation.",
            "neural_panel_limit": "One data/model seed and a fixed 500-update budget; exploratory robustness panel, not SOTA.",
            "next_gate": "E1 remains blocked until T1/H1/B2 and explicit evaluation approval are complete.",
        },
        "output_dir": _display_path(output_dir),
    }


def _write_report_markdown(output_dir: Path, report: dict[str, Any]) -> None:
    dataset = report["checks"]["dataset"]
    cli = report["checks"]["runner_cli_help"]
    external = report["checks"]["external_checkout_validation"]
    methods = report["provenance"]["methods"]
    lines = [
        "# D1 — Baseline Provenance & Unified Runner Preflight",
        "",
        f"Status: **{report['status']}**",
        "",
        "This is a development-only, no-fit preflight. It validates the shared data/protocol contract, runner CLI entry points, and pinned baseline provenance. It does not open evaluation data.",
        "",
        "## Result",
        "",
        f"- Dataset manifest: `{dataset['manifest_sha256']}`",
        f"- Development records checked: `{dataset['manifest_record_count']}`; physical buffers hashed: `{dataset['verified_physical_buffers']}`",
        f"- Selected panel: budget `{dataset['selected_budget']}|S||A|`, training seed(s) `{dataset['selected_training_seeds']}`, groups `{dataset['selected_groups']}`",
        f"- Selected prefixes: `{dataset['selected_training_prefixes']}`; registered tabular runs: `{dataset['selected_registered_tabular_runs']}`",
        f"- Sealed-evaluation guard: **{dataset['sealed_evaluation_guard']['status']}** — `{dataset['sealed_evaluation_guard']['evaluation_buffers_opened']}` evaluation buffers opened",
        f"- Learner fitting called: **{report['fit_called']}**",
        "",
        "## Provenance matrix",
        "",
        "| Method | Source type | Source repository | Pinned commit | Runner | Configuration |",
        "|---|---|---|---|---|---|",
    ]
    for method in methods:
        lines.append(
            f"| `{method['method']}` | {method['source_type']} | {method['source_repository']} | `{method['source_commit']}` | `{method['runner']}` | {method['configuration']} |"
        )
    lines.extend([
        "",
        "The full matrix is also available as `baseline_provenance.csv`.",
        "",
        "## Runner checks",
        "",
        "| Runner | CLI `--help` |",
        "|---|---|",
    ])
    for result in cli:
        lines.append(f"| `{result['runner']}` | **{result['status']}** |")
    lines.extend([
        "",
        "## External checkout checks",
        "",
        "| Checkout | Status | Observed commit |",
        "|---|---|---|",
    ])
    for name, result in external.items():
        lines.append(f"| `{name}` | **{result['status']}** | `{result.get('observed_commit') or '—'}` |")
    if report["checks"]["external_checkout_recorded_only"]:
        lines.extend([
            "",
            "`recorded_only` means that the exact commit is present in the committed provenance/audit and prior result artifacts, but the checkout was not supplied to this invocation. It is not treated as a clean-checkout pass.",
        ])
    lines.extend([
        "",
        "## Scope and remaining gate",
        "",
        "The preflight covers development data only. E1 must remain untouched: the evaluation split is sealed and can only be opened after the dependency gates and explicit approval recorded in W6 are satisfied.",
        "",
        "## Re-run",
        "",
        "```powershell",
        "python scripts/validation/run_baseline_preflight.py `",
        "  --dataset-dir <development-bundle> `",
        "  --output-dir artifacts/week6/d1_unified_preflight_YYYY-MM-DD",
        "```",
        "",
        "Generated files: `preflight_report.json`, `baseline_provenance.csv`, and this report.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=list(DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--spibb-dir", type=Path)
    parser.add_argument("--vi-lcb-dir", dest="vi_lcb_dir", type=Path)
    parser.add_argument("--bcq-dir", type=Path)
    parser.add_argument("--d3rlpy-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.output_dir.resolve()
    reserve_output_dir(output_dir)
    report = _build_report(args, output_dir)
    write_json(output_dir / "preflight_report.json", report)
    write_csv_rows(
        output_dir / "baseline_provenance.csv",
        report["provenance"]["methods"],
        PROVENANCE_FIELDS,
    )
    _write_report_markdown(output_dir, report)
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir)}, indent=2))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
