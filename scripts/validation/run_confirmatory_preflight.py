#!/usr/bin/env python
"""Preflight the frozen Week 6 confirmatory execution contract.

This command is deliberately safe to run before E1.  It rejects evaluation
metadata, validates the complete CFQI/IQL/BC seed and budget plan, exercises
the hash-verified missing-job runner on a fixture, and runs the paired
refit-bootstrap harness with the registered 2,000-draw default.  Development
mode can additionally call the registered train/score learners and the real
refit callback.  Neither mode opens sealed evaluation data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH  # noqa: E402
from evaluation.confirmatory import (  # noqa: E402
    CONFIRMATORY_OUTPUT_DIR_NAME,
    CONFIRMATORY_RUN_ID,
    DEFAULT_BASELINE_METHODS,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    build_job_plan,
    json_sha256,
    prepare_execution,
    run_missing_jobs_only,
    run_paired_refit_bootstrap,
)
from evaluation.development_confirmatory import (  # noqa: E402
    build_development_train_score_callback,
    load_development_refit_panel,
)
from evaluation.io import sha256_file, write_json  # noqa: E402
from evaluation.training_protocol import (  # noqa: E402
    validate_training_manifest,
    verify_training_buffers,
)
from scripts.validation.verify_week6_t1_freeze import verify as verify_t1_freeze  # noqa: E402
from scripts.training.run_empirical_requirement import _training_seed_index  # noqa: E402


T1_MANIFEST_REL = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"


def _fixture_records() -> list[dict[str, Any]]:
    """Create two logical groups with the full 10-seed/6-budget contract."""

    records: list[dict[str, Any]] = []
    sizes = tuple(int(value) for value in PROTOCOL_CONFIG["collection"]["size_multipliers"])
    seeds = tuple(int(value) for value in PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    for group_index, family in enumerate(("fixture_behavior_a", "fixture_behavior_b")):
        instance_id = f"development_fixture_{group_index:03d}"
        for size in sizes:
            for seed in seeds:
                file_name = f"fixture-{group_index}-{family}-x{size}-seed{seed}.npz"
                digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
                records.append(
                    {
                        "dataset_id": f"{instance_id}-{family}-x{size}-seed{seed}",
                        "instance_id": instance_id,
                        "policy_family": family,
                        "stream": "training",
                        "stream_seed": seed,
                        "training_seed": seed,
                        "size_multiplier": size,
                        "size": size,
                        "max_size": max(sizes),
                        "file": file_name,
                        "sha256": digest,
                    }
                )
    return records


def _fixture_bootstrap_callback(
    cluster_values: Mapping[str, tuple[float, float]],
):
    def fit_and_score(multiplicities: Mapping[str, int], _draw_index: int) -> Mapping[str, float]:
        denominator = float(sum(multiplicities.values()))
        if denominator <= 0:
            raise ValueError("fixture bootstrap sample is empty")
        selected = [
            (cluster_values[cluster], count)
            for cluster, count in multiplicities.items()
        ]
        return {
            "v4e": sum(values[0] * count for values, count in selected) / denominator,
            "fold_best_fixed": sum(values[1] * count for values, count in selected) / denominator,
        }

    return fit_and_score


def _fixture_bootstrap() -> tuple[dict[str, str], dict[str, float], Any]:
    values = {
        f"fixture_mdp_{index:02d}": (0.40 + index * 0.01, 0.38 + index * 0.009)
        for index in range(12)
    }
    strata = {
        cluster: ("outer_0" if index < 6 else "outer_1")
        for index, cluster in enumerate(sorted(values))
    }
    observed = {
        "v4e": sum(pair[0] for pair in values.values()) / len(values),
        "fold_best_fixed": sum(pair[1] for pair in values.values()) / len(values),
    }
    return strata, observed, _fixture_bootstrap_callback(values)


def _load_development(dataset_dir: Path) -> tuple[list[Mapping[str, Any]], dict[str, str], dict[str, Any]]:
    manifest_path = dataset_dir.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing development manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = manifest.get("instances", [])
    records = manifest.get("datasets", [])
    if any(item.get("split") != "development" for item in instances):
        raise ValueError("Confirmatory preflight refuses non-development instances.")
    if any(item.get("benchmark_split") != "development" for item in records):
        raise ValueError("Confirmatory preflight refuses non-development dataset records.")
    contract = validate_training_manifest(manifest)
    verified_buffers = verify_training_buffers(dataset_dir.resolve(), manifest)
    input_hashes = {"manifest.json": sha256_file(manifest_path)}
    for record in records:
        if record.get("stream") == "training":
            name = str(record["file"])
            input_hashes[f"buffer:{name}"] = str(record["sha256"]).lower()
    return list(records), input_hashes, {
        "scope": "development bundle contract preflight",
        "manifest_sha256": input_hashes["manifest.json"],
        "verified_physical_buffers": verified_buffers,
        "contract_counts": contract,
        "dataset_dir": str(dataset_dir.resolve()),
    }


def _source_hashes() -> dict[str, str]:
    manifest_path = ROOT / T1_MANIFEST_REL
    verification = verify_t1_freeze(ROOT, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        verification["version"] == "1.1.0"
        and payload.get("integrity_revision") == "r1"
    ):
        raise ValueError("Confirmatory preflight requires T1 manifest v1.1.0-r1")
    paths = tuple(str(path) for path in payload["files"])
    hashes = {path: sha256_file(ROOT / path) for path in paths}
    hashes[T1_MANIFEST_REL] = sha256_file(manifest_path)
    return hashes


def _subset_job_plan(plan: Mapping[str, Any], logical_prefixes: int | None) -> dict[str, Any]:
    """Select complete method-parallel cells for a bounded development probe."""

    if logical_prefixes is None:
        return dict(plan)
    if logical_prefixes <= 0:
        raise ValueError("train_score_job_limit must be positive")
    cells = sorted(
        {
            (
                str(row["instance_id"]),
                str(row["policy_family"]),
                int(row["size_multiplier"]),
                int(row["training_seed"]),
            )
            for row in plan["jobs"]
        }
    )[:logical_prefixes]
    wanted = set(cells)
    jobs = [
        dict(row)
        for row in plan["jobs"]
        if (
            str(row["instance_id"]),
            str(row["policy_family"]),
            int(row["size_multiplier"]),
            int(row["training_seed"]),
        ) in wanted
    ]
    jobs.sort(key=lambda row: str(row["job_key"]))
    result = dict(plan)
    result["jobs"] = jobs
    result["job_keys"] = [str(row["job_key"]) for row in jobs]
    result["groups"] = [
        {"instance_id": instance_id, "policy_family": family}
        for instance_id, family in sorted({(cell[0], cell[1]) for cell in cells})
    ]
    result["counts"] = {
        "groups": len(result["groups"]),
        "logical_prefixes": len(cells),
        "jobs": len(jobs),
        "jobs_by_method": {
            method: sum(str(row["method"]) == method for row in jobs)
            for method in plan["methods"]
        },
    }
    contract = dict(plan.get("contract", {}))
    contract["execution_scope"] = "bounded_development_train_score_probe"
    result["contract"] = contract
    return result


def _identity_job_runner(job: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fixture/development contract runner; it intentionally does not score."""

    return {
        "status": "fixture_contract_passed",
        "method": str(job["method"]),
        "buffer_sha256": str(job["buffer_sha256"]),
        "size_multiplier": int(job["size_multiplier"]),
        "training_seed": int(job["training_seed"]),
        "evaluation_opened": False,
    }


def run_preflight(
    *,
    output_dir: Path,
    dataset_dir: Path | None = None,
    fixture: bool = False,
    execute_jobs: bool = False,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resume: bool = False,
    real_bootstrap: bool = False,
    panel_dir: Path | None = None,
    train_score_job_limit: int | None = None,
) -> dict[str, Any]:
    if fixture == (dataset_dir is not None):
        raise ValueError("Choose exactly one of --fixture or --dataset-dir.")
    if not fixture and draws != DEFAULT_BOOTSTRAP_DRAWS:
        raise ValueError("Development confirmatory preflight is frozen at 2,000 draws.")
    if fixture and (real_bootstrap or panel_dir is not None or train_score_job_limit is not None):
        raise ValueError("Real development options cannot be combined with --fixture.")

    if fixture:
        records = _fixture_records()
        input_hashes = {"fixture_records": json_sha256(records)}
        data_scope = {
            "scope": "fixture-only execution contract",
            "manifest_sha256": input_hashes["fixture_records"],
            "verified_physical_buffers": 0,
        }
    else:
        records, input_hashes, data_scope = _load_development(dataset_dir)  # type: ignore[arg-type]

    plan = build_job_plan(
        records,
        methods=DEFAULT_BASELINE_METHODS,
        training_seeds=tuple(PROTOCOL_CONFIG["collection"]["training_seed_indices"]),
        size_multipliers=tuple(PROTOCOL_CONFIG["collection"]["size_multipliers"]),
        run_id=CONFIRMATORY_RUN_ID,
        seed_index_fn=None if fixture else _training_seed_index,
    )
    source_hashes = _source_hashes()
    execution_plan = _subset_job_plan(plan, train_score_job_limit) if not fixture else plan
    if execute_jobs or fixture:
        if fixture:
            job_runner = _identity_job_runner
        else:
            job_runner = build_development_train_score_callback(
                dataset_dir.resolve(), records
            )  # type: ignore[union-attr]
        job_summary = run_missing_jobs_only(
            output_dir,
            execution_plan,
            job_runner,
            source_hashes=source_hashes,
            input_hashes=input_hashes,
            resume=resume,
        )
        if execution_plan is not plan:
            job_summary["execution_scope"] = execution_plan["contract"]["execution_scope"]
            job_summary["planned_full_grid_jobs"] = len(plan["jobs"])
    else:
        prepare_execution(
            output_dir,
            plan,
            source_hashes=source_hashes,
            input_hashes=input_hashes,
            resume=resume,
        )
        job_summary = {
            "status": "plan_only",
            "total_jobs": len(plan["jobs"]),
            "completed_jobs": 0,
            "missing_jobs": len(plan["jobs"]),
            "stop_on_error": True,
        }
        write_json(
            output_dir / "checkpoint_ledger.json",
            {
                "schema_version": "w6-confirmatory-execution-v1",
                "run_id": CONFIRMATORY_RUN_ID,
                "plan_sha256": json_sha256(plan),
                "jobs": {},
            },
        )
        write_json(
            output_dir / "execution_summary.json",
            {
                "schema_version": "w6-confirmatory-execution-v1",
                "run_id": CONFIRMATORY_RUN_ID,
                "status": "plan_only",
                "stop_on_error": True,
                "total_jobs": len(plan["jobs"]),
                "completed_jobs": 0,
                "missing_jobs": len(plan["jobs"]),
                "runtime_estimate_basis": "no_jobs_executed",
                "estimated_total_runtime_seconds": None,
                "peak_tracemalloc_bytes": 0,
                "memory_measurement": "tracemalloc_peak_python_heap_bytes",
            },
        )

    if fixture or not real_bootstrap:
        strata, observed, fit_and_score = _fixture_bootstrap()
        bootstrap_mode = "fixture_contract"
        bootstrap_source_hashes: dict[str, str] = {}
    else:
        panel = load_development_refit_panel(root=ROOT, panel_dir=panel_dir)
        strata = dict(panel.cluster_strata)
        observed = dict(panel.observed)
        fit_and_score = panel.fit_and_score
        bootstrap_mode = "development_refit"
        bootstrap_source_hashes = dict(panel.source_hashes)
        input_hashes.update(
            {f"refit_panel:{key}": value for key, value in panel.source_hashes.items()}
        )
    bootstrap = run_paired_refit_bootstrap(
        cluster_strata=strata,
        observed=observed,
        fit_and_score=fit_and_score,
        draws=draws,
        seed=seed,
        checkpoint_path=output_dir / "paired_refit_bootstrap.json",
        pairwise_margins={"v4e_minus_fold_best_fixed": 0.005},
        checkpoint_context={
            "data_scope": data_scope["scope"],
            "bootstrap_mode": bootstrap_mode,
            "bootstrap_source_sha256": bootstrap_source_hashes,
            "protocol_config_sha256": sha256_file(PROTOCOL_CONFIG_PATH),
        },
        resume=resume,
    )
    report = {
        "schema_version": "w6-confirmatory-preflight-report-v1",
        "status": "passed",
        "run_id": CONFIRMATORY_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_opened": False,
        "oracle_imported": False,
        "scope": data_scope,
        "plan": {
            "sha256": json_sha256(plan),
            "methods": plan["methods"],
            "training_seeds": plan["training_seeds"],
            "size_multipliers": plan["size_multipliers"],
            "counts": plan["counts"],
            "same_buffer_across_methods": True,
            "same_budget_across_methods": True,
        },
        "execution_plan": {
            "sha256": json_sha256(execution_plan),
            "counts": execution_plan["counts"],
            "scope": execution_plan.get("contract", {}).get("execution_scope", "full_grid"),
        },
        "job_execution": job_summary,
        "paired_refit_bootstrap": bootstrap,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_provenance": {
            "clusters": len(strata),
            "source_sha256": bootstrap_source_hashes,
        },
        "runtime_memory": {
            "job_runtime_seconds": job_summary.get("runtime_seconds", 0.0),
            "job_estimated_total_runtime_seconds": job_summary.get(
                "estimated_total_runtime_seconds"
            ),
            "job_runtime_estimate_basis": job_summary.get("runtime_estimate_basis"),
            "job_peak_tracemalloc_bytes": job_summary.get("peak_tracemalloc_bytes", 0),
            "bootstrap_runtime_seconds": bootstrap["runtime_seconds"],
            "bootstrap_estimated_runtime_seconds": bootstrap.get(
                "estimated_runtime_seconds"
            ),
            "bootstrap_runtime_estimate_basis": bootstrap.get(
                "runtime_estimate_basis"
            ),
            "bootstrap_peak_tracemalloc_bytes": bootstrap["peak_tracemalloc_bytes"],
            "memory_measurement": "tracemalloc_peak_python_heap_bytes",
        },
        "stop_rule": "stop on first job/draw error; preserve verified partial checkpoints",
        "resume_rule": "resume only when run ID, plan, source/input hashes and completed output hashes match",
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "artifact_manifest": "artifact_manifest.json",
    }
    write_json(output_dir / "preflight_report.json", report)
    output_names = (
        "execution_plan.json",
        "run_metadata.json",
        "checkpoint_ledger.json",
        "execution_summary.json",
        "paired_refit_bootstrap.json",
        "bootstrap_summary.json",
        "preflight_report.json",
    )
    write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "w6-confirmatory-artifact-manifest-v1",
            "run_id": CONFIRMATORY_RUN_ID,
            "outputs": {name: sha256_file(output_dir / name) for name in output_names},
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", action="store_true")
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "week6" / CONFIRMATORY_OUTPUT_DIR_NAME,
    )
    parser.add_argument("--execute-jobs", action="store_true")
    parser.add_argument(
        "--train-score-job-limit",
        type=int,
        help="Development probe only: execute this many complete method-parallel logical cells.",
    )
    parser.add_argument(
        "--real-bootstrap",
        action="store_true",
        help="Development only: refit the frozen return selector on every bootstrap draw.",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        help="Root containing the tracked development refit panel directories.",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.draws <= 0:
        parser.error("--draws must be positive")
    run_preflight(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        fixture=args.fixture,
        execute_jobs=args.execute_jobs,
        draws=args.draws,
        seed=args.seed,
        resume=args.resume,
        real_bootstrap=args.real_bootstrap,
        panel_dir=args.panel_dir,
        train_score_job_limit=args.train_score_job_limit,
    )
    print(json.dumps({"status": "passed", "run_id": CONFIRMATORY_RUN_ID}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
