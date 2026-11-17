"""Integrity-first execution helpers for the Week 6 confirmatory gate.

The public development runners deliberately remain evaluation-safe.  This
module contains the smaller, testable pieces that the confirmatory runner must
not re-implement ad hoc: a frozen job plan, hash-verified missing-job resume,
and a paired cluster refit bootstrap with a hard stop on the first error.

The bootstrap callback is intentionally injected.  The statistical procedure
owns the refit; this module owns the sampling, pairing, checkpointing and
execution evidence.  That separation lets the exact same control path be
tested on a fixture without opening the sealed evaluation cohort.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time
import traceback
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.io import sha256_file, sha256_text, write_json


CONFIRMATORY_RUN_ID = "week6_oehv4e_confirmatory_v1_1"
CONFIRMATORY_OUTPUT_DIR_NAME = "oehv4e_confirmatory_v1_1"
CONFIRMATORY_SCHEMA_VERSION = "w6-confirmatory-execution-v1"
BOOTSTRAP_SCHEMA_VERSION = "w6-paired-refit-bootstrap-v1"
DEFAULT_BOOTSTRAP_DRAWS = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_921
DEFAULT_BASELINE_METHODS = (
    "cfqi_tabular",
    "iql_tabular",
    "behavior_cloning_tabular",
)


class ConfirmatoryIntegrityError(ValueError):
    """Raised when a plan, checkpoint, input or output hash is inconsistent."""


class BootstrapExecutionError(RuntimeError):
    """Raised after a paired refit bootstrap stops on a draw failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical representation."""

    return sha256_text(_canonical_json(value))


def _record_seed_index(
    record: Mapping[str, Any], seed_index_fn: Callable[[Mapping[str, Any]], int] | None
) -> int:
    for name in ("training_seed_index", "training_seed", "seed"):
        if name in record:
            return int(record[name])
    if seed_index_fn is None:
        raise ValueError(
            "Training record has no training seed index and no seed_index_fn was supplied."
        )
    return int(seed_index_fn(record))


def _job_key(
    method: str,
    instance_id: str,
    policy_family: str,
    size_multiplier: int,
    training_seed: int,
) -> str:
    return (
        f"{method}|{instance_id}|{policy_family}|"
        f"x{size_multiplier}|seed{training_seed}"
    )


def build_job_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = DEFAULT_BASELINE_METHODS,
    training_seeds: Sequence[int] = tuple(range(10)),
    size_multipliers: Sequence[int] = (1, 2, 5, 10, 20, 50),
    run_id: str = CONFIRMATORY_RUN_ID,
    seed_index_fn: Callable[[Mapping[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """Build and validate the complete paired CFQI/IQL/BC job plan.

    A method is not allowed to choose a different buffer or budget: methods
    are expanded from the same manifest record, and each logical
    ``(instance, behaviour, size, seed)`` cell must exist exactly once.
    """

    methods = tuple(str(value) for value in methods)
    training_seeds = tuple(int(value) for value in training_seeds)
    size_multipliers = tuple(int(value) for value in size_multipliers)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training_seeds must be non-empty and unique")
    if not size_multipliers or len(set(size_multipliers)) != len(size_multipliers):
        raise ValueError("size_multipliers must be non-empty and unique")

    expected_seed_set = set(training_seeds)
    expected_size_set = set(size_multipliers)
    groups: dict[tuple[str, str], dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    for raw in records:
        if raw.get("stream", "training") != "training":
            continue
        instance_id = str(raw["instance_id"])
        policy_family = str(raw["policy_family"])
        size = int(raw["size_multiplier"])
        seed = _record_seed_index(raw, seed_index_fn)
        if seed not in expected_seed_set:
            raise ValueError(
                f"Unregistered training seed {seed} in {instance_id}/{policy_family}."
            )
        if size not in expected_size_set:
            raise ValueError(
                f"Unregistered size multiplier {size} in {instance_id}/{policy_family}."
            )
        key = (size, seed)
        group = groups[(instance_id, policy_family)]
        if key in group:
            raise ValueError(
                f"Duplicate logical training cell {instance_id}/{policy_family}/{key}."
            )
        digest = str(raw.get("sha256", ""))
        file_name = str(raw.get("file", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Missing or invalid buffer SHA-256 for {instance_id}/{policy_family}/{key}.")
        if not file_name:
            raise ValueError(f"Missing buffer file for {instance_id}/{policy_family}/{key}.")
        group[key] = raw

    if not groups:
        raise ValueError("No training records were supplied.")

    expected_cells = {(size, seed) for size in size_multipliers for seed in training_seeds}
    jobs: list[dict[str, Any]] = []
    for (instance_id, policy_family), group in sorted(groups.items()):
        actual = set(group)
        if actual != expected_cells:
            raise ValueError(
                f"{instance_id}/{policy_family} job grid mismatch; "
                f"missing={sorted(expected_cells - actual)}, extra={sorted(actual - expected_cells)}."
            )
        for size, seed in sorted(expected_cells):
            record = group[(size, seed)]
            shared = {
                "instance_id": instance_id,
                "policy_family": policy_family,
                "size_multiplier": size,
                "training_seed": seed,
                "stream_seed": int(record.get("stream_seed", seed)),
                "buffer_file": str(record["file"]),
                "buffer_sha256": str(record["sha256"]).lower(),
                "buffer_size": int(record.get("size", 0)),
                "max_size": int(record.get("max_size", record.get("size", 0))),
            }
            for method in methods:
                key = _job_key(method, instance_id, policy_family, size, seed)
                jobs.append({"job_key": key, "method": method, **shared})

    jobs.sort(key=lambda row: row["job_key"])
    plan = {
        "schema_version": CONFIRMATORY_SCHEMA_VERSION,
        "run_id": run_id,
        "methods": list(methods),
        "training_seeds": list(training_seeds),
        "size_multipliers": list(size_multipliers),
        "groups": [
            {"instance_id": instance_id, "policy_family": family}
            for instance_id, family in sorted(groups)
        ],
        "jobs": jobs,
        "job_keys": [row["job_key"] for row in jobs],
        "counts": {
            "groups": len(groups),
            "logical_prefixes": len(groups) * len(expected_cells),
            "jobs": len(jobs),
            "jobs_by_method": {method: len(jobs) // len(methods) for method in methods},
        },
        "contract": {
            "same_buffer_across_methods": True,
            "same_budget_across_methods": True,
            "seed_grid_complete": True,
            "missing_jobs_only_resume": True,
        },
    }
    validate_job_plan(plan)
    return plan


def validate_job_plan(plan: Mapping[str, Any]) -> None:
    """Validate plan identity, complete seed/budget cells and method parity."""

    if plan.get("schema_version") != CONFIRMATORY_SCHEMA_VERSION:
        raise ConfirmatoryIntegrityError("Unsupported or missing execution-plan schema.")
    if plan.get("run_id") != CONFIRMATORY_RUN_ID:
        raise ConfirmatoryIntegrityError("Execution plan run_id does not match the frozen run ID.")
    methods = tuple(str(value) for value in plan.get("methods", ()))
    seeds = tuple(int(value) for value in plan.get("training_seeds", ()))
    sizes = tuple(int(value) for value in plan.get("size_multipliers", ()))
    jobs = list(plan.get("jobs", ()))
    keys = [str(row.get("job_key", "")) for row in jobs]
    if not methods or len(set(methods)) != len(methods):
        raise ConfirmatoryIntegrityError("Execution plan methods are empty or duplicated.")
    if tuple(sorted(keys)) != tuple(plan.get("job_keys", ())):
        raise ConfirmatoryIntegrityError("Execution plan job_keys are not canonical.")
    if len(keys) != len(set(keys)):
        raise ConfirmatoryIntegrityError("Execution plan contains duplicate job keys.")
    if not seeds or not sizes:
        raise ConfirmatoryIntegrityError("Execution plan has no registered seeds or budgets.")

    by_cell: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in jobs:
        method = str(row.get("method", ""))
        if method not in methods:
            raise ConfirmatoryIntegrityError(f"Unknown method in job plan: {method}")
        cell = (
            str(row.get("instance_id", "")),
            str(row.get("policy_family", "")),
            int(row.get("size_multiplier", -1)),
            int(row.get("training_seed", -1)),
        )
        if cell[2] not in sizes or cell[3] not in seeds:
            raise ConfirmatoryIntegrityError(f"Job outside frozen seed/budget grid: {cell}")
        if len(str(row.get("buffer_sha256", ""))) != 64 or not str(row.get("buffer_file", "")):
            raise ConfirmatoryIntegrityError(f"Job has incomplete buffer identity: {row.get('job_key')}")
        by_cell[cell].append(row)

    expected_methods = set(methods)
    for cell, rows in by_cell.items():
        if {str(row["method"]) for row in rows} != expected_methods:
            raise ConfirmatoryIntegrityError(f"Methods do not share complete cell {cell}.")
        hashes = {str(row["buffer_sha256"]) for row in rows}
        files = {str(row["buffer_file"]) for row in rows}
        sizes_seen = {int(row.get("size_multiplier", -1)) for row in rows}
        if len(hashes) != 1 or len(files) != 1 or sizes_seen != {cell[2]}:
            raise ConfirmatoryIntegrityError(f"Methods do not share buffer/budget for {cell}.")

    counts = plan.get("counts", {})
    if int(counts.get("jobs", -1)) != len(jobs):
        raise ConfirmatoryIntegrityError("Execution plan job count is inconsistent.")
    for method in methods:
        if int(counts.get("jobs_by_method", {}).get(method, -1)) != sum(
            row["method"] == method for row in jobs
        ):
            raise ConfirmatoryIntegrityError(f"Execution plan count is inconsistent for {method}.")


def plan_sha256(plan: Mapping[str, Any]) -> str:
    validate_job_plan(plan)
    return json_sha256(plan)


def _job_filename(job_key: str) -> str:
    return f"{sha256_text(job_key)[:24]}.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmatoryIntegrityError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfirmatoryIntegrityError(f"Expected a JSON object in {path}.")
    return value


def _measure(call: Callable[[], Any]) -> tuple[Any, float, int]:
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        value = call()
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, elapsed, int(peak)


def _execution_metadata(
    plan: Mapping[str, Any], source_hashes: Mapping[str, str], input_hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": CONFIRMATORY_SCHEMA_VERSION,
        "run_id": CONFIRMATORY_RUN_ID,
        "plan_sha256": plan_sha256(plan),
        "source_sha256": dict(sorted((str(k), str(v)) for k, v in source_hashes.items())),
        "input_sha256": dict(sorted((str(k), str(v)) for k, v in input_hashes.items())),
    }


def prepare_execution(
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str] | None = None,
    input_hashes: Mapping[str, str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Create a run or verify every identity field before a resume."""

    validate_job_plan(plan)
    if output_dir.name != CONFIRMATORY_OUTPUT_DIR_NAME:
        raise ConfirmatoryIntegrityError(
            f"Output directory must be named {CONFIRMATORY_OUTPUT_DIR_NAME!r}; got {output_dir.name!r}."
        )
    metadata = _execution_metadata(plan, source_hashes or {}, input_hashes or {})
    plan_path = output_dir / "execution_plan.json"
    metadata_path = output_dir / "run_metadata.json"
    if output_dir.exists():
        has_content = any(output_dir.iterdir())
        if has_content and not resume:
            raise FileExistsError(
                f"Refusing non-empty confirmatory output without --resume: {output_dir}"
            )
        if resume:
            if not has_content or not plan_path.is_file() or not metadata_path.is_file():
                raise ConfirmatoryIntegrityError("Verified resume requires run metadata and execution plan.")
            saved_metadata = _read_json(metadata_path)
            if saved_metadata != metadata:
                raise ConfirmatoryIntegrityError("Run/source/config/input hash mismatch on resume.")
            saved_plan = _read_json(plan_path)
            if saved_plan.get("plan_sha256") != metadata["plan_sha256"]:
                raise ConfirmatoryIntegrityError("Execution plan hash mismatch on resume.")
            saved_plan_value = saved_plan.get("plan")
            if not isinstance(saved_plan_value, dict) or plan_sha256(saved_plan_value) != metadata["plan_sha256"]:
                raise ConfirmatoryIntegrityError("Saved execution plan contents do not match its hash.")
            return metadata
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume missing output directory: {output_dir}")
        output_dir.mkdir(parents=True)

    write_json(plan_path, {"plan_sha256": metadata["plan_sha256"], "plan": plan})
    write_json(metadata_path, metadata)
    (output_dir / "jobs").mkdir(parents=True, exist_ok=True)
    return metadata


def _load_ledger(output_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path = output_dir / "checkpoint_ledger.json"
    if not path.exists():
        return {
            "schema_version": CONFIRMATORY_SCHEMA_VERSION,
            "run_id": CONFIRMATORY_RUN_ID,
            "plan_sha256": metadata["plan_sha256"],
            "jobs": {},
        }
    ledger = _read_json(path)
    if (
        ledger.get("schema_version") != CONFIRMATORY_SCHEMA_VERSION
        or ledger.get("run_id") != CONFIRMATORY_RUN_ID
        or ledger.get("plan_sha256") != metadata["plan_sha256"]
    ):
        raise ConfirmatoryIntegrityError("Checkpoint ledger identity mismatch.")
    if not isinstance(ledger.get("jobs"), dict):
        raise ConfirmatoryIntegrityError("Checkpoint ledger jobs must be an object.")
    return ledger


def _verify_completed_jobs(
    output_dir: Path, plan: Mapping[str, Any], ledger: Mapping[str, Any]
) -> set[str]:
    expected = set(str(key) for key in plan["job_keys"])
    expected_jobs = {str(row["job_key"]): row for row in plan["jobs"]}
    jobs = ledger.get("jobs", {})
    unknown = set(jobs) - expected
    if unknown:
        raise ConfirmatoryIntegrityError(f"Checkpoint contains unknown job keys: {sorted(unknown)[:3]}")
    completed: set[str] = set()
    jobs_dir = output_dir / "jobs"

    def verify_output(path: Path, key: str) -> str:
        if path.name != _job_filename(key):
            raise ConfirmatoryIntegrityError(f"Checkpoint path mismatch for {key}.")
        payload = _read_json(path)
        if payload.get("run_id") != CONFIRMATORY_RUN_ID or payload.get("job_key") != key:
            raise ConfirmatoryIntegrityError(f"Output identity mismatch for completed job {key}.")
        payload_body = {
            field: value for field, value in payload.items() if field != "payload_sha256"
        }
        if payload.get("payload_sha256") != json_sha256(payload_body):
            raise ConfirmatoryIntegrityError(f"Output payload hash mismatch for {key}.")
        if payload.get("job_sha256") != json_sha256(expected_jobs[key]):
            raise ConfirmatoryIntegrityError(f"Output job identity mismatch for {key}.")
        return sha256_file(path)

    for key, entry in jobs.items():
        if not isinstance(entry, dict):
            raise ConfirmatoryIntegrityError(f"Malformed checkpoint entry for {key}.")
        expected_name = f"jobs/{_job_filename(key)}"
        if entry.get("file") != expected_name:
            raise ConfirmatoryIntegrityError(f"Checkpoint path mismatch for {key}.")
        path = output_dir / expected_name
        if not path.is_file():
            raise ConfirmatoryIntegrityError(f"Checkpoint output missing for {key}.")
        observed_hash = sha256_file(path)
        if observed_hash != str(entry.get("sha256", "")):
            raise ConfirmatoryIntegrityError(f"Output hash mismatch for completed job {key}.")
        verify_output(path, key)
        completed.add(key)

    # A job output can be durable while the batch ledger write is interrupted.
    # Reconcile such outputs by their self-identified job key and content hash;
    # this preserves missing-jobs-only resume without rewriting the whole ledger
    # after every job.
    if jobs_dir.exists():
        for path in sorted(jobs_dir.glob("*.json")):
            payload = _read_json(path)
            key = str(payload.get("job_key", ""))
            if key not in expected:
                raise ConfirmatoryIntegrityError(
                    f"Unexpected unregistered checkpoint file: {path.name}"
                )
            observed_hash = verify_output(path, key)
            entry = jobs.get(key)
            if entry is None:
                jobs[key] = {
                    "file": f"jobs/{path.name}",
                    "sha256": observed_hash,
                    "runtime_seconds": float(payload.get("runtime_seconds", 0.0)),
                    "peak_tracemalloc_bytes": int(payload.get("peak_tracemalloc_bytes", 0)),
                }
            elif str(entry.get("sha256", "")) != observed_hash:
                raise ConfirmatoryIntegrityError(f"Output hash mismatch for completed job {key}.")
            completed.add(key)
    return completed


def run_missing_jobs_only(
    output_dir: Path,
    plan: Mapping[str, Any],
    job_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str] | None = None,
    input_hashes: Mapping[str, str] | None = None,
    resume: bool = False,
    checkpoint_interval: int = 25,
) -> dict[str, Any]:
    """Run only missing jobs and stop immediately on the first job error."""

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    metadata = prepare_execution(
        output_dir,
        plan,
        source_hashes=source_hashes or {},
        input_hashes=input_hashes or {},
        resume=resume,
    )
    ledger = _load_ledger(output_dir, metadata)
    completed = _verify_completed_jobs(output_dir, plan, ledger)
    total = len(plan["jobs"])
    for job in plan["jobs"]:
        key = str(job["job_key"])
        if key in completed:
            continue
        try:
            result, elapsed, peak_memory = _measure(lambda job=job: job_runner(job))
            if not isinstance(result, Mapping):
                raise TypeError("job_runner must return a JSON object")
            payload_body = {
                "schema_version": CONFIRMATORY_SCHEMA_VERSION,
                "run_id": CONFIRMATORY_RUN_ID,
                "job_key": key,
                "job_sha256": json_sha256(job),
                "result": dict(result),
                "runtime_seconds": elapsed,
                "peak_tracemalloc_bytes": peak_memory,
                "memory_measurement": "tracemalloc_peak_python_heap_bytes",
            }
            payload = {**payload_body, "payload_sha256": json_sha256(payload_body)}
            path = output_dir / "jobs" / _job_filename(key)
            write_json(path, payload)
            ledger["jobs"][key] = {
                "file": f"jobs/{path.name}",
                "sha256": sha256_file(path),
                "runtime_seconds": elapsed,
                "peak_tracemalloc_bytes": peak_memory,
            }
            completed.add(key)
            if len(completed) % checkpoint_interval == 0:
                write_json(output_dir / "checkpoint_ledger.json", ledger)
        except Exception as exc:
            failure = {
                "schema_version": CONFIRMATORY_SCHEMA_VERSION,
                "run_id": CONFIRMATORY_RUN_ID,
                "job_key": key,
                "status": "failed_stop_on_error",
                "completed_jobs": len(completed),
                "total_jobs": total,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "created_at_utc": _utc_now(),
            }
            write_json(output_dir / "failure.json", failure)
            raise

    summary = {
        "schema_version": CONFIRMATORY_SCHEMA_VERSION,
        "run_id": CONFIRMATORY_RUN_ID,
        "status": "complete",
        "stop_on_error": True,
        "total_jobs": total,
        "completed_jobs": len(completed),
        "missing_jobs": total - len(completed),
        "resumed": resume,
        "checkpoint_interval": checkpoint_interval,
        "runtime_seconds": sum(
            float(value.get("runtime_seconds", 0.0))
            for value in ledger["jobs"].values()
        ),
        "mean_job_runtime_seconds": (
            sum(float(value.get("runtime_seconds", 0.0)) for value in ledger["jobs"].values())
            / len(ledger["jobs"])
            if ledger["jobs"]
            else None
        ),
        "estimated_total_runtime_seconds": (
            sum(float(value.get("runtime_seconds", 0.0)) for value in ledger["jobs"].values())
            / len(ledger["jobs"])
            * total
            if ledger["jobs"]
            else None
        ),
        "runtime_estimate_basis": "mean_completed_job_runtime_x_total_jobs",
        "peak_tracemalloc_bytes": max(
            (int(value.get("peak_tracemalloc_bytes", 0)) for value in ledger["jobs"].values()),
            default=0,
        ),
        "memory_measurement": "tracemalloc_peak_python_heap_bytes",
    }
    write_json(output_dir / "checkpoint_ledger.json", ledger)
    write_json(output_dir / "execution_summary.json", summary)
    return summary


def _validate_bootstrap_checkpoint(
    checkpoint: Mapping[str, Any], metadata: Mapping[str, Any], draws: int
) -> dict[int, dict[str, Any]]:
    if checkpoint.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        raise ConfirmatoryIntegrityError("Unsupported bootstrap checkpoint schema.")
    if checkpoint.get("metadata_sha256") != json_sha256(metadata):
        raise ConfirmatoryIntegrityError("Bootstrap checkpoint metadata hash mismatch.")
    raw_draws = checkpoint.get("draws", {})
    if not isinstance(raw_draws, dict):
        raise ConfirmatoryIntegrityError("Bootstrap checkpoint draws must be an object.")
    result: dict[int, dict[str, Any]] = {}
    for raw_index, raw_record in raw_draws.items():
        index = int(raw_index)
        if index < 0 or index >= draws or not isinstance(raw_record, dict):
            raise ConfirmatoryIntegrityError("Bootstrap checkpoint contains an invalid draw index.")
        saved_hash = raw_record.get("sha256")
        body = {key: value for key, value in raw_record.items() if key != "sha256"}
        if saved_hash != json_sha256(body):
            raise ConfirmatoryIntegrityError(f"Bootstrap draw hash mismatch at draw {index}.")
        if int(body.get("draw_index", -1)) != index:
            raise ConfirmatoryIntegrityError(f"Bootstrap draw identity mismatch at draw {index}.")
        result[index] = raw_record
    return result


def run_paired_refit_bootstrap(
    *,
    cluster_strata: Mapping[str, str | int],
    observed: Mapping[str, float],
    fit_and_score: Callable[[Mapping[str, int], int], Mapping[str, float]],
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    checkpoint_path: Path,
    checkpoint_context: Mapping[str, Any] | None = None,
    pairwise_margins: Mapping[str, float] | None = None,
    resume: bool = False,
    checkpoint_interval: int = 25,
) -> dict[str, Any]:
    """Run paired refit-bootstrap draws with deterministic per-draw resume.

    The callback receives cluster multiplicities and must refit the complete
    meta-procedure once, then return every endpoint from that same fit.  One
    cluster sample is therefore paired across all endpoints.  A numerical,
    schema or callback error stops the run immediately and leaves the last
    verified checkpoint on disk.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if not cluster_strata:
        raise ValueError("cluster_strata must be nonempty")
    raw_endpoint_names = {str(name) for name in observed}
    preferred_endpoint_order = ("v4e", "fold_best_fixed")
    endpoint_names = tuple(
        name
        for name in preferred_endpoint_order
        if name in raw_endpoint_names
    ) + tuple(sorted(raw_endpoint_names - set(preferred_endpoint_order)))
    if len(endpoint_names) < 2:
        raise ValueError("paired bootstrap requires at least two endpoints")
    observed_values = {name: float(observed[name]) for name in endpoint_names}
    if not all(np.isfinite(value) for value in observed_values.values()):
        raise ValueError("observed endpoint values must be finite")
    margin_values = {
        str(key): float(value)
        for key, value in sorted((pairwise_margins or {}).items())
    }
    if any(not np.isfinite(value) or value < 0.0 for value in margin_values.values()):
        raise ValueError("pairwise margins must be finite and non-negative")
    strata: dict[str, list[str]] = defaultdict(list)
    for cluster, stratum in cluster_strata.items():
        strata[str(stratum)].append(str(cluster))
    for members in strata.values():
        members.sort()
    metadata = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "run_id": CONFIRMATORY_RUN_ID,
        "draws": int(draws),
        "seed": int(seed),
        "rng": "numpy.random.PCG64",
        "endpoint_names": list(endpoint_names),
        "cluster_strata": {str(k): str(v) for k, v in sorted(cluster_strata.items())},
        "resampling_unit": "base_mdp",
        "strata": {key: value for key, value in sorted(strata.items())},
        "refit_per_draw": True,
        "paired_all_endpoints": True,
        "pairwise_margins": margin_values,
        "context": dict(checkpoint_context or {}),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing existing bootstrap checkpoint without --resume: {checkpoint_path}"
            )
        checkpoint = _read_json(checkpoint_path)
        saved_metadata = checkpoint.get("metadata")
        if saved_metadata != metadata:
            raise ConfirmatoryIntegrityError("Bootstrap config/input metadata mismatch on resume.")
        completed = _validate_bootstrap_checkpoint(checkpoint, metadata, draws)
    else:
        if resume:
            raise FileNotFoundError(f"Cannot resume missing bootstrap checkpoint: {checkpoint_path}")
        completed = {}

    def save_checkpoint() -> None:
        write_json(
            checkpoint_path,
            {
                "schema_version": BOOTSTRAP_SCHEMA_VERSION,
                "metadata": metadata,
                "metadata_sha256": json_sha256(metadata),
                "draws": {str(index): value for index, value in sorted(completed.items())},
            },
        )

    save_checkpoint()
    for draw_index in range(draws):
        if draw_index in completed:
            continue
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([int(seed), draw_index]))
        )
        multiplicities: Counter[str] = Counter()
        for members in strata.values():
            picked = rng.choice(np.asarray(members, dtype=object), size=len(members), replace=True)
            multiplicities.update(str(value) for value in picked.tolist())
        try:
            raw_values, elapsed, peak_memory = _measure(
                lambda: fit_and_score(dict(sorted(multiplicities.items())), draw_index)
            )
            if not isinstance(raw_values, Mapping):
                raise TypeError("fit_and_score must return a mapping of endpoint values")
            values = {name: float(raw_values[name]) for name in endpoint_names}
            if set(str(key) for key in raw_values) != set(endpoint_names):
                raise ValueError("fit_and_score endpoint set differs from the frozen paired set")
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite endpoint at bootstrap draw {draw_index}")
            body = {
                "draw_index": draw_index,
                "multiplicities": dict(sorted(multiplicities.items())),
                "endpoints": values,
                "runtime_seconds": elapsed,
                "peak_tracemalloc_bytes": peak_memory,
                "memory_measurement": "tracemalloc_peak_python_heap_bytes",
            }
            completed[draw_index] = {**body, "sha256": json_sha256(body)}
            if (draw_index + 1) % checkpoint_interval == 0 or draw_index == draws - 1:
                save_checkpoint()
        except Exception as exc:
            failure = {
                "schema_version": BOOTSTRAP_SCHEMA_VERSION,
                "run_id": CONFIRMATORY_RUN_ID,
                "status": "failed_stop_on_error",
                "draw_index": draw_index,
                "draws_completed": len(completed),
                "draws_requested": draws,
                "mean_completed_draw_runtime_seconds": (
                    sum(float(value.get("runtime_seconds", 0.0)) for value in completed.values())
                    / len(completed)
                    if completed
                    else None
                ),
                "estimated_runtime_seconds": (
                    sum(float(value.get("runtime_seconds", 0.0)) for value in completed.values())
                    / len(completed)
                    * draws
                    if completed
                    else None
                ),
                "runtime_estimate_basis": "mean_completed_draw_runtime_x_requested_draws",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "created_at_utc": _utc_now(),
            }
            write_json(checkpoint_path.with_name("bootstrap_failure.json"), failure)
            raise BootstrapExecutionError(
                f"Paired refit-bootstrap stopped at draw {draw_index}: {exc}"
            ) from exc

    matrices = {
        name: np.asarray([completed[index]["endpoints"][name] for index in range(draws)], dtype=float)
        for name in endpoint_names
    }
    pairwise: dict[str, Any] = {}
    for left, right in combinations(endpoint_names, 2):
        values = matrices[left] - matrices[right]
        pair_name = f"{left}_minus_{right}"
        observed_difference = observed_values[left] - observed_values[right]
        margin = float(margin_values.get(pair_name, 0.0))
        q025 = float(np.quantile(values, 0.025))
        q975 = float(np.quantile(values, 0.975))
        pairwise[f"{left}_minus_{right}"] = {
            "observed": observed_difference,
            "draw_mean": float(np.mean(values)),
            "draw_q025": q025,
            "draw_q975": q975,
            "margin": margin,
            "ci_method": "basic_95_percent",
            "ci_low": 2.0 * observed_difference - q975,
            "ci_high": 2.0 * observed_difference - q025,
            "p_value": float(
                (1 + np.count_nonzero(
                    np.abs(values - observed_difference)
                    >= abs(observed_difference - margin)
                ))
                / (draws + 1)
            ),
        }
    report = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "run_id": CONFIRMATORY_RUN_ID,
        "status": "complete",
        "draws_requested": draws,
        "draws_completed": len(completed),
        "seed": seed,
        "rng": "numpy.random.PCG64",
        "resampling_unit": "base_mdp",
        "strata": {key: len(value) for key, value in sorted(strata.items())},
        "refit_per_draw": True,
        "paired_all_endpoints": True,
        "observed": observed_values,
        "endpoint_draw_summary": {
            name: {
                "mean": float(np.mean(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
            }
            for name, values in matrices.items()
        },
        "paired_differences": pairwise,
        "runtime_seconds": float(
            sum(float(completed[index].get("runtime_seconds", 0.0)) for index in range(draws))
        ),
        "mean_draw_runtime_seconds": float(
            sum(float(completed[index].get("runtime_seconds", 0.0)) for index in range(draws))
            / draws
        ),
        "estimated_runtime_seconds": float(
            sum(float(completed[index].get("runtime_seconds", 0.0)) for index in range(draws))
            / draws
            * draws
        ),
        "runtime_estimate_basis": "mean_completed_draw_runtime_x_requested_draws",
        "peak_tracemalloc_bytes": max(
            int(completed[index].get("peak_tracemalloc_bytes", 0)) for index in range(draws)
        ),
        "memory_measurement": "tracemalloc_peak_python_heap_bytes",
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(checkpoint_path.with_name("bootstrap_summary.json"), report)
    return report


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "CONFIRMATORY_RUN_ID",
    "CONFIRMATORY_OUTPUT_DIR_NAME",
    "CONFIRMATORY_SCHEMA_VERSION",
    "DEFAULT_BASELINE_METHODS",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "BootstrapExecutionError",
    "ConfirmatoryIntegrityError",
    "build_job_plan",
    "json_sha256",
    "plan_sha256",
    "prepare_execution",
    "run_missing_jobs_only",
    "run_paired_refit_bootstrap",
    "validate_job_plan",
]
