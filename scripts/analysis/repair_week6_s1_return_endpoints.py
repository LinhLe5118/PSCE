#!/usr/bin/env python
"""Repair S1 return endpoints while preserving the sealed bootstrap samples.

The first S1 adapter omitted the registered salt when assigning the inner
return-selection folds.  Survival endpoints used their registered salts and
remain valid.  This migration verifies the original checkpoint, preserves its
2,000 cluster samples and survival endpoints, and recomputes only the two
return endpoints with the exact E1/T1 procedure.  It writes a separate audit
directory and never mutates the invalidated checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.confirmatory import (  # noqa: E402
    _validate_bootstrap_checkpoint,
    json_sha256,
    run_paired_refit_bootstrap,
)
from evaluation.e1_analysis import load_e1_analysis_panel  # noqa: E402
from evaluation.io import sha256_file, write_json  # noqa: E402
from scripts.analysis.run_week6_s1 import (  # noqa: E402
    CONTRASTS,
    DEFAULT_E1,
    _bootstrap_metadata,
    _input_hashes,
    _multiplicities_for_draw,
    _oriented_difference,
    _sampling_strata,
)


ANALYSIS_LABEL = "week6_s1_corrected_registered_return_folds_v2"
SURVIVAL_ENDPOINTS = (
    "survival_B1",
    "survival_B2",
    "survival_B2_OEH3_LC_PMS",
)
_WORKER_PANEL = None


def _initialize_worker(output_dir: str) -> None:
    global _WORKER_PANEL
    _WORKER_PANEL = load_e1_analysis_panel(Path(output_dir))


def _repair_return_worker(task: tuple[int, dict[str, int]]) -> dict:
    draw_index, multiplicities = task
    if _WORKER_PANEL is None:
        raise RuntimeError("S1 repair worker panel was not initialized")
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        endpoints = _WORKER_PANEL.fit_return_and_score(multiplicities, draw_index)
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "draw_index": draw_index,
        "multiplicities": multiplicities,
        "return_endpoints": {str(key): float(value) for key, value in endpoints.items()},
        "return_repair_runtime_seconds": elapsed,
        "return_repair_peak_tracemalloc_bytes": int(peak),
    }


def _save_checkpoint(path: Path, metadata: dict, completed: dict[int, dict]) -> None:
    write_json(
        path,
        {
            "schema_version": metadata["schema_version"],
            "metadata": metadata,
            "metadata_sha256": json_sha256(metadata),
            "draws": {str(index): value for index, value in sorted(completed.items())},
        },
    )


def _verdict(report: Mapping[str, object], inputs: Mapping[str, str], provenance: dict) -> dict:
    rows = []
    raw_p = []
    for hypothesis, contrast, margin in CONTRASTS:
        result = _oriented_difference(report, contrast, margin)
        raw_p.append(float(result["p_value"]))
        rows.append({
            "hypothesis": hypothesis,
            "contrast": contrast,
            "margin": margin,
            "estimate": float(result["observed"]),
            "ci_low": float(result["ci_low"]),
            "ci_high": float(result["ci_high"]),
            "p_value": float(result["p_value"]),
        })
    adjusted, rejected = holm_adjust(raw_p, alpha=0.05)
    for row, p_holm, reject in zip(rows, adjusted, rejected):
        row["p_holm"] = float(p_holm)
        row["holm_reject"] = bool(reject)
        row["passes_registered_gate"] = bool(
            reject and float(row["ci_low"]) > float(row["margin"])
        )
    return {
        "schema_version": "week6-s1-analysis-corrected-v2",
        "status": "complete",
        "input_sha256": dict(inputs),
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20260921,
        "resampling_unit": "base_mdp",
        "hypotheses": rows,
        "primary_h1_pass": rows[0]["passes_registered_gate"],
        "claim_rule": "H1 claim only when CI lower bound exceeds 0.005 and Holm rejects",
        "repair_provenance": provenance,
    }


def run(output_dir: Path, *, workers: int, blas_threads: int) -> dict:
    if workers <= 0 or blas_threads <= 0:
        raise ValueError("workers and blas_threads must be positive")
    output = output_dir.resolve()
    source_dir = output / "s1"
    corrected_dir = output / "s1_corrected"
    source_path = source_dir / "bootstrap_checkpoint.json"
    corrected_path = corrected_dir / "bootstrap_checkpoint.json"
    panel = load_e1_analysis_panel(output)
    inputs = _input_hashes(output)

    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_metadata = _bootstrap_metadata(panel, inputs)
    if source_payload.get("metadata") != source_metadata:
        raise ValueError("source S1 checkpoint no longer matches its frozen metadata")
    source_draws = _validate_bootstrap_checkpoint(source_payload, source_metadata, 2000)
    if len(source_draws) != 2000:
        raise ValueError("source S1 checkpoint is incomplete")
    source_sha = sha256_file(source_path)
    source_summary = source_dir / "bootstrap_summary.json"
    source_verdict = source_dir / "s1_verdict.json"
    provenance = {
        "reason": "original S1 return adapter omitted the registered salted inner-fold assignment",
        "source_checkpoint": "s1/bootstrap_checkpoint.json",
        "source_checkpoint_sha256": source_sha,
        "source_summary_sha256": sha256_file(source_summary),
        "source_verdict_sha256": sha256_file(source_verdict),
        "preserved_draw_multiplicities": True,
        "preserved_survival_endpoints": list(SURVIVAL_ENDPOINTS),
        "recomputed_return_endpoints": ["v4e", "fold_best_fixed"],
        "e1_rerun": False,
    }
    corrected_metadata = _bootstrap_metadata(panel, inputs)
    corrected_metadata["context"] = {
        "input_sha256": inputs,
        "analysis": ANALYSIS_LABEL,
        "source_checkpoint_sha256": source_sha,
    }
    corrected_dir.mkdir(parents=True, exist_ok=True)
    if corrected_path.is_file():
        saved = json.loads(corrected_path.read_text(encoding="utf-8"))
        if saved.get("metadata") != corrected_metadata:
            raise ValueError("corrected checkpoint metadata mismatch")
        completed = _validate_bootstrap_checkpoint(saved, corrected_metadata, 2000)
    else:
        completed = {}
        _save_checkpoint(corrected_path, corrected_metadata, completed)

    sampling_strata = _sampling_strata(panel)
    for index, source in source_draws.items():
        expected = _multiplicities_for_draw(source_metadata, index, sampling_strata)
        if source["multiplicities"] != expected:
            raise ValueError(f"source multiplicities differ at draw {index}")
        if not all(name in source["endpoints"] for name in SURVIVAL_ENDPOINTS):
            raise ValueError(f"source survival endpoint is missing at draw {index}")

    missing = [index for index in range(2000) if index not in completed]
    thread_variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in thread_variables}
    for name in thread_variables:
        os.environ[name] = str(blas_threads)
    batch_size = max(workers * 2, 1)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(output),),
        ) as pool:
            for offset in range(0, len(missing), batch_size):
                indices = missing[offset:offset + batch_size]
                tasks = [(index, source_draws[index]["multiplicities"]) for index in indices]
                for repaired in pool.map(_repair_return_worker, tasks, chunksize=1):
                    index = int(repaired["draw_index"])
                    source = source_draws[index]
                    endpoints = {
                        **{name: float(source["endpoints"][name]) for name in SURVIVAL_ENDPOINTS},
                        **repaired["return_endpoints"],
                    }
                    if set(endpoints) != set(corrected_metadata["endpoint_names"]):
                        raise ValueError(f"corrected endpoint set differs at draw {index}")
                    if not all(math.isfinite(value) for value in endpoints.values()):
                        raise ValueError(f"non-finite corrected endpoint at draw {index}")
                    body = {
                        "draw_index": index,
                        "multiplicities": source["multiplicities"],
                        "endpoints": endpoints,
                        "runtime_seconds": repaired["return_repair_runtime_seconds"],
                        "peak_tracemalloc_bytes": repaired["return_repair_peak_tracemalloc_bytes"],
                        "memory_measurement": "tracemalloc_peak_python_heap_bytes",
                        "source_draw_sha256": source["sha256"],
                        "source_draw_runtime_seconds": source["runtime_seconds"],
                        "repair_scope": "return_endpoints_only",
                    }
                    completed[index] = {**body, "sha256": json_sha256(body)}
                _save_checkpoint(corrected_path, corrected_metadata, completed)
                print(json.dumps({
                    "stage": "s1_return_endpoint_repair",
                    "draws_completed": len(completed),
                    "draws_requested": 2000,
                    "workers": workers,
                    "blas_threads_per_worker": blas_threads,
                }), flush=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def unexpected_refit(_multiplicities, _draw_index):
        raise RuntimeError("complete corrected checkpoint unexpectedly requested a refit")

    report = run_paired_refit_bootstrap(
        cluster_strata=panel.cluster_strata,
        observed=panel.observed,
        fit_and_score=unexpected_refit,
        draws=2000,
        seed=20260921,
        checkpoint_path=corrected_path,
        checkpoint_context=corrected_metadata["context"],
        pairwise_margins={name: margin for _, name, margin in CONTRASTS},
        resume=True,
        checkpoint_interval=25,
    )
    verdict = _verdict(report, inputs, provenance)
    write_json(corrected_dir / "s1_verdict.json", verdict)
    write_json(corrected_dir / "repair_provenance.json", provenance)
    invalidation = {
        "schema_version": "week6-s1-invalidation-v1",
        "status": "invalidated",
        **provenance,
        "replacement_checkpoint": "s1_corrected/bootstrap_checkpoint.json",
        "replacement_checkpoint_sha256": sha256_file(corrected_path),
        "replacement_verdict": "s1_corrected/s1_verdict.json",
    }
    write_json(source_dir / "INVALIDATED_BY_INNER_FOLD_MISMATCH.json", invalidation)
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--blas-threads", type=int, default=1)
    args = parser.parse_args(argv)
    result = run(args.output_dir, workers=args.workers, blas_threads=args.blas_threads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
