#!/usr/bin/env python
"""Run the frozen S1 analysis after the sealed E1 runner is complete."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.confirmatory import (  # noqa: E402
    BOOTSTRAP_SCHEMA_VERSION,
    CONFIRMATORY_RUN_ID,
    _validate_bootstrap_checkpoint,
    json_sha256,
    run_paired_refit_bootstrap,
)
from evaluation.e1_analysis import load_e1_analysis_panel  # noqa: E402
from evaluation.io import sha256_file, write_json  # noqa: E402


DEFAULT_E1 = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1"
CONTRASTS = (
    ("H1", "v4e_minus_fold_best_fixed", 0.005),
    ("H2", "survival_B2_OEH3_LC_PMS_minus_survival_B2", 0.0),
    ("H3", "survival_B2_minus_survival_B1", 0.0),
)
_WORKER_PANEL = None


def _input_hashes(output: Path) -> dict[str, str]:
    inputs = {
        name: sha256_file(output / name)
        for name in (
            "e1_execution_summary.json",
            "stochastic_primary_core_run_results.csv",
            "stochastic_primary_selector_decisions.csv",
        )
    }
    inputs["pilot_features/features.csv"] = sha256_file(
        output / "pilot_features" / "features.csv"
    )
    return inputs


def _bootstrap_metadata(panel, inputs: Mapping[str, str]) -> dict:
    raw_endpoint_names = {str(name) for name in panel.observed}
    preferred = ("v4e", "fold_best_fixed")
    endpoint_names = tuple(name for name in preferred if name in raw_endpoint_names) + tuple(
        sorted(raw_endpoint_names - set(preferred))
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for cluster, stratum in panel.cluster_strata.items():
        strata[str(stratum)].append(str(cluster))
    for members in strata.values():
        members.sort()
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "run_id": CONFIRMATORY_RUN_ID,
        "draws": 2000,
        "seed": 20260921,
        "rng": "numpy.random.PCG64",
        "endpoint_names": list(endpoint_names),
        "cluster_strata": {
            str(key): str(value) for key, value in sorted(panel.cluster_strata.items())
        },
        "resampling_unit": "base_mdp",
        "strata": {key: value for key, value in sorted(strata.items())},
        "refit_per_draw": True,
        "paired_all_endpoints": True,
        "pairwise_margins": {
            name: float(margin) for _, name, margin in CONTRASTS
        },
        "context": {
            "input_sha256": dict(inputs),
            "analysis": "week6_s1_frozen_v1",
        },
    }


def _sampling_strata(panel) -> tuple[tuple[str, ...], ...]:
    strata: dict[str, list[str]] = defaultdict(list)
    for cluster, stratum in panel.cluster_strata.items():
        strata[str(stratum)].append(str(cluster))
    for members in strata.values():
        members.sort()
    return tuple(tuple(members) for members in strata.values())


def _multiplicities_for_draw(
    metadata: Mapping[str, object],
    draw_index: int,
    sampling_strata: tuple[tuple[str, ...], ...] | None = None,
) -> dict[str, int]:
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(metadata["seed"]), draw_index]))
    )
    multiplicities: Counter[str] = Counter()
    strata_members = sampling_strata or tuple(
        tuple(members) for members in metadata["strata"].values()
    )
    for members in strata_members:
        member_array = np.asarray(members, dtype=object)
        picked = rng.choice(member_array, size=len(member_array), replace=True)
        multiplicities.update(str(value) for value in picked.tolist())
    return dict(sorted(multiplicities.items()))


def _initialize_s1_worker(output_dir: str) -> None:
    global _WORKER_PANEL
    _WORKER_PANEL = load_e1_analysis_panel(Path(output_dir))


def _fit_s1_worker(task: tuple[int, dict[str, int]]) -> dict:
    draw_index, multiplicities = task
    if _WORKER_PANEL is None:
        raise RuntimeError("S1 worker panel was not initialized")
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        raw_values = _WORKER_PANEL.fit_and_score(multiplicities, draw_index)
    finally:
        elapsed = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    if not isinstance(raw_values, Mapping):
        raise TypeError("fit_and_score must return a mapping of endpoint values")
    return {
        "draw_index": draw_index,
        "multiplicities": multiplicities,
        "endpoints": {str(name): float(value) for name, value in raw_values.items()},
        "runtime_seconds": elapsed,
        "peak_tracemalloc_bytes": int(peak_memory),
        "memory_measurement": "tracemalloc_peak_python_heap_bytes",
    }


def _save_parallel_checkpoint(path: Path, metadata: dict, completed: dict[int, dict]) -> None:
    write_json(
        path,
        {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "metadata": metadata,
            "metadata_sha256": json_sha256(metadata),
            "draws": {str(index): value for index, value in sorted(completed.items())},
        },
    )


def _parallel_resume(
    output: Path,
    panel,
    inputs: Mapping[str, str],
    *,
    workers: int,
    blas_threads: int,
) -> None:
    checkpoint_path = output / "s1" / "bootstrap_checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("parallel S1 requires an existing verified checkpoint")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    metadata = _bootstrap_metadata(panel, inputs)
    if checkpoint.get("metadata") != metadata:
        raise ValueError("parallel S1 metadata differs from the frozen sequential checkpoint")
    completed = _validate_bootstrap_checkpoint(checkpoint, metadata, int(metadata["draws"]))
    missing = [index for index in range(int(metadata["draws"])) if index not in completed]
    if not missing:
        return
    sampling_strata = _sampling_strata(panel)

    thread_variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    previous_environment = {name: os.environ.get(name) for name in thread_variables}
    for name in thread_variables:
        os.environ[name] = str(blas_threads)
    batch_size = max(workers * 2, 1)
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_s1_worker,
            initargs=(str(output),),
        ) as pool:
            for offset in range(0, len(missing), batch_size):
                indices = missing[offset:offset + batch_size]
                tasks = [
                    (index, _multiplicities_for_draw(metadata, index, sampling_strata))
                    for index in indices
                ]
                bodies = list(pool.map(_fit_s1_worker, tasks, chunksize=1))
                for body in bodies:
                    draw_index = int(body["draw_index"])
                    endpoints = body["endpoints"]
                    if set(endpoints) != set(metadata["endpoint_names"]):
                        raise ValueError(f"endpoint set differs at draw {draw_index}")
                    if not all(math.isfinite(float(value)) for value in endpoints.values()):
                        raise ValueError(f"non-finite endpoint at draw {draw_index}")
                    expected_multiplicities = _multiplicities_for_draw(
                        metadata, draw_index, sampling_strata
                    )
                    if body["multiplicities"] != expected_multiplicities:
                        raise ValueError(f"multiplicity mismatch at draw {draw_index}")
                    completed[draw_index] = {**body, "sha256": json_sha256(body)}
                _save_parallel_checkpoint(checkpoint_path, metadata, completed)
                print(
                    json.dumps({
                        "stage": "parallel_s1",
                        "draws_completed": len(completed),
                        "draws_requested": int(metadata["draws"]),
                        "workers": workers,
                        "blas_threads_per_worker": blas_threads,
                    }),
                    flush=True,
                )
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def verify_checkpoint_draw(output_dir: Path, draw_index: int) -> dict:
    output = output_dir.resolve()
    panel = load_e1_analysis_panel(output)
    inputs = _input_hashes(output)
    metadata = _bootstrap_metadata(panel, inputs)
    checkpoint_path = output / "s1" / "bootstrap_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("metadata") != metadata:
        raise ValueError("checkpoint metadata differs from the frozen S1 configuration")
    completed = _validate_bootstrap_checkpoint(checkpoint, metadata, int(metadata["draws"]))
    if draw_index not in completed:
        raise ValueError(f"draw {draw_index} is not present in the checkpoint")
    multiplicities = _multiplicities_for_draw(
        metadata, draw_index, _sampling_strata(panel)
    )
    if completed[draw_index]["multiplicities"] != multiplicities:
        raise ValueError(f"draw {draw_index} multiplicities differ")
    recomputed = panel.fit_and_score(multiplicities, draw_index)
    differences = {
        name: abs(float(recomputed[name]) - float(completed[draw_index]["endpoints"][name]))
        for name in metadata["endpoint_names"]
    }
    if any(value > 1e-12 for value in differences.values()):
        raise ValueError(f"draw {draw_index} endpoint reproduction differs: {differences}")
    return {"draw_index": draw_index, "status": "exact_within_1e-12", "differences": differences}


def _oriented_difference(report: Mapping[str, object], contrast: str, margin: float) -> dict:
    differences = report["paired_differences"]
    if contrast in differences:
        return dict(differences[contrast])
    left, separator, right = contrast.partition("_minus_")
    if not separator:
        raise KeyError(f"invalid paired contrast name: {contrast}")
    reverse = f"{right}_minus_{left}"
    if reverse not in differences:
        raise KeyError(f"paired contrast is absent in both directions: {contrast}")
    if margin != 0.0:
        raise ValueError("a reversed nonzero-margin contrast requires explicit recomputation")
    source = differences[reverse]
    return {
        **source,
        "observed": -float(source["observed"]),
        "draw_mean": -float(source["draw_mean"]),
        "draw_q025": -float(source["draw_q975"]),
        "draw_q975": -float(source["draw_q025"]),
        "ci_low": -float(source["ci_high"]),
        "ci_high": -float(source["ci_low"]),
    }


def run(
    output_dir: Path,
    *,
    resume: bool = False,
    workers: int = 1,
    blas_threads: int = 1,
) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if blas_threads <= 0:
        raise ValueError("blas_threads must be positive")
    if workers > 1 and not resume:
        raise ValueError("parallel S1 is a checkpoint-only resume mode")
    output = output_dir.resolve()
    panel = load_e1_analysis_panel(output)
    s1 = output / "s1"
    s1.mkdir(parents=True, exist_ok=True)
    inputs = _input_hashes(output)
    if workers > 1:
        _parallel_resume(
            output, panel, inputs, workers=workers, blas_threads=blas_threads
        )
    report = run_paired_refit_bootstrap(
        cluster_strata=panel.cluster_strata,
        observed=panel.observed,
        fit_and_score=panel.fit_and_score,
        draws=2000,
        seed=20260921,
        checkpoint_path=s1 / "bootstrap_checkpoint.json",
        checkpoint_context={"input_sha256": inputs, "analysis": "week6_s1_frozen_v1"},
        pairwise_margins={name: margin for _, name, margin in CONTRASTS},
        resume=resume,
        checkpoint_interval=25,
    )
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
    verdict = {
        "schema_version": "week6-s1-analysis-v1",
        "status": "complete",
        "input_sha256": inputs,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20260921,
        "resampling_unit": "base_mdp",
        "hypotheses": rows,
        "primary_h1_pass": rows[0]["passes_registered_gate"],
        "claim_rule": "H1 claim only when CI lower bound exceeds 0.005 and Holm rejects",
    }
    write_json(s1 / "s1_verdict.json", verdict)
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--verify-draw", type=int)
    args = parser.parse_args(argv)
    if args.verify_draw is not None:
        result = verify_checkpoint_draw(args.output_dir, args.verify_draw)
    else:
        result = run(
            args.output_dir,
            resume=args.resume,
            workers=args.workers,
            blas_threads=args.blas_threads,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
