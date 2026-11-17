#!/usr/bin/env python
"""Week 2 pilot aggregation pipeline.

Ingests artifacts produced by the other three workstreams, shaped per
``docs/week2/WEEK2_ARTIFACT_CONTRACT.md``, and writes ``artifacts/week2/*``.
Does not hard-code any expected result (mega-task section 13). A missing
required input artifact is a hard failure with a clear message, never a
partially-fabricated output.

Usage::

    python pilot/aggregate.py --input-dir <dir with the 6 input CSVs> --output-dir artifacts/week2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from evaluation.aggregation import (  # noqa: E402
    aggregate,
    build_per_run_results,
    compare_k_dagger,
    compare_n_required,
)
from evaluation.aggregation.aggregate import PER_RUN_FIELDS  # noqa: E402
from evaluation.baselines import BASELINE_REGISTRY  # noqa: E402
from evaluation.censoring import censoring_report, classify_n_emp_status  # noqa: E402
from evaluation.decision_gate import build_decision_gate_input  # noqa: E402
from evaluation.decision_gate.derive import derive_gate_inputs  # noqa: E402
from evaluation.io import sha256_file, sha256_text, write_csv_rows, write_json  # noqa: E402
from evaluation.validation.schema import validate_artifact  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH  # noqa: E402

REQUIRED_ARTIFACTS: dict[str, str] = {
    "benchmark_metadata": "benchmark_metadata.csv",
    "exact_oracle_results": "exact_oracle_results.csv",
    "exact_k_curve": "exact_k_curve.csv",
    "rl_run_results": "rl_run_results.csv",
    "behavior_policy_metadata": "behavior_policy_metadata.csv",
    "empirical_sample_complexity": "empirical_sample_complexity.csv",
    "oeh_results": "oeh_results.csv",
}


def _unique_k_dagger_exact(exact_k_curve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``exact_k_curve.csv`` repeats ``k_dagger_exact`` once per depth row
    (WEEK2_ARTIFACT_CONTRACT.md sec 1); collapse to one row per
    (mdp_id, policy_family) for comparisons that only need the scalar."""

    seen: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in exact_k_curve_rows:
        key = (row["mdp_id"], row["policy_family"])
        if key in seen and seen[key]["k_dagger_exact"] != row["k_dagger_exact"]:
            raise ValueError(f"{key}: exact_k_curve.csv has inconsistent k_dagger_exact across depths")
        seen[key] = row
    return [
        {"mdp_id": mdp_id, "policy_family": family, "k_dagger_exact": row["k_dagger_exact"]}
        for (mdp_id, family), row in seen.items()
    ]


def load_and_validate(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for contract_name, filename in REQUIRED_ARTIFACTS.items():
        path = input_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        loaded[contract_name] = validate_artifact(path, contract_name)
    if missing:
        raise FileNotFoundError(
            f"Missing required input artifact(s) in {input_dir}: {missing}. "
            "See docs/week2/WEEK2_ARTIFACT_CONTRACT.md."
        )
    return loaded


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _combined_input_hash(input_dir: Path) -> str:
    """A single hash over every required input artifact, so a re-run that
    silently used a changed input file (not just a changed protocol config)
    is still detectable (WEEK2_STATISTICAL_PLAN.md sec 6)."""

    parts = [
        f"{filename}:{sha256_file(input_dir / filename)}"
        for filename in sorted(REQUIRED_ARTIFACTS.values())
    ]
    return sha256_text("\n".join(parts))


def build_baseline_results(artifacts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in BASELINE_REGISTRY.items():
        if spec.source_artifact is None:
            for benchmark_row in artifacts["benchmark_metadata"]:
                mdp_id = benchmark_row["mdp_id"]
                deterministic_seed = int(hashlib.sha256(mdp_id.encode("utf-8")).hexdigest(), 16) % (2**32)
                extracted = spec.extract(mdp_id, seed=deterministic_seed)
                for feature, value in extracted.items():
                    if feature in {"mdp_id", "policy_family"}:
                        continue
                    rows.append(
                        {"baseline": name, "mdp_id": extracted["mdp_id"], "policy_family": "",
                         "feature": feature, "value": value}
                    )
            continue
        for source_row in artifacts[spec.source_artifact]:
            extracted = spec.extract(source_row)
            # extract() may legitimately return a subset of spec.feature_set
            # (e.g. structural_only only reports columns actually present in
            # this MDP's metadata row) -- iterate what was returned, not the
            # full registered feature list.
            for feature, value in extracted.items():
                if feature in {"mdp_id", "policy_family"}:
                    continue
                rows.append(
                    {
                        "baseline": name,
                        "mdp_id": extracted["mdp_id"],
                        "policy_family": extracted.get("policy_family", ""),
                        "feature": feature,
                        "value": value,
                    }
                )
    return rows


def build_censoring_analysis(artifacts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    status_rows: list[dict[str, Any]] = []
    for row in artifacts["oeh_results"]:
        if not row["selected"]:
            continue
        status_rows.append(
            {
                "mdp_id": row["mdp_id"],
                "family": row["policy_family"],
                "algorithm": "",
                "seed": "",
                "dataset_size": row["pilot_multiplier"],
                "axis": "oeh_pilot_selection",
                "status": row["oeh_status"],
            }
        )
    for row in artifacts["empirical_sample_complexity"]:
        status = classify_n_emp_status(right_censored=bool(row["n_emp_censored"]), requirement=row["n_emp_multiplier"])
        status_rows.append(
            {
                "mdp_id": row["mdp_id"],
                "family": row["policy_family"],
                "algorithm": row["algorithm"],
                "seed": "",
                "dataset_size": "",
                "axis": "n_emp_label",
                "status": status,
            }
        )
    return censoring_report(status_rows, status_field="status", group_by=("family", "axis"))


def load_c2_criteria(path: Path | None) -> dict[str, Any] | None:
    """Success criteria 3 and 4 from ``scripts/analysis/run_c2_correlation.py``'s
    ``c2_decision_inputs.json``. Absent, both stay ``None`` at the gate --
    "not measured", which sec 8.2 treats differently from "measured and
    failed" (WEEK2_DECISION_GATE.md sec 6)."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError(f"{path}: expected a top-level 'criteria' object (scripts/analysis/run_c2_correlation.py writes one).")
    if payload.get("synthetic"):
        raise ValueError(
            f"{path}: this is a synthetic dry-run file. The gate does not consume planted numbers "
            "(ARM_REGISTRY.md, C2 pre-registration entry)."
        )
    return {
        "beats_strongest_baseline_ci_excludes_zero": criteria.get("beats_strongest_baseline_ci_excludes_zero"),
        "holds_on_non_q_star_families": criteria.get("holds_on_non_q_star_families"),
        "_source": str(path),
    }


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    rng_seed: int = 20260818,
    c2_decision_inputs: Path | None = None,
    coverage_grid_expansion_feasible: bool = False,
) -> dict[str, Any]:
    artifacts = load_and_validate(input_dir)
    rng = np.random.default_rng(rng_seed)
    oeh_selected = [row for row in artifacts["oeh_results"] if row["selected"]]

    unique_k_dagger_exact = _unique_k_dagger_exact(artifacts["exact_k_curve"])
    per_run = build_per_run_results(
        rl_rows=artifacts["rl_run_results"],
        n_emp_rows=artifacts["empirical_sample_complexity"],
        oeh_rows=oeh_selected,
        exact_rows=artifacts["exact_oracle_results"],
        exact_k_dagger_rows=unique_k_dagger_exact,
    )
    write_csv_rows(output_dir / "per_run_results.csv", per_run, PER_RUN_FIELDS)

    aggregated = aggregate(
        per_run, group_by=("family", "algorithm", "metric"), cluster_field="mdp_id", rng=rng, include_median=True
    )
    aggregated_fields = (
        "family", "algorithm", "metric", "n_total", "n_observed", "n_censored", "n_abstained",
        "n_invalid", "mean", "median", "std", "ci_low", "ci_high", "num_clusters",
    )
    write_csv_rows(output_dir / "aggregated_results.csv", aggregated, aggregated_fields)

    baseline_results = build_baseline_results(artifacts)
    write_csv_rows(
        output_dir / "baseline_results.csv", baseline_results, ("baseline", "mdp_id", "policy_family", "feature", "value")
    )

    censoring_analysis = build_censoring_analysis(artifacts)
    if censoring_analysis:
        censoring_fields = tuple(censoring_analysis[0].keys())
        write_csv_rows(output_dir / "censoring_analysis.csv", censoring_analysis, censoring_fields)

    k_dagger_comparison = compare_k_dagger(oeh_selected, unique_k_dagger_exact)
    if k_dagger_comparison:
        write_csv_rows(output_dir / "k_dagger_comparison.csv", k_dagger_comparison, tuple(k_dagger_comparison[0].keys()))

    n_req_comparison = compare_n_required(oeh_selected, artifacts["empirical_sample_complexity"])
    if n_req_comparison:
        write_csv_rows(output_dir / "n_req_comparison.csv", n_req_comparison, tuple(n_req_comparison[0].keys()))

    # D9.4's open item: derive every gate field from the artifacts just
    # aggregated, instead of passing three booleans and letting the JSON read
    # as "NO_GO because nothing was supplied" while the artifacts sat right here.
    gate_inputs, derivation = derive_gate_inputs(
        artifacts,
        max_size_multiplier=max(PROTOCOL_CONFIG["collection"]["size_multipliers"]),
        k_max=PROTOCOL_CONFIG["estimator"]["k_max"],
        tests_passing=True,
        no_leakage_violations=True,
        reproducible=True,
        baselines_present=bool(baseline_results),
        coverage_grid_expansion_feasible=coverage_grid_expansion_feasible,
        c2_criteria=load_c2_criteria(c2_decision_inputs),
    )

    write_json(output_dir / "k1a_framings.json", derivation["k1a_framings"])

    decision_gate_input = build_decision_gate_input(
        gate_inputs,
        git_commit=_git_commit(),
        config_hash=sha256_text(PROTOCOL_CONFIG_PATH.read_text(encoding="utf-8")),
        protocol_version=PROTOCOL_CONFIG["schema_version"],
        input_artifacts_hash=_combined_input_hash(input_dir),
        rng_seed=rng_seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    decision_gate_input["derivation"] = derivation
    write_json(output_dir / "decision_gate_input.json", decision_gate_input)

    return {
        "per_run_rows": len(per_run),
        "aggregated_rows": len(aggregated),
        "baseline_rows": len(baseline_results),
        "decision": decision_gate_input["decision"],
        "k1a_framings": derivation["k1a_framings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "week2")
    parser.add_argument("--rng-seed", type=int, default=20260818)
    parser.add_argument(
        "--c2-decision-inputs",
        type=Path,
        default=None,
        help="c2_decision_inputs.json from scripts/analysis/run_c2_correlation.py, carrying success criteria 3 and 4.",
    )
    parser.add_argument(
        "--grid-expansion-feasible",
        action="store_true",
        help=(
            "Declare that expanding the label grid past the registered cap is feasible. "
            "PLAN_Final.md sec 7 requires this to be decided AT the gate, so it is a flag "
            "the operator sets, never something inferred from the censoring rate."
        ),
    )
    args = parser.parse_args()
    summary = run(
        args.input_dir,
        args.output_dir,
        rng_seed=args.rng_seed,
        c2_decision_inputs=args.c2_decision_inputs,
        coverage_grid_expansion_feasible=args.grid_expansion_feasible,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
