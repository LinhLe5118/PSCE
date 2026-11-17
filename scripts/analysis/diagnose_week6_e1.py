#!/usr/bin/env python
"""Produce fixed, exploratory diagnostics for a completed Week-6 E1 run.

These diagnostics explain heterogeneity; they do not alter the registered S1
family or authorize a claim that failed its confirmatory gate.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from environments.benchmark_environments import benchmark_specs  # noqa: E402
from evaluation.io import sha256_file, write_json  # noqa: E402


DEFAULT_E1 = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1"
CORE = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
EXPLANATORY_FEATURES = (
    "observed_sa_fraction",
    "weighted_transition_entropy",
    "low_count_cell_fraction",
    "bootstrap_return_mean_std",
    "oeh3_risk_at_pilot",
    "lcoeh4_risk_at_pilot",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Iterable[float]) -> float:
    sequence = list(values)
    if not sequence:
        raise ValueError("cannot average an empty sequence")
    return float(np.mean(sequence))


def _cell_means(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["training_size_multiplier"]) == 10:
            grouped[(row["mdp_id"], row["policy_family"], row["algorithm"])].append(row)
    result = {}
    for key, values in grouped.items():
        seeds = {int(row["training_seed"]) for row in values}
        if len(values) != 10 or seeds != set(range(10)):
            raise ValueError(f"incomplete x10 seed cell: {key}")
        result[key] = {
            "return": _mean(float(row["normalized_return"]) for row in values),
            "success": _mean(1.0 if str(row["success"]).lower() == "true" else 0.0 for row in values),
        }
    return result


def _summarize_method_strata(
    cells: dict[tuple[str, str, str], dict[str, float]],
    families: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (mdp_id, policy, method), values in cells.items():
        key = (method, families[mdp_id], policy)
        grouped[key][mdp_id].append(values["return"])
    return [
        {
            "algorithm": method,
            "environment_family": family,
            "policy_family": policy,
            "base_mdps": len(by_mdp),
            "mean_normalized_return": _mean(_mean(values) for values in by_mdp.values()),
        }
        for (method, family, policy), by_mdp in sorted(grouped.items())
    ]


def _selector_rows(
    decisions: list[dict[str, str]],
    cells: dict[tuple[str, str, str], dict[str, float]],
    families: dict[str, str],
) -> list[dict[str, Any]]:
    result = []
    for row in decisions:
        group = (row["mdp_id"], row["policy_family"])
        returns = {method: cells[(*group, method)]["return"] for method in CORE}
        selected = row["selected_algorithm"]
        fixed = row["fold_best_fixed_algorithm"]
        result.append({
            "mdp_id": group[0],
            "policy_family": group[1],
            "environment_family": families[group[0]],
            "selected_algorithm": selected,
            "fold_best_fixed_algorithm": fixed,
            "selection_gain": returns[selected] - returns[fixed],
            "oracle_regret": max(returns.values()) - returns[selected],
            "selected_return": returns[selected],
            "fallback": str(row["fallback"]).lower() == "true",
        })
    if len(result) != 84 * 5:
        raise ValueError("selector decision panel is incomplete")
    return result


def _selector_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment_family"], row["policy_family"])].append(row)
    return [
        {
            "environment_family": family,
            "policy_family": policy,
            "groups": len(values),
            "mean_selection_gain": _mean(row["selection_gain"] for row in values),
            "mean_oracle_regret": _mean(row["oracle_regret"] for row in values),
            "fallback_rate": _mean(1.0 if row["fallback"] else 0.0 for row in values),
            "selection_share": {
                method: _mean(1.0 if row["selected_algorithm"] == method else 0.0 for row in values)
                for method in CORE
            },
        }
        for (family, policy), values in sorted(grouped.items())
    ]


def _stable_quartile_bins(
    values: np.ndarray, keys: list[tuple[str, str]]
) -> np.ndarray:
    """Assign near-equal quartiles with deterministic tie-breaking."""

    if len(values) != len(keys) or len(values) < 4:
        raise ValueError("quartile assignment requires at least four matched rows")
    order = sorted(
        range(len(values)),
        key=lambda index: (float(values[index]), keys[index]),
    )
    bins = np.empty(len(values), dtype=int)
    for rank, index in enumerate(order):
        bins[index] = min(3, rank * 4 // len(values))
    return bins


def _feature_quartiles(
    selector: list[dict[str, Any]], features: list[dict[str, str]]
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in features:
        key = (row["mdp_id"], row["policy_family"])
        for feature in EXPLANATORY_FEATURES:
            by_group[key][feature].append(float(row[feature]))
    selector_by_group = {(row["mdp_id"], row["policy_family"]): row for row in selector}
    result = []
    for feature in EXPLANATORY_FEATURES:
        keys = sorted(selector_by_group)
        values = np.asarray([_mean(by_group[key][feature]) for key in keys], dtype=float)
        bins = _stable_quartile_bins(values, keys)
        for quartile in range(4):
            selected = [selector_by_group[key] for key, bucket in zip(keys, bins) if bucket == quartile]
            result.append({
                "feature": feature,
                "quartile": quartile + 1,
                "groups": len(selected),
                "mean_selection_gain": _mean(row["selection_gain"] for row in selected),
                "mean_oracle_regret": _mean(row["oracle_regret"] for row in selected),
                "mean_selected_return": _mean(row["selected_return"] for row in selected),
            })
    return result


def run(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    summary_path = output / "e1_execution_summary.json"
    if not summary_path.is_file() or json.loads(summary_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise FileNotFoundError("diagnostics require a completed E1 summary")
    core_path = output / "stochastic_primary_core_run_results.csv"
    external_path = output / "external_run_results.csv"
    decisions_path = output / "stochastic_primary_selector_decisions.csv"
    features_path = output / "pilot_features" / "features.csv"
    core_cells = _cell_means(_read(core_path))
    all_cells = {**core_cells, **_cell_means(_read(external_path))}
    families = {
        spec.instance_id: spec.family
        for spec in benchmark_specs()
        if spec.split == "evaluation"
    }
    selector = _selector_rows(_read(decisions_path), core_cells, families)
    report = {
        "schema_version": "week6-e1-diagnostics-v1",
        "status": "complete",
        "role": "exploratory_explanation_only",
        "input_sha256": {
            path.name: sha256_file(path)
            for path in (summary_path, core_path, external_path, decisions_path, features_path)
        },
        "method_by_environment_and_behavior": _summarize_method_strata(all_cells, families),
        "selector_by_environment_and_behavior": _selector_strata(selector),
        "selector_feature_quartiles": _feature_quartiles(selector, _read(features_path)),
        "interpretation_limits": [
            "Associations are exploratory and do not establish causality.",
            "External methods are descriptive and outside the confirmatory family.",
            "No row, seed, behavior, or environment is filtered.",
        ],
    }
    write_json(output / "e1_diagnostics.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E1)
    args = parser.parse_args(argv)
    report = run(args.output_dir)
    print(json.dumps({"status": report["status"], "output": "e1_diagnostics.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
