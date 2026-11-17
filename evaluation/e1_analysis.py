"""Frozen S1 analysis adapter for the sealed Week-6 E1 outputs.

The adapter reconstructs the registered return and discrete-survival panels
from preserved E1 artifacts.  Bootstrap multiplicities are implemented by
integer replication of whole base-MDP clusters, while the original outer-fold
assignment remains fixed.  Every draw therefore refits standardization,
inner ridge selection, the return selectors, survival calibrators, and the
outer-training best-fixed comparator from one paired cluster sample.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from evaluation.aggregation.folds import assign_folds
from evaluation.development_confirmatory import DevelopmentPanelRow
from scripts.analysis.run_calibrated_comparison import per_instance_log_likelihood
from return_selection import _fit_ridge, tune_return_ridge
from stochastic_evaluation import (
    SurvivalInstance,
    _fit_predict,
    suffix_requirement,
    tune_ridge_grouped,
)


GRID = (1, 2, 5, 10, 20, 50)
ALGORITHMS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
RETURN_FEATURE_SETS = ("B2_OEH3", "B2_OEH3_LCOEH4")
SURVIVAL_FEATURE_SETS = ("B1", "B2", "B2_OEH3_LC_PMS")


def _select_primary_features(
    features: list[dict[str, str]],
    outcome_keys: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Select feature rows that belong to the stochastic-primary outcomes."""

    selected: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in features:
        key = (row["mdp_id"], row["policy_family"], row["algorithm"])
        if key not in outcome_keys:
            continue
        if key in selected:
            raise ValueError(f"duplicate stochastic feature row for {key}")
        selected[key] = row
    if set(selected) != outcome_keys:
        raise ValueError("E1 feature/outcome identities differ")
    return selected


def _nested_survival_oof_fixed(
    instances: list[SurvivalInstance],
    *,
    outer: Mapping[str, int],
    ridges: tuple[float, ...],
) -> list[dict[str, object]]:
    """Nested survival OOF with the registered outer partition held fixed."""

    mdp_ids = {row.mdp_id for row in instances}
    if mdp_ids - set(outer):
        raise ValueError("registered outer assignment is incomplete")
    output: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        panel = [row for row in instances if row.algorithm == algorithm]
        for feature_set in SURVIVAL_FEATURE_SETS:
            for fold in range(5):
                train = [row for row in panel if outer[row.mdp_id] != fold]
                test = [row for row in panel if outer[row.mdp_id] == fold]
                if not test:
                    continue
                selected, inner_scores = tune_ridge_grouped(
                    train,
                    feature_set=feature_set,
                    grid=GRID,
                    ridge_grid=ridges,
                    num_folds=4,
                    salt=f"outer={fold}|algorithm={algorithm}|features={feature_set}",
                )
                hazards = _fit_predict(
                    train, test, feature_set=feature_set, grid=GRID, ridge=selected
                )
                for item, hazard in zip(test, hazards):
                    output.append({
                        "mdp_id": item.mdp_id,
                        "policy_family": item.policy_family,
                        "algorithm": item.algorithm,
                        "feature_set": feature_set,
                        "outer_fold": fold,
                        "selected_ridge": selected,
                        "inner_mean_log_likelihood": inner_scores[selected],
                        "grid_index": item.grid_index,
                        "observed": item.observed,
                        "hazards": tuple(float(value) for value in hazard),
                    })
    expected = len(instances) * len(SURVIVAL_FEATURE_SETS)
    if len(output) != expected:
        raise ValueError(f"incomplete survival OOF predictions: {len(output)}/{expected}")
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


@dataclass(frozen=True)
class E1AnalysisPanel:
    return_rows: tuple[DevelopmentPanelRow, ...]
    return_groups: tuple[tuple[str, str], ...]
    survival_rows: tuple[SurvivalInstance, ...]
    outer_assignment: Mapping[str, int]
    cluster_strata: Mapping[str, str]
    observed: Mapping[str, float]
    fit_return_and_score: Callable[[Mapping[str, int], int], Mapping[str, float]]
    fit_survival_and_score: Callable[[Mapping[str, int], int], Mapping[str, float]]
    fit_and_score: Callable[[Mapping[str, int], int], Mapping[str, float]]


def _registered_return_endpoints(
    rows: tuple[DevelopmentPanelRow, ...],
    groups: tuple[tuple[str, str], ...],
    outer: Mapping[str, int],
    multiplicities: Mapping[str, int],
    ridges: tuple[float, ...],
) -> dict[str, float]:
    """Refit the exact salted return procedure registered in T1 and used by E1."""

    row_by_key = {
        (row.mdp_id, row.policy_family, row.algorithm): row for row in rows
    }
    selected_values: list[tuple[str, float, float]] = []
    for fold in range(5):
        test_groups = [
            group
            for group in groups
            if outer[group[0]] == fold and int(multiplicities.get(group[0], 0)) > 0
        ]
        if not test_groups:
            continue
        predictions: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for algorithm in ALGORITHMS:
            algorithm_rows = [row for row in rows if row.algorithm == algorithm]
            train = [
                row
                for row in algorithm_rows
                if outer[row.mdp_id] != fold
                for _ in range(int(multiplicities.get(row.mdp_id, 0)))
            ]
            test = [
                row
                for row in algorithm_rows
                if outer[row.mdp_id] == fold
                and int(multiplicities.get(row.mdp_id, 0)) > 0
            ]
            if not train or not test:
                raise ValueError("registered return refit has an empty outer fold")
            for feature_set in RETURN_FEATURE_SETS:
                selected_ridge, _ = tune_return_ridge(
                    train,
                    feature_set=feature_set,
                    ridge_grid=ridges,
                    num_folds=4,
                    salt=f"return|{fold}|{algorithm}|{feature_set}",
                )
                values = _fit_ridge(
                    train, test, feature_set=feature_set, ridge=selected_ridge
                )
                predictions[(algorithm, feature_set)] = {
                    (row.mdp_id, row.policy_family): float(value)
                    for row, value in zip(test, values)
                }

        fixed_scores = {algorithm: 0.0 for algorithm in ALGORITHMS}
        fixed_mass = {algorithm: 0.0 for algorithm in ALGORITHMS}
        for row in rows:
            if outer[row.mdp_id] == fold:
                continue
            weight = float(multiplicities.get(row.mdp_id, 0))
            fixed_scores[row.algorithm] += weight * row.normalized_return
            fixed_mass[row.algorithm] += weight
        if any(value <= 0.0 for value in fixed_mass.values()):
            raise ValueError("registered return refit has empty best-fixed training mass")
        fixed_algorithm = max(
            ALGORITHMS,
            key=lambda algorithm: (
                fixed_scores[algorithm] / fixed_mass[algorithm],
                -ALGORITHMS.index(algorithm),
            ),
        )
        for mdp_id, policy_family in test_groups:
            predicted = {
                algorithm: float(
                    np.mean([
                        predictions[(algorithm, feature_set)][(mdp_id, policy_family)]
                        for feature_set in RETURN_FEATURE_SETS
                    ])
                )
                for algorithm in ALGORITHMS
            }
            selected_algorithm = max(
                ALGORITHMS,
                key=lambda algorithm: (
                    predicted[algorithm], -ALGORITHMS.index(algorithm)
                ),
            )
            selected_values.append((
                mdp_id,
                row_by_key[(mdp_id, policy_family, selected_algorithm)].normalized_return,
                row_by_key[(mdp_id, policy_family, fixed_algorithm)].normalized_return,
            ))

    by_mdp: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for mdp_id, selected, fixed in selected_values:
        by_mdp[mdp_id].append((selected, fixed))
    denominator = 0.0
    selected_total = 0.0
    fixed_total = 0.0
    for mdp_id, values in by_mdp.items():
        weight = float(multiplicities.get(mdp_id, 0))
        selected_total += weight * float(np.mean([value[0] for value in values]))
        fixed_total += weight * float(np.mean([value[1] for value in values]))
        denominator += weight
    if denominator <= 0.0:
        raise ValueError("registered return refit has no positive-weight clusters")
    return {
        "v4e": selected_total / denominator,
        "fold_best_fixed": fixed_total / denominator,
    }


def _survival_endpoints(
    rows: tuple[SurvivalInstance, ...],
    outer: Mapping[str, int],
    multiplicities: Mapping[str, int],
    ridges: tuple[float, ...],
) -> dict[str, float]:
    expanded = [
        row
        for row in rows
        for _ in range(int(multiplicities.get(row.mdp_id, 0)))
    ]
    if not expanded:
        raise ValueError("survival bootstrap sample is empty")
    predictions = _nested_survival_oof_fixed(expanded, outer=outer, ridges=ridges)
    values: dict[str, list[float]] = defaultdict(list)
    for prediction in predictions:
        values[str(prediction["feature_set"])].append(
            per_instance_log_likelihood(
                np.asarray(prediction["hazards"], dtype=float),
                int(prediction["grid_index"]),
                bool(prediction["observed"]),
            )
        )
    if set(values) != set(SURVIVAL_FEATURE_SETS):
        raise ValueError("incomplete survival endpoint panel")
    return {
        "survival_B1": float(np.mean(values["B1"])),
        "survival_B2": float(np.mean(values["B2"])),
        "survival_B2_OEH3_LC_PMS": float(np.mean(values["B2_OEH3_LC_PMS"])),
    }


def load_e1_analysis_panel(
    output_dir: Path,
    *,
    require_complete: bool = True,
) -> E1AnalysisPanel:
    """Load and validate the stochastic-primary E1 panel without filtering."""

    output = output_dir.resolve()
    summary_path = output / "e1_execution_summary.json"
    if require_complete:
        if not summary_path.is_file():
            raise FileNotFoundError("E1 execution summary is not complete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or int(summary.get("core_fits", -1)) != 108_000:
            raise ValueError("E1 execution summary is incomplete")

    contract = json.loads(
        (ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json").read_text(
            encoding="utf-8"
        )
    )
    feature_names = contract["feature_sets"]
    features = _read_csv(output / "pilot_features" / "features.csv")
    runs = _read_csv(output / "stochastic_primary_core_run_results.csv")
    outcomes: dict[tuple[str, str, str], dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in runs:
        if row["algorithm"] not in ALGORITHMS:
            raise ValueError(f"unexpected core algorithm: {row['algorithm']}")
        if row["run_status"] != "observed":
            raise ValueError("non-observed core result in E1 primary panel")
        key = (row["mdp_id"], row["policy_family"], row["algorithm"])
        outcomes[key][int(row["training_size_multiplier"])].append(row)
    feature_by_key = _select_primary_features(features, set(outcomes))
    if len(feature_by_key) != 84 * 5 * len(ALGORITHMS):
        raise ValueError("stochastic feature panel does not contain 84x5x3 rows")

    return_rows: list[DevelopmentPanelRow] = []
    survival_rows: list[SurvivalInstance] = []
    for key in sorted(outcomes):
        by_budget = outcomes[key]
        if set(by_budget) != set(GRID) or any(len(by_budget[value]) != 10 for value in GRID):
            raise ValueError(f"incomplete training grid for {key}")
        x10 = [float(row["normalized_return"]) for row in by_budget[10]]
        if not all(math.isfinite(value) for value in x10):
            raise ValueError(f"non-finite x10 return for {key}")
        feature_row = feature_by_key[key]
        return_rows.append(
            DevelopmentPanelRow(
                mdp_id=key[0],
                policy_family=key[1],
                algorithm=key[2],
                normalized_return=float(np.mean(x10)),
                features={
                    name: tuple(float(feature_row[field]) for field in feature_names[name])
                    for name in RETURN_FEATURE_SETS
                },
            )
        )
        success_counts = [
            sum(_as_bool(row["success"]) for row in by_budget[multiplier])
            for multiplier in GRID
        ]
        grid_index, observed = suffix_requirement(success_counts, minimum_successes=8)
        survival_rows.append(
            SurvivalInstance(
                mdp_id=key[0],
                policy_family=key[1],
                algorithm=key[2],
                grid_index=grid_index,
                observed=observed,
                features={
                    name: tuple(float(feature_row[field]) for field in feature_names[name])
                    for name in SURVIVAL_FEATURE_SETS
                },
            )
        )

    mdp_ids = sorted({row.mdp_id for row in return_rows})
    if len(mdp_ids) != 84:
        raise ValueError("E1 stochastic panel must contain 84 base MDPs")
    outer = assign_folds(mdp_ids, num_folds=5)
    groups = tuple(sorted({(row.mdp_id, row.policy_family) for row in return_rows}))
    return_tuple = tuple(return_rows)
    survival_tuple = tuple(survival_rows)
    ridges = tuple(float(value) for value in contract["evaluation"]["survival_ridges"])
    return_ridges = tuple(
        float(value) for value in contract["evaluation"]["return_ridges"]
    )

    def fit_return_and_score(
        multiplicities: Mapping[str, int], _draw_index: int
    ) -> Mapping[str, float]:
        weights = {str(key): int(value) for key, value in multiplicities.items()}
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("invalid bootstrap multiplicities")
        return _registered_return_endpoints(
            return_tuple, groups, outer, weights, return_ridges
        )

    def fit_survival_and_score(
        multiplicities: Mapping[str, int], _draw_index: int
    ) -> Mapping[str, float]:
        weights = {str(key): int(value) for key, value in multiplicities.items()}
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("invalid bootstrap multiplicities")
        return _survival_endpoints(survival_tuple, outer, weights, ridges)

    def fit_and_score(multiplicities: Mapping[str, int], _draw_index: int) -> Mapping[str, float]:
        return {
            **fit_return_and_score(multiplicities, _draw_index),
            **fit_survival_and_score(multiplicities, _draw_index),
        }

    unit_weights = {mdp_id: 1 for mdp_id in mdp_ids}
    observed = fit_and_score(unit_weights, -1)
    return E1AnalysisPanel(
        return_rows=return_tuple,
        return_groups=groups,
        survival_rows=survival_tuple,
        outer_assignment=outer,
        cluster_strata={mdp_id: str(outer[mdp_id]) for mdp_id in mdp_ids},
        observed=observed,
        fit_return_and_score=fit_return_and_score,
        fit_survival_and_score=fit_survival_and_score,
        fit_and_score=fit_and_score,
    )


__all__ = ["E1AnalysisPanel", "load_e1_analysis_panel"]
