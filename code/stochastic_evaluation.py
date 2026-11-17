"""Leakage-safe development evaluation for stochastic tabular features.

The functions in this module contain the statistical machinery shared by the
B0--B3 predictive evaluation and learner-selection audit.  They deliberately
accept already assembled data-only covariates: loading pilot buffers and
joining post-hoc utility labels remain responsibilities of the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from evaluation.aggregation.folds import assign_folds
from evaluation.calibration.discrete_survival import DiscreteTimeHazardCalibrator
from scripts.analysis.run_calibrated_comparison import per_instance_log_likelihood


@dataclass(frozen=True)
class SurvivalInstance:
    """One learner-specific sample-complexity label and its nested features."""

    mdp_id: str
    policy_family: str
    algorithm: str
    grid_index: int
    observed: bool
    features: Mapping[str, tuple[float, ...]]


def suffix_requirement(
    success_counts: Sequence[int], *, minimum_successes: int = 8
) -> tuple[int, bool]:
    """Return the first grid index whose complete suffix meets the criterion.

    If no suffix qualifies, the case is right-censored at the final grid
    interval.  This is the same monotone 8/10 rule used by the empirical
    sample-complexity labels; it avoids declaring an isolated successful
    budget sufficient when a larger budget subsequently fails.
    """

    counts = [int(value) for value in success_counts]
    if not counts:
        raise ValueError("success_counts must be nonempty")
    if minimum_successes < 1:
        raise ValueError("minimum_successes must be positive")
    if any(value < 0 for value in counts):
        raise ValueError("success_counts cannot be negative")
    for index in range(len(counts)):
        if min(counts[index:]) >= minimum_successes:
            return index, True
    return len(counts) - 1, False


def deterministic_balanced_folds(
    groups: Sequence[str], *, num_folds: int, salt: str
) -> dict[str, int]:
    """Assign unique groups to deterministic folds with size imbalance <= 1."""

    unique = sorted(set(groups))
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if len(unique) < num_folds:
        raise ValueError("fewer groups than folds")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest(),
    )
    return {group: position % num_folds for position, group in enumerate(ordered)}


def fit_standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit column scaling; constant columns remain finite and map to zero."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must be nonempty and two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains non-finite values")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return mean, scale


def transform_standardized(
    matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    result = (np.asarray(matrix, dtype=float) - mean) / scale
    if not np.all(np.isfinite(result)):
        raise ValueError("standardized matrix contains non-finite values")
    return result


def _matrix(instances: Sequence[SurvivalInstance], feature_set: str) -> np.ndarray:
    try:
        matrix = np.asarray([item.features[feature_set] for item in instances], dtype=float)
    except KeyError as exc:
        raise ValueError(f"missing feature set {feature_set}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(instances):
        raise ValueError("feature vectors must be aligned and rectangular")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix contains non-finite values")
    return matrix


def _fit_predict(
    train: Sequence[SurvivalInstance],
    test: Sequence[SurvivalInstance],
    *,
    feature_set: str,
    grid: Sequence[int],
    ridge: float,
) -> np.ndarray:
    train_matrix = _matrix(train, feature_set)
    test_matrix = _matrix(test, feature_set)
    mean, scale = fit_standardizer(train_matrix)
    model = DiscreteTimeHazardCalibrator(grid=grid, ridge=ridge).fit(
        transform_standardized(train_matrix, mean, scale),
        [item.grid_index for item in train],
        [item.observed for item in train],
    )
    predicted = model.predict_hazard(transform_standardized(test_matrix, mean, scale))
    if predicted.shape != (len(test), len(grid)) or not np.all(np.isfinite(predicted)):
        raise ValueError("calibrator returned invalid hazards")
    return predicted


def tune_ridge_grouped(
    train: Sequence[SurvivalInstance],
    *,
    feature_set: str,
    grid: Sequence[int],
    ridge_grid: Sequence[float],
    num_folds: int,
    salt: str,
) -> tuple[float, dict[float, float]]:
    """Choose ridge by inner grouped OOF mean log-likelihood."""

    if not train or not ridge_grid:
        raise ValueError("training instances and ridge_grid must be nonempty")
    folds = deterministic_balanced_folds(
        [item.mdp_id for item in train], num_folds=num_folds, salt=salt
    )
    scores: dict[float, float] = {}
    for raw_ridge in ridge_grid:
        ridge = float(raw_ridge)
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge values must be finite and positive")
        held_out_scores: list[float] = []
        valid = True
        for fold in range(num_folds):
            inner_train = [item for item in train if folds[item.mdp_id] != fold]
            inner_test = [item for item in train if folds[item.mdp_id] == fold]
            try:
                hazards = _fit_predict(
                    inner_train, inner_test, feature_set=feature_set,
                    grid=grid, ridge=ridge,
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                valid = False
                break
            held_out_scores.extend(
                per_instance_log_likelihood(hazard, item.grid_index, item.observed)
                for item, hazard in zip(inner_test, hazards)
            )
        scores[ridge] = (
            float(np.mean(held_out_scores))
            if valid and len(held_out_scores) == len(train)
            else float("-inf")
        )
    # ridge_grid order is irrelevant: an exact score tie chooses less shrinkage.
    selected = min(scores, key=lambda ridge: (-scores[ridge], ridge))
    if not math.isfinite(scores[selected]):
        raise ValueError("all inner ridge candidates failed")
    return selected, scores


def nested_grouped_oof(
    instances: Sequence[SurvivalInstance],
    *,
    feature_sets: Sequence[str],
    grid: Sequence[int],
    ridge_grid: Sequence[float],
    outer_folds: int = 5,
    inner_folds: int = 4,
) -> list[dict[str, object]]:
    """Return nested-CV hazards for every instance and feature set.

    Outer folds use the repository's registered MDP hash assignment.  Inner
    folds are re-balanced deterministically within each outer training split,
    so a small development cohort cannot lose an entire validation fold after
    one outer fold is removed.
    """

    values = list(instances)
    if not values:
        raise ValueError("instances must be nonempty")
    outer = assign_folds({item.mdp_id for item in values}, num_folds=outer_folds)
    output: list[dict[str, object]] = []
    algorithms = sorted({item.algorithm for item in values})
    for algorithm in algorithms:
        panel = [item for item in values if item.algorithm == algorithm]
        for feature_set in feature_sets:
            for fold in range(outer_folds):
                train = [item for item in panel if outer[item.mdp_id] != fold]
                test = [item for item in panel if outer[item.mdp_id] == fold]
                if not test:
                    continue
                selected, inner_scores = tune_ridge_grouped(
                    train,
                    feature_set=feature_set,
                    grid=grid,
                    ridge_grid=ridge_grid,
                    num_folds=inner_folds,
                    salt=f"outer={fold}|algorithm={algorithm}|features={feature_set}",
                )
                hazards = _fit_predict(
                    train, test, feature_set=feature_set, grid=grid, ridge=selected
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
    expected = len(values) * len(feature_sets)
    if len(output) != expected:
        raise ValueError(f"incomplete OOF predictions: {len(output)}/{expected}")
    return output


__all__ = [
    "SurvivalInstance", "suffix_requirement", "deterministic_balanced_folds",
    "fit_standardizer", "transform_standardized", "tune_ridge_grouped",
    "nested_grouped_oof",
]
