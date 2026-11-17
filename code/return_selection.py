"""Nested, MDP-grouped return prediction for leakage-safe learner selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from evaluation.aggregation.folds import assign_folds
from stochastic_evaluation import (
    deterministic_balanced_folds, fit_standardizer, transform_standardized,
)


@dataclass(frozen=True)
class ReturnInstance:
    mdp_id: str
    policy_family: str
    algorithm: str
    normalized_return: float
    features: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class AdvantageInstance:
    """One paired learner contrast on the same MDP and behavior policy.

    ``advantage`` is alternative return minus reference return.  Pairing
    removes the large environment/policy component shared by both outcomes,
    which is the quantity a selector does not need to predict.
    """

    mdp_id: str
    policy_family: str
    reference_algorithm: str
    alternative_algorithm: str
    advantage: float
    features: Mapping[str, tuple[float, ...]]

    @property
    def normalized_return(self) -> float:
        """Expose the paired outcome to the common ridge implementation."""

        return self.advantage


def _matrix(rows: Sequence[ReturnInstance], feature_set: str) -> np.ndarray:
    try:
        result = np.asarray([row.features[feature_set] for row in rows], dtype=float)
    except KeyError as exc:
        raise ValueError(f"missing feature set {feature_set}") from exc
    if result.ndim != 2 or result.shape[0] != len(rows) or not np.all(np.isfinite(result)):
        raise ValueError("invalid return feature matrix")
    return result


def _fit_ridge(train: Sequence[ReturnInstance], test: Sequence[ReturnInstance],
               *, feature_set: str, ridge: float) -> np.ndarray:
    x_train = _matrix(train, feature_set)
    x_test = _matrix(test, feature_set)
    mean, scale = fit_standardizer(x_train)
    design = np.column_stack((np.ones(len(train)),
                              transform_standardized(x_train, mean, scale)))
    test_design = np.column_stack((np.ones(len(test)),
                                   transform_standardized(x_test, mean, scale)))
    penalty = np.diag([0.0] + [float(ridge)] * x_train.shape[1])
    outcome = np.asarray([row.normalized_return for row in train], dtype=float)
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ outcome)
    prediction = test_design @ coefficients
    if not np.all(np.isfinite(prediction)):
        raise ValueError("return model produced non-finite predictions")
    return prediction


def tune_return_ridge(train: Sequence[ReturnInstance], *, feature_set: str,
                      ridge_grid: Sequence[float], num_folds: int,
                      salt: str) -> tuple[float, dict[float, float]]:
    if not train or not ridge_grid:
        raise ValueError("train and ridge_grid must be nonempty")
    folds = deterministic_balanced_folds(
        [row.mdp_id for row in train], num_folds=num_folds, salt=salt)
    scores = {}
    for raw_ridge in ridge_grid:
        ridge = float(raw_ridge)
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge values must be finite and positive")
        errors = []
        for fold in range(num_folds):
            inner_train = [row for row in train if folds[row.mdp_id] != fold]
            inner_test = [row for row in train if folds[row.mdp_id] == fold]
            predicted = _fit_ridge(inner_train, inner_test,
                                   feature_set=feature_set, ridge=ridge)
            observed = np.asarray([row.normalized_return for row in inner_test])
            errors.extend(np.square(predicted - observed))
        scores[ridge] = float(np.mean(errors))
    # Exact ties choose stronger regularization, a conservative deterministic
    # rule fixed independently of the held-out outer fold.
    selected = min(scores, key=lambda value: (scores[value], -value))
    return selected, scores


def nested_return_oof(instances: Sequence[ReturnInstance], *,
                      feature_sets: Sequence[str], ridge_grid: Sequence[float],
                      outer_folds: int = 5, inner_folds: int = 4):
    rows = list(instances)
    if not rows:
        raise ValueError("instances must be nonempty")
    outer = assign_folds({row.mdp_id for row in rows}, num_folds=outer_folds)
    output = []
    for algorithm in sorted({row.algorithm for row in rows}):
        panel = [row for row in rows if row.algorithm == algorithm]
        for feature_set in feature_sets:
            for fold in range(outer_folds):
                train = [row for row in panel if outer[row.mdp_id] != fold]
                test = [row for row in panel if outer[row.mdp_id] == fold]
                if not test:
                    continue
                selected, scores = tune_return_ridge(
                    train, feature_set=feature_set, ridge_grid=ridge_grid,
                    num_folds=inner_folds,
                    salt=f"return|{fold}|{algorithm}|{feature_set}")
                predictions = _fit_ridge(
                    train, test, feature_set=feature_set, ridge=selected)
                for row, prediction in zip(test, predictions):
                    output.append({
                        "mdp_id": row.mdp_id,
                        "policy_family": row.policy_family,
                        "algorithm": row.algorithm,
                        "feature_set": feature_set,
                        "outer_fold": fold,
                        "selected_ridge": selected,
                        "inner_mse": scores[selected],
                        "predicted_return": float(prediction),
                        "normalized_return": float(row.normalized_return),
                    })
    if len(output) != len(rows) * len(feature_sets):
        raise ValueError("incomplete return OOF panel")
    return output


def _inner_oof_predictions(
    train: Sequence[AdvantageInstance], *, feature_set: str, ridge: float,
    num_folds: int, salt: str,
) -> list[tuple[AdvantageInstance, float]]:
    """Return honest inner-fold predictions for decision calibration."""

    folds = deterministic_balanced_folds(
        [row.mdp_id for row in train], num_folds=num_folds, salt=salt)
    output = []
    for fold in range(num_folds):
        inner_train = [row for row in train if folds[row.mdp_id] != fold]
        inner_test = [row for row in train if folds[row.mdp_id] == fold]
        predicted = _fit_ridge(
            inner_train, inner_test, feature_set=feature_set, ridge=ridge)
        for row, prediction in zip(inner_test, predicted):
            output.append((row, float(prediction)))
    if len(output) != len(train):
        raise ValueError("incomplete inner OOF prediction panel")
    return output


def _tune_switch_threshold(
    predictions: Sequence[tuple[AdvantageInstance, float]],
) -> tuple[float, float, int]:
    """Choose a nonnegative switch margin by inner-OOF decision utility.

    Candidate boundaries are induced by the observed OOF scores, so there is
    no hand-selected numerical grid.  Exact utility ties choose the larger
    margin and therefore fewer unsupported switches.
    """

    grouped: dict[tuple[str, str], list[tuple[AdvantageInstance, float]]] = {}
    for row, score in predictions:
        grouped.setdefault((row.mdp_id, row.policy_family), []).append((row, score))
    candidates = {0.0}
    candidates.update(float(score) for _, score in predictions if score >= 0)
    if candidates:
        candidates.add(float(np.nextafter(max(candidates), math.inf)))
    evaluated = []
    for threshold in sorted(candidates):
        gains = []
        switches = 0
        for panel in grouped.values():
            row, score = max(panel, key=lambda item: (
                item[1], item[0].alternative_algorithm))
            if score > threshold:
                gains.append(float(row.advantage))
                switches += 1
            else:
                gains.append(0.0)
        evaluated.append((float(np.mean(gains)), threshold, switches))
    utility, threshold, switches = max(
        evaluated, key=lambda item: (item[0], item[1]))
    return threshold, utility, switches


def nested_advantage_oof(
    instances: Sequence[AdvantageInstance], *, feature_sets: Sequence[str],
    ridge_grid: Sequence[float], algorithms: Sequence[str],
    outer_folds: int = 5, inner_folds: int = 4,
    lower_quantile: float = .05,
) -> list[dict[str, object]]:
    """Predict paired learner advantages with a training-only fallback.

    The best fixed reference learner is re-selected inside every outer
    training split.  Ridge strength and the conservative residual correction
    are also estimated without the outer test groups.  For the correction,
    each training MDP contributes its minimum inner-OOF residual across
    behavior policies; this prevents five policies from masquerading as five
    independent uncertainty observations.
    """

    rows = list(instances)
    methods = tuple(dict.fromkeys(str(value) for value in algorithms))
    if not rows or len(methods) < 2 or not feature_sets:
        raise ValueError("instances, feature_sets, and at least two algorithms required")
    if not 0 < lower_quantile < .5:
        raise ValueError("lower_quantile must lie strictly between 0 and .5")
    outer = assign_folds({row.mdp_id for row in rows}, num_folds=outer_folds)
    groups = {(row.mdp_id, row.policy_family) for row in rows}
    outcome: dict[tuple[str, str, str], float] = {}
    for row in rows:
        outcome[(row.mdp_id, row.policy_family, row.reference_algorithm)] = 0.0
        outcome[(row.mdp_id, row.policy_family, row.alternative_algorithm)] = (
            outcome.get((row.mdp_id, row.policy_family,
                         row.reference_algorithm), 0.0) + row.advantage)
    # Reconstructing absolute outcomes from arbitrary pair rows is ambiguous.
    # The best-fixed learner is invariant to a common per-group offset, so use
    # the mean pairwise advantage: E[R_a-R_b] ranks algorithms identically to
    # E[R_a] when the ordered panel is complete.
    expected_pairs = len(groups) * len(methods) * (len(methods) - 1)
    keys = {(r.mdp_id, r.policy_family, r.reference_algorithm,
             r.alternative_algorithm) for r in rows}
    if len(rows) != expected_pairs or len(keys) != expected_pairs:
        raise ValueError("advantage panel must contain every ordered learner pair")

    output: list[dict[str, object]] = []
    for fold in range(outer_folds):
        train_groups = {group for group in groups if outer[group[0]] != fold}
        test_groups = {group for group in groups if outer[group[0]] == fold}
        mean_pair_score = {}
        for algorithm in methods:
            values = [
                row.advantage
                for row in rows
                if (row.mdp_id, row.policy_family) in train_groups
                and row.alternative_algorithm == algorithm
            ]
            mean_pair_score[algorithm] = float(np.mean(values))
        reference = max(methods, key=lambda name: (mean_pair_score[name], name))
        threshold_training: dict[
            str, list[tuple[AdvantageInstance, float]]
        ] = {feature_set: [] for feature_set in feature_sets}
        for alternative in methods:
            if alternative == reference:
                continue
            panel = [row for row in rows
                     if row.reference_algorithm == reference
                     and row.alternative_algorithm == alternative]
            train = [row for row in panel
                     if (row.mdp_id, row.policy_family) in train_groups]
            test = [row for row in panel
                    if (row.mdp_id, row.policy_family) in test_groups]
            for feature_set in feature_sets:
                salt = (f"advantage|outer={fold}|reference={reference}|"
                        f"alternative={alternative}|features={feature_set}")
                selected, scores = tune_return_ridge(
                    train, feature_set=feature_set, ridge_grid=ridge_grid,
                    num_folds=inner_folds, salt=salt)
                predicted = _fit_ridge(
                    train, test, feature_set=feature_set, ridge=selected)
                inner_predictions = _inner_oof_predictions(
                    train, feature_set=feature_set, ridge=selected,
                    num_folds=inner_folds, salt=salt)
                threshold_training[feature_set].extend(inner_predictions)
                residuals: dict[str, list[float]] = {}
                for inner_row, inner_prediction in inner_predictions:
                    residuals.setdefault(inner_row.mdp_id, []).append(
                        float(inner_row.advantage - inner_prediction))
                cluster_minima = np.asarray(
                    [min(values) for values in residuals.values()], dtype=float)
                # ``method='lower'`` chooses an observed training-cluster
                # residual and is deterministic across NumPy interpolation
                # conventions.  This is a conservative score, not a formal
                # finite-sample conformal guarantee.
                correction = float(np.quantile(
                    cluster_minima, lower_quantile, method="lower"))
                for row, point in zip(test, predicted):
                    output.append({
                        "mdp_id": row.mdp_id,
                        "policy_family": row.policy_family,
                        "reference_algorithm": reference,
                        "alternative_algorithm": alternative,
                        "feature_set": feature_set,
                        "outer_fold": fold,
                        "selected_ridge": selected,
                        "inner_mse": scores[selected],
                        "predicted_advantage": float(point),
                        "conservative_advantage": float(point + correction),
                        "residual_correction": correction,
                        "advantage": float(row.advantage),
                    })
        for feature_set in feature_sets:
            threshold, utility, switches = _tune_switch_threshold(
                threshold_training[feature_set])
            for row in output:
                if (row["outer_fold"] == fold
                        and row["feature_set"] == feature_set):
                    row["decision_threshold"] = threshold
                    row["threshold_training_utility"] = utility
                    row["threshold_training_switches"] = switches
    expected = len(groups) * (len(methods) - 1) * len(feature_sets)
    if len(output) != expected:
        raise ValueError("incomplete advantage OOF panel")
    return output


__all__ = [
    "AdvantageInstance", "ReturnInstance", "nested_advantage_oof",
    "nested_return_oof", "tune_return_ridge",
]
