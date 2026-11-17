"""Matched core reference and original E1 return comparator."""
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence
import numpy as np
from .numerics import deterministic_balanced_folds, fit_standardizer, transform_standardized

METHODS = ('cfqi_tabular', 'iql_tabular', 'behavior_cloning_tabular')
FEATURES = ('B2_OEH3', 'B2_OEH3_LCOEH4')

@dataclass(frozen=True)
class ReturnInstance:
    mdp_id: str
    policy_family: str
    algorithm: str
    normalized_return: float
    features: Mapping[str, tuple[float, ...]]

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


def grouped_split(panel, mdps, multiplicities=None):
    weights = {mdp:1 for mdp in mdps} if multiplicities is None else dict(multiplicities)
    if set(weights) != set(mdps) or any(not isinstance(x, (int, np.integer)) or x < 0 for x in weights.values()):
        raise ValueError('Invalid whole-MDP multiplicities')
    groups = sorted(g for g in panel['runs'] if g[0] in weights and weights[g[0]] > 0)
    return weights, groups

def original_routes(panel, mdps, multiplicities=None):
    weights, groups = grouped_split(panel, mdps, multiplicities)
    choices, predictions, tuning = {}, {}, []
    for fold in range(5):
        train_groups = [g for g in groups if panel['outer'][g[0]] != fold]
        test_groups = [g for g in groups if panel['outer'][g[0]] == fold]
        if not test_groups:
            continue
        predicted = {g:{} for g in test_groups}
        for method in METHODS:
            train = []
            for g in train_groups:
                target = np.mean([x[0] for x in panel['runs'][g][method].values()])
                row = ReturnInstance(*g, method, float(target), panel['features'][(*g, method)])
                train.extend([row]*weights[g[0]])
            test = [ReturnInstance(*g, method, 0., panel['features'][(*g, method)]) for g in test_groups]
            blocks = []
            for feature in FEATURES:
                ridge, scores = tune_return_ridge(train, feature_set=feature, ridge_grid=panel['return_ridges'],
                    num_folds=4, salt=f'return|{fold}|{method}|{feature}')
                blocks.append(_fit_ridge(train, test, feature_set=feature, ridge=ridge))
                tuning.append(dict(fold=fold, method=method, feature=feature, ridge=ridge, inner_mse=scores[ridge]))
            for g, value in zip(test_groups, np.mean(blocks, axis=0)):
                predicted[g][method] = [float(value), 0.]
        for g in test_groups:
            choices[g] = METHODS[int(np.argmax([predicted[g][method][0] for method in METHODS]))]
            predictions[g] = predicted[g]
    return choices, predictions, tuning

def score_routes(panel, choices, mdps, *, multiplicities=None):
    weights, expected = grouped_split(panel, mdps, multiplicities)
    if set(choices) != set(expected) or any(m not in METHODS for m in choices.values()):
        raise ValueError('Incomplete scored MDP panel or invalid method')
    per_mdp = defaultdict(list)
    for g in sorted(choices):
        methods = (choices[g], METHODS[0], panel['v4e_decisions'][g])
        means = [np.mean(list(panel['runs'][g][method].values()), axis=0) for method in methods]
        per_mdp[g[0]].append(np.concatenate(means))
    clusters = {mdp:np.mean(values, axis=0) for mdp, values in per_mdp.items()}
    active = [mdp for mdp in mdps if weights[mdp] > 0]
    if set(active) != set(clusters):
        raise ValueError('Incomplete scored MDP panel')
    return np.average([clusters[m] for m in active], axis=0, weights=[weights[m] for m in active]), clusters
