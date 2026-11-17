"""Exact original E1 route reproduction and matched scoring, without new data."""
from collections import defaultdict

import numpy as np

from return_selection import ReturnInstance, _fit_ridge, tune_return_ridge

METHODS = ('cfqi_tabular', 'iql_tabular', 'behavior_cloning_tabular')
FEATURES = ('B2_OEH3', 'B2_OEH3_LCOEH4')


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
