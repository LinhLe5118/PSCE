"""E1 adapter agrees with the unchanged registered return implementation."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from evaluation.aggregation.folds import assign_folds
from evaluation.e1_routing import FEATURES, METHODS, original_routes, score_routes
from return_selection import ReturnInstance, nested_return_oof


def fixture_panel():
    mdps = tuple(f'fixture{i:02}' for i in range(15))
    panel = dict(runs={}, features={}, outer=assign_folds(mdps, num_folds=5), v4e_decisions={}, return_ridges=(.1, 1.))
    for i, mdp in enumerate(mdps):
        for behavior in ('a', 'b'):
            group = (mdp, behavior); panel['runs'][group] = {}; panel['v4e_decisions'][group] = METHODS[0]
            for j, method in enumerate(METHODS):
                value = .7+.01*i+.02*j
                panel['runs'][group][method] = {seed:(value, int(value >= .95)) for seed in range(10)}
                panel['features'][(*group, method)] = {f:(i/15, float(j)) for f in FEATURES}
    return panel, mdps


def test_original_adapter_matches_registered_nested_predictions_and_routes():
    panel, mdps = fixture_panel()
    rows = [ReturnInstance(*g, m, float(np.mean([v[0] for v in runs[m].values()])), panel['features'][(*g, m)])
            for g, runs in panel['runs'].items() for m in METHODS]
    registered = nested_return_oof(rows, feature_sets=FEATURES, ridge_grid=panel['return_ridges'])
    choices, predicted, _ = original_routes(panel, mdps)
    expected = {}
    for row in registered:
        expected.setdefault((row['mdp_id'], row['policy_family'], row['algorithm']), []).append(row['predicted_return'])
    for g in choices:
        values = [np.mean(expected[(*g, m)]) for m in METHODS]
        assert choices[g] == METHODS[int(np.argmax(values))]
        np.testing.assert_allclose([predicted[g][m][0] for m in METHODS], values, rtol=0, atol=1e-12)


def test_original_adapter_does_not_use_own_test_outcomes_and_scores_full_grid():
    panel, mdps = fixture_panel(); a, forecasts, _ = original_routes(panel, mdps)
    changed = copy.deepcopy(panel)
    for g in changed['runs']:
        if panel['outer'][g[0]] == 0:
            for m in METHODS:
                changed['runs'][g][m] = {seed:(-5., 0) for seed in range(10)}
    b, other, _ = original_routes(changed, mdps)
    for g in a:
        if panel['outer'][g[0]] == 0:
            assert a[g] == b[g] and forecasts[g] == other[g]
    mean, clusters = score_routes(panel, a, mdps)
    np.testing.assert_allclose(mean, np.mean(list(clusters.values()), axis=0))
    a.pop(next(iter(a)))
    with pytest.raises(ValueError, match='Incomplete'):
        score_routes(panel, a, mdps)
