"""Reference scoring, grouping and exclusion of a group's held-out labels."""
import copy

import numpy as np
import pytest

from oeh import core, router
from oeh.reference import METHODS, FEATURES, grouped_split, original_routes, score_routes, tune_return_ridge
from test_router import fixture_panel


def test_original_reference_excludes_own_test_labels_and_matches_wrappers():
    panel = fixture_panel()
    panel['candidate_methods'] = (*METHODS, 'extra')
    mdps = tuple(panel['outer'])
    a, forecast, _ = original_routes(panel, mdps)
    assert core.fit_routes(panel, mdps, original_v4e=True)[:2] == (a, forecast)
    assert router.fit_routes(panel, mdps, original_v4e=True)[:2] == (a, forecast)
    changed = copy.deepcopy(panel)
    for g in changed['runs']:
        if panel['outer'][g[0]] == 0:
            for m in METHODS:
                changed['runs'][g][m] = {seed:(-10., 0) for seed in range(10)}
    b, other, _ = original_routes(changed, mdps)
    for g in a:
        if panel['outer'][g[0]] == 0:
            assert a[g] == b[g] and forecast[g] == other[g]
    mean, clusters = score_routes(panel, a, mdps)
    np.testing.assert_allclose(mean, np.mean(list(clusters.values()), axis=0))
    a.pop(next(iter(a)))
    with pytest.raises(ValueError, match='Incomplete'):
        score_routes(panel, a, mdps)


def test_whole_mdp_multiplicities_and_zero_draws():
    panel = fixture_panel(); mdps = tuple(panel['outer'])
    weights = {m:int(i != 0) for i,m in enumerate(mdps)}
    w, groups = grouped_split(panel, mdps, weights)
    assert not any(g[0] == mdps[0] for g in groups)
    routes = {g:METHODS[0] for g in groups}
    actual, clusters = score_routes(panel, routes, mdps, multiplicities=w)
    np.testing.assert_allclose(actual, np.mean(list(clusters.values()), axis=0))
    for bad in ({}, {m:-1 for m in mdps}, {m:1.5 for m in mdps}):
        with pytest.raises(ValueError): grouped_split(panel, mdps, bad)
    # Empty test folds are possible in bootstrap draws and must not create routes.
    for g in list(panel['runs']):
        if panel['outer'][g[0]] == 0:
            weights[g[0]] = 0
    routes, _, _ = original_routes(panel, mdps, weights)
    assert all(panel['outer'][g[0]] != 0 for g in routes)


def test_reference_ridge_validation():
    with pytest.raises(ValueError):
        tune_return_ridge([], feature_set=FEATURES[0], ridge_grid=[.1], num_folds=4, salt='test')
