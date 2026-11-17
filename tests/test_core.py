"""Independent explicit polynomial solution, quadratic control, no test labels."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from oeh.core import (FEATURES, METHODS, fit_routes, joint_routes, kernel_predictions, paired_targets, simplex)


def test_weighted_kernel_agrees_with_independent_explicit_quadratic_ridge():
    x = np.array([[-2.], [-1.], [.5], [2.]])
    query = np.array([[-1.5], [1.5]])
    y = np.column_stack((x[:, 0]**2, x[:, 0]+2))
    w = np.array([1, 3, 2, 1]); ridge = .2
    actual = kernel_predictions(x, query, y, w, [ridge])[0]
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    z = ((x-mean)/scale)[:, 0]; q = ((query-mean)/scale)[:, 0]
    # (1+z*q)^2 = 1 + 2*z*q + z^2*q^2: this primal solution
    # uses explicit monomials rather than the candidate's centered kernel.
    design = np.column_stack((np.ones(len(x)), np.sqrt(2)*z, z**2))
    test = np.column_stack((np.ones(len(q)), np.sqrt(2)*q, q**2))
    penalty = np.diag([0., ridge, ridge])
    coefficients = np.linalg.solve(design.T@(w[:, None]*design)+penalty, design.T@(w[:, None]*y))
    np.testing.assert_allclose(actual, test@coefficients, rtol=1e-10, atol=1e-10)


def test_quadratic_interaction_control_and_outcome_translation():
    x = np.array([[a, b] for a in (-2., -1., 1., 2.) for b in (-2., -1., 1., 2.)])
    query = np.array([[.5, 1.5], [-.5, 1.5]])
    y = (x[:, 0]*x[:, 1])[:, None]
    predicted = kernel_predictions(x, query, y, np.ones(len(x), int), [1e-4])[0, :, 0]
    np.testing.assert_allclose(predicted, query[:, 0]*query[:, 1], atol=1e-4)
    shifted = kernel_predictions(x, query, y+7, np.ones(len(x), int), [1e-4])[0, :, 0]
    np.testing.assert_allclose(shifted, predicted+7, atol=1e-10)


def test_repeated_rows_equal_weighted_fit_and_feature_relabeling():
    x = np.array([[0., 1.], [1., 2.], [2., -1.], [-1., 3.]])
    y = np.array([[1.], [2.], [3.], [4.]])
    w = np.array([1, 2, 3, 1]); query = np.array([[.5, 1.]])
    a = kernel_predictions(x, query, y, w, [.1, 1.])
    b = kernel_predictions(np.repeat(x, w, axis=0), query, np.repeat(y, w, axis=0), np.ones(w.sum(), int), [.1, 1.])
    np.testing.assert_allclose(a, b, atol=1e-10)
    c = kernel_predictions(x[:, ::-1], query[:, ::-1], y, w, [.1, 1.])
    np.testing.assert_allclose(a, c, atol=1e-10)


def test_constant_columns_and_null_joint_gains_remain_finite():
    predicted = kernel_predictions(np.ones((5, 3)), np.ones((2, 3)), np.ones((5, 4)), np.ones(5, int), [.1, 1.])
    np.testing.assert_allclose(predicted, 1.)
    route, evidence = joint_routes(np.zeros((2, 3)), np.zeros((2, 3)))
    assert not route.any() and evidence['predicted_joint_gains'] == [0., 0.]


@pytest.mark.parametrize('case', ['nonfinite', 'bad_weights', 'bad_grid', 'query_shape'])
def test_invalid_kernel_input_rejected(case):
    x = np.ones((5, 2)); q = np.ones((2, 2)); y = np.ones((5, 4)); w = np.ones(5, int); grid = [.1]
    if case == 'nonfinite': x[0, 0] = np.nan
    if case == 'bad_weights': w[0] = 0
    if case == 'bad_grid': grid = [0]
    if case == 'query_shape': q = np.ones((2, 3))
    with pytest.raises(ValueError):
        kernel_predictions(x, q, y, w, grid)


def test_paired_categories_and_joint_scale_invariance():
    reference = {0:(.98, 1), 1:(.4, 0), 2:(.98, 1), 3:(.4, 0)}
    alternative = {3:(.4, 0), 2:(.4, 0), 1:(.98, 1), 0:(.98, 1)}
    np.testing.assert_allclose(paired_targets(reference, alternative), [0., .25, .25, .5])
    with pytest.raises(ValueError): paired_targets(reference, {})
    with pytest.raises(ValueError): paired_targets({0:(.4, 2)}, {0:(.4, 0)})
    r, s = np.array([[0, .9, .6]]), np.array([[0, .001, .4]])
    a, _ = joint_routes(r, s); b, _ = joint_routes(2*r, 5*s)
    assert a.tolist() == b.tolist() == [2]
    np.testing.assert_allclose(simplex([[-1, 2, 0]]), [[0, 1, 0]])
    with pytest.raises(ValueError): simplex([[np.nan, 0, 0]])
    with pytest.raises(ValueError): joint_routes([[0, 1]], [[0, 1]], [0])


def test_test_outcomes_cannot_change_own_fold_decisions():
    panel = dict(runs={}, features={}, outer={}, v4e_decisions={}, return_ridges=(.1, 1.))
    for i in range(15):
        mdp = f'fixture{i:02}'; panel['outer'][mdp] = i%5
        for behavior in ('a', 'b'):
            g = (mdp, behavior); panel['runs'][g] = {}; panel['v4e_decisions'][g] = METHODS[0]
            for j, method in enumerate(METHODS):
                value = .9+.01*(i%3)+.02*j
                panel['runs'][g][method] = {seed:(value, int(value >= .95)) for seed in range(10)}
                panel['features'][(*g, method)] = {f:(i/15, float(behavior == 'a'), float(j)) for f in FEATURES}
    mdps = tuple(panel['outer']); a, predicted, _ = fit_routes(panel, mdps)
    changed = copy.deepcopy(panel)
    for g in changed['runs']:
        if panel['outer'][g[0]] == 0:
            for m in METHODS:
                changed['runs'][g][m] = {seed:(-10., 0) for seed in range(10)}
    b, other, _ = fit_routes(changed, mdps)
    for g in a:
        if panel['outer'][g[0]] == 0:
            assert a[g] == b[g] and predicted[g] == other[g]
