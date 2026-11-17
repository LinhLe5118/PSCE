"""Numerical and grouping input boundaries, independent of research outcomes."""
import numpy as np
import pytest

from oeh.numerics import deterministic_balanced_folds, fit_standardizer, transform_standardized


def test_grouping_is_order_independent_and_balanced():
    groups = [f'm{i}' for i in range(13)]
    a = deterministic_balanced_folds(groups, num_folds=4, salt='fixed')
    assert a == deterministic_balanced_folds(groups[::-1]+groups, num_folds=4, salt='fixed')
    counts = [list(a.values()).count(i) for i in range(4)]
    assert max(counts)-min(counts) <= 1
    with pytest.raises(ValueError): deterministic_balanced_folds(groups, num_folds=1, salt='fixed')
    with pytest.raises(ValueError): deterministic_balanced_folds(['one'], num_folds=4, salt='fixed')


def test_standardization_fits_only_training_rows_and_handles_constants():
    mean, scale = fit_standardizer([[1., 5.], [3., 5.]])
    np.testing.assert_allclose(transform_standardized([[101., 5.]], mean, scale), [[99., 0.]])
    with pytest.raises(ValueError): fit_standardizer([])
    with pytest.raises(ValueError): fit_standardizer([[np.nan]])
    with pytest.raises(ValueError): transform_standardized([[np.nan]], mean, scale)
