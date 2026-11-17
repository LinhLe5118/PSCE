"""Extrapolation-endpoint scorer (task card T1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import extrapolation_endpoint as X


def test_threshold_masks_cover_the_budgets_at_and_above_the_pilot():
    assert X.FULL == (0, 1, 2, 3, 4, 5)
    assert X.GE == (3, 4, 5) and X.GT == (4, 5)


def test_full_mask_is_the_standard_integrated_brier():
    rng = np.random.default_rng(0)
    p, y = rng.dirichlet(np.ones(7), 50), rng.integers(0, 7, 50)
    cdf = np.cumsum(p, 1)[:, :6]
    np.testing.assert_allclose(X.ibs(p, y), ((cdf - (y[:, None] <= np.arange(6))) ** 2).mean(1))


def test_coarsening_merges_the_retrospective_classes():
    rng = np.random.default_rng(1)
    p, y = rng.dirichlet(np.ones(7), 40), rng.integers(0, 7, 40)
    q, z = X.coarsen(p, y)
    np.testing.assert_allclose(q.sum(1), 1)
    np.testing.assert_allclose(q[:, 0], p[:, :4].sum(1))
    assert set(np.unique(z)) <= {0, 1, 2, 3} and (z[y <= 3] == 0).all() and (z[y == 6] == 3).all()


def test_a_forecast_that_differs_only_below_the_pilot_scores_the_same_in_the_extrapolation_region():
    rng = np.random.default_rng(2)
    p, y = rng.dirichlet(np.ones(7), 30), rng.integers(0, 7, 30)
    r = p.copy()
    mass = r[:, :4].sum(1)
    r[:, :4] = 0.0
    r[:, 3] = mass  # all retrospective mass moved to x10
    a, b = X.row_scores(p, y), X.row_scores(r, y)
    np.testing.assert_allclose(a["ibs_ge"], b["ibs_ge"])
    np.testing.assert_allclose(a["ll_ext"], b["ll_ext"])
    assert not np.allclose(a["ibs"], b["ibs"])


def test_recorded_scores_reproduce_the_sealed_summaries():
    out = X.OUTPUT
    eq = pd.read_csv(out / "equivalence.csv")
    assert len(eq) >= 150 and eq[["ibs_diff", "ll_diff"]].to_numpy().max() <= 1e-12
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive"
    for name, digest in manifest["outputs_sha256"].items():
        assert X.sha(out / name) == digest
