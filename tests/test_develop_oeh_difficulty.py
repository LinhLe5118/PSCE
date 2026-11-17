"""Functional and fresh-process checks for the separate difficulty experiment."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.analysis.develop_oeh_difficulty import probabilities_from_cdf, score_probabilities
from scripts.analysis.develop_oeh_blend import fit_blend_weight
from scripts.analysis.confirm_oeh_difficulty import simultaneous_lower


ROOT = Path(__file__).resolve().parents[1]


def test_cli_starts_in_fresh_process():
    for module in ("scripts.analysis.develop_oeh_difficulty", "scripts.analysis.develop_oeh_blend",
                   "scripts.analysis.confirm_oeh_difficulty",
                   "scripts.analysis.compare_oeh_difficulty_e1"):
        result = subprocess.run([sys.executable, "-m", module, "--help"],
                                cwd=ROOT, capture_output=True)
        assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_six_threshold_projection_is_valid_and_preserves_ordered_rows():
    raw = np.array([[.1, .4, .2, .5, .5, .8], [.0, .1, .2, .3, .4, .5]])
    probs = probabilities_from_cdf(raw)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(probs[0].cumsum()[:6], [.1, .4, .4, .5, .5, .8])
    assert np.allclose(probs[1].cumsum()[:6], raw[1])
    assert np.all(np.isfinite(score_probabilities(probs, np.array([6, 2]))[0]))


def test_invalid_probability_inputs_fail_closed():
    with pytest.raises(ValueError):
        probabilities_from_cdf([[.1, .2]])
    with pytest.raises(ValueError):
        score_probabilities(np.ones((1, 7)), np.array([2]))


def test_global_blend_weight_uses_only_probabilities_and_labels():
    correct = np.eye(7)[[0, 6, 2]]
    weak = np.full((3, 7), 1 / 7)
    weight, loss = fit_blend_weight(correct, weak, np.array([0, 6, 2]))
    assert .99 < weight <= 1.
    assert loss < .01
    with pytest.raises(ValueError):
        fit_blend_weight(correct, np.ones((3, 7)), np.array([0, 6, 2]))


def test_simultaneous_mdp_bound_uses_shared_resamples():
    differences = np.array([[.2, .4], [.3, .3], [.4, .5], [.5, .4]])
    point, lower, critical = simultaneous_lower(differences, seed=7, draws=500)
    assert np.allclose(point, [.35, .4])
    assert np.all(lower <= point)
    assert np.isfinite(critical)
    with pytest.raises(ValueError):
        simultaneous_lower(np.array([[.1, .2]]))
