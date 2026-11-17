"""Check pilot-only comparator ingredients and fresh CLI startup."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from scripts.analysis.tabular_e1_quality_signals import pilot_signals


def test_pilot_signals_excludes_incomplete_trajectory_and_relabels_actions():
    arrays = {
        "state": np.array([0, 0, 1, 1]),
        "action": np.array([0, 1, 0, 1]),
        "reward": np.array([1., 0., 2., 100.]),
        "trajectory_id": np.array([0, 0, 1, 2]),
        "timestep": np.array([0, 1, 0, 0]),
        "done": np.array([False, True, True, False]),
    }
    first = pilot_signals(arrays, 2, 2)
    assert first["complete_trajectory_count"] == 2
    assert first["incomplete_trajectory_count"] == 1
    assert first["mean_complete_discounted_return"] == pytest.approx(1.5)
    assert first["observed_unique_sa"] == 4
    assert first["categorical_eas_adaptation"] == pytest.approx(1.)
    assert first["eri_observed_floor_adaptation"] == pytest.approx(1.)
    swapped = {**arrays, "action": 1 - arrays["action"]}
    second = pilot_signals(swapped, 2, 2)
    assert second == first


def test_undefined_eri_is_preserved():
    arrays = {
        "state": np.array([0]), "action": np.array([0]),
        "reward": np.array([1.]), "trajectory_id": np.array([0]),
        "timestep": np.array([0]), "done": np.array([True]),
    }
    assert pilot_signals(arrays, 1, 2)["eri_observed_floor_adaptation"] is None


def test_cli_help_starts_in_fresh_process():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.analysis.tabular_e1_quality_signals", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "prepare" in result.stdout
    assert "score" in result.stdout
