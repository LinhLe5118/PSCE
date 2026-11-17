"""Sustained events retain censoring and do not accept transient success."""
import numpy as np

from benchmarks.bridge.labels import sustained_event


def test_sustained_event_requires_every_larger_budget():
    assert sustained_event(np.array([0, 9, 7, 8, 10, 8])) == 3
    assert sustained_event(np.array([8, 8, 8, 8, 8, 8])) == 0
    assert sustained_event(np.array([0, 0, 0, 0, 0, 7])) == 6
