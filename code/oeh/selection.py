"""Parameter-free learner selection from posterior-predictive outcomes.

Every registered learner is scored by the same rule: it is trained on the same
simulated posterior datasets and evaluated in the draw's own MDP. The selected
learner maximizes the posterior-predictive probability of reaching the
protocol threshold at the decision budget (the thesis decision rule);
expected normalized return breaks ties, then registry order. No weight,
confidence level or switching margin enters the rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tabular import SUCCESS_THRESHOLD


@dataclass(frozen=True)
class Selection:
    learner: str
    success_probability: dict[str, float]
    expected_return: dict[str, float]
    draws: int


def select_learner(normalized_return: np.ndarray, learners: Sequence[str], *,
                   threshold: float = SUCCESS_THRESHOLD) -> Selection:
    """Choose one learner from ``(draws, learners)`` paired simulated returns."""
    values = np.asarray(normalized_return, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(learners) or not len(learners):
        raise ValueError("Expected a (draws, learners) return matrix")
    if not values.shape[0]:
        raise ValueError("Selection needs at least one posterior draw")
    if not np.all(np.isfinite(values)):
        raise ValueError("Simulated returns must be finite")
    success = (values >= threshold).mean(0)
    expected = values.mean(0)
    order = sorted(range(len(learners)), key=lambda i: (-success[i], -expected[i], i))
    return Selection(learner=learners[order[0]],
                     success_probability={name: float(success[i]) for i, name in enumerate(learners)},
                     expected_return={name: float(expected[i]) for i, name in enumerate(learners)},
                     draws=int(values.shape[0]))
