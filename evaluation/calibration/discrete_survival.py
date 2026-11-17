"""Discrete-time hazard calibrator (``PLAN_Final.md`` sec 6, Stage 3).

``h_j(x) = Pr(N_emp = N_j | N_emp >= N_j, x)``, ``logit h_j(x) = alpha_j +
beta^T x``. Fit by pooled logistic regression on the person-period expansion
of the data: each instance contributes one row per grid interval it survived,
with a 1/0 event indicator, and a right-censored instance contributes a row
for every grid interval with event=0 throughout -- this is exactly how a
censored observation contributes its correct partial-information likelihood
term instead of being imputed or dropped (mega-task section 11/section 21).

This is one concrete implementation behind ``SurvivalCalibrator``
(``evaluation/calibration/interface.py``). Regularization strength is a fixed
small ridge default, deliberately not tuned -- per mega-task section 9, the
Week 2 goal is proving the pipeline runs end-to-end, not producing a final
calibration model. Model selection (if any) must happen inside the training
fold only (``WEEK2_STATISTICAL_PLAN.md`` sec 1/4), which is the caller's
responsibility, not this class's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evaluation.calibration.interface import FeatureFrame

_DEFAULT_RIDGE = 1e-3


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Evaluate the logistic link without overflowing at separated folds."""

    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _expand_person_periods(
    grid_index: Sequence[int], observed: Sequence[bool], num_intervals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (subject_index, interval_index, event) person-period rows."""

    subjects: list[int] = []
    intervals: list[int] = []
    events: list[int] = []
    for subject, (position, is_observed) in enumerate(zip(grid_index, observed)):
        if position < 0 or position >= num_intervals:
            raise ValueError(f"grid_index[{subject}]={position} out of range [0,{num_intervals})")
        last_interval = position if is_observed else num_intervals - 1
        for interval in range(last_interval + 1):
            subjects.append(subject)
            intervals.append(interval)
            events.append(1 if (is_observed and interval == position) else 0)
    return np.asarray(subjects), np.asarray(intervals), np.asarray(events, dtype=np.float64)


@dataclass(frozen=True)
class _Fitted:
    alpha: np.ndarray  # shape (num_intervals,)
    beta: np.ndarray  # shape (num_features,)
    grid: tuple[int, ...]


class DiscreteTimeHazardCalibrator:
    def __init__(self, *, grid: Sequence[int], ridge: float = _DEFAULT_RIDGE, max_newton_steps: int = 50):
        if len(grid) < 2:
            raise ValueError("grid must have at least two levels.")
        self._grid = tuple(int(value) for value in grid)
        self._ridge = float(ridge)
        self._max_newton_steps = int(max_newton_steps)
        self._fitted: _Fitted | None = None

    def fit(
        self,
        features: FeatureFrame,
        grid_index: Sequence[int],
        observed: Sequence[bool],
    ) -> "DiscreteTimeHazardCalibrator":
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(grid_index):
            raise ValueError("features must be a 2D array with one row per instance.")
        num_intervals = len(self._grid)
        subjects, intervals, events = _expand_person_periods(grid_index, observed, num_intervals)
        num_features = matrix.shape[1]
        # Design matrix: one-hot interval intercepts + covariates.
        design = np.zeros((len(subjects), num_intervals + num_features), dtype=np.float64)
        design[np.arange(len(subjects)), intervals] = 1.0
        design[:, num_intervals:] = matrix[subjects]
        weights = np.zeros(num_intervals + num_features, dtype=np.float64)
        penalty = np.eye(num_intervals + num_features) * self._ridge
        penalty[:num_intervals, :num_intervals] = 0.0  # never regularize the baseline hazards
        for _ in range(self._max_newton_steps):
            logits = design @ weights
            probabilities = _stable_sigmoid(logits)
            gradient = design.T @ (events - probabilities) - penalty @ weights
            diagonal = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
            hessian = -(design.T * diagonal) @ design - penalty
            step = np.linalg.solve(hessian, gradient)
            weights = weights - step
            if float(np.max(np.abs(step))) < 1e-10:
                break
        self._fitted = _Fitted(
            alpha=weights[:num_intervals], beta=weights[num_intervals:], grid=self._grid
        )
        return self

    def predict_hazard(self, features: FeatureFrame) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Calibrator must be fit before predicting.")
        matrix = np.asarray(features, dtype=np.float64)
        linear = matrix @ self._fitted.beta
        logits = self._fitted.alpha[None, :] + linear[:, None]
        return _stable_sigmoid(logits)

    def predict_median_requirement(self, features: FeatureFrame) -> np.ndarray:
        hazard = self.predict_hazard(features)
        survival = np.cumprod(1.0 - hazard, axis=1)
        grid = np.asarray(self._fitted.grid, dtype=np.float64)
        result = np.full(hazard.shape[0], np.nan, dtype=np.float64)
        for i in range(hazard.shape[0]):
            below_half = np.flatnonzero(survival[i] <= 0.5)
            result[i] = grid[below_half[0]] if len(below_half) else np.nan
        return result


__all__ = ["DiscreteTimeHazardCalibrator"]
