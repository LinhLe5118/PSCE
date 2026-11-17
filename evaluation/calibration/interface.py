"""Calibrator abstraction (mega-task section 9: prove the pipeline runs
end-to-end; do not claim any one model is final).

``evaluation.aggregation`` and ``evaluation.decision_gate`` depend only on
this Protocol, never on ``DiscreteTimeHazardCalibrator`` directly, so a
different survival model can be swapped in later without touching either.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

FeatureFrame = Sequence[Mapping[str, float]]


class SurvivalCalibrator(Protocol):
    def fit(
        self,
        features: FeatureFrame,
        grid_index: Sequence[int],
        observed: Sequence[bool],
    ) -> "SurvivalCalibrator":
        """``grid_index[i]`` is the 0-based grid position of the event for an
        observed instance, or the last grid position for a right-censored one
        (it survived through the whole registered grid)."""
        ...

    def predict_median_requirement(self, features: FeatureFrame) -> np.ndarray:
        ...

    def predict_hazard(self, features: FeatureFrame) -> np.ndarray:
        """Returns shape ``(len(features), len(grid))``."""
        ...


__all__ = ["FeatureFrame", "SurvivalCalibrator"]
