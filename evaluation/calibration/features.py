"""Feature builder for the calibration stage.

The most plausible place for an oracle leak to enter is here, so every call
goes through ``assert_no_oracle_leakage`` before touching numpy
(``docs/week2/WEEK2_EVALUATION_PROTOCOL.md`` sec 7). Unlike the
structural-only baseline, this builder is allowed pilot-derived features
(coverage, visit entropy, OEH's own ``k_dagger_oeh``/``n_req_hat`` outputs) --
only true oracle fields and the ``N_emp`` label itself are forbidden.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from evaluation.calibration.interface import FeatureFrame
from evaluation.validation.leakage import assert_no_oracle_leakage


def build_features(
    rows: FeatureFrame, feature_names: Sequence[str], *, context: str
) -> np.ndarray:
    assert_no_oracle_leakage(feature_names, context=context)
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            if name not in row:
                raise ValueError(f"{context}: row {i} is missing feature '{name}'")
            value = float(row[name])
            if not np.isfinite(value):
                raise ValueError(f"{context}: row {i} feature '{name}' is not finite")
            matrix[i, j] = value
    return matrix


__all__ = ["build_features"]
