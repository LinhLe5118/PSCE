"""Decision-gate thresholds.

Every threshold below is quoted from ``PLAN_Final.md`` sec 8.1/8.2/10 (already
frozen there as candidate pending the 21/08 sign-off, mega-task section 3/8).
None is invented here. Where ``protocol_config.json`` does not yet carry a
machine-readable copy of a threshold, it is marked ``[DECISION REQUIRED]`` in
``docs/week2/WEEK2_EVALUATION_PROTOCOL.md`` sec 8 -- this module is the single
place that would need to change if/when that section is added, so the
prose-vs-code duplication is at least confined to one file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProvisionalThresholds:
    k1a_median_ratio_max: float = 0.25  # PLAN_Final.md sec 8.1, K1a
    k2_max_factor_error: float = 2.0  # sec 8.1, K2
    k3_within_factor: float = 3.0  # sec 8.1, K3
    k3_min_rate: float = 0.60  # sec 8.1 / 8.2 criterion 5
    k4_within_depth: int = 2  # sec 8.1, K4
    k4_min_rate: float = 0.80  # sec 8.1, K4
    max_abstention_rate_on_solvable: float = 0.50  # sec 8.2 criterion 6
    max_censoring_rate_before_grid_expansion: float = 0.40  # sec 7 / sec 10
    weak_k1a_boundary: float = 0.50  # sec 10, "K1a dat yeu (N_k ~= N_emp/2)"


PROVISIONAL_THRESHOLDS = ProvisionalThresholds()

__all__ = ["PROVISIONAL_THRESHOLDS", "ProvisionalThresholds"]
