from __future__ import annotations

from evaluation.aggregation.aggregate import aggregate, build_per_run_results, cluster_bootstrap_ci
from evaluation.aggregation.comparisons import (
    compare_k_dagger,
    compare_n_required,
    early_identification_ratio,
    stabilized_pilot_budget,
)
from evaluation.aggregation.folds import assign_folds

# The C2 layer (evaluation.aggregation.c2_analysis / .correlation) is
# deliberately not re-exported here. Its consumers import those modules
# directly, and a second export list for the same symbols only creates a way
# for the two to drift apart.

__all__ = [
    "aggregate",
    "assign_folds",
    "build_per_run_results",
    "cluster_bootstrap_ci",
    "compare_k_dagger",
    "compare_n_required",
    "early_identification_ratio",
    "stabilized_pilot_budget",
]
