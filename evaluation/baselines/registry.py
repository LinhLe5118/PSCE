"""Uniform handle onto the six registered baselines.

All six run through the same downstream calibration/aggregation pipeline as
OEH (``PLAN_Final.md`` sec 6.1) -- this registry exists so a caller (e.g.
``pilot/aggregate.py``) can iterate over baselines without special-casing any
one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from evaluation.baselines import (
    behavior_quality,
    behavior_value,
    coverage,
    random_control,
    structural_only,
    visit_entropy,
)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    source_artifact: str | None
    feature_set: tuple[str, ...]
    extract: Callable[..., Mapping[str, Any]]


BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    "behavior_quality": BaselineSpec(
        "behavior_quality", "exact_oracle_results", behavior_quality.FEATURE_SET, behavior_quality.extract
    ),
    "coverage": BaselineSpec(
        "coverage", "behavior_policy_metadata", coverage.FEATURE_SET, coverage.extract
    ),
    "behavior_value": BaselineSpec(
        "behavior_value", "exact_oracle_results", behavior_value.FEATURE_SET, behavior_value.extract
    ),
    "visit_entropy": BaselineSpec(
        "visit_entropy", "behavior_policy_metadata", visit_entropy.FEATURE_SET, visit_entropy.extract
    ),
    "structural_only": BaselineSpec(
        "structural_only", "benchmark_metadata", structural_only.FEATURE_SET, structural_only.extract
    ),
    "random_control": BaselineSpec(
        "random_control", None, random_control.FEATURE_SET, random_control.extract
    ),
}


__all__ = ["BASELINE_REGISTRY", "BaselineSpec"]
