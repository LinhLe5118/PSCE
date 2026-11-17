"""Baseline 4 -- visitation entropy.

Data-only: ``normalized_visitation_entropy`` is computed from observed pilot
visitation counts only (``scripts/data/generate_offline_datasets.py::coverage_metrics``),
never from information unavailable at prediction time (no exact occupancy,
no oracle policy).
"""

from __future__ import annotations

from typing import Any, Mapping

FEATURE_SET: tuple[str, ...] = ("normalized_visitation_entropy",)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    if "normalized_visitation_entropy" not in row:
        raise ValueError("behavior_policy_metadata row is missing 'normalized_visitation_entropy'")
    return {
        "mdp_id": row["mdp_id"],
        "policy_family": row["policy_family"],
        "normalized_visitation_entropy": row["normalized_visitation_entropy"],
    }


__all__ = ["FEATURE_SET", "extract"]
