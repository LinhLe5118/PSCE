"""Baseline 1 -- behavior policy quality.

Never recomputes exact returns: reads them from Workstream 1's
``exact_oracle_results.csv`` (``WEEK2_ARTIFACT_CONTRACT.md`` sec 1), which is
the single place exact policy evaluation is allowed to happen.
"""

from __future__ import annotations

from typing import Any, Mapping

FEATURE_SET: tuple[str, ...] = (
    "behavior_return",
    "behavior_normalized_return",
    "behavior_optimality_gap",
)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in FEATURE_SET if name not in row]
    if missing:
        raise ValueError(f"exact_oracle_results row is missing {missing}")
    return {
        "mdp_id": row["mdp_id"],
        "policy_family": row["policy_family"],
        **{name: row[name] for name in FEATURE_SET},
    }


__all__ = ["FEATURE_SET", "extract"]
