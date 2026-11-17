"""Baseline 2 -- coverage.

Only the registered coverage fields already computed by
``scripts/data/generate_offline_datasets.py::coverage_metrics`` and mirrored in
``behavior_policy_metadata.csv``. Per mega-task section 6: "if the repository
only defines a subset, follow the frozen protocol, do not extend it" -- no new
coverage submetric (e.g. a separate transition-level coverage number) is
introduced here.
"""

from __future__ import annotations

from typing import Any, Mapping

FEATURE_SET: tuple[str, ...] = (
    "observed_reachable_sa_fraction",
    "observed_sa_fraction_all",
    "observed_policy_support_fraction",
    "fraction_policy_support_at_least_n_min",
    "minimum_policy_support_count",
)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in FEATURE_SET if name not in row]
    if missing:
        raise ValueError(f"behavior_policy_metadata row is missing {missing}")
    return {
        "mdp_id": row["mdp_id"],
        "policy_family": row["policy_family"],
        **{name: row[name] for name in FEATURE_SET},
    }


__all__ = ["FEATURE_SET", "extract"]
