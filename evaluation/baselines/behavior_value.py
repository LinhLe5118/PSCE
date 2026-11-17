"""Baseline 3 -- behavior value V(beta_hat).

Per ``WEEK2_EVALUATION_PROTOCOL.md`` sec 6.3, this is the exact evaluation of
the *unrestricted* behavior policy beta under the true P, r -- the same
quantity already produced by Workstream 1 as ``behavior_return`` in
``exact_oracle_results.csv``. It is not recomputed a second time under a
different name; this module documents the equivalence and exposes it under
the baseline's registered name so the aggregation layer can address it as
``behavior_value`` without a caller having to know the two names coincide.
"""

from __future__ import annotations

from typing import Any, Mapping

FEATURE_SET: tuple[str, ...] = ("behavior_value",)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    if "behavior_return" not in row:
        raise ValueError("exact_oracle_results row is missing 'behavior_return'")
    return {
        "mdp_id": row["mdp_id"],
        "policy_family": row["policy_family"],
        "behavior_value": row["behavior_return"],
    }


__all__ = ["FEATURE_SET", "extract"]
