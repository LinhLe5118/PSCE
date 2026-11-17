"""Baseline 5 -- structural-only predictor.

Answers the sharpest objection: is ``k_dagger``/``N_hat_req`` just a rewrite
of a structural MDP constant? Uses only the closed structural allowlist
(``evaluation.validation.leakage.STRUCTURAL_FEATURE_ALLOWLIST``) drawn from
``benchmark_metadata.csv``. No pilot outcome, no ``N_emp``, no exact
``k_dagger``, no OEH output, no test-set performance may enter -- enforced by
``assert_no_oracle_leakage`` with the closed allowlist, not just the
pattern-based default.
"""

from __future__ import annotations

from typing import Any, Mapping

from evaluation.validation.leakage import STRUCTURAL_FEATURE_ALLOWLIST, assert_no_oracle_leakage

FEATURE_SET = STRUCTURAL_FEATURE_ALLOWLIST


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    available = [name for name in FEATURE_SET if name in row]
    assert_no_oracle_leakage(available, context="structural_only baseline", allowlist=STRUCTURAL_FEATURE_ALLOWLIST)
    if not available:
        raise ValueError(
            "benchmark_metadata row exposes none of the structural allowlist columns"
        )
    return {"mdp_id": row["mdp_id"], **{name: row[name] for name in available}}


__all__ = ["FEATURE_SET", "extract"]
