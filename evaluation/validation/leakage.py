"""Mechanical leakage checks (necessary, not sufficient -- see
``docs/week2/WEEK2_EVALUATION_PROTOCOL.md`` sec 7).

Catches a column literally named like an oracle field ending up in a feature
set passed to the OEH estimator's callers, the structural-only baseline, or
the calibration feature builder. It does not prove the absence of leakage in
general -- a disguised leak (e.g. a feature that happens to correlate with
``k_dagger_exact`` without being named after it) is outside what a name-based
check can catch.
"""

from __future__ import annotations

from typing import Iterable

# Closed feature namelist for the structural-only baseline
# (docs/week2/WEEK2_EVALUATION_PROTOCOL.md sec 6.5). Nothing derived from the
# pilot, from N_emp, or from any OEH/exact output belongs here.
STRUCTURAL_FEATURE_ALLOWLIST: tuple[str, ...] = (
    "num_states",
    "num_actions",
    "gamma",
    "branching_factor",
    "mixing_time_proxy",
    "horizon",
    "k_max",
    "diameter",
)

_FORBIDDEN_PREFIXES: tuple[str, ...] = ("exact_", "oracle_", "true_")
# Oracle-derived quantities and the N_emp label itself. Deliberately does NOT
# include k_dagger_oeh / n_req_hat / effective_horizon / oeh_status / coverage
# / visit-entropy fields: those are pilot-derived, data-only, and are exactly
# the legitimate predictors the calibration stage (WEEK2_STATISTICAL_PLAN.md
# sec 4) is built to use. The structural-only baseline gets a *stricter*,
# closed-list check (STRUCTURAL_FEATURE_ALLOWLIST) instead of relying on this
# pattern list, because "not an oracle field" is not the same bar as
# "structural."
_FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "k_dagger_exact",
        "n_emp",
        "n_emp_multiplier",
        "n_emp_transitions",
        "n_emp_censored",
        "normalized_return",
        "optimal_return",
        "random_return",
        "behavior_return",
        "behavior_normalized_return",
        "policy_normalized_return",
        "success",
    }
)


class LeakageError(ValueError):
    pass


def _looks_like_oracle_field(name: str) -> bool:
    lowered = name.lower()
    if lowered in _FORBIDDEN_EXACT_NAMES:
        return True
    return any(lowered.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)


def assert_no_oracle_leakage(
    feature_names: Iterable[str],
    *,
    context: str,
    allowlist: tuple[str, ...] | None = None,
) -> None:
    """Reject a feature set that contains an oracle-derived or label field.

    If ``allowlist`` is given (e.g. ``STRUCTURAL_FEATURE_ALLOWLIST``), every
    name must be a member of it -- a closed-list check, stricter than the
    pattern-based default. Otherwise every name is checked against the
    forbidden-pattern list.
    """

    names = tuple(feature_names)
    if allowlist is not None:
        disallowed = [name for name in names if name not in allowlist]
        if disallowed:
            raise LeakageError(
                f"{context}: feature(s) outside the closed structural allowlist: {disallowed}"
            )
        return
    forbidden = [name for name in names if _looks_like_oracle_field(name)]
    if forbidden:
        raise LeakageError(
            f"{context}: forbidden oracle-derived or label feature name(s): {forbidden}"
        )


__all__ = [
    "STRUCTURAL_FEATURE_ALLOWLIST",
    "LeakageError",
    "assert_no_oracle_leakage",
]
