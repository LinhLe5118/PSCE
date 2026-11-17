"""k-dagger and N_req comparisons (mega-task sections 11-12,
``WEEK2_STATISTICAL_PLAN.md`` sec 3).

Two comparisons here deliberately treat a censored ``N_emp`` differently, and
that difference is intentional, not an inconsistency:

* ``compare_n_required`` / ``relative_error`` / ``ratio``: a censored
  ``N_emp`` has no defined point value, so these are ``NaN`` -- there is no
  correct number to compute against (PLAN_Final.md sec 3.1: never replaced by
  N_max).
* ``early_identification_ratio`` (K1a, PLAN_Final.md sec 8.1): the criterion
  itself specifies using ``N_max`` for a censored observation, explicitly as a
  *conservative upper bound* -- since the true (unknown) ``N_emp`` is at least
  ``N_max``, ``N_k / N_max`` is provably an upper bound on the true (smaller,
  unknown) ratio. Passing K1a's threshold against this pessimistic value is
  still informative; that is why the criterion defines it this way instead of
  leaving it undefined like the general comparison above.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def compare_k_dagger(
    oeh_rows: Sequence[Mapping[str, Any]], exact_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``oeh_rows``: selected-only k_dagger_estimates rows. ``exact_rows``:
    one row per (mdp_id, policy_family) with ``k_dagger_exact``.

    Only rows with ``oeh_status in {observed, censored}`` are compared -- an
    abstained/invalid row has no k_dagger_oeh value to compare, and is
    reported through the abstention rate instead (never coerced into this
    comparison).
    """

    exact_by_key = {(row["mdp_id"], row["policy_family"]): row for row in exact_rows}
    results: list[dict[str, Any]] = []
    for row in oeh_rows:
        if row["oeh_status"] not in {"observed", "censored"}:
            continue
        key = (row["mdp_id"], row["policy_family"])
        k_oeh = row["k_dagger_oeh"]
        exact_row = exact_by_key.get(key)
        if exact_row is None:
            # The oracle can be legitimately undefined for a selected OEH pair
            # (scripts/validation/run_oeh_validation.py's own "fewer than two support-valid
            # oracle datasets" case, tracked there under oracle_failures) --
            # this is an expected, common outcome on real data (23/39 pairs
            # in the 2026-08-24 development-split run), not a data-integrity
            # violation, so it is recorded as its own comparison_status
            # rather than raised. See PROTOCOL_DECISIONS_2026-08-16.md D9.
            results.append(
                {
                    "mdp_id": row["mdp_id"],
                    "policy_family": row["policy_family"],
                    "oeh_status": row["oeh_status"],
                    "k_dagger_oeh": k_oeh,
                    "k_dagger_exact": None,
                    "k_error": None,
                    "absolute_k_error": None,
                    "exact_match": False,
                    "within_1": False,
                    "within_2": False,
                    "comparison_status": "oracle_undefined",
                }
            )
            continue
        k_exact = exact_row["k_dagger_exact"]
        if k_oeh is None or k_exact is None:
            raise ValueError(f"{key}: oeh_status={row['oeh_status']!r} but k_dagger_oeh or k_dagger_exact is empty")
        k_error = int(k_oeh) - int(k_exact)
        results.append(
            {
                "mdp_id": row["mdp_id"],
                "policy_family": row["policy_family"],
                "oeh_status": row["oeh_status"],
                "k_dagger_oeh": k_oeh,
                "k_dagger_exact": k_exact,
                "k_error": k_error,
                "absolute_k_error": abs(k_error),
                "exact_match": k_error == 0,
                "within_1": abs(k_error) <= 1,
                "within_2": abs(k_error) <= 2,
                "comparison_status": "compared",
            }
        )
    return results


def compare_n_required(
    oeh_rows: Sequence[Mapping[str, Any]], n_emp_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``oeh_rows``: selected-only rows with ``n_req_hat``. ``n_emp_rows``:
    ``empirical_sample_complexity``-shaped rows, one per
    ``(mdp_id, policy_family, algorithm)``.
    """

    oeh_by_key = {(row["mdp_id"], row["policy_family"]): row for row in oeh_rows}
    results: list[dict[str, Any]] = []
    for row in n_emp_rows:
        key = (row["mdp_id"], row["policy_family"])
        oeh_row = oeh_by_key.get(key)
        n_req_hat = oeh_row["n_req_hat"] if oeh_row is not None and oeh_row["oeh_status"] in {"observed", "censored"} else None
        censored = bool(row["n_emp_censored"])
        n_emp_transitions = row["n_emp_transitions"]
        entry: dict[str, Any] = {
            "mdp_id": row["mdp_id"],
            "policy_family": row["policy_family"],
            "algorithm": row["algorithm"],
            "n_req_hat": n_req_hat,
            "n_emp_transitions": n_emp_transitions,
            "n_emp_censored": censored,
        }
        if n_req_hat is None or censored:
            entry["absolute_error"] = float("nan")
            entry["relative_error"] = float("nan")
            entry["ratio"] = float("nan")
        else:
            absolute_error = abs(float(n_req_hat) - float(n_emp_transitions))
            entry["absolute_error"] = absolute_error
            entry["relative_error"] = absolute_error / float(n_emp_transitions)
            entry["ratio"] = float(n_req_hat) / float(n_emp_transitions)
        results.append(entry)
    return results


def stabilized_pilot_budget(
    attempts: Sequence[tuple[int, int | None]], *, num_states_actions: int
) -> int | None:
    """``attempts``: ``(pilot_multiplier, k_dagger_oeh)`` pairs sorted
    ascending by multiplier, covering every attempted pilot size for one
    (mdp, policy). Returns the smallest transition budget from which
    ``k_dagger_oeh`` stays equal to its final value through the largest
    attempted pilot -- ``None`` if the final attempt itself has no value
    (never stabilized within the measured attempts).
    """

    if not attempts:
        raise ValueError("attempts must be non-empty.")
    final_k = attempts[-1][1]
    if final_k is None:
        return None
    n = len(attempts)
    for start in range(n):
        if all(attempts[j][1] == final_k for j in range(start, n)):
            return num_states_actions * attempts[start][0]
    raise AssertionError("unreachable: start=n-1 always satisfies the condition")


def early_identification_ratio(
    *,
    n_k: int | None,
    n_emp_transitions: int | None,
    n_emp_censored: bool,
    n_max_transitions: int,
) -> float:
    """K1a/K1b ratio (PLAN_Final.md sec 8.1). See module docstring for why a
    censored N_emp uses N_max here but not in ``compare_n_required``."""

    if n_k is None:
        return float("nan")
    denominator = n_max_transitions if n_emp_censored else n_emp_transitions
    if denominator is None or denominator <= 0:
        raise ValueError("A resolved N_emp denominator must be a positive integer.")
    return n_k / denominator


__all__ = [
    "compare_k_dagger",
    "compare_n_required",
    "early_identification_ratio",
    "stabilized_pilot_budget",
]
