"""Target-aligned cross-fitted OEH requirement estimator.

Version 1 incorrectly required the current pilot to certify every relevant
action gap before it would estimate a *future* data requirement.  This module
separates those questions.  A model fold fixes the limited-lookahead policy and
critical top/runner pairs.  An evidence fold estimates their positive gaps,
variances, and behavior frequencies, then extrapolates the sample size needed
for simultaneous identification.  Current-budget sufficiency is reported
separately.

The aggregation is return-aware: comparisons may be omitted only while their
total discounted impact fits inside the epsilon left after planning regret.
Unsupported low-impact states therefore do not invalidate the whole depth.
This remains a local empirical-Bernstein approximation to the alternative-MDP
KL target, not an exact instance-optimal theorem.
"""
from __future__ import annotations

import math

import numpy as np

from oeh_crossfit import _evidence_statistics, _model, _trajectory_folds
from oeh_eh_aligned import (
    _check_distribution, _discounted_occupancy, _policy_value,
    _restricted_optimum, _restricted_policy,
)
from oeh_final_profile import build_final_proxy_profile


def _critical_subset(items, impact_budget):
    """Drop hardest comparisons only while their total impact fits epsilon."""
    ordered = sorted(items, key=lambda item: (math.inf if item["requirement"] is None
                                               else item["requirement"]), reverse=True)
    ignored, retained = [], []
    used = 0.0
    for item in ordered:
        impact = float(item["impact"])
        if used + impact <= impact_budget + 1e-15:
            ignored.append(item)
            used += impact
        else:
            retained.append(item)
    return retained, ignored, used


def _future_requirement(gap, variances, frequencies, log_factor):
    """Two-sample empirical-Bernstein planning requirement.

    The variance term is the variance of a difference of independent action
    means under a future stationary behavior stream.  The linear term preserves
    bounded-outcome protection without replacing the estimated gap by its
    current-pilot lower confidence bound.
    """
    if gap <= 0 or any(value <= 0 for value in frequencies):
        return None
    variance_scale = sum(v / d for v, d in zip(variances, frequencies))
    inverse_coverage = sum(1.0 / d for d in frequencies)
    return float(max(
        1.0,
        2.0 * log_factor * variance_scale / (gap * gap),
        7.0 * log_factor * inverse_coverage / (3.0 * gap),
    ))


def _direction(model_arrays, evidence_arrays, initial_state_distribution, *,
               num_states, num_actions, gamma, k_max, reward_bound, epsilon,
               delta):
    p, r, behavior, model_counts, terminals = _model(
        model_arrays, num_states=num_states, num_actions=num_actions,
        reward_bound=reward_bound)
    rho = _check_distribution(initial_state_distribution, num_states)
    terminal = np.zeros(num_states, dtype=bool)
    terminal[np.asarray(terminals, dtype=int)] = True
    eligible = model_counts > 0
    eligible[terminal] = False
    behavior_use = np.where(eligible, behavior, 0.0)
    totals = behavior_use.sum(1, keepdims=True)
    behavior_use = np.divide(behavior_use, totals, out=np.zeros_like(behavior_use),
                             where=totals > 0)
    behavior_use[~eligible.any(1), 0] = 1.0
    p_beta = np.einsum("sa,san->sn", behavior_use, p)
    r_beta = np.sum(behavior_use * r, axis=1)
    leaf = np.linalg.solve(np.eye(num_states) - gamma * p_beta, r_beta)
    optimum = _restricted_optimum(p, r, gamma, eligible)
    worst = _restricted_optimum(p, r, gamma, eligible, minimize=True)
    value_range = float(rho @ (optimum - worst))
    rows = np.arange(num_states)
    records = []

    for depth in range(k_max + 1):
        q = r + gamma * np.einsum("san,n->sa", p, leaf)
        policy = _restricted_policy(q, eligible)
        top = q[rows, policy]
        greedy = eligible & (q >= top[:, None] - 1e-12)
        greedy[~eligible.any(1), 0] = True
        if np.all(greedy.sum(1) == 1):
            worst_greedy = _policy_value(p, r, gamma, greedy.argmax(1))
        else:
            worst_greedy = _restricted_optimum(p, r, gamma, greedy, minimize=True)
        numerator = max(0.0, float(rho @ (optimum - worst_greedy)))
        planning_regret = 0.0 if value_range <= 1e-15 else numerator / value_range
        if planning_regret > epsilon + 1e-12:
            records.append({
                "depth": depth, "qvi_index": depth + 1,
                "status": "not_epsilon_solvable", "admissible": False,
                "identified": False, "eb_sample_requirement": None,
                "eb_evidence": None, "planning_regret": planning_regret,
                "relevant_states": 0, "retained_states": 0, "ignored_states": 0,
                "ignored_impact": 0.0, "unsupported_states": 0,
                "unresolved_states": 0, "minimum_empirical_gap": None,
                "current_budget_sufficient": False,
            })
            leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
            leaf[~eligible.any(1)] = 0.0
            continue

        occupancy = _discounted_occupancy(p, gamma, rho, policy)
        relevant = (~terminal) & (occupancy > 0)
        evidence_counts, evidence_means, evidence_variances = _evidence_statistics(
            evidence_arrays, leaf, num_states=num_states, num_actions=num_actions,
            gamma=gamma, reward_bound=reward_bound)
        evidence_total = float(evidence_counts.sum())
        # Bonferroni is over the comparisons actually considered, not all
        # state-action-depth coordinates in the public space.
        comparison_count = max(1, int(relevant.sum()))
        log_factor = float(np.log(4 * comparison_count * (k_max + 1) / delta))
        comparisons = []
        regret_scale = max((1.0 - gamma) * value_range, 1e-15)
        for state in np.flatnonzero(relevant):
            supported = np.flatnonzero(eligible[state])
            occupancy_impact = float(min(
                1.0, occupancy[state] / ((1.0 - gamma) * regret_scale)))
            if len(supported) < 2:
                comparisons.append({
                    "state": int(state), "impact": occupancy_impact,
                    "status": "unsupported", "requirement": None, "gap": None,
                })
                continue
            ranked = supported[np.argsort(q[state, supported], kind="stable")]
            chosen, runner = int(ranked[-1]), int(ranked[-2])
            counts = (float(evidence_counts[state, chosen]),
                      float(evidence_counts[state, runner]))
            if min(counts) <= 0 or evidence_total <= 0:
                comparisons.append({
                    "state": int(state), "impact": occupancy_impact,
                    "status": "unsupported", "requirement": None, "gap": None,
                })
                continue
            gap = float(evidence_means[state, chosen] - evidence_means[state, runner])
            model_gap = max(0.0, float(q[state, chosen] - q[state, runner]))
            # Performance difference: relative loss contribution is
            # d_pi(s) * advantage / ((1-gamma) * empirical value range).
            impact = float(min(1.0, occupancy[state] * model_gap / regret_scale))
            if gap <= 0:
                comparisons.append({
                    "state": int(state), "impact": max(impact, occupancy_impact * 1e-12),
                    "status": "unresolved", "requirement": None, "gap": gap,
                })
                continue
            frequencies = tuple(count / evidence_total for count in counts)
            variances = (float(evidence_variances[state, chosen]),
                         float(evidence_variances[state, runner]))
            requirement = _future_requirement(gap, variances, frequencies, log_factor)
            comparisons.append({
                "state": int(state), "impact": impact, "status": "estimated",
                "requirement": requirement, "gap": gap,
            })

        remaining_epsilon = max(0.0, epsilon - planning_regret)
        retained, ignored, ignored_impact = _critical_subset(
            comparisons, remaining_epsilon)
        unsupported = sum(item["status"] == "unsupported" for item in retained)
        unresolved = sum(item["status"] == "unresolved" for item in retained)
        finite = [item["requirement"] for item in retained
                  if item["requirement"] is not None]
        if not retained:
            # No action distinction is needed within the remaining normalized
            # return-loss budget.  This is a zero-information target, not an
            # identified comparison with an artificial one-sample floor.
            status = "epsilon_trivial"
        elif unsupported:
            status = "unsupported"
        elif unresolved:
            status = "unresolved"
        else:
            status = "identified"
        if status == "epsilon_trivial":
            requirement = 0.0
        elif status == "identified":
            requirement = max(finite)
        else:
            requirement = None
        evidence = (0.0 if requirement == 0.0 else
                    math.log(requirement, num_actions)
                    if requirement is not None and num_actions > 1 else None)
        records.append({
            "depth": depth, "qvi_index": depth + 1, "status": status,
            "admissible": True, "identified": status == "identified",
            "eb_sample_requirement": requirement, "eb_evidence": evidence,
            "planning_regret": planning_regret,
            "relevant_states": int(relevant.sum()),
            "retained_states": len(retained), "ignored_states": len(ignored),
            "ignored_impact": ignored_impact,
            "unsupported_states": unsupported, "unresolved_states": unresolved,
            "minimum_empirical_gap": min(
                (item["gap"] for item in retained if item["gap"] is not None),
                default=None),
            "current_budget_sufficient": (
                requirement is not None and evidence_total >= requirement),
        })
        leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
        leaf[~eligible.any(1)] = 0.0
    return records


def estimate_crossfit_oeh_v2(arrays, initial_state_distribution, *, num_states,
                             num_actions, gamma=.95, k_max=100, reward_bound=1.0,
                             epsilon=.05, delta=.05, seed=20260920,
                             compute_budget=None):
    """Estimate a target-aligned future data requirement profile."""
    if num_actions < 2:
        raise ValueError("OEH action identification requires at least two actions")
    trajectory_count = len(np.unique(np.asarray(arrays["trajectory_id"], dtype=int)))
    if trajectory_count < 2:
        records = [{
            "depth": depth, "qvi_index": depth + 1, "status": "unsupported",
            "crossfit_status": "unsupported",
            "admissible": True, "identified": False,
            "eb_sample_requirement": None, "eb_evidence": None,
            # Keep the same public record schema as the two-fold path.  The
            # values are unknown because cross-fitting is impossible, rather
            # than numerical zeroes that could be mistaken for evidence.
            "planning_regret_max": None, "relevant_states_max": None,
            "retained_states_max": None, "ignored_states_max": None,
            "ignored_impact_max": None, "unsupported_states_max": None,
            "unresolved_states_max": None, "minimum_empirical_gap": None,
            "current_budget_sufficient_both": False,
        } for depth in range(k_max + 1)]
        final = build_final_proxy_profile(records, compute_budget=compute_budget)
        final.update({
            "formula_version": "oeh_target_aligned_crossfit_v2",
            "cross_fitted": False,
            "crossfit_unavailable_reason": "fewer_than_two_trajectories",
            "fold_trajectory_counts": [trajectory_count, 0], "records": records,
        })
        return final
    first, second = _trajectory_folds(arrays, seed=seed)
    directions = [
        _direction(first, second, initial_state_distribution,
                   num_states=num_states, num_actions=num_actions, gamma=gamma,
                   k_max=k_max, reward_bound=reward_bound, epsilon=epsilon,
                   delta=delta),
        _direction(second, first, initial_state_distribution,
                   num_states=num_states, num_actions=num_actions, gamma=gamma,
                   k_max=k_max, reward_bound=reward_bound, epsilon=epsilon,
                   delta=delta),
    ]
    precedence = {"unsupported": 4, "unresolved": 3,
                  "not_epsilon_solvable": 2, "identified": 1,
                  "epsilon_trivial": 0}
    combined = []
    for left, right in zip(*directions):
        status = max((left["status"], right["status"]), key=precedence.get)
        requirement = (max(left["eb_sample_requirement"], right["eb_sample_requirement"])
                       if status in {"identified", "epsilon_trivial"} else None)
        combined.append({
            "depth": left["depth"], "qvi_index": left["qvi_index"],
            "crossfit_status": status,
            "admissible": left["admissible"] and right["admissible"],
            "identified": status in {"identified", "epsilon_trivial"},
            "eb_sample_requirement": requirement,
            "eb_evidence": (0.0 if requirement == 0.0 else
                            math.log(requirement, num_actions)
                            if requirement is not None else None),
            "planning_regret_max": max(
                value for value in (left["planning_regret"], right["planning_regret"])
                if value is not None),
            "relevant_states_max": max(left["relevant_states"], right["relevant_states"]),
            "retained_states_max": max(left["retained_states"], right["retained_states"]),
            "ignored_states_max": max(left["ignored_states"], right["ignored_states"]),
            "ignored_impact_max": max(left["ignored_impact"], right["ignored_impact"]),
            "unsupported_states_max": max(left["unsupported_states"], right["unsupported_states"]),
            "unresolved_states_max": max(left["unresolved_states"], right["unresolved_states"]),
            "minimum_empirical_gap": min(
                (value for value in (left["minimum_empirical_gap"], right["minimum_empirical_gap"])
                 if value is not None), default=None),
            "current_budget_sufficient_both": (
                status in {"identified", "epsilon_trivial"}
                and left["current_budget_sufficient"]
                and right["current_budget_sufficient"]),
        })
    final = build_final_proxy_profile(combined, compute_budget=compute_budget)
    final.update({
        "formula_version": "oeh_target_aligned_crossfit_v2",
        "estimator": "target_aligned_crossfit_empirical_bernstein_requirement",
        "exact_kl_target_implemented": False, "cross_fitted": True,
        "fold_trajectory_counts": [
            len(np.unique(first["trajectory_id"])),
            len(np.unique(second["trajectory_id"])),
        ],
        "records": combined,
    })
    return final


__all__ = ["estimate_crossfit_oeh_v2"]
