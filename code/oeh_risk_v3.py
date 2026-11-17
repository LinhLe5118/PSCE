"""Pilot-only, target-return OEH risk forecast for stochastic tabular MDPs.

The estimator keeps the original effective-horizon decomposition

    EH(k) = k + log_A N_k,

but defines ``N_k`` through a return-weighted action-identification risk.  At
depth ``k``, every supported top/runner comparison contributes its normalized
performance-difference impact multiplied by an empirical-Bernstein error tail.
``N_k`` is the smallest future total sample size for which the sum of planning
regret and those error risks is at most ``epsilon``.

The target is explicitly support constrained.  A state with exactly one
observed action therefore needs no action identification; it is not evidence
of full-MDP optimality.  All inputs are pilot transitions.  Learner returns,
success labels, training buffers, and benchmark oracle values are neither
accepted nor read by this module.

This is a forecasting proxy, not a finite-sample confidence certificate or an
exact alternative-MDP KL optimizer.  Two trajectory halves are used only to
report policy instability.  The requirement itself uses the complete pilot so
rare supported actions are not discarded merely because they miss one half.
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


def _tail_scale(gap, variances, frequencies):
    """Return the sample scale in a two-action empirical-Bernstein tail.

    For a future stationary behavior stream of total size ``N``, action
    ``a`` receives about ``N d_a`` observations.  Inverting the square-root
    and bounded linear Bernstein terms gives the two terms below.  The gap is
    the normalized Bellman-target gap and is strictly positive.
    """
    if gap <= 0 or any(value <= 0 for value in frequencies):
        return None
    variance_scale = sum(v / d for v, d in zip(variances, frequencies))
    inverse_coverage = sum(1.0 / d for d in frequencies)
    return float(max(
        np.finfo(float).tiny,
        2.0 * variance_scale / (gap * gap),
        7.0 * inverse_coverage / (3.0 * gap),
    ))


def _decision_risk(sample_size, comparisons):
    """Expected normalized loss proxy at a future total sample size."""
    total = 0.0
    for item in comparisons:
        exponent = -float(sample_size) / float(item["tail_scale"])
        error_probability = min(1.0, 2.0 * math.exp(max(-745.0, exponent)))
        total += float(item["impact"]) * error_probability
    return total


def _invert_risk(comparisons, risk_budget, *, maximum=1e18):
    """Find the smallest nonnegative N whose monotone risk fits the budget."""
    if risk_budget < 0:
        return None
    if not comparisons or _decision_risk(0.0, comparisons) <= risk_budget:
        return 0.0
    if risk_budget <= 0:
        return None
    high = 1.0
    while high < maximum and _decision_risk(high, comparisons) > risk_budget:
        high *= 2.0
    if high > maximum:
        high = maximum
    if _decision_risk(high, comparisons) > risk_budget:
        return None
    low = 0.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if _decision_risk(middle, comparisons) <= risk_budget:
            high = middle
        else:
            low = middle
    return float(high)


def _fold_policies(arrays, *, num_states, num_actions, gamma, k_max,
                   reward_bound):
    p, r, behavior, counts, terminals = _model(
        arrays, num_states=num_states, num_actions=num_actions,
        reward_bound=reward_bound)
    terminal = np.zeros(num_states, dtype=bool)
    terminal[np.asarray(terminals, dtype=int)] = True
    eligible = counts > 0
    eligible[terminal] = False
    beta = np.where(eligible, behavior, 0.0)
    totals = beta.sum(1, keepdims=True)
    beta = np.divide(beta, totals, out=np.zeros_like(beta), where=totals > 0)
    beta[~eligible.any(1), 0] = 1.0
    leaf = np.linalg.solve(
        np.eye(num_states) - gamma * np.einsum("sa,san->sn", beta, p),
        np.sum(beta * r, axis=1),
    )
    policies = []
    for _ in range(k_max + 1):
        q = r + gamma * np.einsum("san,n->sa", p, leaf)
        policies.append(_restricted_policy(q, eligible))
        leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
        leaf[~eligible.any(1)] = 0.0
    return np.asarray(policies, dtype=int)


def estimate_oeh_risk_v3(arrays, initial_state_distribution, *, num_states,
                         num_actions, gamma=.95, k_max=100, reward_bound=1.0,
                         epsilon=.05, seed=20260920, compute_budget=None):
    """Estimate the compact support-constrained OEH-v3 profile.

    ``epsilon`` is normalized by the empirical return range inside the pilot's
    observed action support.  The result must therefore be interpreted as a
    pilot-only forecast for that support, never as a full-MDP certificate.
    """
    if num_actions < 2:
        raise ValueError("OEH action identification requires at least two actions")
    if not isinstance(k_max, int) or k_max < 0 or not 0 < epsilon < 1:
        raise ValueError("Invalid OEH profile parameters")
    required = ("state", "action", "reward", "next_state", "trajectory_id")
    if any(name not in arrays for name in required):
        raise ValueError("Missing pilot transition field")

    p, r, behavior, counts, terminals = _model(
        arrays, num_states=num_states, num_actions=num_actions,
        reward_bound=reward_bound)
    rho = _check_distribution(initial_state_distribution, num_states)
    terminal = np.zeros(num_states, dtype=bool)
    terminal[np.asarray(terminals, dtype=int)] = True
    eligible = counts > 0
    eligible[terminal] = False
    beta = np.where(eligible, behavior, 0.0)
    totals = beta.sum(1, keepdims=True)
    beta = np.divide(beta, totals, out=np.zeros_like(beta), where=totals > 0)
    beta[~eligible.any(1), 0] = 1.0
    leaf = np.linalg.solve(
        np.eye(num_states) - gamma * np.einsum("sa,san->sn", beta, p),
        np.sum(beta * r, axis=1),
    )
    optimum = _restricted_optimum(p, r, gamma, eligible)
    worst = _restricted_optimum(p, r, gamma, eligible, minimize=True)
    value_range = float(rho @ (optimum - worst))
    sample_size = float(np.asarray(arrays["state"]).size)
    frequencies = counts / max(sample_size, 1.0)
    rows = np.arange(num_states)
    records = []
    policies = []
    occupancies = []

    for depth in range(k_max + 1):
        q = r + gamma * np.einsum("san,n->sa", p, leaf)
        policy = _restricted_policy(q, eligible)
        policies.append(policy.copy())
        top = q[rows, policy]
        greedy = eligible & (q >= top[:, None] - 1e-12)
        greedy[~eligible.any(1), 0] = True
        if np.all(greedy.sum(1) == 1):
            worst_greedy = _policy_value(p, r, gamma, greedy.argmax(1))
        else:
            worst_greedy = _restricted_optimum(p, r, gamma, greedy, minimize=True)
        numerator = max(0.0, float(rho @ (optimum - worst_greedy)))
        planning_regret = 0.0 if value_range <= 1e-15 else numerator / value_range
        occupancy = _discounted_occupancy(p, gamma, rho, policy)
        occupancies.append(occupancy.copy())
        relevant = (~terminal) & (occupancy > 0)
        evidence_counts, _, evidence_variances = _evidence_statistics(
            arrays, leaf, num_states=num_states, num_actions=num_actions,
            gamma=gamma, reward_bound=reward_bound)
        comparisons = []
        single_action_states = tied_states = 0
        regret_scale = max((1.0 - gamma) * value_range, 1e-15)
        for state in np.flatnonzero(relevant):
            supported = np.flatnonzero(eligible[state])
            if len(supported) < 2:
                # The estimand is explicitly restricted to observed support.
                # With one feasible action there is no ordering to identify.
                single_action_states += 1
                continue
            ranked = supported[np.argsort(q[state, supported], kind="stable")]
            chosen, runner = int(ranked[-1]), int(ranked[-2])
            model_gap = max(0.0, float(q[state, chosen] - q[state, runner]))
            if model_gap <= 1e-15:
                tied_states += 1
                continue
            gap = (1.0 - gamma) * model_gap
            impact = float(min(
                1.0, occupancy[state] * model_gap / regret_scale))
            if impact <= 0:
                continue
            pair_counts = (float(evidence_counts[state, chosen]),
                           float(evidence_counts[state, runner]))
            pair_frequencies = (float(frequencies[state, chosen]),
                                float(frequencies[state, runner]))
            # A bounded [0,1] target can have variance 1/4.  The vanishing
            # floor avoids declaring a deterministic process from one draw.
            pair_variances = tuple(max(
                float(evidence_variances[state, action]),
                1.0 / (4.0 * (float(evidence_counts[state, action]) + 1.0)),
            ) for action in (chosen, runner))
            scale = _tail_scale(gap, pair_variances, pair_frequencies)
            if scale is None or min(pair_counts) <= 0:
                raise RuntimeError("A selected supported pair has no pilot count")
            comparisons.append({
                "state": int(state), "chosen": chosen, "runner": runner,
                "gap": gap, "impact": impact, "tail_scale": scale,
            })

        if planning_regret > epsilon + 1e-12:
            status, requirement = "not_epsilon_solvable", None
        else:
            requirement = _invert_risk(comparisons, epsilon - planning_regret)
            if requirement is None:
                status = "unresolved"
            elif requirement <= 1e-12:
                status = "epsilon_trivial"
                requirement = 0.0
            else:
                status = "identified"
        current_risk = min(1.0, planning_regret + _decision_risk(
            sample_size, comparisons))
        evidence = (math.log(requirement, num_actions)
                    if requirement is not None and requirement > 0 else 0.0
                    if requirement == 0 else None)
        records.append({
            "depth": depth, "qvi_index": depth + 1,
            "crossfit_status": status, "status": status,
            "admissible": planning_regret <= epsilon + 1e-12,
            "identified": status in {"identified", "epsilon_trivial"},
            "eb_sample_requirement": requirement, "eb_evidence": evidence,
            "planning_regret": planning_regret,
            "decision_risk_at_zero": min(1.0, _decision_risk(0.0, comparisons)),
            "risk_at_pilot": current_risk,
            "relevant_states": int(relevant.sum()),
            "comparison_states": len(comparisons),
            "single_action_states": single_action_states,
            "tied_states": tied_states,
            "minimum_gap": min((item["gap"] for item in comparisons), default=None),
            "maximum_impact": max((item["impact"] for item in comparisons), default=0.0),
            "current_budget_sufficient": (
                requirement is not None and sample_size >= requirement),
        })
        leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
        leaf[~eligible.any(1)] = 0.0

    trajectory_count = len(np.unique(np.asarray(arrays["trajectory_id"], dtype=int)))
    if trajectory_count >= 2:
        first, second = _trajectory_folds(arrays, seed=seed)
        fold_policies = [
            _fold_policies(part, num_states=num_states, num_actions=num_actions,
                           gamma=gamma, k_max=k_max, reward_bound=reward_bound)
            for part in (first, second)
        ]
        fold_counts = [
            len(np.unique(np.asarray(part["trajectory_id"], dtype=int)))
            for part in (first, second)
        ]
        for depth, record in enumerate(records):
            full = policies[depth]
            occupancy = occupancies[depth]
            relevant = (~terminal) & (occupancy > 0)
            instability = 0.5 * sum(float(np.sum(
                occupancy[relevant] * (candidate[depth, relevant] != full[relevant])))
                for candidate in fold_policies)
            record["policy_instability"] = min(1.0, instability)
    else:
        fold_counts = [trajectory_count, 0]
        for record in records:
            record["policy_instability"] = 1.0

    final = build_final_proxy_profile(records, compute_budget=compute_budget)
    final.update({
        "formula_version": "oeh_target_return_risk_v3",
        "estimator": "support_constrained_return_weighted_bernstein_risk",
        "estimand": "pilot_observed_action_support",
        "exact_kl_target_implemented": False,
        "finite_sample_certificate": False,
        "uses_outcome_labels": False,
        "crossfit_role": "trajectory_half policy_instability_only",
        "fold_trajectory_counts": fold_counts,
        "records": records,
    })
    return final


__all__ = ["estimate_oeh_risk_v3"]
