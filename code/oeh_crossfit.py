"""Trajectory-cross-fitted empirical-Bernstein OEH profile.

One trajectory fold fixes the empirical QVI sequence and action comparisons;
the other estimates the corresponding Bellman-target gaps, variances and
coverage.  Swapping folds prevents the same observations from both proposing
and validating a comparison.  This is a conservative development estimator,
not the exact alternative-MDP KL optimizer or an instance-optimal theorem.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import beta as beta_distribution

from oeh_certificate_adapter import trajectory_statistics
from oeh_eh_aligned import (
    _check_distribution,
    _discounted_occupancy,
    _policy_value,
    _restricted_optimum,
    _restricted_policy,
)
from oeh_final_profile import build_final_proxy_profile
from oeh_selective import nominal_model


def _trajectory_folds(arrays, *, seed):
    ids = np.asarray(arrays["trajectory_id"], dtype=np.int64)
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("Cross-fitting requires at least two trajectories")
    keyed = []
    for trajectory_id in unique:
        digest = hashlib.sha256(f"{seed}:{int(trajectory_id)}".encode()).digest()
        keyed.append((digest, int(trajectory_id)))
    ordered = np.array([item[1] for item in sorted(keyed)], dtype=np.int64)
    first_ids, second_ids = ordered[::2], ordered[1::2]
    if not len(second_ids):
        raise ValueError("Cross-fitting produced an empty trajectory fold")

    def select(chosen):
        mask = np.isin(ids, chosen)
        return {name: np.asarray(value)[mask] for name, value in arrays.items()}

    return select(first_ids), select(second_ids)


def _model(arrays, *, num_states, num_actions, reward_bound):
    counts3, reward_sums, _, terminals = trajectory_statistics(
        arrays, num_states=num_states, num_actions=num_actions,
        reward_bound=reward_bound)
    return (*nominal_model(
        counts3, reward_sums, terminals, reward_bound=reward_bound), terminals)


def _evidence_statistics(arrays, leaf, *, num_states, num_actions, gamma,
                         reward_bound):
    state = np.asarray(arrays["state"], dtype=int)
    action = np.asarray(arrays["action"], dtype=int)
    reward = np.asarray(arrays["reward"], dtype=float)/reward_bound
    next_state = np.asarray(arrays["next_state"], dtype=int)
    target = (1-gamma)*(reward+gamma*leaf[next_state])
    counts = np.zeros((num_states, num_actions), dtype=float)
    sums = np.zeros_like(counts)
    squares = np.zeros_like(counts)
    np.add.at(counts, (state, action), 1)
    np.add.at(sums, (state, action), target)
    np.add.at(squares, (state, action), target*target)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = squares-np.divide(
        sums*sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.maximum(0, np.divide(
        centered, counts-1, out=np.zeros_like(sums), where=counts > 1))
    return counts, means, variances


def _bernstein_radius(variance, count, log_factor):
    if count <= 1:
        return math.inf
    return (math.sqrt(2*variance*log_factor/count)
            + 7*log_factor/(3*(count-1)))


def _coverage_lower(count, total, alpha):
    if count <= 0 or total <= 0:
        return 0.0
    return float(beta_distribution.ppf(alpha, count, total-count+1))


def _direction(model_arrays, evidence_arrays, initial_state_distribution, *,
               num_states, num_actions, gamma, k_max, reward_bound, epsilon,
               delta, reachability_tolerance):
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
    behavior_use = np.divide(
        behavior_use, totals, out=np.zeros_like(behavior_use), where=totals > 0)
    behavior_use[~eligible.any(1), 0] = 1.0
    p_beta = np.einsum("sa,san->sn", behavior_use, p)
    r_beta = np.sum(behavior_use*r, axis=1)
    leaf = np.linalg.solve(np.eye(num_states)-gamma*p_beta, r_beta)
    optimum = _restricted_optimum(p, r, gamma, eligible)
    worst = _restricted_optimum(p, r, gamma, eligible, minimize=True)
    value_range = float(rho@(optimum-worst))
    rows = np.arange(num_states)
    log_factor = float(np.log(8*num_states*num_actions*(k_max+1)/delta))
    alpha = delta/(8*num_states*num_actions*(k_max+1))
    records = []

    for depth in range(k_max+1):
        q = r+gamma*np.einsum("san,n->sa", p, leaf)
        policy = _restricted_policy(q, eligible)
        top = q[rows, policy]
        greedy = eligible & (q >= top[:, None]-1e-12)
        greedy[~eligible.any(1), 0] = True
        if np.all(greedy.sum(1) == 1):
            worst_greedy = _policy_value(p, r, gamma, greedy.argmax(1))
        else:
            worst_greedy = _restricted_optimum(p, r, gamma, greedy, minimize=True)
        numerator = max(0.0, float(rho@(optimum-worst_greedy)))
        planning_regret = 0.0 if value_range <= 1e-15 else numerator/value_range
        admissible = planning_regret <= epsilon+1e-12
        occupancy = _discounted_occupancy(p, gamma, rho, policy)
        relevant = (~terminal) & (occupancy > reachability_tolerance)
        evidence_counts, evidence_means, evidence_variances = _evidence_statistics(
            evidence_arrays, leaf, num_states=num_states, num_actions=num_actions,
            gamma=gamma, reward_bound=reward_bound)
        evidence_total = float(evidence_counts.sum())
        unsupported = unresolved = 0
        requirements, lower_gaps = [], []
        for state in np.flatnonzero(relevant):
            if np.any(model_counts[state] <= 0) or np.any(evidence_counts[state] <= 0):
                unsupported += 1
                continue
            maximizers = q[state] >= top[state]-1e-12
            if np.count_nonzero(maximizers) != 1:
                unresolved += 1
                continue
            runner = int(np.where(~maximizers, q[state], -np.inf).argmax())
            chosen = int(policy[state])
            gap_hat = float(evidence_means[state, chosen]-evidence_means[state, runner])
            radii = [
                _bernstein_radius(evidence_variances[state, action],
                                  evidence_counts[state, action], log_factor)
                for action in (chosen, runner)
            ]
            lower_gap = max(0.0, gap_hat-sum(radii))
            if lower_gap <= 0.0:
                unresolved += 1
                continue
            local = []
            for action in (chosen, runner):
                d_lower = _coverage_lower(
                    evidence_counts[state, action], evidence_total, alpha)
                if d_lower <= 0.0:
                    unsupported += 1
                    local = []
                    break
                variance = evidence_variances[state, action]
                local.append(max(
                    1.0,
                    128*variance*log_factor/(d_lower*lower_gap**2),
                    (1+56*log_factor/(3*lower_gap))/d_lower,
                ))
            if local:
                lower_gaps.append(lower_gap)
                requirements.append(max(local))
        if unsupported:
            status = "unsupported"
        elif unresolved:
            status = "unresolved"
        elif not admissible:
            status = "not_epsilon_solvable"
        else:
            status = "identified"
        requirement = max(requirements, default=1.0) if status == "identified" else None
        evidence = (math.log(requirement, num_actions)
                    if requirement is not None and num_actions > 1 else None)
        records.append({
            "depth": depth, "qvi_index": depth+1,
            "planning_regret": planning_regret,
            "admissible": admissible, "identified": status == "identified",
            "status": status, "relevant_states": int(relevant.sum()),
            "unsupported_states": unsupported, "unresolved_states": unresolved,
            "minimum_lower_gap": min(lower_gaps, default=None),
            "eb_sample_requirement": requirement, "eb_evidence": evidence,
        })
        leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
        leaf[~eligible.any(1)] = 0.0
    return records


def estimate_crossfit_oeh(arrays, initial_state_distribution, *, num_states,
                          num_actions, gamma=.95, k_max=100, reward_bound=1.0,
                          epsilon=.05, delta=.05, seed=20260919,
                          reachability_tolerance=1e-12, compute_budget=None):
    """Return a two-direction trajectory-cross-fitted EB-OEH profile."""
    if num_actions < 2:
        raise ValueError("OEH action identification requires at least two actions")
    trajectory_count = len(np.unique(np.asarray(arrays["trajectory_id"], dtype=int)))
    if trajectory_count < 2:
        records = [{
            "depth": depth, "qvi_index": depth+1,
            "admissible": True, "identified": False,
            "eb_sample_requirement": None, "eb_evidence": None,
            "crossfit_status": "unsupported",
            "planning_regret_max": None, "relevant_states_max": None,
            "unsupported_states_max": None, "unresolved_states_max": None,
            "minimum_lower_gap": None,
        } for depth in range(k_max+1)]
        final = build_final_proxy_profile(records, compute_budget=compute_budget)
        final.update({
            "estimator": "trajectory_crossfit_eb_oeh_development_proxy",
            "cross_fitted": False,
            "crossfit_unavailable_reason": "fewer_than_two_trajectories",
            "fold_trajectory_counts": [trajectory_count, 0],
            "records": records,
        })
        return final
    first, second = _trajectory_folds(arrays, seed=seed)
    directions = [
        _direction(first, second, initial_state_distribution,
                   num_states=num_states, num_actions=num_actions, gamma=gamma,
                   k_max=k_max, reward_bound=reward_bound, epsilon=epsilon,
                   delta=delta, reachability_tolerance=reachability_tolerance),
        _direction(second, first, initial_state_distribution,
                   num_states=num_states, num_actions=num_actions, gamma=gamma,
                   k_max=k_max, reward_bound=reward_bound, epsilon=epsilon,
                   delta=delta, reachability_tolerance=reachability_tolerance),
    ]
    combined = []
    precedence = {
        "unsupported": 3, "unresolved": 2,
        "not_epsilon_solvable": 1, "identified": 0,
    }
    for left, right in zip(*directions):
        status = max((left["status"], right["status"]), key=precedence.get)
        requirement = (max(left["eb_sample_requirement"], right["eb_sample_requirement"])
                       if status == "identified" else None)
        evidence = (math.log(requirement, num_actions)
                    if requirement is not None else None)
        combined.append({
            "depth": left["depth"], "qvi_index": left["qvi_index"],
            "admissible": left["admissible"] and right["admissible"],
            "identified": status == "identified",
            "eb_sample_requirement": requirement, "eb_evidence": evidence,
            "crossfit_status": status,
            "planning_regret_max": max(left["planning_regret"], right["planning_regret"]),
            "relevant_states_max": max(left["relevant_states"], right["relevant_states"]),
            "unsupported_states_max": max(left["unsupported_states"], right["unsupported_states"]),
            "unresolved_states_max": max(left["unresolved_states"], right["unresolved_states"]),
            "minimum_lower_gap": min(
                (value for value in (left["minimum_lower_gap"], right["minimum_lower_gap"])
                 if value is not None), default=None),
        })
    final = build_final_proxy_profile(combined, compute_budget=compute_budget)
    final.update({
        "estimator": "trajectory_crossfit_eb_oeh_development_proxy",
        "cross_fitted": True,
        "fold_trajectory_counts": [
            len(np.unique(first["trajectory_id"])),
            len(np.unique(second["trajectory_id"])),
        ],
        "records": combined,
    })
    return final


__all__ = ["estimate_crossfit_oeh"]
