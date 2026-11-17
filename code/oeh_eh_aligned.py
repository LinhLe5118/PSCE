"""EH-aligned, coverage-aware diagnostics for stochastic tabular pilots.

This module is exploratory research code.  It preserves the two terms in the
stochastic effective horizon of Laidlaw et al. (ICLR 2024): planning depth and
the logarithm of the samples needed to resolve a greedy-action gap.  Unlike the
frozen OEH-v1 estimator, it never solves a thresholded loss with ``mu >= 1``;
therefore it has no artificial all-one multiplier floor.

The empirical-Bernstein calculation is a plug-in diagnostic, not a confidence
certificate.  Its role is to test a proposed formula on development data before
any frozen evaluation protocol is changed.
"""
from __future__ import annotations

import numpy as np

from oeh_certificate_adapter import trajectory_statistics
from oeh_final_profile import build_final_proxy_profile
from oeh_selective import nominal_model


def _check_distribution(rho, states):
    rho = np.asarray(rho, dtype=float)
    if (rho.shape != (states,) or not np.all(np.isfinite(rho))
            or np.any(rho < 0) or not np.isclose(rho.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Invalid initial-state distribution")
    return rho


def _policy_value(p, r, gamma, policy):
    states = len(policy)
    rows = np.arange(states)
    return np.linalg.solve(np.eye(states) - gamma*p[rows, policy], r[rows, policy])


def _restricted_optimum(p, r, gamma, eligible, *, minimize=False, tolerance=1e-12):
    """Optimal value with a state-wise action mask; action zero fills empty rows."""
    states = p.shape[0]
    mask = np.asarray(eligible, bool).copy()
    mask[~mask.any(1), 0] = True
    initial = np.where(mask, r, np.inf if minimize else -np.inf)
    policy = initial.argmin(1) if minimize else initial.argmax(1)
    for _ in range(1000):
        value = _policy_value(p, r, gamma, policy)
        q = r + gamma*np.einsum("san,n->sa", p, value)
        score = np.where(mask, q, np.inf if minimize else -np.inf)
        new_policy = score.argmin(1) if minimize else score.argmax(1)
        target = score.min(1) if minimize else score.max(1)
        if (np.array_equal(new_policy, policy)
                or np.max(np.abs(target-value)) <= tolerance*(1-gamma)):
            return value
        policy = new_policy
    raise RuntimeError("Restricted policy iteration did not converge")


def _restricted_policy(q, eligible):
    score = np.where(eligible, q, -np.inf)
    empty = ~eligible.any(1)
    score[empty, 0] = 0
    return score.argmax(1)


def _discounted_occupancy(p, gamma, rho, policy):
    rows = np.arange(len(policy))
    # Row-vector solution d(I-gamma P_pi)=(1-gamma)rho.
    return np.linalg.solve(
        np.eye(len(policy))-gamma*p[rows, policy].T, (1-gamma)*rho)


def _validate_model(p, r, beta, counts, terminals, gamma, reward_bound):
    p, r, beta, counts = map(lambda x: np.asarray(x), (p, r, beta, counts))
    if (p.ndim != 3 or p.shape[0] != p.shape[2] or r.shape != p.shape[:2]
            or beta.shape != r.shape or counts.shape != r.shape):
        raise ValueError("Invalid model shapes")
    if (not 0 <= gamma < 1 or not np.isfinite(reward_bound) or reward_bound <= 0
            or np.any(p < 0) or not np.allclose(p.sum(2), 1, atol=1e-12, rtol=0)
            or not np.all(np.isfinite(r)) or np.any(r < 0) or np.any(r > 1)
            or np.any(counts < 0)):
        raise ValueError("Invalid model values")
    terminal_mask = np.zeros(p.shape[0], dtype=bool)
    terminal_mask[np.asarray(terminals, dtype=int)] = True
    return p.astype(float), r.astype(float), beta.astype(float), counts.astype(float), terminal_mask


def _profile(p, r, beta, counts, rho, terminals, variance_at_leaf, *, gamma=.95,
             k_max=100, epsilon=.05, delta=.05, occupancy_threshold=None,
             reward_bound=1., exact_model=False):
    """Compute QVI profiles. QVI index 1 has the behavior-policy leaf.

    ``variance_at_leaf(leaf)`` returns Var[(1-gamma)(R/Rmax+gamma*leaf(S'))]
    for every state-action.  ``r`` and ``leaf`` are already reward-normalized.
    """
    p, r, beta, counts, terminal = _validate_model(
        p, r, beta, counts, terminals, gamma, reward_bound)
    states, actions = r.shape
    rho = _check_distribution(rho, states)
    if not isinstance(k_max, int) or k_max < 0 or not 0 < delta < 1 or epsilon < 0:
        raise ValueError("Invalid profile parameters")
    if actions < 2:
        raise ValueError("Effective-horizon gaps require at least two actions")
    threshold = 1/(10*states) if occupancy_threshold is None else float(occupancy_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid occupancy threshold")
    eligible = counts > 0
    eligible[terminal] = False
    beta_use = np.where(eligible, beta, 0.)
    totals = beta_use.sum(1, keepdims=True)
    beta_use = np.divide(beta_use, totals, out=np.zeros_like(beta_use), where=totals > 0)
    beta_use[~eligible.any(1), 0] = 1
    p_beta = np.einsum("sa,san->sn", beta_use, p)
    r_beta = np.sum(beta_use*r, axis=1)
    leaf = np.linalg.solve(np.eye(states)-gamma*p_beta, r_beta)
    optimum = _restricted_optimum(p, r, gamma, eligible)
    worst = _restricted_optimum(p, r, gamma, eligible, minimize=True)
    empirical_range = float(rho@(optimum-worst))
    total_samples = float(counts.sum())
    frequencies = counts/max(total_samples, 1.)
    log_factor = float(np.log(4*states*actions*(k_max+1)/delta))
    rows = np.arange(states)
    policies, records = [], []
    q = None
    for depth in range(k_max+1):
        # Original-paper indexing: Q^1 = Q^beta, then Q^{k+1}=TQ^k.
        q = r + gamma*np.einsum("san,n->sa", p, leaf)
        policy = _restricted_policy(q, eligible)
        policies.append(policy)
        top = q[rows, policy]
        greedy = eligible & (q >= top[:, None]-1e-12)
        greedy[~eligible.any(1)] = False
        greedy[~eligible.any(1), 0] = True
        if np.all(greedy.sum(1) == 1):
            worst_greedy = _policy_value(p, r, gamma, greedy.argmax(1))
        else:
            worst_greedy = _restricted_optimum(p, r, gamma, greedy, minimize=True)
        # The ICLR 2024 empirical analysis calls a depth approximately
        # k-QVI-solvable at 95% of the optimal return measured from the minimum
        # return.  This ratio is also invariant to positive reward rescaling.
        numerator = max(0., float(rho@(optimum-worst_greedy)))
        planning_regret = 0. if empirical_range <= 1e-15 else numerator/empirical_range
        occupancy = _discounted_occupancy(p, gamma, rho, policy)
        critical = (~terminal) & (occupancy >= threshold) & (eligible.sum(1) >= 2)
        outcome_variance = np.asarray(variance_at_leaf(leaf), dtype=float)
        if outcome_variance.shape != r.shape or np.any(~np.isfinite(outcome_variance)) or np.any(outcome_variance < 0):
            raise ValueError("Invalid Bellman-target variance")
        gap_values, point_requirements, eb_requirements = [], [], []
        pairs, unresolved_ties = 0, 0
        for state in np.flatnonzero(critical):
            maximizers = eligible[state] & (q[state] >= top[state]-1e-12)
            if maximizers.sum() > 1 and not exact_model:
                # An empirical co-maximizer may be an exact true tie, but the
                # same data do not establish that fact. Treat its zero margin
                # as non-identification rather than silently tie-breaking.
                unresolved_ties += 1
                continue
            nonmax = eligible[state] & (q[state] < top[state]-1e-12)
            if not nonmax.any():
                # In a known model this is a genuine all-action tie and needs no
                # discrimination.  In a finite pilot it is non-identification:
                # empirical equality is not evidence of true equality.
                unresolved_ties += int(not exact_model)
                continue
            runner = int(np.where(nonmax, q[state], -np.inf).argmax())
            gap = float(np.clip((1-gamma)*(top[state]-q[state, runner]), 0, 1))
            if gap <= 0:
                continue
            pairs += 1
            gap_values.append(gap)
            point_requirements.append(1/gap**2)
            action_requirements = []
            for action in (int(policy[state]), runner):
                d = frequencies[state, action]
                # Split the half-gap safety margin equally between square-root
                # and linear empirical-Bernstein terms for each of two actions.
                n_variance = 128*outcome_variance[state, action]*log_factor/(d*gap**2)
                n_linear = (1 + 56*log_factor/(3*gap))/d
                action_requirements.append(max(1., n_variance, n_linear))
            eb_requirements.append(max(action_requirements))
        identified = unresolved_ties == 0
        point_n = max(point_requirements, default=1.) if identified else None
        eb_n = max(eb_requirements, default=1.) if identified else None
        qvi_index = depth+1
        point_evidence = None if point_n is None else float(np.log(point_n)/np.log(actions))
        eb_evidence = None if eb_n is None else float(np.log(eb_n)/np.log(actions))
        admissible = planning_regret <= epsilon+1e-12
        records.append(dict(
            depth=depth, qvi_index=qvi_index, planning_regret=planning_regret,
            admissible=bool(admissible), identified=identified,
            evidence_status='observed' if identified else 'unresolved_empirical_tie',
            critical_states=int(critical.sum()), gap_pairs=pairs,
            unresolved_ties=unresolved_ties,
            minimum_gap=min(gap_values, default=None), point_sample_requirement=point_n,
            eb_sample_requirement=eb_n, point_evidence=point_evidence,
            eb_evidence=eb_evidence,
            point_raw=None if point_evidence is None else qvi_index+point_evidence,
            eb_raw=None if eb_evidence is None else qvi_index+eb_evidence,
            point_score=qvi_index+point_evidence if admissible and identified else None,
            eb_score=qvi_index+eb_evidence if admissible and identified else None))
        leaf = np.max(np.where(eligible, q, -np.inf), axis=1)
        leaf[~eligible.any(1)] = 0

    def selected(field):
        candidates = [(record[field], i) for i, record in enumerate(records)
                      if record[field] is not None]
        return min(candidates)[1] if candidates else None

    return dict(records=records, policies=np.asarray(policies, dtype=int),
                point_selected_depth=selected("point_score"),
                eb_selected_depth=selected("eb_score"),
                eligible=eligible, behavior=beta_use, log_factor=log_factor,
                occupancy_threshold=threshold, empirical_return_range=empirical_range)


def estimate_eh_aligned(arrays, initial_state_distribution, *, num_states,
                        num_actions, gamma=.95, k_max=100, reward_bound=1.,
                        epsilon=.05, delta=.05, occupancy_threshold=None):
    """Data-only EH-aligned plug-in profile from a trajectory prefix."""
    counts3, reward_sums, _, terminals = trajectory_statistics(
        arrays, num_states=num_states, num_actions=num_actions,
        reward_bound=reward_bound)
    p, r, beta, counts = nominal_model(
        counts3, reward_sums, terminals, reward_bound=reward_bound)
    state = np.asarray(arrays["state"], dtype=int)
    action = np.asarray(arrays["action"], dtype=int)
    reward = np.asarray(arrays["reward"], dtype=float)/reward_bound
    next_state = np.asarray(arrays["next_state"], dtype=int)

    def empirical_variance(leaf):
        target = (1-gamma)*(reward + gamma*leaf[next_state])
        sums = np.zeros((num_states, num_actions))
        squares = np.zeros_like(sums)
        np.add.at(sums, (state, action), target)
        np.add.at(squares, (state, action), target*target)
        # Unbiased sample variance is the quantity in the empirical-Bernstein
        # radius.  At n=1 it is undefined; zero plus the nonzero linear term is
        # the conservative finite convention used by the plug-in diagnostic.
        centered = squares-np.divide(sums*sums, counts, out=np.zeros_like(sums),
                                     where=counts > 0)
        return np.maximum(0, np.divide(centered, counts-1,
                                       out=np.zeros_like(sums), where=counts > 1))

    return _profile(p, r, beta, counts, initial_state_distribution, terminals,
                    empirical_variance, gamma=gamma, k_max=k_max,
                    epsilon=epsilon, delta=delta,
                    occupancy_threshold=occupancy_threshold,
                    reward_bound=reward_bound)


def estimate_final_oeh_proxy(arrays, initial_state_distribution, *, num_states,
                             num_actions, gamma=.95, k_max=100,
                             reward_bound=1., epsilon=.05, delta=.05,
                             occupancy_threshold=None, compute_budget=None):
    """Run the EB plug-in and expose the final ``(k, N_hat, status)`` contract.

    The returned metadata states explicitly that this development proxy is not
    cross-fitted and does not implement the general alternative-MDP KL target.
    The legacy estimator remains available so historical artifacts reproduce.
    """
    legacy = estimate_eh_aligned(
        arrays, initial_state_distribution, num_states=num_states,
        num_actions=num_actions, gamma=gamma, k_max=k_max,
        reward_bound=reward_bound, epsilon=epsilon, delta=delta,
        occupancy_threshold=occupancy_threshold)
    final = build_final_proxy_profile(
        legacy["records"], compute_budget=compute_budget)
    return {**final, "legacy_plugin": legacy}


def oracle_eh_aligned(mdp, counts, behavior, *, k_max=100, epsilon=.05,
                      delta=.05, reward_bound=1., occupancy_threshold=None):
    """Oracle counterpart for development scoring; never call from estimator."""
    p = np.asarray(mdp.P, dtype=float)
    reward3 = np.asarray(mdp.R, dtype=float)/reward_bound
    r = np.sum(p*reward3, axis=2)
    rho = np.asarray(mdp.initial_state_distribution, dtype=float)
    gamma = float(mdp.gamma)

    def exact_variance(leaf):
        target = (1-gamma)*(reward3 + gamma*leaf[None, None, :])
        mean = np.sum(p*target, axis=2)
        return np.maximum(0, np.sum(p*target*target, axis=2)-mean*mean)

    return _profile(p, r, behavior, counts, rho, mdp.terminal_states,
                    exact_variance, gamma=gamma, k_max=k_max,
                    epsilon=epsilon, delta=delta,
                    occupancy_threshold=occupancy_threshold,
                    reward_bound=reward_bound, exact_model=True)
