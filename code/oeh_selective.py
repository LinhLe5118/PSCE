"""Exploratory selective pessimism and coupled policy-difference bounds.

SPVI proposals adapt Appendix B of arXiv:2302.00284v4 to discounted,
receding-horizon stationary candidates. Count bonuses are proposal heuristics,
NOT confidence certificates. Certification uses the existing KL model set.
Coupled bounds are a separate discounted derivation inspired by shift-dependent
uncertainty propagation, not an implementation of SUP Theorem 4.1.
"""
from __future__ import annotations

import numpy as np

from oeh_certificate import _expectation, _policy_matrix, robust_values, certify_depths
from oeh_certificate_adapter import trajectory_statistics
from oeh_refinements import kl_confidence_box, terminal_box, empirical_depth_policies


def nominal_model(counts, reward_sums, terminals, *, reward_bound=1.):
    n = counts.sum(-1)
    s, a = n.shape
    p = counts / np.maximum(n, 1)[..., None]
    r = reward_sums / np.maximum(n, 1) / reward_bound
    for state, action in zip(*np.where(n == 0)):
        p[state, action, state] = 1
    for state in terminals:
        p[state], r[state] = 0, 0
        p[state, :, state] = 1
    total = n.sum(-1, keepdims=True)
    beta = np.divide(n, total, out=np.full(n.shape, 1/a), where=total > 0)
    return p, r, beta, n


def selective_proposals(p, r, beta, bonus, *, gamma=.95, k_max=100, selective=True):
    """H=1..k_max+1 first-step actions deployed as stationary proposals.

    gamma=1 gives the finite-horizon Appendix B recurrence for supplied inputs.
    Both PVI and SPVI use the same bonus and initialization. Action indexing in
    Appendix B value updates is interpreted as the selected action throughout.
    k=0 means H=1, unlike the empirical-behavior-leaf proposal family.
    """
    p, r, bonus = np.asarray(p, float), np.asarray(r, float), np.asarray(bonus, float)
    if p.ndim != 3 or p.shape[0] != p.shape[2] or r.shape != p.shape[:2]:
        raise ValueError('Invalid model shapes')
    s, a = r.shape
    beta = _policy_matrix(beta, s, a)
    if (not np.isfinite(gamma) or not 0 <= gamma <= 1 or not isinstance(k_max, int)
            or k_max < 0 or bonus.shape != r.shape or not np.all(np.isfinite(p))
            or np.any(p < 0) or not np.allclose(p.sum(-1), 1, rtol=0, atol=1e-12)
            or not np.all(np.isfinite(r)) or np.any(r < 0) or np.any(r > 1)
            or not np.all(np.isfinite(bonus)) or np.any(bonus < 0)):
        raise ValueError('Invalid proposal inputs')
    shift = p - np.einsum('sa,san->sn', beta, p)[:, None, :]
    v, low, high = np.zeros(s), np.zeros(s), np.zeros(s)
    cap, policies, penalties = 0., [], []
    index = np.arange(s)
    for _ in range(k_max+1):
        cap = 1 + gamma*cap
        q = r + gamma*np.einsum('san,n->sa', p, v)
        propagated = gamma*np.einsum('san,n->sa', np.abs(shift), high-low)
        lower_q = r + gamma*np.einsum('san,n->sa', p, low) - bonus
        score = q - bonus - propagated if selective else lower_q
        pi = score.argmax(-1)
        upper_q = r + gamma*np.einsum('san,n->sa', p, high) + bonus
        low = np.maximum(0, lower_q[index, pi])
        high = np.minimum(cap, upper_q[index, pi])
        v = np.clip(q[index, pi], low, high)
        policies.append(pi)
        penalties.append(float(propagated.max()))
    return np.array(policies), np.array(penalties)


def _signed_value(box, gamma, policy, reward, maximize):
    """Robust evaluation with a signed auxiliary state reward."""
    v = np.zeros(len(reward))
    for _ in range(100000):
        update = reward + gamma*np.sum(policy*_expectation(
            v, box.p_lower, box.p_upper, maximize=maximize), axis=-1)
        error = gamma*np.max(np.abs(update-v))/(1-gamma)
        v = update
        if error <= 1e-10:
            return v, error + 100*np.finfo(float).eps*len(v)/(1-gamma)**2
    raise RuntimeError('Signed robust iteration failed to converge')


def _value_interval(box, gamma, pi, cache):
    key = pi.tobytes()
    if key not in cache:
        low, le = robust_values(box, gamma, policy=pi, maximize=False)
        high, he = robust_values(box, gamma, policy=pi, maximize=True)
        cache[key] = (np.maximum(0, low-le), np.minimum(1/(1-gamma), high+he))
    return cache[key]


def coupled_difference(box, gamma, rho0, policy, reference, *, cache=None):
    """Bound (1-gamma)*(J(policy)-J(reference))/Rmax over one model set.

    D = (r_pi-r_ref) + gamma*(P_pi-P_ref)*V_pi + gamma*P_ref*D.
    Shared action coefficients cancel before uncertainty is propagated.
    The returned interval intersects the independent-value bound, so it cannot
    be wider. This is policy improvement, NOT global near-optimality.
    """
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError('Discount must be below one')
    s, a = box.r_lower.shape
    pi, ref = _policy_matrix(policy, s, a), _policy_matrix(reference, s, a)
    rho = np.asarray(rho0, float)
    if (rho.shape != (s,) or not np.all(np.isfinite(rho)) or np.any(rho < 0)
            or not np.isclose(rho.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError('Known initial distribution required')
    cache = {} if cache is None else cache
    vl, vu = _value_interval(box, gamma, pi, cache)
    bl, bu = _value_interval(box, gamma, ref, cache)
    oldlo = max(-1., float((1-gamma)*(rho@(vl-bu))))
    oldhi = min(1., float((1-gamma)*(rho@(vu-bl))))
    w = pi-ref
    if np.all(w == 0):
        return dict(lower=0., upper=0., independent_lower=oldlo,
                    independent_upper=oldhi, identical_policy=True)
    pos, neg = np.maximum(w, 0), np.minimum(w, 0)
    rl, ru = box.r_lower/box.reward_bound, box.r_upper/box.reward_bound
    reward_lo = np.sum(pos*rl+neg*ru, -1)
    reward_hi = np.sum(pos*ru+neg*rl, -1)
    mid, half = (vl+vu)/2, (vu-vl)/2
    expect_lo = _expectation(mid, box.p_lower, box.p_upper, maximize=False)
    expect_hi = _expectation(mid, box.p_lower, box.p_upper, maximize=True)
    shift_mid_lo = np.sum(pos*expect_lo+neg*expect_hi, -1)
    shift_mid_hi = np.sum(pos*expect_hi+neg*expect_lo, -1)
    shift_lo = np.einsum('sa,san->sn', pos, box.p_lower) + np.einsum('sa,san->sn', neg, box.p_upper)
    shift_hi = np.einsum('sa,san->sn', pos, box.p_upper) + np.einsum('sa,san->sn', neg, box.p_lower)
    # Both inequalities hold for every true shift in this confidence set.
    radius = np.minimum(np.maximum(np.abs(shift_lo), np.abs(shift_hi))@half,
                        np.abs(w).sum(-1)*half.max())
    g_low = reward_lo + gamma*(shift_mid_lo-radius)
    g_high = reward_hi + gamma*(shift_mid_hi+radius)
    dl, el = _signed_value(box, gamma, ref, g_low, False)
    du, eu = _signed_value(box, gamma, ref, g_high, True)
    lower = max(oldlo, float((1-gamma)*(rho@dl-el)))
    upper = min(oldhi, float((1-gamma)*(rho@du+eu)))
    if lower > upper + 1e-8:
        raise RuntimeError('Inconsistent coupled interval')
    return dict(lower=lower, upper=upper, independent_lower=oldlo,
                independent_upper=oldhi, identical_policy=False)


def lookahead_support_audit(p, box, policy, *, xi=.25, depths=(1, 2, 5, 10)):
    """LP-MDP applicability diagnostic, not an LP-MDP learner or guarantee.

    Use a TV upper bound and contrast nominal support with confidence-possible
    support. No true transition or true TV error is accepted by this function.
    """
    s, a = box.r_lower.shape
    pi = _policy_matrix(policy, s, a)
    radius = .5*np.sum(np.maximum(np.abs(p-box.p_lower), np.abs(p-box.p_upper)), -1)
    uncertain = np.minimum(1, radius) >= xi
    bad_states = np.any(uncertain & (pi > 0), axis=-1)
    results = {}
    for name, support in [('nominal', p > 0), ('confidence', box.p_upper > 0)]:
        next_support = np.any(support & (pi > 0)[..., None], axis=1)
        reach = support.copy()
        rescued = np.zeros((s, a), dtype=bool)
        for k in range(1, max(depths)+1):
            rescued |= uncertain & ~np.any(reach & bad_states, axis=-1)
            if k in depths:
                results[f'{name}_rescued_k{k}'] = int(rescued.sum())
            reach = np.einsum('san,nm->sam', reach.astype(np.int64), next_support.astype(np.int64)) > 0
    results['uncertain_rows'] = int(uncertain.sum())
    results['total_rows'] = s*a
    return results


def selective_profiles(arrays, rho0, *, num_states, num_actions, gamma=.95,
                       k_max=100, reward_bound=1., delta=.05, epsilon=.05):
    c, rs, rss, terminals = trajectory_statistics(arrays, num_states=num_states,
        num_actions=num_actions, reward_bound=reward_bound)
    box = terminal_box(kl_confidence_box(c, rs, rss, reward_bound=reward_bound, delta=delta), terminals)
    p, r, beta, n = nominal_model(c, rs, terminals, reward_bound=reward_bound)
    bonus = np.sqrt(np.log(num_states*num_actions*(k_max+1)/delta)/np.maximum(n, 1))
    bonus[terminals] = 0
    proposals = {'empirical': empirical_depth_policies(c, rs, terminals,
        gamma=gamma, k_max=k_max, reward_bound=reward_bound)}
    for name in ('pvi', 'spvi'):
        proposals[name], _ = selective_proposals(p, r, beta, bonus, gamma=gamma,
                                                k_max=k_max, selective=name == 'spvi')
    reference = proposals['empirical'][0]
    combined = certify_depths(box, gamma, rho0, np.concatenate(list(proposals.values())), epsilon=epsilon)
    result, cache, pairs = {}, {}, {}
    for j, (name, policies) in enumerate(proposals.items()):
        intervals = []
        for pi in policies:
            key = pi.tobytes()
            if key not in pairs:
                pairs[key] = coupled_difference(box, gamma, rho0, pi, reference, cache=cache)
            intervals.append(pairs[key])
        u = np.array(combined['normalized_regret_upper'][j*(k_max+1):(j+1)*(k_max+1)])
        lo = np.array([x['lower'] for x in intervals])
        selected = int(np.flatnonzero(u <= u.min()+1e-9)[0])
        improved = int(np.argmax(lo)) if lo.max() > 1e-9 else None
        result[name] = dict(policies=policies, upper=u, intervals=intervals,
            selected_depth=selected, improvement_depth=improved)
    return result, reference, lookahead_support_audit(p, box, proposals['empirical'][-1])
