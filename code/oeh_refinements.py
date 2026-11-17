"""Exploratory OEH refinements: KL confidence and separate policy proposals.

These are mechanism ablations, not implementations of LP-MDP or SUP. Frozen
OEH-R remains untouched. See docs/research/OEH_REFINEMENTS_2026-09-18.md.
"""
from __future__ import annotations

import numpy as np
from scipy.special import rel_entr

from oeh_certificate import (ModelBox, confidence_box, certify_depths,
                             pessimistic_depth_policies)
from oeh_certificate_adapter import trajectory_statistics


def kl_intervals(means, visits, *, coordinates: int, delta: float):
    """Time-uniform coordinate intervals for bounded [0,1] observations.

    n*kl(mean, candidate_mean) <= log(2*d*n*(n+1)/delta).
    Two Chernoff tails and the sum over all visit counts give coverage.
    Bisection returns OUTER endpoints, including p=0/1 and n=0.
    This is not a multinomial KL ball or an anytime KL-UCB implementation.
    """
    p, n = np.broadcast_arrays(np.asarray(means, float), np.asarray(visits, float))
    if (not np.isfinite(delta) or not 0 < delta < 1 or coordinates < 1
            or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1)
            or not np.all(np.isfinite(n)) or np.any(n < 0) or np.any(n != np.floor(n))):
        raise ValueError("Invalid KL inputs.")
    safe_n = np.maximum(n, 1)
    threshold = (np.log(2*coordinates) + np.log(safe_n) + np.log(safe_n+1)
                 - np.log(delta)) / safe_n
    def divergence(q):
        return rel_entr(p, q) + rel_entr(1-p, 1-q)
    low, high = np.zeros_like(p), p.copy()
    for _ in range(64):
        mid = (low+high)/2
        outside = divergence(mid) > threshold
        low = np.where(outside, mid, low)
        high = np.where(outside, high, mid)
    lower = np.where(n == 0, 0, low)
    low, high = p.copy(), np.ones_like(p)
    for _ in range(64):
        mid = (low+high)/2
        outside = divergence(mid) > threshold
        high = np.where(outside, mid, high)
        low = np.where(outside, low, mid)
    upper = np.where(n == 0, 1, high)
    return lower, upper


def kl_confidence_box(counts, reward_sums, reward_square_sums, *, reward_bound=1., delta=.05):
    # Use the existing strict validation of integer counts and bounded moments.
    confidence_box(counts, reward_sums, reward_square_sums,
                   reward_bound=reward_bound, delta=delta, method="hoeffding")
    n = np.asarray(counts).sum(-1)
    denom = np.maximum(n, 1)
    p = counts / denom[..., None]
    r = np.asarray(reward_sums) / denom / reward_bound
    s, a = n.shape
    d = s*a*(s+1)
    plo, phi = kl_intervals(p, n[..., None], coordinates=d, delta=delta)
    rlo, rhi = kl_intervals(np.clip(r, 0, 1), n, coordinates=d, delta=delta)
    return ModelBox(plo, phi, reward_bound*rlo, reward_bound*rhi, reward_bound, delta)


def terminal_box(box, terminals):
    plo, phi, rlo, rhi = (x.copy() for x in (box.p_lower, box.p_upper, box.r_lower, box.r_upper))
    for s in terminals:
        plo[s], phi[s], rlo[s], rhi[s] = 0, 0, 0, 0
        plo[s, :, s], phi[s, :, s] = 1, 1
    return ModelBox(plo, phi, rlo, rhi, box.reward_bound, box.delta)


def empirical_depth_policies(counts, reward_sums, terminals, *, gamma, k_max, reward_bound=1.):
    """Propose supported nominal policies; uncertainty remains in certification.

    Unvisited rows use zero-reward self loops only to define the PROPOSAL model.
    They are never inserted into the confidence set. At visited states exclude
    unobserved actions from greedy choice. At unvisited states choose lowest id.
    k=0 is greedy w.r.t. empirical behavior's value, as in the prior convention.
    """
    if not 0 <= gamma < 1 or not isinstance(k_max, int) or k_max < 0:
        raise ValueError("Invalid discount/depth.")
    n = np.asarray(counts).sum(-1)
    states, actions = n.shape
    p = np.asarray(counts, dtype=float)/np.maximum(n, 1)[..., None]
    r = np.asarray(reward_sums, dtype=float)/np.maximum(n, 1)/reward_bound
    for s, a in zip(*np.where(n == 0)):
        p[s, a, s] = 1
    for s in terminals:
        p[s], r[s] = 0, 0
        p[s, :, s] = 1
    total = n.sum(-1, keepdims=True)
    beta = np.divide(n, total, out=np.full(n.shape, 1/actions), where=total > 0)
    p_beta = np.einsum('sa,san->sn', beta, p)
    r_beta = np.sum(beta*r, axis=-1)
    v = np.linalg.solve(np.eye(states)-gamma*p_beta, r_beta)
    eligible = n > 0
    eligible[total[:, 0] == 0] = True
    policies = []
    for _ in range(k_max+1):
        q = r + gamma*np.einsum('san,n->sa', p, v)
        q = np.where(eligible, q, -np.inf)
        policies.append(np.argmax(q, axis=-1))
        v = q.max(-1)
    return np.array(policies)


def refinement_profiles(arrays, rho0, *, num_states, num_actions, gamma=.95,
                        k_max=100, reward_bound=1., epsilon=.05, delta=.05):
    """Fixed 2x2 ablation; no choice of method based on oracle/validation labels.

    delta applies per confidence construction, not to post-hoc selection across
    both constructions. All candidates within one construction are simultaneous.
    """
    c, rs, rss, terminals = trajectory_statistics(
        arrays, num_states=num_states, num_actions=num_actions, reward_bound=reward_bound)
    visits = c.sum(-1)
    totals = visits.sum(-1, keepdims=True)
    beta = np.divide(visits, totals, out=np.full(visits.shape, 1/num_actions), where=totals > 0)
    empirical = empirical_depth_policies(c, rs, terminals, gamma=gamma, k_max=k_max, reward_bound=reward_bound)
    results = {}
    for confidence in ("bernstein", "kl"):
        box = (confidence_box(c, rs, rss, reward_bound=reward_bound, delta=delta)
               if confidence == "bernstein" else
               kl_confidence_box(c, rs, rss, reward_bound=reward_bound, delta=delta))
        box = terminal_box(box, terminals)
        pessimistic = pessimistic_depth_policies(box, gamma, beta, k_max)
        # One simultaneous certificate computation also shares evaluations for
        # policies repeated between proposal families.
        combined = certify_depths(box, gamma, rho0, np.concatenate([pessimistic, empirical]), epsilon=epsilon)
        for j, (name, policies) in enumerate((("pessimistic", pessimistic), ("empirical", empirical))):
            upper = combined["normalized_regret_upper"][j*(k_max+1):(j+1)*(k_max+1)]
            passed = [k for k, u in enumerate(upper) if u <= epsilon]
            results[f"{confidence}_{name}"] = {
                "upper": np.array(upper), "policies": policies,
                "certified_depth": min(passed) if passed else None,
                "selected_depth": int(np.flatnonzero(np.array(upper) <= min(upper)+1e-9)[0]),
                "unique_policies": len({p.tobytes() for p in policies}),
                "scope": "per-method confidence certificate; no joint selection guarantee across methods",
            }
    return results
