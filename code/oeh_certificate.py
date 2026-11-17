"""Exploratory OEH redesign: a normalized policy-regret certificate.

Independent of the frozen OEH estimator. See docs/research/OEH_REDESIGN_2026-09-18.md.
Counts must summarize raw, unfiltered stationary Markov observations, with
bounded rewards and the complete public state/action space. No bootstrap counts,
oracle model, future-count multiplier, or benchmark-normalized-return guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelBox:
    """Rectangular confidence set; direct construction is for analytical tests."""

    p_lower: np.ndarray
    p_upper: np.ndarray
    r_lower: np.ndarray
    r_upper: np.ndarray
    reward_bound: float
    delta: float | None = None

    def __post_init__(self):
        for name in ("p_lower", "p_upper", "r_lower", "r_upper"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError("Confidence endpoints must be finite.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        lo, hi = self.p_lower, self.p_upper
        if lo.ndim != 3 or min(lo.shape) < 1 or lo.shape[0] != lo.shape[2]:
            raise ValueError("Transitions must have shape (S, A, S).")
        if hi.shape != lo.shape or self.r_lower.shape != lo.shape[:2]:
            raise ValueError("Inconsistent model dimensions.")
        if self.r_upper.shape != self.r_lower.shape:
            raise ValueError("Inconsistent reward dimensions.")
        if not np.isfinite(self.reward_bound) or self.reward_bound <= 0:
            raise ValueError("A positive, known reward bound is required.")
        if self.delta is not None and not 0 < self.delta < 1:
            raise ValueError("delta must be in (0,1).")
        if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi):
            raise ValueError("Invalid transition intervals.")
        if np.any(lo.sum(-1) > 1 + 1e-12) or np.any(hi.sum(-1) < 1 - 1e-12):
            raise ValueError("Transition intervals have no simplex intersection.")
        if (np.any(self.r_lower < 0) or np.any(self.r_upper > self.reward_bound)
                or np.any(self.r_lower > self.r_upper)):
            raise ValueError("Reward intervals must lie in [0, reward_bound].")


def confidence_box(next_counts, reward_sums, reward_square_sums, *,
                   reward_bound: float, delta: float = 0.05,
                   method: str = "bernstein") -> ModelBox:
    """Simultaneous coordinate confidence intervals, uniform over visit counts.

    There are d=S*A*(S+1) scalar means. Allocate delta/[d*n*(n+1)]
    to each coordinate at count n. Two-sided empirical Bernstein uses
    Maurer--Pontil (2009), Theorem 4; Hoeffding is an ablation.
    Count 0 retains full uncertainty; Bernstein count 1 does too.
    Markov dependence is handled via per-(s,a) outcome stacks and the union
    over all n, NOT by assuming the complete transition log is iid.
    """
    c = np.asarray(next_counts, dtype=float)
    rs, rss = np.asarray(reward_sums, dtype=float), np.asarray(reward_square_sums, dtype=float)
    if c.ndim != 3 or min(c.shape) < 1 or c.shape[0] != c.shape[2]:
        raise ValueError("Counts must have shape (S,A,S).")
    if (not np.all(np.isfinite(c)) or np.any(c < 0)
            or np.any(c != np.floor(c)) or np.any(c > 2**53)):
        raise ValueError("Counts must be finite, nonnegative, exactly represented integers.")
    if not np.isfinite(reward_bound) or reward_bound <= 0 or not 0 < delta < 1:
        raise ValueError("Invalid reward bound or delta.")
    if method not in ("bernstein", "hoeffding"):
        raise ValueError("Unknown confidence method.")
    n = c.sum(-1)
    if np.any(n > 2**53):
        raise ValueError("Visit counts exceed exact floating-point representation.")
    if rs.shape != n.shape or rss.shape != n.shape:
        raise ValueError("Reward summaries must have shape (S,A).")
    # Work in public reward units; the observed maximum is not a valid bound.
    u, u2 = rs / reward_bound, rss / reward_bound**2
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(u2)):
        raise ValueError("Reward moments must be finite.")
    tol = 1e-10 * np.maximum(1, n)
    if (np.any(u < 0) or np.any(u > n + tol) or np.any(u2 < 0)
            or np.any(u2 > u + tol)
            or np.any(u2 + tol < u**2 / np.maximum(n, 1))
            or np.any((n == 0) & ((u != 0) | (u2 != 0)))):
        raise ValueError("Inconsistent bounded-reward moments.")
    denom = np.maximum(n, 1)
    p = c / denom[..., None]
    r = u / denom
    vp = p * (1 - p) * (n / np.maximum(n - 1, 1))[..., None]
    vr = np.maximum(0, (u2 - u**2 / denom) / np.maximum(n - 1, 1))
    d = c.shape[0] * c.shape[1] * (c.shape[2] + 1)
    # Log-space evaluation avoids underflow for small delta / large counts.
    ell = np.log(4 if method == "bernstein" else 2) + np.log(d) - np.log(delta)
    ell = ell + np.log(denom) + np.log(denom + 1)
    if method == "bernstein":
        remainder = 7 * ell / (3 * np.maximum(n - 1, 1))
        rp = np.sqrt(2 * vp * (ell / denom)[..., None]) + remainder[..., None]
        rr = np.sqrt(2 * vr * ell / denom) + remainder
        rp[n < 2], rr[n < 2] = 1, 1
    else:
        rr = np.sqrt(ell / (2 * denom))
        rp = np.broadcast_to(rr[..., None], c.shape).copy()
        rp[n == 0], rr[n == 0] = 1, 1
    return ModelBox(
        np.clip(p - rp, 0, 1), np.clip(p + rp, 0, 1),
        reward_bound * np.clip(r - rr, 0, 1),
        reward_bound * np.clip(r + rr, 0, 1), reward_bound, delta,
    )


def _expectation(values, lower, upper, *, maximize):
    """Exact linear optimization on each box intersected with the simplex."""
    mass = np.maximum(0, 1 - lower.sum(-1))
    answer = np.sum(lower * values, axis=-1)
    order = np.argsort(values, kind="stable")
    if maximize:
        order = order[::-1]
    for state in order:
        extra = np.minimum(mass, upper[..., state] - lower[..., state])
        answer += extra * values[state]
        mass = np.maximum(0, mass - extra)
    if np.any(mass > 1e-10):
        raise ValueError("Empty transition confidence set.")
    return answer


def _policy_matrix(policy, states, actions):
    p = np.asarray(policy)
    if p.shape == (states,):
        if (not np.all(np.isfinite(p)) or np.any(p != np.floor(p))
                or np.any(p < 0) or np.any(p >= actions)):
            raise ValueError("Invalid deterministic policy.")
        p = np.eye(actions)[p.astype(int)]
    if (p.shape != (states, actions) or not np.all(np.isfinite(p))
            or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Policy rows must be probability vectors.")
    return p


def _validate_solver(gamma, tolerance, max_iterations):
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0,1).")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A positive solver tolerance is required.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")


def robust_values(box: ModelBox, gamma: float, *, policy=None, maximize=True,
                  tolerance=1e-10, max_iterations=100000):
    """Return fixed-point approximation and a sup-norm error allowance.

    policy=None performs control; otherwise evaluate a fixed policy.
    Calculations use rewards divided by the PUBLIC bound for scale invariance.
    Error allowance is the contraction residual bound, with a floating-point
    margin; it is not interval-arithmetic verification of machine operations.
    """
    _validate_solver(gamma, tolerance, max_iterations)
    states, actions = box.r_lower.shape
    pi = None if policy is None else _policy_matrix(policy, states, actions)
    rewards = (box.r_upper if maximize else box.r_lower) / box.reward_bound
    v = np.zeros(states)
    for _ in range(max_iterations):
        q = rewards + gamma * _expectation(v, box.p_lower, box.p_upper, maximize=maximize)
        updated = q.max(-1) if pi is None else np.sum(pi * q, axis=-1)
        change = np.max(np.abs(updated - v))
        v = updated
        residual_error = gamma * change / (1 - gamma)
        if residual_error <= tolerance:
            # Rounding allowance can exceed the requested iteration tolerance.
            # Include it in the reported bound instead of making convergence
            # impossible for large state spaces or gamma close to one.
            error = residual_error + 100 * np.finfo(float).eps * states / (1 - gamma)**2
            return v, float(error)
    raise RuntimeError("Robust value iteration did not meet its error tolerance.")


def pessimistic_depth_policies(box: ModelBox, gamma: float, baseline_policy,
                              k_max: int, *, tolerance=1e-10):
    """Exploratory candidate family, distinct from frozen OEH-v1 backups.

    Begin at the pessimistic value of the baseline. k=0 means one greedy
    improvement relative to that leaf value, matching the original indexing.
    Include all public actions, retaining uncertainty at unobserved pairs.
    """
    if not isinstance(k_max, int) or k_max < 0:
        raise ValueError("k_max must be a nonnegative integer.")
    v, _ = robust_values(box, gamma, policy=baseline_policy, maximize=False, tolerance=tolerance)
    policies = []
    for _ in range(k_max + 1):
        q = box.r_lower / box.reward_bound + gamma * _expectation(
            v, box.p_lower, box.p_upper, maximize=False)
        policies.append(np.argmax(q, axis=-1))
        v = q.max(-1)
    return np.array(policies)


def certify_depths(box: ModelBox, gamma: float, rho0, policies, *,
                   epsilon: float, tolerance=1e-10):
    """Conservative regret bounds and the first CERTIFIED candidate depth.

    Requires a known initial distribution, not an unaccounted estimate.
    Success is conditional on the true model lying in box. Failure to
    certify means unresolved, never 'unlearnable' or a sample lower bound.
    This function intentionally does not return a future data requirement.
    """
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1).")
    states, actions = box.r_lower.shape
    rho = np.asarray(rho0, dtype=float)
    if (rho.shape != (states,) or not np.all(np.isfinite(rho)) or np.any(rho < 0)
            or not np.isclose(rho.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("rho0 must be a known probability vector.")
    candidates = [_policy_matrix(p, states, actions) for p in policies]
    if not candidates:
        raise ValueError("At least one candidate policy is required.")
    v_opt, e_opt = robust_values(box, gamma, maximize=True, tolerance=tolerance)
    upper = []
    evaluated = {}
    for pi in candidates:
        key = pi.tobytes()
        if key not in evaluated:
            evaluated[key] = robust_values(box, gamma, policy=pi, maximize=False, tolerance=tolerance)
        v_pi, e_pi = evaluated[key]
        bound = (1 - gamma) * (rho @ (v_opt - v_pi) + e_opt + e_pi)
        # One available action admits no control regret in any compatible MDP.
        upper.append(0.0 if actions == 1 else float(np.clip(bound, 0, 1)))
    passed = [i for i, u in enumerate(upper) if u <= epsilon]
    return {
        "status": "certified" if passed else "uncertified_within_depth_grid",
        "certified_depth": min(passed) if passed else None,
        "normalized_regret_upper": upper,
        "epsilon": epsilon,
        "delta": box.delta,
        "normalization": "reward_bound/(1-gamma); not benchmark NR",
        "sample_requirement": None,
        "unique_policy_count": len(evaluated),
        "scope": "exploratory confidence certificate; not validated difficulty predictor",
    }
