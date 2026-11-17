"""Information-theoretic reference calculations for the final OEH definition.

This module deliberately implements only reductions with a closed-form solution.
It must not be presented as the general alternative-MDP KL optimizer described in
``docs/research/OEH_FINAL_FORMULATION_2026-09-19.md``.
"""
from __future__ import annotations

import math


def bernoulli_kl(p: float, q: float) -> float:
    """Return KL(Bernoulli(p) || Bernoulli(q)), including boundary cases."""
    p, q = float(p), float(q)
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("Bernoulli parameters must lie in [0, 1]")
    if p == q:
        return 0.0
    if q == 0.0:
        return math.inf if p > 0.0 else 0.0
    if q == 1.0:
        return math.inf if p < 1.0 else 0.0
    left = 0.0 if p == 0.0 else p * math.log(p / q)
    right = 0.0 if p == 1.0 else (1.0-p) * math.log((1.0-p) / (1.0-q))
    return left + right


def two_action_bernoulli_information(
        p_first: float, p_second: float,
        d_first: float, d_second: float,
        *, tie_tolerance: float = 0.0) -> float:
    """Fixed-behavior KL rate for identifying one best action in a two-arm bandit.

    The sampling weights ``d_first`` and ``d_second`` are expected observations
    per locked sampling unit.  This is the exact-best (epsilon=0) problem.  At a
    true tie, arbitrarily close alternatives make either fixed answer suboptimal,
    so the uniform-certification information rate is zero.  Missing coverage also
    gives rate zero.
    """
    p_first, p_second = float(p_first), float(p_second)
    d_first, d_second = float(d_first), float(d_second)
    if not (0.0 <= p_first <= 1.0 and 0.0 <= p_second <= 1.0):
        raise ValueError("Bernoulli means must lie in [0, 1]")
    if not (math.isfinite(d_first) and math.isfinite(d_second)
            and d_first >= 0.0 and d_second >= 0.0):
        raise ValueError("Sampling weights must be finite and nonnegative")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if abs(p_first-p_second) <= tie_tolerance:
        return 0.0
    if d_first == 0.0 or d_second == 0.0:
        return 0.0
    total = d_first+d_second
    boundary_mean = (d_first*p_first+d_second*p_second)/total
    return (d_first*bernoulli_kl(p_first, boundary_mean)
            + d_second*bernoulli_kl(p_second, boundary_mean))


def identification_scale(information_rate: float, *, delta: float = 0.05,
                         comparison_factor: float = 1.0) -> float:
    """Map an information rate to the final OEH characteristic sample scale."""
    information_rate = float(information_rate)
    delta, comparison_factor = float(delta), float(comparison_factor)
    if (math.isnan(information_rate) or information_rate < 0.0
            or not 0.0 < delta < 1.0
            or not math.isfinite(comparison_factor)
            or comparison_factor < 1.0):
        raise ValueError("Invalid information-scale arguments")
    if information_rate == 0.0:
        return math.inf
    if information_rate == math.inf:
        return 0.0
    return math.log(comparison_factor/delta)/information_rate
