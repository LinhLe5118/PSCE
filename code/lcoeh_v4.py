"""Learner-conditional, pilot-only effective-horizon diagnostics.

LC-OEH uses the three registered nested pilot prefixes rather than learner
outcomes.  At each prefix it trains the actual candidate learner on four fifths
of the pilot trajectories and evaluates that policy in a pessimistically
completed empirical model built from the held-out fifth.  An unsupported
validation action receives zero continuation and zero reward; it is never
silently renormalized away.

The resulting supported-return risk is fit with the rate model

    risk(N) = risk(N0) * (N/N0)^(-alpha),  0 <= alpha <= 1.

The exponent is the median log--log slope across the registered prefix pairs.
The interval [0,1] is a structural rate constraint: zero represents no observed
contraction and one permits deterministic fast rates.  It is not selected from
outcome labels.  A one-sided 95% Gaussian cross-fit lower score is also reported
for comparison with pessimistic/FQE model-selection practice.

All inputs are pilot transitions.  The module has no interface for training
streams, learner returns, empirical requirements, or benchmark oracles.
"""
from __future__ import annotations

import hashlib
import math
from typing import Mapping

import numpy as np

from adaptive_pilot import transitions_from_arrays
from empirical_requirement import (
    _model, behavior_cloning_tabular, cfqi_tabular, iql_tabular,
)


ALGORITHMS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
Z_ONE_SIDED_95 = 1.6448536269514722


def _subset(arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[mask] for name, values in arrays.items()}


def _trajectory_folds(arrays: Mapping[str, np.ndarray], *, folds: int,
                      seed_key: str) -> tuple[np.ndarray, str, int]:
    trajectory = np.asarray(arrays["trajectory_id"], dtype=int)
    unique = np.unique(trajectory)
    if len(unique) == 1:
        # Some long-horizon MDPs produce a single episode at the smallest
        # registered prefix. Two contiguous time blocks retain a held-out
        # partition instead of pretending that transitions are independent
        # trajectories. The returned mode exposes this weaker evidence.
        if len(trajectory) < 2:
            raise ValueError("LC-OEH requires at least two pilot transitions")
        assignment = np.zeros(len(trajectory), dtype=int)
        assignment[len(trajectory) // 2:] = 1
        return assignment, "single_trajectory_time_blocks", 1
    actual_folds = min(folds, len(unique))
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    assignment = {
        int(value): position % actual_folds
        for position, value in enumerate(shuffled)
    }
    return (np.asarray([assignment[int(value)] for value in trajectory], dtype=int),
            "trajectory", min(len(unique), folds))


def _fit_policy(algorithm: str, arrays: Mapping[str, np.ndarray], *,
                num_states: int, num_actions: int, gamma: float) -> np.ndarray:
    transitions = transitions_from_arrays(arrays)
    if algorithm == "cfqi_tabular":
        return cfqi_tabular(
            transitions, num_states=num_states, num_actions=num_actions,
            gamma=gamma,
        )
    if algorithm == "iql_tabular":
        return iql_tabular(
            transitions, num_states=num_states, num_actions=num_actions,
            gamma=gamma,
        )
    if algorithm == "behavior_cloning_tabular":
        return behavior_cloning_tabular(
            transitions, num_states=num_states, num_actions=num_actions,
        )
    raise ValueError(f"Unknown learner: {algorithm}")


def _policy_value(p: np.ndarray, r: np.ndarray, policy: np.ndarray,
                  rho: np.ndarray, gamma: float) -> float:
    transition = np.einsum("sa,san->sn", policy, p)
    reward = np.sum(policy * r, axis=1)
    values = np.linalg.solve(np.eye(len(rho)) - gamma * transition, reward)
    return float(rho @ values)


def _restricted_value(p: np.ndarray, r: np.ndarray, counts: np.ndarray,
                      rho: np.ndarray, gamma: float, *, minimize: bool) -> float:
    supported = counts > 0
    values = np.zeros(len(rho), dtype=float)
    for _ in range(10_000):
        q = r + gamma * np.einsum("san,n->sa", p, values)
        updated = np.zeros_like(values)
        for state in range(len(values)):
            choices = q[state, supported[state]]
            if len(choices):
                updated[state] = float(np.min(choices) if minimize else np.max(choices))
        if float(np.max(np.abs(updated - values))) <= 1e-11:
            values = updated
            break
        values = updated
    return float(rho @ values)


def _crossfit_score(arrays: Mapping[str, np.ndarray], rho: np.ndarray, *,
                    algorithm: str, num_states: int, num_actions: int,
                    gamma: float, folds: int, seed_key: str) -> dict[str, float]:
    assignment, fold_mode, independent_units = _trajectory_folds(
        arrays, folds=folds, seed_key=seed_key)
    actual_folds = int(np.max(assignment)) + 1
    scores, support_scores = [], []
    for fold in range(actual_folds):
        train = _subset(arrays, assignment != fold)
        validation = _subset(arrays, assignment == fold)
        policy = _fit_policy(
            algorithm, train, num_states=num_states,
            num_actions=num_actions, gamma=gamma,
        )
        p, r, counts, _terminal = _model(
            transitions_from_arrays(validation), num_states, num_actions)
        # `_model` leaves unsupported rows at zero.  Policy mass on such rows
        # therefore receives the declared pessimistic zero completion.
        value = _policy_value(p, r, policy, rho, gamma)
        best = _restricted_value(p, r, counts, rho, gamma, minimize=False)
        worst = _restricted_value(p, r, counts, rho, gamma, minimize=True)
        scale = best - worst
        score = 0.5 if scale <= 1e-12 else (value - worst) / scale
        scores.append(float(np.clip(score, 0.0, 1.0)))
        empirical_state_mass = np.bincount(
            np.asarray(validation["state"], dtype=int), minlength=num_states,
        ).astype(float)
        empirical_state_mass /= max(empirical_state_mass.sum(), 1.0)
        fold_supported_mass = np.sum(policy * (counts > 0), axis=1)
        support_scores.append(float(empirical_state_mass @ fold_supported_mass))
    values = np.asarray(scores, dtype=float)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(actual_folds))
    return {
        "crossfit_return_mean": mean,
        "crossfit_return_se": standard_error,
        "pessimistic_return": float(np.clip(
            mean - Z_ONE_SIDED_95 * standard_error, 0.0, 1.0)),
        "validation_support_mass": float(np.mean(support_scores)),
        "fold_mode": fold_mode,
        "folds": actual_folds,
        "independent_trajectory_units": independent_units,
    }


def _rate_and_requirement(sizes: np.ndarray, risks: np.ndarray, *,
                          epsilon: float) -> tuple[float, float | None]:
    sizes = np.asarray(sizes, dtype=float)
    risks = np.asarray(risks, dtype=float)
    if (sizes.ndim != 1 or risks.shape != sizes.shape or len(sizes) < 2
            or np.any(~np.isfinite(sizes)) or np.any(~np.isfinite(risks))
            or np.any(sizes <= 0) or np.any(np.diff(sizes) <= 0)
            or np.any(risks <= 0) or not 0 < epsilon < 1):
        raise ValueError("Invalid LC-OEH rate curve")
    slopes = []
    for left in range(len(sizes)):
        for right in range(left + 1, len(sizes)):
            slope = -math.log(risks[right] / risks[left]) / math.log(
                sizes[right] / sizes[left])
            slopes.append(slope)
    alpha = float(np.clip(np.median(slopes), 0.0, 1.0))
    crossing = np.flatnonzero(risks <= epsilon)
    if len(crossing):
        return alpha, float(sizes[int(crossing[0])])
    if alpha <= 1e-12:
        return alpha, None
    log_requirement = (math.log(sizes[-1])
                       + math.log(risks[-1] / epsilon) / alpha)
    requirement = (math.exp(log_requirement)
                   if log_requirement <= math.log(np.finfo(float).max) else None)
    return alpha, requirement


def _rate_summary(sizes: np.ndarray, risks: np.ndarray, *, epsilon: float):
    """Return rate, stable log requirement, and whether crossing was observed."""
    alpha, requirement = _rate_and_requirement(sizes, risks, epsilon=epsilon)
    crossing = np.flatnonzero(risks <= epsilon)
    if len(crossing):
        value = float(sizes[int(crossing[0])])
        return alpha, value, math.log(value), True
    if alpha <= 1e-12:
        return alpha, None, math.log(float(sizes[-1])), False
    log_requirement = (math.log(float(sizes[-1]))
                       + math.log(float(risks[-1]) / epsilon) / alpha)
    return alpha, requirement, log_requirement, False


def estimate_lcoeh_v4(
    pilot_prefixes: Mapping[int, Mapping[str, np.ndarray]],
    initial_state_distribution,
    *, num_states: int, num_actions: int, gamma: float = .95,
    epsilon: float = .05, folds: int = 5, seed_key: str = "",
) -> dict[str, dict[str, float | None]]:
    """Return learner-conditional pilot diagnostics for every candidate learner."""
    if tuple(sorted(pilot_prefixes)) != (2, 5, 10):
        raise ValueError("LC-OEH-v4 requires the registered pilot multipliers 2, 5, 10")
    if folds != 5:
        raise ValueError("LC-OEH-v4 freezes five trajectory folds")
    if not 0.0 < gamma < 1.0 or not 0.0 < epsilon < 1.0:
        raise ValueError("gamma and epsilon must lie strictly between zero and one")
    rho = np.asarray(initial_state_distribution, dtype=float)
    if rho.shape != (num_states,) or np.any(rho < 0) or not np.isclose(rho.sum(), 1):
        raise ValueError("Invalid initial-state distribution")
    result: dict[str, dict[str, float | None]] = {}
    for algorithm in ALGORITHMS:
        curve = []
        for multiplier in (2, 5, 10):
            arrays = pilot_prefixes[multiplier]
            score = _crossfit_score(
                arrays, rho, algorithm=algorithm, num_states=num_states,
                num_actions=num_actions, gamma=gamma, folds=folds,
                seed_key=f"{seed_key}|{multiplier}",
            )
            curve.append({"multiplier": multiplier, "size": len(arrays["state"]), **score})
        sizes = np.asarray([item["size"] for item in curve], dtype=float)
        risks = np.clip(
            1.0 - np.asarray([item["crossfit_return_mean"] for item in curve]),
            1e-8, 1.0,
        )
        alpha, requirement, log_requirement, observed = _rate_summary(
            sizes, risks, epsilon=epsilon)
        primary = curve[-1]
        result[algorithm] = {
            # log(1 + exp(x)) is evaluated stably and never requires clipping a
            # potentially enormous extrapolated sample requirement.
            "lcoeh4_log1p_requirement": float(np.logaddexp(0.0, log_requirement)),
            "lcoeh4_requirement_censored": float(not observed),
            "lcoeh4_risk_at_pilot": float(risks[-1]),
            "lcoeh4_contraction_rate": alpha,
            "lcoeh4_pessimistic_return": primary["pessimistic_return"],
            "lcoeh4_return_se": primary["crossfit_return_se"],
            "lcoeh4_validation_support_mass": primary["validation_support_mass"],
            "lcoeh4_trajectory_independence_fraction": min(
                float(item["independent_trajectory_units"]) for item in curve
            ) / folds,
            "requirement": requirement,
            "curve": curve,
        }
    return result


LC_OEH_V4_FEATURES = (
    "lcoeh4_log1p_requirement",
    "lcoeh4_requirement_censored",
    "lcoeh4_risk_at_pilot",
    "lcoeh4_contraction_rate",
    "lcoeh4_pessimistic_return",
    "lcoeh4_return_se",
    "lcoeh4_validation_support_mass",
    "lcoeh4_trajectory_independence_fraction",
)


__all__ = ["ALGORITHMS", "LC_OEH_V4_FEATURES", "estimate_lcoeh_v4"]
