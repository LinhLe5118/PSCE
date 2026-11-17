"""Data-only tabular learners and registered empirical-requirement rule.

This module deliberately contains no oracle imports.  A caller supplies an
evaluation callback only after fitting policies to independent training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np


class TransitionLike(Protocol):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: bool


@dataclass(frozen=True)
class EmpiricalRequirement:
    requirement: int | None
    right_censored: bool
    success_counts: tuple[int, ...]
    success_rates: tuple[float, ...]


def _model(
    transitions: Iterable[TransitionLike], states: int, actions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if states <= 0 or actions <= 0:
        raise ValueError("State and action counts must be positive.")
    counts = np.zeros((states, actions), dtype=np.float64)
    rewards = np.zeros((states, actions), dtype=np.float64)
    next_counts = np.zeros((states, actions, states), dtype=np.float64)
    terminal = np.zeros(states, dtype=np.bool_)
    for item in transitions:
        state, action, next_state = int(item.state), int(item.action), int(item.next_state)
        if not (0 <= state < states and 0 <= next_state < states and 0 <= action < actions):
            raise ValueError("Transition index outside model dimensions.")
        if not np.isfinite(item.reward):
            raise ValueError("Transition reward must be finite.")
        counts[state, action] += 1.0
        rewards[state, action] += float(item.reward)
        if bool(item.terminal):
            terminal[next_state] = True
        else:
            next_counts[state, action, next_state] += 1.0
    p_hat = np.divide(next_counts, counts[:, :, None], out=np.zeros_like(next_counts), where=counts[:, :, None] > 0)
    r_hat = np.divide(rewards, counts, out=np.zeros_like(rewards), where=counts > 0)
    return p_hat, r_hat, counts, terminal


def _greedy_policy(q: np.ndarray, counts: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    states, actions = counts.shape
    policy = np.full((states, actions), 1.0 / actions, dtype=np.float64)
    for state in range(states):
        supported = np.flatnonzero(counts[state] > 0.0)
        if terminal[state] or not len(supported):
            continue
        winner = min(supported, key=lambda action: (-float(q[state, action]), int(action)))
        policy[state] = 0.0
        policy[state, winner] = 1.0
    return policy


def behavior_cloning_tabular(
    transitions: Sequence[TransitionLike], *, num_states: int, num_actions: int,
) -> np.ndarray:
    """Maximum-likelihood tabular behavior cloning.

    Observed states reproduce the empirical action distribution.  Terminal and
    unseen states use the protocol's registered uniform fallback.  This learner
    is a development-only sanity check, not a registered ``N_emp`` target.
    """
    _, _, counts, terminal = _model(transitions, num_states, num_actions)
    policy = np.full((num_states, num_actions), 1.0 / num_actions, dtype=np.float64)
    totals = counts.sum(axis=1)
    observed = (totals > 0.0) & ~terminal
    policy[observed] = counts[observed] / totals[observed, None]
    return policy


def cfqi_tabular(
    transitions: Sequence[TransitionLike], *, num_states: int, num_actions: int,
    gamma: float, iterations: int = 100, tolerance: float = 1e-10,
    diagnostics: dict[str, float | int | bool] | None = None,
) -> np.ndarray:
    """Support-conservative fitted Q iteration; unsupported actions never max."""
    p_hat, r_hat, counts, terminal = _model(transitions, num_states, num_actions)
    q = np.zeros((num_states, num_actions), dtype=np.float64)
    if iterations <= 0 or not np.isfinite(tolerance) or tolerance < 0 or not 0 <= gamma < 1:
        raise ValueError("Invalid iteration budget, tolerance or discount.")
    for iteration in range(iterations):
        maxima = np.zeros(num_states, dtype=np.float64)
        for state in range(num_states):
            supported = np.flatnonzero(counts[state] > 0.0)
            if len(supported) and not terminal[state]:
                maxima[state] = float(np.max(q[state, supported]))
        updated = r_hat + gamma * np.einsum("san,n->sa", p_hat, maxima)
        updated[counts == 0.0] = 0.0
        final_update_norm = float(np.max(np.abs(updated - q)))
        if diagnostics is not None:
            diagnostics.update(iterations_used=iteration + 1, final_update_norm=final_update_norm,
                               converged=final_update_norm <= tolerance)
        if final_update_norm <= tolerance:
            q = updated
            break
        q = updated
    return _greedy_policy(q, counts, terminal)


def _expectile(values: np.ndarray, weights: np.ndarray, expectile: float) -> float:
    """Return the exact finite-support weighted expectile.

    Between two adjacent sorted action values, the asymmetric least-squares
    first-order condition is linear.  Enumerating those intervals is exact and
    avoids a fixed 100-step bisection inside every IQL state/update.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    for split in range(len(sorted_values) + 1):
        coefficients = np.empty_like(sorted_weights)
        coefficients[:split] = 1.0 - expectile
        coefficients[split:] = expectile
        effective = sorted_weights * coefficients
        candidate = float(np.dot(effective, sorted_values) / effective.sum())
        lower = float(sorted_values[split - 1]) if split else -np.inf
        upper = float(sorted_values[split]) if split < len(sorted_values) else np.inf
        if lower - 1e-15 <= candidate <= upper + 1e-15:
            return candidate
    raise RuntimeError("Failed to solve a finite-support weighted expectile.")


def iql_tabular(
    transitions: Sequence[TransitionLike], *, num_states: int, num_actions: int,
    gamma: float, expectile: float = 0.7, inverse_temperature: float = 3.0,
    iterations: int = 100, tolerance: float = 1e-10,
    diagnostics: dict[str, float | int | bool] | None = None,
) -> np.ndarray:
    """Tabular IQL with behavior-weighted expectile values and AWBC extraction."""
    if not 0.0 < expectile < 1.0 or (not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0):
        raise ValueError("IQL expectile and inverse temperature must be positive registered values.")
    p_hat, r_hat, counts, terminal = _model(transitions, num_states, num_actions)
    behavior = np.divide(counts, counts.sum(axis=1, keepdims=True), out=np.zeros_like(counts), where=counts.sum(axis=1, keepdims=True) > 0)
    values = np.zeros(num_states, dtype=np.float64)
    if iterations <= 0 or not np.isfinite(tolerance) or tolerance < 0 or not 0 <= gamma < 1:
        raise ValueError("Invalid iteration budget, tolerance or discount.")
    for iteration in range(iterations):
        q = r_hat + gamma * np.einsum("san,n->sa", p_hat, values)
        q[counts == 0.0] = np.nan
        updated = np.zeros_like(values)
        for state in range(num_states):
            supported = np.flatnonzero(counts[state] > 0.0)
            if len(supported) and not terminal[state]:
                updated[state] = _expectile(q[state, supported], behavior[state, supported], expectile)
        final_update_norm = float(np.max(np.abs(updated - values)))
        if diagnostics is not None:
            diagnostics.update(iterations_used=iteration + 1, final_update_norm=final_update_norm,
                               converged=final_update_norm <= tolerance)
        if final_update_norm <= tolerance:
            values = updated
            break
        values = updated
    q = r_hat + gamma * np.einsum("san,n->sa", p_hat, values)
    policy = np.full((num_states, num_actions), 1.0 / num_actions, dtype=np.float64)
    for state in range(num_states):
        supported = np.flatnonzero(counts[state] > 0.0)
        if terminal[state] or not len(supported):
            continue
        logits = inverse_temperature * (q[state, supported] - values[state])
        logits -= logits.max()
        weighted = behavior[state, supported] * np.exp(logits)
        policy[state] = 0.0
        policy[state, supported] = weighted / weighted.sum()
    return policy


def empirical_requirement(
    prefixes: Sequence[tuple[int, Sequence[Sequence[TransitionLike]]]], *,
    learner: Callable[[Sequence[TransitionLike]], np.ndarray],
    evaluate: Callable[[np.ndarray], float], success_threshold: float = 0.95,
    required_successes: int = 8,
) -> EmpiricalRequirement:
    """Return first grid size whose suffix has at least required successes."""
    if not prefixes:
        raise ValueError("At least one registered dataset size is required.")
    if required_successes <= 0:
        raise ValueError("required_successes must be positive.")
    if not np.isfinite(success_threshold):
        raise ValueError("success_threshold must be finite.")
    sizes = tuple(int(size) for size, _ in prefixes)
    if any(size <= 0 for size in sizes) or any(
        right <= left for left, right in zip(sizes, sizes[1:])
    ):
        raise ValueError("Registered dataset sizes must be positive and strictly increasing.")
    run_count = len(prefixes[0][1])
    if run_count < required_successes or any(
        len(runs) != run_count for _, runs in prefixes
    ):
        raise ValueError(
            "Every registered size must have the same number of runs, and that "
            "number must be at least required_successes."
        )

    counts: list[int] = []
    for _, runs in prefixes:
        counts.append(sum(evaluate(learner(run)) >= success_threshold for run in runs))
    for index, (size, _) in enumerate(prefixes):
        if all(count >= required_successes for count in counts[index:]):
            return EmpiricalRequirement(
                int(size),
                False,
                tuple(counts),
                tuple(count / run_count for count in counts),
            )
    return EmpiricalRequirement(
        None,
        True,
        tuple(counts),
        tuple(count / run_count for count in counts),
    )


__all__ = [
    "EmpiricalRequirement",
    "behavior_cloning_tabular",
    "cfqi_tabular",
    "empirical_requirement",
    "iql_tabular",
]
