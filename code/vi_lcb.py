"""Paper-spec VI-LCB for discounted tabular offline RL.

This implements Algorithm 1 and equation (28) from Li et al. (2024). The
authors' public repository contains the finite-horizon Gambler experiment, so
this discounted implementation is a transparent local port rather than an
unmodified import of author code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class TransitionLike(Protocol):
    state: int
    action: int
    reward: float
    next_state: int


@dataclass(frozen=True)
class VILCBResult:
    policy: np.ndarray
    q_values: np.ndarray
    counts: np.ndarray
    empirical_transitions: np.ndarray
    empirical_rewards: np.ndarray


def vi_lcb_discounted(
    transitions: Sequence[TransitionLike],
    *,
    num_states: int,
    num_actions: int,
    gamma: float,
    cb: float,
    delta: float = 0.05,
    iterations: int = 100,
    terminal_states: Sequence[int] = (),
) -> VILCBResult:
    """Fit discounted VI-LCB with the Bernstein penalty from equation (28).

    Rewards are estimated by the empirical mean because the project protocol
    does not expose the true reward function to learners. Unseen transition
    rows use the paper's uniform convention. Known terminal semantics are used
    to make terminal states absorbing with zero reward.
    """

    if num_states <= 0 or num_actions <= 0:
        raise ValueError("state and action counts must be positive")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if cb < 0.0 or not 0.0 < delta < 1.0 or iterations <= 0:
        raise ValueError("cb, delta, or iterations is invalid")
    if not transitions:
        raise ValueError("VI-LCB requires at least one transition")

    counts = np.zeros((num_states, num_actions), dtype=np.int64)
    transition_counts = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    reward_sums = np.zeros((num_states, num_actions), dtype=np.float64)
    for item in transitions:
        state, action, next_state = int(item.state), int(item.action), int(item.next_state)
        if not (0 <= state < num_states and 0 <= next_state < num_states):
            raise ValueError("transition state is outside the declared state space")
        if not 0 <= action < num_actions:
            raise ValueError("transition action is outside the declared action space")
        reward = float(item.reward)
        if not np.isfinite(reward):
            raise ValueError("transition reward must be finite")
        counts[state, action] += 1
        transition_counts[state, action, next_state] += 1.0
        reward_sums[state, action] += reward

    empirical = np.full(
        (num_states, num_actions, num_states), 1.0 / num_states, dtype=np.float64
    )
    observed = counts > 0
    empirical[observed] = transition_counts[observed] / counts[observed, None]
    rewards = np.divide(
        reward_sums, counts, out=np.zeros_like(reward_sums), where=observed
    )
    terminals = tuple(sorted(set(int(state) for state in terminal_states)))
    if any(state < 0 or state >= num_states for state in terminals):
        raise ValueError("terminal state is outside the declared state space")
    for state in terminals:
        empirical[state] = 0.0
        empirical[state, :, state] = 1.0
        rewards[state] = 0.0

    sample_size = len(transitions)
    log_term = float(np.log(sample_size / ((1.0 - gamma) * delta)))
    if log_term <= 0.0:
        raise ValueError("equation (28) logarithm must be positive")
    horizon = 1.0 / (1.0 - gamma)
    q_values = np.zeros((num_states, num_actions), dtype=np.float64)
    for _ in range(iterations):
        values = np.max(q_values, axis=1)
        mean = np.einsum("san,n->sa", empirical, values)
        second = np.einsum("san,n->sa", empirical, values * values)
        variance = np.maximum(second - mean * mean, 0.0)
        denominator = counts.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance_term = np.sqrt(cb * log_term * variance / denominator)
            count_term = 2.0 * cb * log_term / ((1.0 - gamma) * denominator)
        inner = np.maximum(variance_term, count_term)
        inner[~observed] = np.inf
        penalty = np.minimum(inner, horizon) + 5.0 / sample_size
        q_values = np.maximum(rewards + gamma * mean - penalty, 0.0)
        if terminals:
            q_values[list(terminals)] = 0.0

    actions = np.argmax(q_values, axis=1)
    policy = np.zeros_like(q_values)
    policy[np.arange(num_states), actions] = 1.0
    return VILCBResult(policy, q_values, counts, empirical, rewards)


__all__ = ["VILCBResult", "vi_lcb_discounted"]
