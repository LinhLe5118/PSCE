"""Data-only Offline Effective Horizon estimator.

The estimator consumes trajectory-labelled transitions plus public state-space
dimensions and the initial-state distribution.  It never accepts an MDP,
manifest, structural-family label, or exact P/R/V*/Q* values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402


FloatArray = NDArray[np.float64]
DEPTH_TIE_ATOL = 1e-12


class TransitionLike(Protocol):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: bool
    truncated: bool
    trajectory_id: int
    timestep: int


@dataclass(frozen=True)
class OEHConfig:
    gamma: float
    tau_coefficient: float
    n_min: int
    k_max: int
    num_bootstrap: int
    num_exact_replicates: int
    n0_coefficient: int
    epsilon: float
    horizon: int
    min_valid_bootstrap_fraction: float
    bootstrap_scheme: str = "bayesian_trajectory_dirichlet"
    bootstrap_dirichlet_alpha: float = 1.0
    min_trajectory_support: int = 2

    @classmethod
    def load(cls) -> "OEHConfig":
        """Load the estimator section from the canonical protocol config."""

        raw = PROTOCOL_CONFIG["estimator"]
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class OEHResult:
    """Estimator output, including explicit abstention/censoring states."""

    abstained: bool
    reason: str | None
    censored_right: bool
    num_states: int
    num_actions: int
    tau: float
    counts: NDArray[np.int64]
    trajectory_counts: NDArray[np.int64]
    supported_actions: tuple[tuple[int, ...], ...]
    deltas: FloatArray | None = None
    sigmas: FloatArray | None = None
    z_scores: FloatArray | None = None
    losses_at_one: FloatArray | None = None
    multipliers: FloatArray | None = None
    gap_occupancy: FloatArray | None = None
    occupancy_weights: FloatArray | None = None
    value_range: FloatArray | None = None
    valid_bootstrap_fraction: float | None = None
    k_dagger: int | None = None
    effective_horizon: float | None = None
    n_req_hat: float | None = None


def _supported_actions(
    counts: NDArray[np.int64],
    trajectory_counts: NDArray[np.int64],
    config: OEHConfig,
) -> tuple[tuple[int, ...], ...]:
    support_mask = (counts >= config.n_min) & (
        trajectory_counts >= config.min_trajectory_support
    )
    return tuple(tuple(np.flatnonzero(row)) for row in support_mask)


def _trajectory_counts(
    transitions: Iterable[TransitionLike],
    states: int,
    actions: int,
) -> NDArray[np.int64]:
    counts = np.zeros((states, actions), dtype=np.int64)
    observed = {
        (int(item.trajectory_id), int(item.state), int(item.action))
        for item in transitions
    }
    for _, state, action in observed:
        counts[state, action] += 1
    return counts


def _terminal_state_mask(
    transitions: Sequence[TransitionLike],
    states: int,
) -> NDArray[np.bool_]:
    """Infer episodic terminal destinations and validate their data semantics."""

    terminal_states = {
        int(item.next_state) for item in transitions if bool(item.terminal)
    }
    for item in transitions:
        state = int(item.state)
        next_state = int(item.next_state)
        terminal = bool(item.terminal)
        truncated = bool(item.truncated)
        if terminal and truncated:
            raise ValueError("A transition cannot be terminal and truncated.")
        if state in terminal_states:
            raise ValueError("Natural trajectories cannot act from a terminal state.")
        if (next_state in terminal_states) != terminal:
            raise ValueError(
                "terminal flags must be consistent for each destination state."
            )
    mask = np.zeros(states, dtype=np.bool_)
    if terminal_states:
        mask[list(terminal_states)] = True
    return mask


def _smallest_near_minimum(values: FloatArray) -> int:
    """Return the first minimizer under the registered absolute tolerance."""

    minimum = float(np.nanmin(values))
    if np.isinf(minimum):
        tied = np.flatnonzero(values == minimum)
    else:
        tied = np.flatnonzero(
            np.isclose(values, minimum, rtol=0.0, atol=DEPTH_TIE_ATOL)
        )
    if len(tied) == 0:
        raise ValueError("The evidence curve has no defined minimum.")
    return int(tied[0])


def _abstain(
    reason: str,
    counts: NDArray[np.int64],
    trajectory_counts: NDArray[np.int64],
    config: OEHConfig,
    *,
    valid_bootstrap_fraction: float | None = None,
) -> OEHResult:
    states, actions = counts.shape
    return OEHResult(
        abstained=True,
        reason=reason,
        censored_right=False,
        num_states=states,
        num_actions=actions,
        tau=config.tau_coefficient / states,
        counts=counts,
        trajectory_counts=trajectory_counts,
        supported_actions=_supported_actions(counts, trajectory_counts, config),
        valid_bootstrap_fraction=valid_bootstrap_fraction,
    )


def _observed_dimensions(
    transitions: Sequence[TransitionLike],
) -> tuple[int, int]:
    if not transitions:
        raise ValueError("D0 must contain at least one transition.")
    max_state = max(max(int(item.state), int(item.next_state)) for item in transitions)
    max_action = max(int(item.action) for item in transitions)
    if min(
        min(int(item.state), int(item.next_state), int(item.action))
        for item in transitions
    ) < 0:
        raise ValueError("State and action indices must be non-negative.")
    return max_state + 1, max_action + 1


def _validate_dimensions(
    transitions: Sequence[TransitionLike],
    num_states: int | None,
    num_actions: int | None,
) -> tuple[int, int]:
    observed_states, observed_actions = _observed_dimensions(transitions)
    states = observed_states if num_states is None else num_states
    actions = observed_actions if num_actions is None else num_actions
    if (
        isinstance(states, bool)
        or not isinstance(states, int)
        or states < observed_states
        or isinstance(actions, bool)
        or not isinstance(actions, int)
        or actions < observed_actions
        or actions < 2
    ):
        raise ValueError(
            "num_states must cover every observed state and num_actions must "
            "cover every observed action and be at least 2."
        )
    return states, actions


def _model(
    transitions: Iterable[TransitionLike],
    states: int,
    actions: int,
    supported: tuple[tuple[int, ...], ...],
    trajectory_weights: Mapping[int, float] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None:
    counts = np.zeros((states, actions), dtype=np.float64)
    rewards = np.zeros((states, actions), dtype=np.float64)
    next_counts = np.zeros((states, actions, states), dtype=np.float64)
    for item in transitions:
        state = int(item.state)
        action = int(item.action)
        next_state = int(item.next_state)
        reward = float(item.reward)
        if not np.isfinite(reward):
            raise ValueError("Transition rewards must be finite.")
        weight = (
            1.0
            if trajectory_weights is None
            else float(trajectory_weights[int(item.trajectory_id)])
        )
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("Every trajectory weight must be finite and positive.")
        counts[state, action] += weight
        rewards[state, action] += weight * reward
        next_counts[state, action, next_state] += weight
    if any(
        counts[state, action] == 0
        for state, row in enumerate(supported)
        for action in row
    ):
        return None
    p_hat = np.zeros_like(next_counts)
    r_hat = np.zeros((states, actions), dtype=np.float64)
    beta = np.zeros((states, actions), dtype=np.float64)
    for state, row in enumerate(supported):
        if not row:
            continue
        row_counts = counts[state, list(row)]
        beta[state, list(row)] = row_counts / row_counts.sum()
        for action in row:
            p_hat[state, action] = next_counts[state, action] / counts[state, action]
            r_hat[state, action] = rewards[state, action] / counts[state, action]
    return p_hat, r_hat, beta, counts


def _q_sequence(
    p_hat: FloatArray,
    r_hat: FloatArray,
    beta: FloatArray,
    supported: tuple[tuple[int, ...], ...],
    gamma: float,
    k_max: int,
) -> list[FloatArray] | None:
    import os
    backend = os.environ.get('DAORL_OEH_BACKEND', 'reference')
    if backend != 'reference':
        from tabular_acceleration import q_sequence
        return q_sequence(p_hat, r_hat, beta, supported, gamma, k_max, backend=backend)
    states, actions = beta.shape
    p_beta = np.einsum("sa,san->sn", beta, p_hat)
    r_beta = np.einsum("sa,sa->s", beta, r_hat)
    try:
        v_beta = np.linalg.solve(np.eye(states) - gamma * p_beta, r_beta)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(v_beta)):
        return None
    q_values = np.full((states, actions), np.nan, dtype=np.float64)
    for state, row in enumerate(supported):
        for action in row:
            q_values[state, action] = (
                r_hat[state, action] + gamma * p_hat[state, action] @ v_beta
            )
    if any(
        not np.isfinite(q_values[state, action])
        for state, row in enumerate(supported)
        for action in row
    ):
        return None
    sequence = [q_values]
    for _ in range(k_max):
        previous = sequence[-1]
        maxima = np.zeros(states, dtype=np.float64)
        for state, row in enumerate(supported):
            if row:
                maxima[state] = np.max(previous[state, list(row)])
        updated = np.full((states, actions), np.nan, dtype=np.float64)
        for state, row in enumerate(supported):
            for action in row:
                updated[state, action] = (
                    r_hat[state, action]
                    + gamma * p_hat[state, action] @ maxima
                )
        if any(
            not np.isfinite(updated[state, action])
            for state, row in enumerate(supported)
            for action in row
        ):
            return None
        sequence.append(updated)
    return sequence


def _occupancy(
    p_hat: FloatArray,
    q_values: FloatArray,
    supported: tuple[tuple[int, ...], ...],
    initial: FloatArray,
    gamma: float,
    terminal_mask: NDArray[np.bool_],
) -> FloatArray | None:
    states = len(supported)
    p_policy = np.zeros((states, states), dtype=np.float64)
    for state, row in enumerate(supported):
        if terminal_mask[state]:
            continue
        if row:
            action = row[int(np.nanargmax(q_values[state, list(row)]))]
            policy_row = p_hat[state, action].copy()
            terminal_mass = float(policy_row[terminal_mask].sum())
            policy_row[terminal_mask] = 0.0
            policy_row += terminal_mass * initial
            p_policy[state] = policy_row
        else:
            p_policy[state, state] = 1.0
    try:
        occupancy = (1.0 - gamma) * np.linalg.solve(
            (np.eye(states) - gamma * p_policy).T,
            initial,
        )
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(occupancy)) or np.any(occupancy < -1e-12):
        return None
    return np.maximum(occupancy, 0.0)


def _loss_and_multiplier(
    deltas: FloatArray,
    sigmas: FloatArray,
    weights: FloatArray,
    epsilon: float,
) -> tuple[float, float, FloatArray]:
    z_scores = np.zeros_like(deltas)
    positive_sigma = sigmas > 0.0
    z_scores[positive_sigma] = deltas[positive_sigma] / sigmas[positive_sigma]
    z_scores[(~positive_sigma) & (deltas > 0.0)] = np.inf

    def loss(multiplier: float) -> float:
        return float(
            np.sum(
                weights
                * deltas
                * norm.cdf(-z_scores * np.sqrt(multiplier))
            )
        )

    loss_at_one = loss(1.0)
    if loss_at_one <= epsilon:
        return loss_at_one, 1.0, z_scores
    upper = 2.0
    while loss(upper) > epsilon:
        upper *= 2.0
        if not np.isfinite(upper):
            return loss_at_one, np.inf, z_scores
    multiplier = brentq(lambda value: loss(value) - epsilon, 1.0, upper)
    return loss_at_one, float(multiplier), z_scores


def estimate_oeh(
    transitions: Sequence[TransitionLike],
    initial_state_distribution: Sequence[float],
    *,
    num_states: int | None = None,
    num_actions: int | None = None,
    config: OEHConfig | None = None,
    rng: np.random.Generator | None = None,
) -> OEHResult:
    """Estimate OEH from a data-only pilot prefix."""

    cfg = config or OEHConfig.load()
    if cfg.bootstrap_scheme != "bayesian_trajectory_dirichlet":
        raise ValueError("Unsupported bootstrap_scheme.")
    if cfg.bootstrap_dirichlet_alpha != 1.0:
        raise ValueError(
            "bootstrap_dirichlet_alpha must be 1.0 for the registered scheme."
        )
    if (
        isinstance(cfg.min_trajectory_support, bool)
        or not isinstance(cfg.min_trajectory_support, int)
        or cfg.min_trajectory_support < 2
    ):
        raise ValueError("min_trajectory_support must be an integer >= 2.")
    states, actions = _validate_dimensions(transitions, num_states, num_actions)
    terminal_mask = _terminal_state_mask(transitions, states)
    initial = np.asarray(initial_state_distribution, dtype=np.float64)
    if (
        initial.shape != (states,)
        or np.any(initial < 0.0)
        or not np.isclose(initial.sum(), 1.0)
    ):
        raise ValueError(
            "initial_state_distribution must be a probability vector over S."
        )
    if np.any(initial[terminal_mask] > 0.0):
        raise ValueError(
            "initial_state_distribution cannot place mass on terminal states."
        )
    trajectory_ids = sorted({int(item.trajectory_id) for item in transitions})
    counts = np.zeros((states, actions), dtype=np.int64)
    for item in transitions:
        counts[int(item.state), int(item.action)] += 1
    trajectory_counts = _trajectory_counts(transitions, states, actions)
    if len(trajectory_ids) < 2:
        return _abstain(
            "fewer_than_two_trajectories", counts, trajectory_counts, cfg
        )
    supported = _supported_actions(counts, trajectory_counts, cfg)
    model = _model(transitions, states, actions, supported)
    if model is None:
        return _abstain(
            "support_model_undefined", counts, trajectory_counts, cfg
        )
    p_hat, r_hat, beta, _ = model
    q_sequence = _q_sequence(
        p_hat, r_hat, beta, supported, cfg.gamma, cfg.k_max
    )
    if q_sequence is None:
        return _abstain(
            "singular_behavior_evaluation", counts, trajectory_counts, cfg
        )

    top_pairs: list[list[tuple[int, int] | None]] = []
    for q_values in q_sequence:
        pairs: list[tuple[int, int] | None] = []
        for state, row in enumerate(supported):
            if len(row) < 2:
                pairs.append(None)
            else:
                order = sorted(
                    row,
                    key=lambda action: (
                        -float(q_values[state, action]),
                        action,
                    ),
                )
                pairs.append((order[0], order[1]))
        top_pairs.append(pairs)

    generator = rng if rng is not None else np.random.default_rng()
    replicate_gaps: list[list[FloatArray]] = []
    for _ in range(cfg.num_bootstrap):
        weights = generator.dirichlet(
            np.full(len(trajectory_ids), cfg.bootstrap_dirichlet_alpha)
        )
        trajectory_weights = dict(zip(trajectory_ids, weights, strict=True))
        replicate_model = _model(
            transitions,
            states,
            actions,
            supported,
            trajectory_weights=trajectory_weights,
        )
        if replicate_model is None:
            continue
        replicate_p, replicate_r, replicate_beta, _ = replicate_model
        sequence = _q_sequence(
            replicate_p,
            replicate_r,
            replicate_beta,
            supported,
            cfg.gamma,
            cfg.k_max,
        )
        if sequence is None:
            continue
        replicate_gaps.append(
            [
                np.array(
                    [
                        np.nan
                        if pair is None
                        else q_values[state, pair[0]]
                        - q_values[state, pair[1]]
                        for state, pair in enumerate(pairs)
                    ]
                )
                for q_values, pairs in zip(sequence, top_pairs)
            ]
        )
    valid_fraction = len(replicate_gaps) / cfg.num_bootstrap
    if valid_fraction < cfg.min_valid_bootstrap_fraction:
        return _abstain(
            "insufficient_numerically_valid_bootstrap_replicates",
            counts,
            trajectory_counts,
            cfg,
            valid_bootstrap_fraction=valid_fraction,
        )

    deltas = np.full((cfg.k_max + 1, states), np.nan)
    sigmas = np.full_like(deltas, np.nan)
    z_scores = np.full_like(deltas, np.nan)
    losses = np.full(cfg.k_max + 1, np.nan)
    multipliers = np.full(cfg.k_max + 1, np.nan)
    gap_occupancy = np.zeros(cfg.k_max + 1)
    # w_k(s) is reported, not just consumed: without it the loss L_k = sum_s
    # w_k(s) Delta_k(s) Phi(-z_k(s)) cannot be decomposed after the fact, and a
    # diagnostic that rebuilt the weights itself would be checking its own
    # arithmetic rather than the estimator's.
    occupancy_weights = np.zeros((cfg.k_max + 1, states))
    # The spread of Q_hat over supported pairs is the only data-only value
    # scale available to a caller: r_max/(1-gamma) overstates it by orders of
    # magnitude under sparse rewards, and the benchmark's dynamic_range is
    # derived from V* and may never enter a data-only quantity.
    value_range = np.zeros(cfg.k_max + 1)
    for depth, (q_values, pairs) in enumerate(zip(q_sequence, top_pairs)):
        if np.any(np.isfinite(q_values)):
            value_range[depth] = float(
                np.nanmax(q_values) - np.nanmin(q_values)
            )
        occupancy = _occupancy(
            p_hat,
            q_values,
            supported,
            initial,
            cfg.gamma,
            terminal_mask,
        )
        if occupancy is None:
            return _abstain(
                "singular_or_nonfinite_occupancy",
                counts,
                trajectory_counts,
                cfg,
                valid_bootstrap_fraction=valid_fraction,
            )
        critical = (
            occupancy >= cfg.tau_coefficient / states
        ) & ~terminal_mask
        if any(critical[state] and not supported[state] for state in range(states)):
            return _abstain(
                "empty_supported_action_set_in_critical_state",
                counts,
                trajectory_counts,
                cfg,
                valid_bootstrap_fraction=valid_fraction,
            )
        eligible = np.array(
            [critical[state] and pairs[state] is not None for state in range(states)]
        )
        critical_mass = occupancy[critical].sum()
        weights = np.zeros(states)
        if critical_mass > 0.0:
            weights[eligible] = occupancy[eligible] / critical_mass
            gap_occupancy[depth] = float(
                occupancy[critical & ~eligible].sum() / critical_mass
            )
        occupancy_weights[depth] = weights
        for state, pair in enumerate(pairs):
            if pair is not None:
                deltas[depth, state] = (
                    q_values[state, pair[0]] - q_values[state, pair[1]]
                )
        values = np.asarray([replicate[depth] for replicate in replicate_gaps])
        for state in range(states):
            finite_values = values[np.isfinite(values[:, state]), state]
            if len(finite_values) >= 2:
                sigmas[depth, state] = float(np.std(finite_values, ddof=1))
        loss, multiplier, depth_z = _loss_and_multiplier(
            np.nan_to_num(deltas[depth]),
            np.nan_to_num(sigmas[depth]),
            weights,
            cfg.epsilon,
        )
        losses[depth] = loss
        multipliers[depth] = multiplier
        z_scores[depth] = depth_z

    k_dagger = _smallest_near_minimum(multipliers)
    censored_right = bool(gap_occupancy[k_dagger] > cfg.epsilon)
    minimum = float(multipliers[k_dagger])
    effective_horizon = float(
        min(
            depth
            + np.log(max(1.0, float(multipliers[depth]))) / np.log(actions)
            for depth in range(cfg.k_max + 1)
        )
    )
    return OEHResult(
        abstained=False,
        reason=None,
        censored_right=censored_right,
        num_states=states,
        num_actions=actions,
        tau=cfg.tau_coefficient / states,
        counts=counts,
        trajectory_counts=trajectory_counts,
        supported_actions=supported,
        deltas=deltas,
        sigmas=sigmas,
        z_scores=z_scores,
        losses_at_one=losses,
        multipliers=multipliers,
        gap_occupancy=gap_occupancy,
        occupancy_weights=occupancy_weights,
        value_range=value_range,
        valid_bootstrap_fraction=valid_fraction,
        k_dagger=k_dagger,
        effective_horizon=effective_horizon,
        n_req_hat=len(transitions) * minimum,
    )


__all__ = [
    "OEHConfig",
    "OEHResult",
    "TransitionLike",
    "estimate_oeh",
]
