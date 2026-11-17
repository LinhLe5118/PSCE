"""Five pre-registered behavior-policy families for the DAORL benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from exact_solver import value_iteration  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG
from environments.tabular_environments import TabularMDP


_POLICY_CONFIG = PROTOCOL_CONFIG["behavior_policies"]
POLICY_FAMILIES = tuple(_POLICY_CONFIG["families"])
TEMPERATURE_GRID = tuple(float(value) for value in _POLICY_CONFIG["temperature_grid"])
REWARD_NOISE_GRID = tuple(float(value) for value in _POLICY_CONFIG["reward_noise_grid"])
EPSILON_GRID = tuple(float(value) for value in _POLICY_CONFIG["epsilon_grid"])
MIXTURE_WEIGHT_GRID = tuple(float(value) for value in _POLICY_CONFIG["mixture_weight_grid"])
DYNAMICS_SHIFT_GRID = tuple(float(value) for value in _POLICY_CONFIG["dynamics_shift_grid"])
STRUCTURED_SCORE_RANK = int(_POLICY_CONFIG["structured_score_rank"])


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True)
class BehaviorPolicy:
    family: str
    probabilities: np.ndarray
    parameters: dict[str, Any]
    seed: int
    uses_q_star: bool

    def validate(self, mdp: TabularMDP) -> None:
        if self.family not in POLICY_FAMILIES:
            raise ValueError(f"Unknown behavior-policy family: {self.family}")
        if self.probabilities.shape != (mdp.num_states, mdp.num_actions):
            raise ValueError("Policy must have shape (S,A).")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Policy probabilities must be finite.")
        if np.any(self.probabilities < 0.0) or not np.allclose(
            self.probabilities.sum(axis=1), 1.0, atol=1e-12
        ):
            raise ValueError("Every policy row must be a probability distribution.")


def _mdp_view(mdp: TabularMDP, *, P: np.ndarray, R: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        P=P,
        R=R,
        gamma=mdp.gamma,
        initial_state_distribution=mdp.initial_state_distribution,
    )


def _epsilon_soft(
    deterministic_actions: np.ndarray, num_actions: int, epsilon: float
) -> np.ndarray:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1].")
    policy = np.full(
        (len(deterministic_actions), num_actions),
        epsilon / num_actions,
        dtype=np.float64,
    )
    policy[np.arange(len(deterministic_actions)), deterministic_actions] += 1.0 - epsilon
    return policy


def _stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    centered = (scores - scores.max(axis=1, keepdims=True)) / temperature
    exponentials = np.exp(centered)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _structured_scores(
    num_states: int,
    num_actions: int,
    rng: np.random.Generator,
    *,
    rank: int = 3,
) -> np.ndarray:
    """Draw a reproducible low-rank score field rather than IID action labels."""

    if rank <= 0:
        raise ValueError("rank must be positive.")
    state_latent = rng.normal(size=(num_states, rank))
    action_latent = rng.normal(size=(num_actions, rank))
    scores = state_latent @ action_latent.T / np.sqrt(rank)
    scores += 0.05 * rng.normal(size=scores.shape)
    return scores


def q_star_softmax_policy(
    mdp: TabularMDP, *, temperature_normalized: float, seed: int
) -> BehaviorPolicy:
    solution = value_iteration(mdp)
    # Multiplying Q by (1-gamma) places continuing and episodic instances on a
    # comparable reward scale.  The equivalent raw-Q temperature is recorded.
    normalized_q = (1.0 - mdp.gamma) * solution.q_values
    probabilities = _stable_softmax(normalized_q, temperature_normalized)
    policy = BehaviorPolicy(
        family="q_star_softmax",
        probabilities=probabilities,
        parameters={
            "temperature_normalized": temperature_normalized,
            "temperature_raw_q": temperature_normalized / (1.0 - mdp.gamma),
            "q_normalization": "multiply_by_1_minus_gamma",
        },
        seed=seed,
        uses_q_star=True,
    )
    policy.validate(mdp)
    return policy


def perturbed_reward_policy(
    mdp: TabularMDP,
    *,
    noise_scale: float,
    epsilon: float,
    seed: int,
) -> BehaviorPolicy:
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")
    rng = _rng(seed)
    noise = rng.normal(size=(mdp.num_states, mdp.num_actions))
    noise = (noise - noise.mean()) / max(float(noise.std()), 1e-12)
    perturbed_rewards = np.clip(
        mdp.R + noise_scale * noise[:, :, None], 0.0, 1.0
    )
    if mdp.terminal_states:
        perturbed_rewards[np.asarray(mdp.terminal_states)] = 0.0
    solution = value_iteration(_mdp_view(mdp, P=mdp.P, R=perturbed_rewards))
    actions = np.argmax(solution.q_values, axis=1)
    probabilities = _epsilon_soft(actions, mdp.num_actions, epsilon)
    policy = BehaviorPolicy(
        family="perturbed_reward",
        probabilities=probabilities,
        parameters={
            "noise_scale": noise_scale,
            "noise_distribution": "standardized_gaussian_state_action",
            "reward_transform": "clip_to_0_1",
            "epsilon": epsilon,
            "softening": "epsilon_uniform",
        },
        seed=seed,
        uses_q_star=False,
    )
    policy.validate(mdp)
    return policy


def structured_random_policy(
    mdp: TabularMDP, *, epsilon: float, seed: int, rank: int = STRUCTURED_SCORE_RANK
) -> BehaviorPolicy:
    rng = _rng(seed)
    scores = _structured_scores(mdp.num_states, mdp.num_actions, rng, rank=rank)
    actions = np.argmax(scores, axis=1)
    probabilities = _epsilon_soft(actions, mdp.num_actions, epsilon)
    policy = BehaviorPolicy(
        family="structured_random",
        probabilities=probabilities,
        parameters={
            "epsilon": epsilon,
            "score_model": "low_rank_gaussian",
            "score_rank": rank,
            "local_noise_std": 0.05,
        },
        seed=seed,
        uses_q_star=False,
    )
    policy.validate(mdp)
    return policy


def two_policy_mixture(
    mdp: TabularMDP,
    *,
    weight: float,
    seed: int,
    rank: int = STRUCTURED_SCORE_RANK,
) -> BehaviorPolicy:
    if not 0.0 < weight < 1.0:
        raise ValueError("weight must be strictly between zero and one.")
    rng = _rng(seed)
    scores = _structured_scores(mdp.num_states, mdp.num_actions, rng, rank=rank)
    first = np.argmax(scores, axis=1)
    second = np.argmin(scores, axis=1)
    probabilities = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
    rows = np.arange(mdp.num_states)
    probabilities[rows, first] += weight
    probabilities[rows, second] += 1.0 - weight
    disagreement = float(np.mean(first != second))
    policy = BehaviorPolicy(
        family="two_policy_mixture",
        probabilities=probabilities,
        parameters={
            "weight_first": weight,
            "operator_source": "argmax_and_argmin_of_shared_low_rank_score",
            "score_rank": rank,
            "operator_disagreement_fraction": disagreement,
        },
        seed=seed,
        uses_q_star=False,
    )
    policy.validate(mdp)
    return policy


def shifted_dynamics_policy(
    mdp: TabularMDP,
    *,
    shift: float,
    epsilon: float,
    seed: int,
) -> BehaviorPolicy:
    if not 0.0 <= shift <= 1.0:
        raise ValueError("shift must lie in [0,1].")
    rng = _rng(seed)
    action_permutation = rng.permutation(mdp.num_actions)
    if mdp.num_actions > 1 and np.array_equal(
        action_permutation, np.arange(mdp.num_actions)
    ):
        action_permutation = np.roll(action_permutation, 1)
    shifted = (1.0 - shift) * mdp.P + shift * mdp.P[:, action_permutation, :]
    solution = value_iteration(_mdp_view(mdp, P=shifted, R=mdp.R))
    actions = np.argmax(solution.q_values, axis=1)
    probabilities = _epsilon_soft(actions, mdp.num_actions, epsilon)
    policy = BehaviorPolicy(
        family="shifted_dynamics",
        probabilities=probabilities,
        parameters={
            "shift": shift,
            "shift_model": "convex_mix_with_action_permuted_kernel",
            "action_permutation": action_permutation.tolist(),
            "epsilon": epsilon,
            "softening": "epsilon_uniform",
        },
        seed=seed,
        uses_q_star=False,
    )
    policy.validate(mdp)
    return policy


def behavior_policy_suite(
    mdp: TabularMDP, *, seed: int, variant_index: int = 0
) -> list[BehaviorPolicy]:
    """Return one configuration from each family for an MDP.

    Parameter grids are rotated deterministically across base MDPs, avoiding
    parameter selection after outcomes are inspected.
    """

    grid_size = len(TEMPERATURE_GRID)
    if not all(
        len(grid) == grid_size
        for grid in (
            REWARD_NOISE_GRID,
            EPSILON_GRID,
            MIXTURE_WEIGHT_GRID,
            DYNAMICS_SHIFT_GRID,
        )
    ):
        raise AssertionError("Behavior-policy parameter grids must have equal length.")
    index = variant_index % grid_size
    epsilon = EPSILON_GRID[index]
    policies = [
        q_star_softmax_policy(
            mdp,
            temperature_normalized=TEMPERATURE_GRID[index],
            seed=seed + 1,
        ),
        perturbed_reward_policy(
            mdp,
            noise_scale=REWARD_NOISE_GRID[index],
            epsilon=epsilon,
            seed=seed + 2,
        ),
        structured_random_policy(mdp, epsilon=epsilon, seed=seed + 3),
        two_policy_mixture(
            mdp, weight=MIXTURE_WEIGHT_GRID[index], seed=seed + 4
        ),
        shifted_dynamics_policy(
            mdp,
            shift=DYNAMICS_SHIFT_GRID[index],
            epsilon=epsilon,
            seed=seed + 5,
        ),
    ]
    if sum(policy.uses_q_star for policy in policies) != 1:
        raise AssertionError("Exactly one behavior family may use Q-star.")
    return policies


__all__ = [
    "BehaviorPolicy",
    "DYNAMICS_SHIFT_GRID",
    "EPSILON_GRID",
    "MIXTURE_WEIGHT_GRID",
    "POLICY_FAMILIES",
    "REWARD_NOISE_GRID",
    "TEMPERATURE_GRID",
    "behavior_policy_suite",
    "perturbed_reward_policy",
    "q_star_softmax_policy",
    "shifted_dynamics_policy",
    "structured_random_policy",
    "two_policy_mixture",
]
