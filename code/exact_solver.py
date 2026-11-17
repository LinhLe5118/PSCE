"""Exact utilities for finite discounted tabular MDPs.

This module is oracle infrastructure: it may be used to generate benchmark
labels and to score estimators, but it must not be imported by the data-only
OEH estimator.  The implementation only depends on NumPy.

Accepted MDP objects expose ``P`` with shape ``(S, A, S)``, ``R`` or ``r``
with shape ``(S, A)`` or ``(S, A, S)``, ``gamma``, and either ``rho0`` or
``initial_state_distribution``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class MDPValidationError(ValueError):
    """Raised when an object does not satisfy the finite-MDP interface."""


class DegenerateMDPError(ValueError):
    """Raised when normalized return is undefined for an MDP."""


@dataclass(frozen=True)
class ExactSolution:
    """Result of exact value iteration."""

    values: FloatArray
    q_values: FloatArray
    optimal_action_mask: BoolArray
    policy: FloatArray
    bellman_residual: float
    iterations: int


@dataclass(frozen=True)
class PolicyEvaluation:
    """Exact evaluation of a deterministic or stochastic policy."""

    values: FloatArray
    q_values: FloatArray
    expected_return: float
    bellman_residual: float


@dataclass(frozen=True)
class KStepBackupResult:
    """Exact supported backup path from ``Q^(0)`` through ``Q^(K)``.

    The leading dimension of every sequence field is the depth, so index
    ``k`` contains the quantity associated with ``Q^(k)``.
    """

    q_values: FloatArray
    greedy_action_masks: BoolArray
    greedy_policies: FloatArray
    action_gaps: FloatArray
    restricted_behavior_policy: FloatArray

    @property
    def k_max(self) -> int:
        return int(self.q_values.shape[0] - 1)


@dataclass(frozen=True)
class _Model:
    transitions: FloatArray
    rewards: FloatArray
    expected_rewards: FloatArray
    gamma: float
    initial_distribution: FloatArray

    @property
    def num_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def num_actions(self) -> int:
        return int(self.transitions.shape[1])


def _get_attribute(mdp: Any, *names: str) -> Any:
    for name in names:
        if hasattr(mdp, name):
            return getattr(mdp, name)
    joined = " hoặc ".join(names)
    raise MDPValidationError(f"MDP thiếu thuộc tính bắt buộc: {joined}.")


def _as_model(mdp: Any, *, probability_tolerance: float = 1e-10) -> _Model:
    """Validate an MDP-like object and return normalized NumPy views."""

    transitions = np.asarray(_get_attribute(mdp, "P"), dtype=np.float64)
    rewards = np.asarray(_get_attribute(mdp, "R", "r"), dtype=np.float64)
    initial = np.asarray(
        _get_attribute(mdp, "rho0", "initial_state_distribution"),
        dtype=np.float64,
    )
    gamma = float(_get_attribute(mdp, "gamma"))

    if transitions.ndim != 3:
        raise MDPValidationError("P phải có shape (S, A, S).")
    num_states, num_actions, num_next_states = transitions.shape
    if num_states == 0 or num_actions == 0 or num_states != num_next_states:
        raise MDPValidationError("P phải có shape (S, A, S) với S,A > 0.")
    if rewards.shape not in {
        (num_states, num_actions),
        (num_states, num_actions, num_states),
    }:
        raise MDPValidationError("Reward phải có shape (S,A) hoặc (S,A,S).")
    if initial.shape != (num_states,):
        raise MDPValidationError("rho0 phải có shape (S,).")
    if not 0.0 <= gamma < 1.0:
        raise MDPValidationError("gamma phải thuộc [0,1).")
    if not np.all(np.isfinite(transitions)) or not np.all(np.isfinite(rewards)):
        raise MDPValidationError("P và reward không được chứa NaN hoặc infinity.")
    if not np.all(np.isfinite(initial)):
        raise MDPValidationError("rho0 không được chứa NaN hoặc infinity.")
    if np.any(transitions < -probability_tolerance):
        raise MDPValidationError("Xác suất chuyển trạng thái không được âm.")
    if not np.allclose(
        transitions.sum(axis=2), 1.0, rtol=0.0, atol=probability_tolerance
    ):
        raise MDPValidationError("Mỗi P[s,a,:] phải có tổng bằng 1.")
    if np.any(initial < -probability_tolerance) or not np.isclose(
        initial.sum(), 1.0, rtol=0.0, atol=probability_tolerance
    ):
        raise MDPValidationError("rho0 phải là một phân phối xác suất.")

    expected_rewards = (
        rewards
        if rewards.ndim == 2
        else np.sum(transitions * rewards, axis=2)
    )
    return _Model(
        transitions=transitions,
        rewards=rewards,
        expected_rewards=np.asarray(expected_rewards, dtype=np.float64),
        gamma=gamma,
        initial_distribution=initial,
    )


def validate_mdp(mdp: Any) -> None:
    """Raise :class:`MDPValidationError` when ``mdp`` is malformed."""

    _as_model(mdp)


def _q_from_values(model: _Model, values: FloatArray) -> FloatArray:
    # E[R + gamma V] = E[R] + gamma E[V]. Expected rewards are
    # already cached during validation; avoid two (S,A,S) temporaries per
    # Bellman update on larger tabular environments.
    continuation = np.einsum("san,n->sa", model.transitions, values)
    return model.expected_rewards + model.gamma * continuation


def q_values_from_values(mdp: Any, values: NDArray[np.floating[Any]]) -> FloatArray:
    """Apply one exact Bellman lookahead to a supplied state-value vector."""

    model = _as_model(mdp)
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (model.num_states,):
        raise ValueError("values phải có shape (S,).")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values không được chứa NaN hoặc infinity.")
    return _q_from_values(model, values_array)


def _supported_action_mask(action_mask: Any, shape: tuple[int, int]) -> BoolArray:
    if action_mask is None:
        return np.ones(shape, dtype=np.bool_)
    supported = np.asarray(action_mask, dtype=np.bool_)
    if supported.shape != shape:
        raise ValueError("action_mask phải có shape (S,A).")
    if np.any(supported.sum(axis=1) == 0):
        raise ValueError("Mỗi trạng thái phải có ít nhất một hành động được hỗ trợ.")
    return supported


def greedy_policy(
    q_values: NDArray[np.floating[Any]],
    *,
    action_mask: Any = None,
    tie_tolerance: float = 1e-10,
) -> tuple[BoolArray, FloatArray]:
    """Return all supported greedy actions and a uniform policy over ties."""

    q_array = np.asarray(q_values, dtype=np.float64)
    if q_array.ndim != 2 or q_array.shape[1] == 0:
        raise ValueError("q_values phải có shape (S,A) với A > 0.")
    if tie_tolerance < 0.0:
        raise ValueError("tie_tolerance không được âm.")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q_values không được chứa NaN hoặc infinity.")

    supported = _supported_action_mask(action_mask, q_array.shape)
    supported_q = np.where(supported, q_array, -np.inf)
    row_maxima = supported_q.max(axis=1, keepdims=True)
    mask = supported & np.isclose(
        q_array, row_maxima, rtol=0.0, atol=tie_tolerance
    )
    policy = mask.astype(np.float64) / mask.sum(axis=1, keepdims=True)
    return mask, policy


def action_gaps_from_q_values(
    q_values: NDArray[np.floating[Any]], *, action_mask: Any = None
) -> FloatArray:
    """Return the best-minus-second-best supported action gap per state.

    A state with only one supported action has no second-best action and is
    reported as ``NaN``. Tied best actions correctly produce a gap of zero.
    """

    q_array = np.asarray(q_values, dtype=np.float64)
    if q_array.ndim != 2 or q_array.shape[1] == 0:
        raise ValueError("q_values phải có shape (S,A) với A > 0.")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q_values không được chứa NaN hoặc infinity.")
    supported = _supported_action_mask(action_mask, q_array.shape)

    gaps = np.full(q_array.shape[0], np.nan, dtype=np.float64)
    for state in range(q_array.shape[0]):
        supported_values = q_array[state, supported[state]]
        if supported_values.size >= 2:
            top_two = np.partition(supported_values, -2)[-2:]
            gaps[state] = float(top_two.max() - top_two.min())
    return gaps


def value_iteration(
    mdp: Any,
    *,
    action_mask: Any = None,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
    tie_tolerance: float = 1e-10,
) -> ExactSolution:
    """Compute ``V*``, ``Q*`` and the complete supported optimal-action set.

    ``tie_tolerance`` is only a floating-point equality tolerance.  It is not
    the action-gap tolerance used by the OEH definition.

    ``action_mask`` is optional.  K4 can pass the population/support mask here
    so the exact reference and pilot estimator optimize over identical actions.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance phải dương.")
    if max_iterations <= 0:
        raise ValueError("max_iterations phải dương.")

    model = _as_model(mdp)
    supported = _supported_action_mask(
        action_mask, (model.num_states, model.num_actions)
    )
    values = np.zeros(model.num_states, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        q_values = _q_from_values(model, values)
        updated = np.where(supported, q_values, -np.inf).max(axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta <= tolerance:
            break
    else:
        raise RuntimeError(
            f"Value iteration không hội tụ sau {max_iterations} vòng; "
            f"gamma={model.gamma}."
        )

    q_values = _q_from_values(model, values)
    supported_maximum = np.where(supported, q_values, -np.inf).max(axis=1)
    bellman_residual = float(np.max(np.abs(supported_maximum - values)))
    optimal_action_mask, policy = greedy_policy(
        q_values, action_mask=supported, tie_tolerance=tie_tolerance
    )
    return ExactSolution(
        values=values,
        q_values=q_values,
        optimal_action_mask=optimal_action_mask,
        policy=policy,
        bellman_residual=bellman_residual,
        iterations=iterations,
    )


def _policy_matrix(policy: Any, model: _Model) -> FloatArray:
    policy_array = np.asarray(policy)
    if policy_array.shape == (model.num_states,):
        if not np.issubdtype(policy_array.dtype, np.integer):
            raise ValueError("Policy deterministic phải chứa chỉ số hành động nguyên.")
        if np.any(policy_array < 0) or np.any(policy_array >= model.num_actions):
            raise ValueError("Policy chứa hành động nằm ngoài [0,A).")
        matrix = np.zeros((model.num_states, model.num_actions), dtype=np.float64)
        matrix[np.arange(model.num_states), policy_array.astype(np.int64)] = 1.0
        return matrix

    matrix = np.asarray(policy, dtype=np.float64)
    if matrix.shape != (model.num_states, model.num_actions):
        raise ValueError("Policy phải có shape (S,) hoặc (S,A).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Policy không được chứa NaN hoặc infinity.")
    if np.any(matrix < -1e-12) or not np.allclose(
        matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("Mỗi hàng của policy phải là một phân phối xác suất.")
    return matrix


def _restrict_policy_to_actions(
    policy: FloatArray, supported: BoolArray
) -> FloatArray:
    restricted = policy * supported
    supported_mass = restricted.sum(axis=1, keepdims=True)
    if np.any(supported_mass <= 0.0):
        missing_states = np.flatnonzero(supported_mass[:, 0] <= 0.0).tolist()
        raise ValueError(
            "Behavior policy không có xác suất trên action được hỗ trợ tại "
            f"các state {missing_states}."
        )
    return restricted / supported_mass


def evaluate_policy(mdp: Any, policy: Any) -> PolicyEvaluation:
    """Evaluate a policy exactly by solving ``(I - gamma P_pi)V = r_pi``."""

    model = _as_model(mdp)
    policy_matrix = _policy_matrix(policy, model)
    policy_transitions = np.einsum(
        "sa,san->sn", policy_matrix, model.transitions
    )
    policy_rewards = np.sum(policy_matrix * model.expected_rewards, axis=1)
    system = np.eye(model.num_states) - model.gamma * policy_transitions
    try:
        values = np.linalg.solve(system, policy_rewards)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Không giải được hệ tuyến tính đánh giá policy.") from error

    q_values = _q_from_values(model, values)
    bellman_update = np.sum(policy_matrix * q_values, axis=1)
    bellman_residual = float(np.max(np.abs(bellman_update - values)))
    expected_return = float(model.initial_distribution @ values)
    return PolicyEvaluation(
        values=values,
        q_values=q_values,
        expected_return=expected_return,
        bellman_residual=bellman_residual,
    )


def k_step_backups(
    mdp: Any,
    behavior_policy: Any,
    k_max: int,
    *,
    action_mask: Any = None,
    tie_tolerance: float = 1e-10,
) -> KStepBackupResult:
    """Compute the exact supported sequence ``Q^(0), ..., Q^(k_max)``.

    ``Q^(0)`` is the action-value function of the behavior policy after it is
    renormalized over supported actions. Each later element follows

    ``Q^(k+1)(s,a) = r(s,a) + gamma * P(s,a) @ max_supported Q^(k)``.

    This is the exact/population reference used to score the data-only OEH
    estimator. It does not perform bootstrapping or estimate a model from an
    offline dataset.
    """

    if isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)):
        raise TypeError("k_max phải là số nguyên không âm.")
    if k_max < 0:
        raise ValueError("k_max phải là số nguyên không âm.")

    model = _as_model(mdp)
    supported = _supported_action_mask(
        action_mask, (model.num_states, model.num_actions)
    )
    full_behavior_policy = _policy_matrix(behavior_policy, model)
    restricted_behavior_policy = _restrict_policy_to_actions(
        full_behavior_policy, supported
    )
    baseline = evaluate_policy(mdp, restricted_behavior_policy)

    q_sequence = np.empty(
        (k_max + 1, model.num_states, model.num_actions), dtype=np.float64
    )
    greedy_masks = np.empty_like(q_sequence, dtype=np.bool_)
    greedy_policies = np.empty_like(q_sequence)
    action_gaps = np.empty((k_max + 1, model.num_states), dtype=np.float64)
    q_sequence[0] = baseline.q_values

    for depth in range(k_max + 1):
        greedy_masks[depth], greedy_policies[depth] = greedy_policy(
            q_sequence[depth],
            action_mask=supported,
            tie_tolerance=tie_tolerance,
        )
        action_gaps[depth] = action_gaps_from_q_values(
            q_sequence[depth], action_mask=supported
        )
        if depth < k_max:
            backed_values = np.where(
                supported, q_sequence[depth], -np.inf
            ).max(axis=1)
            q_sequence[depth + 1] = _q_from_values(model, backed_values)

    return KStepBackupResult(
        q_values=q_sequence,
        greedy_action_masks=greedy_masks,
        greedy_policies=greedy_policies,
        action_gaps=action_gaps,
        restricted_behavior_policy=restricted_behavior_policy,
    )


def random_policy(mdp: Any) -> FloatArray:
    """Return the uniform random policy used by normalized-return labels."""

    model = _as_model(mdp)
    return np.full(
        (model.num_states, model.num_actions),
        1.0 / model.num_actions,
        dtype=np.float64,
    )


def discounted_state_occupancy(
    mdp: Any, policy: Any, *, normalized: bool = True
) -> FloatArray:
    """Compute exact discounted state occupancy under ``policy``.

    With ``normalized=True`` the output is
    ``(1-gamma) * sum_t gamma**t Pr(s_t=s)`` and sums to one.
    """

    model = _as_model(mdp)
    policy_matrix = _policy_matrix(policy, model)
    policy_transitions = np.einsum(
        "sa,san->sn", policy_matrix, model.transitions
    )
    scale = 1.0 - model.gamma if normalized else 1.0
    system = np.eye(model.num_states) - model.gamma * policy_transitions.T
    try:
        occupancy = np.linalg.solve(system, scale * model.initial_distribution)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Không giải được hệ tuyến tính occupancy.") from error
    occupancy[np.abs(occupancy) < 1e-14] = 0.0
    if np.any(occupancy < -1e-10):
        raise RuntimeError("Occupancy chính xác có giá trị âm ngoài sai số số học.")
    return occupancy


def dynamic_range(
    mdp: Any, *, solution: ExactSolution | None = None
) -> tuple[float, float, float]:
    """Return ``(J_star - J_random, J_star, J_random)``."""

    model = _as_model(mdp)
    exact = solution if solution is not None else value_iteration(mdp)
    if exact.values.shape != (model.num_states,):
        raise ValueError("ExactSolution không khớp số trạng thái của MDP.")
    optimal_return = float(model.initial_distribution @ exact.values)
    random_return = evaluate_policy(mdp, random_policy(mdp)).expected_return
    return optimal_return - random_return, optimal_return, random_return


def passes_dynamic_range_filter(
    mdp: Any,
    *,
    threshold: float | None = None,
    solution: ExactSolution | None = None,
) -> bool:
    """Apply the PLAN_Final dynamic-range exclusion rule.

    Rewards must already have been affinely scaled to ``[0,1]`` by the
    benchmark generator.  The primary threshold is ``0.01/(1-gamma)``.
    """

    model = _as_model(mdp)
    actual_threshold = (
        0.01 / (1.0 - model.gamma) if threshold is None else float(threshold)
    )
    if actual_threshold < 0.0:
        raise ValueError("threshold không được âm.")
    range_value, _, _ = dynamic_range(mdp, solution=solution)
    return range_value >= actual_threshold


def normalized_return(
    mdp: Any,
    policy: Any,
    *,
    solution: ExactSolution | None = None,
    minimum_dynamic_range: float = 1e-12,
) -> float:
    """Compute ``(J(policy)-J_random)/(J_star-J_random)`` exactly."""

    if minimum_dynamic_range < 0.0:
        raise ValueError("minimum_dynamic_range không được âm.")
    range_value, _, random_return = dynamic_range(mdp, solution=solution)
    if range_value <= minimum_dynamic_range:
        raise DegenerateMDPError(
            "Normalized return không xác định vì J* - J(random) quá nhỏ."
        )
    policy_return = evaluate_policy(mdp, policy).expected_return
    return (policy_return - random_return) / range_value


__all__ = [
    "DegenerateMDPError",
    "ExactSolution",
    "KStepBackupResult",
    "MDPValidationError",
    "PolicyEvaluation",
    "action_gaps_from_q_values",
    "discounted_state_occupancy",
    "dynamic_range",
    "evaluate_policy",
    "greedy_policy",
    "k_step_backups",
    "normalized_return",
    "passes_dynamic_range_filter",
    "q_values_from_values",
    "random_policy",
    "validate_mdp",
    "value_iteration",
]
