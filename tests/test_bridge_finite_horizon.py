"""The exact scorer uses only the prespecified horizon and transition kernel."""
import numpy as np

from benchmarks.bridge.finite_horizon import FiniteHorizonEvaluator
from environments.tabular_environments import TabularMDP
from oeh.tabular import finite_horizon_normalized_returns
from oeh.tabular import normalized_returns


def test_two_action_one_step_normalized_return():
    P = np.zeros((2, 2, 2))
    R = np.zeros_like(P)
    P[0, :, 1] = 1
    P[1, :, 1] = 1
    R[0, 0, 1] = 1
    mdp = TabularMDP("fixture", P, R, np.array([1.0, 0.0]), (1,), .99, {})
    mdp.validate()
    evaluator = FiniteHorizonEvaluator(mdp, 100)
    assert np.allclose(evaluator.reference(), (.5, 1.0))
    good = np.array([[1, 0], [.5, .5]], dtype=float)
    bad = np.array([[0, 1], [.5, .5]], dtype=float)
    assert np.isclose(evaluator.normalized(good), 1.0)
    assert np.isclose(evaluator.normalized(bad), -1.0)
    expected_reward = np.einsum("san,san->sa", P, R)
    predicted = finite_horizon_normalized_returns(
        P[None], expected_reward[None], mdp.initial_state_distribution[None],
        np.stack([good, bad])[None], mdp.gamma, 100)
    assert np.allclose(predicted, [[evaluator.normalized(good), evaluator.normalized(bad)]])


def test_finite_cap_matches_oracle_when_infinite_horizon_is_different():
    P = np.zeros((2, 2, 2))
    R = np.zeros_like(P)
    P[0, 0, 0] = P[0, 1, 1] = 1
    P[1, 0, 0] = P[1, 1, 1] = 1
    R[0, 0, 0] = 1
    mdp = TabularMDP("fixture-continuing", P, R, np.array([1.0, 0.0]), (), .9, {})
    mdp.validate()
    evaluator = FiniteHorizonEvaluator(mdp, horizon=2)
    good = np.array([[1, 0], [1, 0]], dtype=float)
    bad = np.array([[0, 1], [0, 1]], dtype=float)
    policies = np.stack([good, bad])[None]
    reward = np.einsum("san,san->sa", P, R)
    finite = finite_horizon_normalized_returns(
        P[None], reward[None], mdp.initial_state_distribution[None],
        policies, mdp.gamma, 2)
    infinite = normalized_returns(
        P[None], reward[None], mdp.initial_state_distribution[None],
        policies, mdp.gamma)
    assert np.allclose(finite, [[evaluator.normalized(good), evaluator.normalized(bad)]])
    assert not np.allclose(finite, infinite)
