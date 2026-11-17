"""Oracle-only exact 100-step returns for a published BRIDGE tabular task.

Policies are stationary over the adapter's observed Markov state. The finite
episode cap is applied during evaluation, matching the authors' tabular setup.
These values are for benchmark labels and normalization, never pilot features.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from environments.tabular_environments import TabularMDP


class FiniteHorizonEvaluator:
    def __init__(self, mdp: TabularMDP, horizon: int = 100):
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        self.mdp = mdp
        self.horizon = horizon
        self.states, self.actions = mdp.num_states, mdp.num_actions
        s, a, next_s = np.nonzero(mdp.P)
        prob = mdp.P[s, a, next_s]
        self.transition = csr_matrix((prob, (s * self.actions + a, next_s)),
                                     shape=(self.states * self.actions, self.states))
        self.reward = np.bincount(s * self.actions + a,
                                  weights=prob * mdp.R[s, a, next_s],
                                  minlength=self.states * self.actions).reshape(self.states, self.actions)
        self._reference = None

    def value(self, policy: np.ndarray | None = None, *, optimal: bool = False) -> float:
        if optimal and policy is not None:
            raise ValueError("Choose policy or optimal, not both")
        if policy is None and not optimal:
            policy = np.full((self.states, self.actions), 1 / self.actions)
        if policy is not None and (policy.shape != (self.states, self.actions) or
                                   not np.isfinite(policy).all() or
                                   not np.allclose(policy.sum(1), 1, atol=1e-10) or
                                   np.any(policy < 0)):
            raise ValueError("Invalid stationary policy")
        v = np.zeros(self.states)
        for _ in range(self.horizon):
            q = self.reward + self.mdp.gamma * (self.transition @ v).reshape(self.states, self.actions)
            v = q.max(axis=1) if optimal else np.sum(policy * q, axis=1)
        return float(self.mdp.initial_state_distribution @ v)

    def reference(self) -> tuple[float, float]:
        if self._reference is None:
            self._reference = (self.value(), self.value(optimal=True))
        return self._reference

    def normalized(self, policy: np.ndarray) -> float:
        random_return, optimal_return = self.reference()
        span = optimal_return - random_return
        if not np.isfinite(span) or span <= 1e-8:
            raise ValueError("Degenerate finite-horizon return range")
        return (self.value(policy) - random_return) / span
