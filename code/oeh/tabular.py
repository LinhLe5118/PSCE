"""Batched tabular models, registered core learners and exact evaluation.

Every function accepts a leading batch axis so that many simulated datasets can
be processed with NumPy array operations. The three learners reproduce the
registered E1 definitions of ``cfqi_tabular``, ``iql_tabular`` and
``behavior_cloning_tabular``: the same empirical model, convergence rule,
support mask, tie order and uniform fallback. Agreement with the archived
per-dataset implementation is checked in ``tests/test_tabular.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Registered learning constants of the original protocol configuration. They
# belong to the learners whose outcomes define the labels, not to OEH.
ITERATIONS = 100
TOLERANCE = 1e-10
IQL_EXPECTILE = 0.7
IQL_INVERSE_TEMPERATURE = 3.0
SUCCESS_THRESHOLD = 0.95


@dataclass(frozen=True)
class CountModel:
    """Sufficient statistics of a batch of transition datasets.

    ``next_counts`` excludes transitions into terminal states, matching the
    registered learners: such transitions end the return, so the empirical
    transition row keeps the missing mass as an absorbing zero-value outcome.
    """

    counts: np.ndarray        # (B, S, A) all transitions
    next_counts: np.ndarray   # (B, S, A, S) non-terminal transitions
    reward_sums: np.ndarray   # (B, S, A)
    terminal: np.ndarray      # (B, S) next states observed as terminal

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.counts.shape

    def empirical(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the maximum-likelihood ``p_hat`` and ``r_hat``."""
        denominator = self.counts[..., None]
        p_hat = np.divide(self.next_counts, denominator,
                          out=np.zeros_like(self.next_counts), where=denominator > 0)
        r_hat = np.divide(self.reward_sums, self.counts,
                          out=np.zeros_like(self.reward_sums), where=self.counts > 0)
        return p_hat, r_hat


def count_model(state: np.ndarray, action: np.ndarray, reward: np.ndarray,
                next_state: np.ndarray, terminal: np.ndarray, *,
                num_states: int, num_actions: int) -> CountModel:
    """Build sufficient statistics from aligned ``(B, N)`` transition arrays.

    One-dimensional inputs are treated as a batch of one dataset.
    """
    arrays = [np.atleast_2d(np.asarray(value)) for value in
              (state, action, reward, next_state, terminal)]
    s, a, r, n, t = arrays
    if len({item.shape for item in arrays}) != 1:
        raise ValueError("Transition arrays must share one (batch, size) shape")
    if s.size and (s.min() < 0 or n.min() < 0 or a.min() < 0
                   or s.max() >= num_states or n.max() >= num_states
                   or a.max() >= num_actions):
        raise ValueError("Transition index outside model dimensions")
    if not np.all(np.isfinite(r)):
        raise ValueError("Transition reward must be finite")
    batch = s.shape[0]
    t = t.astype(bool)
    offsets = np.arange(batch)[:, None]
    pair = (offsets * num_states + s) * num_actions + a
    cells = batch * num_states * num_actions
    counts = np.bincount(pair.ravel(), minlength=cells).astype(float)
    rewards = np.bincount(pair.ravel(), weights=r.ravel().astype(float), minlength=cells)
    live = ~t
    triple = pair[live] * num_states + n[live]
    next_counts = np.bincount(triple, minlength=cells * num_states).astype(float)
    terminal = np.zeros((batch, num_states), dtype=bool)
    rows, cols = np.nonzero(t)
    terminal[rows, n[rows, cols]] = True
    return CountModel(
        counts=counts.reshape(batch, num_states, num_actions),
        next_counts=next_counts.reshape(batch, num_states, num_actions, num_states),
        reward_sums=rewards.reshape(batch, num_states, num_actions),
        terminal=terminal,
    )


def _apply(p: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Batched ``einsum('san,n->sa')``; bitwise equal to the per-item product."""
    return np.einsum("bsan,bn->bsa", p, values)


def _greedy(q: np.ndarray, supported: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy; ties choose the smallest supported action."""
    batch, states, actions = q.shape
    policy = np.full(q.shape, 1.0 / actions)
    decide = supported.any(-1) & ~terminal
    score = np.where(supported, q, -np.inf)
    winner = score.argmax(-1)
    chosen = np.zeros_like(policy)
    np.put_along_axis(chosen, winner[..., None], 1.0, axis=-1)
    policy[decide] = chosen[decide]
    return policy


def cfqi(model: CountModel, gamma: float, *, iterations: int = ITERATIONS,
         tolerance: float = TOLERANCE) -> np.ndarray:
    """Support-conservative fitted Q iteration (registered ``cfqi_tabular``)."""
    p_hat, r_hat = model.empirical()
    supported = model.counts > 0
    active_rows = supported.any(-1) & ~model.terminal
    q = np.zeros(model.counts.shape)
    active = np.ones(q.shape[0], dtype=bool)
    for _ in range(iterations):
        if not active.any():
            break
        maxima = np.where(supported, q, -np.inf).max(-1)
        maxima = np.where(active_rows, maxima, 0.0)
        updated = r_hat + gamma * _apply(p_hat, maxima)
        updated[~supported] = 0.0
        norm = np.abs(updated - q).max(axis=(1, 2))
        q = np.where(active[:, None, None], updated, q)  # converged items stay frozen
        active &= norm > tolerance
    return _greedy(q, supported, model.terminal)


def _expectile(values: np.ndarray, weights: np.ndarray, mask: np.ndarray,
               expectile: float) -> np.ndarray:
    """Exact weighted expectile over supported entries of the last axis.

    Splits are tested in the registered order with its absolute ``1e-15``
    tolerance, so every value the archived learner returns is reproduced
    bitwise. The archived learner raises when rounding places every candidate
    outside its interval (e.g. tied values of magnitude above about 4); there
    the split with the smallest interval violation is returned, which is the
    exact solution up to rounding.
    """
    order = np.argsort(np.where(mask, values, np.inf), axis=-1, kind="stable")
    v = np.take_along_axis(values, order, -1)
    w = np.take_along_axis(np.where(mask, weights, 0.0), order, -1)
    m = np.take_along_axis(mask, order, -1)
    n = m.sum(-1)
    v = np.where(m, v, 0.0)
    result = np.full(values.shape[:-1], np.nan)
    width = values.shape[-1]
    positions = np.arange(width)
    fallback = np.full(values.shape[:-1], np.nan)
    violation = np.full(values.shape[:-1], np.inf)
    for split in range(width + 1):
        coefficients = np.where(positions < split, 1.0 - expectile, expectile)
        effective = w * coefficients
        with np.errstate(invalid="ignore", divide="ignore"):
            candidate = (effective * v).sum(-1) / effective.sum(-1)
        lower = (np.take_along_axis(v, np.full(n.shape + (1,), max(split - 1, 0)), -1)[..., 0]
                 if split else np.full(n.shape, -np.inf))
        upper_index = np.minimum(split, width - 1)
        upper = np.where(split < n,
                         np.take_along_axis(v, np.full(n.shape + (1,), upper_index), -1)[..., 0],
                         np.inf)
        valid = ((split <= n) & np.isnan(result)
                 & (lower - 1e-15 <= candidate) & (candidate <= upper + 1e-15))
        result = np.where(valid, candidate, result)
        with np.errstate(invalid="ignore"):
            miss = np.maximum(lower - candidate, candidate - upper)
        better = (split <= n) & (miss < violation)
        violation = np.where(better, miss, violation)
        fallback = np.where(better, candidate, fallback)
    return np.where(np.isnan(result) & (n > 0), fallback, result)


def iql(model: CountModel, gamma: float, *, expectile: float = IQL_EXPECTILE,
        inverse_temperature: float = IQL_INVERSE_TEMPERATURE,
        iterations: int = ITERATIONS, tolerance: float = TOLERANCE) -> np.ndarray:
    """Tabular IQL with exact expectile values and AWBC extraction."""
    p_hat, r_hat = model.empirical()
    supported = model.counts > 0
    totals = model.counts.sum(-1, keepdims=True)
    behavior = np.divide(model.counts, totals, out=np.zeros_like(model.counts),
                         where=totals > 0)
    rows = supported.any(-1) & ~model.terminal
    values = np.zeros(model.counts.shape[:2])
    active = np.ones(values.shape[0], dtype=bool)
    for _ in range(iterations):
        if not active.any():
            break
        q = r_hat + gamma * _apply(p_hat, values)
        solved = _expectile(q, behavior, supported, expectile)
        updated = np.where(rows, solved, 0.0)
        norm = np.abs(updated - values).max(-1)
        values = np.where(active[:, None], updated, values)  # converged items stay frozen
        active &= norm > tolerance
    q = r_hat + gamma * _apply(p_hat, values)
    logits = inverse_temperature * (q - values[..., None])
    logits = np.where(supported, logits, -np.inf)
    with np.errstate(invalid="ignore"):
        logits = logits - logits.max(-1, keepdims=True)
        weighted = np.where(supported, behavior * np.exp(logits), 0.0)
        extracted = weighted / weighted.sum(-1, keepdims=True)
    policy = np.full(q.shape, 1.0 / q.shape[-1])
    policy[rows] = extracted[rows]
    return policy


def behavior_cloning(model: CountModel) -> np.ndarray:
    """Maximum-likelihood behavior cloning with uniform fallback."""
    totals = model.counts.sum(-1)
    observed = (totals > 0) & ~model.terminal
    policy = np.full(model.counts.shape, 1.0 / model.counts.shape[-1])
    policy[observed] = model.counts[observed] / totals[observed][:, None]
    return policy


CORE_LEARNERS = {
    "cfqi_tabular": lambda model, gamma: cfqi(model, gamma),
    "iql_tabular": lambda model, gamma: iql(model, gamma),
    "behavior_cloning_tabular": lambda model, gamma: behavior_cloning(model),
}


def policy_values(P: np.ndarray, r: np.ndarray, policy: np.ndarray,
                  gamma: float) -> np.ndarray:
    """Exact state values of stochastic policies; batch axis is leading."""
    p_pi = np.einsum("...sa,...san->...sn", policy, P)
    r_pi = np.einsum("...sa,...sa->...s", policy, r)
    eye = np.eye(P.shape[-1])
    return np.linalg.solve(eye - gamma * p_pi, r_pi[..., None])[..., 0]


def optimal_values(P: np.ndarray, r: np.ndarray, gamma: float, *,
                   max_iterations: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimal values and greedy policy by batched policy iteration."""
    actions = P.shape[-2]
    policy_index = r.argmax(-1)
    for _ in range(max_iterations):
        policy = np.eye(actions)[policy_index]
        values = policy_values(P, r, policy, gamma)
        q = r + gamma * np.einsum("...san,...n->...sa", P, values)
        current = np.take_along_axis(q, policy_index[..., None], -1)[..., 0]
        improve = q.max(-1) > current + 1e-12 * (1.0 + np.abs(current))
        if not improve.any():
            return values, policy_index
        policy_index = np.where(improve, q.argmax(-1), policy_index)
    raise RuntimeError("Policy iteration did not converge")


def normalized_returns(P: np.ndarray, r: np.ndarray, rho: np.ndarray,
                       policies: np.ndarray, gamma: float) -> np.ndarray:
    """Protocol-normalized return ``(J(pi)-J_rand)/(J*-J_rand)``.

    ``P`` has shape ``(M, S, A, S)``, ``r`` ``(M, S, A)``, ``rho`` ``(M, S)`` and
    ``policies`` ``(M, K, S, A)``: ``K`` policies are scored in each MDP.
    Returns an ``(M, K)`` array.
    """
    j_star, j_rand = reference_returns(P, r, rho, gamma)
    values = policy_values(P[:, None], r[:, None], policies, gamma)
    j = np.einsum("mks,ms->mk", values, rho)
    return (j - j_rand[:, None]) / (j_star - j_rand)[:, None]


def finite_horizon_normalized_returns(P: np.ndarray, r: np.ndarray, rho: np.ndarray,
                                      policies: np.ndarray, gamma: float,
                                      horizon: int) -> np.ndarray:
    """Normalize exact finite-horizon returns by random and time-dependent optimal.

    This uses the same terminal-step cap as the BRIDGE label evaluator. The
    historical infinite-horizon benchmark continues to use ``normalized_returns``.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if P.ndim != 4 or r.shape != P.shape[:3] or rho.shape != (P.shape[0], P.shape[1]):
        raise ValueError("Invalid batched MDP shapes")
    if policies.shape[:1] != P.shape[:1] or policies.shape[2:] != P.shape[1:3]:
        raise ValueError("Invalid policy shapes")
    j_rand, j_opt = finite_horizon_references(P, r, rho, gamma, horizon)
    j_policy = finite_horizon_policy_returns(P, r, rho, policies, gamma, horizon)
    span = j_opt - j_rand
    if not np.isfinite(span).all() or np.any(span <= 1e-8):
        raise ValueError("Degenerate finite-horizon return range")
    return (j_policy - j_rand[:, None]) / span[:, None]


def finite_horizon_references(P: np.ndarray, r: np.ndarray, rho: np.ndarray,
                              gamma: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Random and time-dependent optimal 100-step references, batched by MDP."""
    m, s = rho.shape
    value_opt = np.zeros((m, s))
    value_rand = np.zeros((m, s))
    p_rand = P.mean(axis=2)
    r_rand = r.mean(axis=2)
    for _ in range(horizon):
        q_opt = r + gamma * np.einsum("msan,mn->msa", P, value_opt)
        value_opt = q_opt.max(axis=-1)
        value_rand = r_rand + gamma * np.einsum("msn,mn->ms", p_rand, value_rand)
    return np.sum(rho * value_rand, axis=-1), np.sum(rho * value_opt, axis=-1)


def finite_horizon_policy_returns(P: np.ndarray, r: np.ndarray, rho: np.ndarray,
                                  policies: np.ndarray, gamma: float,
                                  horizon: int) -> np.ndarray:
    """Exact finite-horizon return of stationary policies, batched by MDP."""
    m, k, s, _ = policies.shape
    p_policy = np.einsum("mksa,msan->mksn", policies, P)
    r_policy = np.einsum("mksa,msa->mks", policies, r)
    value = np.zeros((m, k, s))
    for _ in range(horizon):
        value = r_policy + gamma * np.einsum("mksn,mkn->mks", p_policy, value)
    return np.einsum("ms,mks->mk", rho, value)


def reference_returns(P: np.ndarray, r: np.ndarray, rho: np.ndarray,
                      gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal and uniform-random returns used by the protocol normalization."""
    optimum, _ = optimal_values(P, r, gamma)
    uniform = np.full(r.shape, 1.0 / r.shape[-1])
    random = policy_values(P, r, uniform, gamma)
    return (np.einsum("...s,...s->...", rho, optimum),
            np.einsum("...s,...s->...", rho, random))
