"""Numerically checked accelerators for a new BRIDGE transfer run only.

The sealed historical OEH code is untouched. Call ``install()`` inside the
fresh runner after development parity checks; it replaces only three kernels
in that process. All posterior draws, learners and endpoints remain the same.
"""
from __future__ import annotations

import numpy as np


def simulate_streams_searchsorted(sample, rng, *, streams: int, length: int,
                                  horizon: int) -> dict[str, np.ndarray]:
    """Same RNG draw order as ``oeh.posterior.simulate_streams``.

    Offset-sorted CDFs permit C-level binary search across rows, avoiding the
    original O(S) comparison for every simulated transition.
    """
    B, S, A, _ = sample.P.shape
    rows = B * streams
    owner = np.repeat(np.arange(B), streams)
    rho_cdf = np.cumsum(sample.rho)
    beta_cdf = np.cumsum(sample.beta, axis=-1)
    p_cdf = np.cumsum(sample.P, axis=-1)
    beta_offset = 2.0 * np.arange(B * S, dtype=float)[:, None]
    p_offset = 2.0 * np.arange(B * S * A, dtype=float)[:, None]
    beta_sorted = (beta_cdf.reshape(-1, A) + beta_offset).ravel()
    p_sorted = (p_cdf.reshape(-1, S) + p_offset).ravel()
    result = {name: np.empty((rows, length), dtype=dtype) for name, dtype in
              (("state", np.int64), ("action", np.int64), ("reward", float),
               ("next_state", np.int64), ("terminal", bool), ("truncated", bool),
               ("trajectory_id", np.int64), ("timestep", np.int64))}
    state = np.minimum(np.searchsorted(
        rho_cdf, rng.random(rows) * rho_cdf[-1], side="right"), S - 1)
    clock = np.zeros(rows, dtype=np.int64)
    episode = np.zeros(rows, dtype=np.int64)
    for step in range(length):
        beta_row = owner * S + state
        action = (np.searchsorted(beta_sorted,
                    2.0 * beta_row + rng.random(rows) * beta_cdf[owner, state, -1],
                    side="right") % A).astype(np.int64)
        p_row = (owner * S + state) * A + action
        next_state = (np.searchsorted(p_sorted,
                          2.0 * p_row + rng.random(rows) * p_cdf[owner, state, action, -1],
                          side="right") % S).astype(np.int64)
        reward = sample.R[owner, state, action, next_state]
        terminal = sample.terminal[next_state]
        result["state"][:, step] = state
        result["action"][:, step] = action
        result["reward"][:, step] = reward
        result["next_state"][:, step] = next_state
        result["terminal"][:, step] = terminal
        result["truncated"][:, step] = (clock + 1 >= horizon) & ~terminal
        result["trajectory_id"][:, step] = episode
        result["timestep"][:, step] = clock
        restart = terminal | (clock + 1 >= horizon)
        clock = np.where(restart, 0, clock + 1)
        episode += restart
        fresh = np.minimum(np.searchsorted(
            rho_cdf, rng.random(rows) * rho_cdf[-1], side="right"), S - 1)
        state = np.where(restart, fresh, next_state)
    return {key: value.reshape(B, streams, length) for key, value in result.items()}


def finite_horizon_references_cuda(P, r, rho, gamma, horizon):
    from oeh.tabular import finite_horizon_references as cpu
    if P.shape[1] < 400:
        return cpu(P, r, rho, gamma, horizon)
    import torch
    if not torch.cuda.is_available():
        return cpu(P, r, rho, gamma, horizon)
    random, optimal = [], []
    with torch.inference_mode():
        for i in range(len(P)):
            p = torch.as_tensor(P[i], dtype=torch.float64, device="cuda")
            reward = torch.as_tensor(r[i], dtype=torch.float64, device="cuda")
            initial = torch.as_tensor(rho[i], dtype=torch.float64, device="cuda")
            value_opt = torch.zeros(P.shape[1], dtype=torch.float64, device="cuda")
            value_rand = torch.zeros_like(value_opt)
            p_rand = p.mean(dim=1)
            r_rand = reward.mean(dim=1)
            for _ in range(horizon):
                value_opt = (reward + gamma * torch.einsum("san,n->sa", p, value_opt)).max(dim=-1).values
                value_rand = r_rand + gamma * torch.mv(p_rand, value_rand)
            optimal.append(float(torch.dot(initial, value_opt).cpu()))
            random.append(float(torch.dot(initial, value_rand).cpu()))
    return np.asarray(random), np.asarray(optimal)


def finite_horizon_policy_returns_cuda(P, r, rho, policies, gamma, horizon):
    from oeh.tabular import finite_horizon_policy_returns as cpu
    if P.shape[1] < 400:
        return cpu(P, r, rho, policies, gamma, horizon)
    import torch
    if not torch.cuda.is_available():
        return cpu(P, r, rho, policies, gamma, horizon)
    output = []
    with torch.inference_mode():
        for i in range(len(P)):
            p = torch.as_tensor(P[i], dtype=torch.float64, device="cuda")
            reward = torch.as_tensor(r[i], dtype=torch.float64, device="cuda")
            initial = torch.as_tensor(rho[i], dtype=torch.float64, device="cuda")
            policy = torch.as_tensor(policies[i], dtype=torch.float64, device="cuda")
            p_pi = torch.einsum("ksa,san->ksn", policy, p)
            r_pi = torch.einsum("ksa,sa->ks", policy, reward)
            value = torch.zeros_like(r_pi)
            for _ in range(horizon):
                value = r_pi + gamma * torch.matmul(p_pi, value.unsqueeze(-1)).squeeze(-1)
            output.append(torch.einsum("s,ks->k", initial, value).cpu().numpy())
    return np.stack(output)


def install() -> None:
    """Use the accelerators only in this process's ``oeh.predictive`` module."""
    from oeh import posterior, predictive
    posterior.simulate_streams = simulate_streams_searchsorted
    predictive.finite_horizon_references = finite_horizon_references_cuda
    predictive.finite_horizon_policy_returns = finite_horizon_policy_returns_cuda
