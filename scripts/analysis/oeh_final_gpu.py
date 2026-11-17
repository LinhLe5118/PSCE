"""CUDA float64 versions of the batched registered core learners.

Posterior sampling, random numbers and trajectory simulation stay in
``oeh.predictive`` on the CPU, so a GPU run sees exactly the same simulated
datasets as the NumPy reference. Only the fitted-value iterations move to the
GPU. They mirror :mod:`oeh.tabular` operation by operation in float64; results
can differ from the CPU only by floating-point summation order, which is
audited by :func:`agreement` before use.
"""
from __future__ import annotations

import numpy as np
import torch

from oeh import tabular as T


def _tensor(x, device):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)


def _model(model: T.CountModel, device):
    counts = _tensor(model.counts, device)
    nxt = _tensor(model.next_counts, device)
    rewards = _tensor(model.reward_sums, device)
    terminal = torch.as_tensor(model.terminal, device=device)
    denominator = counts[..., None]
    p_hat = torch.where(denominator > 0, nxt / torch.where(denominator > 0, denominator, 1.0), 0.0)
    r_hat = torch.where(counts > 0, rewards / torch.where(counts > 0, counts, 1.0), 0.0)
    return counts, p_hat, r_hat, terminal


def _greedy(q, supported, terminal):
    actions = q.shape[-1]
    policy = torch.full_like(q, 1.0 / actions)
    decide = supported.any(-1) & ~terminal
    winner = torch.where(supported, q, -torch.inf).argmax(-1)
    chosen = torch.nn.functional.one_hot(winner, actions).to(q.dtype)
    return torch.where(decide[..., None], chosen, policy)


def cfqi(model: T.CountModel, gamma: float, device="cuda") -> np.ndarray:
    counts, p_hat, r_hat, terminal = _model(model, device)
    supported = counts > 0
    rows = supported.any(-1) & ~terminal
    q = torch.zeros_like(counts)
    active = torch.ones(q.shape[0], dtype=torch.bool, device=device)
    for _ in range(T.ITERATIONS):
        if not bool(active.any()):
            break
        maxima = torch.where(supported, q, -torch.inf).amax(-1)
        maxima = torch.where(rows, maxima, 0.0)
        updated = r_hat + gamma * torch.einsum("bsan,bn->bsa", p_hat, maxima)
        updated = torch.where(supported, updated, 0.0)
        norm = (updated - q).abs().amax((1, 2))
        q = torch.where(active[:, None, None], updated, q)
        active &= norm > T.TOLERANCE
    return _greedy(q, supported, terminal).cpu().numpy()


def _expectile(values, weights, mask, expectile):
    order = torch.argsort(torch.where(mask, values, torch.inf), dim=-1, stable=True)
    v = torch.take_along_dim(values, order, -1)
    w = torch.take_along_dim(torch.where(mask, weights, 0.0), order, -1)
    m = torch.take_along_dim(mask, order, -1)
    n = m.sum(-1)
    v = torch.where(m, v, 0.0)
    width = values.shape[-1]
    positions = torch.arange(width, device=values.device)
    result = torch.full(values.shape[:-1], torch.nan, dtype=values.dtype, device=values.device)
    fallback = torch.full_like(result, torch.nan)
    violation = torch.full_like(result, torch.inf)
    for split in range(width + 1):
        coefficients = torch.where(positions < split, 1.0 - expectile, expectile).to(values.dtype)
        effective = w * coefficients
        candidate = (effective * v).sum(-1) / effective.sum(-1)
        if split:
            lower = v[..., split - 1]
        else:
            lower = torch.full_like(candidate, -torch.inf)
        upper = torch.where(split < n, v[..., min(split, width - 1)], torch.inf)
        valid = ((split <= n) & torch.isnan(result)
                 & (lower - 1e-15 <= candidate) & (candidate <= upper + 1e-15))
        result = torch.where(valid, candidate, result)
        miss = torch.maximum(lower - candidate, candidate - upper)
        better = (split <= n) & (miss < violation)
        violation = torch.where(better, miss, violation)
        fallback = torch.where(better, candidate, fallback)
    return torch.where(torch.isnan(result) & (n > 0), fallback, result)


def iql(model: T.CountModel, gamma: float, device="cuda") -> np.ndarray:
    counts, p_hat, r_hat, terminal = _model(model, device)
    supported = counts > 0
    totals = counts.sum(-1, keepdim=True)
    behavior = torch.where(totals > 0, counts / torch.where(totals > 0, totals, 1.0), 0.0)
    rows = supported.any(-1) & ~terminal
    values = torch.zeros(counts.shape[:2], dtype=torch.float64, device=device)
    active = torch.ones(values.shape[0], dtype=torch.bool, device=device)
    for _ in range(T.ITERATIONS):
        if not bool(active.any()):
            break
        q = r_hat + gamma * torch.einsum("bsan,bn->bsa", p_hat, values)
        solved = _expectile(q, behavior, supported, T.IQL_EXPECTILE)
        updated = torch.where(rows, solved, 0.0)
        norm = (updated - values).abs().amax(-1)
        values = torch.where(active[:, None], updated, values)
        active &= norm > T.TOLERANCE
    q = r_hat + gamma * torch.einsum("bsan,bn->bsa", p_hat, values)
    logits = T.IQL_INVERSE_TEMPERATURE * (q - values[..., None])
    logits = torch.where(supported, logits, -torch.inf)
    logits = logits - logits.amax(-1, keepdim=True)
    weighted = torch.where(supported, behavior * torch.exp(logits), 0.0)
    extracted = weighted / weighted.sum(-1, keepdim=True)
    policy = torch.full_like(q, 1.0 / q.shape[-1])
    return torch.where(rows[..., None], extracted, policy).cpu().numpy()


def learners(device="cuda") -> dict:
    """Drop-in replacement for :data:`oeh.tabular.CORE_LEARNERS`."""
    return {
        "cfqi_tabular": lambda model, gamma: cfqi(model, gamma, device),
        "iql_tabular": lambda model, gamma: iql(model, gamma, device),
        "behavior_cloning_tabular": lambda model, gamma: T.behavior_cloning(model),
    }


def agreement(models: list[T.CountModel], gamma: float, device="cuda") -> dict:
    """Compare GPU and NumPy policies on the same count models."""
    gpu, cpu = learners(device), T.CORE_LEARNERS
    report = {}
    for name in gpu:
        exact = total = 0
        worst = 0.0
        for model in models:
            a, b = gpu[name](model, gamma), cpu[name](model, gamma)
            exact += int(np.sum(np.all(a == b, axis=(1, 2))))
            total += a.shape[0]
            worst = max(worst, float(np.abs(a - b).max()))
        report[name] = {"datasets": total, "identical_policies": exact, "max_abs_difference": worst}
    return report
