"""Posterior-predictive offline requirement (OEH-B).

For one pilot, MDPs and behavior policies are drawn from the hyperparameter-free
posterior of :mod:`oeh.posterior`. In each draw, the registered labeling
protocol is replayed exactly: independent natural trajectory streams, nested
budget prefixes, the learner itself, exact evaluation, the normalized-return
success threshold and the sustained ``8/10`` suffix rule. The predictive
category law is the posterior mixture of the per-draw outcomes, estimated from
``draws`` Monte Carlo draws with the Krichevsky-Trofimov (Jeffreys) estimator,
the minimax-regret estimator under logarithmic loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from . import posterior as post
from .tabular import (CORE_LEARNERS, SUCCESS_THRESHOLD, count_model,
                      finite_horizon_policy_returns, finite_horizon_references,
                      normalized_returns)

GRID = (1, 2, 5, 10, 20, 50)
SUSTAINED_SEEDS = 8
SEEDS = 10
HORIZON = 100
CHUNK_DRAWS = 8  # memory bound only; fixed so random streams are reproducible

DatasetLearner = Callable[[dict, dict], np.ndarray]


@dataclass(frozen=True)
class Outcomes:
    """Normalized returns with shape ``(draws, seeds, budgets, learners)``."""

    learners: tuple[str, ...]
    budgets: tuple[int, ...]
    normalized_return: np.ndarray
    posterior: dict
    sampling: dict


def suffix_event(successes: np.ndarray, minimum: int = SUSTAINED_SEEDS) -> np.ndarray:
    """First budget index from which every larger budget has ``minimum`` successes.

    ``successes`` has shape ``(..., budgets)`` with per-budget success counts.
    Returns ``budgets`` (censored) when no suffix qualifies.
    """
    ok = successes >= minimum
    suffix = np.flip(np.logical_and.accumulate(np.flip(ok, -1), -1), -1)
    width = successes.shape[-1]
    first = np.where(suffix.any(-1), suffix.argmax(-1), width)
    return first


def kt_probabilities(events: np.ndarray, categories: int) -> np.ndarray:
    """Krichevsky-Trofimov estimate of a categorical law from samples."""
    events = np.asarray(events, dtype=np.int64).ravel()
    counts = np.bincount(events, minlength=categories)[:categories].astype(float)
    return (counts + 0.5) / (counts.sum() + 0.5 * categories)


def posterior_outcomes(pilot: Mapping[str, np.ndarray], rho: Sequence[float], *,
                       num_states: int, num_actions: int, gamma: float,
                       draws: int, seed: Sequence[int] | int,
                       budgets: Sequence[int] = GRID, seeds: int = SEEDS,
                       batched: Mapping[str, Callable] | None = None,
                       per_dataset: Mapping[str, DatasetLearner] | None = None,
                       minimum_dynamic_range: float | None = None,
                       reward_bound: float = 1.0, horizon: int = HORIZON,
                       evaluation_horizon: int | None = None,
                       reward_prior: str = "exchangeable",
                       transition_base: str = "uniform", behavior_prior: str = "single",
                       transition_prior: str = "single") -> Outcomes:
    """Replay the registered offline protocol inside posterior MDP draws.

    ``batched`` learners map a :class:`oeh.tabular.CountModel` batch and
    ``gamma`` to policies. ``per_dataset`` learners receive one dataset's
    arrays and a context dictionary (dimensions, ``gamma``, terminal states,
    the draw's behavior probabilities and a deterministic integer seed) and
    return one policy. All learners see identical simulated datasets.
    """
    batched = dict(CORE_LEARNERS if batched is None else batched)
    per_dataset = dict(per_dataset or {})
    names = tuple(batched) + tuple(per_dataset)
    if not names or len(set(names)) != len(names):
        raise ValueError("Learner names must be nonempty and distinct")
    budgets = tuple(int(b) for b in budgets)
    if list(budgets) != sorted(set(budgets)) or budgets[0] < 1:
        raise ValueError("Budgets must be strictly increasing positive multipliers")
    if minimum_dynamic_range is None:
        minimum_dynamic_range = 0.01 / (1.0 - gamma)  # public benchmark filter
    stats = post.pilot_statistics(pilot, rho, num_states=num_states, num_actions=num_actions,
                                  gamma=gamma, reward_bound=reward_bound)
    model = post.fit_posterior(stats, reward_prior=reward_prior, transition_base=transition_base,
                               behavior_prior=behavior_prior, transition_prior=transition_prior)
    main_seed, learner_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(main_seed)
    seed_rng = np.random.default_rng(learner_seed)
    base = num_states * num_actions
    length = budgets[-1] * base
    results, reports = [], []
    remaining = draws
    while remaining > 0:
        chunk = min(CHUNK_DRAWS, remaining)
        sample, report = post.sample_mdps(model, rng, chunk,
                                          minimum_dynamic_range=minimum_dynamic_range)
        reports.append(report)
        got = len(sample.P)
        remaining -= chunk
        if not got:
            continue
        streams = post.simulate_streams(sample, rng, streams=seeds, length=length, horizon=horizon)
        r = sample.expected_rewards
        if evaluation_horizon is not None:
            reference_random, reference_optimal = finite_horizon_references(
                sample.P, r, np.broadcast_to(sample.rho, (got, num_states)),
                gamma, evaluation_horizon)
            reference_span = reference_optimal - reference_random
            if not np.isfinite(reference_span).all() or np.any(reference_span <= 1e-8):
                raise ValueError("Degenerate posterior finite-horizon return range")
        values = np.empty((got, seeds, len(budgets), len(names)))
        for j, budget in enumerate(budgets):
            size = budget * base
            flat = {key: value[:, :, :size].reshape(got * seeds, size) for key, value in streams.items()}
            counts = count_model(flat["state"], flat["action"], flat["reward"],
                                 flat["next_state"], flat["terminal"],
                                 num_states=num_states, num_actions=num_actions)
            policies = []
            for name, learner in batched.items():
                policies.append(learner(counts, gamma).reshape(got, seeds, num_states, num_actions))
            for name, learner in per_dataset.items():
                block = np.empty((got, seeds, num_states, num_actions))
                for b in range(got):
                    for i in range(seeds):
                        arrays = {key: value[b * seeds + i] for key, value in flat.items()}
                        context = {"num_states": num_states, "num_actions": num_actions,
                                   "gamma": gamma, "terminal_states": tuple(np.flatnonzero(sample.terminal)),
                                   "behavior": sample.beta[b], "seed": int(seed_rng.integers(2**31 - 1))}
                        block[b, i] = learner(arrays, context)
                policies.append(block)
            stacked = np.stack(policies, axis=2)  # (got, seeds, learners, S, A)
            flattened = stacked.reshape(got, seeds * len(names), num_states, num_actions)
            if evaluation_horizon is None:
                scored = normalized_returns(
                    sample.P, r, np.broadcast_to(sample.rho, (got, num_states)),
                    flattened, gamma)
            else:
                policy_return = finite_horizon_policy_returns(
                    sample.P, r, np.broadcast_to(sample.rho, (got, num_states)),
                    flattened, gamma, evaluation_horizon)
                scored = (policy_return - reference_random[:, None]) / reference_span[:, None]
            values[:, :, j, :] = scored.reshape(got, seeds, len(names))
        results.append(values)
    sampling = {"draws_requested": draws,
                "draws_returned": int(sum(len(item) for item in results)),
                "proposals": int(sum(item["proposals"] for item in reports)),
                "accepted": int(sum(round(item["acceptance_rate"] * item["proposals"]) for item in reports))}
    empty = np.empty((0, seeds, len(budgets), len(names)))
    return Outcomes(names, budgets, np.concatenate(results) if results else empty,
                    model.describe(), sampling)


def requirement_forecast(outcomes: Outcomes, *, threshold: float = SUCCESS_THRESHOLD,
                         minimum: int = SUSTAINED_SEEDS) -> dict[str, dict]:
    """Seven-category requirement law and scalar OEH-B for every learner."""
    if outcomes.budgets != GRID:
        raise ValueError("Requirement forecasts use the registered budget grid")
    successes = (outcomes.normalized_return >= threshold).sum(axis=1)  # (draws, budgets, learners)
    # A posterior with no draw compatible with the public range filter cannot
    # score any learner as successful: its requested draws count as censored.
    missing = outcomes.sampling["draws_requested"] if not len(successes) else 0
    forecast = {}
    for index, name in enumerate(outcomes.learners):
        events = np.concatenate([suffix_event(successes[:, :, index], minimum),
                                 np.full(missing, len(GRID), dtype=np.int64)])
        probabilities = kt_probabilities(events, len(GRID) + 1)
        median_event = int(np.sort(events)[(len(events) - 1) // 2])
        forecast[name] = {"probabilities": probabilities, "events": events,
                          "median_event": median_event,
                          "censored_fraction": float(np.mean(events == len(GRID)))}
    return forecast


def offline_horizon(median_event: int, num_states: int, num_actions: int) -> float:
    """Scalar OEH-B: ``log_A`` of the posterior-median required transitions."""
    if median_event >= len(GRID):
        return math.inf
    return math.log(GRID[median_event] * num_states * num_actions, num_actions)
