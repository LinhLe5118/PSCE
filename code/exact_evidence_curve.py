"""Oracle evidence curve for scoring OEH, never for the data-only estimator.

For a known probe/benchmark MDP and behaviour policy, the numerator is the
true finite-depth action gap. Its denominator is estimated from ``R=200``
independently collected datasets, rather than by bootstrap. This is therefore
an oracle scoring reference, not an alternative estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from oeh_estimator import (
    OEHConfig,
    _loss_and_multiplier,
    _model,
    _occupancy,
    _q_sequence,
    _smallest_near_minimum,
)
from pilot_collector import collect_pilot_dataset


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExactEvidenceCurve:
    deltas: FloatArray
    sigmas: FloatArray
    multipliers: FloatArray
    k_dagger: int
    valid_sample_fraction: float


def _true_q_sequence(
    mdp: object,
    beta: FloatArray,
    action_mask: NDArray[np.bool_],
    k_max: int,
) -> list[FloatArray]:
    p = np.asarray(getattr(mdp, "P"), dtype=np.float64)
    rewards = np.asarray(getattr(mdp, "R"), dtype=np.float64)
    expected_r = rewards if rewards.ndim == 2 else np.sum(p * rewards, axis=2)
    gamma = float(getattr(mdp, "gamma"))
    # Match the data-only empirical model exactly: a non-terminal state with
    # no supported action has zero behaviour mass and therefore zero
    # continuation value.  Such a state is only a certificate failure if it
    # is critical; it must not make the oracle reference undefined merely by
    # existing outside S_crit.
    p_beta = np.einsum("sa,san->sn", beta, p)
    r_beta = np.einsum("sa,sa->s", beta, expected_r)
    try:
        values = np.linalg.solve(np.eye(p.shape[0]) - gamma * p_beta, r_beta)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Exact restricted behaviour evaluation is singular.") from error
    q = np.full((p.shape[0], p.shape[1]), np.nan, dtype=np.float64)
    for state, row in enumerate(np.asarray(action_mask, dtype=np.bool_)):
        for action in np.flatnonzero(row):
            q[state, action] = expected_r[state, action] + gamma * p[state, action] @ values
    sequence = [q]
    for _ in range(k_max):
        maxima = np.zeros(action_mask.shape[0], dtype=np.float64)
        for state in range(action_mask.shape[0]):
            row = np.flatnonzero(action_mask[state])
            if len(row):
                maxima[state] = np.max(sequence[-1][state, row])
        sequence.append(expected_r + gamma * np.einsum("san,n->sa", p, maxima))
    return sequence


def exact_evidence_curve(
    mdp: object,
    beta: FloatArray,
    *,
    action_mask: NDArray[np.bool_] | None = None,
    config: OEHConfig | None = None,
    rng: np.random.Generator | None = None,
) -> ExactEvidenceCurve:
    """Compute the oracle curve using independent datasets from the true MDP.

    The caller supplies ``rng`` before observing results; it controls only
    Monte-Carlo sampling.  ``action_mask`` fixes the support restriction for
    both the true curve and every sampled empirical model.
    """
    cfg = config or OEHConfig.load()
    p = np.asarray(getattr(mdp, "P"), dtype=np.float64)
    states, actions = p.shape[:2]
    beta = np.asarray(beta, dtype=np.float64)
    if (
        beta.shape != (states, actions)
        or not np.all(np.isfinite(beta))
        or np.any(beta < 0.0)
        or not np.allclose(beta.sum(axis=1), 1.0)
    ):
        raise ValueError("beta must be a (S,A) policy matrix.")
    terminal_mask = np.zeros(states, dtype=np.bool_)
    terminal_mask[
        np.asarray(getattr(mdp, "terminal_states"), dtype=np.int64)
    ] = True
    initial = np.asarray(
        getattr(mdp, "initial_state_distribution"), dtype=np.float64
    )
    if np.any(initial[terminal_mask] > 0.0):
        raise ValueError("The exact reference cannot start in a terminal state.")
    if action_mask is None:
        support = np.ones((states, actions), dtype=np.bool_)
        support[terminal_mask] = False
    else:
        support = np.asarray(action_mask, dtype=np.bool_)
    if support.shape != (states, actions):
        raise ValueError(
            "action_mask must have shape (S,A)."
        )
    supported = tuple(
        tuple(np.flatnonzero(support[state])) for state in range(states)
    )
    restricted_beta = beta.copy()
    for state, row in enumerate(supported):
        if terminal_mask[state]:
            continue
        if not row:
            restricted_beta[state] = 0.0
            continue
        mass = restricted_beta[state, list(row)].sum()
        if mass <= 0.0:
            raise ValueError(
                "beta must assign positive mass to retained actions in every "
                "non-terminal state."
            )
        restricted_beta[state] = 0.0
        restricted_beta[state, list(row)] = beta[state, list(row)] / mass
    true_q = _true_q_sequence(mdp, restricted_beta, support, cfg.k_max)
    pairs = [
        [
            tuple(
                sorted(
                    row,
                    key=lambda action: (-float(q[state, action]), action),
                )[:2]
            )
            if len(row) >= 2
            else None
            for state, row in enumerate(supported)
        ]
        for q in true_q
    ]
    deltas = np.array(
        [
            [
                np.nan if pair is None else q[s, pair[0]] - q[s, pair[1]]
                for s, pair in enumerate(depth_pairs)
            ]
            for q, depth_pairs in zip(true_q, pairs)
        ]
    )
    sample_size = cfg.n0_coefficient * states * actions
    generator = rng if rng is not None else np.random.default_rng()
    sampled_gaps: list[np.ndarray] = []
    for _ in range(cfg.num_exact_replicates):
        dataset = collect_pilot_dataset(
            mdp,
            beta,
            num_transitions=sample_size,
            horizon=cfg.horizon,
            seed=int(generator.integers(0, 2**32 - 1)),
        )
        model = _model(dataset.transitions, states, actions, supported)
        if model is None:
            continue
        sequence = _q_sequence(*model[:3], supported, cfg.gamma, cfg.k_max)
        if sequence is None:
            continue
        sampled_gaps.append(
            np.array(
                [
                    [
                        np.nan
                        if pair is None
                        else q[s, pair[0]] - q[s, pair[1]]
                        for s, pair in enumerate(depth_pairs)
                    ]
                    for q, depth_pairs in zip(sequence, pairs)
                ]
            )
        )
    if len(sampled_gaps) < 2:
        raise RuntimeError(
            "Fewer than two support-valid oracle datasets; curve undefined."
        )
    sampled_array = np.asarray(sampled_gaps)
    sigmas = np.full(deltas.shape, np.nan, dtype=np.float64)
    for depth in range(cfg.k_max + 1):
        for state in range(states):
            finite = sampled_array[
                np.isfinite(sampled_array[:, depth, state]), depth, state
            ]
            if len(finite) >= 2:
                sigmas[depth, state] = float(np.std(finite, ddof=1))
    multipliers = np.empty(cfg.k_max + 1)
    for k, q in enumerate(true_q):
        occupancy = _occupancy(
            p,
            q,
            supported,
            initial,
            float(getattr(mdp, "gamma")),
            terminal_mask,
        )
        if occupancy is None:
            raise RuntimeError("Exact occupancy is singular or non-finite.")
        critical = (
            occupancy >= cfg.tau_coefficient / states
        ) & ~terminal_mask
        eligible = np.array(
            [critical[s] and pairs[k][s] is not None for s in range(states)]
        )
        weights = np.zeros(states)
        weights[eligible] = occupancy[eligible] / occupancy[critical].sum()
        _, multipliers[k], _ = _loss_and_multiplier(
            np.nan_to_num(deltas[k]),
            np.nan_to_num(sigmas[k]),
            weights,
            cfg.epsilon,
        )
    return ExactEvidenceCurve(
        deltas,
        sigmas,
        multipliers,
        _smallest_near_minimum(multipliers),
        len(sampled_gaps) / cfg.num_exact_replicates,
    )
