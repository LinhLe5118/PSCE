"""Hyperparameter-free posterior over tabular MDPs and behavior policies.

Only pilot transitions, the public dimensions, the public initial-state
distribution and public protocol constants are used. No concentration,
smoothing weight or threshold is chosen by hand:

* Transition and behavior rows are Dirichlet with a symmetric base measure.
  Their single concentration is the empirical-Bayes maximizer of the exact
  Dirichlet-multinomial marginal likelihood of the observed rows.
* Rewards are deterministic functions of a factorization key. The key is the
  coarsest of ``next_state``, ``(state, action)`` and ``(state, action,
  next_state)`` that is consistent with every observed reward (minimum
  description length). An unobserved key is rewarding with probability
  ``theta ~ Beta(k + 1/2, n - k + 1/2)`` (Jeffreys reference prior updated by
  the ``k`` rewarding keys among ``n`` observed keys); a rewarding value is
  drawn from the observed nonzero values, or equals the public reward bound if
  none was observed.
* Posterior draws are conditioned on the public dynamic-range filter that
  every benchmark MDP satisfies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tabular import reference_returns

_LOG_GRID = np.linspace(-14.0, 14.0, 561)  # numerical search domain of log(alpha)


def _rising_log_terms(counts: np.ndarray) -> np.ndarray:
    """Histogram ``h[j] = #{entries with count > j}`` for integer counts."""
    counts = np.asarray(counts, dtype=np.int64).ravel()
    counts = counts[counts > 0]
    if not counts.size:
        return np.zeros(0)
    tally = np.bincount(counts)
    return tally[::-1].cumsum()[::-1][1:].astype(float)


def dirichlet_multinomial_concentration(rows: np.ndarray,
                                        base: np.ndarray | None = None) -> float:
    """Empirical-Bayes total concentration of a Dirichlet with fixed base measure.

    ``base`` defaults to the symmetric (uniform) measure.

    ``rows`` holds one observed count vector per row. The exact log marginal
    likelihood uses ``log Gamma(x+n) - log Gamma(x) = sum_{j<n} log(x+j)``.
    Rows with fewer than two observations carry no information and are ignored.
    Returns ``inf`` (uniform limit) or a value near ``0`` (point-mass limit) when
    the maximizer lies at the edge of the numerical domain.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("Concentration needs a (rows, categories>=2) count matrix")
    rows = rows[rows.sum(1) >= 2]
    if not len(rows):
        return float("nan")
    width = rows.shape[1]
    base = np.full(width, 1.0 / width) if base is None else np.asarray(base, dtype=float)
    if base.shape != (width,) or np.any(base < 0) or not np.isclose(base.sum(), 1.0):
        raise ValueError("Base measure must be a probability vector")
    if np.any(rows[:, base <= 0] > 0):
        raise ValueError("Observed category outside the base-measure support")
    total = _rising_log_terms(rows.sum(1))
    j_total = np.arange(total.size)
    cells = [(base[k], _rising_log_terms(rows[:, k])) for k in np.flatnonzero(base > 0)]
    cells = [(mass, tally, np.arange(tally.size)) for mass, tally in cells if tally.size]

    def log_likelihood(log_alpha: np.ndarray) -> np.ndarray:
        alpha = np.exp(log_alpha)[:, None]
        value = -(total * np.log(alpha + j_total)).sum(1)
        for mass, tally, j in cells:
            value = value + (tally * np.log(alpha * mass + j)).sum(1)
        return value

    values = log_likelihood(_LOG_GRID)
    best = int(values.argmax())
    if best == len(_LOG_GRID) - 1:
        return float("inf")
    low = _LOG_GRID[max(best - 1, 0)]
    high = _LOG_GRID[min(best + 1, len(_LOG_GRID) - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):  # golden-section refinement to machine resolution
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if log_likelihood(np.array([a]))[0] >= log_likelihood(np.array([b]))[0]:
            high = b
        else:
            low = a
    return float(np.exp((low + high) / 2.0))


def _rising_table(values: np.ndarray, maximum: int) -> np.ndarray:
    """``T[i, n] = sum_{j<n} log(values[i] + j)`` for ``n = 0..maximum``."""
    j = np.arange(max(maximum, 1))
    logs = np.log(values[:, None] + j[None, :])
    return np.concatenate([np.zeros((len(values), 1)), np.cumsum(logs, axis=1)], axis=1)[:, :maximum + 1]


def row_log_likelihood(rows: np.ndarray, alpha: float, base: np.ndarray) -> np.ndarray:
    """Dirichlet-multinomial log likelihood of each count row (up to a constant).

    The multinomial coefficient does not depend on ``alpha`` and is omitted, so
    values are comparable across concentrations and mixture components.
    """
    rows = np.asarray(rows, dtype=np.int64)
    support = base > 0
    totals = rows.sum(1)
    maximum = int(rows.max()) if rows.size else 0
    cells = _rising_table(alpha * base[support], maximum)
    per_cell = np.take_along_axis(cells, rows[:, support].T, axis=1).sum(0)
    total = _rising_table(np.array([alpha]), int(totals.max()) if totals.size else 0)[0][totals]
    return per_cell - total


def _golden(objective, low: float, high: float, iterations: int = 60) -> float:
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iterations):
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if objective(a) >= objective(b):
            high = b
        else:
            low = a
    return (low + high) / 2.0


def fit_dirichlet_mixture(rows: np.ndarray, base: np.ndarray | None = None) -> dict:
    """Empirical-Bayes finite mixture of Dirichlet rows, size chosen by BIC.

    Components share the base measure and differ in total concentration, so a
    mixture can represent rows that are nearly deterministic and rows that are
    nearly uniform in the same MDP. Each size ``K`` is fit by EM from several
    deterministic starts; ``K`` grows while the Bayesian information criterion
    improves. Only rows with at least two observations inform the fit.
    Returns weights and log-concentrations (``-inf``/``inf`` mark the limits).
    """
    rows = np.asarray(rows, dtype=np.int64)
    width = rows.shape[1]
    base = np.full(width, 1.0 / width) if base is None else np.asarray(base, dtype=float)
    informative = rows[rows.sum(1) >= 2]
    single = dirichlet_multinomial_concentration(rows, base)
    best = {"weights": np.array([1.0]), "log_alpha": np.array([np.log(single) if single > 0 else -np.inf]),
            "components": 1, "bic": np.inf, "rows": int(len(informative))}
    if len(informative) < 4 or not np.isfinite(single) or single <= 0:
        return best
    low, high = _LOG_GRID[0], _LOG_GRID[-1]

    def loglik(log_alpha: float) -> np.ndarray:
        return row_log_likelihood(informative, float(np.exp(log_alpha)), base)

    count = len(informative)
    ll1 = float(loglik(np.log(single)).sum())
    best["bic"] = -2.0 * ll1 + np.log(count)
    components = 1
    while components < count:
        components += 1
        candidate = None
        for start in range(3):  # deterministic spread of initial concentrations
            spread = np.linspace(-1.0, 1.0, components) * (2.0 + 2.0 * start)
            log_alpha = np.clip(np.log(single) + spread, low, high)
            weights = np.full(components, 1.0 / components)
            previous = -np.inf
            for _ in range(200):
                table = np.stack([loglik(value) for value in log_alpha], axis=1) + np.log(weights)
                peak = table.max(1, keepdims=True)
                total = float((peak[:, 0] + np.log(np.exp(table - peak).sum(1))).sum())
                resp = np.exp(table - peak)
                resp /= resp.sum(1, keepdims=True)
                weights = np.maximum(resp.mean(0), 1e-300)
                for k in range(components):
                    log_alpha[k] = _golden(lambda x: float((resp[:, k] * loglik(x)).sum()), low, high)
                if total - previous <= 1e-9 * (1.0 + abs(total)):
                    break
                previous = total
            if candidate is None or total > candidate[0]:
                candidate = (total, weights.copy(), log_alpha.copy())
        bic = -2.0 * candidate[0] + (2 * components - 1) * np.log(count)
        if bic >= best["bic"]:
            break
        order = np.argsort(candidate[2])
        best = {"weights": candidate[1][order], "log_alpha": candidate[2][order],
                "components": components, "bic": float(bic), "rows": int(count)}
    return best


def mixture_responsibilities(rows: np.ndarray, mixture: dict, base: np.ndarray) -> np.ndarray:
    """Posterior component probabilities of every row (uninformative rows -> weights)."""
    rows = np.asarray(rows, dtype=np.int64)
    weights, log_alpha = mixture["weights"], mixture["log_alpha"]
    if len(weights) == 1:
        return np.ones((len(rows), 1))
    table = np.stack([row_log_likelihood(rows, float(np.exp(a)), base) for a in log_alpha], axis=1)
    table = table + np.log(weights)
    table -= table.max(1, keepdims=True)
    resp = np.exp(table)
    return resp / resp.sum(1, keepdims=True)


def sample_dirichlet(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """Dirichlet draws on the last axis, exact in the point-mass limit.

    Rows whose gamma variates all underflow are the ``alpha -> 0`` limit: one
    category, chosen with probability proportional to ``alpha``.
    """
    alpha = np.asarray(alpha, dtype=float)
    gammas = rng.standard_gamma(alpha)
    totals = gammas.sum(-1, keepdims=True)
    empty = totals[..., 0] <= 0
    if empty.any():
        weights = alpha[empty] / alpha[empty].sum(-1, keepdims=True)
        cdf = weights.cumsum(-1)
        pick = (rng.random(cdf.shape[:-1] + (1,)) > cdf).sum(-1)
        pick = np.minimum(pick, alpha.shape[-1] - 1)
        replacement = np.zeros_like(weights)
        np.put_along_axis(replacement, pick[..., None], 1.0, axis=-1)
        gammas[empty] = replacement
        totals = gammas.sum(-1, keepdims=True)
    return gammas / totals


@dataclass(frozen=True)
class PilotStatistics:
    num_states: int
    num_actions: int
    gamma: float
    rho: np.ndarray
    transition_counts: np.ndarray   # (S, A, S), terminal destinations included
    action_counts: np.ndarray       # (S, A)
    terminal: np.ndarray            # (S,) observed terminal states
    reward_key: str
    reward_table: dict              # observed key -> reward
    reward_bound: float


REWARD_KEYS = ("next_state", "state_action", "triple")


def _keys(kind: str, s, a, n):
    if kind == "next_state":
        return list(zip(np.asarray(n).tolist()))
    if kind == "state_action":
        return list(zip(np.asarray(s).tolist(), np.asarray(a).tolist()))
    return list(zip(np.asarray(s).tolist(), np.asarray(a).tolist(), np.asarray(n).tolist()))


def pilot_statistics(arrays: dict, rho, *, num_states: int, num_actions: int,
                     gamma: float, reward_bound: float = 1.0) -> PilotStatistics:
    """Summarize one pilot buffer into posterior sufficient statistics."""
    s = np.asarray(arrays["state"], dtype=np.int64)
    a = np.asarray(arrays["action"], dtype=np.int64)
    n = np.asarray(arrays["next_state"], dtype=np.int64)
    r = np.asarray(arrays["reward"], dtype=float)
    t = np.asarray(arrays["terminal"], dtype=bool)
    if not (len(s) == len(a) == len(n) == len(r) == len(t)) or not len(s):
        raise ValueError("Pilot arrays must be aligned and nonempty")
    if np.any(r < 0) or np.any(r > reward_bound):
        raise ValueError("Pilot reward outside the public bound")
    transition_counts = np.zeros((num_states, num_actions, num_states))
    np.add.at(transition_counts, (s, a, n), 1.0)
    action_counts = transition_counts.sum(-1)
    terminal = np.zeros(num_states, dtype=bool)
    terminal[n[t]] = True
    chosen, table = "triple", {}
    for kind in REWARD_KEYS:
        observed: dict = {}
        consistent = True
        for key, value in zip(_keys(kind, s, a, n), r.tolist()):
            if observed.setdefault(key, value) != value:
                consistent = False
                break
        if consistent:
            chosen, table = kind, observed
            break
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (num_states,) or np.any(rho < 0) or not np.isclose(rho.sum(), 1.0):
        raise ValueError("Invalid initial-state distribution")
    return PilotStatistics(num_states, num_actions, float(gamma), rho, transition_counts,
                           action_counts, terminal, chosen, table, float(reward_bound))


REWARD_PRIORS = ("exchangeable", "unseen_zero")
TRANSITION_BASES = ("uniform", "pooled")
ROW_PRIORS = ("single", "mixture")


@dataclass(frozen=True)
class Posterior:
    stats: PilotStatistics
    transition_concentration: float
    behavior_concentration: float
    reward_rate: tuple[float, float]   # Beta parameters for unobserved keys
    reward_values: np.ndarray
    transition_base: np.ndarray        # (S,) Dirichlet base measure of next states
    reward_prior: str = "exchangeable"
    transition_base_kind: str = "uniform"
    behavior_mixture: dict | None = None      # weights, log_alpha, responsibilities
    transition_mixture: dict | None = None

    def describe(self) -> dict:
        mixtures = {name: {"components": int(value["components"]),
                           "weights": np.asarray(value["weights"]).tolist(),
                           "log_alpha": np.asarray(value["log_alpha"]).tolist()}
                    for name, value in (("behavior_mixture", self.behavior_mixture),
                                        ("transition_mixture", self.transition_mixture)) if value}
        return {**mixtures, "reward_prior": self.reward_prior,
                "transition_base": self.transition_base_kind,
                "transition_concentration": self.transition_concentration,
                "behavior_concentration": self.behavior_concentration,
                "reward_key": self.stats.reward_key,
                "reward_rate_beta": list(self.reward_rate),
                "observed_reward_keys": len(self.stats.reward_table)}


def fit_posterior(stats: PilotStatistics, *, reward_prior: str = "exchangeable",
                  transition_base: str = "uniform", behavior_prior: str = "single",
                  transition_prior: str = "single") -> Posterior:
    """Empirical-Bayes posterior; every constant is derived from the pilot.

    ``reward_prior="unseen_zero"`` assigns zero reward to every unobserved key
    (the support-conservative reading the registered learners also use).
    ``transition_base="pooled"`` replaces the uniform next-state base measure
    by the pooled empirical next-state distribution of the pilot.
    """
    if (reward_prior not in REWARD_PRIORS or transition_base not in TRANSITION_BASES
            or behavior_prior not in ROW_PRIORS or transition_prior not in ROW_PRIORS):
        raise ValueError("Unknown posterior structure")
    S, A = stats.num_states, stats.num_actions
    if transition_base == "pooled":
        pooled = stats.transition_counts.sum((0, 1))
        base = pooled / pooled.sum()
    else:
        base = np.full(S, 1.0 / S)
    alpha_p = dirichlet_multinomial_concentration(stats.transition_counts.reshape(-1, S), base)
    alpha_b = dirichlet_multinomial_concentration(stats.action_counts[~stats.terminal])
    values = np.array(list(stats.reward_table.values()), dtype=float)
    rewarding = int(np.count_nonzero(values > 0))
    nonzero = values[values > 0]
    return Posterior(
        stats=stats,
        transition_concentration=alpha_p,
        behavior_concentration=alpha_b,
        reward_rate=(rewarding + 0.5, len(values) - rewarding + 0.5),
        reward_values=nonzero if nonzero.size else np.array([stats.reward_bound]),
        transition_base=base,
        reward_prior=reward_prior,
        transition_base_kind=transition_base,
        behavior_mixture=_mixture(stats.action_counts, np.full(A, 1.0 / A), ~stats.terminal)
        if behavior_prior == "mixture" else None,
        transition_mixture=_mixture(stats.transition_counts.reshape(-1, S), base, None)
        if transition_prior == "mixture" else None,
    )


def _mixture(rows: np.ndarray, base: np.ndarray, keep: np.ndarray | None) -> dict:
    fit_rows = rows if keep is None else rows[keep]
    mixture = fit_dirichlet_mixture(fit_rows, base)
    mixture["responsibilities"] = mixture_responsibilities(rows, mixture, base)
    return mixture


def _row_prior(rng: np.random.Generator, count: int, mixture: dict, base: np.ndarray) -> np.ndarray:
    """Per-draw, per-row Dirichlet parameters under a fitted mixture prior."""
    resp = mixture["responsibilities"]
    cdf = np.cumsum(resp, axis=1)
    u = rng.random((count, len(resp), 1))
    component = np.minimum((u > cdf[None]).sum(-1), resp.shape[1] - 1)
    alpha = np.exp(np.asarray(mixture["log_alpha"]))[component]          # (count, rows)
    alpha = np.where(np.isfinite(alpha), alpha, 1e12)
    return alpha[..., None] * base[None, None, :]


def _prior(concentration: float, base: np.ndarray) -> np.ndarray:
    """Dirichlet prior parameters ``alpha * base``.

    An unidentified concentration (no row with two observations) falls back to
    the Jeffreys prior ``1/2`` per supported category; the uniform limit
    ``alpha = inf`` is represented by a numerically flat Dirichlet.
    """
    if np.isnan(concentration):
        return np.where(base > 0, 0.5, 0.0)
    if concentration == float("inf"):
        return base * 1e12
    return concentration * base


@dataclass(frozen=True)
class MDPSample:
    P: np.ndarray        # (B, S, A, S)
    R: np.ndarray        # (B, S, A, S)
    beta: np.ndarray     # (B, S, A)
    terminal: np.ndarray  # (S,)
    rho: np.ndarray      # (S,)
    gamma: float

    @property
    def expected_rewards(self) -> np.ndarray:
        return np.einsum("bsan,bsan->bsa", self.P, self.R)


def _draw(posterior: Posterior, rng: np.random.Generator, count: int):
    stats = posterior.stats
    S, A = stats.num_states, stats.num_actions
    if posterior.transition_mixture is None:
        a_p = _prior(posterior.transition_concentration, posterior.transition_base)
        P = sample_dirichlet(rng, np.broadcast_to(stats.transition_counts + a_p, (count, S, A, S)))
    else:
        a_p = _row_prior(rng, count, posterior.transition_mixture, posterior.transition_base)
        P = sample_dirichlet(rng, stats.transition_counts[None] + a_p.reshape(count, S, A, S))
    if posterior.behavior_mixture is None:
        a_b = _prior(posterior.behavior_concentration, np.full(A, 1.0 / A))
        beta = sample_dirichlet(rng, np.broadcast_to(stats.action_counts + a_b, (count, S, A)))
    else:
        a_b = _row_prior(rng, count, posterior.behavior_mixture, np.full(A, 1.0 / A))
        beta = sample_dirichlet(rng, stats.action_counts[None] + a_b)
    terminal = stats.terminal
    P[:, terminal] = 0.0
    P[:, terminal, :, terminal] = 1.0
    if stats.reward_key == "next_state":
        shape, index = (S,), lambda key: key
    elif stats.reward_key == "state_action":
        shape, index = (S, A), lambda key: key
    else:
        shape, index = (S, A, S), lambda key: key
    theta = rng.beta(*posterior.reward_rate, size=count)
    if posterior.reward_prior == "unseen_zero":
        theta = np.zeros(count)
    draws = rng.random((count,) + shape) < theta.reshape((count,) + (1,) * len(shape))
    values = rng.choice(posterior.reward_values, size=(count,) + shape)
    table = np.where(draws, values, 0.0)
    for key, value in stats.reward_table.items():
        table[(slice(None),) + tuple(index(key))] = value
    if stats.reward_key == "next_state":
        R = np.broadcast_to(table[:, None, None, :], (count, S, A, S)).copy()
    elif stats.reward_key == "state_action":
        R = np.broadcast_to(table[..., None], (count, S, A, S)).copy()
    else:
        R = table
    R[:, terminal] = 0.0
    return P, R, beta


def sample_mdps(posterior: Posterior, rng: np.random.Generator, count: int, *,
                minimum_dynamic_range: float, max_rounds: int = 64) -> tuple[MDPSample, dict]:
    """Draw ``count`` posterior MDPs that satisfy the public range filter.

    Rejection sampling implements exact conditioning on the filter. If the
    acceptance rate is so small that ``max_rounds`` batches of ``count``
    proposals yield fewer draws, the accepted draws are returned and the
    shortfall is reported; callers must record it.
    """
    stats = posterior.stats
    kept = []
    proposals = accepted = 0
    no_reward = (posterior.reward_prior == "unseen_zero"
                 and not any(value > 0 for value in stats.reward_table.values()))
    if no_reward and minimum_dynamic_range > 0:
        # Every draw has identically zero reward, so J* = J_rand and the
        # filter rejects it with probability one; sampling would be futile.
        empty = MDPSample(np.empty((0,) + stats.transition_counts.shape), np.empty((0,) + stats.transition_counts.shape),
                          np.empty((0,) + stats.action_counts.shape), stats.terminal.copy(), stats.rho.copy(), stats.gamma)
        return empty, {"requested": count, "returned": 0, "proposals": 0, "acceptance_rate": 0.0,
                       "rejected_with_certainty": True}
    for _ in range(max_rounds):
        P, R, beta = _draw(posterior, rng, count)
        r = np.einsum("bsan,bsan->bsa", P, R)
        j_star, j_rand = reference_returns(P, r, np.broadcast_to(stats.rho, (count, stats.num_states)),
                                           stats.gamma)
        ok = (j_star - j_rand) >= minimum_dynamic_range
        proposals += count
        accepted += int(ok.sum())
        kept.append((P[ok], R[ok], beta[ok]))
        if sum(len(item[0]) for item in kept) >= count:
            break
    P = np.concatenate([item[0] for item in kept])[:count]
    R = np.concatenate([item[1] for item in kept])[:count]
    beta = np.concatenate([item[2] for item in kept])[:count]
    report = {"requested": count, "returned": int(len(P)), "proposals": proposals,
              "acceptance_rate": accepted / proposals if proposals else 0.0}
    return MDPSample(P, R, beta, stats.terminal.copy(), stats.rho.copy(), stats.gamma), report


def simulate_streams(sample: MDPSample, rng: np.random.Generator, *, streams: int,
                     length: int, horizon: int) -> dict[str, np.ndarray]:
    """Natural trajectory streams under each sampled behavior and MDP.

    Returns arrays of shape ``(B, streams, length)`` with the generator's
    semantics: restart from ``rho`` after termination or after ``horizon``
    steps; transitions into terminal states are flagged terminal.
    """
    B, S, A, _ = sample.P.shape
    rows = B * streams
    owner = np.repeat(np.arange(B), streams)
    rho_cdf = np.cumsum(sample.rho)
    beta_cdf = np.cumsum(sample.beta, -1)
    p_cdf = np.cumsum(sample.P, -1)
    out = {name: np.empty((rows, length), dtype=dtype) for name, dtype in
           (("state", np.int64), ("action", np.int64), ("reward", float),
            ("next_state", np.int64), ("terminal", bool), ("truncated", bool),
            ("trajectory_id", np.int64), ("timestep", np.int64))}

    def draw(cdf: np.ndarray) -> np.ndarray:
        u = rng.random(cdf.shape[0])[:, None] * cdf[:, -1:]
        return np.minimum((u >= cdf).sum(-1), cdf.shape[-1] - 1)

    state = draw(np.broadcast_to(rho_cdf, (rows, S)))
    clock = np.zeros(rows, dtype=np.int64)
    episode = np.zeros(rows, dtype=np.int64)
    for step in range(length):
        action = draw(beta_cdf[owner, state])
        next_state = draw(p_cdf[owner, state, action])
        reward = sample.R[owner, state, action, next_state]
        terminal = sample.terminal[next_state]
        out["state"][:, step] = state
        out["action"][:, step] = action
        out["reward"][:, step] = reward
        out["next_state"][:, step] = next_state
        out["terminal"][:, step] = terminal
        out["truncated"][:, step] = (clock + 1 >= horizon) & ~terminal
        out["trajectory_id"][:, step] = episode
        out["timestep"][:, step] = clock
        restart = terminal | (clock + 1 >= horizon)
        clock = np.where(restart, 0, clock + 1)
        episode = episode + restart
        fresh = draw(np.broadcast_to(rho_cdf, (rows, S)))
        state = np.where(restart, fresh, next_state)
    return {name: value.reshape(B, streams, length) for name, value in out.items()}
