"""OEH-H: offline effective horizon of an MDP under a behavior policy.

Effective horizon (Laidlaw et al., 2023) scores an MDP by
``H = min_k [k + log_A m_k]``. Here ``k`` is the number of value-iteration
backups from the random-policy Q-function after which greedy action selection
is optimal, and ``m_k`` counts random rollouts per leaf. In the offline
setting the agent cannot choose which actions are tried; the logged behavior
does. OEH-H keeps the same backup sequence, but starts from the behavior leaf
``Q^1 = Q^beta`` and replaces the online sampling cost by the cost of
identifying each return-critical action ranking from behavior data.

``N_k = max over critical (s, a') of [Var Q_hat(s,a*) + Var Q_hat(s,a')] / gap^2``

``Var Q_hat(s,a) = sum_{(x,b)} W_k[(s,a),(x,b)]^2 sigma_k^2(x,b) / d_beta(x,b)``

``W_k`` is the discounted visitation of the depth-``k`` lookahead tree,
``sigma_k^2`` the one-step Bellman-target variance and ``d_beta`` the
state-action frequency of logged natural trajectories. The formula is the
first-order (delta-method) variance of the model-based estimate at unit
signal-to-noise ratio. EH's gap bound uses the same convention, so no
confidence constant is introduced. A competitor is *critical* when replacing
the greedy action by it at that state alone drops the protocol-normalized
return below ``tau``; non-critical misrankings cannot violate the target.
``OEH_H = min_k log_A N_k`` over depths whose greedy policy reaches ``tau``.

The model-free lookahead cost ``log_A 1/kappa_k`` reduces exactly to EH's
``k`` term under a uniform behavior policy (see :func:`lookahead_cost`).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .tabular import optimal_values, policy_values, reference_returns

GAP_TOLERANCE = 1e-10  # numerical tie tolerance, as in the published EH code
FORMS = ("critical", "risk", "coverage_risk", "missing_mass_risk")

try:  # SciPy is present in every analysis environment; math.erfc is the fallback.
    from scipy.special import erfc as _erfc
except ImportError:  # pragma: no cover
    _erfc = np.vectorize(math.erfc)


def _misrank_probability(gap: np.ndarray, variance: np.ndarray, samples: float) -> np.ndarray:
    """Gaussian (CLT) probability that a noisy comparison reverses ``gap > 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.isfinite(variance) & (variance > 0), gap / np.sqrt(variance / samples),
                     np.where(np.isfinite(variance) & (gap > 0), np.inf, 0.0))
    return 0.5 * _erfc(z / math.sqrt(2.0))


def missing_mass_requirement(visits: np.ndarray, coverage: np.ndarray, slack: float) -> float:
    """Smallest N with expected target-occupancy mass on unobserved pairs <= slack.

    ``visits`` is the target policy's normalized state occupancy and
    ``coverage`` the logged frequency of its chosen pair at each state; a pair
    with frequency ``d`` is still unobserved after ``N`` transitions with
    probability ``exp(-N d)`` (independent-draw approximation).
    """
    need = visits > 0
    if np.any(need & (coverage <= 0)):
        return math.inf
    mass = lambda n: float((visits[need] * np.exp(-n * coverage[need])).sum())  # noqa: E731
    if mass(1.0) <= slack:
        return 1.0
    if slack <= 0:
        return math.inf
    low, high = 0.0, 200.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if mass(math.exp(middle)) <= slack:
            high = middle
        else:
            low = middle
    return math.exp(high)


def _risk_requirement(gap, variance, weight, slack) -> float:
    """Smallest N whose expected normalized regret (noise only) is within ``slack``."""
    loss = lambda n: float((weight * _misrank_probability(gap, variance, n)).sum())  # noqa: E731
    if loss(1.0) <= slack:
        return 1.0
    if loss(math.exp(200.0)) > slack:
        return math.inf
    low, high = 0.0, 200.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if loss(math.exp(middle)) <= slack:
            high = middle
        else:
            low = middle
    return math.exp(high)


LOG_GRID = np.arange(0.0, 200.0 + 1e-9, 0.25)   # numerical resolution of log N


def _coverage_risk_requirement(gap, variance, advantage, compete, alternatives, visits, scale_factor,
                               d_chosen, tree, d_flat, slack) -> float:
    """Requirement when unobserved pairs, not only noise, can corrupt a decision.

    At state ``s`` the chosen action is unusable with probability
    ``u_s = min(1, exp(-N d(s,a*)) + tree_s . exp(-N d))``; the learner then
    picks among the other supported actions (mean advantage ``Y_s``). Otherwise
    noise reverses the ranking with CLT probability (``X_s``). The expected
    normalized regret is ``sum_s visits_s [(1-u_s) X_s + u_s Y_s] / scale``.
    The requirement uses the nonincreasing envelope ``sup_{N' >= N}`` so that a
    budget is accepted only if every larger budget is accepted too.
    """
    n = np.exp(LOG_GRID)
    with np.errstate(over="ignore", under="ignore"):
        unseen = np.minimum(1.0, np.exp(-n[:, None] * d_chosen[None, :])
                            + np.exp(-n[:, None] * d_flat[None, :]) @ tree.T)        # (grid, S)
    y = np.where(alternatives.any(1), (advantage * alternatives).sum(1)
                 / np.maximum(alternatives.sum(1), 1), 0.0)                          # (S,)
    x = np.stack([(np.where(compete, advantage, 0.0) * _misrank_probability(gap, variance, value)).sum(1)
                  for value in n])                                                    # (grid, S)
    loss = ((1.0 - unseen) * x + unseen * y[None, :]) @ visits / scale_factor
    envelope = np.maximum.accumulate(loss[::-1])[::-1]
    ok = np.flatnonzero(envelope <= slack)
    return math.inf if not len(ok) else float(n[ok[0]])


@dataclass(frozen=True)
class HorizonResult:
    value: float                 # OEH-H in log_A units; inf if unidentifiable
    depth: int | None            # minimizing QVI index k
    requirement: float           # N_k in transitions (inf if unidentifiable)
    statistical: float           # log_A of variance-to-gap ratio at unit coverage
    coverage: float              # log_A of the variance-weighted inverse coverage
    solvable_depths: int
    status: str

    def as_dict(self) -> dict:
        return {"oeh_h": self.value, "oeh_h_depth": self.depth,
                "oeh_h_requirement": self.requirement,
                "oeh_h_statistical": self.statistical,
                "oeh_h_coverage": self.coverage,
                "oeh_h_solvable_depths": self.solvable_depths,
                "oeh_h_status": self.status}


def stream_visitation(P: np.ndarray, beta: np.ndarray, rho: np.ndarray,
                      terminal: np.ndarray, horizon: int) -> np.ndarray:
    """Long-run state-action frequency of restarted natural trajectories.

    Episodes start from ``rho``, follow ``beta`` and end at a terminal state or
    after ``horizon`` transitions. By the renewal-reward theorem the fraction
    of logged transitions at ``(s, a)`` is expected visits per episode divided
    by expected episode length.
    """
    live = (~terminal).astype(float)
    mu = rho * live
    visits = np.zeros(beta.shape)
    for _ in range(horizon):
        step = mu[:, None] * beta
        visits += step
        mu = np.einsum("sa,san->n", step, P) * live
    return visits / visits.sum()


def lookahead_cost(P: np.ndarray, beta: np.ndarray, policies: list[np.ndarray],
                   start_action: np.ndarray | None = None) -> np.ndarray:
    """``log_A 1/kappa_k(s, a)`` for the model-free (rollout) estimator.

    ``kappa_k(s,a) = beta(a|s) E[prod_t beta(pi_{k-t}(s_t)|s_t)]`` is the
    probability that one logged trajectory from ``s`` executes ``a`` followed
    by the ``k-1`` lookahead actions ``policies`` (deterministic indices, the
    first applied first). Under a uniform behavior it equals ``A^{-k}``.
    """
    S, A = beta.shape
    survival = np.ones(S)
    for policy in reversed(policies):
        follow = beta[np.arange(S), policy]
        survival = follow * np.einsum("sn,n->s", P[np.arange(S), policy], survival)
    kappa = beta * np.einsum("san,n->sa", P, survival)
    with np.errstate(divide="ignore"):
        return -np.log(kappa) / math.log(A)


def _deviation_returns(P, r, rho, gamma, policy, values, q):
    """Exact return of every one-state deviation from a deterministic policy."""
    S = len(policy)
    rows = np.arange(S)
    p_pi = P[rows, policy]
    M = np.linalg.inv(np.eye(S) - gamma * p_pi)
    advantage = q - values[:, None]
    occupancy = rho @ M                                   # unnormalized, per state
    delta_p = P - p_pi[:, None, :]                        # (S, A, S)
    denominator = 1.0 - gamma * np.einsum("san,ns->sa", delta_p, M)
    return float(rho @ values) + occupancy[:, None] * advantage / denominator


def offline_effective_horizon(P: np.ndarray, R: np.ndarray, beta: np.ndarray,
                              rho: np.ndarray, gamma: float, *, terminal: np.ndarray,
                              tau: float, horizon: int, max_depth: int | None = None,
                              form: str = "critical", depth_rule: str = "min") -> HorizonResult:
    """Compute OEH-H for one MDP ``(P, R)`` and behavior ``beta``.

    ``R`` has shape ``(S, A, S)``. ``tau`` and ``horizon`` are the public
    protocol threshold and data-collection horizon; ``max_depth`` defaults to
    ``horizon``.

    ``form="critical"`` takes the worst critical ranking (EH-style worst case).
    ``form="risk"`` uses the performance-difference lemma: the expected
    normalized regret ``sum_s d^{pi_k}(s) sum_a' A_k(s,a') P(misrank)`` /
    ``((1-gamma) Range)`` with CLT misranking probabilities must not exceed the
    slack ``NR(pi_k) - tau``; ``N_k`` is the smallest such sample size.
    """
    if form not in FORMS or depth_rule not in ("min", "converged"):
        raise ValueError(f"Unknown OEH-H form or depth rule: {form}/{depth_rule}")
    S, A, _ = P.shape
    max_depth = horizon if max_depth is None else int(max_depth)
    r = np.einsum("san,san->sa", P, R)
    j_star, j_rand = (float(x[0]) for x in reference_returns(P[None], r[None], rho[None], gamma))
    scale = j_star - j_rand
    if not scale > 0:
        # No reward signal: the protocol-normalized return is undefined.
        return HorizonResult(math.inf, None, math.inf, math.inf, math.inf, 0, "degenerate_range")
    d_beta = stream_visitation(P, beta, rho, terminal, horizon)
    rows = np.arange(S)
    live = ~terminal
    # The learnable lookahead is restricted to the behavior's support: an action
    # with zero behavior probability never appears in the data.
    support = (beta > 0) & live[:, None]
    support[~support.any(1), 0] = True
    v_beta = policy_values(P, r, beta, gamma)
    q = r + gamma * np.einsum("san,n->sa", P, v_beta)
    # Lookahead-tree visitation W_1 = (I - gamma P_beta)^-1 on state-action pairs.
    transfer_beta = np.einsum("san,nb->sanb", P, beta).reshape(S * A, S * A)
    W = np.linalg.inv(np.eye(S * A) - gamma * transfer_beta)
    continuation = v_beta
    best = HorizonResult(math.inf, None, math.inf, math.inf, math.inf, 0, "not_tau_solvable")
    solvable = 0
    for depth in range(1, max_depth + 1):
        policy = np.where(support, q, -np.inf).argmax(1)
        values = policy_values(P, r, np.eye(A)[policy], gamma)
        nr = (float(rho @ values) - j_rand) / scale
        if depth_rule == "converged" and depth < max_depth:
            pass  # only the converged lookahead is scored
        elif nr >= tau:
            solvable += 1
            target = R + gamma * continuation[None, None, :]
            mean = np.einsum("san,san->sa", P, target)
            sigma2 = np.einsum("san,san->sa", P, (target - mean[..., None]) ** 2)
            weight = (W ** 2) * sigma2.reshape(1, -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                unit = weight.sum(1).reshape(S, A)               # variance at d = 1
                scaled = np.where(weight > 0, weight / d_beta.reshape(1, -1), 0.0)
                inverse = np.where((weight > 0) & (d_beta.reshape(1, -1) <= 0), np.inf, scaled)
                variance = inverse.sum(1).reshape(S, A)
            q_pi = r + gamma * np.einsum("san,n->sa", P, values)
            deviations = (_deviation_returns(P, r, rho, gamma, policy, values, q_pi) - j_rand) / scale
            occupancy = rho @ np.linalg.inv(np.eye(S) - gamma * P[rows, policy])
            gap = q[rows, policy][:, None] - q
            critical = ((deviations < tau) & (gap > GAP_TOLERANCE) & support
                        & (occupancy[:, None] > 0) & live[:, None])
            critical[rows, policy] = False
            if form in ("risk", "coverage_risk"):
                visits = (1.0 - gamma) * occupancy
                loss = np.maximum(q_pi[rows, policy][:, None] - q_pi, 0.0)
                compete = support & live[:, None] & (visits[:, None] > 0) & (loss > 0)
                compete[rows, policy] = False
                weight = np.where(compete, visits[:, None] * loss / ((1.0 - gamma) * scale), 0.0)
                pair = variance[rows, policy][:, None] + variance
                unit_pair = unit[rows, policy][:, None] + unit
                gap_c = np.where(compete, np.maximum(gap, 0.0), 0.0)
                statistical = _risk_requirement(gap_c, np.where(compete, unit_pair, 0.0), weight, nr - tau)
                if form == "coverage_risk":
                    chosen_rows = rows * A + policy
                    # Downstream lookahead mass; terminal states need no data.
                    tree = (1.0 - gamma) * W[chosen_rows]
                    tree[rows, chosen_rows] -= (1.0 - gamma)          # the chosen pair enters via d_chosen
                    needs_data = np.repeat(live, A)
                    tree = np.where(needs_data[None, :], np.maximum(tree, 0.0), 0.0)
                    alternatives = support & live[:, None]
                    alternatives[rows, policy] = False
                    requirement = _coverage_risk_requirement(
                        gap_c, np.where(compete, pair, 0.0), loss, compete, alternatives,
                        np.where(live, visits, 0.0), (1.0 - gamma) * scale, d_beta[rows, policy],
                        tree, np.where(needs_data, d_beta.reshape(-1), np.inf), nr - tau)
                else:
                    requirement = _risk_requirement(gap_c, np.where(compete, pair, 0.0), weight, nr - tau)
                if form == "missing_mass_risk":
                    chosen_cover = np.where(live, d_beta[rows, policy], 1.0)
                    requirement = requirement + missing_mass_requirement(
                        np.where(live, visits, 0.0), chosen_cover, nr - tau)
            elif critical.any():
                chosen = variance[rows, policy][:, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(critical, (chosen + variance) / gap ** 2, -np.inf)
                    unit_ratio = np.where(critical, (unit[rows, policy][:, None] + unit) / gap ** 2, -np.inf)
                flat = int(np.argmax(ratio))
                requirement = float(ratio.flat[flat])
                statistical = float(unit_ratio.flat[flat])
            else:
                requirement = statistical = 1.0  # no critical ranking: one sample suffices
            value = math.log(requirement, A) if requirement > 0 else -math.inf
            if value < best.value:
                stat_term = math.log(statistical, A) if 0 < statistical < math.inf else math.inf
                best = HorizonResult(value, depth, requirement, stat_term,
                                     value - stat_term if math.isfinite(stat_term) else math.inf,
                                     solvable, "identified" if math.isfinite(value) else "coverage_trap")
        # Next backup: Q^{k+1} = r + gamma P V^k with V^k = max_a Q^k.
        greedy = np.where(support, q, -np.inf).argmax(1)
        continuation = np.where(live, q[rows, greedy], 0.0)
        select = np.zeros((S, S * A))
        select[rows, rows * A + greedy] = 1.0
        W = np.eye(S * A) + gamma * (P.reshape(S * A, S) @ (select @ W))
        q = r + gamma * np.einsum("san,n->sa", P, continuation)
    status = best.status
    if status == "not_tau_solvable" and bool((~(beta > 0) & live[:, None]).any()):
        status = "coverage_trap"   # needed actions lie outside the behavior support
    return HorizonResult(best.value, best.depth, best.requirement, best.statistical,
                         best.coverage, solvable, status)


def stationary_eh(P: np.ndarray, R: np.ndarray, gamma: float, *, max_depth: int,
                  terminal: np.ndarray | None = None) -> tuple[float, int | None]:
    """Gap-bound effective horizon ``min_k k + 2 log_A(1/gap_k)`` (stationary form).

    Follows the published definition with a uniform-random leaf: depth ``k``
    is admissible when every greedy action of ``Q^k`` is optimal at every
    nonterminal state; ``gap_k`` is the smallest greedy/non-greedy separation.
    """
    S, A, _ = P.shape
    terminal = np.zeros(S, dtype=bool) if terminal is None else terminal
    live = ~terminal
    r = np.einsum("san,san->sa", P, R)
    uniform = np.full((S, A), 1.0 / A)
    v_star, _ = optimal_values(P, r, gamma)
    q_star = r + gamma * np.einsum("san,n->sa", P, v_star)
    optimal = q_star >= q_star.max(1, keepdims=True) - GAP_TOLERANCE
    q = r + gamma * np.einsum("san,n->sa", P, policy_values(P, r, uniform, gamma))
    best, best_k = math.inf, None
    for depth in range(1, max_depth + 1):
        greedy = q >= q.max(1, keepdims=True) - GAP_TOLERANCE
        if not np.any(greedy[live] & ~optimal[live]):
            second = np.where(~greedy, q, -np.inf).max(1)
            gaps = (q.max(1) - second)[live & np.isfinite(second)]
            if gaps.size and gaps.min() > 0:
                value = depth - 2.0 * math.log(float(gaps.min()), A)
                if value < best:
                    best, best_k = value, depth
        q = r + gamma * np.einsum("san,n->sa", P, np.where(live, q.max(1), 0.0))
    return best, best_k


def offline_theory_measures(P: np.ndarray, R: np.ndarray, beta: np.ndarray, rho: np.ndarray,
                            gamma: float, *, terminal: np.ndarray, horizon: int) -> dict:
    """Oracle offline-RL difficulty quantities from the sample-complexity literature.

    * single-policy concentrability ``C* = max d^{pi*}(s,a) / d_beta(s,a)``
      (Rashidinejad et al., 2021) and its clipped form with ``min(d^{pi*}, 1/S)``
      in the numerator (Li et al., 2022);
    * the intrinsic, instance-dependent bound of Yin & Wang (2021) in discounted
      form, ``sum_{s,a} d^{pi*}(s,a) sqrt(Var_P[R + gamma V*](s,a) / d_beta(s,a)) / (1-gamma)``
      per ``sqrt(N)``.
    ``d_beta`` is the logged-stream visitation used by OEH-H. Values are natural
    logarithms; ``inf`` marks an optimal pair outside the behavior's support.
    """
    S, A, _ = P.shape
    r = np.einsum("san,san->sa", P, R)
    v_star, best = optimal_values(P, r, gamma)
    rows = np.arange(S)
    occupancy = rho @ np.linalg.inv(np.eye(S) - gamma * P[rows, best]) * (1.0 - gamma)
    d_star = np.zeros((S, A))
    d_star[rows, best] = occupancy
    d_star[terminal] = 0.0
    d_beta = stream_visitation(P, beta, rho, terminal, horizon)
    target = R + gamma * np.where(terminal, 0.0, v_star)[None, None, :]
    mean = np.einsum("san,san->sa", P, target)
    variance = np.einsum("san,san->sa", P, (target - mean[..., None]) ** 2)
    need = d_star > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(need, d_star / d_beta, 0.0)
        clipped = np.where(need, np.minimum(d_star, 1.0 / S) / d_beta, 0.0)
        bound = np.where(need, d_star * np.sqrt(variance / d_beta), 0.0)
    missing = bool(np.any(need & (d_beta <= 0)))
    log = lambda x: math.inf if missing or not np.isfinite(x) else math.log(max(x, 1e-300))  # noqa: E731
    return {"log_concentrability": log(float(ratio.max())),
            "log_clipped_concentrability": log(float(clipped.max())),
            "log_intrinsic_bound": log(float(bound.sum()) / (1.0 - gamma))}


def asymptotic_ceilings(P: np.ndarray, R: np.ndarray, beta: np.ndarray, rho: np.ndarray, gamma: float,
                        *, supported: np.ndarray, terminal: np.ndarray) -> dict:
    """Protocol-normalized returns that learners approach with unlimited data.

    ``behavior``: the behavior policy itself (the behavior-cloning ceiling).
    ``support``: the best policy using only ``supported`` actions (the ceiling of
    support-conservative value iteration). ``nan`` when the range is degenerate.
    """
    S, A, _ = P.shape
    r = np.einsum("san,san->sa", P, R)
    j_star, j_rand = (float(x[0]) for x in reference_returns(P[None], r[None], rho[None], gamma))
    scale = j_star - j_rand
    if not scale > 0:
        return {"behavior_ceiling": math.nan, "support_ceiling": math.nan}
    mask = supported & ~terminal[:, None]
    mask[~mask.any(1), 0] = True
    values = np.zeros(S)
    for _ in range(10_000):
        q = r + gamma * np.einsum("san,n->sa", P, values)
        updated = np.where(mask, q, -np.inf).max(1)
        if np.max(np.abs(updated - values)) <= 1e-12:
            values = updated
            break
        values = updated
    policy = np.where(mask, r + gamma * np.einsum("san,n->sa", P, values), -np.inf).argmax(1)
    support_value = float(rho @ policy_values(P, r, np.eye(A)[policy], gamma))
    behavior_value = float(rho @ policy_values(P, r, beta, gamma))
    return {"behavior_ceiling": (behavior_value - j_rand) / scale,
            "support_ceiling": (support_value - j_rand) / scale}


def risk_profile(P: np.ndarray, R: np.ndarray, beta: np.ndarray, rho: np.ndarray, gamma: float, *,
                 terminal: np.ndarray, tau: float, horizon: int, budgets: np.ndarray) -> dict:
    """OEH-H risk and missing-mass curves at given sample sizes.

    Evaluated at the converged support-constrained optimal policy (the estimand
    of value-iteration learners). ``noise_regret(N)`` is the expected normalized
    regret from CLT misrankings with delta-method variances over the converged
    lookahead tree; ``missing_mass(N)`` is the target occupancy on chosen pairs
    still unobserved after ``N`` logged transitions; ``slack = NR(pi) - tau``.
    """
    S, A, _ = P.shape
    r = np.einsum("san,san->sa", P, R)
    j_star, j_rand = (float(x[0]) for x in reference_returns(P[None], r[None], rho[None], gamma))
    scale = j_star - j_rand
    budgets = np.asarray(budgets, dtype=float)
    if not scale > 0:
        nan = np.full(len(budgets), np.nan)
        return {"noise_regret": nan, "missing_mass": nan, "slack": math.nan}
    live = ~terminal
    support = (beta > 0) & live[:, None]
    support[~support.any(1), 0] = True
    rows = np.arange(S)
    values = np.zeros(S)
    for _ in range(10_000):
        q = r + gamma * np.einsum("san,n->sa", P, values)
        updated = np.where(live, np.where(support, q, -np.inf).max(1), 0.0)
        if np.max(np.abs(updated - values)) <= 1e-12:
            values = updated
            break
        values = updated
    q = r + gamma * np.einsum("san,n->sa", P, values)
    policy = np.where(support, q, -np.inf).argmax(1)
    v_pi = policy_values(P, r, np.eye(A)[policy], gamma)
    q_pi = r + gamma * np.einsum("san,n->sa", P, v_pi)
    slack = (float(rho @ v_pi) - j_rand) / scale - tau
    visits = (1.0 - gamma) * (rho @ np.linalg.inv(np.eye(S) - gamma * P[rows, policy]))
    d_beta = stream_visitation(P, beta, rho, terminal, horizon)
    select = np.zeros((S, S * A))
    select[rows, rows * A + policy] = 1.0
    W = np.linalg.inv(np.eye(S * A) - gamma * (P.reshape(S * A, S) @ select))
    target = R + gamma * np.where(live, v_pi, 0.0)[None, None, :]
    mean = np.einsum("san,san->sa", P, target)
    sigma2 = np.einsum("san,san->sa", P, (target - mean[..., None]) ** 2)
    weight = (W ** 2) * sigma2.reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where((weight > 0) & (d_beta.reshape(1, -1) <= 0), np.inf,
                            np.where(weight > 0, weight / d_beta.reshape(1, -1), 0.0)).sum(1).reshape(S, A)
    loss = np.maximum(q_pi[rows, policy][:, None] - q_pi, 0.0)
    compete = support & live[:, None] & (visits[:, None] > 0) & (loss > 0)
    compete[rows, policy] = False
    regret_weight = np.where(compete, visits[:, None] * loss / ((1.0 - gamma) * scale), 0.0)
    gap = np.where(compete, np.maximum(q_pi[rows, policy][:, None] - q_pi, 0.0), 0.0)
    pair = np.where(compete, variance[rows, policy][:, None] + variance, 0.0)
    noise = np.array([float((regret_weight * _misrank_probability(gap, pair, n)).sum()) for n in budgets])
    chosen = np.where(live, d_beta[rows, policy], 1.0)
    target_visits = np.where(live, visits, 0.0)
    missing = np.array([float((target_visits * np.exp(-n * chosen)).sum()) for n in budgets])
    return {"noise_regret": noise, "missing_mass": missing, "slack": slack}
