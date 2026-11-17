"""Workstream 3 scientific validation of the data-only OEH estimator.

This module answers the questions the estimator's own unit tests cannot: is the
depth-zero reduction what the protocol says it is, does the trajectory bootstrap
have the dispersion property the definition leans on, what does the evidence
curve look like on real pilots, and how does the data-only depth compare with
the exact reference and the empirical requirement.

Exact quantities are read only *after* a data-only result exists, exactly as
``scripts/validation/run_week1_validation.py`` does, and no threshold anywhere in the protocol is
chosen from anything computed here.  The depth-zero and bootstrap checks
deliberately rebuild the empirical model instead of importing
``oeh_estimator._model``: a check that reuses the code under test cannot
detect a fault in that code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from adaptive_pilot import (
    AdaptivePilotSelection,
    select_effective_pilot,
    transitions_from_arrays,
)
from oeh_estimator import OEHConfig, OEHResult, TransitionLike, estimate_oeh

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------
# Running one pilot buffer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotRun:
    """One pilot buffer taken through the registered sequential selector.

    ``results`` keeps every prefix the selector actually assessed, keyed by its
    size multiplier.  The selector stops at the first accepted prefix, so a
    multiplier missing from this mapping was never evaluated, not silently
    dropped.
    """

    selection: AdaptivePilotSelection[OEHResult]
    results: Mapping[int, OEHResult]
    transitions: tuple[TransitionLike, ...]

    @property
    def selected_multiplier(self) -> int:
        return int(self.selection.n0_effective_multiplier)

    @property
    def selected_result(self) -> OEHResult:
        return self.results[self.selected_multiplier]


def prefix_bootstrap_rng(
    bootstrap_seed: int,
    prefix_length: int,
    num_states: int,
    num_actions: int,
) -> np.random.Generator:
    """Rebuild the exact generator ``select_effective_pilot_with_oeh`` uses.

    The diagnostics below have to draw the *same* Dirichlet weights as the
    estimator did, or they would describe a different bootstrap than the one
    that produced the result being validated.
    """

    seed = np.random.SeedSequence(
        [bootstrap_seed % (2**32), prefix_length, num_states, num_actions]
    )
    return np.random.default_rng(seed)


def run_pilot(
    arrays: Mapping[str, Any],
    initial_state_distribution: Sequence[float],
    *,
    num_states: int,
    num_actions: int,
    bootstrap_seed: int,
) -> PilotRun:
    """Run the registered adaptive selector, keeping every assessed result."""

    transitions = transitions_from_arrays(arrays)
    captured: dict[int, OEHResult] = {}

    def assess_prefix(
        prefix: Sequence[TransitionLike],
        initial: FloatArray,
        states: int,
        actions: int,
    ) -> OEHResult:
        result = estimate_oeh(
            prefix,
            initial,
            num_states=states,
            num_actions=actions,
            rng=prefix_bootstrap_rng(
                bootstrap_seed, len(prefix), states, actions
            ),
        )
        captured[len(prefix) // (states * actions)] = result
        return result

    selection = select_effective_pilot(
        transitions,
        initial_state_distribution,
        num_states=num_states,
        num_actions=num_actions,
        assess_prefix=assess_prefix,
    )
    return PilotRun(
        selection=selection, results=dict(captured), transitions=transitions
    )


def oeh_status(result: OEHResult) -> str:
    """Map an ``OEHResult`` onto the Axis-A status vocabulary."""

    if result.abstained:
        return "abstained"
    if result.censored_right:
        return "censored"
    return "observed"


# --------------------------------------------------------------------------
# Independent empirical model, used by both scientific checks
# --------------------------------------------------------------------------


def _weighted_model(
    transitions: Iterable[TransitionLike],
    states: int,
    actions: int,
    supported: Sequence[Sequence[int]],
    weights: Mapping[int, float] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Rebuild ``P_hat, r_hat, beta_hat^K`` from raw transitions."""

    mass = np.zeros((states, actions), dtype=np.float64)
    reward_sum = np.zeros((states, actions), dtype=np.float64)
    successor = np.zeros((states, actions, states), dtype=np.float64)
    for item in transitions:
        weight = 1.0 if weights is None else float(weights[int(item.trajectory_id)])
        state, action = int(item.state), int(item.action)
        mass[state, action] += weight
        reward_sum[state, action] += weight * float(item.reward)
        successor[state, action, int(item.next_state)] += weight
    p_hat = np.zeros_like(successor)
    r_hat = np.zeros((states, actions), dtype=np.float64)
    beta = np.zeros((states, actions), dtype=np.float64)
    for state, row in enumerate(supported):
        row = list(row)
        if not row:
            continue
        if np.any(mass[state, row] <= 0.0):
            return None
        beta[state, row] = mass[state, row] / mass[state, row].sum()
        for action in row:
            p_hat[state, action] = successor[state, action] / mass[state, action]
            r_hat[state, action] = reward_sum[state, action] / mass[state, action]
    return p_hat, r_hat, beta


def _restricted_behavior_q(
    p_hat: FloatArray,
    r_hat: FloatArray,
    beta: FloatArray,
    supported: Sequence[Sequence[int]],
    gamma: float,
) -> tuple[FloatArray, float] | None:
    """``Q^(0) = Q^{beta^K}`` plus the Bellman residual of the linear solve."""

    states = beta.shape[0]
    p_beta = np.einsum("sa,san->sn", beta, p_hat)
    r_beta = np.einsum("sa,sa->s", beta, r_hat)
    try:
        values = np.linalg.solve(np.eye(states) - gamma * p_beta, r_beta)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(values)):
        return None
    residual = float(
        np.max(np.abs(values - (r_beta + gamma * p_beta @ values)))
    )
    q_values = np.full(beta.shape, np.nan, dtype=np.float64)
    for state, row in enumerate(supported):
        for action in row:
            q_values[state, action] = (
                r_hat[state, action] + gamma * p_hat[state, action] @ values
            )
    return q_values, residual


def top_two(
    q_values: FloatArray, row: Sequence[int]
) -> tuple[int, int] | None:
    """Best and second-best supported action, ties broken by action id."""

    if len(row) < 2:
        return None
    order = sorted(row, key=lambda action: (-float(q_values[action]), action))
    return order[0], order[1]


# --------------------------------------------------------------------------
# Check 1 — the depth-zero (K=0) reduction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthZeroCheck:
    """Depth-zero agreement between the estimator and an independent solve."""

    states_compared: int
    max_absolute_gap_error: float
    max_bellman_residual: float
    gaps_are_non_negative: bool
    passed: bool
    reason: str | None = None


def check_depth_zero(
    run: PilotRun,
    result: OEHResult,
    *,
    gamma: float,
    prefix_length: int,
    tolerance: float = 1e-9,
) -> DepthZeroCheck:
    """Verify ``Q_hat^(0)`` really is the restricted behavior policy value.

    Protocol section 3.2 defines depth zero as the value of ``beta^K`` on the
    empirical model, not of ``beta``, and section 3.3 defines the margin as the
    non-negative top-two gap.  Both are re-derived here from the raw
    transitions and compared with what the estimator reported.

    ``prefix_length`` must be the prefix the estimator actually consumed: the
    selector assesses ``transitions[:size]``, so comparing against the whole
    buffer would compare two different empirical models.
    """

    if result.deltas is None:
        return DepthZeroCheck(0, np.nan, np.nan, True, False, "no depth-zero deltas")
    supported = result.supported_actions
    model = _weighted_model(
        run.transitions[:prefix_length],
        result.num_states,
        result.num_actions,
        supported,
    )
    if model is None:
        return DepthZeroCheck(0, np.nan, np.nan, True, False, "model undefined")
    solved = _restricted_behavior_q(*model, supported, gamma)
    if solved is None:
        return DepthZeroCheck(0, np.nan, np.nan, True, False, "singular solve")
    q_values, residual = solved

    errors: list[float] = []
    non_negative = True
    for state, row in enumerate(supported):
        pair = top_two(q_values[state], row)
        if pair is None:
            continue
        gap = float(q_values[state, pair[0]] - q_values[state, pair[1]])
        non_negative &= gap >= -tolerance
        reported = float(result.deltas[0, state])
        if np.isfinite(reported):
            errors.append(abs(gap - reported))
    max_error = float(max(errors)) if errors else 0.0
    return DepthZeroCheck(
        states_compared=len(errors),
        max_absolute_gap_error=max_error,
        max_bellman_residual=residual,
        gaps_are_non_negative=bool(non_negative),
        passed=bool(
            errors
            and max_error <= tolerance
            and residual <= tolerance
            and non_negative
        ),
        reason=None if errors else "no state had two supported actions",
    )


# --------------------------------------------------------------------------
# Check 2 — trajectory bootstrap dispersion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapDispersion:
    """Does bootstrapping the *difference* actually buy what section 3.3 claims?

    Section 3.3 requires the bootstrap to be taken on the gap rather than on
    each action value, because the two values share successor-value estimates
    and are strongly positively correlated, so taking a sigma per value "would
    inflate the uncertainty of the difference".

    ``median_ratio`` is therefore ``sigma(Q1 - Q2) / sqrt(sigma(Q1)^2 +
    sigma(Q2)^2)``: the realised gap dispersion over the dispersion one would
    get by treating the two values as independent, which is exactly the
    inflation the section warns about.  Under equal variances the ratio is
    ``sqrt(1 - rho)``, so a ratio near 1 would mean the shared-error argument
    does not hold on real pilots and ``z_k`` is built on a false premise.
    """

    replicates_used: int
    replicates_requested: int
    states_compared: int
    median_gap_sigma: float
    median_independent_sigma: float
    median_ratio: float
    max_ratio: float
    median_correlation: float


def bootstrap_dispersion(
    run: PilotRun,
    result: OEHResult,
    *,
    config: OEHConfig,
    bootstrap_seed: int,
    prefix_length: int,
) -> BootstrapDispersion:
    """Re-run the registered Dirichlet bootstrap, keeping both dispersions.

    The generator is rebuilt with the estimator's own seed derivation, so the
    replicate weights are the same draws the reported ``sigma`` came from.
    Only depth zero is recomputed: it is the depth where the two action values
    share the most estimation error, and one linear solve per replicate keeps
    this affordable at benchmark scale.
    """

    supported = result.supported_actions
    states, actions = result.num_states, result.num_actions
    trajectory_ids = sorted(
        {int(item.trajectory_id) for item in run.transitions[:prefix_length]}
    )
    transitions = run.transitions[:prefix_length]
    generator = prefix_bootstrap_rng(
        bootstrap_seed, prefix_length, states, actions
    )

    base_model = _weighted_model(transitions, states, actions, supported)
    if base_model is None:
        return BootstrapDispersion(0, config.num_bootstrap, 0, *([np.nan] * 5))
    base_solved = _restricted_behavior_q(*base_model, supported, config.gamma)
    if base_solved is None:
        return BootstrapDispersion(0, config.num_bootstrap, 0, *([np.nan] * 5))
    pairs = [top_two(base_solved[0][state], row) for state, row in enumerate(supported)]

    best: list[list[float]] = [[] for _ in range(states)]
    second: list[list[float]] = [[] for _ in range(states)]
    used = 0
    for _ in range(config.num_bootstrap):
        weights = generator.dirichlet(
            np.full(len(trajectory_ids), config.bootstrap_dirichlet_alpha)
        )
        model = _weighted_model(
            transitions,
            states,
            actions,
            supported,
            dict(zip(trajectory_ids, weights, strict=True)),
        )
        if model is None:
            continue
        solved = _restricted_behavior_q(*model, supported, config.gamma)
        if solved is None:
            continue
        used += 1
        for state, pair in enumerate(pairs):
            if pair is None:
                continue
            best[state].append(float(solved[0][state, pair[0]]))
            second[state].append(float(solved[0][state, pair[1]]))

    gap_sigmas: list[float] = []
    independent_sigmas: list[float] = []
    ratios: list[float] = []
    correlations: list[float] = []
    for state in range(states):
        if len(best[state]) < 2:
            continue
        first = np.asarray(best[state])
        other = np.asarray(second[state])
        gap_sigma = float(np.std(first - other, ddof=1))
        first_sigma = float(np.std(first, ddof=1))
        other_sigma = float(np.std(other, ddof=1))
        independent_sigma = float(np.hypot(first_sigma, other_sigma))
        gap_sigmas.append(gap_sigma)
        independent_sigmas.append(independent_sigma)
        if independent_sigma > 0.0:
            ratios.append(gap_sigma / independent_sigma)
        if first_sigma > 0.0 and other_sigma > 0.0:
            correlations.append(float(np.corrcoef(first, other)[0, 1]))

    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    return BootstrapDispersion(
        replicates_used=used,
        replicates_requested=config.num_bootstrap,
        states_compared=len(gap_sigmas),
        median_gap_sigma=_median(gap_sigmas),
        median_independent_sigma=_median(independent_sigmas),
        median_ratio=_median(ratios),
        max_ratio=float(max(ratios)) if ratios else float("nan"),
        median_correlation=_median(correlations),
    )


# --------------------------------------------------------------------------
# Artifact rows (WEEK2_ARTIFACT_CONTRACT.md sections 1 and 3)
# --------------------------------------------------------------------------


def _blank(value: Any) -> Any:
    """Contract convention: an undefined scalar is an empty cell, not a NaN."""

    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def oeh_results_rows(
    mdp_id: str, family: str, run: PilotRun
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for multiplier, result in sorted(run.results.items()):
        status = oeh_status(result)
        observed = status != "abstained"
        gap = (
            float(result.gap_occupancy[result.k_dagger])
            if result.gap_occupancy is not None and result.k_dagger is not None
            else None
        )
        rows.append(
            {
                "mdp_id": mdp_id,
                "policy_family": family,
                "pilot_multiplier": multiplier,
                "selected": multiplier == run.selected_multiplier,
                "oeh_status": status,
                "abstain_reason": result.reason or "",
                "k_dagger_oeh": _blank(result.k_dagger if observed else None),
                "effective_horizon": _blank(
                    result.effective_horizon if observed else None
                ),
                "n_req_hat": _blank(result.n_req_hat if observed else None),
                "valid_bootstrap_fraction": _blank(result.valid_bootstrap_fraction),
                "gap_occupancy_at_k_dagger": _blank(gap if observed else None),
            }
        )
    return rows


def evidence_curve_rows(
    mdp_id: str, family: str, run: PilotRun
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for multiplier, result in sorted(run.results.items()):
        if result.multipliers is None:
            continue
        for depth, value in enumerate(result.multipliers):
            rows.append(
                {
                    "mdp_id": mdp_id,
                    "policy_family": family,
                    "pilot_multiplier": multiplier,
                    "depth": depth,
                    "multiplier": _blank(float(value)),
                    "loss_at_one": _blank(
                        float(result.losses_at_one[depth])
                        if result.losses_at_one is not None
                        else None
                    ),
                    "gap_occupancy": _blank(
                        float(result.gap_occupancy[depth])
                        if result.gap_occupancy is not None
                        else None
                    ),
                }
            )
    return rows


def bootstrap_results_rows(
    mdp_id: str, family: str, run: PilotRun, *, bootstrap_seed: int, num_bootstrap: int
) -> list[dict[str, Any]]:
    return [
        {
            "mdp_id": mdp_id,
            "policy_family": family,
            "pilot_multiplier": multiplier,
            "num_bootstrap": num_bootstrap,
            "valid_bootstrap_fraction": _blank(result.valid_bootstrap_fraction),
            "bootstrap_seed": bootstrap_seed,
        }
        for multiplier, result in sorted(run.results.items())
    ]


def selected_subset_rows(
    oeh_rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    """``k_dagger_estimates`` / ``n_required_estimates``: selected rows only."""

    return [
        {name: row[name] for name in columns}
        for row in oeh_rows
        if row["selected"]
    ]


def abstention_rows(
    oeh_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in oeh_rows:
        status = row["oeh_status"]
        if status not in {"abstained", "censored"}:
            continue
        rows.append(
            {
                "mdp_id": row["mdp_id"],
                "policy_family": row["policy_family"],
                "pilot_multiplier": row["pilot_multiplier"],
                "oeh_status": status,
                "reason": (
                    row["abstain_reason"]
                    if status == "abstained"
                    else "gap_occupancy_exceeds_epsilon"
                ),
            }
        )
    return rows


def exact_k_curve_rows(mdp_id: str, family: str, curve: Any) -> list[dict[str, Any]]:
    """One row per depth.

    ``delta_hat`` and ``sigma_hat`` are per-state arrays on ``ExactEvidenceCurve``
    but scalars in the contract, which asks for "the aggregate".  The aggregate
    used here is the unweighted mean over states with a defined value
    (``nanmean``); it is recorded in the run report so no reader has to guess
    which aggregate was taken.
    """

    rows: list[dict[str, Any]] = []
    for depth, multiplier in enumerate(curve.multipliers):
        with np.errstate(invalid="ignore"):
            delta = float(np.nanmean(curve.deltas[depth]))
            sigma = float(np.nanmean(curve.sigmas[depth]))
        rows.append(
            {
                "mdp_id": mdp_id,
                "policy_family": family,
                "depth": depth,
                "delta_hat": _blank(delta),
                "sigma_hat": _blank(sigma),
                "multiplier": _blank(float(multiplier)),
                "k_dagger_exact": int(curve.k_dagger),
                "valid_sample_fraction": float(curve.valid_sample_fraction),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Comparisons and failure analysis
# --------------------------------------------------------------------------


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def compare_k_dagger(
    selected_rows: Sequence[Mapping[str, Any]],
    exact_by_key: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    """``k_dagger_oeh`` against ``k_dagger_exact`` (check K4, PLAN section 8.1)."""

    errors: list[float] = []
    oeh_depths: list[float] = []
    exact_depths: list[float] = []
    for row in selected_rows:
        key = (row["mdp_id"], row["policy_family"])
        if key not in exact_by_key or row["k_dagger_oeh"] == "":
            continue
        oeh = int(row["k_dagger_oeh"])
        exact = int(exact_by_key[key])
        errors.append(abs(oeh - exact))
        oeh_depths.append(oeh)
        exact_depths.append(exact)
    within_two = (
        sum(1 for value in errors if value <= 2) / len(errors) if errors else None
    )
    return {
        "pairs_compared": len(errors),
        "absolute_error": _summary(errors),
        "k_dagger_oeh_distribution": dict(
            sorted(_counts(oeh_depths).items())
        ),
        "k_dagger_exact_distribution": dict(
            sorted(_counts(exact_depths).items())
        ),
        "k4_fraction_within_two": within_two,
        "k_dagger_oeh_is_constant": len(set(oeh_depths)) <= 1,
        "k_dagger_exact_is_constant": len(set(exact_depths)) <= 1,
    }


def _counts(values: Sequence[float]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def compare_with_empirical_requirement(
    selected_rows: Sequence[Mapping[str, Any]],
    n_emp_rows: Sequence[Mapping[str, Any]],
    pilot_sizes: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    """``N_hat_req`` and the pilot budget against the measured ``N_emp``.

    Right-censored ``N_emp`` rows are reported as their own count and never
    replaced by ``N_max`` or dropped, per PLAN section 3.1.  The pilot-budget
    ratio is a *proxy* for K1a: K1a is defined on the budget at which ``k†``
    stabilizes, and this uses the selected ``N0_eff`` instead, which is the
    budget actually spent.  It is reported as ``k1a_proxy`` for that reason.
    """

    per_algorithm: dict[str, dict[str, Any]] = {}
    algorithms = sorted({str(row["algorithm"]) for row in n_emp_rows})
    for algorithm in algorithms:
        ratios: list[float] = []
        budget_ratios: list[float] = []
        censored = 0
        unmatched = 0
        for row in n_emp_rows:
            if str(row["algorithm"]) != algorithm:
                continue
            key = (str(row["mdp_id"]), str(row["policy_family"]))
            selected = next(
                (
                    item
                    for item in selected_rows
                    if (item["mdp_id"], item["policy_family"]) == key
                ),
                None,
            )
            if selected is None:
                unmatched += 1
                continue
            if str(row["n_emp_censored"]).lower() in {"true", "1"}:
                censored += 1
                continue
            n_emp = float(row["n_emp_transitions"])
            if n_emp <= 0.0:
                continue
            if key in pilot_sizes:
                budget_ratios.append(pilot_sizes[key] / n_emp)
            if selected["n_req_hat"] != "":
                ratios.append(float(selected["n_req_hat"]) / n_emp)
        per_algorithm[algorithm] = {
            "n_emp_censored": censored,
            "unmatched_keys": unmatched,
            "n_req_hat_over_n_emp": _summary(ratios),
            "k1a_proxy_pilot_budget_over_n_emp": _summary(budget_ratios),
            "k1a_proxy_threshold": 0.25,
            "k1a_proxy_met": (
                bool(np.median(budget_ratios) <= 0.25) if budget_ratios else None
            ),
        }
    return per_algorithm


def failure_analysis(
    oeh_rows: Sequence[Mapping[str, Any]],
    coverage_by_key: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Where the estimator declines to answer, and what those cases look like."""

    selected = [row for row in oeh_rows if row["selected"]]
    by_status: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    coverage_by_status: dict[str, list[float]] = {}
    for row in selected:
        status = str(row["oeh_status"])
        family = str(row["policy_family"])
        by_status[status] = by_status.get(status, 0) + 1
        by_family.setdefault(family, {})
        by_family[family][status] = by_family[family].get(status, 0) + 1
        if status == "abstained" and row["abstain_reason"]:
            key = str(row["abstain_reason"])
            reasons[key] = reasons.get(key, 0) + 1
        key = (row["mdp_id"], row["policy_family"])
        if key in coverage_by_key:
            coverage_by_status.setdefault(status, []).append(coverage_by_key[key])
    answered = by_status.get("observed", 0)
    total = len(selected)
    return {
        "selected_cases": total,
        "by_status": dict(sorted(by_status.items())),
        "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
        "abstain_reasons": dict(sorted(reasons.items())),
        "coverage_by_status": {
            status: _summary(values)
            for status, values in sorted(coverage_by_status.items())
        },
        "non_answer_rate": (total - answered) / total if total else None,
        "success_criterion_6_threshold": 0.5,
    }


__all__ = [
    "BootstrapDispersion",
    "DepthZeroCheck",
    "PilotRun",
    "abstention_rows",
    "bootstrap_dispersion",
    "bootstrap_results_rows",
    "check_depth_zero",
    "compare_k_dagger",
    "compare_with_empirical_requirement",
    "evidence_curve_rows",
    "exact_k_curve_rows",
    "failure_analysis",
    "oeh_results_rows",
    "oeh_status",
    "prefix_bootstrap_rng",
    "run_pilot",
    "selected_subset_rows",
]
