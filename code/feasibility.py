"""Feasibility of difficulty estimation, decided from the pilot dataset alone.

This is the practitioner-facing side of Theorem 1 and Corollary 2.2
(``docs/DERIVATIONS.md`` sec 2, 3.4). Those results say that a resolution-failure
statistic whose per-state term is homogeneous of degree **one** in
``(Delta, sigma, V_hat)`` -- the registered definition among them -- obeys

    L_k(1) <= G * sigma_bar_k,      G = max_{z>=0} z*Phi(-z) = 0.169971

so escaping saturation *requires* ``sigma_bar >= epsilon / G``. Both sides are
computable from a pilot buffer, before any estimator is trusted and before any
training run is paid for.

That turns a theorem into a decision procedure, and the decision procedure is
what makes the pipeline difficulty-**aware** in the sense the thesis title
claims: the system reports not only an estimate but whether an estimate of this
kind is obtainable from the data at hand at all.

**Scope, stated rather than implied.** The necessary condition applies to the
degree-one class. A degree-zero statistic (``materiality``, ``max_state`` --
Corollary 2.3) is scale-invariant and is *not* bounded by ``G * sigma_bar``, so
a verdict here does not transfer to it. ``ADMISSIBILITY`` records which class
each registered definition falls in, and the report says which class it judged.

**What this module is not.** It does not decide whether a statistic is
*useful*, only whether it can be non-degenerate. Axioms A1--A3 are necessary for
admissibility, not sufficient for information: the C2 run
(``artifacts/week2/c2_correlation_2026-09-02/``) measured two axiom-satisfying
definitions and neither beat a structural baseline. Feasible is not the same as
informative, and the report says so in as many words.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from oeh_estimator import OEHResult
from positive_control import MAX_Z_PHI, required_sigma_bar, weighted_sigma_bar

#: Homogeneity degree in ``(Delta, sigma, V_hat)`` per Theorem 2. Degree one
#: violates A1 and is subject to Corollary 2.2's necessary condition; degree
#: zero satisfies A1 and is not.
ADMISSIBILITY: dict[str, dict[str, Any]] = {
    "registered": {
        "homogeneity_degree": 1,
        "satisfies_a1": False,
        "necessary_condition_applies": True,
        "note": "psi = Delta*Phi(-z sqrt(mu)) = sigma * [z*Phi(-z sqrt(mu))]; the sigma factor "
        "outside the scale-free bracket is what makes it degree one (Corollary 2.1).",
    },
    "probability": {
        "homogeneity_degree": 0,
        "satisfies_a1": True,
        "necessary_condition_applies": False,
        "note": "Scale-invariant, but violates A2 and A3: a tied state contributes w/2 at every "
        "mu, so m can be infinite. Admissible under A1 alone is not admissible.",
    },
    "materiality": {
        "homogeneity_degree": 0,
        "satisfies_a1": True,
        "necessary_condition_applies": False,
        "note": "h(z, rho) = 1[rho >= f] * Phi(-z sqrt(mu)); satisfies A1-A3 (Corollary 2.3).",
    },
    "max_state": {
        "homogeneity_degree": 0,
        "satisfies_a1": True,
        "necessary_condition_applies": False,
        "note": "h(z, rho) = Phi(-z sqrt(mu)) aggregated by max over {rho >= f}; A1-A3 hold.",
    },
}

#: Ordered strongest-evidence-first. The first tier whose condition holds is the
#: verdict, because a proof of impossibility outranks a failed necessary
#: condition, which in turn outranks an observed shortfall.
VERDICTS = (
    "no_curve",
    "impossible_at_any_dispersion",
    "blocked_by_theorem_1",
    "underpowered",
    "feasible",
)

MECHANISMS = (
    "reward_starvation",
    "determinism",
    "occupancy_dilution",
    "insufficient_evidence",
    "none",
)

#: A critical state carrying this much occupancy while excluded from the sum is
#: reported as dilution. Not a threshold anything is decided on -- the verdict
#: never depends on it; it only names the mechanism.
DILUTION_NOTICE = 0.10


@dataclass(frozen=True)
class FeasibilityReport:
    """Can a difficulty statistic of the judged class work on this dataset?

    ``verdict`` answers the question, ``mechanism`` says why, and the two
    headroom numbers say how far off it is. The headrooms are ratios rather than
    differences so they can be read without knowing ``epsilon``: a value of 1.0
    is exactly at the bar, 0.04 is 24x short.
    """

    label: str
    definition: str
    homogeneity_degree: int
    necessary_condition_applies: bool
    verdict: str
    mechanism: str
    epsilon: float
    sigma_bar: float | None
    required_sigma_bar: float
    sigma_headroom: float | None
    exact_ceiling: float | None
    ceiling_headroom: float | None
    max_loss_at_one: float | None
    dispersion_multiplier_needed: float | None
    max_gap: float | None
    max_gap_occupancy: float | None
    num_supported_states: int
    explanation: str
    remedy: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _nanmax(values: Any) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.max()) if finite.size else None


def exact_ceiling(result: OEHResult) -> float | None:
    """``max_k (1/2) * sum_s w_k(s) Delta_k(s)`` -- DERIVATIONS.md sec 4.3.

    This is not an upper bound that a lucky instance might beat: it is the value
    ``L_k(1)`` converges to when *every* eligible state becomes maximally
    uncertain, since ``Phi(-0) = 1/2``. Below ``epsilon`` it is a proof of
    impossibility at any dispersion, which is why it outranks the ``sigma_bar``
    condition in the verdict ordering.
    """

    if result.deltas is None or result.occupancy_weights is None:
        return None
    weighted = 0.5 * np.nansum(
        np.nan_to_num(result.occupancy_weights) * np.nan_to_num(result.deltas), axis=1
    )
    return _nanmax(weighted)


def dispersion_multiplier_needed(result: OEHResult, epsilon: float) -> float | None:
    """``c*`` of DERIVATIONS.md sec 6: how many times more dispersed the data
    would have to be, or ``None`` when no finite inflation suffices.

    Inflating ``sigma`` by ``c`` sends ``z -> z/c``. A state with ``sigma``
    exactly zero has ``z = +inf`` and stays there for every finite ``c``, which
    is why this is reported next to ``exact_ceiling`` and never instead of it.
    """

    if result.deltas is None or result.sigmas is None or result.occupancy_weights is None:
        return None
    deltas = np.nan_to_num(result.deltas)
    sigmas = np.nan_to_num(result.sigmas)
    weights = np.nan_to_num(result.occupancy_weights)

    def loss_at(multiplier: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigmas > 0.0, deltas / np.maximum(sigmas, 1e-300), np.inf)
        inflated = z / multiplier
        per_depth = np.sum(weights * deltas * norm.cdf(-inflated), axis=1)
        return float(np.nanmax(per_depth))

    if loss_at(1.0) >= epsilon:
        return 1.0
    low, high = 1.0, 2.0
    for _ in range(60):  # 2^60 is far past any physically meaningful inflation
        if loss_at(high) >= epsilon:
            break
        low, high = high, high * 2.0
    else:
        return None
    for _ in range(80):
        mid = 0.5 * (low + high)
        if loss_at(mid) >= epsilon:
            high = mid
        else:
            low = mid
    return float(high)


def _mechanism_from_scalars(
    *,
    max_gap: float | None,
    max_gap_occupancy: float | None,
    zero_sigma_fraction: float | None,
) -> str:
    """Name the blocking mechanism. Diagnostic only -- never decides a verdict."""

    if max_gap is not None and max_gap <= 0.0:
        return "reward_starvation"
    if zero_sigma_fraction is not None and zero_sigma_fraction >= 1.0:
        return "determinism"
    if max_gap_occupancy is not None and max_gap_occupancy >= DILUTION_NOTICE:
        return "occupancy_dilution"
    return "insufficient_evidence"


_MECHANISM_REMEDY = {
    "reward_starvation": (
        "No supported critical state has a non-zero action gap, so every loss in this class is "
        "identically zero by construction. The dataset, not the statistic, has to change: the "
        "behavior policy must reach rewarding states often enough to record them."
    ),
    "determinism": (
        "Every state with a gap has zero bootstrap dispersion, and inflating sigma cannot rescue "
        "z = +inf at any finite factor (DERIVATIONS.md sec 6). A deterministic instance cannot "
        "serve this class of statistic; use stochastic rewards or fewer trajectories."
    ),
    "occupancy_dilution": (
        "States carrying the gaps sit outside the sum, so occupancy mass is spent on states that "
        "contribute nothing. Raising coverage on the excluded critical states is the lever, not "
        "raising dispersion."
    ),
    "insufficient_evidence": (
        "Gaps and dispersion are both present but their ratio sits far from the maximising "
        "z* = 0.7518. Fewer trajectories or higher-variance rewards move z down toward it."
    ),
    "none": "",
}


def decide(
    *,
    label: str,
    definition: str,
    epsilon: float,
    sigma_bar: float | None,
    exact_ceiling: float | None,
    max_loss_at_one: float | None,
    dispersion_multiplier: float | None,
    max_gap: float | None,
    max_gap_occupancy: float | None,
    zero_sigma_fraction: float | None,
    num_supported_states: int,
    abstained: bool,
    abstain_reason: str | None = None,
) -> FeasibilityReport:
    """The verdict, from scalars only.

    Both entry points funnel through here -- a live estimator run
    (:func:`assess`) and a recorded artifact (:func:`assess_from_record`) --
    because the same evidence must not be tiered two different ways depending
    on which door it came in.

    The ordering is the point. A proof of impossibility (``exact_ceiling <
    epsilon``) outranks a failed necessary condition (``sigma_bar`` short),
    which outranks an observed shortfall. Reporting the weakest true statement
    would understate what is known about the instance.
    """

    if definition not in ADMISSIBILITY:
        raise ValueError(
            f"Unknown definition {definition!r}; expected one of {sorted(ADMISSIBILITY)}."
        )
    profile = ADMISSIBILITY[definition]
    needed = required_sigma_bar(epsilon)
    headroom = None if sigma_bar is None else sigma_bar / needed
    ceiling_headroom = None if exact_ceiling is None else exact_ceiling / epsilon

    if abstained:
        return FeasibilityReport(
            label=label,
            definition=definition,
            homogeneity_degree=profile["homogeneity_degree"],
            necessary_condition_applies=profile["necessary_condition_applies"],
            verdict="no_curve",
            mechanism="insufficient_evidence",
            epsilon=epsilon,
            sigma_bar=None,
            required_sigma_bar=needed,
            sigma_headroom=None,
            exact_ceiling=None,
            ceiling_headroom=None,
            max_loss_at_one=None,
            dispersion_multiplier_needed=None,
            max_gap=None,
            max_gap_occupancy=None,
            num_supported_states=num_supported_states,
            explanation=(
                "The estimator abstained, so no depth-evidence curve exists and feasibility is "
                f"not decidable from this buffer: {abstain_reason or 'no reason recorded'}"
            ),
            remedy=(
                "The instance needs supported action pairs in its critical states before any "
                "statistic of this class has an input at all."
            ),
        )

    mechanism = _mechanism_from_scalars(
        max_gap=max_gap,
        max_gap_occupancy=max_gap_occupancy,
        zero_sigma_fraction=zero_sigma_fraction,
    )

    if max_loss_at_one is not None and max_loss_at_one >= epsilon:
        verdict, mechanism = "feasible", "none"
        explanation = (
            f"L_k(1) reaches {max_loss_at_one:.4g} >= epsilon = {epsilon:g} at some depth, so the "
            "statistic is not degenerate on this dataset. Feasible is not the same as "
            "informative: whether it predicts the empirical requirement better than |S||A| is a "
            "separate, measured question (artifacts/week2/c2_correlation_2026-09-02/)."
        )
        remedy = ""
    elif exact_ceiling is not None and exact_ceiling < epsilon:
        verdict = "impossible_at_any_dispersion"
        explanation = (
            f"The exact ceiling max_k (1/2) sum_s w_k(s) Delta_k(s) = {exact_ceiling:.4g} is below "
            f"epsilon = {epsilon:g}. That is the value the loss converges to when every eligible "
            "state becomes maximally uncertain, so no amount of dispersion reaches epsilon on "
            "this instance. This is a proof, not a bound (DERIVATIONS.md sec 4.3)."
        )
        remedy = _MECHANISM_REMEDY[mechanism]
    elif profile["necessary_condition_applies"] and sigma_bar is not None and sigma_bar < needed:
        verdict = "blocked_by_theorem_1"
        explanation = (
            f"sigma_bar = {sigma_bar:.4g} is below the necessary {needed:.4g} = epsilon / "
            f"{MAX_Z_PHI:.6g}, so Theorem 1 bounds L_k(1) below epsilon whatever the gaps are. "
            f"The instance is short by a factor of {needed / max(sigma_bar, 1e-300):.3g}."
        )
        remedy = _MECHANISM_REMEDY[mechanism]
    else:
        verdict = "underpowered"
        loss_text = "unrecorded" if max_loss_at_one is None else f"{max_loss_at_one:.4g}"
        sigma_text = "unrecorded" if sigma_bar is None else f"{sigma_bar:.4g}"
        explanation = (
            f"sigma_bar = {sigma_text} clears the necessary bound but the observed "
            f"L_k(1) = {loss_text} still misses epsilon = {epsilon:g}, so the gaps are mis-scaled "
            "relative to the dispersion rather than absent."
        )
        remedy = _MECHANISM_REMEDY[mechanism]

    return FeasibilityReport(
        label=label,
        definition=definition,
        homogeneity_degree=profile["homogeneity_degree"],
        necessary_condition_applies=profile["necessary_condition_applies"],
        verdict=verdict,
        mechanism=mechanism,
        epsilon=epsilon,
        sigma_bar=sigma_bar,
        required_sigma_bar=needed,
        sigma_headroom=headroom,
        exact_ceiling=exact_ceiling,
        ceiling_headroom=ceiling_headroom,
        max_loss_at_one=max_loss_at_one,
        dispersion_multiplier_needed=None if verdict == "feasible" else dispersion_multiplier,
        max_gap=max_gap,
        max_gap_occupancy=max_gap_occupancy,
        num_supported_states=num_supported_states,
        explanation=explanation,
        remedy=remedy,
    )


def assess(
    result: OEHResult,
    *,
    epsilon: float,
    definition: str = "registered",
    label: str = "pilot",
) -> FeasibilityReport:
    """Judge feasibility from a live estimator run."""

    supported = sum(1 for actions in result.supported_actions if len(actions) >= 2)
    if result.abstained:
        return decide(
            label=label,
            definition=definition,
            epsilon=epsilon,
            sigma_bar=None,
            exact_ceiling=None,
            max_loss_at_one=None,
            dispersion_multiplier=None,
            max_gap=None,
            max_gap_occupancy=None,
            zero_sigma_fraction=None,
            num_supported_states=supported,
            abstained=True,
            abstain_reason=result.reason,
        )

    zero_sigma_fraction = None
    if result.deltas is not None and result.sigmas is not None:
        deltas = np.nan_to_num(result.deltas)
        sigmas = np.nan_to_num(result.sigmas)
        eligible = deltas > 0.0
        if eligible.any():
            zero_sigma_fraction = float(np.mean(sigmas[eligible] == 0.0))

    return decide(
        label=label,
        definition=definition,
        epsilon=epsilon,
        sigma_bar=weighted_sigma_bar(result),
        exact_ceiling=exact_ceiling(result),
        max_loss_at_one=_nanmax(result.losses_at_one),
        dispersion_multiplier=dispersion_multiplier_needed(result, epsilon),
        max_gap=_nanmax(result.deltas),
        max_gap_occupancy=_nanmax(result.gap_occupancy),
        zero_sigma_fraction=zero_sigma_fraction,
        num_supported_states=supported,
        abstained=False,
    )


def _optional(row: Any, key: str) -> float | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return None if np.isnan(number) else number


def assess_from_record(
    row: Any,
    *,
    definition: str = "registered",
    label: str | None = None,
    sigma_bar_value: float | None = None,
) -> FeasibilityReport:
    """Judge feasibility from an already-committed audit row.

    ``DEMO_PLAN.md`` sec 3 requires the defence demo to read the result store
    rather than train anything on stage, and re-running the estimator over 825
    buffers is hours of compute for numbers already on disk. This reads
    ``saturation_audit_2026-08-25/audit_loss_ceiling.csv``, whose columns are
    exactly the quantities the verdict needs.

    ``sigma_bar`` is not in that file -- it lives in
    ``alternative_definitions_2026-09-01/alternative_definitions.csv`` as
    ``sigma_weighted_mean`` -- so the caller joins it and passes it here rather
    than this function guessing at a second artifact's location.
    """

    status = str(row.get("oeh_status", "")).strip().lower()
    return decide(
        label=label or f"{row.get('mdp_id')}/{row.get('policy_family')}",
        definition=definition,
        epsilon=float(row["epsilon"]),
        sigma_bar=sigma_bar_value,
        exact_ceiling=_optional(row, "exact_ceiling"),
        max_loss_at_one=_optional(row, "max_loss_at_one"),
        dispersion_multiplier=_optional(row, "sigma_inflation_to_reach_epsilon"),
        max_gap=_optional(row, "max_delta"),
        max_gap_occupancy=None,
        zero_sigma_fraction=_optional(row, "zero_sigma_fraction"),
        num_supported_states=int(_optional(row, "eligible_states_at_k_dagger") or 0),
        abstained=status == "abstained",
        abstain_reason="recorded as abstained in the audit artifact",
    )


REPORT_FIELDS: tuple[str, ...] = (
    "label",
    "definition",
    "homogeneity_degree",
    "necessary_condition_applies",
    "verdict",
    "mechanism",
    "epsilon",
    "sigma_bar",
    "required_sigma_bar",
    "sigma_headroom",
    "exact_ceiling",
    "ceiling_headroom",
    "max_loss_at_one",
    "dispersion_multiplier_needed",
    "max_gap",
    "max_gap_occupancy",
    "num_supported_states",
    "explanation",
    "remedy",
)

__all__ = [
    "ADMISSIBILITY",
    "assess_from_record",
    "decide",
    "DILUTION_NOTICE",
    "MECHANISMS",
    "REPORT_FIELDS",
    "VERDICTS",
    "FeasibilityReport",
    "assess",
    "dispersion_multiplier_needed",
    "exact_ceiling",
]
