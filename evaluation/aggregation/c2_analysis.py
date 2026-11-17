"""C2 -- does ``n_req_hat`` beat the structural-only baseline ``S*A``?

This module is the analysis frame, written *before* the real numbers are
looked at (``PHASE_B_PLAN.md`` week 1, Statistical Evaluation: "viet truoc khung
phan tich ... de ngay 4 chi cam so that"). Everything that could be tuned after
seeing a coefficient -- which pairs enter, what is divided out, which method is
reported -- is fixed here and exercised on synthetic data. Plugging in the real
join changes inputs only.

What it computes, per ``WEEK2_DECISION_GATE.md`` sec 10b:

* **Spearman and Pearson**, both, always. sec 10b's finding hinges on the two
  disagreeing in sign (0.206 vs -0.169), so reporting one is reporting half.
* **Two scales.** ``raw`` correlates ``n_req_hat`` against ``n_emp_transitions``.
  ``per_sa`` divides ``S*A`` out of *both* sides, isolating the part of
  ``n_req_hat`` that actually depends on the pilot data (the selected
  multiplier) against ``N_emp`` in the same units. That second scale is the
  whole check: ``n_req_hat = N_0 * min_k m_k`` and ``N_0 = c * S*A``, so a raw
  correlation is mostly ``S*A`` correlating with itself.
* **The structural baseline in the same table**, so "beats the baseline" is
  read off one table rather than assembled by eye. Note in advance that the
  baseline's ``per_sa`` cell is *undefined*, not zero: ``S*A / (S*A) = 1`` is a
  constant. That is the correct output and the reason
  ``correlation.correlate`` reports a reason string instead of 0.0.
* **Four definitions plus the registered one**, at every registered
  materiality threshold, because ``max_state``'s ``n_req_hat`` moves 1.48x
  across delta (``alternative_definitions_2026-09-01/README.md``) and a
  coefficient quoted without that spread is a coefficient quoted without its
  sensitivity.

Two censoring axes, and neither is ever filled in with a stand-in:

* ``n_emp_censored`` -- the label side. The run needed more than the whole size
  grid; the true value is unknown (``PLAN_Final.md`` sec 3.1: never replaced by
  ``N_max``).
* ``n_req_hat = inf`` -- the estimator side. The loss never reaches
  ``epsilon``, so the estimator's answer *is* "no finite amount of data
  suffices." Definite, not missing.

Those are different objects and the rules below treat them differently. An
empty ``n_req_hat`` (the estimator abstained) is a third thing again, and is
dropped under every rule -- there is no answer to rank.

``EXPERIMENT_RULE.md`` rule 2 requires the rule to be fixed *before* any
coefficient is seen, which is why ``scripts/analysis/run_c2_correlation.py`` refuses to run on
real inputs until the choice is recorded in ``ARM_REGISTRY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.aggregation.correlation import (
    DEFAULT_CONFIDENCE,
    DEFAULT_NUM_RESAMPLES,
    METHODS,
    correlate,
    correlate_difference,
    holm_adjust,
)

SCALES = ("raw", "per_sa")
STRUCTURAL_PREDICTOR = "structural_sa"
OEH_PREDICTOR = "n_req_hat"
POOLED_SCOPE = "pooled"

#: Admissible dispositions of the two censoring axes. Every rule shares one
#: invariant: no censored or infinite observation is ever replaced by a finite
#: number. They differ only in whether an ordered-but-unbounded observation is
#: allowed to contribute its *rank*.
CENSORING_RULES: dict[str, str] = {
    "complete_case": (
        "Keep only pairs with a finite n_req_hat and an observed N_emp. Both Spearman and "
        "Pearson are defined. Conservative and the only rule under which the Pearson column "
        "is populated; drops the most rows, and the dropped count is reported next to n."
    ),
    "infinite_n_req_ranked_last": (
        "complete_case, plus pairs whose n_req_hat is infinite, ranked above every finite "
        "n_req_hat (tied among themselves). Justified because 'no finite amount of data "
        "suffices' is a definite answer and the largest one on that axis. Spearman only; "
        "Pearson is reported undefined rather than computed on a substitute."
    ),
    "both_axes_ranked": (
        "infinite_n_req_ranked_last, plus right-censored N_emp ranked above every observed "
        "N_emp. Valid only when every censored observation's known lower bound exceeds every "
        "observed value in the analysis scale -- true per-multiplier (censored means > 50x, "
        "observed means <= 50x) but NOT true in raw transition counts across MDPs of different "
        "size. The rule verifies this and refuses rather than assuming it. Spearman only."
    ),
}
DEFAULT_CENSORING_RULE = "complete_case"

#: ``ARM_REGISTRY.md`` "Quy tac C2-2" clause 6, registered 2026-08-31 by
#: Workstream 2: C2 fixes ``materiality_fraction = 0.01`` for ``materiality``
#: and ``max_state``. ``registered`` and ``probability`` do not depend on a
#: threshold at all. Any coefficient at 0.005 or 0.02 is *exploratory* and may
#: not be used to pick a result -- so the frame computes it (a sensitivity that
#: is not reported is a sensitivity that was hidden) and labels it, rather than
#: leaving the reader to check which delta a number came from.
PRIMARY_MATERIALITY_FRACTION = 0.01

#: The one behavior family generated from the true ``Q*``
#: (``PROJECT_CONTEXT_READ_ONLY.md`` sec 7). ``PLAN_Final.md`` sec 8.2
#: criterion 4 requires the criterion-3 conclusion to survive when this family
#: is removed: a result that only holds because the behavior policy was built
#: from the answer is not a result about predicting difficulty from data.
Q_STAR_DERIVED_FAMILY = "q_star_softmax"
ALL_FAMILIES_SCOPE = "all_families"
NON_Q_STAR_SCOPE = "non_q_star_families"
THRESHOLD_DEPENDENT_VARIANTS = ("materiality", "max_state")

DISPOSITIONS = (
    "kept",
    "kept_n_req_infinite",
    "kept_n_emp_censored",
    "dropped_n_req_abstained",
    "dropped_n_req_infinite",
    "dropped_n_emp_censored",
    "dropped_n_emp_missing",
)


@dataclass(frozen=True)
class JoinedPair:
    """One ``(mdp_id, policy_family, algorithm, variant, delta)`` row.

    ``n_req_hat`` is ``None`` when the estimator abstained, ``inf`` when it
    answered "never", finite otherwise. ``n_emp_transitions`` is ``None``
    exactly when ``n_emp_censored`` is True.
    """

    mdp_id: str
    policy_family: str
    algorithm: str
    variant: str
    materiality_fraction: float | None
    n_req_hat: float | None
    oeh_status: str
    n_emp_transitions: float | None
    n_emp_censored: bool
    num_states: int
    num_actions: int

    @property
    def state_action_product(self) -> int:
        return self.num_states * self.num_actions


def analysis_role(variant: str, materiality_fraction: float | None) -> str:
    """``"primary"`` or ``"exploratory"`` per registry rule C2-2 clause 6.

    A variant that does not depend on a materiality threshold is always
    primary -- there is no other delta it could have been run at.
    """

    if variant not in THRESHOLD_DEPENDENT_VARIANTS:
        return "primary"
    if materiality_fraction is None:
        return "primary"
    return "primary" if materiality_fraction == PRIMARY_MATERIALITY_FRACTION else "exploratory"


@dataclass
class AnalysisCell:
    """One row of the output table: a predictor, a scale, a method, a number."""

    predictor: str
    variant: str
    materiality_fraction: float | None
    algorithm_scope: str
    scale: str
    censoring_rule: str
    estimate: Any
    counts: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "predictor": self.predictor,
            "variant": self.variant,
            "materiality_fraction": self.materiality_fraction,
            "analysis_role": analysis_role(self.variant, self.materiality_fraction),
            "algorithm_scope": self.algorithm_scope,
            "scale": self.scale,
            "censoring_rule": self.censoring_rule,
        }
        row.update(self.estimate.as_row())
        row.update({f"n_{name}": self.counts.get(name, 0) for name in DISPOSITIONS})
        row["n_candidate_pairs"] = sum(self.counts.get(name, 0) for name in DISPOSITIONS)
        return row


OUTPUT_FIELDS: tuple[str, ...] = (
    "predictor",
    "variant",
    "materiality_fraction",
    "analysis_role",
    "algorithm_scope",
    "scale",
    "censoring_rule",
    "method",
    "coefficient",
    "ci_low",
    "ci_high",
    "confidence",
    "n_pairs",
    "num_clusters",
    "num_resamples",
    "num_defined_replicates",
    "undefined_reason",
    "ci_undefined_reason",
    "n_candidate_pairs",
    *(f"n_{name}" for name in DISPOSITIONS),
)


def _rule_keeps_infinite_n_req(rule: str) -> bool:
    return rule in {"infinite_n_req_ranked_last", "both_axes_ranked"}


def _rule_keeps_censored_n_emp(rule: str) -> bool:
    return rule == "both_axes_ranked"


def _scaled(value: float, pair: JoinedPair, scale: str) -> float:
    if scale == "raw":
        return value
    if scale == "per_sa":
        return value / pair.state_action_product
    raise ValueError(f"Unknown scale {scale!r}; expected one of {SCALES}.")


def _predictor_value(pair: JoinedPair, predictor: str) -> float | None:
    if predictor == OEH_PREDICTOR:
        return pair.n_req_hat
    if predictor == STRUCTURAL_PREDICTOR:
        return float(pair.state_action_product)
    raise ValueError(f"Unknown predictor {predictor!r}.")


def select_pairs(
    pairs: Sequence[JoinedPair],
    *,
    predictor: str,
    scale: str,
    censoring_rule: str,
    censored_n_emp_lower_bound_multiplier: float,
) -> tuple[list[float], list[float], list[str], dict[str, int]]:
    """Apply the censoring rule and the scale; return ``(x, y, clusters, counts)``.

    ``counts`` always sums to ``len(pairs)``: every candidate pair leaves this
    function through exactly one disposition, so "how many were dropped and
    why" is answerable from the output row alone
    (``WEEK2_STATISTICAL_PLAN.md`` sec 2's four-counts rule, applied to the
    correlation table).
    """

    if censoring_rule not in CENSORING_RULES:
        raise ValueError(
            f"Unregistered censoring rule {censoring_rule!r}. Admissible: "
            f"{sorted(CENSORING_RULES)}. EXPERIMENT_RULE.md rule 2 requires the choice to be "
            "fixed and recorded before any coefficient is looked at."
        )
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; expected one of {SCALES}.")

    counts = {name: 0 for name in DISPOSITIONS}
    kept: list[tuple[float, float, str]] = []
    censored_rows: list[JoinedPair] = []

    for pair in pairs:
        value = _predictor_value(pair, predictor)
        if value is None:
            counts["dropped_n_req_abstained"] += 1
            continue
        if np.isinf(value):
            if not _rule_keeps_infinite_n_req(censoring_rule):
                counts["dropped_n_req_infinite"] += 1
                continue
        if pair.n_emp_censored:
            if not _rule_keeps_censored_n_emp(censoring_rule):
                counts["dropped_n_emp_censored"] += 1
                continue
            censored_rows.append(pair)
            continue
        if pair.n_emp_transitions is None:
            # Not censored and no value: an incomplete label row, not a
            # scientific outcome. Never imputed.
            counts["dropped_n_emp_missing"] += 1
            continue
        x = value if np.isinf(value) else _scaled(value, pair, scale)
        y = _scaled(float(pair.n_emp_transitions), pair, scale)
        counts["kept_n_req_infinite" if np.isinf(value) else "kept"] += 1
        kept.append((x, y, pair.mdp_id))

    if censored_rows:
        observed_max = max((y for _, y, _ in kept), default=float("-inf"))
        for pair in censored_rows:
            lower_bound = _scaled(
                censored_n_emp_lower_bound_multiplier * pair.state_action_product, pair, scale
            )
            if lower_bound < observed_max:
                raise ValueError(
                    f"censoring rule {censoring_rule!r} ranks a right-censored N_emp above every "
                    f"observed one, but {pair.mdp_id}/{pair.policy_family}/{pair.algorithm} has a "
                    f"lower bound of {lower_bound} below the observed maximum {observed_max} in "
                    f"scale {scale!r}. The ordering the rule assumes does not hold here; use "
                    "'infinite_n_req_ranked_last' or the 'per_sa' scale instead of assuming it."
                )
            value = _predictor_value(pair, predictor)
            assert value is not None  # abstained rows never reach here
            x = value if np.isinf(value) else _scaled(value, pair, scale)
            counts["kept_n_emp_censored"] += 1
            kept.append((x, float("inf"), pair.mdp_id))

    xs = [item[0] for item in kept]
    ys = [item[1] for item in kept]
    clusters = [item[2] for item in kept]
    return xs, ys, clusters, counts


def _variant_keys(pairs: Iterable[JoinedPair]) -> list[tuple[str, float | None]]:
    """Distinct ``(variant, delta)`` keys, ordered deterministically.

    ``delta`` is ``None`` for the threshold-independent variants (``registered``,
    ``probability``), so the sort key coerces it rather than comparing ``None``
    against a float.
    """

    seen: dict[tuple[str, float | None], None] = {}
    for pair in pairs:
        seen.setdefault((pair.variant, pair.materiality_fraction), None)
    return sorted(seen, key=lambda key: (key[0], float("-inf") if key[1] is None else key[1]))


def _algorithm_scopes(pairs: Iterable[JoinedPair]) -> list[str]:
    algorithms = sorted({pair.algorithm for pair in pairs})
    return [*algorithms, POOLED_SCOPE]


def run_c2_analysis(
    pairs: Sequence[JoinedPair],
    *,
    censoring_rule: str,
    rng: np.random.Generator,
    censored_n_emp_lower_bound_multiplier: float,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[dict[str, Any]]:
    """The full pre-specified table: every predictor x scale x method x scope.

    Deliberately computes *all* cells rather than letting a caller request one.
    A function that returns a single coefficient invites running it repeatedly
    and reporting the run that came out best; this one makes the whole grid
    appear at once, so a selective read is visible as a selective read.
    """

    cells: list[AnalysisCell] = []
    for scope in _algorithm_scopes(pairs):
        scoped = [p for p in pairs if scope == POOLED_SCOPE or p.algorithm == scope]
        if not scoped:
            continue
        for variant, delta in _variant_keys(scoped):
            subset = [
                p for p in scoped if p.variant == variant and p.materiality_fraction == delta
            ]
            for scale in SCALES:
                cells.extend(
                    _cells_for(
                        subset,
                        predictor=OEH_PREDICTOR,
                        variant=variant,
                        materiality_fraction=delta,
                        scope=scope,
                        scale=scale,
                        censoring_rule=censoring_rule,
                        rng=rng,
                        censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
                        num_resamples=num_resamples,
                        confidence=confidence,
                    )
                )

        # The structural baseline does not depend on the variant, so it is
        # computed once per scope over the deduplicated label rows -- not once
        # per variant, which would repeat the same number eight times and make
        # the table look like eight independent baselines.
        baseline_rows = _deduplicate_label_rows(scoped)
        for scale in SCALES:
            cells.extend(
                _cells_for(
                    baseline_rows,
                    predictor=STRUCTURAL_PREDICTOR,
                    variant=STRUCTURAL_PREDICTOR,
                    materiality_fraction=None,
                    scope=scope,
                    scale=scale,
                    censoring_rule=censoring_rule,
                    rng=rng,
                    censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
                    num_resamples=num_resamples,
                    confidence=confidence,
                )
            )
    return [cell.as_row() for cell in cells]


def _deduplicate_label_rows(pairs: Sequence[JoinedPair]) -> list[JoinedPair]:
    """One row per ``(mdp_id, policy_family, algorithm)``.

    The structural predictor is a property of the MDP, so it must be scored on
    the label table, not on the variant-expanded table where each label row
    appears once per definition.
    """

    seen: dict[tuple[str, str, str], JoinedPair] = {}
    for pair in pairs:
        seen.setdefault((pair.mdp_id, pair.policy_family, pair.algorithm), pair)
    return list(seen.values())


def _cells_for(
    subset: Sequence[JoinedPair],
    *,
    predictor: str,
    variant: str,
    materiality_fraction: float | None,
    scope: str,
    scale: str,
    censoring_rule: str,
    rng: np.random.Generator,
    censored_n_emp_lower_bound_multiplier: float,
    num_resamples: int,
    confidence: float,
) -> list[AnalysisCell]:
    xs, ys, clusters, counts = select_pairs(
        subset,
        predictor=predictor,
        scale=scale,
        censoring_rule=censoring_rule,
        censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
    )
    return [
        AnalysisCell(
            predictor=predictor,
            variant=variant,
            materiality_fraction=materiality_fraction,
            algorithm_scope=scope,
            scale=scale,
            censoring_rule=censoring_rule,
            estimate=correlate(
                xs,
                ys,
                clusters=clusters,
                method=method,
                rng=rng,
                num_resamples=num_resamples,
                confidence=confidence,
            ),
            counts=counts,
        )
        for method in METHODS
    ]


BASELINE_COMPARISON_FIELDS: tuple[str, ...] = (
    "variant",
    "materiality_fraction",
    "analysis_role",
    "algorithm_scope",
    "scale",
    "censoring_rule",
    "method",
    "coefficient_predictor",
    "coefficient_baseline",
    "difference",
    "ci_low",
    "ci_high",
    "confidence",
    "n_pairs",
    "num_clusters",
    "num_resamples",
    "num_defined_replicates",
    "undefined_reason",
    "ci_undefined_reason",
)


def select_paired(
    pairs: Sequence[JoinedPair],
    *,
    scale: str,
    censoring_rule: str,
    censored_n_emp_lower_bound_multiplier: float,
) -> tuple[list[float], list[float], list[float], list[str], dict[str, int]]:
    """Variant and baseline predictors on **identical** rows.

    ``select_pairs`` scores the structural baseline on its own natural
    coverage, which is larger: ``S*A`` never abstains and never returns
    infinity, so it produces a number on rows where ``n_req_hat`` does not.
    That difference is itself a finding (``aggregation_2026-08-24/README.md``
    calls it a coverage disadvantage), so it is reported.

    A head-to-head, though, has to be like-for-like, and this function supplies
    that: exactly the rows the variant survives, with ``S*A`` read off the same
    rows. Both framings appear in the output; neither one silently stands in
    for the other.
    """

    xs, ys, clusters, counts = select_pairs(
        pairs,
        predictor=OEH_PREDICTOR,
        scale=scale,
        censoring_rule=censoring_rule,
        censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
    )
    baseline_xs, _, baseline_clusters, _ = select_pairs(
        pairs,
        predictor=STRUCTURAL_PREDICTOR,
        scale=scale,
        censoring_rule=censoring_rule,
        censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
    )
    if len(baseline_xs) != len(xs) or baseline_clusters != clusters:
        # The two selections differ only where n_req_hat is absent or
        # infinite, so re-run the baseline restricted to the variant's rows.
        eligible = {
            (pair.mdp_id, pair.policy_family, pair.algorithm)
            for pair in pairs
            if _is_kept(pair, censoring_rule)
        }
        matched = [
            pair
            for pair in pairs
            if (pair.mdp_id, pair.policy_family, pair.algorithm) in eligible
        ]
        baseline_xs, _, baseline_clusters, _ = select_pairs(
            matched,
            predictor=STRUCTURAL_PREDICTOR,
            scale=scale,
            censoring_rule=censoring_rule,
            censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
        )
    if len(baseline_xs) != len(xs) or baseline_clusters != clusters:
        raise ValueError(
            "Could not align the baseline to the variant's rows; a head-to-head must be "
            f"like-for-like ({len(baseline_xs)} baseline rows vs {len(xs)} variant rows)."
        )
    return xs, baseline_xs, ys, clusters, counts


def _is_kept(pair: JoinedPair, censoring_rule: str) -> bool:
    """Whether this row survives the rule with ``n_req_hat`` as the predictor."""

    if pair.n_req_hat is None:
        return False
    if np.isinf(pair.n_req_hat) and not _rule_keeps_infinite_n_req(censoring_rule):
        return False
    if pair.n_emp_censored:
        return _rule_keeps_censored_n_emp(censoring_rule)
    return pair.n_emp_transitions is not None


def run_baseline_comparison(
    pairs: Sequence[JoinedPair],
    *,
    censoring_rule: str,
    rng: np.random.Generator,
    censored_n_emp_lower_bound_multiplier: float,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[dict[str, Any]]:
    """The registered head-to-head: ``corr(n_req_hat) - corr(S*A)``, paired.

    Emitted **per algorithm** at the ``raw`` scale only. Both restrictions are
    statements, not omissions. The pooled-across-algorithms cell belongs to
    ``run_family_robustness`` (its ``all_families`` scope) so that one number
    has one home and one interval. At ``per_sa`` the baseline is the constant 1, so it has no
    correlation to subtract; "does the pilot-dependent part beat structure
    there" is not a difference at all but the question *is this coefficient
    distinguishable from zero*, which the interval in ``c2_correlations.csv``
    already answers. Emitting a difference against a constant would look like a
    test and be nothing.
    """

    rows: list[dict[str, Any]] = []
    for scope in _algorithm_scopes(pairs):
        # The pooled cell is deliberately NOT emitted here. It is the same
        # statistic as run_family_robustness's "all_families" row, and
        # computing it twice from independent bootstrap draws would publish two
        # slightly different intervals for one number -- an inconsistency a
        # reader has no way to resolve. The pooled head-to-head lives in
        # c2_family_robustness.csv, where it also carries its Holm correction.
        if scope == POOLED_SCOPE:
            continue
        scoped = [p for p in pairs if p.algorithm == scope]
        if not scoped:
            continue
        for variant, delta in _variant_keys(scoped):
            subset = [p for p in scoped if p.variant == variant and p.materiality_fraction == delta]
            xs, baseline_xs, ys, clusters, _ = select_paired(
                subset,
                scale="raw",
                censoring_rule=censoring_rule,
                censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
            )
            for method in METHODS:
                estimate = correlate_difference(
                    xs,
                    baseline_xs,
                    ys,
                    clusters=clusters,
                    method=method,
                    rng=rng,
                    num_resamples=num_resamples,
                    confidence=confidence,
                )
                rows.append(
                    {
                        "variant": variant,
                        "materiality_fraction": delta,
                        "analysis_role": analysis_role(variant, delta),
                        "algorithm_scope": scope,
                        "scale": "raw",
                        "censoring_rule": censoring_rule,
                        **estimate.as_row(),
                    }
                )
    return rows


FAMILY_ROBUSTNESS_FIELDS: tuple[str, ...] = (
    "family_scope",
    "families_included",
    "variant",
    "materiality_fraction",
    "analysis_role",
    "method",
    "coefficient_predictor",
    "coefficient_baseline",
    "difference",
    "ci_low",
    "ci_high",
    "p_value",
    "p_holm",
    "significant_after_holm",
    "beats_baseline",
    "n_pairs",
    "num_clusters",
    "undefined_reason",
)


def family_scopes(pairs: Sequence[JoinedPair]) -> list[tuple[str, tuple[str, ...]]]:
    """The registered robustness scopes, in reporting order.

    ``PLAN_Final.md`` sec 8.2 criterion 4 asks for the non-``Q*`` restriction;
    sec 7 additionally registers a leave-one-family-out check. Both are
    enumerated here rather than chosen per run, so the set of scopes cannot
    grow after a result is seen.
    """

    families = tuple(sorted({pair.policy_family for pair in pairs}))
    scopes: list[tuple[str, tuple[str, ...]]] = [(ALL_FAMILIES_SCOPE, families)]
    non_q_star = tuple(name for name in families if name != Q_STAR_DERIVED_FAMILY)
    if non_q_star and non_q_star != families:
        scopes.append((NON_Q_STAR_SCOPE, non_q_star))
    for dropped in families:
        remaining = tuple(name for name in families if name != dropped)
        if remaining:
            scopes.append((f"drop_{dropped}", remaining))
    return scopes


def run_family_robustness(
    pairs: Sequence[JoinedPair],
    *,
    censoring_rule: str,
    rng: np.random.Generator,
    censored_n_emp_lower_bound_multiplier: float,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Success criteria 3 and 4 (``PLAN_Final.md`` sec 8.2), Holm-corrected.

    Criterion 3 is "beats the strongest baseline, 95% CI of the difference
    excluding zero, after Holm correction". Criterion 4 is that same conclusion
    restricted to the four behavior families **not** generated from ``Q*``.
    Both are answered from one table here, pooled across algorithms, at the
    ``raw`` scale where the baseline has a correlation to be compared against.

    Holm runs **within each family scope**, over that scope's definition x
    method comparisons. Correcting across scopes as well would be correcting
    across restatements of the same rows, which inflates the family size
    without adding independent tests.
    """

    rows: list[dict[str, Any]] = []
    for scope_name, families in family_scopes(pairs):
        scoped = [pair for pair in pairs if pair.policy_family in families]
        if not scoped:
            continue
        scope_rows: list[dict[str, Any]] = []
        for variant, delta in _variant_keys(scoped):
            subset = [p for p in scoped if p.variant == variant and p.materiality_fraction == delta]
            xs, baseline_xs, ys, clusters, _ = select_paired(
                subset,
                scale="raw",
                censoring_rule=censoring_rule,
                censored_n_emp_lower_bound_multiplier=censored_n_emp_lower_bound_multiplier,
            )
            for method in METHODS:
                estimate = correlate_difference(
                    xs, baseline_xs, ys, clusters=clusters, method=method,
                    rng=rng, num_resamples=num_resamples, confidence=confidence,
                )
                scope_rows.append(
                    {
                        "family_scope": scope_name,
                        "families_included": "|".join(families),
                        "variant": variant,
                        "materiality_fraction": delta,
                        "analysis_role": analysis_role(variant, delta),
                        "method": method,
                        "coefficient_predictor": estimate.coefficient_predictor,
                        "coefficient_baseline": estimate.coefficient_baseline,
                        "difference": estimate.difference,
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                        "p_value": estimate.p_value,
                        "n_pairs": estimate.n_pairs,
                        "num_clusters": estimate.num_clusters,
                        "undefined_reason": estimate.undefined_reason,
                    }
                )

        adjusted, significant = holm_adjust([row["p_value"] for row in scope_rows], alpha=alpha)
        for row, p_holm, is_significant in zip(scope_rows, adjusted, significant):
            row["p_holm"] = p_holm
            row["significant_after_holm"] = is_significant
            # "Beats the baseline" is a one-sided claim: significant AND on the
            # right side of zero. A significant negative difference is the
            # opposite finding and must never be counted as passing.
            row["beats_baseline"] = bool(is_significant and row["difference"] > 0.0)
        rows.extend(scope_rows)
    return rows


def success_criteria_3_and_4(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the robustness table to the two decision-gate booleans.

    Returns the fields ``evaluation/decision_gate/gate.py`` reads as
    ``beats_strongest_baseline_ci_excludes_zero`` and
    ``holds_on_non_q_star_families``. Each is ``None`` only when the scope was
    not measured at all -- never ``False`` by default, because "unmeasured"
    and "measured and failed" are different states and the gate treats them
    differently.
    """

    def _verdict(scope: str) -> tuple[bool | None, list[str]]:
        scoped = [row for row in rows if row["family_scope"] == scope]
        if not scoped:
            return None, []
        winners = [
            f"{row['variant']}/{row['method']}" for row in scoped if row["beats_baseline"]
        ]
        losers = [
            f"{row['variant']}/{row['method']} (diff {row['difference']:+.3f}, "
            f"p_holm {row['p_holm']:.3f})"
            for row in scoped
            if row["significant_after_holm"] and row["difference"] < 0.0
        ]
        return bool(winners), winners or losers

    beats, beats_detail = _verdict(ALL_FAMILIES_SCOPE)
    holds, holds_detail = _verdict(NON_Q_STAR_SCOPE)
    return {
        "beats_strongest_baseline_ci_excludes_zero": beats,
        "holds_on_non_q_star_families": holds,
        "all_families_detail": beats_detail,
        "non_q_star_detail": holds_detail,
        "criterion_3": "PLAN_Final.md sec 8.2 item 3 (Holm-corrected CI of the difference)",
        "criterion_4": "PLAN_Final.md sec 8.2 item 4 (same conclusion on the four non-Q* families)",
    }


def multiplicity_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many cells were computed, and how many intervals exclude zero.

    A 95% interval excludes zero in roughly one cell in twenty by chance. This
    table has dozens of cells, so the count of exclusions is only readable next
    to the count of cells -- and the cells that do exclude zero are listed by
    name so that quoting one of them in isolation is visibly a choice.
    """

    considered = [row for row in rows if row["undefined_reason"] is None]
    excluding: list[dict[str, Any]] = []
    for row in considered:
        low, high = row["ci_low"], row["ci_high"]
        if np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0):
            excluding.append(
                {
                    "cell": f"{row['algorithm_scope']}/{row['variant']}/{row['scale']}/{row['method']}",
                    "coefficient": row["coefficient"],
                    "ci": [low, high],
                    "n_pairs": row["n_pairs"],
                    "num_clusters": row["num_clusters"],
                }
            )
    by_scale: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        scale_cells = [row for row in considered if row["scale"] == scale]
        scale_excluding = [item for item in excluding if f"/{scale}/" in item["cell"]]
        by_scale[scale] = {
            "cells_with_a_defined_coefficient": len(scale_cells),
            "cells_whose_ci_excludes_zero": len(scale_excluding),
            "expected_by_chance_at_95_percent": round(0.05 * len(scale_cells), 1),
            "cells_excluding_zero": scale_excluding,
        }

    return {
        "cells_emitted": len(rows),
        "cells_with_a_defined_coefficient": len(considered),
        "cells_whose_ci_excludes_zero": len(excluding),
        "expected_by_chance_at_95_percent": round(0.05 * len(considered), 1),
        "by_scale": by_scale,
        "cells_excluding_zero": excluding,
        "note": (
            "Uncorrected, and the split by scale is the part to read. A raw-scale interval "
            "excluding zero says only that bigger problems need more data -- S*A is inside "
            "n_req_hat by construction, so those cells were never in question. The scientific "
            "question lives at per_sa, and a single significant per_sa cell must be read against "
            "the per_sa denominator here (PLAN_Final.md sec 7 requires Holm correction across "
            "the ablation family before any such cell is reported as a finding)."
        ),
    }


def beats_structural_baseline(
    rows: Sequence[Mapping[str, Any]],
    *,
    scale: str = "per_sa",
    method: str = "spearman",
    algorithm_scope: str = POOLED_SCOPE,
) -> list[dict[str, Any]]:
    """Read C2 off the table: variant coefficient vs the baseline's, same cell.

    Returns one verdict row per variant. The verdict is deliberately *not* a
    hypothesis test on the difference -- with the baseline undefined at
    ``per_sa`` (it is a constant there) the comparison at that scale is
    "does the pilot-dependent part correlate at all", which is exactly what
    ``WEEK2_DECISION_GATE.md`` sec 10b asked and answered with 0.206 / -0.169.
    The honest summary is the coefficient and its CI, so that is what this
    returns; ``ci_excludes_zero`` is stated as a fact about the interval, not
    dressed up as a p-value.
    """

    verdicts: list[dict[str, Any]] = []
    for row in rows:
        if row["scale"] != scale or row["method"] != method:
            continue
        if row["algorithm_scope"] != algorithm_scope:
            continue
        if row["predictor"] != OEH_PREDICTOR:
            continue
        low, high = row["ci_low"], row["ci_high"]
        defined = row["undefined_reason"] is None
        ci_defined = defined and np.isfinite(low) and np.isfinite(high)
        verdicts.append(
            {
                "variant": row["variant"],
                "materiality_fraction": row["materiality_fraction"],
                "analysis_role": row["analysis_role"],
                "scale": scale,
                "method": method,
                "algorithm_scope": algorithm_scope,
                "coefficient": row["coefficient"],
                "ci_low": low,
                "ci_high": high,
                "n_pairs": row["n_pairs"],
                "num_clusters": row["num_clusters"],
                "ci_excludes_zero": bool(ci_defined and (low > 0.0 or high < 0.0)),
                "undefined_reason": row["undefined_reason"],
            }
        )
    return verdicts


__all__ = [
    "ALL_FAMILIES_SCOPE",
    "BASELINE_COMPARISON_FIELDS",
    "FAMILY_ROBUSTNESS_FIELDS",
    "NON_Q_STAR_SCOPE",
    "Q_STAR_DERIVED_FAMILY",
    "CENSORING_RULES",
    "DEFAULT_CENSORING_RULE",
    "DISPOSITIONS",
    "PRIMARY_MATERIALITY_FRACTION",
    "THRESHOLD_DEPENDENT_VARIANTS",
    "analysis_role",
    "OEH_PREDICTOR",
    "OUTPUT_FIELDS",
    "POOLED_SCOPE",
    "SCALES",
    "STRUCTURAL_PREDICTOR",
    "JoinedPair",
    "beats_structural_baseline",
    "family_scopes",
    "multiplicity_report",
    "run_baseline_comparison",
    "run_family_robustness",
    "success_criteria_3_and_4",
    "run_c2_analysis",
    "select_paired",
    "select_pairs",
]
