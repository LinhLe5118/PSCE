"""Rank and linear correlation with a cluster-bootstrap confidence interval.

Written for C2 (``PHASE_B_PLAN.md`` sec 1): *does ``n_req_hat`` under a
corrected definition carry information beyond the structural-only baseline
``S*A``?* The pre-registered check it serves is ``PLAN_Final.md`` sec 6.1's
structural-only baseline, and the shape of the answer is fixed by
``WEEK2_DECISION_GATE.md`` sec 10b: Spearman **and** Pearson, with and without
``S*A`` divided out.

Three design points, each of which is a decision and not an implementation
detail:

* **The CI resamples base MDPs, not rows** (``PLAN_Final.md`` sec 7). The 150
  joined rows are 15 MDPs x 5 behavior families x 2 algorithms; rows sharing an
  ``mdp_id`` are not independent, so a row bootstrap would report an interval
  several times too narrow. ``cluster_bootstrap_ci`` in
  ``evaluation/aggregation/aggregate.py`` already does cluster resampling but
  takes a statistic over one pooled vector; a correlation needs *paired*
  values, so it cannot be expressed through that signature. Same principle,
  different arity -- not a second bootstrap mechanism.
* **An undefined correlation is ``nan`` with a stated reason, never 0.0.**
  Fewer than three pairs, or a constant vector on either side, means the
  question was not answered. Reporting 0.0 there would read as "measured, no
  relationship," which is a different claim. This matters concretely: the
  structural baseline divided by ``S*A`` *is* a constant, so this path is
  taken by a row that will appear in the real output table.
* **``pearson`` refuses non-finite input; ``spearman`` accepts ``+inf``.**
  ``n_req_hat = inf`` is a definite answer ("no finite amount of data
  suffices", ``alternative_definitions_2026-09-01/README.md``), so it has a
  well-defined *rank* but no product-moment contribution. Coercing it to a
  large finite stand-in is the substitution ``PLAN_Final.md`` sec 3.1
  prohibits for ``N_emp``, and the same argument applies on this axis. ``nan``
  (a missing measurement) is rejected by both: the caller must dispose of it
  through an explicit censoring rule, not by letting a correlation function
  quietly drop it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

MIN_PAIRS = 3
METHODS = ("spearman", "pearson")
DEFAULT_NUM_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
# A bootstrap whose replicates are mostly undefined has not estimated an
# interval; it has reported that most resamples could not answer the question.
MIN_DEFINED_REPLICATE_FRACTION = 0.5


@dataclass(frozen=True)
class CorrelationEstimate:
    """A coefficient plus everything needed to read it honestly.

    ``undefined_reason`` is ``None`` exactly when ``coefficient`` is finite.
    ``ci_low``/``ci_high`` may still be ``nan`` on a defined coefficient (too
    few clusters, or too few defined bootstrap replicates), in which case
    ``ci_undefined_reason`` says which.
    """

    method: str
    coefficient: float
    ci_low: float
    ci_high: float
    confidence: float
    n_pairs: int
    num_clusters: int
    num_resamples: int
    num_defined_replicates: int
    undefined_reason: str | None = None
    ci_undefined_reason: str | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _as_pair_arrays(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must be the same length; got {xs.shape} and {ys.shape}.")
    if xs.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if np.isnan(xs).any() or np.isnan(ys).any():
        raise ValueError(
            "nan in correlation input. A missing measurement must be disposed of by an "
            "explicit censoring rule before it reaches this function (EXPERIMENT_RULE.md "
            "rule 2), never dropped silently here."
        )
    return xs, ys


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged. ``+inf`` ranks last, as it should: it is an
    ordered value, not a missing one."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, sorted_values.shape[0] + 1):
        if index == sorted_values.shape[0] or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index
    return ranks


def _product_moment(xs: np.ndarray, ys: np.ndarray) -> float:
    x_centered = xs - xs.mean()
    y_centered = ys - ys.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Product-moment correlation. ``nan`` if either side is constant.

    Raises on ``+/-inf``: an infinite ``n_req_hat`` has no product-moment
    contribution, and substituting a finite stand-in for it would manufacture
    the number. Under a censoring rule that keeps infinities, Pearson is
    reported as undefined -- that is the honest outcome, not a gap to fill.
    """

    xs, ys = _as_pair_arrays(x, y)
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError(
            "Pearson correlation is undefined on non-finite values. An infinite n_req_hat is "
            "a definite answer with a rank but no product-moment contribution; report Pearson "
            "as undefined rather than substituting a finite stand-in."
        )
    if xs.shape[0] < MIN_PAIRS:
        return float("nan")
    return _product_moment(xs, ys)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, ties averaged. ``+inf`` is a legitimate largest value."""

    xs, ys = _as_pair_arrays(x, y)
    if xs.shape[0] < MIN_PAIRS:
        return float("nan")
    return _product_moment(average_ranks(xs), average_ranks(ys))


def _coefficient(method: str, xs: np.ndarray, ys: np.ndarray) -> float:
    if method == "spearman":
        return spearman(xs, ys)
    if method == "pearson":
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            return float("nan")
        return pearson(xs, ys)
    raise ValueError(f"Unknown correlation method {method!r}; expected one of {METHODS}.")


def _undefined_reason(method: str, xs: np.ndarray, ys: np.ndarray) -> str | None:
    if xs.shape[0] < MIN_PAIRS:
        return f"fewer than {MIN_PAIRS} pairs (n={xs.shape[0]})"
    if method == "pearson" and not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        return "non-finite value present; Pearson has no product-moment contribution for it"
    values = (average_ranks(xs), average_ranks(ys)) if method == "spearman" else (xs, ys)
    for name, vector in zip(("predictor", "label"), values):
        if float(np.ptp(vector)) == 0.0:
            return f"{name} is constant, so the correlation is undefined (not zero)"
    return None


def correlate(
    x: Sequence[float],
    y: Sequence[float],
    *,
    clusters: Sequence[Any],
    method: str,
    rng: np.random.Generator,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CorrelationEstimate:
    """Correlate ``x`` against ``y`` and bracket it with a cluster bootstrap.

    ``clusters`` is one label per pair (``mdp_id`` in this project). Resampling
    is over the distinct labels with replacement; every row of a resampled MDP
    enters together, so the interval carries the within-MDP dependence that a
    row bootstrap would throw away.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown correlation method {method!r}; expected one of {METHODS}.")
    xs, ys = _as_pair_arrays(x, y)
    labels = np.asarray(list(clusters), dtype=object)
    if labels.shape[0] != xs.shape[0]:
        raise ValueError(
            f"clusters must have one label per pair; got {labels.shape[0]} for {xs.shape[0]} pairs."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    reason = _undefined_reason(method, xs, ys)
    point = float("nan") if reason is not None else _coefficient(method, xs, ys)

    distinct = list(dict.fromkeys(labels.tolist()))
    index_by_cluster = {label: np.flatnonzero(labels == label) for label in distinct}
    num_clusters = len(distinct)

    if reason is not None:
        ci_reason: str | None = "coefficient is undefined"
        replicates = np.empty(0, dtype=np.float64)
    elif num_clusters < 2:
        ci_reason = f"cluster bootstrap needs >= 2 clusters; got {num_clusters}"
        replicates = np.empty(0, dtype=np.float64)
    else:
        ci_reason = None
        drawn = np.empty(num_resamples, dtype=np.float64)
        for b in range(num_resamples):
            picked = rng.integers(0, num_clusters, size=num_clusters)
            index = np.concatenate([index_by_cluster[distinct[int(p)]] for p in picked])
            drawn[b] = _coefficient(method, xs[index], ys[index])
        replicates = drawn[np.isfinite(drawn)]
        if replicates.shape[0] < MIN_DEFINED_REPLICATE_FRACTION * num_resamples:
            ci_reason = (
                f"only {replicates.shape[0]}/{num_resamples} bootstrap replicates were defined; "
                "a resampled table that usually cannot answer the question does not bracket one "
                "that can"
            )

    if ci_reason is None and replicates.shape[0] > 0:
        alpha = (1.0 - confidence) / 2.0
        low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
        ci_low, ci_high = float(low), float(high)
    else:
        ci_low = ci_high = float("nan")

    return CorrelationEstimate(
        method=method,
        coefficient=point,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_pairs=int(xs.shape[0]),
        num_clusters=num_clusters,
        num_resamples=num_resamples if ci_reason is None else 0,
        num_defined_replicates=int(replicates.shape[0]),
        undefined_reason=reason,
        ci_undefined_reason=ci_reason,
    )


@dataclass(frozen=True)
class DifferenceEstimate:
    """``corr(predictor, y) - corr(baseline, y)`` on one set of rows.

    This is the statistic ``PLAN_Final.md`` sec 7 registers for "beats the
    strongest baseline": the interval of the **difference**, not two intervals
    side by side. Two overlapping CIs do not imply the difference contains
    zero, and two disjoint CIs are a stronger claim than the data supports --
    reading a head-to-head off a pair of marginal intervals is the mistake this
    type exists to prevent.
    """

    method: str
    coefficient_predictor: float
    coefficient_baseline: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float
    n_pairs: int
    num_clusters: int
    num_resamples: int
    num_defined_replicates: int
    undefined_reason: str | None = None
    ci_undefined_reason: str | None = None

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def correlate_difference(
    x_predictor: Sequence[float],
    x_baseline: Sequence[float],
    y: Sequence[float],
    *,
    clusters: Sequence[Any],
    method: str,
    rng: np.random.Generator,
    num_resamples: int = DEFAULT_NUM_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> DifferenceEstimate:
    """Paired cluster bootstrap of the difference between two correlations.

    Both correlations are recomputed on **the same resampled clusters** in each
    replicate. Pairing is the whole point: predictor and baseline are scored on
    identical rows, so the resampling noise they share cancels out of the
    difference instead of being counted twice.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown correlation method {method!r}; expected one of {METHODS}.")
    xs_p, ys = _as_pair_arrays(x_predictor, y)
    xs_b, _ = _as_pair_arrays(x_baseline, y)
    labels = np.asarray(list(clusters), dtype=object)
    if not labels.shape[0] == xs_p.shape[0] == xs_b.shape[0]:
        raise ValueError("predictor, baseline, label and cluster arrays must all be the same length.")

    reason = _undefined_reason(method, xs_p, ys) or _undefined_reason(method, xs_b, ys)
    point_p = float("nan") if reason else _coefficient(method, xs_p, ys)
    point_b = float("nan") if reason else _coefficient(method, xs_b, ys)
    difference = point_p - point_b

    distinct = list(dict.fromkeys(labels.tolist()))
    index_by_cluster = {label: np.flatnonzero(labels == label) for label in distinct}
    num_clusters = len(distinct)

    if reason is not None:
        ci_reason: str | None = "difference is undefined"
        replicates = np.empty(0, dtype=np.float64)
    elif num_clusters < 2:
        ci_reason = f"cluster bootstrap needs >= 2 clusters; got {num_clusters}"
        replicates = np.empty(0, dtype=np.float64)
    else:
        ci_reason = None
        drawn = np.empty(num_resamples, dtype=np.float64)
        for b in range(num_resamples):
            picked = rng.integers(0, num_clusters, size=num_clusters)
            index = np.concatenate([index_by_cluster[distinct[int(p)]] for p in picked])
            drawn[b] = _coefficient(method, xs_p[index], ys[index]) - _coefficient(
                method, xs_b[index], ys[index]
            )
        replicates = drawn[np.isfinite(drawn)]
        if replicates.shape[0] < MIN_DEFINED_REPLICATE_FRACTION * num_resamples:
            ci_reason = (
                f"only {replicates.shape[0]}/{num_resamples} bootstrap replicates were defined"
            )

    if ci_reason is None and replicates.shape[0] > 0:
        alpha = (1.0 - confidence) / 2.0
        low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
        ci_low, ci_high = float(low), float(high)
    else:
        ci_low = ci_high = float("nan")

    # Two-sided bootstrap p-value, needed because Holm correction
    # (PLAN_Final.md sec 7) is defined on p-values and an interval alone
    # cannot be adjusted for multiplicity.
    #
    # Both tails are counted **inclusive of zero**. That detail decides real
    # cells here: when predictor and baseline are near-identical variables,
    # most replicates land exactly on 0.0, and counting only the strict tails
    # would report p = 0 for a distribution that is mostly a point mass at no
    # difference at all -- manufacturing significance out of ties. Inclusive
    # tails give p -> 1 there, which is the honest reading.
    if replicates.shape[0] > 0:
        at_or_below = float(np.mean(replicates <= 0.0))
        at_or_above = float(np.mean(replicates >= 0.0))
        p_value = min(1.0, 2.0 * min(at_or_below, at_or_above))
    else:
        p_value = float("nan")

    return DifferenceEstimate(
        method=method,
        coefficient_predictor=point_p,
        coefficient_baseline=point_b,
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        confidence=confidence,
        n_pairs=int(xs_p.shape[0]),
        num_clusters=num_clusters,
        num_resamples=num_resamples if ci_reason is None else 0,
        num_defined_replicates=int(replicates.shape[0]),
        undefined_reason=reason,
        ci_undefined_reason=ci_reason,
    )


def holm_adjust(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down over one family of comparisons.

    ``PLAN_Final.md`` sec 7 registers Holm correction across the ablation
    family, and sec 8.2 criterion 3 is stated in terms of the corrected
    result. Applying it is therefore not a conservatism choice made here --
    it is the registered procedure, and reporting the uncorrected cell instead
    would be reporting a different test than the one pre-specified.

    ``nan`` p-values (an undefined comparison) never count toward the family
    size and never come back significant; they are returned as ``nan``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    values = list(p_values)
    testable = [(index, p) for index, p in enumerate(values) if not math.isnan(p)]
    family_size = len(testable)

    adjusted = [float("nan")] * len(values)
    significant = [False] * len(values)
    running = 0.0
    for rank, (index, p) in enumerate(sorted(testable, key=lambda item: item[1])):
        running = max(running, min(1.0, (family_size - rank) * p))
        adjusted[index] = running
        significant[index] = running <= alpha
    return adjusted, significant


__all__ = [
    "CorrelationEstimate",
    "DifferenceEstimate",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_NUM_RESAMPLES",
    "METHODS",
    "MIN_PAIRS",
    "average_ranks",
    "correlate",
    "correlate_difference",
    "holm_adjust",
    "pearson",
    "spearman",
]
