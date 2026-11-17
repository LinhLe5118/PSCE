"""Observed / censored / abstained / invalid classification.

Two independent axes exist and are never merged (``WEEK2_EVALUATION_PROTOCOL.md``
sec 5):

* Axis A -- OEH pilot-selection outcome (``code/adaptive_pilot.py`` /
  ``code/oeh_estimator.py``).
* Axis B -- ``N_emp`` label outcome (``code/empirical_requirement.py``).

``invalid`` is never produced by the classifiers here: it is assigned by
``evaluation.validation.schema`` when an artifact row fails schema validation,
before it ever reaches these functions. A classifier that could itself
produce ``invalid`` from otherwise-valid fields would blur "the estimator
declined" with "the pipeline is broken," which is exactly the distinction
mega-task section 7 requires.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

STATUS_VALUES = ("observed", "censored", "abstained", "invalid")

_SELECTION_STATUS_MAP = {
    "selected": "observed",
    "censored_at_cap": "censored",
    "abstained_at_cap": "abstained",
    "insufficient_bootstrap_at_cap": "abstained",
}


def classify_oeh_status(
    selection_status: str | None = None,
    *,
    abstained: bool | None = None,
    censored_right: bool | None = None,
) -> str:
    """Classify an Axis-A outcome.

    Either pass ``selection_status`` (the string from
    ``AdaptivePilotSelection.status``) or the raw ``OEHResult`` booleans
    (``abstained``, ``censored_right``). Passing both is an error -- pick one
    source of truth per call site instead of letting them silently disagree.
    """

    if selection_status is not None:
        if abstained is not None or censored_right is not None:
            raise ValueError(
                "Pass either selection_status or (abstained, censored_right), not both."
            )
        if selection_status not in _SELECTION_STATUS_MAP:
            raise ValueError(
                f"Unrecognized adaptive_pilot selection status: {selection_status!r}"
            )
        return _SELECTION_STATUS_MAP[selection_status]

    if abstained is None or censored_right is None:
        raise ValueError("Must supply both abstained and censored_right.")
    if abstained:
        return "abstained"
    if censored_right:
        return "censored"
    return "observed"


def classify_n_emp_status(*, right_censored: bool, requirement: int | None) -> str:
    """Classify an Axis-B outcome. Axis B has no ``abstained`` state."""

    if right_censored:
        if requirement is not None:
            raise ValueError(
                "A right-censored N_emp observation must not carry a resolved requirement."
            )
        return "censored"
    if requirement is None:
        raise ValueError(
            "A non-censored N_emp observation must carry a resolved requirement."
        )
    return "observed"


def censoring_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    status_field: str,
    group_by: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate per-row status into censoring/abstention/valid rates per group.

    Every group total is the group's row count; the four status counts always
    sum to it. A group with zero rows of a given status still reports rate
    0.0 for it, never an omitted field.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status = row[status_field]
        if status not in STATUS_VALUES:
            raise ValueError(f"Unknown status {status!r}; expected one of {STATUS_VALUES}.")
        key = tuple(row[field] for field in group_by)
        groups[key].append(row)

    report: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        total = len(group_rows)
        counts = {status: 0 for status in STATUS_VALUES}
        for row in group_rows:
            counts[row[status_field]] += 1
        entry: dict[str, Any] = dict(zip(group_by, key))
        entry["n_total"] = total
        for status in STATUS_VALUES:
            entry[f"n_{status}"] = counts[status]
            entry[f"{status}_rate"] = counts[status] / total if total else float("nan")
        entry["valid_rate"] = entry["observed_rate"]
        report.append(entry)
    return report


__all__ = [
    "STATUS_VALUES",
    "censoring_report",
    "classify_n_emp_status",
    "classify_oeh_status",
]
