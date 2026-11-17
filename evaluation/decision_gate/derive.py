"""Derive ``DecisionGateInputs`` from the contracted Week 2 artifacts.

Closes the gap ``docs/protocol/PROTOCOL_DECISIONS_2026-08-16.md`` D9.4 left
open: ``pilot/aggregate.py::run()`` called ``build_decision_gate_input`` with
only ``tests_passing``/``no_leakage_violations``/``reproducible``, so the JSON
it wrote read as *"NO_GO because no artifacts were supplied"* even when the
artifacts were right there. The real decision was carried by a hand-assembled
file (D7.4). Hand-assembly is exactly the manual-synthesis risk
``docs/week2/WEEK2_ARTIFACT_CONTRACT.md`` sec 5 was written to remove, so the
derivation lives here instead, in one place, testable.

**This module does not decide anything.** It reads columns and returns the
inputs; ``gate.py::decide`` applies ``PLAN_Final.md`` sec 10's table to them.

The one place a judgement call was unavoidable is K1a, and it is not hidden:

K1a is ``median N_k / N_emp`` (sec 8.1). ``N_emp`` is right-censored on most
development rows, and sec 8.1 says to treat those conservatively -- the
criterion's own words are *"bao thu voi ca bi censor"*. ``comparisons.py``
implements that as ``N_k / N_max``: the true ``N_emp`` exceeds ``N_max``, so
``N_k / N_max`` is an upper bound on the true, smaller ratio, and clearing the
threshold against it is sound.

Sound, but on this dataset it is also nearly vacuous, and that has to be
visible rather than inferred. The pilot budget is capped at ``10*|S||A|`` and
``N_max = 50*|S||A|``, so **every** censored row yields a ratio of at most
``0.2`` by arithmetic alone, under the 0.25 threshold, whatever the estimator
did. With 69% of labels censored the median is decided by that arithmetic.

So ``derive_gate_inputs`` computes three framings and reports all three
(``k1a_framings``), while feeding the registered one to the gate:

* ``registered_conservative`` -- sec 8.1 as written, ``N_max`` for censored
  rows. **This is the one the gate consumes.**
* ``finite_labels_only`` -- censored rows dropped instead of bounded. Answers
  "among problems we actually resolved, was the depth identified early?"
* ``selected_budget_finite_labels`` -- the selected pilot budget rather than
  the stabilization budget ``N_k``, on finite labels. This is the framing the
  2026-08-24 hand-assembled `decision_gate_input.json` used, reproduced here
  so the two files can be compared instead of merely differing.

Reporting all three is ``PLAN_Final.md`` sec 11 rule 7 applied to a definition
choice rather than to a result: the framings disagree about K1a, and a reader
is entitled to see that they do.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.aggregation.comparisons import early_identification_ratio, stabilized_pilot_budget
from evaluation.decision_gate.gate import DecisionGateInputs

OBSERVED = "observed"
ANSWERED_STATUSES = frozenset({"observed", "censored"})


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return int(value)


@dataclass(frozen=True)
class K1aFraming:
    """One way of computing K1a, with the sample it was computed on."""

    name: str
    description: str
    median_by_algorithm: dict[str, float]
    count_by_algorithm: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "median_by_algorithm": self.median_by_algorithm,
            "count_by_algorithm": self.count_by_algorithm,
        }


def states_actions_by_mdp(benchmark_metadata: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        row["mdp_id"]: int(row["num_states"]) * int(row["num_actions"])
        for row in benchmark_metadata
    }


def stabilization_budgets(
    oeh_rows: Sequence[Mapping[str, Any]], states_actions: Mapping[str, int]
) -> dict[tuple[str, str], int | None]:
    """``N_k`` per (mdp_id, policy_family): the smallest pilot budget from which
    ``k_dagger_oeh`` no longer moves. Needs **every** attempted pilot size, not
    just the selected one, which is why ``pilot/aggregate.py`` passes the
    unfiltered ``oeh_results`` rows here and the selected-only rows elsewhere.
    """

    attempts_by_pair: dict[tuple[str, str], list[tuple[int, int | None]]] = defaultdict(list)
    for row in oeh_rows:
        key = (row["mdp_id"], row["policy_family"])
        attempts_by_pair[key].append(
            (int(row["pilot_multiplier"]), _optional_int(row.get("k_dagger_oeh")))
        )

    budgets: dict[tuple[str, str], int | None] = {}
    for key, attempts in attempts_by_pair.items():
        mdp_id = key[0]
        if mdp_id not in states_actions:
            raise KeyError(
                f"{mdp_id} appears in oeh_results.csv but not in benchmark_metadata.csv; "
                "K1a cannot be computed without |S||A| (WEEK2_ARTIFACT_CONTRACT.md sec 1)."
            )
        budgets[key] = stabilized_pilot_budget(
            sorted(attempts), num_states_actions=states_actions[mdp_id]
        )
    return budgets


def selected_budgets(
    oeh_rows: Sequence[Mapping[str, Any]], states_actions: Mapping[str, int]
) -> dict[tuple[str, str], int]:
    return {
        (row["mdp_id"], row["policy_family"]): int(row["pilot_multiplier"]) * states_actions[row["mdp_id"]]
        for row in oeh_rows
        if _is_true(row.get("selected"))
    }


def k1a_framings(
    *,
    oeh_rows: Sequence[Mapping[str, Any]],
    n_emp_rows: Sequence[Mapping[str, Any]],
    benchmark_metadata: Sequence[Mapping[str, Any]],
    max_size_multiplier: int,
) -> list[K1aFraming]:
    states_actions = states_actions_by_mdp(benchmark_metadata)
    n_k = stabilization_budgets(oeh_rows, states_actions)
    selected = selected_budgets(oeh_rows, states_actions)

    buckets: dict[str, dict[str, list[float]]] = {
        "registered_conservative": defaultdict(list),
        "finite_labels_only": defaultdict(list),
        "selected_budget_finite_labels": defaultdict(list),
    }

    for row in n_emp_rows:
        key = (row["mdp_id"], row["policy_family"])
        sa = states_actions.get(row["mdp_id"])
        if sa is None or key not in n_k:
            continue
        algorithm = row["algorithm"]
        censored = _is_true(row.get("n_emp_censored"))
        transitions = _optional_int(row.get("n_emp_transitions"))

        conservative = early_identification_ratio(
            n_k=n_k[key],
            n_emp_transitions=transitions,
            n_emp_censored=censored,
            n_max_transitions=max_size_multiplier * sa,
        )
        if conservative == conservative:  # not NaN
            buckets["registered_conservative"][algorithm].append(conservative)
            if not censored:
                buckets["finite_labels_only"][algorithm].append(conservative)

        if not censored and transitions and key in selected:
            buckets["selected_budget_finite_labels"][algorithm].append(selected[key] / transitions)

    descriptions = {
        "registered_conservative": (
            "PLAN_Final.md sec 8.1 as written: N_k / N_emp, with N_max substituted for a "
            "right-censored N_emp as the registered conservative bound. Fed to the gate. "
            "Note that a capped pilot budget makes every censored row <= "
            "pilot_cap / N_max by arithmetic, so on a heavily censored split this "
            "framing is close to vacuous."
        ),
        "finite_labels_only": (
            "Same N_k, but censored labels dropped rather than bounded: among the problems "
            "whose data requirement was actually resolved, was the depth identified early?"
        ),
        "selected_budget_finite_labels": (
            "The selected pilot budget instead of the stabilization budget N_k, on finite "
            "labels. Reproduces the framing of the hand-assembled "
            "artifacts/week2/decision_gate_input.json (2026-08-24)."
        ),
    }

    framings: list[K1aFraming] = []
    for name, per_algorithm in buckets.items():
        framings.append(
            K1aFraming(
                name=name,
                description=descriptions[name],
                median_by_algorithm={
                    algorithm: statistics.median(values)
                    for algorithm, values in sorted(per_algorithm.items())
                    if values
                },
                count_by_algorithm={
                    algorithm: len(values) for algorithm, values in sorted(per_algorithm.items())
                },
            )
        )
    return framings


def exact_curve_has_interior_minimum(exact_k_curve_rows: Sequence[Mapping[str, Any]], *, k_max: int) -> bool:
    """K5 (``PLAN_Final.md`` sec 8.1): does the exact ``m_k`` curve have an
    interior minimum? True only if some curve puts ``k_dagger_exact`` strictly
    inside ``(0, k_max)`` **and** its multiplier is not constant -- a flat
    curve whose argmin lands at an interior index by tie-break is not a
    minimum, it is a tie, and reporting it as K5 would manufacture the
    project's central claim out of a tolerance.
    """

    by_curve: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in exact_k_curve_rows:
        by_curve[(row["mdp_id"], row["policy_family"])].append(row)

    for rows in by_curve.values():
        k_dagger = _optional_int(rows[0].get("k_dagger_exact"))
        if k_dagger is None or not (0 < k_dagger < k_max):
            continue
        multipliers = {
            float(row["multiplier"])
            for row in rows
            if row.get("multiplier") not in (None, "", "None")
        }
        if len(multipliers) > 1:
            return True
    return False


def derive_gate_inputs(
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_size_multiplier: int,
    k_max: int,
    tests_passing: bool,
    no_leakage_violations: bool,
    reproducible: bool,
    baselines_present: bool,
    coverage_grid_expansion_feasible: bool,
    k1a_framing: str = "registered_conservative",
    c2_criteria: Mapping[str, Any] | None = None,
) -> tuple[DecisionGateInputs, dict[str, Any]]:
    """Returns the gate inputs plus a provenance record of how each number was
    derived, so the JSON the gate writes can be audited without rereading this
    module.

    ``coverage_grid_expansion_feasible`` is a caller-supplied decision, not a
    measurement: ``PLAN_Final.md`` sec 7 requires it to be settled *at the
    gate* rather than after seeing results, so it is never inferred from the
    censoring rate here.

    ``c2_criteria`` optionally carries success criteria 3 and 4 from
    ``scripts/analysis/run_c2_correlation.py``'s ``c2_decision_inputs.json``. Absent it, both
    stay ``None`` -- "not measured", which the gate treats differently from
    ``False`` ("measured and failed").
    """

    oeh_rows = list(artifacts["oeh_results"])
    selected = [row for row in oeh_rows if _is_true(row.get("selected"))]
    if not selected:
        raise ValueError("oeh_results.csv contains no selected rows; K1a and censoring are undefined.")

    statuses = [row["oeh_status"] for row in selected]
    observed = sum(1 for status in statuses if status == OBSERVED)
    censored = sum(1 for status in statuses if status == "censored")
    abstained = sum(1 for status in statuses if status == "abstained")

    framings = k1a_framings(
        oeh_rows=oeh_rows,
        n_emp_rows=artifacts["empirical_sample_complexity"],
        benchmark_metadata=artifacts["benchmark_metadata"],
        max_size_multiplier=max_size_multiplier,
    )
    by_name = {framing.name: framing for framing in framings}
    if k1a_framing not in by_name:
        raise KeyError(f"Unknown K1a framing {k1a_framing!r}; known: {sorted(by_name)}")
    medians = by_name[k1a_framing].median_by_algorithm
    if not medians:
        raise ValueError(f"K1a framing {k1a_framing!r} produced no ratios; cannot reach the gate.")

    # One gate call per algorithm is what the 2026-08-24 record does; the
    # worst (largest) median is the one that decides a single combined call,
    # since a larger ratio means later identification.
    k1a_worst = max(medians.values())

    k5 = exact_curve_has_interior_minimum(artifacts["exact_k_curve"], k_max=k_max)

    criteria = dict(c2_criteria or {})
    inputs = DecisionGateInputs(
        tests_passing=tests_passing,
        no_leakage_violations=no_leakage_violations,
        reproducible=reproducible,
        oeh_observed_fraction=observed / len(selected),
        k1a_median_ratio=k1a_worst,
        k3_within_factor3_rate=None,
        censoring_rate=censored / len(selected),
        coverage_grid_expansion_feasible=coverage_grid_expansion_feasible,
        baselines_present=baselines_present,
        projected_ci_width_exceeds_mde=None,
        k5_interior_minimum_found=k5,
        beats_strongest_baseline_ci_excludes_zero=criteria.get(
            "beats_strongest_baseline_ci_excludes_zero"
        ),
        holds_on_non_q_star_families=criteria.get("holds_on_non_q_star_families"),
        abstention_rate_on_solvable=None,
    )

    derivation = {
        "selected_pilots": len(selected),
        "oeh_status_counts": {"observed": observed, "censored": censored, "abstained": abstained},
        "k1a_framing_fed_to_gate": k1a_framing,
        "k1a_median_by_algorithm": medians,
        "k1a_framings": [framing.as_dict() for framing in framings],
        "k5_source": "exact_k_curve.csv: interior k_dagger_exact with a non-constant multiplier",
        "k3_note": (
            "K3 (blind extrapolation) is not computed here: it needs predictions made at N_k "
            "and scored against N_emp afterwards, which no committed artifact carries. Left "
            "None so the gate reads it as unmeasured rather than failed."
        ),
        "abstention_rate_on_solvable_note": (
            "Undefined while solvability is itself right-censored for most rows "
            "(WEEK2_DECISION_GATE.md sec 7). Left None rather than reported as the "
            "unrestricted rate, which criterion 6 is not defined on."
        ),
        "c2_criteria_source": criteria.get("_source", "not supplied"),
    }
    return inputs, derivation


__all__ = [
    "K1aFraming",
    "derive_gate_inputs",
    "exact_curve_has_interior_minimum",
    "k1a_framings",
    "selected_budgets",
    "stabilization_budgets",
    "states_actions_by_mdp",
]
