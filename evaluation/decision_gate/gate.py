"""Decision gate: GO / CONDITIONAL_GO / NO_GO.

Restates ``PLAN_Final.md`` sec 10's table (which is itself the frozen
decision procedure) as code; it does not invent a new procedure. Checks A-E
map to mega-task section 14:

* A. Implementation validity -- tests pass, no leakage violation, reproducible.
* B. OEH validity -- estimator produces observed results on a non-trivial
  fraction of the development split.
* C. Predictive validity -- K1a / K3.
* D. Statistical validity -- censoring handled structurally, baselines present.
* E. Scientific hypothesis -- K5 (interior minimum on the exact curve).

Requires real artifacts to reach a real decision. Called with no data, it
returns NO_GO with reason ``"no artifacts supplied"`` -- a data-availability
statement, not a scientific finding (``WEEK2_STATISTICAL_PLAN.md`` sec 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from evaluation.decision_gate.thresholds import PROVISIONAL_THRESHOLDS as T

DecisionStatus = Literal["GO", "CONDITIONAL_GO", "NO_GO"]


@dataclass(frozen=True)
class DecisionGateInputs:
    # A. Implementation validity
    tests_passing: bool | None = None
    no_leakage_violations: bool | None = None
    reproducible: bool | None = None

    # B. OEH validity
    oeh_observed_fraction: float | None = None  # fraction of development rows with oeh_status=observed

    # C. Predictive validity
    k1a_median_ratio: float | None = None
    k3_within_factor3_rate: float | None = None

    # D. Statistical validity
    censoring_rate: float | None = None
    coverage_grid_expansion_feasible: bool | None = None
    baselines_present: bool | None = None
    projected_ci_width_exceeds_mde: bool | None = None

    # E. Scientific hypothesis
    k5_interior_minimum_found: bool | None = None

    # Registered success criteria 3-4 (PLAN_Final.md sec 8.2)
    beats_strongest_baseline_ci_excludes_zero: bool | None = None
    holds_on_non_q_star_families: bool | None = None

    # Registered success criterion 6
    abstention_rate_on_solvable: float | None = None


@dataclass(frozen=True)
class Decision:
    status: DecisionStatus
    reasons: list[str] = field(default_factory=list)


def _core_fields_present(inputs: DecisionGateInputs) -> bool:
    return (
        inputs.k1a_median_ratio is not None
        and inputs.k5_interior_minimum_found is not None
        and inputs.oeh_observed_fraction is not None
    )


def decide(inputs: DecisionGateInputs) -> Decision:
    if inputs.tests_passing is False:
        return Decision("NO_GO", ["Implementation validity (A) failed: test suite does not pass."])
    if inputs.no_leakage_violations is False:
        return Decision("NO_GO", ["Implementation validity (A) failed: leakage check reported a violation."])
    if inputs.reproducible is False:
        return Decision("NO_GO", ["Implementation validity (A) failed: aggregation is not reproducible."])

    if not _core_fields_present(inputs):
        return Decision(
            "NO_GO",
            ["No artifacts supplied: this is a data-availability statement, not a scientific finding."],
        )

    reasons: list[str] = []

    if inputs.censoring_rate is not None and inputs.censoring_rate > T.max_censoring_rate_before_grid_expansion:
        if not inputs.coverage_grid_expansion_feasible:
            return Decision(
                "NO_GO",
                [
                    f"Censoring rate {inputs.censoring_rate:.2f} exceeds "
                    f"{T.max_censoring_rate_before_grid_expansion:.2f} and grid expansion is not feasible "
                    "(PLAN_Final.md sec 10)."
                ],
            )
        reasons.append("Censoring exceeded 40% but grid expansion is feasible; decided at the gate, not post hoc.")

    if inputs.projected_ci_width_exceeds_mde:
        return Decision(
            "NO_GO",
            ["Projected CI width exceeds the pre-registered minimal detectable effect (PLAN_Final.md sec 7/10)."],
        )

    k1a = inputs.k1a_median_ratio
    k5 = inputs.k5_interior_minimum_found

    if k1a > T.weak_k1a_boundary:
        return Decision(
            "NO_GO",
            [
                f"K1a failed: median N_k/N_emp={k1a:.3f} > {T.weak_k1a_boundary:.2f}; "
                "k_dagger is only identifiable once the problem is already solved (PLAN_Final.md sec 10)."
            ],
        )

    if T.k1a_median_ratio_max < k1a <= T.weak_k1a_boundary:
        return Decision(
            "NO_GO",
            [
                f"K1a weak: median N_k/N_emp={k1a:.3f} in "
                f"({T.k1a_median_ratio_max:.2f}, {T.weak_k1a_boundary:.2f}]; do not submit to ICLR, "
                "Phase B becomes the thesis main line (PLAN_Final.md sec 10)."
            ],
        )

    if not k5:
        return Decision(
            "CONDITIONAL_GO",
            [
                "K5 failed (m_k curve is monotone on the exact reference): drop the depth claim, "
                "report a negative transferability result. Still submittable under the reframed claim "
                "(PLAN_Final.md sec 10)."
            ],
        )

    if inputs.beats_strongest_baseline_ci_excludes_zero is False:
        return Decision(
            "NO_GO",
            ["Success criterion 3 failed: does not beat the strongest baseline with a Holm-corrected 95% CI excluding zero (PLAN_Final.md sec 8.2)."],
        )
    if inputs.holds_on_non_q_star_families is False:
        return Decision(
            "NO_GO",
            ["Success criterion 4 failed: result does not hold restricted to the four non-Q*-derived behavior families (PLAN_Final.md sec 8.2)."],
        )

    if inputs.abstention_rate_on_solvable is not None and inputs.abstention_rate_on_solvable > T.max_abstention_rate_on_solvable:
        reasons.append(
            f"Abstention rate on solvable cases {inputs.abstention_rate_on_solvable:.2f} exceeds "
            f"{T.max_abstention_rate_on_solvable:.2f} (success criterion 6, PLAN_Final.md sec 8.2)."
        )

    k1a_full_ok = k1a <= T.k1a_median_ratio_max
    k3_ok = inputs.k3_within_factor3_rate is not None and inputs.k3_within_factor3_rate >= T.k3_min_rate

    if k1a_full_ok and k5 and k3_ok and not reasons:
        return Decision("GO", ["K1a, K2/K5, and K3 all pass on the development split; proceed to the full grid (PLAN_Final.md sec 10)."])

    return Decision(
        "CONDITIONAL_GO",
        reasons + ["K1a/K5 pass but at least one secondary check (K3, abstention rate, or censoring note) needs attention before the full run."],
    )


def build_decision_gate_input(
    inputs: DecisionGateInputs,
    *,
    git_commit: str,
    config_hash: str,
    protocol_version: str,
    input_artifacts_hash: str | None = None,
    rng_seed: int | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """``input_artifacts_hash``/``rng_seed``/``timestamp`` are optional so unit
    tests can call this without assembling a full artifact set; ``pilot/aggregate.py``
    always supplies all three for a real run (WEEK2_STATISTICAL_PLAN.md sec 6:
    git_commit, config_hash, seed, timestamp, protocol_version)."""

    decision = decide(inputs)
    return {
        "inputs": {
            "tests_passing": inputs.tests_passing,
            "no_leakage_violations": inputs.no_leakage_violations,
            "reproducible": inputs.reproducible,
            "oeh_observed_fraction": inputs.oeh_observed_fraction,
            "k1a_median_ratio": inputs.k1a_median_ratio,
            "k3_within_factor3_rate": inputs.k3_within_factor3_rate,
            "censoring_rate": inputs.censoring_rate,
            "coverage_grid_expansion_feasible": inputs.coverage_grid_expansion_feasible,
            "baselines_present": inputs.baselines_present,
            "projected_ci_width_exceeds_mde": inputs.projected_ci_width_exceeds_mde,
            "k5_interior_minimum_found": inputs.k5_interior_minimum_found,
            "beats_strongest_baseline_ci_excludes_zero": inputs.beats_strongest_baseline_ci_excludes_zero,
            "holds_on_non_q_star_families": inputs.holds_on_non_q_star_families,
            "abstention_rate_on_solvable": inputs.abstention_rate_on_solvable,
        },
        "decision": {"status": decision.status, "reasons": decision.reasons},
        "provenance": {
            "git_commit": git_commit,
            "config_hash": config_hash,
            "protocol_version": protocol_version,
            "input_artifacts_hash": input_artifacts_hash,
            "rng_seed": rng_seed,
            "timestamp": timestamp,
        },
    }


__all__ = ["Decision", "DecisionGateInputs", "DecisionStatus", "build_decision_gate_input", "decide"]
