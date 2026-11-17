"""Executable metric contract.

Single source of truth for what each Week 2 metric means. Nothing here
redefines a formula that already exists in ``PLAN_Final.md`` §3 or in the
reference implementations (``code/oeh_estimator.py``,
``code/exact_evidence_curve.py``, ``code/empirical_requirement.py``,
``scripts/data/generate_offline_datasets.py``); each entry's ``formula`` field quotes the
existing source rather than restating a derivation.

``docs/week2/WEEK2_EVALUATION_PROTOCOL.md`` §4 is a human-readable index of
this registry; ``tests/test_evaluation_metrics.py`` fails if the two diverge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    definition: str
    formula: str
    inputs: tuple[str, ...]
    output_type: str
    valid_range: tuple[float, float] | None
    aggregation: str
    failure_handling: str
    is_oracle: bool
    source: str


METRIC_CONTRACT: dict[str, MetricSpec] = {
    "return": MetricSpec(
        name="return",
        definition="Expected discounted return of a policy under the true MDP.",
        formula="J(pi) = E_tau~pi[ sum_t gamma^t r_t ]",
        inputs=("policy", "mdp"),
        output_type="float",
        valid_range=(0.0, None),
        aggregation="mean, median, std across seeds",
        failure_handling="undefined policy/MDP pair -> status=invalid, never 0.0",
        is_oracle=True,
        source="code/exact_solver.py::evaluate_policy",
    ),
    "normalized_return": MetricSpec(
        name="normalized_return",
        definition="Return rescaled so 0 is the random policy and 1 is optimal.",
        formula="NR(pi) = (J(pi) - J(pi_rand)) / (J* - J(pi_rand))",
        inputs=("J(pi)", "J(pi_rand)", "J*"),
        output_type="float",
        valid_range=None,
        aggregation="median (success threshold), mean+CI (reporting)",
        failure_handling="MDP excluded pre-generation if J*-J(pi_rand) < 0.01/(1-gamma)",
        is_oracle=True,
        source="PLAN_Final.md sec 3.1",
    ),
    "optimality_gap": MetricSpec(
        name="optimality_gap",
        definition="Fraction of the achievable return gap left unclosed.",
        formula="optimality_gap(pi) = 1 - NR(pi)",
        inputs=("normalized_return",),
        output_type="float",
        valid_range=None,
        aggregation="mean, median, CI",
        failure_handling="propagates normalized_return's failure handling",
        is_oracle=True,
        source="derived, single definition, not reimplemented elsewhere",
    ),
    "success": MetricSpec(
        name="success",
        definition="Indicator that a training run reached the success threshold.",
        formula="success(pi) = 1[NR(pi) >= 0.95]",
        inputs=("normalized_return",),
        output_type="bool",
        valid_range=(0.0, 1.0),
        aggregation="rate with Wilson interval",
        failure_handling="invalid run excluded from the rate's denominator and counted in n_invalid",
        is_oracle=True,
        source="protocol_config.json:learning.success_normalized_return",
    ),
    "coverage": MetricSpec(
        name="coverage",
        definition="observed_reachable_sa_fraction, the registered primary coverage metric, plus its registered secondary fields.",
        formula="see scripts/data/generate_offline_datasets.py::coverage_metrics",
        inputs=("pilot transitions", "MDP reachability (construction-time only)"),
        output_type="float",
        valid_range=(0.0, 1.0),
        aggregation="mean, median across instances within a stratum",
        failure_handling="never extended with a new coverage submetric outside the registered set",
        is_oracle=False,
        source="scripts/data/generate_offline_datasets.py::coverage_metrics",
    ),
    "behavior_value": MetricSpec(
        name="behavior_value",
        definition="Exact value of the unrestricted behavior policy beta (not the support-restricted beta^K).",
        formula="V(beta) = rho0 . V^beta, solved exactly",
        inputs=("beta", "mdp"),
        output_type="float",
        valid_range=(0.0, None),
        aggregation="mean, median, CI",
        failure_handling="singular policy evaluation -> status=invalid",
        is_oracle=True,
        source="code/exact_solver.py::evaluate_policy",
    ),
    "visit_entropy": MetricSpec(
        name="visit_entropy",
        definition="Normalized entropy of the observed visitation distribution in the pilot.",
        formula="normalized_visitation_entropy, data-only",
        inputs=("pilot transitions",),
        output_type="float",
        valid_range=(0.0, 1.0),
        aggregation="mean, median across instances",
        failure_handling="empty pilot -> status=invalid, not 0.0",
        is_oracle=False,
        source="scripts/data/generate_offline_datasets.py::coverage_metrics",
    ),
    "N_emp": MetricSpec(
        name="N_emp",
        definition="Smallest grid size at which >=8/10 independent training seeds reach NR>=0.95 at that size and every larger size.",
        formula="N_emp = min{N_j : p_hat_l >= 0.8 for all N_l >= N_j}",
        inputs=("success rates per grid size",),
        output_type="int | censored",
        valid_range=None,
        aggregation="median across (mdp, policy, algorithm); censored observations never imputed",
        failure_handling="no qualifying grid size -> right-censored at N_max, never replaced by N_max",
        is_oracle=False,
        source="code/empirical_requirement.py::empirical_requirement",
    ),
    "N_hat_req": MetricSpec(
        name="N_hat_req",
        definition="Non-parametric data-only prediction of the transitions required.",
        formula="N_hat_req(D0) = N0 * min_k m_k(D0)",
        inputs=("pilot D0",),
        output_type="float | empty (abstained/invalid)",
        valid_range=(0.0, None),
        aggregation="mean, median, CI; compared to N_emp only on uncensored subset",
        failure_handling="abstained/invalid OEH result -> empty, never 0.0 or N_max",
        is_oracle=False,
        source="code/oeh_estimator.py::OEHResult.n_req_hat",
    ),
    "k_dagger_exact": MetricSpec(
        name="k_dagger_exact",
        definition="Argmin of the exact evidence curve under the same support restriction as the estimator.",
        formula="k_dagger = min argmin_k m_k using true P, R",
        inputs=("mdp", "beta", "action_mask"),
        output_type="int",
        valid_range=(0, None),
        aggregation="scoring reference only, never averaged into a predictive feature",
        failure_handling="fewer than 2 support-valid oracle datasets -> RuntimeError, instance excluded and logged",
        is_oracle=True,
        source="code/exact_evidence_curve.py::ExactEvidenceCurve.k_dagger",
    ),
    "k_dagger_oeh": MetricSpec(
        name="k_dagger_oeh",
        definition="Argmin of the data-only evidence curve.",
        formula="k_dagger = min argmin_k m_k using P_hat, r_hat on A_K",
        inputs=("pilot D0",),
        output_type="int | empty (abstained/invalid)",
        valid_range=(0, None),
        aggregation="compared to k_dagger_exact via absolute_k_error, exact_match, within_1, within_2",
        failure_handling="abstained/invalid -> empty, never 0",
        is_oracle=False,
        source="code/oeh_estimator.py::OEHResult.k_dagger",
    ),
    "censoring": MetricSpec(
        name="censoring",
        definition="Status axis, not a numeric metric. Two independent kinds: OEH-pilot censoring (Axis A) and N_emp-label censoring (Axis B). Never merged.",
        formula="see evaluation.censoring.status",
        inputs=("OEHResult / AdaptivePilotSelection", "EmpiricalRequirement"),
        output_type="categorical",
        valid_range=None,
        aggregation="rate per axis, per stratum",
        failure_handling="n/a - this is the failure-handling mechanism itself",
        is_oracle=False,
        source="docs/week2/WEEK2_EVALUATION_PROTOCOL.md sec 5",
    ),
    "abstention": MetricSpec(
        name="abstention",
        definition="Status axis: estimator actively declines due to insufficient evidence/coverage.",
        formula="see evaluation.censoring.status",
        inputs=("OEHResult.abstained", "OEHResult.reason"),
        output_type="categorical",
        valid_range=None,
        aggregation="rate, reported separately for solvable vs unsolvable cases",
        failure_handling="n/a - this is the failure-handling mechanism itself",
        is_oracle=False,
        source="docs/week2/WEEK2_EVALUATION_PROTOCOL.md sec 5",
    ),
}


def metric_names() -> tuple[str, ...]:
    return tuple(METRIC_CONTRACT)


__all__ = ["METRIC_CONTRACT", "MetricSpec", "metric_names"]
