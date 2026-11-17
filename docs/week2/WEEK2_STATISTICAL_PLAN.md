# Week 2 Statistical Plan

Status: candidate, same caveat as the other two `docs/week2/` files. Defines
*how* `evaluation/aggregation/`, `evaluation/censoring/`, `evaluation/calibration/`
and `evaluation/decision_gate/` combine artifacts built to
`WEEK2_ARTIFACT_CONTRACT.md`. Every procedure here is either already specified
in `PLAN_Final.md` §6–§8 (in which case this document only makes it
executable) or explicitly marked `[DECISION REQUIRED]`.

## 1. Cross-validation and confidence intervals

Per `PLAN_Final.md` §7: 5-fold cross-validation, **split by base MDP**, holding
every behavior policy, every pilot size, and every training size of one
instance in the same fold. `evaluation/aggregation/folds.py::assign_folds`
implements this — a deterministic hash of `mdp_id` mod 5, never a random split
that could vary run-to-run.

Confidence intervals: **cluster bootstrap at the base-MDP level** (resample
MDPs with replacement, not rows). Report the interval of the **difference**
between the candidate model and the strongest baseline, Holm-corrected across
the ablation family (§4). `evaluation/aggregation/aggregate.py::cluster_bootstrap_ci`
takes a per-MDP-aggregated statistic and returns `(point_estimate, ci_low,
ci_high, num_clusters)`.

Primary metric: discrete-time likelihood / integrated Brier score (§3).
Secondary: rank correlation and ratio error, **on observed events only**,
always reported with the finite case count (`PLAN_Final.md` §7). Abstention
rate is reported separately for solvable vs. unsolvable cases.

## 2. Aggregation — what must never happen

`evaluation/aggregation/aggregate.py::aggregate(per_run_df) -> aggregated_df`
groups by `(mdp_id, policy_family, algorithm)` (or a coarser key requested by
the caller — family, algorithm, dataset-size bucket) and computes:

- `mean`, `median` (median only where the protocol calls for it — success
  rate, `NR`; do not add median to every metric reflexively),
- `std`,
- `ci_low`, `ci_high` (cluster bootstrap, §1),
- `n_valid`, `n_censored`, `n_abstained`, `n_invalid` — **all four counted
  explicitly, always summing to the group size**.

A group whose `n_valid == 0` still gets a row: `mean`/`ci_*` are `NaN`, not
omitted, and the row is not dropped from the output. This is the direct
implementation of mega-task §10's "never silently drop invalid/censored runs."
`tests/test_evaluation_aggregation.py::test_all_censored_group_still_produces_a_row`
checks this.

Censored `N_emp` values are **never** turned into `N_max` inside aggregation.
`aggregate()` keeps the raw `(n_emp_transitions, n_emp_censored)` pair through
to the output; any downstream analysis that needs a point value from a censored
observation must go through `evaluation/calibration/` (§4), which fits a
survival model precisely so that censored observations contribute correctly
weighted likelihood instead of a substituted number.

## 3. `k†` and `N̂_req` comparisons

`evaluation/aggregation/comparisons.py`.

### `compare_k_dagger(oeh_df, exact_df) -> DataFrame`
Per `(mdp_id, policy_family, pilot_multiplier)`, only where `oeh_status in
{observed, censored}` (an `abstained`/`invalid` row has no `k_dagger_oeh` to
compare — it is reported in the abstention-rate statistic, §5 criterion 6, not
silently coerced into a comparison):

```
k_error            = k_dagger_oeh - k_dagger_exact
absolute_k_error   = |k_error|
exact_match        = k_error == 0
within_1           = absolute_k_error <= 1
within_2           = absolute_k_error <= 2      # K4 threshold
```

K4 (`PLAN_Final.md` §8.1) is reported as the fraction with `within_2 == True`
among support-valid instances (`oeh_status == observed`), never pooling
`censored` rows into the numerator or denominator without saying so.
`compare_k_dagger` returns per-row booleans; the summary rate is computed by
`aggregate()`, not baked into the comparison function, so intermediate results
stay auditable.

### `compare_n_required(oeh_df, n_emp_df) -> DataFrame`
Per `(mdp_id, policy_family)`, joined on the **smallest usable pilot
multiplier** (`WEEK2_EVALUATION_PROTOCOL.md` §2's `N0_eff`, i.e. the row
`adaptive_pilot` actually selected — never averaged across multipliers):

```
absolute_error = |n_req_hat - n_emp_transitions|
relative_error = absolute_error / n_emp_transitions      # undefined if n_emp censored
ratio           = n_req_hat / n_emp_transitions           # undefined if n_emp censored
```

If `n_emp_censored == True`, `relative_error` and `ratio` are `NaN` and the row
carries `n_emp_censored=True` through unchanged — **not** computed against
`N_max` (mega-task §11's explicit prohibition, `PLAN_Final.md` §3.1 "never
replaced by N_max"). Correlation between `n_req_hat` and `n_emp_transitions` is
reported only over the uncensored subset, with the censored fraction stated
alongside so the correlation is never presented as if it covered the full
sample.

K1a/K1b (`PLAN_Final.md` §8.1) use the *pilot budget at which the estimate
stabilizes* (`N_k`), not `n_req_hat` itself, divided by `N_emp` — median ratio
`≤ 0.25` for K1a (depth), reported without a hard threshold for K1b
(multiplier). `evaluation/aggregation/comparisons.py::early_identification_ratio`
computes `N_k / N_emp` per instance, with censored `N_emp` entering as a
conservative upper bound (the ratio can only be an over-estimate of how early
identification really was), exactly as `PLAN_Final.md` §8.1 specifies.

## 4. Calibration prototype

`evaluation/calibration/discrete_survival.py`. Goal per mega-task §9: prove the
pipeline `censored observations → calibration → predicted requirement /
uncertainty` runs end-to-end. **Not** to produce a tuned final model.

Model family, per `PLAN_Final.md` §6 Stage 3 (already specified, not invented
here): discrete-time hazard on the size grid,

```
h_j(x) = Pr(N_emp = N_j | N_emp >= N_j, x)
logit h_j(x) = alpha_j + beta^T x
```

with every censored observation contributing to the likelihood via its
survival term (never dropped, never cut at `N_max`). Regularization is
selected **inside the training fold only** (§7 CV).

`evaluation/calibration/interface.py` defines the abstraction:

```python
class SurvivalCalibrator(Protocol):
    def fit(self, features: FeatureFrame, sizes: Sequence[int], observed: Sequence[bool]) -> "SurvivalCalibrator": ...
    def predict_median_requirement(self, features: FeatureFrame) -> np.ndarray: ...
    def predict_hazard(self, features: FeatureFrame) -> np.ndarray: ...
```

`evaluation/calibration/discrete_survival.py::DiscreteTimeHazardCalibrator`
implements it with per-size logistic regression on the hazard (closed-form
Newton steps, no external ML dependency beyond `numpy`/`scipy`, matching
`requirements.txt`). The feature builder
(`evaluation/calibration/features.py::build_features`) draws **only** from the
allowlist in `WEEK2_EVALUATION_PROTOCOL.md` §7 (pilot-derived + structural) —
this is where a leaked `exact_return` column would most plausibly enter, so it
is the most heavily tested module (`tests/test_evaluation_leakage.py`).

No calibration model is described as final. `DiscreteTimeHazardCalibrator` is
one concrete implementation behind `SurvivalCalibrator`; swapping it later
requires no change to `evaluation/aggregation/` or `evaluation/decision_gate/`.

## 5. Decision gate logic

`evaluation/decision_gate/gate.py::decide(inputs: DecisionGateInputs) -> Decision`.

`Decision` is one of `GO`, `CONDITIONAL_GO`, `NO_GO`, each with a `reasons:
list[str]` explaining which of the five checks (mega-task §14 A–E) drove it.
Thresholds are read from `protocol_config.json` where present and from a
clearly `[DECISION REQUIRED]`-flagged constants block
(`evaluation/decision_gate/thresholds.py::PROVISIONAL_THRESHOLDS`) where not —
see `WEEK2_EVALUATION_PROTOCOL.md` §8 for why K1–K5 thresholds are not all in
`protocol_config.json` yet.

Checks, mapped to `PLAN_Final.md` §10's table (this is a restatement, not a
new decision procedure):

- **A. Implementation validity**: full test suite passes, `protocol_status`
  reachable (not stuck on a validation failure), no leakage-check failure.
- **B. OEH validity**: `oeh_results.csv` has `oeh_status == observed` for a
  non-trivial fraction of development-split rows, `k_dagger_oeh` computable,
  `valid_bootstrap_fraction` distribution sane.
- **C. Predictive validity**: K1a ratio and K3 blind-extrapolation rate from
  `evaluation/aggregation/comparisons.py`.
- **D. Statistical validity**: censoring handled per §2/§3 (checked
  structurally — e.g. no row where `n_emp_censored=True` and `relative_error`
  is non-null), baseline pipeline produces rows for all six baselines,
  `evaluation/validation/leakage.py` reports zero violations.
- **E. Scientific hypothesis**: K5 (interior minimum of `m_k` on the exact
  curve) on the development split.

`decide()` returns `NO_GO` if **A** fails outright (broken pipeline is not a
scientific question). Otherwise it maps B–E onto `PLAN_Final.md` §10's table
literally — `evaluation/decision_gate/gate.py` docstring quotes that table
row-for-row so the mapping is auditable against the source.

**This function requires real artifacts to produce a real decision.** Run
against no artifacts (or synthetic test fixtures), it returns `NO_GO` with
reason `"no artifacts supplied"` — that is a data-availability statement, not
a scientific finding, and `artifacts/week2/WEEK2_DECISION_GATE.md` says so
explicitly rather than presenting it as a negative result.

## 6. Reproducibility

Every `evaluation/` function that touches randomness takes an explicit `rng:
np.random.Generator` or `seed: int`, never a bare `np.random` call — matching
`code/oeh_estimator.py`'s existing convention. `pilot/aggregate.py` records
`git_commit`, `config_hash` (sha256 of `protocol_config.json`), a combined
sha256 over every required input artifact file
(`pilot/aggregate.py::_combined_input_hash`), `rng_seed`, a UTC `timestamp`,
and `protocol_version` (`protocol_config.json:schema_version`) into
`decision_gate_input.json`'s `provenance` block. Same artifacts + same config
must produce byte-identical `per_run_results.csv`/`aggregated_results.csv` —
`tests/test_pilot_aggregate.py::PipelineEndToEndTests::test_reproducible_given_the_same_inputs_and_seed`
checks this by running the full pipeline twice over one fixture and diffing
the output files.

This reuses the existing manifest provenance pattern
(`scripts/data/generate_offline_datasets.py::_generator_provenance`) rather than building a
second reproducibility mechanism, per mega-task §18.
