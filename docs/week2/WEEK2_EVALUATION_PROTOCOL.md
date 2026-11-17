# Week 2 Evaluation Protocol

Status: **candidate** — mirrors `protocol_config.json:protocol_status`. This
document does not itself freeze anything; it formalizes, for Week 2 engineering
purposes, definitions that are already candidate/frozen elsewhere. It becomes
load-bearing only when `protocol_config.json:protocol_status == "frozen"`.

Owner: Workstream 4 (Statistical Evaluation & Research Protocol).

## 0. Real status as of this writing (2026-08-18)

`docs/week1/WEEK1_AUDIT.md` records Week 1 as **NOT COMPLETE**. `protocol_config.json`
carries `protocol_status: "candidate"` with four `pending_validations`. The
`PLAN_Final.md` freeze date is **21/08/2026** — still in the future. The `main`
branch merge titled "Week 1 verified" (`ec7ef17`) added no commits beyond the
tip of `linh` (`255df63`, which itself is the commit that both wrote the
NOT-COMPLETE audit and added the CFQI/IQL/`N_emp` learner infrastructure that
one of the audit's own blockers says is absent). This is a **recorded
inconsistency**, not something this document resolves: the audit text and the
code in the same commit disagree, and no rerun confirming the fix exists yet.

**Consequence for this document and for `evaluation/`:** nothing here
authorizes running the official 120-MDP benchmark (`WEEK2_BACKLOG.md` already
forbids that until Week 1 has no blocker). What this document and the
`evaluation/` package provide is the *contract and infrastructure* so that once
Week 1 is actually frozen, the other three workstreams can produce artifacts
against a single definition instead of three private ones. Every constant
below is quoted from an already-committed source, never invented here.

## 1. Source of truth, in priority order

1. `protocol_config.json` (machine-readable constants) + `config/protocol_config.py`
   (validator).
2. `PLAN_Final.md` §3 (definitions), §5–§8 (benchmark, calibration, evaluation
   protocol, success criteria) and `paper/sections/03_definitions.tex` (the
   same definitions in the paper's formal notation — the two are required to
   agree; if they ever diverge, that is a bug to report, not to silently pick
   one side of).
3. The reference implementations: `code/oeh_estimator.py` (`OEHResult`),
   `code/adaptive_pilot.py` (`AdaptivePilotSelection`, `PilotAttempt`),
   `code/empirical_requirement.py` (`EmpiricalRequirement`),
   `code/exact_evidence_curve.py` (`ExactEvidenceCurve`), `code/exact_solver.py`.
4. This document and `WEEK2_ARTIFACT_CONTRACT.md` / `WEEK2_STATISTICAL_PLAN.md`,
   which operationalize (1)–(3) for Week 2 but define nothing new.

Nothing in `evaluation/` is allowed to redefine a formula that already exists
in (2)/(3). Where `evaluation/` computes a quantity, it computes it exactly as
the reference implementation does, or it imports the reference implementation
directly.

## 2. Dataset split

| Split | Count | Use | May feed OEH estimation? |
|---|---:|---|---|
| Evaluation | 120 | primary metrics, superiority tests (K1–K5, §8) | pilot buffer only |
| Development | 15 | pipeline probes, hyperparameters, censoring calibration, GO/NO-GO gate | pilot buffer only |
| OOD (Combination Lock, Grid Maze, RiverSwim) | 3 | out-of-distribution illustration only | pilot buffer only |

No metric in the main results table is computed on the development split
(`PLAN_Final.md` §5.2). Variants of the same base MDP are never counted as
independent observations.

Within one MDP instance, there are exactly two independent data streams,
namespaced by disjoint seed offsets (`collection.pilot_seed_offset` = `10000000`,
`collection.training_seed_offset` = `20000000` in `protocol_config.json`):

- **Pilot stream** (`D0`): what the OEH estimator sees. Nested prefixes at
  `{2, 5, 10} × |S||A|` (`collection.pilot_size_multipliers`).
- **Training stream**: what `N_emp` is measured against, `R=10` independent
  seeds per grid size `{1, 2, 5, 10, 20, 50} × |S||A|`
  (`collection.training_seed_indices`, `collection.size_multipliers`).

**Exact-oracle information (`P`, `R`, `V*`, `Q*`, exact normalized return, exact
`k†`, family/coverage labels) must never reach the OEH estimator, the
calibration model's feature builder, or any structural-only baseline as an
input feature.** It is legitimate only as: (a) the label being predicted
(`N_emp`), (b) a scoring reference (`k_dagger_exact`, K3/K4/K5), or (c)
benchmark construction. `evaluation/validation/leakage.py` enforces this
mechanically — see §7 below and `tests/test_evaluation_leakage.py`.

## 3. `N_emp` — frozen definition, restated for engineering

Source: `PLAN_Final.md` §3.1, `paper/sections/03_definitions.tex` §3.1,
`code/empirical_requirement.py::empirical_requirement`.

For a trained policy `π` on training stream size `N`:

```
NR(π) = [J(π) - J(π_rand)] / [J* - J(π_rand)]
```

MDPs with `J* - J(π_rand) < 0.01/(1-γ)` (post reward-rescale to `[0,1]`) are
excluded **before dataset generation**; the exclusion count is a required
manifest field, never silently dropped.

Grid `𝒩 = {1, 2, 5, 10, 20, 50}·|S||A|`, `R = 10` independent training seeds
per size, **independent of the pilot stream** (disjoint seed namespace, see
§2). A run succeeds when `NR(π) ≥ 0.95` (`learning.success_normalized_return`).

```
p̂_j = (1/R) Σ_r 1[NR(π_j,r) ≥ 0.95]
N_emp = min{ N_j : p̂_ℓ ≥ 0.8 for every N_ℓ ≥ N_j }     (learning.success_runs_required = 8 of 10)
```

If no grid level qualifies, the observation is **right-censored** at `N_max =
50·|S||A|`. It is entered into every downstream analysis as a censored
observation (`code/empirical_requirement.py::EmpiricalRequirement.right_censored
= True`, `requirement = None`). **It is never imputed as `N_max` and never
dropped.** Every reported `N_emp` must carry a Wilson interval for `p̂_j`
(`PLAN_Final.md` §3.1) — Workstream 2's artifact must include the raw
per-size success counts (`success_counts`, `success_rates` from
`EmpiricalRequirement`), not just the resolved `N_emp`, so that Workstream 4
can compute the interval rather than re-deriving counts after the fact.

This is a **thresholded requirement on a fixed grid**, not a population sample
complexity — do not describe it as the latter in any Week 2 output.

`N_emp` is per-(MDP, behavior policy, algorithm). CFQI and IQL each get their
own `N_emp`; they are never pooled.

## 4. Metric contract

Full formal spec (name, formula, input, output, valid range, aggregation,
failure handling) lives in `evaluation/metrics/contract.py::METRIC_CONTRACT`
— that module is the single executable source of truth. This section is a
human-readable index; a test
(`tests/test_evaluation_metrics.py::test_protocol_doc_lists_every_registered_metric`)
fails the build if this list and the code registry diverge.

| Metric | Computed by | Oracle? |
|---|---|---|
| `return` | exact policy evaluation (`code/exact_solver.py::evaluate_policy`) | yes — scoring only |
| `normalized_return` | `PLAN_Final.md` §3.1 formula | yes — scoring/label only |
| `optimality_gap` | `1 - normalized_return` | yes — scoring only |
| `success` | `1[normalized_return ≥ 0.95]` | yes — scoring/label only |
| `coverage` | `scripts/data/generate_offline_datasets.py::coverage_metrics` fields (`observed_reachable_sa_fraction` primary) | no — data-only |
| `behavior_value` | exact evaluation of the *unrestricted* behavior policy `β` (not `β̂^K`) | yes — scoring/baseline only |
| `visit_entropy` | `normalized_visitation_entropy` (already computed data-only in `scripts/data/generate_offline_datasets.py`) | no — data-only |
| `N_emp` | §3 above | label (measured, not oracle-internal) |
| `N_hat_req` | `OEHResult.n_req_hat` = `N0 · min_k m_k(D0)` | no — data-only |
| `k_dagger_exact` | `ExactEvidenceCurve.k_dagger` | yes — scoring reference only |
| `k_dagger_oeh` | `OEHResult.k_dagger` | no — data-only |
| `censoring` | §5 below | status, not a numeric metric |
| `abstention` | §5 below | status, not a numeric metric |

No metric here has two implementations. Where a metric is "data-only" it must
never be computed from a function that also has access to `P/R/V*/Q*` in the
same call; `evaluation/validation/leakage.py` checks this by signature
inspection for the registered estimator entry points.

## 5. Observed / censored / abstained / invalid — kept separate

Per `evaluation/censoring/status.py`. There are **two independent axes** that
must not be merged into one status:

**Axis A — pilot-selection / OEH outcome** (`code/adaptive_pilot.py` status
strings, `OEHResult` fields):

| Status | Meaning | Source condition |
|---|---|---|
| `observed` | usable `k†_oeh`, `H_OEH`, `N̂_req` | `selection.status == "selected"` and not `censored_right` |
| `censored` | budget/coverage insufficient at the selected depth, **not** a refusal | `selection.status == "censored_at_cap"`, or accepted result has `censored_right=True` (i.e. `w_gap(k†) > ε`, PLAN §3.2 D2) |
| `abstained` | estimator actively declines: insufficient trajectories, singular model, empty support at a critical state, or `< 90%` numerically valid bootstrap replicates | `selection.status in {"abstained_at_cap", "insufficient_bootstrap_at_cap"}`, or `OEHResult.abstained=True` with `OEHResult.reason` set |
| `invalid` | pipeline/schema failure — not a scientific outcome | malformed artifact, exception, schema-validation failure |

`insufficient_bootstrap_at_cap` is classified as `abstained` (not a fourth
category and not `censored`): the bootstrap-validity gate is an
evidence-sufficiency test, matching the estimator's own use of "abstain" for
that condition (`PLAN_Final.md` §3.3).

**Axis B — `N_emp` label outcome** (`code/empirical_requirement.py`):

| Status | Meaning |
|---|---|
| `observed` | `right_censored=False`, `requirement` is an integer grid size |
| `censored` | `right_censored=True`, `requirement=None` |
| `invalid` | fewer than the required number of training runs at some grid size, or a malformed artifact |

Axis B has no `abstained` state — `N_emp` measurement never "declines to
answer"; it either resolves on the grid or right-censors at the cap.

Every per-run record in `per_run_results.csv` (see
`WEEK2_ARTIFACT_CONTRACT.md`) carries **both** axes explicitly
(`oeh_status`, `n_emp_status`) — they are frequently different for the same
(MDP, policy) pair and reporting one in place of the other is a leakage/
mislabeling bug.

## 6. Baselines

All six run through the same calibration/aggregation pipeline as OEH
(`PLAN_Final.md` §6.1); none gets a privileged evaluation path.

1. **Behavior policy quality** — `behavior_return`, `behavior_normalized_return`
   (`NR(β)`), `behavior_optimality_gap`. Exact evaluation of `β`, never `β̂^K`.
2. **Coverage** — the registered subset only: `observed_reachable_sa_fraction`
   (primary) plus the secondary fields already computed in
   `scripts/data/generate_offline_datasets.py::coverage_metrics`. No new coverage metric is
   introduced; nothing beyond what §4 lists as "not extended per protocol."
3. **Behavior value** — `V(β̂)` = value of the exact behavior policy under the
   true `P, r` (oracle evaluation, not `V^{β̂^K}`).
4. **Visit entropy** — `normalized_visitation_entropy`, data-only, using only
   prediction-time information (visitation counts already in the pilot).
5. **Structural-only** — features limited to `1/(1-γ)`, `|S|`, `|A|`, MDP
   diameter, mixing time, and any other structural constant already frozen in
   `protocol_config.json:benchmark.environment_recipes`. Forbidden inputs:
   pilot outcome, `N_emp`, `k†_exact`, any OEH output, test-set performance.
   Enforced by `evaluation/baselines/structural_only.py` accepting only a
   fixed, closed feature namelist.
6. **Random negative control** — features drawn from an RNG seeded
   independently of both the pilot and training streams; exists to catch
   spurious correlation in the statistical pipeline itself, and its result is
   reported but never cited as evidence of OEH's contribution
   (`PLAN_Final.md` §6.1, last line).

## 7. Leakage boundary (mechanically enforced)

`evaluation/validation/leakage.py::assert_no_oracle_leakage` inspects a
feature/column set against a closed allowlist per consumer:

- OEH estimator input: transitions + `S`, `A`, `rho0` only (already true of
  `code/oeh_estimator.py`; `tests/test_evaluation_leakage.py` re-verifies it
  from the evaluation-layer side, independent of the existing
  `test_estimator_does_not_import_exact_solver`).
- Structural-only baseline input: the closed structural namelist in §6.4 —
  rejects any column whose name matches `exact_*`, `oracle_*`, `k_dagger_exact`,
  `n_emp*`, `oeh_*`.
- Calibration feature builder (`evaluation/calibration/`): pilot-derived and
  structural features only; rejects the same forbidden-name patterns.

This is a **necessary, not sufficient** check — it catches a column literally
named `exact_return` ending up in a feature matrix, not a semantically
disguised leak. Its purpose is to fail loudly on the common mistake, not to
prove the absence of leakage in general.

## 8. Decision-gate criteria (pointer, not a redefinition)

K1a–K5 and the six pre-registered success criteria are defined in
`PLAN_Final.md` §8 and `paper/sections/03_definitions.tex` §"Pre-Registered
Claims." `evaluation/decision_gate/gate.py` encodes exactly those thresholds
(`0.25` for K1a, factor-of-2 for K2, `±2` on `≥80%` for K4, factor-of-3 on
`≥60%` for K3, `≤50%` abstention rate for criterion 6) by reading them from
`protocol_config.json` where they exist there, and from a small, explicitly
`[DECISION REQUIRED]`-flagged constants table where they do not yet exist in
machine-readable form (they currently do not — `protocol_config.json` has no
`success_criteria` or `k1a_threshold` section). See
`WEEK2_STATISTICAL_PLAN.md` §5 for the exact gate logic and
`artifacts/week2/WEEK2_DECISION_GATE.md` for the current (not-a-real-run)
report.

**[DECISION REQUIRED]**: `protocol_config.json` should gain a
`success_criteria` section mirroring `PLAN_Final.md` §8.2 the same way
`estimator`/`learning`/`coverage` already mirror §3, so the K1–K5 thresholds
stop living only in prose. This is flagged, not silently added, because
`config/protocol_config.py::validate_protocol_config` is itself part of the frozen
Week 1 surface and changing its required-keys set is a protocol change subject
to the same freeze rule (`PLAN_Final.md` §11 rule 1).
