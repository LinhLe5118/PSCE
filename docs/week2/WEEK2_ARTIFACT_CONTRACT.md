# Week 2 Artifact Contract

Status: candidate, same caveat as `WEEK2_EVALUATION_PROTOCOL.md` §0. This is
the exact interface the other three workstreams produce against so
`pilot/aggregate.py` and `evaluation/validation/schema.py` can ingest their
output without a fourth ad hoc format. Column names reuse existing repo
vocabulary (`code/oeh_estimator.py`, `code/adaptive_pilot.py`,
`code/empirical_requirement.py`, `scripts/data/generate_offline_datasets.py`) wherever a
field already exists there — this contract does not rename anything that is
already public.

If a workstream's current artifact uses different names: **do not rename their
code.** Either (a) they change their output to match this contract (preferred,
since it is new and unshipped), or (b) a new `evaluation/validation/adapter.py`
module is added to carry a documented rename map, once a real mismatch exists
to adapt — no such module exists yet, and none is created speculatively
before there is a concrete naming conflict to fix. Silently accepting
mismatched names in `pilot/aggregate.py` is not allowed either way —
`evaluation/validation/schema.py` fails loudly instead (§4).

### Ownership is content-addressed, not name-addressed

The person named after each workstream heading below (§1–§3) is a contact for
questions, not a gate on whether an artifact is valid. **Do not use "who ran
it" to decide whether an artifact is authoritative.** Use its hash chain
instead: `git_commit` (which script version), plus the `artifact_sha256` map
already required in every validation report JSON (e.g.
`oeh_validation_report.json:artifact_sha256`, `benchmark_oracle_validation_report.json`
equivalent). Two files with the same declared schema and the same SHA-256 are
the same artifact regardless of which workstream's driver script emitted them
— no contract update is needed when that changes.

This matters concretely for `exact_k_curve.csv`: the contract assigns it to
Workstream 1 by section heading, but the actual computation
(`exact_k_curve_rows`, called from `code/exact_evidence_curve.py`) currently
only has a driver in Workstream 3's `scripts/validation/run_oeh_validation.py`.
`scripts/validation/run_benchmark_oracle_validation.py` (Workstream 1's driver) does not
independently generate this file — it takes `--exact-k-curve-source`, runs it
through `validate_artifact(..., "exact_k_curve")`, and copies it byte-for-byte
into its own output directory. Confirmed 2026-08-24: the copies under
`artifacts/week2/benchmark_oracle_handoff_2026-08-23/exact_k_curve.csv` and
`artifacts/week2/oeh_validation_2026-08-23/exact_k_curve.csv` are
bit-identical, SHA-256 `a97b2fe0762fb16c0c9f269973882556aea15be060741c57764ef1e06c172d19`.
That hash match — not a person's name — is what makes both copies
authoritative simultaneously. If Workstream 1 later ships its own generator
for this file, the contract does not need editing: whichever run's output
hash is cited in a downstream report's `artifact_sha256` map is the one that
report actually used, and that citation is already mandatory (§4). Section
headings below stay as contacts only; they are not re-assigned when this
happens.

## 1. Workstream 1 — Benchmark & Oracle Validation (Việt Trần)

### `benchmark_metadata.csv`
One row per benchmark MDP instance.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | stable identifier, matches `data_only_group_id` style hashing already used in `scripts/data/generate_offline_datasets.py` |
| `family` | str | one of `uniform_garnet`, `forward_layered`, `lock_structured`, `bottleneck`, or an OOD family name |
| `benchmark_split` | str | `evaluation` \| `development` \| `ood` |
| `num_states` | int | |
| `num_actions` | int | |
| `gamma` | float | must equal `protocol_config.json:benchmark.gamma` |
| `branching_factor` | float | structural feature, allowed as structural-only baseline input |
| `mixing_time_proxy` | float \| empty | structural feature; empty is valid (not every family has one defined) — must not be silently filled |
| `dynamic_range` | float | `J* - J(π_rand)` before the exclusion filter |
| `excluded_dynamic_range` | bool | true if this MDP was excluded and regenerated per §3.1 |
| `git_commit` | str | provenance, already a manifest field pattern |
| `config_hash` | str | sha256 of `protocol_config.json` used |

### `exact_oracle_results.csv`
One row per `(mdp_id, behavior_policy_family)`.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | one of the five families |
| `behavior_return` | float | `J(β)` |
| `behavior_normalized_return` | float | `NR(β)` |
| `behavior_optimality_gap` | float | `1 - NR(β)` |
| `optimal_return` | float | `J*` |
| `random_return` | float | `J(π_rand)` |
| `uses_q_star` | bool | true only for `q_star_softmax` |

### `exact_k_curve.csv`
One row per `(mdp_id, policy_family, depth)`. Source: `ExactEvidenceCurve`.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | |
| `depth` | int | `0..K_max` |
| `delta_hat` | float | may be NaN at a state-averaged level; store the aggregate, not per-state (per-state arrays are large — keep them in a companion `.npz` if needed, referenced by `mdp_id`) |
| `sigma_hat` | float | |
| `multiplier` | float | `m_k` at this depth |
| `k_dagger_exact` | int | repeated per row for join convenience (same value for every depth of one `mdp_id`/`policy_family`) |
| `valid_sample_fraction` | float | `ExactEvidenceCurve.valid_sample_fraction` |

## 2. Workstream 2 — Offline RL & Data Generation (Nguyễn Quốc Việt)

### `rl_run_results.csv`
One row per `(mdp_id, policy_family, algorithm, training_size_multiplier, training_seed)`.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | behavior policy that generated the data |
| `algorithm` | str | `cfqi_tabular` \| `iql_tabular` |
| `training_size_multiplier` | int | one of `{1,2,5,10,20,50}` |
| `training_seed` | int | one of `collection.training_seed_indices` |
| `normalized_return` | float | `NR(π)` for this run |
| `success` | bool | `normalized_return >= 0.95` |
| `run_status` | str | `observed` \| `invalid` (Axis B has no censored/abstained at the run level — censoring is a property of the resolved `N_emp`, not of one run) |

### `behavior_policy_metadata.csv`
One row per `(mdp_id, policy_family)`. Coverage fields must be exactly the
ones already computed in `scripts/data/generate_offline_datasets.py::coverage_metrics` —
do not add or drop fields.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | |
| `policy_parameters` | str (JSON) | |
| `observed_reachable_sa_fraction` | float | primary coverage metric |
| `observed_sa_fraction_all` | float | |
| `observed_policy_support_fraction` | float | |
| `fraction_policy_support_at_least_n_min` | float | |
| `minimum_policy_support_count` | int | |
| `normalized_visitation_entropy` | float | this **is** the `visit_entropy` metric — do not recompute it separately |
| `num_transitions` | int | |
| `num_trajectories_started` | int | |

### `empirical_sample_complexity.csv`
One row per `(mdp_id, policy_family, algorithm)`. Fields mirror
`EmpiricalRequirement` exactly.

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | |
| `algorithm` | str | |
| `n_emp_multiplier` | int \| empty | grid multiplier (`requirement`'s corresponding index into `{1,2,5,10,20,50}`); empty iff censored |
| `n_emp_transitions` | int \| empty | `n_emp_multiplier * S * A`; empty iff censored |
| `n_emp_censored` | bool | `EmpiricalRequirement.right_censored` — **required column, no default** |
| `success_counts` | str (JSON list[int]) | one entry per grid size, in grid order |
| `success_rates` | str (JSON list[float]) | one entry per grid size, in grid order |

A missing `n_emp_censored` column is a schema error, not an implicit `False`
(`WEEK2_EVALUATION_PROTOCOL.md` §3, and the mega-task's explicit prohibition on
silently filling `censoring=False`).

## 3. Workstream 3 — OEH Core & Scientific Validation (Đỗ Trọng Hưng)

### `oeh_results.csv`
One row per `(mdp_id, policy_family, pilot_multiplier)`. Fields mirror
`OEHResult` (scalar fields only — arrays go in `evidence_curves.csv`).

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | |
| `pilot_multiplier` | int | which of `{2,5,10}` this row used |
| `selected` | bool | true on exactly one row per `(mdp_id, policy_family)` — the `AdaptivePilotSelection` outcome (`N0_eff`): either the first accepted multiplier, or the cap row if none was accepted. `evaluation/validation/schema.py` rejects zero or more than one `selected=True` row per key |
| `oeh_status` | str | `observed` \| `censored` \| `abstained` \| `invalid` (Axis A, `WEEK2_EVALUATION_PROTOCOL.md` §5) |
| `abstain_reason` | str \| empty | `OEHResult.reason`; required non-empty when `oeh_status=abstained` |
| `k_dagger_oeh` | int \| empty | empty iff not `observed`/`censored` (a censored result still has a `k†` value — only `abstained`/`invalid` leave it empty) |
| `effective_horizon` | float \| empty | `H_OEH`, same emptiness rule as `k_dagger_oeh` |
| `n_req_hat` | float \| empty | same emptiness rule |
| `valid_bootstrap_fraction` | float \| empty | |
| `gap_occupancy_at_k_dagger` | float \| empty | `w_gap(k†)`; required when `oeh_status=censored` |

### `evidence_curves.csv`
One row per `(mdp_id, policy_family, pilot_multiplier, depth)`.

| Column | Type |
|---|---|
| `mdp_id` | str |
| `policy_family` | str |
| `pilot_multiplier` | int |
| `depth` | int |
| `multiplier` | float |
| `loss_at_one` | float |
| `gap_occupancy` | float |

### `bootstrap_results.csv`
One row per `(mdp_id, policy_family, pilot_multiplier)` — summary, not one row
per replicate (100 replicates × full grid is not artifact-scale data).

| Column | Type | Notes |
|---|---|---|
| `mdp_id` | str | |
| `policy_family` | str | |
| `pilot_multiplier` | int | |
| `num_bootstrap` | int | must equal `estimator.num_bootstrap` (100) |
| `valid_bootstrap_fraction` | float | |
| `bootstrap_seed` | int | |

### `k_dagger_estimates.csv`
One row per `(mdp_id, policy_family)` — the `selected=True` row only — subset of
`oeh_results.csv` columns kept separate because §11–12 of the mega-task
requires `k†` comparison to be tracked as its own artifact, joinable without
carrying the full OEH result set.

| Column | Type |
|---|---|
| `mdp_id` | str |
| `policy_family` | str |
| `pilot_multiplier` | int |
| `k_dagger_oeh` | int \| empty |
| `oeh_status` | str |

### `n_required_estimates.csv`
One row per `(mdp_id, policy_family)` — the `selected=True` row only.

| Column | Type |
|---|---|
| `mdp_id` | str |
| `policy_family` | str |
| `pilot_multiplier` | int |
| `n_req_hat` | float \| empty |
| `oeh_status` | str |

### `abstention_results.csv`
One row per `(mdp_id, policy_family, pilot_multiplier)` where `oeh_status in
{abstained, censored}` — the explicit non-answer log required by §7 of the
mega-task, kept separate so abstention-rate reporting (success criterion 6,
`PLAN_Final.md` §8.2) never has to filter a mixed table.

| Column | Type |
|---|---|
| `mdp_id` | str |
| `policy_family` | str |
| `pilot_multiplier` | int |
| `oeh_status` | str (`abstained` or `censored` only) |
| `reason` | str (abstain reason, or `gap_occupancy_exceeds_epsilon` for censored) |

## 4. Schema validation behavior

`evaluation/validation/schema.py::validate_artifact(path, contract_name)`:

- Missing required column → `SchemaError: "<file>: missing required column '<name>'"`.
- Wrong dtype (e.g. `mdp_id` not parseable as the declared type, `n_emp_censored`
  not a bool) → `SchemaError` naming column, expected type, and offending value.
- Duplicate `(mdp_id, policy_family, ...)` key within one file → `SchemaError`
  naming the duplicated key tuple.
- `oeh_status`/`n_emp_censored`/any enum-typed column outside its allowed set →
  `SchemaError`.
- A censored/abstained row missing its required companion field (e.g.
  `oeh_status=censored` with empty `gap_occupancy_at_k_dagger`) → `SchemaError`.

**Never**: auto-fill a missing column with a default, coerce a bad dtype
silently, drop a duplicate row without raising, or convert an invalid status to
a nearby valid one. The validator's job is to fail loudly before
`pilot/aggregate.py` touches the data (mega-task §17).

## 5. Workstream 4 outputs (this workstream, via `pilot/aggregate.py`)

Written to `artifacts/week2/`:

| File | Producer |
|---|---|
| `per_run_results.csv` | `evaluation/aggregation/aggregate.py::build_per_run_results` |
| `aggregated_results.csv` | `evaluation/aggregation/aggregate.py::aggregate` |
| `baseline_results.csv` | `evaluation/baselines/*` |
| `censoring_analysis.csv` | `evaluation/censoring/status.py::censoring_report` |
| `k_dagger_comparison.csv` | `evaluation/aggregation/comparisons.py::compare_k_dagger` |
| `n_req_comparison.csv` | `evaluation/aggregation/comparisons.py::compare_n_required` |
| `decision_gate_input.json` | `evaluation/decision_gate/gate.py::build_decision_gate_input` |
| `WEEK2_DECISION_GATE.md` | hand-authored from `decision_gate_input.json`, never auto-generated with prose claims the numbers don't support |

None of these are committed with fabricated content. As of this writing,
`artifacts/week2/` contains only this contract's schema and a README pointing
back here — see `artifacts/week2/README.md`.
