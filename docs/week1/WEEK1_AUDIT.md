# Week 1 audit — 2026-08-17

Status: **NOT COMPLETE — protocol remains `candidate`**.

## Evidence collected before any code change

- Branch: `linh`; starting commit: `4740dc6fe08b15ad56a177ec7cd7dc643596d8b8`.
- Core suite: **68/68 PASS**.
- Smoke: **14/14 path-test environments × 5 behavior families PASS**.
- Development-only bundle: 15 development instances, five policy families,
  825 physical buffers, and 4,725 nested logical datasets.  Its manifest SHA-256
  is `0a5378a3f037612be0b59c54f77ae53fbf49fcbcf353daba355e3fb8266a1001`.
- The bundle was generated in `review` mode with an explicit development-only
  reason; no evaluation instance was materialized.

## Development validation result

The registered data-only selector was run over all 75 development
instance-policy pilots using the registered `B=100`, `K_max=100`, adaptive
prefix grid `{2,5,10}|S||A|`, and deterministic bootstrap seed `20260817`.

| Outcome | Count |
|---|---:|
| Valid selected estimate | 24 |
| Right-censored at cap | 38 |
| Abstained at cap | 13 |

Every attempted bootstrap had valid fraction 1.0. Observed reachable
state-action coverage ranged from 0.10 to 1.00 (median 0.75390625). These are
measurements after natural trajectory generation; no transition was inserted,
filtered, or resampled to change coverage.

Policy-quality normalized-return medians were: `q_star_softmax` 0.9572,
`perturbed_reward` 0.4265, `structured_random` -0.0469,
`two_policy_mixture` -0.0002, and `shifted_dynamics` 0.7399. This confirms
material quality diversity; only `q_star_softmax` is configured to use Q-star.

## Findings and disposition

1. **Fixed: oracle/reference support parity.** `exact_evidence_curve` rejected
   an empty supported-action set at a non-critical non-terminal state, whereas
   OEH correctly permits that state unless it enters `S_crit`. The oracle now
   gives such a state zero restricted-behavior continuation, matching the
   data-only empirical model. A regression test covers this condition.
2. **BLOCKER: `N_emp` infrastructure is absent.** The repository contains the
   definition and an exact normalized-return utility, but no CFQI/IQL (or other
   registered learner) and no implementation of the 10-seed, nested-size
   `N_emp` measurement. Consequently D4 and the required exact `N_emp` output
   cannot be validated.
3. **BLOCKER: Gate C needs a clean post-fix development rerun.** The first
   complete run exposed the oracle parity issue. It must be rerun using the
   unchanged development bundle and seed after the fix, then its evidence-curve
   and `k†` diagnostics must be reviewed. No threshold or estimator parameter
   may be changed based on that rerun.

## Integrity checks retained

- `test_estimator_does_not_import_exact_solver` passes.
- Terminal, unsupported-action, adaptive-pilot, Bayesian trajectory bootstrap,
  reproducibility, and protocol-validation tests pass.
- The estimator receives only transition arrays, dimensions, and the initial
  distribution. Manifest oracle fields and exact solver calls are confined to
  audit/generation paths.

## Freeze decision

Do **not** set `protocol_status` to `frozen`, clear pending validations, tag a
freeze, or start the official evaluation benchmark. Week 1 can be frozen only
after both blockers above are resolved and the full suite/smoke/development
validation are rerun successfully.

## Rerun — 2026-08-18

Both scripts referenced by the two blockers above already existed in the
repository at commit `255df63` (`scripts/training/run_empirical_requirement.py`,
`code/empirical_requirement.py`) but had never actually been executed; no
result from either was on record anywhere. This entry records what happened
when they were actually run, against the unchanged
`datasets/week1_development_validation` bundle (manifest SHA-256
`0a5378a3f037612be0b59c54f77ae53fbf49fcbcf353daba355e3fb8266a1001`, verified
to still match before running). No threshold, tolerance, or definition was
changed based on either result.

### Blocker 3 (Gate C rerun) — resolved

`python -m scripts.validation.run_week1_validation --dataset-dir datasets/week1_development_validation --output <path> --oracle-seed 20260817`, real runtime 23m42s.

Outcome counts across all 75 development instance-policy pilots (all five
behavior families, not only `q_star_softmax`):

| Outcome | Count |
|---|---:|
| Valid selected estimate | 24 |
| Right-censored at cap | 38 |
| Abstained at cap | 13 |

**Bit-for-bit identical** to the counts already recorded above in this file,
and the `q_star_softmax`-only breakdown (11 censored, 2 abstained, 2 valid)
matches `docs/protocol/PROTOCOL_DECISIONS_2026-08-16.md` D6 exactly. This is an independent
confirmation of reproducibility: same bundle, same seed, same pipeline, same
numbers, run from a fresh process. Per-family breakdown (new — D6 covered
`q_star_softmax` only):

| Family | Censored | Abstained | Valid |
|---|---:|---:|---:|
| `q_star_softmax` | 11 | 2 | 2 |
| `perturbed_reward` | 10 | 1 | 4 |
| `structured_random` | 5 | 5 | 5 |
| `two_policy_mixture` | 2 | 2 | 11 |
| `shifted_dynamics` | 10 | 3 | 2 |

Every attempted bootstrap again had valid fraction 1.0 (min/median/max all
`1.0` across 75 pilots). This blocker is resolved: the post-fix development
rerun is clean, reproducible, and reviewed.

**But it surfaces the open gate D6 already flagged, now confirmed at full
development-split scale instead of a 13-case subsample.** Of the 24 valid
cases, 20 have a defined exact-oracle reference (4 have "fewer than two
support-valid oracle datasets"). Across those 20: **`k_dagger_oeh = 0` in
every case, and `oracle_k_dagger = 0` in every case** —
`absolute_k_dagger_error` is `0.0` for all 20 (min = median = max = 0.0). K4
(`PLAN_Final.md` §8.1, "within ±2 on ≥80% of support-valid instances")
technically passes at 100%, but this is not informative: there is still no
observed depth variance to actually test the depth claim against. This is
`PLAN_Final.md` §13's risk #1 ("`m_k` monotone, no interior minimum") and
`docs/protocol/PROTOCOL_DECISIONS_2026-08-16.md` D6's `exact_k_dagger_range_audit` gate,
**still open** — full raw output, including all 75 per-instance outcomes,
committed at `artifacts/week1/week1_validation_rerun_2026-08-18.json` for
anyone who wants to inspect the 20 defined cases directly.

### Blocker 2 (`N_emp` infrastructure) — resolved

`python -m scripts.training.run_empirical_requirement --dataset-dir datasets/week1_development_validation --output <path>`,
real runtime 2h54m55s. The learners (`cfqi_tabular`, `iql_tabular`,
`code/empirical_requirement.py`) had existed in the repository since commit
`255df63` but had never actually been executed before this run; no result
from either had been on record anywhere. This entry records what happened
when it was actually run, over all 75 development instance-policy training
streams (6 grid sizes × 10 independent seeds each, per `PLAN_Final.md` §3.1),
against the same unchanged bundle used for the Gate C rerun above. No
threshold, tolerance, or definition was changed based on the result. Full raw
output, including all 150 per-(instance, family, algorithm) `N_emp` values
and their success counts/rates, committed at
`artifacts/week1/empirical_requirement_rerun_2026-08-18.json`.

| Algorithm | Right-censored | Finite | `N_emp` min | `N_emp` median (finite) | `N_emp` max |
|---|---:|---:|---:|---:|---:|
| `cfqi_tabular` | 42 / 75 | 33 / 75 | 16 | 1440 | 12800 |
| `iql_tabular` | 61 / 75 | 14 / 75 | 16 | 256 | 1800 |

Per-family finite-case median `N_emp` (transitions), censored count out of 15:

| Family | `cfqi_tabular` median / censored | `iql_tabular` median / censored |
|---|---|---|
| `q_star_softmax` | 380 / 3 | 186 / 7 |
| `perturbed_reward` | 3600 / 10 | 960 / 13 |
| `structured_random` | 1440 / 12 | **no finite case** / 15 |
| `two_policy_mixture` | 800 / 12 | **no finite case** / 15 |
| `shifted_dynamics` | 1440 / 5 | 272 / 11 |

This is real, substantive data, not a sanity-check placeholder: `q_star_softmax`
(the reference, highest-quality behavior family) has the lowest median
requirement and lowest censoring rate under both learners, in the expected
direction. `iql_tabular` never resolves within the registered grid
(`{1,2,5,10,20,50}·|S||A|`, `N_max = 50|S||A|`) for `structured_random` or
`two_policy_mixture` on this development split — 100% right-censored for both
— while `cfqi_tabular` resolves a minority of instances in the same two
families. This blocker is resolved in the sense that the pipeline runs
end-to-end and produces real, differentiated, plausible-direction numbers;
it also surfaces a finding worth the team's attention: `iql_tabular`'s high
censoring rate on two of five behavior families needs to be understood before
K1a/K1b (which divide by `N_emp`) can be evaluated meaningfully for those
family/algorithm pairs — a heavily right-censored `N_emp` distribution limits
how informative the ratio can be regardless of how good the OEH-side estimate
is.

**Runtime note for whoever runs this next:** 2h55m on 75×6×10×2 = 9000
learner fits is dominated by `iql_tabular`'s per-state expectile solve
(`code/empirical_requirement.py::_expectile`, a 100-step bisection called
once per state per one of 100 outer iterations, in a pure Python loop). This
is a real cost to budget for when planning the full 120-MDP evaluation grid,
not a bug; it is out of scope for this audit to change (no algorithm code was
touched to produce this rerun).

## Updated freeze decision (2026-08-18)

Still **not frozen**, and this rerun is not authorization to freeze on its
own. Both blockers recorded at the top of this file are now resolved for
real: Gate C reproduced exactly, and `N_emp` measurement now has a real,
recorded, plausible-direction result. What remains open, unaffected by this
rerun:

1. **`exact_k_dagger_range_audit`** (`docs/protocol/PROTOCOL_DECISIONS_2026-08-16.md` D6) —
   `k†=0` in all 20 development cases with a defined oracle reference, both
   for OEH and the exact curve (Gate C rerun above). No depth variance has
   been observed on the development split in any run to date. This is
   `PLAN_Final.md` §13 risk #1 and remains the single most important open
   question before committing to the full 120-MDP grid.
2. The new finding above: `iql_tabular` is 100% right-censored on two of
   five development behavior families. Not previously known because the
   learner had never been run.
3. The other two `protocol_config.json:pending_validations`
   (`development_policy_quality_probe`, `coverage_threshold_group_review`)
   are untouched by this rerun and remain open.
   `adaptive_pilot_weighted_bootstrap_review` is arguably satisfied by the
   audit trail in `docs/protocol/PROTOCOL_DECISIONS_2026-08-16.md` D5/D6, but flipping that
   specific pending-validation flag is a team sign-off decision, not
   something this rerun does unilaterally.

`protocol_status` stays `"candidate"`. `[DECISION REQUIRED]`: the team should
review items 1-2 above before 21/08 — item 1 in particular is exactly the
condition `PLAN_Final.md` §10's decision table treats as risk #1 ("K5 hỏng —
`m_k` đơn điệu"), and the table's prescribed response if it persists at full
scale is to drop the depth claim and reframe as a negative transferability
result, not to adjust `ε`, the `μ≥1` floor, or the tie-break
(`PLAN_Final.md` §11 rule 2).
