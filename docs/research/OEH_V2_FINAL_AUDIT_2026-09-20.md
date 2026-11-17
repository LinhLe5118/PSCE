# OEH v2: final development audit

Date: 2026-09-20. This document supersedes the v1 cross-fit proxy as the current
development implementation. It does not erase the registered OEH-v1 artifacts:
those remain the before/after evidence for the original bottleneck and
mechanical saturation.

## What was fixed

The old cross-fitted proxy asked the current pilot to prove a positive lower
confidence gap before it would estimate a *future* requirement. It also required
every action at every occupancy-positive state. This made finite forecasting
nearly impossible and caused the controlled factorial to return 0 identified
cells.

OEH v2 separates four questions:

1. The model fold fixes the supported limited-lookahead policy and its critical
   top/runner comparisons.
2. The evidence fold estimates the comparison gap, Bellman-target variance, and
   future behavior frequency without replacing the gap by a current-pilot LCB.
3. The performance-difference scaling assigns each comparison a normalized
   return impact. Comparisons may be omitted only within the remaining epsilon
   loss budget.
4. The estimated future requirement and current-budget sufficiency are separate
   outputs. The folds are exchanged and combined conservatively.

For each retained comparison,

\[
\widehat N_{s,k}=\max\!\left\{1,
\frac{2L\sum_{a\in\{a^*,a'\}}\widehat v_a/\widehat d_a}
     {\widehat\Delta^2},
\frac{7L\sum_{a\in\{a^*,a'\}}1/\widehat d_a}
     {3\widehat\Delta}\right\}.
\]

The depth requirement is the maximum over retained comparisons. If no action
distinction is needed inside the epsilon budget, the status is
`epsilon_trivial` with requirement zero. It is no longer disguised as an
`identified` point at the artificial floor one. Missing critical support remains
`unsupported`; a nonpositive independent-fold gap remains `unresolved`.

This is a target-aligned empirical-Bernstein forecasting approximation. It is
not the exact alternative-MDP KL optimum and carries no instance-optimal
finite-sample theorem.

## Frozen development results

The final 225-pilot run completed 22,725 depth rows with no errors:

- 210 cases were `unsupported`;
- 13 were `epsilon_trivial`;
- one had a nontrivial identified profile;
- one was `unresolved`.

Across depths, 95 rows were nontrivially identified and all 95 requirements
were distinct, ranging from 20,212,236 to 43,870,552 transitions. Another 1,313
rows were explicitly epsilon-trivial. No nontrivial identified row used the
floor one.

The matched stochastic factorial gives the stronger mechanism check. Of 810
cells, 254 were identified and 556 lacked critical support. All 254 finite
requirements were distinct and none equaled one. The aggregate resolved rate
rose with the nested budget:

| Budget multiplier | Resolved rate |
|---:|---:|
| 2 | 0.0037 |
| 10 | 0.2185 |
| 50 | 0.7185 |

The corresponding legacy run had 0 identified cells, 568 unsupported, and 242
unresolved. Because both factorial versions use the same generators and random
streams, this before/after comparison isolates the estimator change.

The executable audit passes all 13 implementation/mechanism checks in
`artifacts/research/oeh_v2_final_audit_v2_2026-09-20/`.

## Predictive result

Removing saturation did not make OEH useful automatically. The nested grouped
B0--B3 evaluation still fails all registered development gates:

- B2 does not consistently improve B1;
- B3 does not add significant predictive information beyond B2;
- the B2 selector does not beat the training-fold best-fixed learner;
- B3 selector mean return is 0.5651 versus 0.6803 for best-fixed, a difference
  of -0.1151 with 95% MDP-cluster interval [-0.1901, -0.0442].

This fixes the estimator's logical and saturation problems within its stated
scope, but it does not validate the original learner-selection claim. Further
tuning on the same 75 groups would be leakage. A new positive claim requires a
newly registered development cohort and then the still-sealed evaluation split.

## Reproducible evidence

- Formula/protocol: `docs/research/OEH_V2_PROTOCOL_2026-09-20.md`
- Implementation: `code/oeh_crossfit_v2.py`
- Unit tests: `tests/test_oeh_crossfit_v2.py`
- Full profile: `artifacts/research/oeh_crossfit_v2_development_final_v2_2026-09-20/`
- Controlled factorial: `artifacts/research/controlled_stochastic_factorial_v2_development_2026-09-20/`
- Mechanical audit: `artifacts/research/oeh_v2_final_audit_v2_2026-09-20/`
- B0--B3: `artifacts/research/stochastic_b0_b3_oeh_v2_development_final_2026-09-20/`
