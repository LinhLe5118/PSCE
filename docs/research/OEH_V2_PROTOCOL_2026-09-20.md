# OEH v2 development protocol (frozen before the replacement runs)

Date: 2026-09-20. Scope: development data only. The sealed evaluation split
must not be read while developing or selecting this estimator.

## Estimand and scope

OEH v2 is a **forecasting covariate** for the data scale needed to distinguish
the critical actions of a limited-lookahead, support-constrained tabular policy.
It is not a finite-sample certificate, an exact alternative-MDP KL optimizer,
or an offline-RL learner. Its scientific test is whether it adds out-of-fold
predictive or learner-selection information beyond coverage, return-quality,
and stochasticity features (B2).

The support-constrained target is necessary: an action absent from the logging
distribution is not identifiable from the log. Such a state is `unsupported`
unless its maximum possible normalized-return impact can be spent from the
remaining epsilon budget. A zero or negative validation-fold gap is
`unresolved`, not silently replaced by a positive number. Fewer than two
trajectories is an explicit cross-fit abstention.

## Frozen computation

Trajectories are assigned deterministically to two folds. In each direction,
the model fold fixes the empirical model, limited-lookahead policy, occupancy,
and top/runner comparison. The other fold estimates the Bellman-target gap,
variance, and behavior frequencies. The directions are swapped and combined
conservatively by their maximum requirement and worst status.

For a retained top/runner pair with validation gap `Delta > 0`, Bellman-target
variances `v_a`, future behavior frequencies `d_a`, and simultaneous log factor
`L`, the total-transition planning scale is

```
N_pair = max(1,
             2 L sum_a(v_a / d_a) / Delta^2,
             7 L sum_a(1 / d_a) / (3 Delta)).
```

This is a two-sample empirical-Bernstein planning approximation. Crucially,
`Delta` is the independent empirical gap, not its current-pilot lower confidence
bound. Consequently, OEH can estimate a future requirement larger than the
current pilot instead of returning `unresolved` by construction. Whether the
current fold already reaches that requirement is reported separately.

The normalized return impact of a supported comparison is bounded using the
performance-difference scaling

```
impact(s) = min(1, d_pi(s) * model_gap(s)
                   / ((1-gamma) * model_value_range)).
```

For an unsupported comparison, the implementation uses the corresponding
worst-case bounded-reward impact. Comparisons may be omitted only while their
total impact fits `epsilon - planning_regret`; the hardest comparisons are
considered first. Every remaining comparison must be identified. This avoids
the old requirement that every action at every occupancy-positive state be
resolved, while preserving an explicit return-loss budget.

Frozen constants for the replacement runs are `epsilon=0.05`, `delta=0.05`,
two trajectory folds, deterministic seed `20260920`, and the registered depth
grid. No constant may be changed after inspecting the replacement results.

## Acceptance checks

1. Unit and edge-case tests pass, including future requirement beyond the
   current pilot, missing critical support, exact ties, irrelevant unsupported
   actions under the support-constrained target, and one-trajectory abstention.
2. No NaN/inf is emitted as a selected requirement; all failures have an
   explicit status.
3. The full development profile finishes without runner errors and contains at
   least one finite, nonconstant identified profile. This rejects mechanical
   all-one saturation; it does not require every dataset to be identifiable.
4. The controlled stochasticity x coverage x budget factorial is rerun with the
   frozen estimator and nested prefixes. Identification/abstention and finite
   requirement behavior are reported, without tuning for a desired direction.
5. B0--B3 uses grouped nested out-of-fold evaluation. OEH is retained as a main
   contribution only if B3 adds information beyond B2 under the registered
   gates; otherwise it is reported as a negative or diagnostic result.

No method can guarantee finite identification for every stochastic tabular MDP
from arbitrary offline data: lack of support and exact/near ties are information
limits. Correct abstention in those cases is part of the estimator contract,
not saturation.
