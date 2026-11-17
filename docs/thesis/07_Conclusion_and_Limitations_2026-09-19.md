# VII. Conclusion and Limitations

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

## 7.1 Conclusion

This thesis asked whether a pilot dataset can predict learner-specific data
requirements and support learner choice in stochastic tabular offline
reinforcement learning. It developed that question through three generations
of Offline Effective Horizon.

OEH-v1 exposed a general failure mode: a degree-one uncertainty-weighted loss
can collapse below a fixed tolerance as estimation dispersion decreases. OEH-v2
removed the artificial finite floor and made non-identification explicit, but
its support rule abstained on most primary cases. OEH-v3 replaced the test with
a support-constrained, return-weighted empirical-Bernstein risk forecast. On a
fresh cohort generated after the formula was frozen, OEH-v3 produced varied
requirements for every primary group without accessing learner labels.

The frozen predictive conclusion is narrower: stochasticity and uncertainty
features improved the simpler pooled predictor, while OEH-v3's primary pooled
log-likelihood improvement was uncertain. The adaptive learner-conditional
extension then improved both proper scores, and OEH-v4E improved return over
best-fixed after Holm correction across all 15 screened selectors. The
development objective is therefore positive; independent confirmation remains
open because the sealed evaluation split is unused.

## 7.2 Contributions supported by the evidence

1. A conditional saturation result explains the original OEH failure and
   separates statistic saturation from learner saturation.
2. A pilot-only OEH-v3 formulation combines planning depth, supported action
   gaps, Bellman-target variance, behavior frequency, occupancy, and normalized
   return impact in a monotone future-sample risk curve.
3. A leakage firewall freezes the estimator and endpoint before a single-use
   60-MDP cohort, separates pilot and training streams, groups folds by base
   MDP, and keeps evaluation sealed after a failed gate.
4. A complete 54,000-fit local grid and a same-machine external-method panel
   provide learner outcomes, censoring-aware labels, and implementation
   provenance.
5. The post-freeze analysis shows that non-saturation alone is insufficient,
   then identifies learner-conditional uncertainty and a fixed equal ensemble
   as the components needed for positive development decision value.

## 7.3 Limitations

- OEH-v3 is a plug-in forecasting proxy, not an exact alternative-MDP KL
  optimizer, confidence certificate, or instance-optimal theorem.
- Its estimand is optimality within observed pilot action support. It cannot
  infer the value of unseen actions.
- The empirical requirements are heavily right-censored, particularly for IQL
  and BC.
- The benchmark contains 60 independent MDPs from four synthetic tabular
  families. Function approximation, continuous actions, and real logged systems
  remain outside the evidence.
- The external neural panel has one seed and fixed short training; it is a
  compatibility check rather than a tuned SOTA comparison.
- The secondary Brier improvement was observed alongside a failed primary
endpoint and cannot be promoted to the main result after inspection.
- LC-OEH-v4 and OEH-v4E reuse the consumed cohort. Their predictive and selector
  signals pass their screening corrections but remain adaptive rather than
  independent confirmation.

## 7.4 Allowed and disallowed claims

The thesis may claim that OEH-v1 saturates under the stated conditions; OEH-v3
removes the observed constant-floor and zero-identification defects; B2 improves
pooled prediction on the post-freeze cohort; and OEH-v3 shows secondary
calibration evidence without passing its frozen primary gate. It may also claim
that the adaptive LC-OEH predictor and OEH-v4E selector pass their Holm-corrected
development screens without opening the sealed evaluation split.

The thesis must not claim that OEH-v3 works for every stochastic tabular MDP,
predicts full-MDP optimal sample complexity, beats B2 on the frozen primary
endpoint, has independent confirmation, or establishes neural SOTA. It must not
describe the unused evaluation split as validation evidence.

It may additionally report that the matched factorial separates transition
stochasticity, behavior quality, and finite-sample observability. In particular,
raw observed coverage can increase while learner return decreases. This is a
descriptive development finding, not a universal causal law.

## 7.5 Next valid study

OEH-v4E is now the fixed candidate for independent evaluation. Its formula,
feature blocks, equal weights, ridge grid, fold rules, endpoints, and complete
multiplicity family must remain unchanged. It may be applied once to the sealed
evaluation split and then reported regardless of outcome. This sequence
preserves the value of the remaining evaluation data and prevents
iterative tuning from being mistaken for confirmation.

Under the decision not to generate another development dataset, the remaining
thesis work should deepen the stochastic-data decomposition using existing
factorial and post-freeze artifacts. It should not reopen estimator tuning; the
sealed split is reserved solely for the one frozen OEH-v4E evaluation.
