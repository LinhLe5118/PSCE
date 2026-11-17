# OEH-v4E final candidate and evidence boundary

Date: 20 September 2026.

Candidate freeze commit: `f556bc57c1ae3bda12878a67c955d2e55565ccd5`.

## Final construction

OEH-v4E separates prediction of data requirement from the learner decision.
The requirement model uses the exploratory `B2 + OEH3 + LC-PMS` block because
generic planning risk and learner-conditional held-out uncertainty measure
different sources of difficulty. The learner selector uses two nested-OOF
return regressors:

1. `B2 + OEH3`, the lower-variance generic OEH model;
2. `B2 + OEH3 + LC-OEH4`, the full learner-conditional model.

For learner `l` and group `g`, the decision score is

```text
s_v4E(g,l) = 0.5 s_OEH3(g,l) + 0.5 s_OEH3+LC(g,l).
```

The equal weights follow symmetry between the generic and learner-conditional
views. They are not fitted to return labels. Each component standardizes
features and chooses ridge strength only inside its training folds. In every
outer fold, the comparator is the learner with highest mean return on the
outer-training MDPs. The selector chooses the learner with largest `s_v4E`;
ties follow the registered CFQI, IQL, BC order.

This construction fixes the remaining decision mismatch. OEH-v3 estimated the
probability of crossing a high-return threshold and used that probability to
choose a learner. OEH-v4E retains the censored requirement model for scientific
diagnosis but optimizes expected return for the budget-10 selection decision.
The equal ensemble reduces the variance observed when either the generic or
learner-conditional return model is used alone.

## Complete development family

The final family contains all 15 selectors examined after the frozen v3 run,
including direct LC selection, nine return-model blocks, two paired-advantage
rules, the inner-tuned paired rule, and OEH-v4E. Failed candidates remain in
the Holm family.

OEH-v4E obtains mean return `0.695515` versus `0.685941` for training-fold
best-fixed. The paired gain is `+0.009574`, with 95% MDP-cluster interval
`[+0.002438,+0.018245]`, raw `p=0.0034`, and Holm-adjusted `p=0.0442` across
all 15 selectors. It switches learner in 52 of 300 MDP-policy groups and
matches the per-group oracle in 220 groups.

The development decision endpoint therefore passes for the final exploratory
candidate. The predictive `B2 + OEH3 + LC-PMS` block also passes its exploratory
screen for pooled log likelihood and integrated Brier after Holm correction.

## Leakage and claim boundary

- No additional development MDP was generated.
- The 120-MDP evaluation split was not accessed.
- Every reported selector return is based on outer-OOF predictions.
- Equal ensemble weights were not selected from outcomes.
- The existing development cohort influenced the choice of the final candidate;
  the positive result is therefore adaptive development evidence.
- OEH-v4E is now fixed for an independent run. Its result cannot be called
  confirmatory or SOTA until the sealed cohort is opened once under a recorded
  protocol and succeeds without further method changes.

## Reproduction

```text
python scripts/analysis/run_oeh_ensemble_selector.py \
  --previous-screen-dir artifacts/research/oeh_paired_advantage_selector_v2_2026-09-20 \
  --ensemble-input-dir artifacts/research/lcoeh_v4_exploratory_final_2026-09-20 \
  --output-dir artifacts/research/oeh_equal_ensemble_selector_2026-09-20
```

Primary artifact:
`artifacts/research/oeh_equal_ensemble_selector_2026-09-20/`.
