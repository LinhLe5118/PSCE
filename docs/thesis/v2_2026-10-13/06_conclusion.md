# 6. Conclusion and Future Work

## 6.1. Answers to the Research Questions

**RQ1 — Does OEH forecast the requirement better than a matched pilot-only control?** Yes, within the studied MDP families. On 84 new stochastic MDPs, with forecasts sealed before any outcome, OEH lowered the integrated Brier score by 15.97% and the seven-class log loss by 10.22% relative to the same forecasting procedure fitted on generic pilot statistics. All 22 registered contrasts, including adapted published quality scores and a comparator with privileged true-MDP reference data, have positive simultaneous lower bounds. The same configuration had already passed the pilot-only tier in an earlier independent cohort, so the result has been repeated.

**RQ2 — How does the result vary?** The gain is present for all three learners and all five behaviors on average. It is robust for CFQI and IQL, uncertain for BC because BC rarely has a finite requirement, large for Garnet MDPs, moderate for forward-layered MDPs, and absent for bottleneck MDPs. Most remaining error lies in placing finite requirements on the budget grid. On independently published BRIDGE MDPs the transfer is learner-dependent: consistently favorable for CFQI, unfavorable for IQL on Atari tasks and for BC on the other panels. The exploratory ablation shows that the OEH blocks are largely redundant with each other; the learner simulation adds detectable information for CFQI, the planning descriptors for IQL's log loss, and the legacy diagnostics nothing detectable.

**RQ3 — Does selection at $x10$ improve on the registered rule?** Yes against fixed CFQI, the strongest fixed learner: seed success rose by 6.67 percentage points and normalized return by 0.0187, both with positive simultaneous bounds. It does not improve on an earlier internal router, and it uses MDP-family labels that a pilot does not reveal. This remains a secondary, conditional result.

## 6.2. Contributions and Scope of the Conclusions

The thesis contributes (i) a measurable, learner-specific and right-censored definition of the data requirement of an offline learner, with a controlled benchmark of 120 MDP kernels and 108,000 learner jobs; (ii) OEH, a pilot-only forecaster that simulates the actual learner on posterior MDPs and combines the result with planning-risk and coverage descriptors in a fitted probabilistic model; and (iii) an evaluation that seals forecasts before outcomes, holds the forecasting procedure fixed across comparators, separates information tiers, and reports negative and mixed results alongside the positive one.

The conclusions are scoped accordingly. The forecasting gain is established for new kernels of the three studied stochastic families under a fixed labeling protocol. It is not a claim of state-of-the-art performance across offline RL, not a guarantee that a predicted budget suffices, not evidence of data or compute savings, and not an attribution of the gain to one component. The selection result shows that the forecasts can inform a decision, not that the decision is optimal.

## 6.3. Future Work

In order of priority:

1. **Confirmatory component test.** Freeze the arms suggested by the ablation — B2, B2 plus learner simulation, B2 plus planning descriptors, their union, and the full set — and evaluate them once on an unopened cohort with a registered contrast family.
2. **Full-method transfer to an independently designed family.** Extend the BRIDGE work with a larger, pre-registered panel and analyze the unfavorable IQL and BC cases without retuning on opened labels. Revisit the behavior prior, whose smoothing appears to mislead BC forecasts, only through a new frozen test.
3. **Smaller pilots.** The pilot currently costs $x10$, which makes early budget classes retrospective. Testing pilots of $x1$–$x2$ would show whether forecasts can guide collection before most of the data are gathered.
4. **Measured decision value.** Define a decision protocol — choose a budget from the forecast, collect, train, and pay for under-budgeting — and measure net data and compute savings including OEH's own cost; the ablation suggests that dropping the learner simulation could reduce that cost substantially.
5. **Richer targets and settings.** Use all sixty success counts per row rather than one class label, and move beyond tabular MDPs to function approximation, where the posterior must be replaced by a learned model.
