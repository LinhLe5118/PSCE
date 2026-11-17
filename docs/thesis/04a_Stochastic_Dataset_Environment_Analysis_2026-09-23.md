# IV-A. Stochastic Dataset and Environment Analysis

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

This section supplements the frozen experimental design without changing its
estimands, feature sets, thresholds, or evaluation cohort. It summarizes only
committed development and controlled-probe artifacts. No additional development
dataset was generated, and the sealed evaluation manifest, buffers, and outcomes
were not accessed.

## Separation of stochastic mechanisms

Transition stochasticity and behavior stochasticity are distinct design axes.
An environment is classified as stochastic when at least one reachable,
nonterminal state-action row has two or more successors with probability above
`1e-12`. This classification is verified from the reconstructed kernel and is
not inferred from a finite pilot. In contrast, behavior stochasticity is induced
by the registered policy generator: Q-star softmax, reward-perturbed epsilon-soft,
structured-random epsilon-greedy, two-policy mixture, or shifted-dynamics
epsilon-soft. Empirical visitation entropy depends jointly on the policy,
environment dynamics, and finite sample and therefore cannot identify either
mechanism by itself.

The reconstructed 60-MDP development cohort contains 44 stochastic MDPs and 16
deterministic lock controls. All eligible kernel rows are stochastic in the 16
uniform Garnets (2,115/2,115), 20 forward-layered MDPs (1,588/1,588), and eight
bottleneck MDPs (749/749). None of the 328 eligible rows is stochastic in the 16
lock controls. At the x10 pilot budget, median empirical transition entropy is
0.2123, 0.1425, 0.2496, and 0.0000, respectively. The kernel classification,
rather than these empirical entropy values, determines the prospective primary
and control panels.

## Coverage, trajectories, and budget

Coverage is measured after collection and never used to alter a buffer. On the
75-group registered development artifact, reachable state-action coverage ranges
from 0.1667 to 1.0000 with median 0.9286. The frozen thresholds 0.25, 0.50, and
0.75 produce 2 low, 4 medium, 18 high, and 51 very-high groups. This imbalance is
a benchmark-design limitation; changing the cut points after observation would
be post-hoc threshold selection. The 60-MDP holdout has median all-state-action
coverage 0.6545, but that denominator includes unreachable cells and is not
interchangeable with the registered reachable-state-action metric.

Trajectory-return moments and terminal-trajectory fractions describe the
realized pilot distribution rather than a policy-only quality construct. Across
the 60 MDPs, median trajectory return ranges from 0.2774 for two-policy mixture
to 0.6751 for Q-star softmax. Median terminal fraction is 0.9412 for Q-star
softmax and 0.9524 for shifted dynamics, but zero for perturbed-reward and
structured-random policies. Structural family, reward placement, horizon, and
coverage vary simultaneously, so these summaries must not be interpreted as
causal policy rankings.

Pilot prefixes contain `{2,5,10}|S||A|` transitions. All 900 recorded prefix
rows equal the registered formula. At x10, group sizes range from 120 to 2,560
transitions, with median 720. Learner training uses
`{1,2,5,10,20,50}|S||A|` and ten seed indices. The prospective core workload is
75,600 fits for 84 stochastic MDPs and 32,400 fits for 36 deterministic controls.
These repeated prefixes, policies, and learner seeds improve measurement but do
not increase the number of independent base-MDP clusters.

## Controlled mechanism evidence and inference boundary

The controlled stochastic factorial contains 810 development rows across six
matched topology/reward blocks. Increasing slip from 0.00 to 0.20 raises mean
weighted transition entropy by 0.1800 and lowers CFQI normalized return by
0.0356 across 270 matched pairs. Increasing the budget from x2 to x50 raises
observed state-action coverage by 0.2687, CFQI return by 0.2001, and the resolved
OEH rate by 0.7148. Increasing the behavior-decoy probability from 0.05 to 0.45
raises observed coverage slightly (0.0387) while reducing CFQI return by 0.2440.
Thus dynamics, behavior, coverage, and budget need not move together and should
not be collapsed into a single difficulty score.

These contrasts are descriptive mechanism checks because only six independent
topology/reward blocks are represented. The development artifacts were consumed
during method design and cannot independently confirm v4E. Confirmatory claims
remain limited to the frozen 84-MDP stochastic mixture, fixed learner portfolio,
fixed budgets, and the statistical gates in T1 v1.1.0.

Machine-readable evidence and source hashes are recorded in
`artifacts/week6/viettq_b1_b2_2026-09-23/b1_stochastic_dataset_analysis.json`.
