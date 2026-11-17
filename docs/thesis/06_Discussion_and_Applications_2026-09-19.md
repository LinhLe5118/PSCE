# VI. Discussion and Applications

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

## 6.1 Answer to the research question

Pilot coverage, trajectory quality, stochasticity, and uncertainty predict part
of the variation in learner-specific data requirements on the fresh development
cohort. OEH-v3 adds a favorable secondary calibration signal but does not pass
its frozen primary or decision gate. The adaptive LC-OEH extension improves
both proper scores, and the final equal ensemble improves learner selection over
best-fixed after Holm correction across the complete selector family. The
positive method claim is therefore supported on development data, while
independent confirmation remains outstanding.

This result separates measurement from utility. OEH-v1 collapsed because its
loss shrank with estimation dispersion. OEH-v2 exposed support failure but
abstained too often to provide useful variation. OEH-v3 constructs varied,
finite, support-constrained forecasts for all primary pilot groups. Removing
the measurement defect was necessary. Aligning the decision model with return
at the deployment budget and averaging generic and learner-conditional OOF
predictions supplied the additional decision value.

## 6.2 What the OEH-v3 variables mean

Each term encodes a distinct obstacle to identifying a useful action from a
fixed log. The normalized action gap measures how much evidence is needed to
separate the top supported actions. Conditional Bellman-target variance measures
transition and reward noise. Behavior frequency converts total samples into
samples for each action. Discounted occupancy and action advantage translate a
local ranking error into possible return loss. Planning regret records what
finite lookahead leaves unresolved even if action comparisons were known.

The empirical-Bernstein tail combines these quantities into a monotone error
curve as future sample size grows. Its inversion gives a forecast in data units,
while `k + log_A N` preserves the planning-depth plus evidence-cost structure of
effective horizon. These choices explain the formula; they do not turn it into
an exact finite-sample certificate.

## 6.3 Why the primary gate may still fail

Three explanations remain consistent with the evidence. First, the proxy can be
well behaved but poorly aligned with the empirical requirements of CFQI, IQL,
and BC. Their optimization, regularization, and policy extraction differ from
the support-constrained planning sequence used by OEH-v3. Second, right
censoring is heavy: only 144 CFQI, 55 IQL, and 28 BC groups reach the target on
the registered grid. Third, compressing a depth profile into four features may
discard learner-relevant structure even though it controls model dimension.

The cohort cannot distinguish these explanations without becoming training
data for another method revision. Because the formula and endpoint were frozen
before this single-use cohort, changing them after reading these labels would
convert the apparent holdout into ordinary development data. A valid follow-up
requires a new protocol and new MDP/stream namespace.

## 6.4 Supported practical use

OEH-v3 can be used as a pilot audit within observed action support. It identifies
which comparisons combine small gaps, high transition variance, low behavior
frequency, and high return impact. It can also distinguish an already acceptable
supported policy (`epsilon_trivial`) from a large positive forecast. These
outputs can guide targeted data collection or inspection of logging coverage.

It cannot certify that an unseen action is poor. If a behavior policy assigns
zero probability to an important action, no data-only estimator can recover
that action's consequences without added assumptions. The support-constrained
estimand makes this limitation explicit rather than assigning a misleading
finite full-MDP difficulty.

## 6.5 Relation to the external methods

The same-machine panel shows that the evaluation stack can train and score a
range of tabular and discrete offline-RL methods on common data. It does not
validate OEH, because method return and incremental pilot-feature value are
different questions. It also does not rank the broader field: SPIBB and VI-LCB
use fixed documented configurations, and BCQ/CQL use a short one-seed GPU
budget. A publication claiming neural SOTA would require matched tuning budgets,
multiple initialization seeds, convergence diagnostics, and a benchmark suited
to function approximation.

## 6.6 Threats to validity

The study covers finite stochastic tabular MDPs from four generator families.
Exact policy evaluation makes normalized return measurable but narrows external
validity. The empirical-Bernstein expression treats the learned value targets
as plug-in quantities and does not account fully for their data dependence.
Statewise action-ranking errors are aggregated as a risk proxy rather than a
proved joint bound. The return-impact weights and planning regret inherit error
from the empirical model. Finally, the fresh cohort contains 60 independent MDP
clusters; 300 behavior groups increase observations but not the number of
independent environments.

## 6.7 Publication position

The current work is a complete and defensible thesis with a theorem-backed
saturation diagnosis, a support-constrained risk construction, a controlled
stochastic-data study, and a positive decision-aligned development result. The
paper can be framed as a method-development study rather than a negative-result
paper.

It is not yet ready for an A/A*/Q1 or SOTA claim because OEH-v4E was chosen
after inspecting the development cohort and the external neural panel has an
unequal tuning budget. The next publication step is one frozen, independent
evaluation of OEH-v4E, followed by reporting that result without another method
revision.

## 6.8 Stochastic tabular data as a thesis extension

The thesis can make a broader contribution without claiming that a scalar OEH
solves learner selection. The existing controlled factorial supports a
three-axis view of offline data:

1. **Transition stochasticity** controls successor dispersion conditional on a
   state--action pair. Slip interventions change this axis directly.
2. **Behavior quality and decision coverage** control which actions are logged
   and whether the observed actions are useful. Raw coverage can rise while
   return falls when additional mass is placed on decoy actions.
3. **Finite-sample observability** controls whether stochastic successors and
   action gaps can be estimated at all. Budget changes measured entropy,
   empirical stochastic-cell counts, and bootstrap uncertainty even when the
   underlying kernel is fixed.

This decomposition explains why transition entropy alone is not an environment
difficulty score and why state--action coverage alone is not a dataset-quality
score. It also supplies a coherent role for OEH: planning risk and
learner-conditional uncertainty are diagnostic components within a multivariate
data profile, rather than a universal scalar ranking.

No additional development dataset is needed for this extension. The thesis can
report the 810-cell matched factorial, the 60-MDP post-freeze cohort, and the
historical saturation controls as distinct evidence sources. Since all have
already informed method development, the resulting claims must remain
descriptive and exploratory. Independent positive-method or SOTA claims would
still require new external evidence.
