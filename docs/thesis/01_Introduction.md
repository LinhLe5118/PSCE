# Difficulty-Aware Offline Reinforcement Learning

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

## I. Introduction

### 1.1 The decision before training

Offline reinforcement learning (offline RL) learns a policy from a fixed log of
past interaction. This setting removes the safest response to uncertainty: the
learner cannot visit an uncertain state, try a poorly covered action, and update
its estimate. The usefulness of an offline dataset is consequently determined
by more than its number of transitions. It depends on where the behavior policy
placed those transitions, how stochastic the observed outcomes are, which
decisions control return, and how the learning algorithm uses partial support.

In practice, these factors are often assessed after the expensive step. Several
learners are trained at several dataset sizes, their policies are evaluated, and
the learning curves reveal which method worked. That procedure measures the
answer but does not predict it. A dataset diagnostic would be more useful if it
could be computed from a small pilot, before the full training grid is run.

This thesis asks whether such a diagnostic can support a concrete decision in
finite stochastic MDPs:

> Given a pilot dataset and a future data budget, can pilot-only features predict
> whether each offline learner will reach a target normalized return, and can
> those predictions select a learner with higher return than a learner fixed
> without seeing the test MDP? Does an Offline Effective Horizon add information
> beyond coverage, trajectory quality, and stochastic uncertainty?

The question is learner conditional and budget conditional. It does not seek a
universal ordering of environments from easy to hard. It also requires an
explicit scope for support: no statistic computed from a fixed dataset can
identify the value of an action that the behavior policy never takes without
additional assumptions.

### 1.2 From effective horizon to an offline risk profile

Effective horizon combines two sources of difficulty in interactive RL: how far
an agent must plan and how much rollout evidence is needed to distinguish the
relevant actions. The idea is attractive offline because Bellman depth alone is
not a data requirement. The missing component is evidence under the logged
behavior distribution.

The first Offline Effective Horizon (OEH-v1) attempted to replace rollout
evidence with a bootstrap action-margin loss. On development data it returned
the same minimum multiplier across every assessed depth. A theoretical audit
showed that this was not evidence that all instances were easy. Writing an
action gap as Δ = zσ transforms each registered loss term into
σzΦ(−z), and `sup zΦ(−z) = 0.169972`. The entire loss is therefore bounded by an
occupancy-weighted dispersion term. When that term lies below the fixed
tolerance, every depth passes and the rule collapses to its minimum.

This diagnosis creates two requirements for a replacement. First, it must infer
a future sample requirement rather than ask whether the current pilot already
certifies its own action gaps. Second, it must align action-ranking error with
the target return. OEH-v3 meets these requirements with a support-constrained
risk curve. At each planning depth it combines:

- empirical-Bernstein separation rates for supported top-versus-runner actions;
- future behavior frequencies, so rare actions require more total data;
- discounted state occupancy and action advantage, so errors are weighted by
  their effect on normalized return;
- planning regret inside the same observed action support; and
- a trajectory-half instability measure that records sensitivity to the pilot.

The resulting requirement is the smallest future sample size for which the sum
of planning regret and weighted action-misranking risk falls below the target
tolerance. A state with one observed action has no within-support ranking problem
and contributes zero identification cost. This convention does not claim that
unobserved actions are safe; it defines the estimand as explicitly restricted to
the action support of the pilot.

### 1.3 Evaluation as prediction and decision making

A nonconstant OEH profile is only a necessary mechanical property. Its research
value depends on whether it predicts learner outcomes beyond simpler features.
The thesis therefore evaluates four nested feature blocks:

- **B0:** pilot size and problem-size controls;
- **B1:** B0 plus state-action coverage and trajectory-return summaries;
- **B2:** B1 plus transition stochasticity and bootstrap uncertainty;
- **B3:** B2 plus four pre-specified OEH-v3 summaries: log requirement, selected
  depth, risk at the pilot size, and policy instability.

For each learner, the outcome is a right-censored empirical data requirement on
the fixed grid `{1, 2, 5, 10, 20, 50}|S||A|`. Prediction uses nested out-of-fold
models grouped by base MDP. Learner selection is evaluated at
`10|S||A|` against the strongest learner selected using only outer-training
MDPs. These choices separate three questions that are easily conflated:

1. Does the OEH formula produce a meaningful range of pilot-only values?
2. Does that range contain incremental information about learner success?
3. Does using the prediction improve the downstream learner choice?

### 1.4 Research questions

The thesis answers four questions in order.

1. **Failure mechanism.** Why did OEH-v1 saturate, and under which mathematical
   conditions does the same obstruction apply to related losses?
2. **Measurement.** Can a support-constrained, return-weighted OEH avoid the
   artificial finite floor while responding correctly to gap, variance,
   coverage, planning depth, and data scale?
3. **Prediction.** Do stochastic pilot features improve coverage and trajectory
   features, and does OEH-v3 add information beyond that combined baseline?
4. **Decision utility.** Does the induced learner selector outperform a
   train-fold best-fixed learner at the declared budget?

### 1.5 Contributions

This thesis makes five contributions.

**A conditional saturation result.** It derives the ceiling behind OEH-v1 and
separates statistic saturation from learner saturation. Positive controls and a
controlled corridor show that the implementation can return non-minimal values,
while learner returns can improve even when the old statistic is flat.

**A target-return OEH formulation.** OEH-v3 preserves the effective-horizon form
`depth + log_A(requirement)` but defines the requirement by inverting a
return-weighted empirical-Bernstein risk curve. The formula assigns distinct
roles to action gap, stochastic variance, behavior coverage, discounted impact,
and planning error. Its scope and non-claims are explicit.

**A leakage firewall.** Formula, feature block, cohort generator, statistical
endpoint, and seed namespace are committed before the fresh cohort is generated.
OEH reads pilot transitions only; learner outcomes come from independent
training streams and are joined only after feature extraction. The registered
120-MDP evaluation split remains separate.

**A learner-conditional evaluation.** Nested grouped cross-validation,
censoring-aware proper scores, MDP-cluster bootstrap intervals, and a
train-fold-only best-fixed comparator measure both prediction and decision
utility. The primary OEH endpoint is fixed before outcomes exist.

**A same-data baseline audit.** Tabular CFQI, tabular IQL, behavior cloning,
Pi_b-SPIBB, VI-LCB, discrete BCQ, and discrete CQL are separated by compatibility
and provenance. External results are described according to what was actually
run: author core, paper-spec port, or documented library implementation.

### 1.6 Scope of the evidence

The thesis concerns discounted stochastic tabular MDPs with discrete actions.
Exact MDP quantities are used after training to score policies and construct
normalized returns; they are not estimator inputs. The support-constrained OEH
is a forecast, not an exact alternative-MDP KL optimizer or a finite-sample
certificate. Neural experiments are compute-limited robustness checks rather
than tuned SOTA comparisons.

The historical 15-MDP development cohort was consumed while diagnosing OEH-v1
and OEH-v2. It cannot provide independent evidence for OEH-v3. The v3 analysis
therefore uses a single-use, post-freeze cohort of 60 new development MDPs. The
registered 120-MDP evaluation split may be opened only if the frozen development
criteria are satisfied without modifying the method.

### 1.7 Thesis organization

Chapter II positions the work relative to offline RL, effective horizon,
best-action identification, dataset characterization, and algorithm selection.
Chapter III derives the saturation result and OEH-v3 risk formulation. Chapter
IV specifies the benchmark, leakage controls, learners, outcomes, and statistical
tests. Chapter V reports mechanism, prediction, selection, and baseline results.
Chapter VI interprets the findings and their practical limits. Chapter VII
states the conclusions, claims supported by the evidence, and work required for
independent confirmation.
