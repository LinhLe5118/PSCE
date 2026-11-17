# Abstract

Offline reinforcement learning (RL) trains a policy from a fixed log of past interaction, so the amount and kind of logged data decide what a learner can achieve. Before collecting a large dataset or running an expensive training grid, a practitioner would like to know how much data a specified learner will need. This thesis studies that question as a forecasting problem in finite stochastic Markov decision processes (MDPs). From a pilot log of ten transitions per state–action pair, we predict for one registered learner the first of six data budgets at which at least eight of ten independently trained seeds reach 95% of the optimal normalized return and remain there at every larger budget; when no measured budget suffices, the outcome is a seventh, right-censored class. The target is therefore learner-specific and empirical rather than a property of the MDP alone.

We propose Offline Effective Horizon (OEH), a pilot-only forecaster with three ingredients: a posterior-predictive simulation that replays the actual learner on MDPs compatible with the pilot, a planning-risk descriptor computed on the posterior-mean model, and generic coverage statistics. A fixed ensemble of an accelerated-failure-time model, direct threshold models and a multiclass model converts these signals into seven class probabilities. The same forecasting procedure, fitted on generic pilot statistics only, is the main control.

All forecasts were sealed before training outcomes on a cohort of 120 new MDP kernels and 108,000 learner jobs. On the 84 stochastic MDPs, OEH lowers the integrated Brier score by 15.97% and the seven-class log loss by 10.22% relative to the matched control, and all 22 registered comparisons have positive simultaneous lower bounds, including one against a comparator with privileged access to the true MDP. The gain is clearest for value-based learners and absent for one MDP family. An exploratory block ablation, transfer tests on independently published MDPs and a component test on a published offline dataset show where the method helps and where it does not. As a secondary application, a learner selector improves success and return over the strongest fixed learner but not over an earlier internal router. The evidence supports requirement forecasting within the studied MDP families, not a general superiority claim.

**Keywords:** offline reinforcement learning, data requirement forecasting, right censoring, stochastic tabular MDPs, effective horizon, posterior predictive simulation

# Table of Contents

# List of Abbreviations

Table: Table 0.1. Abbreviations used in this thesis.
| Abbreviation | Meaning |
|---|---|
| AFT | Accelerated failure time (survival model) |
| B2 | The 18 generic pilot statistics used as the main control |
| BC | Behavior cloning |
| BCQ | Batch-constrained deep Q-learning |
| BRIDGE | Published collection of tabular MDPs from the effective-horizon study |
| BWD | Bellman–Wasserstein distance |
| CFQI | Conservative fitted Q-iteration (project implementation) |
| CQL | Conservative Q-learning |
| EH | Effective horizon |
| ERI | Estimated relative return improvement |
| IBS | Integrated Brier score over the six budget thresholds |
| IQL | Implicit Q-learning |
| KT | Krichevsky–Trofimov estimator |
| LL | Seven-class log loss |
| MDP | Markov decision process |
| NR | Normalized return |
| OEH | Offline Effective Horizon (proposed forecaster) |
| OEH-B | Posterior-predictive learner-simulation signal of OEH |
| OEH-H | Planning-risk descriptor of OEH |
| OOF | Out-of-fold |
| RL | Reinforcement learning |
| SACo | State–action coverage |
| SPIBB | Safe policy improvement with baseline bootstrapping |
| TQ | Trajectory quality |
| VI-LCB | Value iteration with lower confidence bounds |
