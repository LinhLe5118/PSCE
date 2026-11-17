# The method: OEH (Offline Effective Horizon)

This page describes the final method as frozen on 14/10/2026 and used in the three final tests. The results are in
[FINAL_TESTS.md](FINAL_TESTS.md); the data in [DATA.md](DATA.md).

## Problem and target

- A *problem* is a finite MDP `M = (S, A, P, R, rho, gamma)` with rewards in [0, 1] and `gamma = 0.95`, together
  with the behavior policy that logs its data. A *learner* is one offline RL algorithm with fixed settings: tabular
  behavior cloning (BC), conservative fitted Q-iteration (CFQI) or implicit Q-learning (IQL). A *case* is one pilot
  and one learner.
- The normalized return of a policy is `NR(pi) = (J(pi) - J_rand) / (J* - J_rand)`: 0 for the uniformly random
  policy, 1 for an optimal one.
- A budget `xb` means `b|S||A|` logged transitions, with `b` in `(1, 2, 5, 10, 20, 50)`. The pilot has size `x10`.
- The *label* of a learner on a problem is the first budget from which at least 8 of 10 independent training runs
  reach `NR >= 0.95` and keep doing so at every larger budget; if no budget qualifies, the label is `>x50`.
- OEH receives the pilot, the learner and the public facts `|S|`, `|A|`, `rho`, `gamma`. It never sees `P`, `R` or
  the behavior policy. It returns a probability for each of the seven classes `x1, x2, x5, x10, x20, x50, >x50`.
- Forecasts are scored by the integrated Brier score over the six budgets (IBS) and the seven-class log loss (LL,
  floored at 1e-12), averaged within each MDP and then across MDPs.

## Baseline and the idea of OEH

The baseline **B2** describes the pilot by 18 generic statistics and turns them into a forecast with the same
forecasting head as OEH. B2 and the full head of OEH therefore differ only in their inputs. B2 leaves three gaps:
its statistics describe the log, not how a particular learner reacts to it; a small pilot leaves much of the MDP
unseen; and a fitted forecaster can be unreliable on pilots unlike its training data. OEH adds evidence obtained by
**running the learner itself**, in MDPs sampled from a posterior fitted to the pilot and on parts of the pilot, a
planning-risk measure adapted from the effective horizon of Laidlaw et al. (2023), and a gate for unfamiliar pilots.

## The six blocks

| Block | Input | Output | Builds on |
|---|---|---|---|
| 1. Generic statistics (B2) | pilot | 18 statistics (group G1) | trajectory quality and state–action coverage (Schweighofer et al., 2022), trajectory bootstrap |
| 2. Learner simulation | pilot, learner | class distribution of the learner in sampled MDPs (G2, G3, G6) | posterior predictive distribution (Gelman et al.) |
| 3. Planning risk | posterior mean model | offline horizon `h_H` and risk profiles (G4, G5) | effective horizon (Laidlaw et al., 2023, 2024) |
| 4. Pilot diagnostics | pilot | action-identification risk and cross-fitted learner diagnostics (G7, G8) | empirical Bernstein bound; learning-curve extrapolation |
| 5. Forecasting heads | 108 features | full head `p`, finite-only head `qO`, robust head `qR` | XGBoost AFT, ordinal threshold models, stacking |
| 6. Shift gate | heads, distance `d` | final forecast | nearest-neighbor outlier score, dataset shift |

**Block 1.** Size and coverage (6), trajectory return (4), transition randomness (6) and two trajectory-bootstrap
spreads with 200 replicates. Coverage is normalized by `|S||A|`, which is public, instead of by reference datasets.

**Block 2.** Each `P(.|s,a)` gets a Dirichlet posterior centered on the uniform distribution with one shared
concentration estimated from the pilot counts; the behavior policy gets a Dirichlet mixture whose size is chosen by
BIC; each reward is a function of the next state, of the pair or of the transition, whichever is the shortest form
that fits every observed reward (unobserved rewards are zero). For 64 drawn MDPs whose optimal return exceeds the
random return by at least 0.2, the block repeats the labeling experiment with the actual learner: ten simulated
training streams, the six nested budgets, exact evaluation in the drawn MDP and the 8-of-10 rule. The class
probabilities use the Krichevsky–Trofimov estimate `(n_c + 1/2) / (B_eff + 7/2)`. The distribution is an input of the
heads, not a forecast on its own.

**Block 3.** In the posterior mean model, starting from the values of the fitted behavior and maximizing over the
actions it can log (in practice all actions), the block computes for each lookahead depth `k = 1..100` the smallest
number of logged transitions `N_k` at which the expected share of return lost through wrongly ranked actions (normal
approximation with a delta-method variance, weighted by state occupancy) fits within the margin above the target.
`h_H = min_k log_|A| N_k` over the depths whose policy reaches the target. Unlike the effective horizon, `h_H` is a
number of transitions on a logarithmic scale, not a number of lookahead steps.

**Block 4.** Without any fitted model: an action-identification risk from the empirical Bernstein inequality in the
empirical model of the pilot (4 features), and the learner cross-fitted on five folds of the pilot prefixes `x2`,
`x5`, `x10` and scored on the held-out fold, summarized by a power law of the shortfall (6 features per learner).
Each case also carries the diagnostics of the other learners.

**Block 5.** A head averages three XGBoost models (200 rounds each): an accelerated-failure-time model with normal
errors and scale 1.2 on interval-censored budgets (depth 2, learning rate 0.05, minimum child weight 5); six binary
models for `E <= j` made monotone by a running maximum; and a seven-class model (depth 3, minimum child weight 10).
The average uses non-negative weights that sum to one, fitted by the out-of-fold log loss with five folds that keep
the cases of an MDP together. The **full head** uses all 108 features on all meta-training cases; the **finite-only
head** uses them on the cases with a finite label; the **robust head** uses the 46 features of groups G1, G7 and G8.
Each head is fitted separately for each learner.

**Block 6.** `d` is the mean distance of the pilot's standardized B2 statistics to its ten nearest meta-training
cases. The forecast is

```
p_OEH = (1 - eps(d)) * [ (1 - p6) * ((1 - lambda(d)) * qO + lambda(d) * qR),  p6 ] + eps(d) * prior_learner
lambda(d) = lambda_max * min(1, d / d0),   eps(d) = eps_max * min(1, d / d0)
```

It keeps the full head's probability `p6` that no budget is enough, takes the finite part from the finite-only head
near the meta-training data and from the robust head far from it, and mixes in a little of the learner's class
frequencies. The frozen values are `lambda_max = 1`, `eps_max = 0.02`, `d0 = 3`
([`oeh_v2_spec.json`](../../artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json)); they were chosen on
five development sets by a rule written before any fit.

## Learners

All three are tabular versions written for the project. BC copies the logged action frequencies. CFQI follows fitted
Q-iteration but updates and acts only among logged actions. IQL weights logged actions by their estimated advantage
(expectile 0.7, inverse temperature 3). CFQI and IQL run at most 100 updates and stop when value changes are at most
1e-10; unlogged states use the uniform random policy. Policies are evaluated exactly in the known MDP.

## Code

The single entry point is [`scripts/analysis/oeh_pipeline.py`](../../scripts/analysis/oeh_pipeline.py): `pilot_inputs`
computes Blocks 1–4, `fit_heads` Block 5 and `final_law` Block 6. Where each block lives is listed in the
[root README](../../README.md#final-method-the-six-blocks-of-oeh).
