# IV. Experimental Design

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

**Version boundary (H1, 22 September 2026):** Sections 4.1–4.7 below describe
the historical frozen v3 study. The current v4E prospective protocol is Section
4.8 and T1 v1.1.0; the earlier v3 opening condition is superseded for v4E.

## 4.1 Evidence partitions

The project uses four distinct sources of evidence.

1. **Historical development:** 15 base MDPs and 75 MDP-behavior groups used to
   diagnose OEH-v1 and OEH-v2. This cohort is consumed for method development.
2. **Controlled factorial:** matched corridor blocks varying slip, behavior
   coverage, and budget. This panel tests mechanism and monotonic response, not
   learner-outcome generalization.
3. **Post-freeze development holdout:** 60 newly generated base MDPs and 300
   MDP-behavior groups used once for the frozen OEH-v3 gate.
4. **Registered evaluation:** 120 base MDPs that remain sealed until all
   development criteria permit one confirmatory run.

The fresh cohort preserves the registered structural proportions: 16 uniform
Garnets, 20 forward-layered MDPs, 16 lock-structured MDPs, and 8 bottleneck MDPs.
It uses a new seed namespace and shares no instance identifier with evaluation.

## 4.2 Leakage firewall

OEH-v3 formula, four-variable feature block, cohort generator, seed namespace,
folding rule, and primary endpoint were committed before the fresh buffers were
generated. The generator refuses to run from a dirty working tree and writes the
freeze commit into the manifest.

Each MDP-behavior group has independent pilot and training seed namespaces. OEH
and B0-B2 features read only pilot transitions. Training streams define learner
outcomes and returns. Feature extraction completes before outcomes are joined.
Exact optimal and random-policy returns appear only in post-training scoring.

All five behavior policies from a base MDP remain in the same outer fold and the
same bootstrap cluster. Standardization and ridge selection are fit within the
appropriate training fold. The best-fixed comparator is also chosen using only
outer-training MDPs.

## 4.3 Learners, budgets, and empirical requirements

The principal panel contains tabular CFQI, tabular IQL, and tabular behavior
cloning. Training uses independent prefixes at

`{1, 2, 5, 10, 20, 50}|S||A|`

with ten training seeds per group and budget. Every learned policy is scored by
the same exact normalized return

`NR(pi) = [J(pi)-J(pi_random)]/[J^*-J(pi_random)]`.

For a learner, the empirical requirement is the first budget at which at least
8 of 10 seeds reach `NR >= 0.95` and every larger tested budget also meets that
criterion. If no suffix qualifies, the requirement is right censored at the
largest grid interval. A single successful intermediate budget does not count as
success when a larger budget fails.

The same-data comparison panel contains Pi_b-SPIBB, discounted VI-LCB, discrete
BCQ, and discrete CQL. Each is labeled by provenance: unchanged author core,
paper-spec port, author learner through an adapter, or documented library
implementation. Fixed configurations make these robustness comparisons rather
than an equal-tuning-budget leaderboard.

## 4.4 Pilot feature blocks

The nested blocks are fixed before the fresh outcomes exist.

- **B0:** log pilot transitions and log state-action count.
- **B1:** B0 plus observed state/action coverage, visitation entropy, minimum
  positive count, trajectory-return moments, and terminal-trajectory fraction.
- **B2:** B1 plus transition entropy, successor diversity, low-count fractions,
  and trajectory-bootstrap uncertainty summaries.
- **B3:** B2 plus OEH-v3 log requirement, selected-depth fraction, current-pilot
  risk, and trajectory-half policy instability.

All features are finite. A profile with no selectable depth receives the
pre-specified worst-risk and worst-instability representation and a censored
requirement scale; the analysis does not replace missing OEH with a favorable
zero.

## 4.5 Predictive model and endpoints

Each learner receives a discrete-time hazard model for the grid-indexed empirical
requirement. Five outer folds provide out-of-fold predictions; four inner grouped
folds choose ridge penalty from `{1e-4, 1e-3, 1e-2, 1e-1, 1}`. Proper scoring uses
mean log likelihood and integrated Brier loss with right censoring represented in
the discrete hazard likelihood.

The pre-committed **primary OEH endpoint** is the pooled B3-B2 difference in mean
log likelihood across the three learners. A paired bootstrap resamples base MDPs
10,000 times and keeps all associated policies and learners together. OEH-v3
passes the primary predictive gate only if the lower endpoint of the 95%
interval is greater than zero.

Integrated Brier loss is a secondary endpoint; improvement requires the upper
endpoint of the B3-B2 interval to be below zero. Per-learner contrasts remain
descriptive secondary analyses. This hierarchy avoids requiring six correlated
tests to be significant simultaneously on a modest number of MDP clusters.

## 4.6 Learner-selection endpoint

At budget `10|S||A|`, the selector chooses the learner with the highest
out-of-fold probability of having reached its requirement. The comparator is the
learner with the highest mean return among outer-training MDPs, applied unchanged
to the held-out fold. Oracle-best is reported only as an upper reference.

The selector endpoint is the paired B3 return minus best-fixed return. The gate
passes only when its 95% MDP-cluster bootstrap interval lies strictly above zero.
Predictive improvement and selector improvement are reported separately because
a better proper score need not change the chosen learner.

## 4.7 Reproducibility and stopping rule

Runners reserve new output directories, verify buffer hashes, record source and
input hashes, and reject non-development records. The fresh cohort is single use.
If the gate is negative, the method cannot be changed and reevaluated on the same
cohort as if it were independent. The result must be reported, and any new method
requires another pre-committed cohort.

The evaluation split is opened once only if the frozen development objective is
met. Otherwise the thesis remains a development study with a scoped positive or
negative conclusion according to the observed endpoints.

## 4.8 Current v4E prospective protocol (T1 v1.1.0)

The 120-MDP cohort is partitioned before fitting into 84 stochastic MDPs
(30 Garnet, 36 forward, 18 bottleneck) and 36 deterministic lock controls.
The primary analysis, all inner/outer fitting, comparator and fallback use only
the 84 stochastic MDPs. Controls have their own nested pipeline and descriptive
analysis. Five behaviors per MDP do not create additional independent MDPs.

Five outer/four inner folds confirm the locked meta-learning procedure on a new
cohort, without pooling development. V4E selects by equal-weight predicted
return, not requirement probability. Exact ties use CFQI/IQL/BC order. Declared
nonfinite diagnostic scores cause whole-group best-fixed fallback; file/feature/
fitting failures stop execution rather than becoming abstentions.

The primary decision margin is .005. Hybrid-versus-B2 and B2-versus-B1 pooled
log-likelihood contrasts form the other two members of the Holm family. Use
2,000 paired base-MDP refit-bootstrap draws stratified by outer fold, with
training-only standardization, ridge selection and best-fixed recomputed in
each draw. Fixed-OOF resampling is descriptive sensitivity only. Gates require
the registered direction, adjusted p<=.05 and the lower CI exceeding the margin.

The .95 label success threshold and full-suffix 8/10 rule remain unchanged.
Prefixes are nested within each training stream; seeds are independent and
pilot namespaces are disjoint. Formula, runner and independent review must pass
before one E1 execution. H1 documents the method and its limitations; it does not
authorize opening data or certify D1/B2 completion. Full operational rules are
in [T1 v1.1.0](../protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0_2026-09-21.md).
