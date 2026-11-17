# OEH difficulty revision: direct event probabilities with global calibration

Status: **development candidate; not promoted**. The original E1 cohort and
the first independent 60-MDP cohort have both been inspected during method
development. Neither can independently confirm this revision. The original
sealed OEH and its unsuccessful artifacts remain unchanged.

## Diagnosed mismatch

The raw OEH requirement is a conservative, support-constrained planning proxy,
whereas the task label is the first of six data budgets at which a specified
offline learner reaches normalized return at least 0.95 on at least eight of
ten seeds **at that and every larger budget**. On the first fresh 42-MDP
stochastic panel, 72.06% of task/behavior/learner rows were right-censored
past x50; raw LCOEH declared 100% censored. Its integrated Brier was 0.1476.
The calibrated 30-feature OEH+AFT model reached 0.0641, but its advantage
over generic B2 had a simultaneous lower bound below zero. The E1-to-fresh
point change in that advantage does not itself establish a distribution shift.
More precisely, the B2-minus-OEH Brier improvement was 0.011340 on the
adaptively reused E1 stochastic panel and 0.007228 on the first fresh panel.
The latter panel contains only 42 independent MDPs; 25 favored OEH and 17
did not, and the MDP-level paired differences had standard deviation about
0.0313. The simultaneous one-sided 95% lower bound was -0.008948. Thus the
*uncertainty-adjusted lower bound* fell below zero because the estimated gain
became smaller relative to heterogeneous MDP-to-MDP variation. OEH's
absolute Brier on the fresh panel (0.064138) was lower than its E1 value
(0.076465); calling that an absolute accuracy deterioration would be wrong.

## Candidate fixed before a new outcome

The prediction target remains seven event categories (x1, x2, x5, x10, x20,
x50, and >x50). For each learner and the predeclared stochastic/control panel,
six equally configured binary XGBoost models predict the cumulative event
probability at each budget from pilot-only B2 and OEH features. Rowwise
cumulative maximum creates a valid nondecreasing CDF. The two near-constant
LCOEH requirement/censor fields are excluded because they represent the
unsupported planning bound rather than the measured learner event. All other
feature rules, rounds, and tree parameters are common across environment
families. No true transition kernel or query outcome enters prediction.

To avoid the zero-probability event bins caused by differencing six separately
fitted CDFs, form a convex mixture with the previously frozen OEH+AFT
forecast. The **one global mixture weight**, 0.394954... on the direct model,
was learned by minimizing seven-category log-loss on MDP-grouped,
out-of-fold E1 predictions. The first 60-MDP cohort was used to choose this
model family and inspect results; it is development evidence only.

Development results on 42 stochastic MDPs: direct model Brier 0.057208,
log-loss 1.013275; mixture Brier 0.058982, log-loss 0.680796; frozen
OEH+AFT Brier 0.064138, log-loss 0.810100; B2 AFT Brier 0.071367,
log-loss 0.860517. The 18 deterministic-control MDPs gave mixture Brier
0.041519 and log-loss 0.440661. OEH mechanism features alone under the
direct model had stochastic Brier 0.082671, so a standalone raw OEH
superiority claim is **not** supported. The present candidate is a
learner-specific, pilot-only OEH-augmented estimator.

A further development-only calibration test traced the direct model's high
log-loss to nine true-event rows assigned zero event-bin probability on the
opened stochastic panel. A single prior-mixture coefficient, 0.248073,
fitted by E1 MDP-grouped out-of-fold log-loss removed this degeneracy, but
yielded Brier 0.061674 and log-loss 0.702052 on the opened panel. Both are
worse than the already frozen direct/AFT mixture (0.058982 and 0.680796),
so this calibration is **not promoted**. Its source, manifest, and scores
remain under `artifacts/oeh_difficulty_calibration_development_2026-10-01/`
to make the negative result auditable; it did not alter the sealed 60-MDP
confirmation.

The opened-panel MDP-level B2-minus-mixture Brier advantage varies by family:
0.001459 on nine bottleneck MDPs, 0.021980 on 18 forward-layered MDPs, and
0.007425 on 15 uniform-Garnet MDPs. Only 29/42 stochastic MDPs favor the
mixture. Thus the aggregate gain cannot be treated as a family-uniform or
universal improvement, and no family-specific correction is applied after
looking at these scores.

An additional **architecture-matched** B2 control uses the same six direct
binary models, the same AFT anchor, and an E1 out-of-fold log-loss-fitted
global blend; it differs only in omitting the OEH mechanism features. On the
opened 42-MDP stochastic panel it yields Brier 0.068061 and log-loss
0.729849, versus 0.058982 and 0.680796 for the OEH blend. That is an
exploratory 13.3% and 6.7% reduction, respectively. On the 18 controls the
relative reductions are smaller: Brier 0.043006 to 0.041519 and log-loss
0.445564 to 0.440661. The matched control was added after the opened-panel
outcomes were available and is **not** an independent confirmatory comparator
in the already sealed new cohort. Its reproducible artifact is under
`artifacts/oeh_difficulty_architecture_control_2026-10-01/`.

## New independent gate

Use 60 **new** MDPs, not a new 120-MDP E1: 15 uniform-garnet, 18
forward-layered, 9 bottleneck, and 18 lock-structured. Draw seeds from a
separate fixed namespace before materializing data, verify no seed overlap
with E1 or the first 60. Keep the original five behavior policies, six
budgets, ten seeds, three core learners, success threshold and suffix rule.
Freeze all predictions before any new learner outcome. The primary unit is
the MDP: average 15 behavior/learner scores inside each MDP, then average
42 stochastic MDPs equally. Controls remain separately reported.

Primary gate: candidate has positive simultaneous one-sided 95% MDP-cluster
bootstrap lower bounds for **both** integrated Brier and seven-category
log-loss against train-frequency, pilot coverage, pilot quality,
coverage+quality, and generic B2 AFT. Use 2,000 predeclared resamples, one
shared resample for all ten contrasts, and the maximum studentized centered
error to obtain a common critical value. Keep the stronger OEH+AFT anchor
and all negative/failed outcomes visible. If a learner run fails, do not
selectively drop it. The original protocol allowed a descriptive comparison
on the original 120 E1 MDPs after a complete GO. The later user criterion
additionally requires a 5% margin and an exact published-original comparison;
until those are established the E1 comparison remains withheld. E1 was used
for meta-training/development and cannot become an independent confirmatory
test.

This gate compares methods implemented on a common learner-specific offline
requirement endpoint. Published EH bounds use full-MDP online information;
TQ/SACo require matched reference datasets, and the original continuous
action versions of ERI/EAS/COI are not exact matches to arbitrary tabular
state/action IDs. The coverage/quality signals here are explicitly tabular
adaptations of published ideas. Passing the gate cannot establish superiority
over every published method or a literature-wide SOTA claim.

The [EWRL 2024 dataset-performance predictor](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf)
uses image representations and a CNN to predict policy return for a fixed
learner/framework, whereas this protocol predicts a seven-category sustained
success budget from arbitrary tabular state IDs. The later
[Bellman--Wasserstein dataset score](https://arxiv.org/abs/2510.12638)
uses a behavioral critic and Euclidean action cost on MuJoCo; E1 action IDs
have no intrinsic Euclidean geometry. Calling either an exact tabular
same-endpoint baseline without a new, declared adaptation would be misleading.

The executable sequence is `python -m scripts.analysis.confirm_oeh_difficulty
freeze`, then `prepare`, `evaluate`, and `analyze`. Every later phase verifies
the frozen source/input hashes and the previous seal. `evaluate --resume`
checks the complete plan and every existing job before continuing after an
interruption. `python -m scripts.analysis.compare_oeh_difficulty_e1` refuses
to run unless the independent result manifest is complete and GO. Its output
remains explicitly post-hoc because the 120 E1 MDPs supplied training labels.

## Completed second-cohort result and subsequent development

The sealed 60-MDP run completed all 54,000 learner-budget-seed jobs with no
recorded learner failure. On its 42 stochastic MDPs the frozen OEH blend has
Brier 0.061334 and seven-class log-loss 0.775273, versus generic B2 AFT
0.073828 and 0.898859. This is a 16.92% and 13.75% point reduction,
respectively. All ten registered simultaneous one-sided 95% lower bounds are
positive; the narrowest Brier and log-loss bounds against B2 are 0.001782
and 0.027598. The original registered numerical gate is GO. The later 5%
numerical audit also passes against **registered** baselines, but the
published-original-method condition is still not established.

Before inspecting second-cohort scores, a late-added, architecture-matched B2
forecast was sealed separately. It gives Brier 0.069438 and log-loss
0.809073 on the stochastic panel. OEH's gains against this stronger control
are 11.67% Brier and only **4.18% log-loss**; an exploratory paired
simultaneous log-loss lower bound is negative (-0.018270). This comparator
was not part of the original gate and is not an original published method.
It nevertheless prevents attributing all registered-baseline improvement to
OEH diagnostics rather than the changed forecasting architecture.

Later development-only experiments on the two now-open 60-MDP cohorts tested
a joint seven-class model and a three-way convex combination of AFT,
six-threshold, and seven-class forecasts. The one global three-way weight
vector for each of OEH and B2 was fitted by E1 MDP-grouped out-of-fold
log-loss. On the second cohort, OEH three-way improves absolute scores to
0.059318 Brier and 0.736330 log-loss, versus matched B2 three-way 0.068902
and 0.770115. The log-loss gain is still only **4.39%**, with a negative
exploratory lower bound (-0.008661); therefore this candidate is not
promoted. A learner-pooled seven-class model with one-hot learner identity
performed worse in absolute stochastic Brier/log-loss (OEH 0.068489 and
0.790142) and is also not promoted. These experiments are visibly labeled
development, not a repeated independent confirmation or proof of the
requested 5% margin.

Further structural alternatives did not close that margin. A conditional
discrete-time hazard model yielded second-cohort OEH Brier/log-loss 0.064694
and 0.882858. A two-stage success-by-x50/conditional-budget model yielded
0.069210 and 0.809914. Applying the same learner-conditional E1 OOF
three-way weight fitting rule to both OEH and B2 gave 0.059587/0.736184 for
OEH and 0.068790/0.765128 for B2; the log-loss gain fell to 3.78%.
These are development results from already-open cohorts. None is promoted;
the strongest absolute candidate so far is the global three-way mixture,
whose matched-B2 log-loss gain remains below 5% and lacks a positive
simultaneous lower bound on the second cohort.

References: [XGBoost AFT documentation](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html),
[proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf),
[TQ/SACo](https://proceedings.mlr.press/v199/schweighofer22a.html),
[ERI/EAS/COI](https://arxiv.org/abs/2111.13461),
[stochastic EH](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf).
