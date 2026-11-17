# Tabular E1 comparator replication contract

Status: completed post-hoc E1 extraction and scoring. The separately sealed
21,000-job routing run also completed. Its 60-MDP superiority gate is NO-GO;
the result manifest is in `artifacts/independent_validation_2026-09-30/`.
This document does not change either protocol or the original 120-MDP E1 cohort.

## Common question and inputs

The operational difficulty target is the first E1 budget in
`{x1,x2,x5,x10,x20,x50}` whose complete measured suffix has at least 8/10
seed successes at normalized return >= 0.95, or right-censoring beyond x50.
One row is a (MDP, behavior dataset, learner) triple; all five behaviors and
three core learners of an MDP stay together in cross-validation. The raw E1
pilot buffers at x10 are the only input to data-only methods. Full MDP kernels
and optimal-action gaps are confined to a separate oracle panel. No new MDP,
dataset, learner outcome or seed is generated for this E1 replication.

The executables `python -m scripts.analysis.tabular_e1_quality_signals prepare`
and `python -m scripts.analysis.tabular_e1_quality_signals score` completed
the E1 step. The first verifies
each original pilot buffer hash and extracts 600 rows of unique state--action
counts, complete-trajectory discounted returns, observed-floor ERI,
categorical EAS and within-MDP COI adaptations. It leaves undefined ratios
missing and records partial trajectories. The second scores each declared
adaptation with the same censored AFT objective and original MDP-grouped E1
folds. The 600-row extraction and 1,800-label scoring manifests are in
`artifacts/tabular_e1_quality_signals_2026-10-01/` and
`artifacts/tabular_e1_quality_scoring_2026-10-01/`. On the 84 stochastic
MDPs, the strongest combined adaptation scores 0.08648 integrated Brier;
B2+OEH AFT scores 0.07646 in the separately recorded E1 comparison. This
is descriptive because E1 was reused and the adaptation is not an exact
implementation of any published original method.
The audit watcher in
`artifacts/independent_validation_2026-09-30/post_routing_e1.ps1` waits for
all 21,000 routing jobs, runs its sealed analysis, then invokes the two E1
commands in order. Its status is `complete`; stdout, stderr and status
artifacts for every step are retained.

The target and information set must be stated for every numerical result.
Published methods with a different target can additionally be assessed by
rank association with the E1 endpoint, but that number is a *transfer test*,
not a reproduction of a published theorem or benchmark score. E1 was reused
during OEH development, so all E1-only comparisons are post-hoc.

## Source-to-E1 mapping

| Literature method | Published quantity | E1 reproducibility decision |
|---|---|---|
| Laidlaw et al. 2023/2024 EH, EPW, covering length | Interactive-RL complexity from true MDP; 2024 EH includes optimal-action gaps | The existing `artifacts/e1_oracle_qvi_posthoc_2026-09-30/` already computes 120 reconstructed-MDP oracle descriptors and offline rank transfer. It is not an offline, pilot-only competitor. Deterministic GORP training and exact random-cover episode quantiles have not been reproduced. |
| Schweighofer et al. 2022 SACo | `unique(S,A in D) / unique(S,A in Dref)`, where `Dref` is a same-MDP, same-size replay reference | E1 contains no designated replay reference dataset. The existing observed-state-action fraction is a **tabular adaptation**, not exact SACo. An exact score is undefined unless an independently justified reference is supplied. |
| Schweighofer et al. 2022 TQ | `(mean discounted trajectory return(D)-mean(Dmin))/(mean(Dexpert)-mean(Dmin))` | E1 contains no designated minimal/expert reference datasets. Existing trajectory-return features are **adaptations**. Picking extrema from the E1 outcome table would leak labels; picking best/worst behavior after viewing results is prohibited. |
| Swazinna et al. 2021 ERI | `(max trajectory return-mean trajectory return)/mean trajectory return` after shifting by a known nonnegative return floor | E1 has signed returns and no registered task-level floor in its data-only input. Any floor estimated from the same pilot must be labeled an adaptation and all zero denominators retained as undefined. |
| Swazinna et al. 2021 EAS/COI | Gaussian behavior-policy action stochasticity; COI combines ERI/EAS ranks 2:1 | E1 actions are categorical. State-conditional categorical entropy is a declared tabular adaptation; the published continuous-action Gaussian EAS and hence original COI cannot run unchanged. COI also ranks multiple datasets together, giving it a different information set from a single-dataset OEH query. |
| Mateos-Melero et al. EWRL 2024 predictor | Domain-specific rendered or state-scatter image fed to a CNN to predict a fixed learner's return | The E1 random tabular MDPs have no canonical state geometry or rendered frames. Rasterizing arbitrary state IDs would change under a state permutation while the MDP is identical. A permutation test and a predeclared representation are prerequisites for a tabular **adaptation**; the published image predictor is not directly executable on all E1 families. |
| Asadulaev et al. ICLR 2026 BWD | Behavior critic `Q_beta`, random-action reference and state-conditional entropic optimal transport with cost `Q_beta(s,a')-||a'-a||²` | E1 has no meaningful Euclidean geometry on arbitrary action IDs. One-hot actions provide a permutation-invariant *tabular adaptation*, but the critic's Q values for unseen actions are not identified by the pilot without an extrapolation assumption. Any such assumption and entropic weight must be frozen and sensitivity-reported before a numerical claim; the original continuous-action BWD is not exactly reproducible. |
| XGBoost-AFT | Censored regression for the same E1 requirement label | Existing `artifacts/difficulty_e1_posthoc_2026-09-30/` and `artifacts/e1_dataset_signals_posthoc_2026-09-30/` contain the 120-MDP grouped-fold AFT transfer comparison with raw rows, source hashes, and bootstrap. The 60-MDP seed-disjoint result is in `artifacts/independent_difficulty_2026-09-30/`. |

## Reporting rule

The common table must separate (1) exact source definitions, (2) declared
tabular adaptations, and (3) privileged full-MDP oracle diagnostics. Never
label an adaptation by a published method's bare name. Preserve undefined
scores, right-censoring, negative contrasts and failures. A claim of beating
the offline dataset-quality literature requires faithful runnable comparators
under matched information access and an independent held-out cohort; the E1
post-hoc table alone cannot establish that claim.

Primary sources: [EH 2023](https://papers.nips.cc/paper_files/paper/2023/file/b8be628bf719550b560de8bec9456e0b-Paper-Conference.pdf),
[stochastic EH 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf),
[TQ/SACo](https://proceedings.mlr.press/v199/schweighofer22a/schweighofer22a.pdf),
[ERI/EAS/COI](https://arxiv.org/pdf/2111.13461),
[image predictor](https://openreview.net/pdf?id=AKU4h6BPG7),
[BWD](https://arxiv.org/pdf/2510.12638), and
[AFT](https://arxiv.org/abs/2006.04920).
