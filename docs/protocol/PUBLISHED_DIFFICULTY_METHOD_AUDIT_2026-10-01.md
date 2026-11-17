# Published difficulty methods: exact-comparison eligibility audit

This audit responds to the stricter requirement added **after** the second
60-MDP confirmation started: at least 5% superiority to original published
methods before any new analysis of the original E1 120 MDPs. It is not a
retroactive amendment to the already sealed protocol. The second cohort has
completed, but no result has been promoted under the new requirement.

For this project, the measured target is the first budget in
{x1,x2,x5,x10,x20,x50} at which a fixed offline learner achieves normalized
return >=0.95 on at least 8/10 seeds at that and every larger budget, or
right-censoring beyond x50. A fair direct comparator must predict this target
from the same x10 pilot data without a true MDP kernel, held-out learner
outcomes, a later-budget dataset, or an externally supplied reference dataset.

| Published method | Original object and input | Exact same-endpoint/data-only comparison on E1? |
|---|---|---|
| [Stochastic effective horizon (Laidlaw et al., ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf) | Environment-dependent online RL hardness/effective horizon using the MDP and random-policy Q iteration | **No**: full-MDP/online quantity, not a pilot-only learner-specific offline requirement. Exact oracle value can be reported in an oracle tier. |
| [Colosseum hardness (Conserva et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5eeb693f46d753e5fe24c97212c22bd2-Abstract-Conference.html) | Theoretical MDP hardness measures and tabular environment benchmark | **No**: requires MDP knowledge and estimates a different hardness object. Oracle diagnostic only. |
| [TQ/SACo (Schweighofer et al., CoLLAs 2022)](https://proceedings.mlr.press/v199/schweighofer22a/schweighofer22a.pdf) | Dataset trajectory quality and normalized unique state-action coverage relative to random/expert or other matched reference datasets | **No**, absent the same reference datasets at prediction time. Pilot return/coverage features are *adaptations*, not the original normalized measures. Using x50 as reference would leak a later-budget dataset. |
| [ERI/EAS/COI (Swazinna et al., 2021)](https://arxiv.org/pdf/2111.13461) | Dataset-selection ranks: ERI uses `(maximum trajectory return - mean return)/mean return` and assumes positive mean; EAS fits a tanh-Gaussian behavior policy and reads action standard deviation; COI combines their ranks 2:1 | ERI can be computed exactly only on eligible positive-mean datasets. Original EAS/COI cannot be applied to arbitrary discrete action IDs without changing the model, so they are **not exact** same-endpoint comparators. |
| [Image-based Dataset Performance Prediction (Mateos-Melero et al., EWRL 2024)](https://plg.uc3m.es/papers/mateos-melero-et-al-ewrl24.pdf) | CNN on an image representation to predict fixed-framework policy return | **No**: E1 uses arbitrary tabular state IDs without the spatial semantics required by the original image representation; target is return, not censored budget. |
| [Bellman--Wasserstein distance (2025/ICLR 2026)](https://arxiv.org/abs/2510.12638) | Value-aware distance from random behavior using a behavioral critic and Euclidean action cost, evaluated on MuJoCo dataset quality | **No** exact use on arbitrary action IDs: their Euclidean action geometry is undefined. A tabular transport cost would be a new adaptation. |
| [XGBoost AFT (Barnwal et al., 2020)](https://arxiv.org/abs/2006.04920) | General censored-outcome prediction algorithm | **Yes as a statistical comparator** when trained with the same E1 meta-labels and pilot features, but it is not an offline-RL-specific published difficulty estimator. The current B2 and B2+OEH AFT controls use it. |

This is an identified-literature audit, not proof that every relevant paper has
been enumerated. The original methods above cannot all be ranked by the
current seven-category Brier and log-loss without changing their task,
information access, or representation. In particular, one cannot claim that
OEH beats “every published original method” by converting nonmatching
quantities to a seven-category forecast and presenting that conversion as
original. Keep exact oracle quantities, faithful reference-dependent scores,
and data-only tabular adaptations in separate tables.

For a future **new, untouched** confirmation under the user's 5% rule,
predeclare a finite set of eligible original same-target methods (if any are
found), their author implementation/commit and inputs, and a relative-gain
calculation. A reasonable precise margin is at least 5% lower integrated
Brier **and** at least 5% lower seven-category log-loss than the strongest
eligible method, with positive simultaneous MDP-cluster lower bounds and no
hidden failures. The completed second-cohort protocol passed its original
registered gate and exceeds 5% numerically versus its registered data-only
baselines. It cannot retrospectively establish the stronger
published-original claim. A late-added, blinded architecture-matched B2
control reduces the OEH log-loss gain to 4.18%, with a negative exploratory
simultaneous lower bound. See
`artifacts/oeh_difficulty_confirmation_2026-10-01/margin_audit.json` and
`artifacts/oeh_difficulty_matched_b2_confirmation_2026-10-01/summary.json`.

The original E1 120 MDPs were used repeatedly for OEH development and
meta-training. Any subsequent E1 table is descriptive even if a new-cohort
gate succeeds; it cannot be called a new one-shot independent test.
