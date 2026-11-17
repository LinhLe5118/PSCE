# Post-hoc E1 dataset-requirement audit

This folder reports a separate **development** analysis on the original E1
120-MDP cohort. E1 outcomes had already been inspected during OEH development;
these cross-fitted predictions and bootstrap intervals are not independent
confirmation. No new MDPs, buffers, or policy-training runs were created.

The empirical target is the first budget in `x1,x2,x5,x10,x20,x50` whose
entire measured suffix has at least 8/10 training seeds with normalized return
at least 0.95, separately for CFQI, IQL and BC. No such budget is a censored
`>x50` label. The x10 pilot features cannot retrospectively establish
prospective prediction for x1, x2 or x5.
An AFT distribution can extrapolate beyond x50; this audit scores only the
six measured thresholds and does not validate eventual success.

`protocol.json` gives the comparison design. `comparison.csv` contains all
five methods and both panels; `predictions.csv` preserves every out-of-fold
probability and per-instance score, including adverse outcomes. `labels.csv`
contains the derived endpoints. `paired_bootstrap.csv` reports 2,000 whole-MDP
resamples of fixed predictions, with no model refit. `manifest.json` records
input, source and output SHA-256 hashes. The paper table is generated directly
from `comparison.csv`.

XGBoost-AFT is a modern censored-survival comparator, not a proven universal
best model for this custom offline-RL endpoint. Its B2+OEH feature version
improves stochastic Brier over B2-only (0.0765 versus 0.0878); the direct
OEH3 and LCOEH4 requirement proxies score worse than a fold-frequency
reference (0.2087 and 0.1520 versus 0.1134). OEH3 estimates a
support-restricted planning-risk requirement, while this empirical target is
learner-specific sustained success. The control Brier difference between AFT
feature sets is only 0.00018, and control log loss worsens.

The original complete E1 outcomes and pilot feature files are local-only in
some clones. With those files and `requirements-difficulty.txt`, run
`python -m scripts.analysis.compare_difficulty` to regenerate this folder.
The public test suite verifies stored prediction arithmetic and file hashes
without private buffers. The original S1 confirmatory requirement gate failed;
that verdict remains in the immutable week-6 bundle.
