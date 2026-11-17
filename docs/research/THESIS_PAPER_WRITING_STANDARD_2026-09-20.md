# Writing standard for the thesis and paper

This document records the editorial standard used for the 20 September 2026
rewrite. It is an internal writing contract, not evidence that a venue will
accept the work.

## Source guidance

- [IEEE Author Center: Structure Your Paper](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/): use a specific title, a self-contained abstract, an introduction ending in the exact question, reproducible methods, and results whose limitations are not exaggerated.
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist): match claims to theory and experiments, state assumptions and limitations, define uncertainty, report compute, and provide a reproducibility path.
- [MIT EECS Communication Lab: Introduction](https://mitcommlab.mit.edu/eecs/commkit/journal-article-introduction/): build the introduction backward from the contribution to the problem and motivation.
- [MIT EECS Communication Lab: Results](https://mitcommlab.mit.edu/eecs/commkit/journal-article-results/): organize results by findings, give rationale and quantitative evidence, and reserve speculation for the discussion.
- [MIT EECS Communication Lab: Planning a Paper](https://mitcommlab.mit.edu/eecs/commkit/journal-or-conference-paper/): every section must answer motivation, question, solution, evidence, and take-away.
- [MIT thesis preparation specifications](https://libraries.mit.edu/distinctive-collections/thesis-specs/): treat the thesis as the durable record of original research; keep the abstract descriptive and distinguish the student's contribution in collaborative work.

## Editorial decisions

1. The paper has one main question: can a pilot-only, support-constrained risk
   profile add predictive and decision value beyond simpler dataset features?
   The saturation theorem is the diagnosis that motivates the estimator, not a
   second competing paper.
2. The narrative order is problem, gap, failed construction, derived repair,
   leakage-resistant test, result, and scope. Chronological debugging history
   moves to the appendix or research log.
3. Every contribution is paired with evidence. Words such as *novel*, *robust*,
   *general*, *effective*, and *state of the art* are removed unless a stated
   comparison supports them.
4. Results state the question, experiment, principal estimate, uncertainty, and
   immediate conclusion. Interpretation and proposed mechanisms belong in the
   discussion.
5. Development, fresh post-freeze holdout, controlled factorial, and sealed
   evaluation are named consistently. A development result is never called
   held-out merely because it was out of fold.
6. The support-constrained estimand is stated wherever OEH-v3 is introduced.
   Zero observed support cannot imply full-MDP optimality.
7. The exact KL target, empirical-Bernstein proxy, learner outcome, and selector
   utility are separate objects. Their symbols and claims are not interchanged.
8. `epsilon_trivial` means the target risk is already below epsilon without an
   action-identification requirement. It is reported separately from positive
   finite requirements and from non-identification.
9. The four-variable OEH-v3 block is described once in Methods. Results report
   its behavior and predictive contrast rather than repeating implementation
   details.
10. The primary endpoint is the frozen pooled B3-B2 log-likelihood contrast on
    the single-use 60-MDP post-freeze cohort. Per-learner contrasts, Brier loss,
    and selector return are distinct secondary or decision endpoints.

## Claim ladder

### Claim supported by mechanism tests

The registered OEH-v1 functional can saturate mechanically under the stated
bounded normalized-loss and vanishing-dispersion conditions. OEH-v3 removes the
observed constant-floor and zero-identification defects on the tested pilot and
factorial panels.

### Claim requiring the fresh cohort

OEH-v3 adds learner-outcome information beyond B2 only if the pre-committed
cluster-bootstrap confidence interval for pooled B3-B2 log likelihood lies
strictly above zero. A useful learner selector additionally requires B3 return
to exceed the train-fold best-fixed learner with a positive interval.

### Claim requiring the sealed evaluation split

Generalization to the registered benchmark distribution requires the method and
analysis to remain unchanged and the evaluation split to be used once. Until
then, the paper may report development evidence but cannot claim confirmed
generalization.

### Claims outside the project evidence

The work does not establish an exact alternative-MDP KL optimum, a finite-sample
certificate, instance-optimality, full-MDP optimality without coverage, neural
SOTA, continuous-action validity, or performance on arbitrary stochastic
tabular environments.

## Revision checklist

- Title and abstract expose the actual estimand and strongest result.
- Introduction ends with questions and contributions that appear again in
  Results in the same order.
- Related work states the precise difference from each closest line of work.
- Methods contain enough detail to reproduce data splits, feature extraction,
  tuning, scoring, uncertainty, compute, and the leakage firewall.
- Tables state direction of better performance, unit of analysis, and interval
  construction.
- Discussion begins with the answer to the research question, then explains
  mechanism, practical implication, limitation, and next valid experiment.
- Thesis chapters contain more pedagogy and derivation than the paper; the paper
  contains only material needed to evaluate its central claim.
- Artifact paths and numeric results are generated from final reports, not
  copied from obsolete runs.
