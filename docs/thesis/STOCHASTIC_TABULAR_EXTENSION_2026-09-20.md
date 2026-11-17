# Thesis extension: stochastic tabular offline data

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

## Research position

The thesis should study stochastic tabular **data**, rather than treating
stochasticity as a single scalar property of an environment. The available
evidence supports three separate axes:

- transition stochasticity of `P(S'|S,A)`;
- behavior quality and coverage of decision-relevant actions;
- finite-sample uncertainty in trajectories and empirical transition cells.

The central extension question is:

> When topology and reward are matched, how do transition stochasticity,
> behavior quality, and data budget separately change measurable dataset
> properties, learner return, and the reliability of pilot diagnostics?

## Evidence already available

1. The 810-cell corridor factorial varies slip, behavior decoy probability and
   budget within six matched topology--reward blocks.
2. The 60-MDP post-freeze cohort contains five behavior policies, independent
   pilot/training streams, 54,000 learner fits and right-censored requirements.
3. Historical OEH saturation, bottleneck and positive-control artifacts expose
   measurement failures that cannot be inferred from learner return alone.
4. LC-OEH-v4 supplies learner-conditional held-out uncertainty from pilot data;
   OEH-v4E combines it with generic OEH return prediction and passes the
   corrected development selector screen.

No new development dataset is required or planned.

## Results that can be claimed

- Slip strongly raises empirical transition stochasticity while its mean return
  effect is modest relative to behavior-quality changes on the matched panel.
- A behavior policy can increase raw observed state--action coverage while
  reducing learner return, so coverage quantity and useful coverage differ.
- Larger budgets improve observed coverage and learner return while also
  changing the detectability of stochastic transitions.
- Diagnostic resolution can improve while learner utility deteriorates;
  identifiability of a statistic and usefulness of a dataset are separate.
- Generic OEH risk and learner-conditional held-out uncertainty improve
  prediction jointly, and their fixed equal return ensemble improves learner
  selection on adaptive development data.

All five are development or exploratory claims. Six matched base blocks do not
support universal causal language.

## Thesis structure

1. Define transition stochasticity, behavior stochasticity/quality, and
   finite-sample uncertainty separately.
2. Present the OEH-v1 saturation theorem as a measurement failure.
3. Present OEH-v3 as the support-constrained mechanical repair and report its
   frozen negative primary gate.
4. Use the matched factorial to show why stochasticity, coverage, and budget
   cannot be collapsed into one scalar.
5. Present LC-OEH-v4 as the learner-conditional explanation of the v3 mismatch
   and OEH-v4E as the positive adaptive development selector.
6. Conclude with a multivariate diagnostic view and an explicit claim boundary.

## Prohibited interpretations

- Transition entropy is not a universal difficulty score.
- Higher observed coverage is not automatically higher-quality coverage.
- A resolved or finite OEH value is not proof that a learner will perform well.
- The LC-OEH screen is not independent confirmation or a SOTA result.
- The sealed 120-MDP evaluation split is independent confirmation and must not
  be used to tune or revise OEH-v4E after it is opened.
