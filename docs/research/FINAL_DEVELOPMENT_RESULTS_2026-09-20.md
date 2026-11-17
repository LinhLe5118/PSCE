# Final development results: OEH-v3 and OEH-v4E

Date: 20 September 2026. The 120-MDP evaluation split was not accessed.

## Frozen evidence boundary

OEH-v3, its four-variable feature block, the fresh-cohort generator, grouped
folds, and the primary pooled log-likelihood endpoint were frozen in commit
`e0eb32ca8a2812625affae266f59b3d5b3e78de6`. The generator then created 60
new development MDPs (family counts 16/20/16/8), five behavior policies per
MDP, disjoint pilot/training seed namespaces, and no evaluation-ID overlap.

## Mechanical result

The pilot-only run completed 900 cases and 90,900 depth rows without error. It
records `outcome_labels_accessed=false`, `training_buffers_accessed=false`, and
`sealed_evaluation_accessed=false`.

At the primary x10 pilot, all 300 groups have a finite selected profile: 240
positive requirements and 60 `epsilon_trivial` cases. The positive values
contain 238 distinct values and range from 9,081.4936 to 1,032,950,163.3561.
OEH-v3 therefore removes the observed OEH-v1 common-floor and OEH-v2
zero-identification defects. This result establishes mechanical variation, not
predictive validity.

## Learner grid

The independent local grid contains 54,000 valid fits: 18,000 each for CFQI,
tabular IQL, and tabular BC. There are no invalid runs. The numbers of observed
finite empirical requirements are 144, 55, and 28 of 300, respectively; all
other labels are retained as right-censored.

## Frozen predictive gate

The nested MDP-grouped analysis contains 3,600 out-of-fold predictions. Pooled
B2 minus B1 improves mean log likelihood by 0.05267 with 95% MDP-cluster
interval [0.01841, 0.08449], and integrated Brier by -0.01012
[-0.01591, -0.00462].

The primary pooled OEH contrast, B3 minus B2 log likelihood, is 0.03697 with
interval [-0.00853, 0.08158] (`p=0.1118`), so the primary gate fails. The
secondary integrated-Brier contrast is -0.00732
[-0.01360, -0.00132] (`p=0.0152`) and favors B3. It remains secondary.

## Learner-selection gate

At budget `10|S||A|`, B3 selector mean return is 0.64397 versus 0.68594 for the
training-fold best-fixed learner. The paired difference is -0.04197 with 95%
interval [-0.06614, -0.02109]. All B0--B3 selectors underperform best-fixed.

## Frozen OEH-v3 decision

`research_objective_complete=false`. OEH-v3 fixes the observed measurement
pathology but does not pass the pre-specified primary predictive or decision
gate. The evaluation split remains sealed. The single-use cohort must not be
used to validate a revised method; a positive OEH-v4 claim needs a new frozen
protocol and a new development cohort.

## External-method context

The historical same-prefix development panel remains the complete compatibility
comparison: CFQI, IQL, BC, Pi_b-SPIBB author core, a documented VI-LCB port,
the discrete BCQ author learner, and d3rlpy CQL. Its common x10/seed-0 mean
returns are 0.6729, 0.5084, 0.3510, 0.4499, 0.2559 (best VI-LCB setting),
0.4929, and 0.4746. Fixed unequal tuning budgets make this a robustness panel,
not a neural SOTA result.

## Primary artifacts

- `artifacts/research/oeh_v3_post_freeze_holdout_2026-09-20/`
- `artifacts/research/oeh_v3_post_freeze_local_methods_2026-09-20/`
- `artifacts/research/oeh_v3_post_freeze_b0_b3_2026-09-20/`

Historical OEH-v1 saturation and positive-control artifacts are retained as
evidence for the algorithm's development. Current numerical tables must be
generated from the three directories above.

## Post-hoc LC-OEH-v4 exploration

After the frozen v3 gates failed, a learner-conditional diagnostic was evaluated
on the same, now-consumed cohort. It cross-fits each actual learner within pilot
trajectories, records held-out supported value, uncertainty and support mass,
and estimates risk contraction across the registered pilot prefixes. The run
completed all 300 groups and 900 learner rows without errors.

The strongest hybrid (`B2 + OEH3 + LC-PMS`) improves pooled log likelihood by
0.06215 [0.01817, 0.11513] and integrated Brier by -0.00975
[-0.01596, -0.00350]; both remain significant after Holm correction across the
screened predictive blocks. Its return-selector gain is only 0.00584
[0.00001, 0.01227] and does not remain significant after correcting across
screened selectors (`Holm p=0.3486`). Direct point-value and pessimistic
selection both underperform best-fixed.

These are exploratory findings because the same cohort informed the redesign.
They do not alter the frozen v3 gate decision or by themselves support a SOTA
claim. Per the final project constraint, no new development dataset was
generated.

## Final OEH-v4E development selector

Two further decision-aligned checks retained the same outer MDP folds. Direct
paired-advantage regression did not improve reliably: its point rule gained
`+0.00182` with an interval crossing zero, the conservative rule made no
switches, and an inner-OOF threshold lost `0.00165`. These failures remain in
the multiple-testing family.

The final OEH-v4E selector takes an equal-weight average of the nested-OOF
return predictions from `B2 + OEH3` and `B2 + OEH3 + full LC-OEH4`. The weights
are fixed at `(0.5, 0.5)` and are not learned from outcome labels. It reaches
mean return `0.695515` versus `0.685941` for best-fixed. The difference is
`+0.009574`, with 95% MDP-cluster interval `[+0.002438,+0.018245]`, raw
`p=0.0034`, and Holm `p=0.0442` across all 15 screened selectors. Thus the
final adaptive development selector passes its corrected decision screen.

This positive result is still adaptive rather than confirmatory: the cohort
informed the redesign, and the sealed evaluation set remains untouched. The
formula, evidence boundary, and reproduction command are documented in
`docs/research/OEH_V4E_FINAL_CANDIDATE_2026-09-20.md`.

## Current project state

The mechanical saturation defect and the development learner-selection gate are
now resolved. OEH-v4E is the fixed final candidate, and no additional
development dataset will be created. The remaining scientific step is a single
independent execution on the already sealed evaluation cohort under the frozen
protocol. Until that run succeeds, the paper must describe the selector result
as adaptive development evidence rather than a confirmatory or SOTA result.
