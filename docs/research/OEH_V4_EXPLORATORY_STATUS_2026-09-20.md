# LC-OEH-v4 exploratory status

Date: 20 September 2026. This document records an exploratory redesign on the
already-consumed OEH-v3 development cohort. It is not a frozen confirmatory
protocol, and no additional development dataset will be generated.

## Problem addressed

OEH-v3 removed the old constant-floor and zero-identification defects, but its
four variables describe a generic support-restricted QVI policy. The prediction
labels are learner-specific (CFQI, tabular IQL, and behavior cloning), and the
decision endpoint is mean return at a fixed budget. This creates two mismatches:

1. generic QVI risk is not the same object as learner-specific estimation error;
2. probability of crossing normalized return 0.95 is not the same utility as
   mean normalized return at budget `10|S||A|`.

LC-OEH-v4 therefore trains each actual candidate learner inside the pilot and
measures its held-out behavior without reading any independent training stream,
learner outcome, `N_emp`, or oracle return.

## Learner-conditional construction

For learner `l`, pilot prefix `N_j` in the registered set `{2,5,10}|S||A|`, and
held-out fold `f`, fit the learner on the other pilot trajectories to obtain
`pi(l,j,-f)`. Build an empirical model from fold `f`. An action absent from that
fold receives zero reward and zero continuation; the evaluator never
renormalizes unsupported mass away.

Let `V(l,j,f)` be the value of this policy in the held-out model, and let
`V_min(j,f)` and `V_max(j,f)` be the worst and best values attainable using
actions supported by that fold. The normalized supported score is

```text
u(l,j,f) = clip((V(l,j,f) - V_min(j,f)) /
                (V_max(j,f) - V_min(j,f)), 0, 1).
```

If the denominator is zero, the score is 0.5 because the fold cannot rank
policies on its observed support. Across folds,

```text
u_bar(l,j) = mean_f u(l,j,f)
r(l,j)     = 1 - u_bar(l,j)
LCB(l,j)   = clip(u_bar(l,j) - z_0.95 SE_f[u(l,j,f)], 0, 1).
```

The one-sided constant `z_0.95=1.6448536` follows the declared 95% confidence
level. It is not selected from outcomes. The observed contraction exponent is

```text
alpha_l = clip(median_{j<k}
    {-log[r(l,k)/r(l,j)] / log[N_k/N_j]}, 0, 1).
```

The lower limit zero represents no observed contraction. The upper limit one
allows a deterministic `1/N` fast rate while excluding unsupported
super-linear extrapolation. For target risk `epsilon=0.05`,

```text
N_hat_l = N_last [r(l,last)/epsilon]^(1/alpha_l)
```

when `alpha_l>0`; otherwise the requirement is right-censored. Computation is
performed on the log scale, so a small positive rate cannot overflow. The
feature stores `log(1+N_hat_l)` and a censor flag separately.

Trajectory folds use at most five independent trajectories. When a small prefix
contains only one long trajectory, two contiguous time blocks provide a weaker
held-out diagnostic and the feature
`trajectory_independence_fraction=1/5` exposes that loss of independence. This
case is not silently treated as five independent folds.

## Why each variable is present

| Variable | Meaning |
|---|---|
| `log1p_requirement`, censor flag | data scale implied by the observed risk contraction, represented without an arbitrary cap |
| `risk_at_pilot` | unresolved learner-specific error at the current pilot budget |
| `contraction_rate` | whether more pilot data actually reduces that risk |
| cross-fit point estimate and standard error | held-out policy quality and its sampling instability |
| pessimistic return | the registered one-sided lower score used by pessimistic model-selection baselines |
| validation support mass | how much selected-policy mass the held-out fold can evaluate |
| trajectory-independence fraction | evidence quality when the prefix contains too few independent episodes |

The strongest exploratory block combines OEH-v3's generic planning-risk
summaries with the four LC-PMS variables: pessimistic return, its standard
error, validation support mass, and trajectory-independence fraction. This
combination tests whether global environment difficulty and learner-conditional
validation uncertainty are complementary.

## Exhaustive candidate screen on the existing cohort

The estimator completed all 300 MDP-policy groups and 900 learner rows without
error. Six base blocks, three hybrid blocks, direct point-value selection, and
direct pessimistic selection were compared with nested MDP-grouped CV. Scaling
and ridge selection occurred only inside training folds. The return ridge grid
was extended from `{1e-4,...,1}` to `{1e-4,...,100}` after every initial inner
fit selected the old upper boundary; the extension follows a decimal grid and
the scale of standardized `X'X`, rather than choosing a favorable coefficient.

The hybrid `B2 + OEH3 + LC-PMS` has the strongest predictive screen:

- pooled log-likelihood improvement over B2: `+0.06215`, 95% cluster interval
  `[+0.01817,+0.11513]`, Holm-adjusted `p=0.0352`;
- integrated-Brier improvement: `-0.00975`, interval
  `[-0.01596,-0.00350]`, Holm-adjusted `p=0.0208`.

At budget 10, its nested return selector achieves `0.69178` versus `0.68594`
for training-fold best-fixed, a difference of `+0.00584` with interval
`[+0.00001,+0.01227]`. This selector comparison does not survive correction
across all screened methods (`Holm p=0.3486`). OEH-v3 alone has the largest
screened gain, `+0.00908 [ +0.00222,+0.01739 ]`, but also misses the familywise
threshold (`Holm p=0.0540`). Direct cross-fit point selection and direct
pessimistic selection are worse than best-fixed by `-0.0444` and `-0.0820`.

## Final decision-aligned selector

The initial selectors above estimated each learner independently. A paired
advantage model was also tested; its point rule gained only `+0.00182` with an
interval crossing zero, while a training-only conservative rule abstained on
every switch. An inner-OOF threshold was unstable across folds and lost
`0.00165`. These candidates remain in the multiplicity family.

The final candidate averages the nested-OOF return predictions from
`B2 + OEH3` and `B2 + OEH3 + full LC-OEH4` with fixed weights `(0.5, 0.5)`.
This equal ensemble has no fitted mixing weight or post-hoc margin. It obtains
return `0.69552` versus `0.68594` for training-fold best-fixed: a gain of
`+0.00957`, 95% MDP-cluster interval `[+0.00244,+0.01825]`, raw `p=0.0034`,
and Holm-adjusted `p=0.0442` across the complete 15-selector family. It is the
first decision-aligned candidate to pass the corrected development screen.

## Gate decision and claim boundary

| Gate | Result |
|---|---|
| complete, finite, pilot-only execution | pass |
| avoids zero-identification/common-floor pathology | pass mechanically |
| predictive improvement after screening correction | pass exploratorily for the hybrid block |
| selector beats best-fixed after complete screening correction | pass exploratorily for OEH-v4E |
| independent confirmatory evidence | unavailable: cohort was reused for design |
| SOTA claim | prohibited: external methods are a compatibility panel with unequal tuning budgets |

The result identifies a defensible formula, a corrected predictive signal, and
a positive corrected development selector. It completes method development but
does not supply independent confirmation because the cohort informed the
redesign. OEH-v3 remains the frozen primary result and the 120-MDP evaluation
split remains sealed. Thesis and paper text may report OEH-v4E as positive
adaptive development evidence, but not as a confirmatory or SOTA result.

## Reproducible artifacts

- `artifacts/research/lcoeh_v4_exploratory_2026-09-20/`
- `artifacts/research/lcoeh_v4_exploratory_final_2026-09-20/`
- `code/lcoeh_v4.py`
- `code/return_selection.py`
- `scripts/analysis/run_lcoeh_v4.py`
- `scripts/analysis/screen_lcoeh_v4.py`
- `scripts/analysis/run_paired_advantage_selector.py`
- `scripts/analysis/run_oeh_ensemble_selector.py`
- `artifacts/research/oeh_paired_advantage_selector_v2_2026-09-20/`
- `artifacts/research/oeh_equal_ensemble_selector_2026-09-20/`
- `docs/research/OEH_V4E_FINAL_CANDIDATE_2026-09-20.md`
