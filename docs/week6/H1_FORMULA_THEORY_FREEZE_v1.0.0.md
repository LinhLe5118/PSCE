# H1 — OEH-v4E Formula & Theory Freeze

Owner: Đỗ Trọng Hưng. Version **1.0.0**, 22 September 2026.
Implementation baseline: `8861511`; statistical authority: **T1 v1.1.0**.

This is the formula/theory handover for independent B2 review. It records the
existing algorithm, not a redesigned candidate. No scientific source pinned by
T1, dataset, historical result, feature, weight, endpoint or hyperparameter is
changed. An authorized T1 packaging repair changes only verifier/pointer metadata;
see [integrity revision r1](../protocol/WEEK6_T1_INTEGRITY_REPAIR_r1.md).
H1 integrity PASS does **not** grant E1 execution GO or B2 approval. The checksum
package must be committed/archived before E1; a checksum alone is not a timestamp.

## 1. Authority, outputs and estimands

- Statistical design: `docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json`
  and its Markdown companion; historical v3 gates remain historical.
- Formula specification: this document and
  `docs/protocol/WEEK6_H1_FORMULA_v1.0.0.json`.
- Integrity: `docs/protocol/WEEK6_H1_FREEZE_v1.0.0.sha256.json` and its
  `.sha256` sidecar, verified by
  `python -m scripts.validation.verify_week6_h1_freeze`.
- Theory in the paper: conditional ceiling in Section 4; implementation details
  and scope propositions in the appendix. Thesis Chapters III–IV are synchronized.

There are **three different targets**, with different normalizations:

1. **OEH3 proxy:** invert a pilot-model risk curve within observed action support;
   normalize planning regret by empirical best-minus-worst supported return.
   Its output is a real-valued future **total** transition count, not extra
   transitions, a learner guarantee or an exact KL sample complexity.
2. **Learner requirement label:** first budget whose entire remaining grid has
   at least 8/10 runs with true-MDP normalized return at least .95. Normalization
   is `(J_policy-J_random)/(J_star-J_random)`, without clipping. No successful
   suffix means right censoring after x50, not a label of 50 or infinity.
3. **V4E decision:** maximize predicted mean normalized learner return at x10.
   The requirement model is a separate survival model; its probability is not
   the final v4E selection score. Neither proxy epsilon=.05 nor LC risk=.05 is
   an algebraic equivalence to label NR=.95.

The feature extractor receives pilot arrays, public dimensions, initial-state
distribution, gamma and reward bound. Outcomes can enter **meta-training folds**
only after features are sealed; outer-test outcomes cannot select features,
ridge, weights, comparator or fallback. Evaluation confirms a locked nested
meta-learning procedure on a fresh cohort, not a single development-trained model.

## 2. Supported empirical model and indexing

Let N0 be the number of transitions; n(s,a) their counts; A_D(s)={a:n(s,a)>0}
at nonterminal states. Rewards lie in [0,Rmax], with Rmax=1 for this protocol.
The empirical model divides mean rewards by Rmax and uses empirical successor
frequencies. Unobserved rows receive zero reward and a self-loop; observed
terminal destination states are absorbing zero-reward states. Truncation does
not mark a state terminal. Empty action sets use action 0 only as a numerical
zero-value completion; they provide no evidence about unobserved actions.

Normalize the empirical behavior policy over A_D. Evaluate its value V_beta,
then Q_0=r_hat+gamma P_hat V_beta=Q^(beta_hat). For k=0,...,100, choose the
supported greedy policy pi_k and update the leaf by max over supported Q_k.
Recorded `qvi_index=k+1`; k=0 is a behavior-policy leaf, not a reward-only leaf.
For exact greedy ties, `_restricted_policy` chooses the lowest action index.

Let R_D=J_D^star-J_D^minus. For planning regret, use the **worst** policy among
actions within 1e-12 of the maximum at every state:

`b_k=max(0,J_D^star-J_D^worst_greedy)/R_D`.

When R_D<=1e-15, set b_k=0. The occupancy used for pair impacts and instability
is instead that of the deterministic pi_k:

`d_k^T=(1-gamma) rho0^T (I-gamma P_pi_k)^(-1)`.

These two tie conventions have different roles and must not be conflated.

## 3. OEH3 risk, inversion and readout

For each nonterminal state with d_k(s)>0 and at least two supported actions,
sort supported Q values stably and take the last two actions as top and runner.
If their raw gap G_sk<=1e-15, omit that pair. The stable sort uses ascending
action order for equal values; a skipped top tie may still have lower-valued
third actions. Thus the pair approximation is not an all-action error bound.

Define:

`Delta_sk=(1-gamma) G_sk`, `f_sa=n(s,a)/N0`,

`Z_i=(1-gamma)[r_i/Rmax+gamma V_leaf(s'_i)]`,

`vtilde_sa=max(sample_variance(Z_sa),1/[4(n(s,a)+1)])`.

The sample variance uses n-1, returns 0 for n<=1, and clamps negative rounding
error to 0. **There is no target clipping in `_evidence_statistics`.** For valid
nonnegative bounded rewards and the empirical discounted model, Z lies in
[0,1] in exact arithmetic. The variance floor is a design regularizer; it is
not a proven confidence upper bound on variance.

`t_sk=max(float64.tiny, 2 sum_a(vtilde_sa/f_sa)/Delta_sk^2,
                      (7/3) sum_a(1/f_sa)/Delta_sk)` (sum over the pair),

`w_sk=min(1, d_k(s) G_sk/max((1-gamma)R_D,1e-15))`,

`p_sk(N)=min(1,2 exp(-N/t_sk))`,

`R_k(N)=b_k+sum_s w_sk p_sk(N)`.

In floating point the exponential exponent is floored at -745. The sum is not
clipped for inversion; only the displayed `risk_at_pilot=min(1,R_k(N0))` and
`decision_risk_at_zero` are clipped. The weights need not sum to one.
Performance-difference scaling motivates w; replacing true advantages,
occupancies and all action errors by pilot top/runner terms is an approximation,
not an application of that identity yielding a learner-return theorem.

The ideal definition is `Nhat_k=inf{N>=0:R_k(N)<=epsilon}`, epsilon=.05.
The existing numerical conventions are:

| Condition | Stored status and value |
|---|---|
| b_k > epsilon+1e-12 | `not_epsilon_solvable`, None |
| Negative residual budget, or no root at/below 1e18 | `unresolved`, None |
| R_k(0)<=epsilon | `epsilon_trivial`, 0 |
| Positive root | `identified`; doubling from 1 then 100 bisections |
| Root <=1e-12 | numerically `epsilon_trivial`, 0 |

The small interval epsilon<b_k<=epsilon+1e-12 is admissible in the diagnostic
flag but inversion returns unresolved, not a false finite root. None with
`requirement_censored_right` is a profile encoding and need not denote the same
statistical censoring event as N_emp. No comparison is required at single-action
states; missing support is not thereby certified harmless in the real MDP.

Select `(Nhat_k,k+1)` lexicographically among resolved depths. The data selection
does not minimize the EH reference `(k+1)+log_A Nhat_k`. At zero requirement the
adapter stores evidence 0; the historical EH adapter treats negative log evidence
as unavailable, so a 0<Nhat<1 record is outside its well-defined EH-reference
path. That path is not a new fallback: such an exception must stop execution,
and any correction requires versioned review before E1. With the frozen Rmax,
gamma, epsilon and positive comparisons, the registered tested positive profiles
are above one; do not assert that for arbitrary API configurations.

The four OEH3 features, in order, are:

| Feature | Exact readout / role |
|---|---|
| `oeh3_log1p_requirement` | ln(1+Nhat_selected), magnitude |
| `oeh3_selected_qvi_fraction` | (k_selected+1)/101, planning depth |
| `oeh3_risk_at_pilot` | min(1,R_selected(N0)), current proxy risk |
| `oeh3_policy_instability` | mean of two occupancy-weighted disagreements with the full-pilot policy, clipped at 1 |

Instability halves are assigned by sorting SHA256(`20260920:trajectory_id`)
and alternating IDs. With fewer than two trajectories instability is 1.
The full pilot still computes the requirement. No eligible depth uses
`[ln(1+10*reference),1,1,1]`, reference=max(nonmissing requirements, default=1).
This is a finite missingness encoding, not proof of a large actual requirement.
The frozen grid has all 101 depths and no additional compute-budget filter.

## 4. Learner-conditional block (LC-OEH4)

At each nested pilot prefix x2,x5,x10, run CFQI/IQL/BC on training folds of the
pilot and evaluate on the held-out empirical model. Up to five trajectory folds
are used: shuffle unique IDs with the seed derived from the first 8 little-endian
bytes of SHA256(`mdp_id|policy_family|multiplier`), then assign round-robin.
With one trajectory use **two contiguous time blocks**, not two independent
trajectories. At least two transitions are needed; shorter input is an error.

CFQI/IQL use 100 updates, tolerance 1e-10, gamma=.95; IQL expectile=.7 and
inverse temperature=3. BC uses observed action frequencies. Unseen/terminal
states of these learners use uniform policy. These are actual pilot fits;
"pilot-only" does not mean "no learner computation".

Validation model: unsupported action rows have P=0,r=0; terminal transitions
have zero continuation. Policy mass is **not renormalized away** from unsupported
actions. This zero completion is conservative for nonnegative missing rewards,
but estimation error on supported rows prevents a general true-value lower bound.

For fold f and prefix j, compute value V_ljf and supported best/worst V+_jf,V-_jf.
If range>1e-12, `u_ljf=clip((V_ljf-V-_jf)/(V+_jf-V-_jf),0,1)`; otherwise u=.5.
With F_j actual folds:

`u_bar_j=mean_f u_ljf`, `SE_j=sample_std_f(u_ljf)/sqrt(F_j)`,

`lower_j=clip(u_bar_j-1.6448536269514722*SE_j,0,1)`.

This is a **nominal Gaussian lower score**, not a calibrated 95% confidence
bound: fold training sets overlap, time blocks may be dependent, and value
models/normalization are estimated. Folds are equally weighted, not size weighted.

`risk_j=clip(1-u_bar_j,1e-8,1)`,

`alpha=clip(median_{i<j}[-ln(risk_j/risk_i)/ln(N_j/N_i)],0,1)`.

If any risk_j<=.05, use the **first** crossing prefix, even if a later risk rises.
This diagnostic differs deliberately from the full-suffix definition of N_emp.
Otherwise, for alpha>1e-12:

`log_Nhat=ln(N_last)+ln(risk_last/.05)/alpha`.

If alpha<=1e-12, requirement=None and log_Nhat=ln(N_last), a placeholder with
censor flag 1. If exponentiation would overflow float64, requirement=None but
keep the finite log extrapolation. The feature is `logaddexp(0,log_Nhat)`.
The censor flag means **no crossing was observed in the pilot prefixes**; it is
1 even when a finite extrapolated requirement exists.

| Full LC feature | Meaning |
|---|---|
| `lcoeh4_log1p_requirement` | stable log forecast/placeholder above |
| `lcoeh4_requirement_censored` | 1 iff no observed pilot crossing |
| `lcoeh4_risk_at_pilot` | risk at x10 |
| `lcoeh4_contraction_rate` | alpha, constrained descriptive rate |
| `lcoeh4_pessimistic_return` | lower score at x10 |
| `lcoeh4_return_se` | SE at x10, not a coverage guarantee |
| `lcoeh4_validation_support_mass` | mean-fold empirical validation-state-frequency-weighted policy mass on supported actions |
| `lcoeh4_trajectory_independence_fraction` | min_j(min(number of trajectories_j,5))/5; single-trajectory time blocks give .2, not .4 |

LC-PMS is the last four columns only. The independence fraction is a diagnostic
count, not a proof that folds or transitions are statistically independent.

## 5. B0–B2 dictionary and meta-models

All feature lists and their order are frozen in T1 JSON: dimensions B0=2,
B1=10, B2=18, B2+OEH3=22, hybrid requirement block=26, full return block=30.
Logs are natural unless explicitly log_A.

| B0/B1 feature | Definition |
|---|---|
| log pilot transitions / log state-action count | ln(N0) / ln(S*A) |
| observed SA / state fraction | count(n_sa>0)/(S*A) / unique source states divided by S |
| normalized visitation entropy | entropy(n_sa/N0)/ln(S*A), 0 when denominator 0 |
| log1p minimum positive count | ln(1+min positive n_sa) |
| trajectory return mean/std/IQR | sum gamma^t r_t per trajectory; sample std (0 for one trajectory); linear quantile .75 minus .25 |
| terminal trajectory fraction | fraction with at least one terminal flag, not truncated flag |

| B2 feature | Definition |
|---|---|
| weighted transition entropy | sum n_sa/N0 * entropy(P_hat_sa)/ln(S) |
| observed-cell transition entropy | mean normalized entropy over n_sa>0 |
| low-count cell fraction | count(0<n_sa<3)/(S*A), **not divided by observed cells** |
| empirical stochastic cell fraction | fraction with >1 observed successor among n_sa>=2; 0 if none |
| low/high coverage entropy | mean normalized entropy on 0<n_sa<3 / n_sa>=3, 0 if empty |
| bootstrap return mean std | sample std of 200 trajectory-resampled return means |
| bootstrap rare-exposure mean std | sample std of 200 trajectory-resampled means of within-trajectory fraction visiting cells with full-pilot n_sa<=2 |

Entropy is 0 for S=1. Bootstrap features use SHA256(seed_key), first 8 bytes
little-endian, NumPy default_rng; with one trajectory their bootstrap std is 0,
which reflects no resampling variation, not certainty. The development builder
passes dataset_id as the seed key (see pinned caller); preserve this at E1.

For each learner, each return component fits ridge regression minimizing
`sum_i(y_i-b-x_i^T theta)^2+lambda||theta||^2`; b is unpenalized.
Standardize within the training fold using population std; std<=1e-12 becomes 1.
Choose lambda from {1e-4,1e-3,1e-2,.1,1,10,100} by inner MSE; exact ties choose
the larger lambda. Return predictions are not clipped.

The requirement model uses logit h_j=intercept_j+theta^T x and ridge on theta
only, grid {1e-4,1e-3,1e-2,.1,1}, inner log likelihood, ties smaller lambda.
The existing optimizer has at most 50 Newton steps, Hessian diagonal floor 1e-9
and step stopping threshold 1e-10. Event likelihood is survival before the event
times hazard at the event; censoring contributes survival through all six bins.
T1 specifies full-grid Brier and hazard clipping [1e-12,1-1e-12].

Five outer base-MDP folds use integer SHA256(id) modulo 5. Four inner folds sort
unique IDs by salted hashes then assign round-robin. Salt is
`return|outer_fold|algorithm|feature_set` for return and
`outer=outer_fold|algorithm=algorithm|features=feature_set` for survival.
All behaviors of the same base MDP stay together. T1's separate stochastic and
control pipelines apply before **any** fitting, including bootstrap refits.

## 6. Ensemble, ties, fallback and failures

`score(g,l)=.5*prediction_B2_OEH3(g,l)+.5*prediction_B2_OEH3_LCOEH4(g,l)`.

Choose the largest score; exact ties follow CFQI, IQL, BC. Equal weights were
not numerically fitted, but the candidate and components were chosen adaptively;
symmetry is a design rationale, not proof of optimal weights or independence.
No switch margin or confidence threshold is added.

Best-fixed is selected solely by outer-training mean return, averaging seeds,
then behavior, then MDP; ties use the same learner order. If a component cannot
produce finite scores for **all three** learners as a declared diagnostic
abstention, fall back for the **whole group** to this best-fixed. Keep that
group in all denominators, record a reason, never average only the surviving
component. Feature exceptions, missing files, fitting/numerical errors stop
execution; they are not valid abstention. Finite high-risk encodings alone do
not trigger fallback. These are existing T1 requirements.

The historical ensemble runner assumes 300 complete development groups and
does not implement the sealed runner's fallback, panel partition or refit
bootstrap. H1 specifies these obligations without pretending that runner is
ready for E1. D1's baseline preflight also does not discharge them.

## 7. Theory and claim boundary

For the old weighted loss with nonnegative gaps/weights and finite dispersion,
define g(z)=z Phi(-z). Its unique maximizer solves Phi(-z)=z phi(z), giving
G=0.169971207479904... . With sigma=0 use contribution 0, including ties.
For every mu>=1, substituting u=(Delta/sigma)*sqrt(mu) proves

`L_k(mu)<=G/sqrt(mu) * sum_s w_k(s) sigma_k(s)`.

For a monotone, positively homogeneous aggregator the same componentwise bound
gives `G/sqrt(mu)*A_k(w_k,sigma_k)` for the registered Gaussian term. At mu=1,
`G*A<=epsilon` implies m_k=1. Therefore m_k>1 requires **A>epsilon/G**;
the non-strict historical necessary inequality is valid but not sharp at equality.
With a fixed finite depth set and max_k A_k->0, all depths eventually saturate
at m=1, selecting depth 0. This does not prove a dispersion rate, nor that more
data harms a learner. Positive controls refute a universal hard-coded floor,
not prove predictive validity. A1–A3 are design desiderata, not a uniqueness or
learner sample-complexity theorem. A2 refers to the top-pair contribution, not
all possible action errors when there are more than two actions.

For fixed finite positive t_sk and nonnegative w_sk, R_k(N) is continuous and
nonincreasing, with limit b_k. If b_k<epsilon and R_k(0)>epsilon, a finite positive
crossing exists mathematically; if b_k>epsilon none exists. At b_k=epsilon with
any positive weight, no finite crossing exists. These facts justify inversion,
not the correctness of estimated inputs or utility for CFQI/IQL/BC. The numeric
cap 1e18 and exponent floor further restrict the executable domain.

**Support counterexample.** A one-state one-step bandit logs only action a0,
whose reward is 1/2. In MDP M+, unlogged a1 has reward 1; in M-, it has reward 0.
The pilot laws are identical, but the optimal actions differ. Any deterministic
choice incurs regret 1/2 in one MDP. A randomized choice of a1 with probability
p has regrets (1-p)/2 and p/2, whose maximum is at least 1/4. Thus no pilot-only
method can uniformly infer full-MDP epsilon-optimality for epsilon<1/4 in this
class. A zero within-support requirement is compatible with this impossibility.

The survival target is an empirical suffix requirement, not the probability of
one training seed's success. Higher threshold probability need not imply higher
expected return. Example: learner A always returns .94; B returns 1 with
probability .8 and 0 otherwise. B has higher .95-threshold success probability,
but lower mean return (.8 vs .94). This motivates decision alignment; it does
not prove that ridge/ensemble estimates those means accurately.

## 8. Alignment, acceptance and handover

| Topic | Executable reference | Paper / thesis alignment |
|---|---|---|
| v1 ceiling, equality, zero sigma | `oeh_estimator.py`, `positive_control.py` | Section 4 + H1 appendix / Chapter III |
| empirical support/Q indexing/risk | `oeh_risk_v3.py`, `oeh_crossfit.py`, `oeh_final_profile.py` | definitions + H1 appendix / Chapter III |
| exact feature ordering/encodings | `stochastic_features.py`, T1 JSON | H1 appendix / this dictionary |
| LC folds/rates/censor flag | `lcoeh_v4.py`, `empirical_requirement.py` | method + H1 appendix / Chapter III |
| return prediction and ensemble | `return_selection.py`, `run_oeh_ensemble_selector.py` | method / Chapters III–IV |
| current endpoint/fallback/panels | T1 v1.1.0 | prospective protocol appendix / Chapter IV |

Numerical settings are locked choices, not all theoretically optimal constants.
The verifier checks exact hashes and the existing T1 package; fixture tests
check feature order/defaults, rate and support edge cases, ceiling and monotone
inversion. Tests complement analytic proofs; they do not validate all theorem
assumptions on unseen data. A float64 requirement with no reference is an error,
not a license to invent a replacement value.

Before E1, B2 must independently check this package and D1 must demonstrate
full-group fallback, panel separation, complete job schema, reproducible resume
and paired refit-bootstrap. No H1 claim of E1 execution, external validity,
SOTA, compute savings or independent confirmation is made. For B1 handover,
Hưng should check transition randomness versus behavior randomness, sample
coverage versus missing population support, raw trajectory quality versus
oracle-normalized scoring, and MDP-level denominators. An unfinished B1 artifact
cannot be signed off by this package.

Any change to a pinned formula, feature order, threshold, weight, normalization,
tie or fallback requires an explicit new H1 version and B2 review. If a T1-pinned
source changes, Linh must also issue a new T1 manifest/version before E1.
