# III. Theoretical Foundation and OEH Formulation

> **Historical development draft.** Current terms and seventh-cohort claim scope are in [T1 v1.1.0](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and the [claim map](../week8/T1_CLAIM_MAP_2026-10-05.md). This chapter awaits P1 integration; keep its dated results separate from the final cohort.

## 3.1 Operational target

For learner `l`, target normalized return `tau`, and training budget `N`, define

`Y_(l,N) = 1{ NR(pi_hat_(l,N)) >= tau }`.

For the current frozen procedure, the requirement model predicts the distribution
of the empirical **full-suffix** requirement (8/10 seeds at every remaining grid
point), not the probability of success of one seed `Y_(l,N)`. The historical v3
selector used probability of reaching that requirement. V4E instead chooses the
learner with greatest predicted mean normalized return at x10, averaging two
return regressors with weights (.5,.5). Requirement prediction and selection
utility are separate questions.

This definition distinguishes the empirical learner outcome from an environment
difficulty parameter. The same pilot can imply different success probabilities
for learners with different inductive biases and optimization procedures.

## 3.2 Why the registered OEH-v1 loss saturates

At planning depth `k` and state `s`, let `Delta_k(s)` be the estimated gap between
the top two supported actions, `sigma_k(s)` its estimated dispersion, and
`w_k(s)` a nonnegative occupancy weight. The registered loss at its minimum
multiplier was

`L_k(1) = sum_s w_k(s) Delta_k(s) Phi(-Delta_k(s)/sigma_k(s))`.

Set `z = Delta/sigma`. Then

`L_k(1) = sum_s w_k(s) sigma_k(s) g(z_k(s))`,

where `g(z) = z Phi(-z)`. Because `sup_(z>=0) g(z) = 0.169971207479904...`,

`L_k(1) <= 0.169971207479904... sum_s w_k(s) sigma_k(s)`.

Whenever this ceiling lies below the fixed tolerance at every depth, every
depth accepts the minimum multiplier and the selected depth is determined by
the tie rule. Increasing data can make the loss smaller by reducing dispersion,
even when the action ranking becomes more reliable. The saturation therefore
comes from the scale of the functional, not from a proof that the decision
requires no data.

The argument extends to nonnegative terms of the form
`psi = sigma g(Delta/sigma; mu)` when `g(.;1)` is bounded and the aggregate is
positively homogeneous and monotone. An asymptotic saturation claim additionally
requires the aggregated dispersion to vanish. The result does not exclude every
data-only statistic; degree-zero functions of standardized gaps fall outside
this obstruction.

## 3.3 Ideal identification target

Let `C_k` be the action comparisons whose reversal could prevent the target
return at depth `k`, and let `Alt_k(tau)` be alternative MDPs that reverse enough
of those decisions to violate the target. Under behavior occupancy `d_beta`, the
information for separating a nominal MDP `M` from an alternative `M'` is

`I_k(M,M';beta) = sum_(s,a) d_beta(s,a) KL(P_M(.|s,a) || P_M'(.|s,a))`,

where the conditional law includes reward and next state. An ideal requirement
has the form

`N_k^ID(delta,tau) = log(1/delta) / inf_(M' in Alt_k(tau)) I_k(M,M';beta)`.

If an alternative can change a critical action at zero information cost, the
denominator is zero and the full-MDP requirement is infinite. OEH-v3 does not
solve this optimization. It instead changes the estimand to the action support
observed in the pilot and builds a local forecasting proxy within that support.

## 3.4 Supported empirical planning sequence

From the pilot, let `n(s,a)` be the transition count and define
`A_D(s) = {a : n(s,a) > 0}`. The empirical transition and reward model is formed
only on these actions. The empirical behavior policy is renormalized on
`A_D(s)`, and its value provides the leaf. Repeated supported Bellman-optimality
backups produce `Q_k` and policy `pi_k` for depths `k = 0,...,K_max`.

The target is explicitly support constrained. If `|A_D(s)| = 1`, the supported
decision at `s` has no competing action and needs no ranking evidence. This is a
zero identification cost within `A_D(s)`; it is not evidence that an unobserved
action is worse.

Planning regret is evaluated against the best policy in the same empirical
support. Let `Range_D` be the return range between the support-restricted best
and worst policies. With discounted occupancy `d_pi_k`, define

`r_plan(k) = [J_D^* - J_D(pi_k)] / Range_D`.

Ties are handled by the worst greedy policy at that depth, so a favorable
tie-break cannot create a false zero regret.

## 3.5 Return-weighted action-identification risk

For a state with at least two supported actions, let `a*` and `a'` denote the top
and runner-up actions under `Q_k`. The normalized Bellman target is

`Z = (1-gamma)[R/R_max + gamma V_leaf(S')]`,

and the normalized action gap is

`Delta_(s,k) = (1-gamma)[Q_k(s,a*) - Q_k(s,a')]`.

Let `d_(s,a) = n(s,a)/N_pilot`. The variance estimate uses

`v_tilde_(s,a) = max(v_hat_(s,a), 1/[4(n(s,a)+1)])`.

The shrinking floor prevents a single deterministic-looking observation from
asserting zero stochastic uncertainty. Inverting the two empirical-Bernstein
terms gives the pair scale

`tau_(s,k) = max{`

`  2 sum_(a in {a*,a'}) v_tilde_(s,a)/d_(s,a) / Delta_(s,k)^2,`

`  7 sum_(a in {a*,a'}) 1/d_(s,a) / [3 Delta_(s,k)] }`.

At a future total sample size `N`, the plug-in misranking tail is

`p_(s,k)(N) = min{1, 2 exp[-N/tau_(s,k)]}`.

This expression preserves the intended monotonic relations: the risk decays
more slowly when the action gap is small, the Bellman target is variable, or the
behavior policy rarely selects either action.

The return impact of a reversal is motivated by performance-difference scaling,
with empirical occupancy and top/runner gaps replacing true-policy quantities.
This substitution is a proxy, not a proven upper bound on learner regret:

`w_(s,k) = min{1, d_pi_k(s)[Q_k(s,a*)-Q_k(s,a')] /`

`                 [(1-gamma) Range_D]}`.

The depth-specific risk curve is

`R_k(N) = r_plan(k) + sum_s w_(s,k) p_(s,k)(N)`.

OEH-v3 defines

`N_hat_k = inf{N >= 0 : R_k(N) <= epsilon}`.

The inversion is monotone and solved by bisection. If `r_plan(k) > epsilon`, the
depth is `not_epsilon_solvable`. If `N_hat_k = 0`, it is `epsilon_trivial` rather
than a positive finite requirement. Otherwise it is `identified`. The primary
data choice minimizes `N_hat_k`; the effective-horizon reference choice minimizes

`OEH(k) = (k+1) + log_A N_hat_k`.

For `N_hat_k = 0`, the adapter records a zero evidence term instead of taking
`log(0)`. This convention keeps trivial targets separate in status and prevents
an artificial one-sample floor.

## 3.6 Stability and compact predictive features

The requirement uses the complete pilot because it is a plug-in forecast rather
than a cross-fitted certificate. Splitting a small pilot into model and evidence
halves caused OEH-v2 to label a supported rare action as missing whenever it
failed to appear in either half. OEH-v3 instead uses trajectory halves to measure
policy instability. For each half, the planning sequence is reconstructed and
compared with the full-pilot policy using discounted occupancy mass.

The predictive block contains four fixed summaries:

1. `log(1 + min_k N_hat_k)`;
2. the selected QVI index divided by the number of depths;
3. `min(1,R_k(N_pilot))` at the selected depth; and
4. trajectory-half policy instability at that depth.

This compact block replaces the eleven OEH-v2 status and curve summaries, which
were too large relative to the number of independent MDP clusters.

## 3.7 Claims and non-claims

OEH-v3 is a data-only, support-constrained forecasting proxy. It removes the
observed algebraic floor of OEH-v1 and the two-half support loss of OEH-v2. It
does not provide a full-MDP guarantee, an exact KL optimum, or a finite-sample
confidence certificate. Its empirical validity is determined only by the frozen
prediction and selection tests in Chapter IV.

## 3.8 H1 formula freeze and learner-conditional extension

The current formula dictionary and all numerical boundary conventions are in
[H1 v1.0.0](../week6/H1_FORMULA_THEORY_FREEZE_v1.0.0.md). It takes precedence
over abbreviated descriptions here; T1 v1.1.0 controls statistical endpoints.
The implementation uses sample Bellman-target variances without target clipping;
valid nonnegative bounded rewards imply targets in [0,1]. Planning regret uses
the worst greedy policy, but impact occupancy uses the deterministic lowest-index
greedy policy. A top tie may hide inferior third actions, so pair omission does
not establish absence of every action-identification error.

LC-OEH4 cross-fits the three actual learners on pilot prefixes x2,x5,x10.
Held-out unsupported actions retain their policy mass but receive zero reward
and continuation. Fold value is normalized by the held-out supported
best-minus-worst range and clipped to [0,1]; a range <=1e-12 gives .5.
The lower score `clip(mean-1.6448536269514722*SE,0,1)` is nominal, not a 95%
coverage theorem. There are up to five trajectory folds. A lone trajectory is
split into two contiguous time blocks, with independence fraction .2; those
blocks are not independent trajectories.

Risk is `clip(1-mean,1e-8,1)`. The median pairwise log-log risk slope is clipped
to [0,1]. An observed crossing uses the first pilot prefix with risk<=.05;
otherwise a positive exponent extrapolates in log space. At zero exponent,
requirement is missing and the log feature encodes the last pilot size with
censor flag 1. This flag remains 1 for a finite extrapolation with no observed
crossing. These are diagnostic conventions, distinct from label censoring.

The v4E score is `.5*s_B2_OEH3+.5*s_B2_OEH3_LCOEH4`, with 22 and 30 input
features. The hybrid requirement model uses B2+OEH3+LC-PMS (26 features).
The two return models are ridge regressors fitted per learner, with training-only
standardization, nested ridge selection and unpenalized intercepts. Exact ties
select CFQI before IQL before BC. Declared group abstention uses the
outer-training best-fixed learner; fitting errors stop execution. No component
is silently dropped. Equal weighting is a frozen design choice, not an optimality
result; choosing the candidate on consumed development remains adaptive.

## 3.9 Proofs and limitations needed for interpretation

For the registered Gaussian loss, substitute `u=Delta*sqrt(mu)/sigma` to obtain
`L_k(mu)<=G/sqrt(mu)*A_k(w,sigma)` (G as in Section 3.2). Zero dispersion
contributes zero by convention. Since the crossing accepts equality, `m_k>1`
requires `A_k(w,sigma)>epsilon/G`. At equality the minimum multiplier is accepted.
Vanishing aggregated dispersion over a fixed finite depth set implies eventual
simultaneous saturation; it does not prove a rate or learner-performance effect.

For fixed positive pair scales and nonnegative impacts, each exponential risk
term is continuous, nonincreasing and tends to zero. The finite sum therefore
tends to planning regret. If planning regret<epsilon<R(0), a finite positive
crossing exists by continuity. If planning regret>epsilon none exists; at equality
and with any positive impact, no finite crossing exists. This proves properties
of the fixed proxy only; the numeric search cap and estimated inputs are separate.

A support counterexample is a one-step two-action bandit where the logger only
observes reward .5 for the first action. An unobserved second action can reward
1 or 0 in two indistinguishable MDPs. A rule choosing it with probability p has
regrets `(1-p)/2` and `p/2`, whose maximum is at least 1/4. Thus zero requirement
on observed support cannot certify full-MDP optimality without additional
assumptions. The paper appendix gives the same proof in LaTeX.
