# 4. Method and Evaluation Design

## 4.1. Pipeline Overview

The study has two branches that meet only at scoring (Figure 1.1). On the **forecast branch**, the pilot $D_0$ of an MDP–behavior pair is turned into features, and the fitted forecaster outputs seven probabilities for each learner. On the **label branch**, the ten training streams of the same pair are used to train BC, CFQI and IQL at six budgets; exact evaluation turns the trained policies into success counts and the empirical requirement $E_l$. The two branches share only the identity of the MDP–behavior pair. Forecasts are hashed and sealed before the label branch is run; afterwards, each forecast is joined to its label and scored.

In operational order, OEH performs six steps for one pilot:

1. count transitions, actions and observed rewards in $D_0$ and compute generic pilot statistics (Section 4.2);
2. fit a Bayesian posterior over transition kernels and behavior policies (Section 4.3);
3. simulate the registered learner on logs drawn from the posterior and record the simulated requirement law (OEH-B, Section 4.3);
4. compute planning-risk, coverage and ceiling descriptors on the posterior-mean model (OEH-H, Section 4.4);
5. assemble the 108-column feature vector for each learner (Section 4.5);
6. apply the fitted forecasting head and return seven non-negative probabilities that sum to one.

## 4.2. Generic Dataset Statistics (B2)

B2 is a group-built set of 18 statistics of the pilot (Table 4.1). It describes what a dataset-centered analysis would look at — size, coverage, return of the logged trajectories, transition stochasticity and sampling variability — without modeling the MDP or the learner. Its role in the thesis is the **generic pilot baseline**: the same forecasting head fitted on B2 alone is the main control, and B2 is also the first block of OEH. B2 is not a published method.

Table: Table 4.1. The 18 B2 statistics. Returns are discounted with $\gamma=0.95$; a cell is a state–action pair.
| Group | Statistics | Count |
|---|---|---:|
| Size and coverage | log pilot size; $\log SA$; fraction of observed cells; fraction of observed states; normalized visitation entropy; $\log(1+{}$smallest positive cell count$)$ | 6 |
| Trajectory return | mean, standard deviation and interquartile range of trajectory returns; fraction of trajectories reaching a terminal | 4 |
| Transition stochasticity | count-weighted and mean next-state entropy; fraction of cells with 1–2 visits; fraction of cells (≥2 visits) with several successors; next-state entropy of low- and high-coverage cells | 6 |
| Sampling variability | trajectory-bootstrap (200 replicates) standard deviation of the mean return and of rare-transition exposure | 2 |

## 4.3. Posterior-Predictive Signal (OEH-B)

**Posterior.** For each observed cell $(s,a)$ with next-state counts $n(s,a,\cdot)$, the transition row receives a Dirichlet posterior $\mathrm{Dir}(\alpha_P u + n(s,a,\cdot))$ with a uniform base $u$; the single concentration $\alpha_P$ is fitted by maximizing the Dirichlet-multinomial marginal likelihood of all rows [@minka2000]. The behavior's action distribution in each state receives a Dirichlet-mixture prior whose components share a uniform base and differ in concentration, fitted in the same way. Observed rewards are deterministic per key, and the reward factorization (by next state, by state–action, or by transition) and the terminal states are inferred from the pilot. Reward keys that were never observed receive the value zero. The frozen structure (uniform transition base, behavior mixture, zero unobserved reward) was chosen during development on opened cohorts.

**Simulation.** For each of $B=64$ posterior draws, OEH-B samples a complete MDP and behavior, conditions on a positive dynamic range (draws whose return span is degenerate are rejected and counted), simulates ten training streams of length $x50$ from the drawn behavior, trains the *same* registered learner on the six nested prefixes, evaluates each policy exactly in the drawn MDP, and applies the requirement rule of Eq. (2.6). This produces up to 64 simulated requirement classes $E_l^{(b)}$ per learner. With $B_{\mathrm{eff}}$ returned draws, the smoothed law is the Krichevsky–Trofimov estimate [@krichevsky1981]

$$\hat p^{\,\mathrm B}_l(c)=\frac{\#\{b:E^{(b)}_l=c\}+\tfrac12}{B_{\mathrm{eff}}+\tfrac72},\qquad c=0,\ldots,6.$$ (4.1)

If no draw is returned, 64 censored events are substituted. OEH-B also stores, for each budget, the simulated seed success rate and mean normalized return. The simulation targets the actual empirical event far more directly than an analytic proxy, because it runs the real learner through the real labeling protocol. Its accuracy is limited by the posterior: more draws reduce Monte Carlo noise but cannot correct a misspecified prior or the zero-reward convention.

## 4.4. Planning-Risk Descriptor (OEH-H)

OEH-H asks how many transitions would be needed before a learner that plans on an estimated model ranks the important actions correctly. It is computed once on the posterior-mean model $(\bar P,\bar R,\bar\beta)$. Starting from $Q_1=Q^{\bar\beta}$, it applies Bellman backups restricted to the modeled support $\{a:\bar\beta(a\mid s)>0\}$, with zero continuation at terminal states. At depth $k$ let $\pi_k$ be greedy with respect to $Q_k$, $\Delta_k(s,a)\ge0$ the backed-up gap between $\pi_k(s)$ and a competitor $a$, $L_k(s,a)$ the loss of choosing $a$ under $Q^{\pi_k}$, and $d^{\pi_k}_\rho$ the normalized discounted occupancy. Motivated by the performance-difference identity [@kakade2002], the modeled normalized regret at a budget of $N$ transitions is

$$\tilde\ell_k(N)=\sum_{s,a}\frac{d^{\pi_k}_\rho(s)\,L_k(s,a)}{(1-\gamma)(J^*-J_{\mathrm{random}})}\;\Phi\!\left(\frac{-\Delta_k(s,a)\sqrt N}{\sqrt{\nu_k(s,\pi_k(s))+\nu_k(s,a)}}\right),$$ (4.2)

where $\Phi$ is the standard normal distribution function and the sum runs over supported, non-terminal competitors with positive loss. The variance coefficient $\nu_k(s,a)=\sum_{x,b}W_k[(s,a),(x,b)]^2\sigma_k^2(x,b)/d^{\mathrm{stream}}_\beta(x,b)$ propagates the Bellman-target variance $\sigma_k^2$ of each cell through the lookahead weights $W_k$ and divides by the expected per-episode visitation of the cell under the behavior, $d^{\mathrm{stream}}_\beta$. The descriptor inverts the risk against the model's return slack and takes the best qualifying depth:

$$\tilde N_k=\inf\{N\ge1:\tilde\ell_k(N)\le \mathrm{NR}(\pi_k)-\tau\},\qquad h_{\mathrm H}=\min_{k:\,\mathrm{NR}(\pi_k)\ge\tau}\log_A\tilde N_k.$$ (4.3)

The scan covers depths 1 to 100. The form $k+\log_A N$ of the original effective horizon [@laidlaw2023] is replaced by this risk inversion because, offline, the relevant effort is the number of logged transitions rather than random rollouts. OEH-H reports, besides $h_{\mathrm H}$ and its status (identified, coverage trap, or not reached):

- the **risk profile** $\tilde\ell(N_j)$ and an **exponential missing-mass** estimate — the target occupancy on cells still unobserved after $N_j$ transitions — at each of the six registered budgets, evaluated at the converged support-constrained policy, together with the return slack;
- two **ceilings**: the normalized return of the posterior-mean behavior (what BC approaches with unlimited data) and the best normalized return achievable using only the actions *observed* in the pilot (what support-constrained value iteration approaches);
- **concentrability** summaries from offline theory evaluated on the posterior-mean model: the single-policy concentrability [@rashidinejad2021], its clipped form [@li2024], and an instance-dependent variance bound.

The Gaussian misranking, the omitted covariances between cells and the independence assumption of the missing-mass estimate are approximations. Equations (4.2)–(4.3) are therefore *descriptors* to be validated empirically; they are not finite-sample bounds for BC, CFQI or IQL, and the status labels describe the model, not the true MDP.

## 4.5. Forecasting Model

**Feature set.** The frozen candidate uses 108 columns, which the M1 lineage assigns to eight information blocks (Table 4.2; per-column details in Appendix A). Every column is computed from the pilot (and its $x2$, $x5$ prefixes) and public metadata. Blocks G7 and G8 are diagnostics of earlier OEH versions that were kept because they helped during development: OEH3 is a learner-agnostic action-identification risk, and LCOEH4 trains the actual learner on four fifths of the pilot trajectories at three nested pilot sizes and evaluates it on a model built from the held-out fifth.

Table: Table 4.2. Information blocks of the 108 OEH columns.
| Block | Content | Columns | Compute component | Learner scope |
|---|---|---:|---|---|
| G1 | B2 generic statistics | 18 | pilot statistics | shared |
| G2 | OEH-B CDF, success and return of the row's learner | 18 | posterior simulation | own |
| G3 | Posterior concentrations | 2 | posterior fit | shared |
| G4 | OEH-H value, status, depth, concentrability, ceilings | 11 | posterior-mean planning | shared |
| G5 | OEH-H risk and missing mass at six budgets, slack | 13 | posterior-mean planning | shared |
| G6 | OEH-B CDF of all three learners | 18 | posterior simulation | cross-learner |
| G7 | Legacy OEH3 (4) and LCOEH4 (6) diagnostics | 10 | legacy risk; pilot cross-fit | shared / own |
| G8 | LCOEH4 diagnostics of all three learners | 18 | pilot cross-fit | cross-learner |

**Head.** For each learner and panel separately, three components map features $x$ to class probabilities:

1. **AFT.** An XGBoost accelerated-failure-time model [@barnwal2022; @chen2016] treats the requirement as an interval-censored latent budget, $(b_{c-1},b_c]$ for $c<6$ and $(50,\infty)$ for the censored class, with a normal error of scale 1.2 on the log scale. Its predicted median $m(x)$ gives $F(j)=\Phi\big((\log b_j-\log m(x))/1.2\big)$.
2. **Direct thresholds.** Six binary XGBoost classifiers estimate $\Pr(E_l\le j\mid x)$ for $j=0,\ldots,5$; a cumulative maximum projects the six values to a monotone CDF.
3. **Multiclass.** A seven-class XGBoost model with depth 3, minimum child weight 10 and 200 boosting rounds.

The first two use depth-2 trees, learning rate 0.05 and 200 rounds, and their CDFs are differenced into seven probabilities. The final forecast is a convex combination

$$\hat p_l(x)=\sum_{t=1}^{3}w_t\,\hat p^{(t)}_l(x),\qquad w_t\ge0,\ \sum_t w_t=1,$$ (4.4)

whose weights minimize the out-of-fold seven-class log loss over five MDP-grouped folds of the meta-training rows; the single-component solutions are also candidates, and the best is kept. All components are then refit on all meta-training rows. Every matched comparator in Section 4.7 is fitted by exactly this procedure with its own feature set; weights are learned separately for each row of the comparison and never copied.

## 4.6. Learners and Rationale

Three learners produce the requirement labels; four more enter only the secondary selection study (Table 4.3).

- **BC** imitates the logged actions: it estimates $\hat\beta(a\mid s)$ by counts and acts with it, falling back to uniform at unobserved states. It can only succeed when the behavior is already good.
- **CFQI** is a conservative variant of fitted Q-iteration [@ernst2005]: on the empirical model it performs up to 100 Bellman backups in which the maximization is restricted to actions observed in the state, so unsupported actions are never chosen.
- **IQL** follows implicit Q-learning [@kostrikov2022] in tabular form: an expectile (0.7) value estimate avoids querying unseen actions, and the policy is extracted by advantage weighting with inverse temperature 3.

These three represent imitation, conservative value iteration and advantage-weighted extraction, which respond differently to coverage and behavior quality, while keeping the 108,000-job label grid feasible. The four selection learners add batch-constrained deep Q-learning (BCQ) [@fujimoto2019; @fujimoto2019discrete], conservative Q-learning (CQL) [@kumar2020; @seno2022], safe policy improvement with baseline bootstrapping (SPIBB) [@laroche2019] and value iteration with lower confidence bounds (VI-LCB) [@li2024].

Table: Table 4.3. Learner implementations and information access. All policies are scored by the same exact evaluator.
| Learner | Implementation | Configuration | Extra access | Role |
|---|---|---|---|---|
| BC | project tabular | action counts, uniform fallback | none | labels, selection |
| CFQI | project tabular | ≤100 backups, observed-action support | none | labels, selection |
| IQL | project tabular | expectile 0.7, inverse temperature 3, ≤100 iterations | none | labels, selection |
| BCQ | author code (discrete), one-hot states | 256×256 network, 500 updates, threshold 0.3 | none | selection |
| CQL | d3rlpy DiscreteCQL, pinned | 256×256 network, 500 updates, $\alpha=1$ | none | selection |
| SPIBB | author $\Pi_b$-SPIBB | $N_\wedge=10$ | **true behavior policy** | selection |
| VI-LCB | project port of the paper specification | $c_b=0.05$, $\delta=0.05$, 100 iterations | terminal metadata | selection |

Equal data budgets do not mean equal optimization budgets or equal information. In particular, SPIBB's access to the true behavior and the selector's access to family labels (Section 4.7) must be kept in mind when reading the selection results.

## 4.7. Comparators

Forecasters differ in two ways that affect the meaning of a comparison: the *head* that turns inputs into probabilities, and the *information* they can access. Table 4.4 lists the registered rows.

Table: Table 4.4. Forecast comparators on the sealed test cohort.
| Row | Inputs | Head | Information tier |
|---|---|---|---|
| B2 matched | 18 B2 statistics | matched three-component | Tier 1, pilot only |
| B2 + ERI adaptation | B2, ERI score and eligibility | matched | Tier 1, pilot only |
| B2 + BWD adaptation | B2, tabular BWD | matched | Tier 1, pilot only |
| B2 generic; coverage, quality and coverage+quality subsets | B2 or subsets | AFT only | Tier 1, different head |
| Train frequency | none (class frequencies of meta-training) | none | Tier 1, no features |
| B2 + oracle EH | B2 and effective-horizon descriptors of the true MDP | matched | Tier 2, privileged |
| B2 + TQ/SACo | B2 and author-code TQ/SACo with reference datasets from the true MDP | matched | Tier 2, privileged |
| Published metrics only | TQ/SACo, ERI, BWD | matched | Tier 2, privileged |
| Raw OEH-B | Eq. (4.1) without a head | none | reported, not gated |
| **OEH** | 108 columns | matched | pilot only |

The **matched** rows answer the central question of the thesis: holding the head, meta-training rows and tuning fixed, which inputs forecast better? ERI and BWD are *adaptations*: the published scores are used as input features of our head, not as end-to-end reproductions of their papers, and BWD's continuous-action geometry is not preserved by our one-hot tabular version. **Privileged** rows use information that a pilot does not provide and serve as stress tests. Rows with other heads separate the contribution of the head from that of the features.

For the selection study, the registered comparator is the best fixed learner of the protocol, fixed CFQI. An earlier internal router (v5), uniform random choice and an outcome oracle that picks the best learner per group in hindsight are reported. The OEH selector fits two ExtraTrees regressors [@geurts2006] (150 trees, minimum leaf 7, maximum feature fraction 0.8) on the opened fourth cohort to predict each learner's success and return at $x10$ from CFQI pilot features, CFQI's OEH probabilities and a one-hot MDP-family label; it maximizes predicted success plus 0.15 times predicted return and leaves CFQI only when the advantage exceeds 0.1.

## 4.8. Metrics

For a forecast $\hat p_l$ with cumulative values $F_{l,j}=\sum_{c\le j}\hat p_{l,c}$ and observed class $E_l$, the two registered metrics are the integrated Brier score over the six thresholds and the seven-class log loss:

$$\mathrm{IBS}=\frac16\sum_{j=0}^{5}\big(F_{l,j}-\mathbf 1\{E_l\le j\}\big)^2,\qquad \mathrm{LL}=-\log\max\big(\hat p_{l,E_l},10^{-12}\big).$$ (4.5)

Both are proper scoring rules [@gneiting2007]: their expectation is minimized by the true distribution. Lower is better for both. IBS rewards calibrated cumulative probabilities at every budget and is the discrete analogue of the integrated Brier score of survival analysis [@graf1999]; LL penalizes confident mistakes heavily. A censored row has $\mathbf 1\{E_l\le j\}=0$ at all six thresholds and contributes $-\log\hat p_{l,6}$ to the log loss, so censored observations are scored without imputing a finite requirement. Scores are first averaged over the 15 rows of an MDP and then over MDPs.

For selection at $x10$, each rule chooses one learner per MDP–behavior group. The **mean normalized return** averages NR of the chosen learner's ten seeds, and the **seed success** is the fraction of those seeds with $\mathrm{NR}\ge0.95$; it differs from the sustained 8/10 label. Higher is better for both.

## 4.9. Experimental Setup and Statistical Inference

The independent unit of inference is the MDP: the five behaviors and three learners of one MDP share its kernel, and the ten seeds share its training distribution, so the 108,000 jobs are not 108,000 independent observations. For a comparator $c$ and metric $m$, the paired difference per MDP is $\delta_i=m_{c,i}-m_{\mathrm{OEH},i}$ (positive favors OEH). Uncertainty is estimated by a cluster bootstrap [@efron1993] that resamples whole MDPs, and simultaneous one-sided 95% lower bounds over the registered family of contrasts use a max-$t$ correction [@westfall1993]. The seventh-cohort protocol registered 22 contrasts (11 comparators × 2 metrics); its critical value is 2.53. A contrast passes when its mean difference is positive and its simultaneous lower bound is strictly positive; there is no minimum effect size. Tier 1, Tier 2 and the combined gate pass only if all their contrasts pass. The selection gate registered two contrasts against fixed CFQI (return and success; critical value 1.95).

The sealed cohort is the fifth new 120-MDP test of the forecasting question. Earlier tests are reported in Appendix D with their own protocols and verdicts: the first three failed their gates, and the fourth passed Tier 1 but failed Tier 2, the combined gate and selection. The fourth and the present test use the same frozen OEH configuration; the present test adds the opened fourth cohort to meta-training. Each gate controls error within its own cohort only; repeated testing of the research question is disclosed rather than corrected [@cawley2010; @dwork2015].

## 4.10. Ablation Design

The ablation asks which information blocks carry the gain of OEH over B2 under the same head. Following the M1 lineage, five arms were fixed before the run, each defined by the blocks it removes, never by outcomes:

Table: Table 4.5. Ablation arms (column order of the sealed candidate is preserved).
| Arm | Removed blocks | Columns | Question |
|---|---|---:|---|
| B2 matched | G2–G8 | 18 | generic control |
| Full OEH | none | 108 | reference |
| Without direct OEH-B | G2, G3, G6 | 70 | value of learner simulation summaries |
| Without OEH-H/risk | G4, G5 | 84 | value of the planning descriptors |
| Without legacy | G7, G8 | 80 | value of earlier diagnostics and their context |

Removing direct OEH-B does not remove all posterior information, because OEH-H is computed on the posterior mean; it does remove the learner simulation, which dominates the cost. Each arm is refit with the identical head, folds, meta-training rows and tuning rule; this is a retrained leave-blocks-out comparison, not a masking of a fitted model. The full and B2 arms must reproduce the sealed predictions, which verifies the shared procedure. Predictions are sealed with hashes of every input and source file before the labels are read. Because the test cohort was opened on 4 October 2026, the ablation is **exploratory**: its intervals describe this cohort and cannot confirm a component claim. A confirmatory component test requires an unopened cohort and a pre-registered contrast family.
