# 1. Introduction

## 1.1. Background and Motivation

Reinforcement learning (RL) trains a decision rule, or *policy*, by trial and error. In many applications the trial-and-error phase is the problem: exploratory control of a building can waste energy or reduce comfort [@liu2022hvac], and exploratory treatment or dialogue policies can harm users. *Offline* RL removes this phase. A learner receives a fixed log of past interactions, collected by some *behavior policy*, and must output a policy without collecting anything new [@levine2020; @prudencio2024]. This makes the log the only source of information about the environment.

The amount of logged data a learner needs is therefore a central planning quantity. Collecting data and training learners at many data sizes cost time and money, so before committing to a large collection or a full training grid one would like to know: *if I continue to log data in the same way, at what size will this particular learner reliably reach a good policy?* Dataset size alone does not answer the question. Two logs of equal size can be very different. One may contain the actions that matter at the states that matter; another may never try the optimal action in a critical state. Stochastic transitions require repeated observations before their effect can be averaged out. The answer also depends on the learner: an imitation learner cannot exceed the behavior it copies, whereas a value-based learner can improve on a poor behavior if the relevant actions have been logged.

This thesis studies whether a small *pilot* log, collected before the main data, can predict this learner-specific data requirement. The study is set in finite (tabular) stochastic Markov decision processes (MDPs), where every policy can be evaluated exactly and the requirement can be measured precisely. Figure 1.1 shows the two independent paths of the study: the forecast path, which sees only the pilot, and the label path, which trains learners on separate data and measures the requirement that the forecast is scored against.

![Figure 1.1. The forecast path and the label path. Each MDP and behavior policy yields an independent pilot and ten independent training streams. The forecaster sees the pilot and the learner identity; six nested budgets and ten seeds per learner produce the measured, possibly censored requirement label. Forecasts are sealed before labels and joined only for scoring.](figures/oeh_pipeline.png){width=16}

## 1.2. Related Work

Four lines of work address neighboring questions. For each we summarize the problem, its input and output, and the limitation that matters for this thesis.

**Dataset characteristics and quality.** Schweighofer et al. [@schweighofer2022] ask how properties of an offline dataset relate to the performance of offline learners. Their inputs are datasets and reference policies collected in the true environment; their outputs are two scalar descriptors, trajectory quality (TQ) and state–action coverage (SACo), that they relate to learner performance mainly in deterministic environments. Swazinna et al. [@swazinna2021] propose indicators such as the estimated relative return improvement (ERI) to choose promising datasets, and Asadulaev et al. [@asadulaev2026] measure dataset quality with a Bellman–Wasserstein distance (BWD). These scores rank datasets; they do not output a distribution over data budgets, and some require reference data from the true environment, which a pilot-only forecaster does not have.

**Coverage and sample complexity.** Offline RL theory explains when a dataset suffices. Bounds depend on how well the data cover the distribution of a good policy (concentrability) [@rashidinejad2021; @zhan2022], and model-based analyses settle the minimax sample complexity under such coverage assumptions [@li2024]. Instance-dependent results show that the action gaps of a particular MDP can make learning much easier than the worst case [@wang2022]. These results take the MDP, or its coverage constants, as given and output worst-case or instance-dependent bounds up to constants. Arnob et al. [@arnob2021] argue that empirical sample-complexity curves should be measured rather than inferred from such bounds. None of this work predicts, from a pilot, the measured requirement of a specific learner.

**Effective horizon.** Laidlaw et al. introduce the effective horizon (EH), which combines the lookahead depth at which greedy actions become optimal with the random-rollout effort needed to estimate the relevant values, and show that it explains the sample complexity of interactive deep RL on the BRIDGE collection of tabular MDPs [@laidlaw2023]. A later study extends the analysis to stochastic environments [@laidlaw2024]. The input is the true MDP and the output a scalar complexity for *online* learning with random exploration. This thesis borrows the depth-and-gap structure but must replace the true MDP by a model inferred from a pilot and the random exploration by the logging behavior; neither substitution preserves the original guarantees.

**Performance prediction and policy or learner selection.** Mateos-Melero et al. [@mateos2024] predict the return that a specified offline learner with given hyperparameters will reach on a dataset, using an image representation of the data. Offline policy selection ranks already-trained policies using held-out logged data [@yang2022; @tang2021; @paine2020]. These tasks predict or compare performance at one configuration or one data size. Our target instead is the *first* data size at which sustained success begins, including the possibility that it does not begin within the measured range.

## 1.3. Research Gap

The four lines above leave a specific question open: **from a pilot, can one forecast the distribution of the first data budget at which a given learner attains sustained success, and do so better than generic dataset statistics under the same forecasting procedure?** Answering it needs (i) a target that is measurable and learner-specific, including a principled treatment of runs that never succeed; (ii) a forecaster that uses only information available at pilot time; and (iii) a comparison that holds the forecasting procedure fixed, so that any gain is attributable to the information extracted from the pilot. Our scoped literature search, last updated in October 2026, found no published method with exactly this input and output. We present this as a task definition and an evaluated method, not as proof that no related work exists.

## 1.4. Problem and Scope

The title contains four key terms. *Difficulty-aware* means that we treat difficulty as the measured amount of data a learner needs, not as an intrinsic scalar of the environment. *Pilot-only* means that the forecaster sees one logged pilot, public metadata such as the number of states and actions, and a model fitted on earlier cohorts, but never the true MDP or any outcome of the test learners. *Learner-specific* means that the target is defined separately for each registered learner. *Stochastic tabular MDPs* fixes the setting: finite state and action sets with random transitions.

The thesis fixes one concrete situation. An MDP and a behavior policy produce a pilot of $10|S||A|$ transitions (written $x10$). The registered learners are behavior cloning (BC), conservative fitted Q-iteration (CFQI) and tabular implicit Q-learning (IQL). Each is trained at six total data budgets $x1, x2, x5, x10, x20, x50$ (multiples of $|S||A|$) with ten independent seeds. The label is the first budget at which at least eight seeds reach 95% of the optimal normalized return at that and every larger measured budget, or the censored class $>x50$. A forecast is a probability distribution over these seven classes. A secondary scope is the choice among seven learners at the fixed budget $x10$, which tests whether the forecasts are useful for a downstream decision.

## 1.5. Research Questions and Objectives

- **RQ1.** How much better does OEH forecast the learner-specific requirement than a matched pilot-only control fitted with the same procedure?
- **RQ2.** How does the result vary by learner, behavior policy and MDP family, and how does it transfer to independently published environments?
- **RQ3 (secondary).** Does a learner selector built on the forecasts improve normalized return and success at $x10$ over the registered fixed-learner rule?

The corresponding objectives are to define and measure the target on a controlled benchmark; to design a pilot-only forecaster; and to evaluate it once, with forecasts sealed before outcomes, against registered comparators with uncertainty clustered at the level of independent MDPs.

## 1.6. Contributions and Thesis Organization

The thesis makes three contributions.

1. **Target and benchmark.** A learner-specific, right-censored empirical requirement with a sustained-success rule, and a controlled benchmark of 120 new MDP kernels, five behavior policies per MDP and 108,000 learner jobs with exact evaluation.
2. **Method.** Offline Effective Horizon (OEH), a pilot-only forecaster that combines a posterior-predictive simulation of the actual learner, a planning-risk descriptor and coverage statistics in a fitted seven-class model.
3. **Evaluation.** A prediction-sealed, pre-registered test with an architecture-matched control and MDP-clustered simultaneous bounds, together with an exploratory block ablation, external transfer tests, negative results, and a secondary selection study.

Chapter 2 defines the RL background, the study's terminology and the forecasting problem. Chapter 3 describes how the dataset was generated and analyzes its properties. Chapter 4 presents the method, the comparators, the metrics and the statistical protocol. Chapter 5 reports the results and discusses them, including negative results and limitations. Chapter 6 answers the research questions and outlines further work. The appendices list the 108 forecast features with their lineage, the learner configurations, the comparator access matrix and the history of evaluation cohorts.
