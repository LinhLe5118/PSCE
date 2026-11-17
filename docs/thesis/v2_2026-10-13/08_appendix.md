# Appendix

## Appendix A. The 108 Forecast Columns and Their Lineage

The table lists the columns of the sealed candidate in their protocol order, as exported by the M1 lineage (`artifacts/week8/m1_matched_ablation_2026-10-13/feature_lineage.csv`), which also records the producing function, source artifact, dependencies and non-finite encoding of every column. Context columns are copies of another learner's value and are identical for the three learner rows of one MDP–behavior pair. No column reads a learner outcome or the true MDP.

Table: Table A.1. Forecast columns (learner names in context columns abbreviated).
| # | Column | Block | Compute component | Scope | Meaning |
|---|---|---|---|---|---|
| 1 | `log_pilot_transitions` | G1 | pilot statistics | shared | log pilot size N0 |
| 2 | `log_state_action_count` | G1 | pilot statistics | shared | log of S times A (public metadata) |
| 3 | `observed_sa_fraction` | G1 | pilot statistics | shared | fraction of state-action pairs seen in the pilot |
| 4 | `observed_state_fraction` | G1 | pilot statistics | shared | fraction of states seen in the pilot |
| 5 | `normalized_visitation_entropy` | G1 | pilot statistics | shared | entropy of state-action visitation divided by log(SA) |
| 6 | `log1p_min_positive_sa_count` | G1 | pilot statistics | shared | log1p of the smallest positive state-action count |
| 7 | `trajectory_return_mean` | G1 | pilot statistics | shared | mean discounted pilot trajectory return |
| 8 | `trajectory_return_std` | G1 | pilot statistics | shared | std of discounted pilot trajectory return |
| 9 | `trajectory_return_iqr` | G1 | pilot statistics | shared | IQR of discounted pilot trajectory return |
| 10 | `terminal_trajectory_fraction` | G1 | pilot statistics | shared | fraction of pilot trajectories reaching a terminal |
| 11 | `weighted_transition_entropy` | G1 | pilot statistics | shared | count-weighted normalized next-state entropy |
| 12 | `observed_cell_transition_entropy` | G1 | pilot statistics | shared | mean normalized next-state entropy over seen cells |
| 13 | `low_count_cell_fraction` | G1 | pilot statistics | shared | fraction of cells with 1-2 visits |
| 14 | `empirical_stochastic_cell_fraction` | G1 | pilot statistics | shared | fraction of cells (>=2 visits) with several successors |
| 15 | `low_coverage_entropy` | G1 | pilot statistics | shared | mean next-state entropy of cells with <3 visits |
| 16 | `high_coverage_entropy` | G1 | pilot statistics | shared | mean next-state entropy of cells with >=3 visits |
| 17 | `bootstrap_return_mean_std` | G1 | pilot statistics | shared | trajectory-bootstrap std of mean return |
| 18 | `bootstrap_rare_exposure_mean_std` | G1 | pilot statistics | shared | trajectory-bootstrap std of rare-transition exposure |
| 19 | `oehb_cdf_0` | G2 | posterior simulation | own | posterior-predictive P(E <= class 0) after KT smoothing |
| 20 | `oehb_cdf_1` | G2 | posterior simulation | own | posterior-predictive P(E <= class 1) after KT smoothing |
| 21 | `oehb_cdf_2` | G2 | posterior simulation | own | posterior-predictive P(E <= class 2) after KT smoothing |
| 22 | `oehb_cdf_3` | G2 | posterior simulation | own | posterior-predictive P(E <= class 3) after KT smoothing |
| 23 | `oehb_cdf_4` | G2 | posterior simulation | own | posterior-predictive P(E <= class 4) after KT smoothing |
| 24 | `oehb_cdf_5` | G2 | posterior simulation | own | posterior-predictive P(E <= class 5) after KT smoothing |
| 25 | `oehb_success_x1` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x1 |
| 26 | `oehb_success_x2` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x2 |
| 27 | `oehb_success_x5` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x5 |
| 28 | `oehb_success_x10` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x10 |
| 29 | `oehb_success_x20` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x20 |
| 30 | `oehb_success_x50` | G2 | posterior simulation | own | mean simulated seed success (NR>=0.95) at budget x50 |
| 31 | `oehb_return_x1` | G2 | posterior simulation | own | mean simulated normalized return at budget x1 |
| 32 | `oehb_return_x2` | G2 | posterior simulation | own | mean simulated normalized return at budget x2 |
| 33 | `oehb_return_x5` | G2 | posterior simulation | own | mean simulated normalized return at budget x5 |
| 34 | `oehb_return_x10` | G2 | posterior simulation | own | mean simulated normalized return at budget x10 |
| 35 | `oehb_return_x20` | G2 | posterior simulation | own | mean simulated normalized return at budget x20 |
| 36 | `oehb_return_x50` | G2 | posterior simulation | own | mean simulated normalized return at budget x50 |
| 37 | `oehh_value` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 38 | `oehh_identified` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 39 | `oehh_trap` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 40 | `oehh_statistical` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 41 | `oehh_coverage` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 42 | `oehh_depth_fraction` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 43 | `oehh_log_concentrability` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 44 | `oehh_log_clipped_concentrability` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 45 | `oehh_log_intrinsic_bound` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 46 | `oehh_behavior_ceiling` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 47 | `oehh_support_ceiling` | G4 | posterior-mean planning | shared | OEH-H descriptor or ceiling on the posterior-mean model |
| 48 | `oehb_log_transition_concentration` | G3 | posterior fit | shared | log empirical-Bayes transition Dirichlet concentration |
| 49 | `oehb_log_behavior_concentration` | G3 | posterior fit | shared | log empirical-Bayes behavior Dirichlet concentration |
| 50 | `other_BC_oehb_cdf_0` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 51 | `other_BC_oehb_cdf_1` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 52 | `other_BC_oehb_cdf_2` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 53 | `other_BC_oehb_cdf_3` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 54 | `other_BC_oehb_cdf_4` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 55 | `other_BC_oehb_cdf_5` | G6 | posterior simulation | cross-learner | OEH-B CDF of BC (context copy) |
| 56 | `other_CFQI_oehb_cdf_0` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 57 | `other_CFQI_oehb_cdf_1` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 58 | `other_CFQI_oehb_cdf_2` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 59 | `other_CFQI_oehb_cdf_3` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 60 | `other_CFQI_oehb_cdf_4` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 61 | `other_CFQI_oehb_cdf_5` | G6 | posterior simulation | cross-learner | OEH-B CDF of CFQI (context copy) |
| 62 | `other_IQL_oehb_cdf_0` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 63 | `other_IQL_oehb_cdf_1` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 64 | `other_IQL_oehb_cdf_2` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 65 | `other_IQL_oehb_cdf_3` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 66 | `other_IQL_oehb_cdf_4` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 67 | `other_IQL_oehb_cdf_5` | G6 | posterior simulation | cross-learner | OEH-B CDF of IQL (context copy) |
| 68 | `oeh3_log1p_requirement` | G7 | OEH3 risk | shared | OEH3 return-weighted identification risk summary |
| 69 | `oeh3_selected_qvi_fraction` | G7 | OEH3 risk | shared | OEH3 return-weighted identification risk summary |
| 70 | `oeh3_risk_at_pilot` | G7 | OEH3 risk | shared | OEH3 return-weighted identification risk summary |
| 71 | `oeh3_policy_instability` | G7 | OEH3 risk | shared | OEH3 return-weighted identification risk summary |
| 72 | `lcoeh4_risk_at_pilot` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 73 | `lcoeh4_contraction_rate` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 74 | `lcoeh4_pessimistic_return` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 75 | `lcoeh4_return_se` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 76 | `lcoeh4_validation_support_mass` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 77 | `lcoeh4_trajectory_independence_fraction` | G7 | LCOEH4 cross-fit | own | LCOEH4 cross-fitted learner diagnostic of the row's learner |
| 78 | `other_BC_lcoeh4_risk_at_pilot` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 79 | `other_BC_lcoeh4_contraction_rate` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 80 | `other_BC_lcoeh4_pessimistic_return` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 81 | `other_BC_lcoeh4_return_se` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 82 | `other_BC_lcoeh4_validation_support_mass` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 83 | `other_BC_lcoeh4_trajectory_independence_fraction` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of BC (context copy) |
| 84 | `other_CFQI_lcoeh4_risk_at_pilot` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 85 | `other_CFQI_lcoeh4_contraction_rate` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 86 | `other_CFQI_lcoeh4_pessimistic_return` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 87 | `other_CFQI_lcoeh4_return_se` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 88 | `other_CFQI_lcoeh4_validation_support_mass` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 89 | `other_CFQI_lcoeh4_trajectory_independence_fraction` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of CFQI (context copy) |
| 90 | `other_IQL_lcoeh4_risk_at_pilot` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 91 | `other_IQL_lcoeh4_contraction_rate` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 92 | `other_IQL_lcoeh4_pessimistic_return` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 93 | `other_IQL_lcoeh4_return_se` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 94 | `other_IQL_lcoeh4_validation_support_mass` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 95 | `other_IQL_lcoeh4_trajectory_independence_fraction` | G8 | LCOEH4 cross-fit | cross-learner | LCOEH4 diagnostic of IQL (context copy) |
| 96 | `oehh_risk_x1` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 97 | `oehh_risk_x2` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 98 | `oehh_risk_x5` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 99 | `oehh_risk_x10` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 100 | `oehh_risk_x20` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 101 | `oehh_risk_x50` | G5 | posterior-mean planning | shared | modeled noise regret at budget |
| 102 | `oehh_miss_x1` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 103 | `oehh_miss_x2` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 104 | `oehh_miss_x5` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 105 | `oehh_miss_x10` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 106 | `oehh_miss_x20` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 107 | `oehh_miss_x50` | G5 | posterior-mean planning | shared | modeled exponential missing mass at budget |
| 108 | `oehh_risk_slack` | G5 | posterior-mean planning | shared | NR slack used by the risk inversion |

## Appendix B. Information-Access Matrix

Table: Table B.1. What each component of the study may read.
| Component | Pilot log | Public metadata | Meta-training labels | True behavior policy | True MDP | Family label | Test outcomes |
|---|---|---|---|---|---|---|---|
| OEH and matched pilot-only rows | yes | yes | for fitting | no | no | no | no |
| Privileged rows (oracle EH, TQ/SACo) | yes | yes | for fitting | no | yes (descriptors or reference data) | no | no |
| BC, CFQI, IQL, BCQ, CQL | training buffer | yes | — | no | no | no | — |
| SPIBB | training buffer | yes | — | **yes** | no | no | — |
| VI-LCB | training buffer | yes + terminal states | — | no | no | no | — |
| OEH selector | yes | yes | opened fourth cohort | no | no | **yes** | no |
| Outcome oracle (bound) | — | — | — | — | — | — | **yes** |
| Scorer | — | — | — | — | yes (exact evaluation) | — | produces them |

## Appendix C. Protocol, Seal and Amendments of the Test Cohort

The protocol of the seventh cohort fixes the 120 kernels and their fingerprints, the family quotas, the five behaviors, the pilot and budget multipliers, the ten training seeds, the three label learners, the requirement rule, the OEH configuration (posterior structure, 64 requirement draws, the 108 columns, the matched head without the raw-law component), the comparator list and tiers, the bootstrap and the decision rules, the selection rule, and the hash of every source file. Four amendments were recorded before any learner outcome existed: a schema token required by the verifier, resolved kernel seeds after a duplicate check, a namespace prefix for human-readable identifiers reused by the fourth and seventh cohorts, and an overlap guard for the selector. Each amendment archived the previous protocol and recorded that zero outcomes had been observed; none changed a decision rule. The pre-outcome seal hashes the pilot features, OEH-H descriptors, published-metric features, oracle descriptors (for the privileged rows), all predictions, the tuning record and the learner choices. The result manifest records 108,000 completed jobs and points back to the protocol, seal and execution plan.

## Appendix D. Evaluation History

Table: Table D.1. Evaluation cohorts of the research program.
| Cohort | Role | Outcome |
|---|---|---|
| E1 (120 MDPs) | development | adaptive development of features and the earlier v5 router; reused as meta-training |
| First and second 60-MDP cohorts | earlier validations | include a sealed routing validation with a NO-GO verdict; reused as meta-training |
| 1st–3rd new 120-MDP tests | sealed tests of earlier OEH formulations | NO-GO (Table 5.6); reused as meta-training |
| 4th new 120-MDP test | sealed test of the current configuration | Tier 1 GO; Tier 2, combined and selection NO-GO; reused as meta-training for the 7th |
| 7th new 120-MDP test | sealed test reported in this thesis | Tier 1, Tier 2, combined GO; selection GO vs fixed CFQI, not superior to v5 |
| Farama FourRooms | published offline dataset, component test | NO-GO |
| BRIDGE MiniGrid-14 | published MDPs, reduced OEH | mixed (BC worse; CFQI, IQL better) |
| BRIDGE Atari-5 and expanded-5 | published MDPs, full OEH, sealed before labels | mixed and learner-dependent |
| M1 block ablation | opened 7th cohort | exploratory; redundancy among blocks |

## Appendix E. Artifacts and Reproduction

Table: Table E.1. Main artifacts (paths relative to the project repository).
| Content | Location |
|---|---|
| Sealed test cohort (protocol, seal, predictions, labels, scores, contrasts, selection) | `artifacts/oeh_confirmatory_unseen_120_2026-10-04/` |
| Block ablation with lineage, cost probe, input and source hashes | `artifacts/week8/m1_matched_ablation_2026-10-13/` |
| Tables and figures of Chapters 3–5 | `artifacts/week8/manuscript_tables_2026-10-13/` |
| BRIDGE transfer analyses | `artifacts/week8/bridge_*_test_analysis_2026-10-0*/` |
| FourRooms component test | `artifacts/published_fourrooms_calibration_2026-10-04/` |
| Forecasting head and feature assembly | `scripts/analysis/oeh_final_forecast.py` |
| OEH posterior, simulation and horizon code | `code/oeh/` |

The ablation is reproduced with `python -m scripts.analysis.run_m1_matched_ablation lineage|cost|forecast|score --output <new directory>`, and every table and figure of Chapters 3–5 with `python -m scripts.analysis.export_manuscript_tables_20261013 --output <new directory>`. Both refuse to overwrite existing outputs, verify the hashes of the sealed inputs, and record the hashes of their own outputs.
