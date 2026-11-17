## 5.9 Verified BRIDGE follow-up results

This section is generated from the strict common seven-class metric audit. Within each panel, methods use identical MDP–behavior–learner rows. Integrated Brier and seven-class log loss are lower when better. Paired differences are full OEH minus comparator; negative favors OEH. Intervals are descriptive 95% MDP-cluster bootstrap intervals, unadjusted for multiple comparisons. Do not pool panels or treat intervals as a new superiority gate.

### Prospective five-MDP BRIDGE follow-up

Status: **sealed before labels**. The published BRIDGE tasks supply MDPs; the team generated the offline pilot logs and learner-requirement labels.

| Learner | Forecast method | MDPs | Rows | Censored >x50 | Integrated Brier | Seven-class log loss |
|---|---|---:|---:|---:|---:|---:|
| BC | B2 internal | 5 | 25 | 92.0% | 0.045499 | 0.214500 |
| BC | XGBoost-CF/18 adapted | 5 | 25 | 92.0% | 0.037074 | 0.157030 |
| BC | XGBoost-CF/108 control | 5 | 25 | 92.0% | 0.065261 | 0.367247 |
| BC | Full OEH | 5 | 25 | 92.0% | 0.056365 | 0.274487 |
| CFQI | B2 internal | 5 | 25 | 40.0% | 0.284935 | 2.023458 |
| CFQI | XGBoost-CF/18 adapted | 5 | 25 | 40.0% | 0.266283 | 4.019918 |
| CFQI | XGBoost-CF/108 control | 5 | 25 | 40.0% | 0.184938 | 3.855647 |
| CFQI | Full OEH | 5 | 25 | 40.0% | 0.153300 | 1.521154 |
| IQL | B2 internal | 5 | 25 | 92.0% | 0.074708 | 0.348738 |
| IQL | XGBoost-CF/18 adapted | 5 | 25 | 92.0% | 0.074309 | 0.287602 |
| IQL | XGBoost-CF/108 control | 5 | 25 | 92.0% | 0.064962 | 0.286363 |
| IQL | Full OEH | 5 | 25 | 92.0% | 0.071049 | 0.338863 |

| Learner | Metric | Full OEH minus comparator | Paired MDP mean | Descriptive 95% interval |
|---|---|---|---:|---|
| BC | Brier | B2 internal | +0.01087 | [-0.01803, +0.04143] |
| BC | Brier | XGBoost-CF/18 adapted | +0.01929 | [-0.00794, +0.05129] |
| BC | Brier | XGBoost-CF/108 control | -0.00890 | [-0.03190, +0.01389] |
| BC | Log loss | B2 internal | +0.05999 | [-0.05227, +0.17577] |
| BC | Log loss | XGBoost-CF/18 adapted | +0.11746 | [+0.00598, +0.22894] |
| BC | Log loss | XGBoost-CF/108 control | -0.09276 | [-0.30673, +0.10168] |
| CFQI | Brier | B2 internal | -0.13163 | [-0.27212, -0.01868] |
| CFQI | Brier | XGBoost-CF/18 adapted | -0.11298 | [-0.22703, -0.01659] |
| CFQI | Brier | XGBoost-CF/108 control | -0.03164 | [-0.08456, -0.00221] |
| CFQI | Log loss | B2 internal | -0.50230 | [-1.01395, +0.07220] |
| CFQI | Log loss | XGBoost-CF/18 adapted | -2.49876 | [-4.55744, -0.54611] |
| CFQI | Log loss | XGBoost-CF/108 control | -2.33449 | [-4.14737, -0.52161] |
| IQL | Brier | B2 internal | -0.00366 | [-0.03100, +0.02440] |
| IQL | Brier | XGBoost-CF/18 adapted | -0.00326 | [-0.03150, +0.02394] |
| IQL | Brier | XGBoost-CF/108 control | +0.00609 | [-0.00870, +0.02214] |
| IQL | Log loss | B2 internal | -0.00988 | [-0.09065, +0.07421] |
| IQL | Log loss | XGBoost-CF/18 adapted | +0.05126 | [+0.01176, +0.11243] |
| IQL | Log loss | XGBoost-CF/108 control | +0.05250 | [+0.01337, +0.10577] |

The comparisons against B2 are internal feature controls. XGBoost-CF/18 adapts a published general survival method to this seven-class target; XGBoost-CF/108 also receives OEH descriptors. Original ERI, TQ, SACo and EH scores are scalar ranking diagnostics with different endpoints or access and are not assigned Brier/log-loss values in this table.

Brier versus XGBoost-CF/18 adapted: OEH has a lower point estimate for 2/3 learners; 1/3 descriptive intervals lie entirely below zero and 0/3 lie entirely above zero. These counts describe this panel only.

### Previously opened 14-MDP BRIDGE MiniGrid analysis

Status: **exploratory; outcomes already opened to the research program**. The published BRIDGE tasks supply MDPs; the team generated the offline pilot logs and learner-requirement labels.

| Learner | Forecast method | MDPs | Rows | Censored >x50 | Integrated Brier | Seven-class log loss |
|---|---|---:|---:|---:|---:|---:|
| BC | B2 internal | 14 | 70 | 90.0% | 0.022425 | 0.169015 |
| BC | Reduced OEH | 14 | 70 | 90.0% | 0.044523 | 0.323741 |
| BC | Full OEH (exploratory) | 14 | 70 | 90.0% | 0.049716 | 0.345453 |
| CFQI | B2 internal | 14 | 70 | 60.0% | 0.155727 | 1.630302 |
| CFQI | Reduced OEH | 14 | 70 | 60.0% | 0.081931 | 1.037709 |
| CFQI | Full OEH (exploratory) | 14 | 70 | 60.0% | 0.069641 | 0.975145 |
| IQL | B2 internal | 14 | 70 | 87.1% | 0.025290 | 0.274767 |
| IQL | Reduced OEH | 14 | 70 | 87.1% | 0.015152 | 0.205251 |
| IQL | Full OEH (exploratory) | 14 | 70 | 87.1% | 0.017511 | 0.216089 |

| Learner | Metric | Full OEH minus comparator | Paired MDP mean | Descriptive 95% interval |
|---|---|---|---:|---|
| BC | Brier | B2 internal | +0.02729 | [+0.01010, +0.04589] |
| BC | Brier | Reduced OEH | +0.00519 | [+0.00202, +0.00856] |
| BC | Log loss | B2 internal | +0.17644 | [+0.04405, +0.32094] |
| BC | Log loss | Reduced OEH | +0.02171 | [+0.00239, +0.04638] |
| CFQI | Brier | B2 internal | -0.08609 | [-0.10888, -0.06102] |
| CFQI | Brier | Reduced OEH | -0.01229 | [-0.01948, -0.00577] |
| CFQI | Log loss | B2 internal | -0.65516 | [-0.81222, -0.48837] |
| CFQI | Log loss | Reduced OEH | -0.06256 | [-0.10581, -0.01133] |
| IQL | Brier | B2 internal | -0.00778 | [-0.01503, -0.00056] |
| IQL | Brier | Reduced OEH | +0.00236 | [-0.00087, +0.00552] |
| IQL | Log loss | B2 internal | -0.05868 | [-0.10692, -0.00911] |
| IQL | Log loss | Reduced OEH | +0.01084 | [-0.00704, +0.02926] |

The comparisons against B2 are internal feature controls. XGBoost-CF/18 adapts a published general survival method to this seven-class target; XGBoost-CF/108 also receives OEH descriptors. Original ERI, TQ, SACo and EH scores are scalar ranking diagnostics with different endpoints or access and are not assigned Brier/log-loss values in this table.

Brier versus B2 internal: OEH has a lower point estimate for 2/3 learners; 2/3 descriptive intervals lie entirely below zero and 1/3 lie entirely above zero. These counts describe this panel only.

### Posterior draw coverage on the prospective panel

The prelabel OEH-B audit returned fewer than 64 posterior draws in **4/25** MDP–behavior groups; **4/25** returned none. For a zero-draw group, the frozen requirement forecast treats all 64 requested draws as censored and applies Krichevsky–Trofimov smoothing. This is a registered fallback, not an empirical 64-draw posterior estimate. The score tables retain these groups and must be read with this coverage limit.

| MDP | Behavior | Posterior draws returned / requested |
|---|---|---:|
| MiniGrid-Empty-16x16-v0 | perturbed_reward | 0/64 |
| MiniGrid-DoorKey-5x5-v0 | structured_random | 0/64 |
| MiniGrid-DoorKey-5x5-v0 | perturbed_reward | 0/64 |
| MiniGrid-DoorKey-5x5-v0 | two_policy_mixture | 0/64 |

