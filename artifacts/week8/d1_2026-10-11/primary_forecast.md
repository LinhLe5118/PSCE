# D1 — Primary forecast matrix

84 stochastic MDPs; lower scores are better. Frozen cohort 7; no new fitting.

| Group | Row | Head | Access | Brier | Log loss |
|---|---|---|---|---:|---:|
| Matched pilot | B2 matched | matched | pilot-only | 0.054727 | 0.627179 |
| Matched pilot | B2 + ERI adaptation | matched | pilot-only | 0.056672 | 0.632970 |
| Matched pilot | B2 + BWD adaptation | matched | pilot-only | 0.053954 | 0.621165 |
| Matched pilot | OEH current (108 features) | matched | pilot-only | 0.045989 | 0.563079 |
| AFT-only pilot | B2 generic | AFT-only | pilot-only | 0.063990 | 0.782395 |
| AFT-only pilot | Coverage adaptation | AFT-only | pilot-only | 0.097201 | 1.124217 |
| AFT-only pilot | Quality adaptation | AFT-only | pilot-only | 0.084791 | 0.967090 |
| AFT-only pilot | Coverage + quality adaptation | AFT-only | pilot-only | 0.068844 | 0.825023 |
| Frequency | Train frequency | frequency | meta-training labels only | 0.109529 | 0.942767 |
| Privileged | B2 + oracle EH | matched | true MDP | 0.054712 | 0.626642 |
| Privileged | B2 + TQ/SACo reference | matched | true-MDP reference datasets | 0.052213 | 0.614164 |
| Privileged | Published-metrics-only block | matched | includes true-MDP references | 0.062988 | 0.678562 |
| Diagnostic (outside gate) | Earlier OEH diagnostics | matched | pilot-only | 0.050907 | 0.607922 |
| Diagnostic (outside gate) | Raw OEH-B law | unfitted posterior | pilot-only | 0.097826 | 0.910682 |

Matched = AFT + direct-threshold + depth-3 multiclass, fitted separately for each row.
Privileged and AFT-only rows are separate comparisons; diagnostics are outside the 22-contrast gate.
