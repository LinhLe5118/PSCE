# Fresh full-method BRIDGE Atari transfer — 2026-10-08

## Protocol and provenance

The frozen [protocol](../../benchmarks/bridge/fresh_panel_2026-10-08/protocol.json) selected five published BRIDGE Atari MDPs before outcome inspection. The 25% sticky-action adapter adds the previous executed action to the state; the team generated all offline pilot trajectories and learner-requirement labels. This is an external **environment** test, not a published offline-log dataset. Task-specific finite horizons, reward scaling by each task's published positive maximum, and `gamma=0.99` differ from the 120-MDP internal panel.

The [prelabel seal](../../benchmarks/bridge/fresh_panel_2026-10-08/prelabel_seal.json) fixed 75 seven-class predictions for each of full OEH (108 inputs), internal matched B2, published XGBoost-CF survival formulation adapted with 18 generic inputs, and XGBoost-CF with the same 108 OEH inputs as an architecture control. It also fixed 25 original scalar-score rows. The [label manifest](../../artifacts/week8/bridge_fresh_test_labels_2026-10-08/manifest.json) records 75 learner labels from five tasks, five behavior policies and BC/CFQI/IQL. The [analysis manifest](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/manifest.json) records `COMPLETE_FRESH_BRIDGE_ATARI_FULL_OEH_TRANSFER`, source/output hashes and 75 rows per forecaster.

## Seven-class forecasting result

All scores below are from the [sealed summary](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/summary.csv). Integrated Brier is the mean of six cumulative-threshold squared errors. Lower is better.

| Learner | Method | Brier | Log loss | `>x50` |
|---|---|---:|---:|---:|
| BC | B2 internal | 0.059865 | 0.497956 | 92% |
| BC | XGBoost-CF/18 adapted | 0.045579 | 2.277528 | 92% |
| BC | XGBoost-CF/108 control | 0.028191 | 1.305974 | 92% |
| BC | Full OEH | 0.028178 | 0.382242 | 92% |
| CFQI | B2 internal | 0.296849 | 2.099780 | 48% |
| CFQI | XGBoost-CF/18 adapted | 0.292000 | 3.924963 | 48% |
| CFQI | XGBoost-CF/108 control | 0.261571 | 2.920034 | 48% |
| CFQI | Full OEH | 0.284121 | 2.111864 | 48% |
| IQL | B2 internal | 0.049781 | 0.324704 | 84% |
| IQL | XGBoost-CF/18 adapted | 0.071166 | 0.358160 | 84% |
| IQL | XGBoost-CF/108 control | 0.117773 | 0.634500 | 84% |
| IQL | Full OEH | 0.112798 | 0.819092 | 84% |

The [paired five-task bootstrap](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/paired_task_bootstrap.csv) gives OEH-minus-B2 Brier deltas of BC `−0.031687 [−0.083220,+0.009082]`, CFQI `−0.012728 [−0.071202,+0.035563]`, and IQL `+0.063018 [+0.004222,+0.138822]`. OEH-minus-XGBoost-CF/18 deltas are BC `−0.017401 [−0.040827,+0.002368]`, CFQI `−0.007879 [−0.069256,+0.044753]`, and IQL `+0.041633 [+0.002215,+0.094243]`. These are **descriptive**, per-contrast intervals with five MDP clusters and no simultaneous superiority test. IQL log loss is also unfavorable versus all three controls. CFQI Brier is worse than the full-input XGBoost-CF control by `+0.022550 [+0.003642,+0.052617]`.

The [original scalar-score table](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/original_rank_comparison.csv) is a different endpoint and information tier. In the [paired easier-requirement ranking](../../artifacts/week8/bridge_fresh_full_test_analysis_2026-10-08/original_rank_paired_transfer.csv), original TQ's Spearman point estimates are `0.470/0.635/0.632` for BC/CFQI/IQL, versus OEH's `0.429/0.596/0.501`; TQ uses privileged true-MDP DQN reference data. ERI, SACo and author EH scores are reported with defined-row counts and access tier. None produces a seven-class probability law on this target. The original BRIDGE work and original XGBoost-CF work did not publish these learner-requirement labels or the scores in our tables.

## Execution and interpretation

OEH-B completed 25/25 pilot groups. The background PowerShell monitor could not start the next Python subprocess (`-1073741790`) after OEH-B completion. The full forecast module was then run directly from the already frozen inputs; it sealed 75 predictions before any labels existed. The prelabel-seal module passed, the one registered label pass produced 25/25 group checkpoints and 75/75 labels, and the scoring module completed with hash checks. No protocol, scientific source, sealed prediction or label was changed to recover the orchestration failure. The monitor log retains its transient `NO_GO`; the progress JSON was subsequently marked `COMPLETE` with the recovery detail. The analysis manifest is the scientific completion record.

The transfer result is mixed. It supports neither uniform learner improvement nor BRIDGE-wide or literature-wide state of the art. The 14-task reduced MiniGrid result remains a separate earlier test with a different OEH arm and different tasks. The [OEH-B manifest](../../artifacts/week8/bridge_fresh_full_oehb_preoutcome_2026-10-08/manifest.json) records a sum of 33,528.44 seconds across 25 group computations, or 9.31 group-compute hours; groups ran in parallel, so this is not wall time. Comparable full-pipeline cost for every arm and net data/compute savings were not established by this score run.
