# Paper and thesis update gate

The 120-MDP internal cohort and the first five-task BRIDGE Atari panel are complete. The prospective five-task BRIDGE follow-up and the 14-task opened MiniGrid full-method analysis are still running. The latter is exploratory because its learner outcomes were previously opened to the research program.

`benchmarks/bridge/final_manuscript_watch.ps1` waits for both analyses and the strict four-panel common-metric audit. Until `artifacts/week8/difficulty_common_metric_final_2026-10-09/manifest.json` says `ALL_PANELS_COMMON_METRIC_VERIFIED`, it does not add their scores to either manuscript.

After the gate passes, `scripts/analysis/finalize_verified_manuscripts.py` checks summary and paired-score hashes and produces:

- `artifacts/week8/difficulty_common_metric_final_2026-10-09/MANUSCRIPT_RESULTS.md`, with every method, learner, metric, censoring rate and paired MDP interval;
- a short main-paper follow-up section, conditional abstract sentence, conditional discussion, and full supplement tables;
- `Drive/Thesis/05a_Bridge_Verified_Results_2026-10-09.md`, `06a_Bridge_Verified_Discussion_2026-10-09.md`, `07a_Bridge_Verified_Conclusion_2026-10-09.md`, and a hash-checked abstract sentence, included in the DOCX build.

The watcher builds `paper/main.pdf`, `paper/supplement.pdf`, and `Drive/Thesis/DAORL_Thesis_Current_2026-10-09.docx`, then records their SHA-256 hashes in `artifacts/week8/difficulty_manuscript_build_manifest_2026-10-09.json`. If an audit, source-generation or build step fails, it records `REVIEW_REQUIRED`. It never overwrites a generated source that a person has edited. The current paper remains subject to its eight-page build gate.

Prospective BRIDGE and opened MiniGrid stay separate. The published BRIDGE data are environment definitions; this group generated the offline logs and requirement labels. B2 is an internal control, XGBoost-CF/18 is an adaptation of a published survival forecaster, and scalar ERI/TQ/SACo/EH scores remain ranking diagnostics with different outputs or access. Favorable and unfavorable outcomes are reported together. Automatic builds are reviewable drafts, not submission approval.

Current status: `powershell -NoProfile -ExecutionPolicy Bypass -File Devoted/benchmarks/bridge/show_progress.ps1` from the workspace root. If Windows powers off, run the checkpoint recovery script from the same location after restart. The manuscript watcher is included in that recovery script.

On 2026-10-10 the paper and thesis openings were revised to define learner-specific empirical difficulty, distinguish an observed budget label from its seven-class forecast, explain the independent pilot and training paths, and separate forecast scores from learned-policy returns. The thesis now gives a clearly post-outcome, explanatory `CFQI=x5 / IQL=x10 / BC=>x50` example with its source in §5.2 and an early rationale for controlled synthetic data. Progress-dependent sentences were made status-neutral so completed BRIDGE follow-up tables can be inserted without contradicting the openings. The BRIDGE design table explicitly records its 2026-10-09 freeze status. The current paper PDF builds at eight pages and the thesis DOCX builds with the pipeline and requirement-distribution figures; Word reports 35 pages and three inline figures. The source generator was compacted for the eight-page main-paper budget and its supplementary LaTeX percentage formatting corrected. Final numerical interpretation remains gated on the strict audit.

A separate layout smoke used synthetic scores in a temporary copy of `paper/`, then deleted that copy. It compiled the automatically generated main-paper BRIDGE section **and conditional abstract at eight pages**, plus the supplement; it did not write synthetic scores into the maintained manuscripts. The synthetic formatter check also verified all expected learner/method rows and escaped percentages. The current thesis build passed with the conditional sources absent. These checks prove the build path and approximate page fit, not the truth of future experimental scores.
