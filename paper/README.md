# OEH forecasting paper

**16/10/2026, naming sync.** Names follow the locked set of the writing kit: Test 1-3, cohort 1-8 numbering, $\bar p_l$ for the class prior, shared row labels; five figures redrawn by `scripts/analysis/figures_final_naming_20261016.py`. Record: [naming sync](../docs/week8/NAMING_SYNC_2026-10-16.md).

**16/10/2026, final revision.** Rewritten around the three final tests: Test 1 (four panels, Table II), Test 2 (fifth panel, Table III, NO-GO vs the robust head) and Test 3 (sixth panel, Table IV and `figures/fig_panel6.pdf` from `scripts/analysis/paper_fig_panel6.py`; registered gates F1 and F2 GO on the extrapolation endpoint). New: the extrapolation endpoint (Eq. 7), the guarantees subsection `sec:theory` (gate safety bound, regret from scoring, no consistency claim), the prefix + FQE and TabPFN baselines, and the post-hoc analyses T1-T6. Decision-rule notation is $q$ (threshold) and $\kappa$ (penalty). Supplement adds `supplement_theory_checked`, `supplement_panel6` and `supplement_posthoc`. Record: [paper update](../docs/week8/PAPER_UPDATE_2026-10-16.md).

**15/10/2026, final-method revision.** The manuscript now presents the shift-gated forecaster as OEH (method: `03_difficulty.tex` + `03b_shift_gate.tex`; setup with development vs final panels in `05_setup.tex`; results in `06_results.tex`: four-panel sealed test (Table II), fifth panel (Table III, NO-GO vs the robust head), mechanism with `figures/fig_gain_by_distance.pdf`, development evidence (Table IV), selection, cost and limitations). The seventh-cohort table and threshold figure moved to `supplement_development.tex`. Renaming: OEH-F/full head = former OEH; robust head = B2 + legacy; pilot diagnostics D = legacy diagnostics (G7, G8).

**15/10/2026 revision.** Adds OEH v2: method subsection `sections/03b_oeh_v2.tex` (Eq. 7, Proposition 2 in the supplement) and results subsection VI.F *Sealed test of OEH v2 on four new panels* with Table IV (`artifacts/oeh_v2_sealed_test_2026-10-14/`). To stay within eight pages the selector table, the BRIDGE transfer table and the exploratory M1 table moved to the supplement (`supplement_selection`, `supplement_reduced_bridge`, new `supplement_ablation`), and a new `supplement_theory_v2` holds Propositions 1-2 and the robustness table. The cross-panel figure `figures/fig_cross_panel.pdf` (script `scripts/analysis/figure_cross_panel_20261014.py`) is used in the thesis. Writing plan: [15/10](../docs/week8/KE_HOACH_VIET_PAPER_THESIS_2026-10-15.md).

**14/10/2026 revision.** Builds to 8 pages (the budget). New section VI.E *Sealed replication and external panels* (`sections/06_results.tex`, Table V) reports the eighth cohort (168 internal MDPs), the two Colosseum panels and the SPIBB panel; abstract, ablation, cost/decision, limitations and conclusion follow the [cross-panel re-analysis](../docs/week8/REANALYSIS_ALL_PANELS_2026-10-14.md): external claims are restricted to integrated Brier, the external mechanism (detection of insufficient data and calibration, worse finite-budget placement) is stated, and the XGBoost-CF log-loss gap carries a floor note. The first Colosseum panel's detail and the authors' TQ/SACo row are in `sections/supplement_colosseum.tex`. Allowed wording and the remaining page-budget decisions are in the [14/10 writing plan](../docs/week8/KE_HOACH_VIET_PAPER_THESIS_2026-10-14.md). Sources: `artifacts/oeh_eighth_cohort_2026-10-14/`, `artifacts/oeh_spibb_panel_2026-10-14/`, `artifacts/week8/colosseum_posthoc_part2_2026-10-14/`, `artifacts/week8/cross_panel_reanalysis_2026-10-14/`.


**13/10/2026 revision.** The manuscript now follows the IEEE outline (I Introduction, II Related Work, III Problem Formulation, IV Proposed Method in `03_difficulty.tex`, V Experimental Setup, VI Results and Discussion with the M1 ablation in `06a_ablation.tex`, VII Conclusion) and builds to 7 pages with `build.ps1`. Superseded section files are in `sections/superseded_2026-10-13/`. Figure `fig_threshold_brier.pdf` (with `oeh_pipeline.pdf`) comes from `artifacts/week8/manuscript_tables_2026-10-13/`. Author metadata is still a placeholder.

Updated 2026-10-05. The maintained source is main.tex and sections/01_introduction.tex through sections/07_discussion.tex. The paper uses the title "Difficulty-Aware Offline Reinforcement Learning" and centers the final pilot-only learner-specific requirement forecaster; its separately registered x10 learner selector is a secondary application. The [T1 terminology protocol](../docs/week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) and [claim map](../docs/week8/T1_CLAIM_MAP_2026-10-05.md) define current terms and evidence scope. Unused historical tables moved to tests/fixtures/paper_history so their consistency checks still run; unused figures were removed from this directory and remain recoverable from Git history. The checked-in main.pdf predates the 05/10 text edits and must be rebuilt from the current source.

The current confirmatory source is ../artifacts/oeh_confirmatory_unseen_120_2026-10-04. Read result_manifest.json, summary.csv, and contrasts.csv for forecasting, then confirmatory_verdict.json and selection_summary.csv for the registered selector result. external_execution_audit.json verifies all 16,800 external jobs. The published external component test is in ../artifacts/published_fourrooms_calibration_2026-10-04/result.json. Protocol, pre-outcome seal, execution plan hashes, and archived pre-outcome amendments document what was fixed before learner outcomes. The fourth-cohort failures remain in ../artifacts/oeh_final_unseen_120_2026-10-02.

Main statements to preserve: seventh-cohort Tier 1, Tier 2 and all-comparator forecast GO, with all 22 simultaneous lower bounds positive on 84 stochastic MDPs; seventh-cohort selector GO against the best fixed learner, CFQI, on both success and return but NO-GO in the supplementary v5 comparison; fourth-cohort Tier 2 and combined NO-GO; published Farama FourRooms component NO-GO. The fixed learner table now lists all seven recorded implementations, including the author-code and library methods. The seventh cohort uses new kernels from existing generator families, so full-method external transfer remains open. A literature-wide SOTA claim is not supported because published methods address different targets or use different information. The raw OEH-B predictor is not calibrated. Author names, affiliation, venue formatting, and a fully redistributable training input package remain submission tasks.

Build on Windows with PowerShell using paper/build.ps1 from the repository root. The script calls pdflatex and bibtex and enforces the current eight-page budget. A functional TeX installation with IEEEtran is required. The exact comparator information-access table is in COMPARATOR_SCOPE.md.

The 08/10 H1 writing handover adds explicit notation, the empirical suffix
label, OEH-B versus final forecast, observed versus modeled support, the
implemented OEH-H risk approximation, and discrete Brier/log-loss formulas.
See [H1 theory/glossary](../docs/week8/H1_THEORY_GLOSSARY_2026-10-08.md) and
[formula/code map](../docs/week8/H1_FORMULA_CODE_MAP_2026-10-08.md). These edits
do not change the algorithm or sealed scores. The checked-in PDF is not silently
replaced by this source-only handover; rebuild and review the page budget before
submission.

The [11/10 D1 matrix](../docs/week8/D1_LEARNER_COMPARATOR_METRICS_2026-10-11.md)
supplies the seven-learner provenance/configuration/access table, B2's 18
features, metric definitions and independently checked primary forecast blocks.
Paper-ready [learner](../artifacts/week8/d1_2026-10-11/learner_matrix.tex) and
[primary forecast](../artifacts/week8/d1_2026-10-11/primary_forecast.tex) fragments
are generated with `python -m scripts.analysis.export_week8_d1`. They are
handover material, not extra tables silently inserted into the eight-page
manuscript. In particular, disclose true logging beta for SPIBB and family
metadata for the secondary selector when integrating the learner matrix.
