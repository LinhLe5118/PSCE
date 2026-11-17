# Fourth fresh 120-MDP test of the final OEH (one shot)

Protocol `protocol.json` (frozen 2026-10-02; four pre-outcome orchestration
amendments recorded inside it with the previous versions saved alongside).
Forecasts and learner choices sealed in `preoutcome_seal.json` before any learner
outcome. Verdicts are in `result_manifest.json`:

* difficulty, tier 1 (same pilot-only information as OEH): **GO**
* difficulty, tier 2 (oracle EH, author-code TQ/SACo with true-MDP reference data): NO-GO
  (only the TQ/SACo contrast fails: Brier +7.8%, log loss +3.7%, lower bounds negative)
* learner selection (supplementary): NO-GO (success +1.76 pp vs CFQI significant;
  return not significant; worse than the OEH-v5 router on success)

Main tables: `contrasts.csv`, `summary.csv`, `mdp_scores.csv`, `selection_summary.csv`,
`selection_contrasts.csv`, `horizon_comparison.csv`. Figures: `paper/figures/oeh_final_*`.
`execution_plan.json.gz` is a byte-exact copy of the frozen plan; its raw and
compressed hashes are in `execution_plan_archive.json`. The expanded plan,
learner-job records and intermediate group files stay local because they contain
over 125,000 generated files. Restore the plan with gzip before a local rerun.
Reproduce: `python -m scripts.analysis.run_oeh_final_120 analyze` (requires the local
dataset and job records) and `python -m scripts.analysis.render_oeh_final`.
