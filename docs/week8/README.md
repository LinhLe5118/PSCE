**Start at the [root README](../../README.md) (11/11/2026).** It describes the final method, the three final tests and the command that reproduces their results; this page is the chronological record of week 8 and later.

# Week 8 difficulty-aware terminology

**Naming sync (16/10/2026):** the [naming record](NAMING_SYNC_2026-10-16.md) locks one set of names for the thesis, paper, supplement and figures: Test 1-3 for the final tests, cohorts numbered chronologically (E1, cohorts 1-8; the earlier 'cohort 4' is cohort 6), $\bar p_l$ for the class prior, shared table row labels, and the H6 review fixes of PR #68. Five figures were redrawn from committed data (`scripts/analysis/figures_final_naming_20261016.py`).

**Paper update (16/10/2026), current entry point for writing:** the [paper update record](PAPER_UPDATE_2026-10-16.md). `paper/main.tex` (IEEE conference, 8 pages) now reports the three final tests (four panels, fifth panel, sixth panel with F1/F2 GO), the checked guarantees (gate safety bound, regret from scoring, local constancy and gate saturation; no consistency claim), the prefix + FQE and TabPFN baselines and the post-hoc analyses T1-T6; the supplement adds their proofs and details.

**Panel 6 results (16/10/2026), current entry point:** [T10 record](T10_PANEL6_RESULTS_2026-10-16.md); sealing in the [T9 record](T9_PANEL6_SEALED_2026-10-15.md). Two new families with a real instance generator (MDP Playground 96, labmaze 96), protocol `artifacts/oeh_panel6_2026-10-15` (sha 49c27071, amendment A1). Registered gates: **F1 GO** (OEH vs B2 + prefix/FQE on the extrapolation endpoint) and **F2 GO** (vs B2); F3 regret estimate favours OEH (no verdict). Descriptively the full head OEH-F beats the gated OEH (-3.3% IBS, -3.6% LL), mostly on MDP Playground where the gate saturates.

**Seal integrity (15/10/2026):** `tests/test_sealed_integrity.py` fails CI if any frozen protocol, seal or manifest entry no longer matches its tracked file.

**Final method and fifth panel (15/10/2026, afternoon) — current entry point:** writing now calls the shift-gated forecaster **OEH** and the earlier forecaster **OEH-F** (full head). A fifth sealed panel (Gymnasium FrozenLake 36, gambler's problem 36; `artifacts/oeh_fifth_panel_2026-10-15`) registered one contrast, OEH vs the robust head (B2 + D): **NO-GO**; OEH again beats its full head and B2 in log loss. The single entry point `scripts/analysis/oeh_pipeline.py` reproduces all four sealed panels; post-hoc analyses (gain by distance, ablations, generic baselines) are in `artifacts/week8/oeh_final_posthoc_2026-10-15`. See the [OEH record](OEH_V2_DEVELOPMENT_2026-10-14.md) sections 5-6 and section 0 of the [15/10 writing plan](KE_HOACH_VIET_PAPER_THESIS_2026-10-15.md).

**OEH v2 sealed test (15/10/2026):** the [OEH v2 record](OEH_V2_DEVELOPMENT_2026-10-14.md) documents the development on opened panels (M1/M4 selected; M6, M2 reward prior and M3 behavior cohort not adopted by their pre-registered rules), the deterministic frozen specification, and the sealed test on four new panels (internal 168, Colosseum 2029 72, SPIBB-2 108, MDPToolbox forest 48; 371,520 learner jobs): steps 1-3 GO, OEH v2 adopted. Descriptively, the margin over B2 + legacy is carried by the forest panel; the margin over OEH holds with any external panel left out. Proposition 2 (per-forecast mixture bound) is in the [D3 theory note](D3_THEORY_NOTE_2026-10-14.md). The [15/10 writing plan](KE_HOACH_VIET_PAPER_THESIS_2026-10-15.md) supersedes the 14/10 plan (venue strategy, thesis status).

**Writing plan (14/10/2026, superseded by the 15/10 plan):** the [14/10 paper/thesis writing plan](KE_HOACH_VIET_PAPER_THESIS_2026-10-14.md) superseded the 13/10 plan: evidence status per panel, contribution positioning, allowed wording, section status of the paper and thesis, and the prioritized next steps (review PR #53; one sealed test of the hybrid head; thesis chapters 1–4; no history rewrite).

**Re-analysis of all panels (14/10/2026):** the [cross-panel re-analysis](REANALYSIS_ALL_PANELS_2026-10-14.md) recomputes every sealed full-OEH panel with one scorer and adds random-effects pooling (within the generator 16.8% IBS / 10.1% LL, I^2 = 0; outside 16.0% IBS, 4.9% LL not significant), censored/finite and swap decompositions, Murphy calibration, floor and influence checks, distribution-shift and label-noise analyses. Descriptive only.

**Eighth cohort, SPIBB and development (14/10/2026, evening):** the [cohort-8 record](COHORT8_SPIBB_DEVELOPMENT_2026-10-14.md) reports the development of OEH-S/OEH-2T/LCE/CB on the opened meta panel, the sealed gatekeeping replication on 168 new internal MDPs (steps 1-2 GO), a new 72-kernel Colosseum tier (step 1 GO, step 2 NO-GO), the SPIBB panel with author MDPs and behaviors (step 1 NO-GO), and the completed Colosseum amendment-1 family with the authors' TQ/SACo. The [D3 theory note](D3_THEORY_NOTE_2026-10-14.md) states the consistency of the raw OEH-B law and its limits.

**Colosseum external panel (14/10/2026):** the [Colosseum record](COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md) reviews the secondary machine's handover, corrects two of its numbers (max-t rule; 42/360 zero-draw OEH-B groups) and reports the 72-MDP Colosseum panel: pre-registered OEH vs B2 matched GO (log-loss bound +0.0003 under the team rule), cohort-7 comparator set tier 1 NO-GO (B2 + ERI), and post-hoc XGBoost-CF, block-factorial Shapley, decision-regret and cost analyses. Everything beyond the pre-registered gate is post hoc.

**Current entry point (14/10/2026):** the [BRIDGE results summary](BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md) reports the three scored BRIDGE panels (reduced MiniGrid-14, full-OEH Atari-5 and expanded-5); the [evidence and retirement ledger](CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md) fixes which results may be cited; the [M1 record](M1_FEATURE_LINEAGE_ABLATION_2026-10-13.md) gives the 108-column lineage and the exploratory block ablation; and the [writing plan](KE_HOACH_VIET_PAPER_THESIS_2026-10-13.md) maps evidence to the paper and thesis. The full-OEH reanalysis of the 14 opened MiniGrid tasks is exploratory and incomplete (55/70 OEH-B groups checkpointed on 13/10); its partial checkpoints are not committed and no score from it is cited.

Use [T1 v1.1.0](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) as the current
one-page definition of the research question, empirical difficulty target,
comparator groups, metrics, and claim limits. The [printable PDF](T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.pdf)
has one A4 page. The [claim map](T1_CLAIM_MAP_2026-10-05.md) links statements to
code, sealed artifacts, and writing locations.

Version 1.0.0 is retained for the record. Team cross-review remains pending;
no reviewer approval is implied by this terminology freeze. Historical code,
protocols, and result artifacts retain their original text and verdicts.

## H1 theory and glossary handover (08/10/2026)

Read [H1 glossary and conditional arguments](H1_THEORY_GLOSSARY_2026-10-08.md),
then the [formula--assumption--code--limit map](H1_FORMULA_CODE_MAP_2026-10-08.md).
They distinguish empirical labels, OEH-B signals, modeled OEH-H and final
forecasts; the active paper's Method and Related Work are aligned with them.
No algorithm, sealed protocol or result is changed.

[Independent block-ablation review](H1_ABLATION_REVIEW_2026-10-08.md) partitions
the 108 sealed columns and flags information retained after a nominal block
removal. Its suggested contrasts await team freeze; no ablation was run.

[B1 pre-review](H1_B1_REVIEW_2026-10-08.md) is a checklist for the forthcoming
Week 8 B1 handover, not approval of an unavailable document. H1's own glossary
and writing deliverables can be handed over first. Human review and the later
B1 cross-review are recorded separately from automated validation.

[H1 validation record and handover hashes](H1_VALIDATION_2026-10-08.json)
records 155 passing tests, 95% branch-aware coverage, evidence/table checks,
and the local environment. LaTeX compilation and the eight-page layout check
remain for a TeX-equipped machine or CI; no rebuilt PDF is claimed.

## BRIDGE transfer and comparator records (07–10/10/2026)

Status as of 10/10/2026 (the expanded panel has since completed; see the 13/10 BRIDGE summary): [quality audit and next research plan](QUALITY_AUDIT_AND_NEXT_RESEARCH_2026-10-10.md), followed by the [evidence and retirement ledger](CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md). They distinguish the completed internal 120-MDP and five-task BRIDGE results from the running five-task prospective extension and the queued 14-task opened-label exploratory reanalysis. The [BRIDGE expanded execution record](BRIDGE_EXPANDED_EXECUTION_2026-10-08.md) and live manifests are the source for progress; an in-progress panel has no result to cite. The [manuscript gate](MANUSCRIPT_UPDATE_GATE_2026-10-09.md) describes the automatic build after strict verification.

The [GVHD problem/data/method evidence audit](GVHD_PROBLEM_DATA_METHOD_EVIDENCE_AUDIT_2026-10-08.md)
maps the advisor's presentation questions to protocol, code, and sealed data;
it includes a checked learner-specific example and the presentation order.

For current comparisons, follow the
[T1 v1.3.0 comparator decision](T1_COMPARATOR_DECISION_v1.3.0_2026-10-08.md):
BWD is excluded because the continuous-action method does not fit this
tabular-discrete protocol. B2 remains an internal control and ERI is a
literature-derived adaptation. The earlier
[v1.2.0 addendum](T1_COMPARATOR_ADDENDUM_v1.2.0_2026-10-07.md) and sealed
seventh-cohort results remain unchanged as historical records.

The [D1 audit and comparator run plan](D1_ORIGINAL_METHODS_AUDIT_AND_RUN_PLAN_2026-10-07.md)
lists original methods reviewed, compatibility gates, owners, and separate
result tables. No original offline-RL forecaster with the exact seven-class
`E_l` endpoint has been identified.

The [published survival comparator audit and results](D5_PUBLISHED_SURVIVAL_COMPARATOR_2026-10-08.md)
records a faithful XGBoost-CF formulation adapted from Kim et al. (2026), run
on the opened 120-MDP panel and sealed before labels on five fresh BRIDGE MDPs.
It also separates a 108-feature architecture control from an independent
published-method adaptation. This does not establish offline-RL SOTA.
The [same-endpoint 120-MDP comparison](SAME_ENDPOINT_COMPARISON_120_2026-10-08.md)
puts B2, full OEH, XGBoost-CF/18 and XGBoost-CF/108 on the same rows and
metrics, separating the 84 stochastic MDPs from 36 controls.

The [completed fresh BRIDGE full-method result](BRIDGE_FRESH_FULL_RESULT_2026-10-08.md)
reports the five-task prelabel seal, 75 completed learner labels, all four
forecaster arms and mixed BC/CFQI/IQL outcomes, including the unfavorable IQL
result. Published BRIDGE MDPs were used with group-generated offline logs and
labels; this is not a published offline-log dataset benchmark.
The [post-outcome failure diagnosis](BRIDGE_FAILURE_DIAGNOSIS_2026-10-08.md)
breaks down the scores by learner and task, traces IQL errors to four early
requirements, and documents the input shift and remaining causal uncertainty.

The [original-comparator execution record](E2_ORIGINAL_COMPARATORS_EXECUTION_2026-10-08.md)
reports ERI, TQ/SACo and author-published BRIDGE EH metadata as scalar-score
rank diagnostics, with hashes, undefined rates, MDP-cluster uncertainty and
method-specific NO-GO decisions. These are not seven-class forecasters or a
claim that OEH surpasses an external author's complete system.

The [OEH versus published-score audit](E3_OEH_VS_PUBLISHED_METHODS_2026-10-08.md)
rechecks the sealed full-OEH predictions against original ERI, TQ and SACo
scores on the 84-MDP stochastic panel, adds the secondary rank table to the
paper, and states the remaining same-endpoint SOTA gap.

The [stochastic-tabular difficulty SOTA audit](D3_STOCHASTIC_TABULAR_DIFFICULTY_SOTA_AUDIT_2026-10-08.md)
checks published performance predictors and sample-complexity work against the
actual `E_l` target. Its new BRIDGE analysis compares the sealed **reduced**
OEH forecast with original ERI, TQ, SACo and author EH descriptors on the
opened 14-task panel; it does not substitute for full OEH on BRIDGE.

The completed [BRIDGE reduced transfer record](E1_EXECUTION_RECORD_2026-10-07.md)
and its [scope note](../../artifacts/week8/bridge_reduced_test_analysis_2026-10-07/INTERPRETATION_SCOPE.md)
compare two **internal** forecasters on published MDPs with team-generated
offline logs and labels. These results do not test an original author's method.

The [second-pass literature review](D1_SECOND_PASS_LITERATURE_2026-10-07.md)
adds the GECCO 2025 dataset-performance predictor and adjacent work on
algorithm choice and scaling. It narrows the novelty claim and updates the
Mateos-style comparator preflight; no new outcomes are claimed.

The [comparator execution plan and ETA](D2_EXECUTION_PLAN_COMPARATORS_2026-10-07.md)
specifies the result tables, preflight gates, run order, measured workload,
and conditional time budget for this machine.


The [local experiment runbook](RUNBOOK_LOCAL_EXPERIMENTS_2026-10-07.md) records
the measured machine resources, preparation commands, and gates for the
controlled ablation and BRIDGE transfer runs. It is a plan, not a result log.

The [local data and ETA audit](LOCAL_EXPERIMENT_ETA_2026-10-07.md) records what
was downloaded, what was timed, and which runtime estimates still depend on
the BRIDGE adapter smoke test.

## B1 and D1 handover update (11/10/2026)

The current [B1 data dictionary and benchmark audit](../data/DATA_WEEK8.md)
was merged in PR #49. Statements above and in the dated 08/10 pre-review about
an unavailable B1 describe that earlier snapshot; the actual B1 is now available.
Its delivery does not imply Hưng/Linh's human approval.

[D1 learner, comparator and metric matrix](D1_LEARNER_COMPARATOR_METRICS_2026-10-11.md)
documents Quốc Việt's seven-learner implementations, fixed optimization budgets,
true-beta/family-label access, the 18 B2 covariates, comparator heads and score
formulas. Its [handover](D1_HANDOVER_2026-10-11.md) links independently checked
primary tables and paper-ready LaTeX fragments in a new namespace. Run
`python -m scripts.analysis.export_week8_d1 --check` to reproduce its checks.
This export does not fit, alter sealed results or grant statistical approval.

## Improvement probes (12/10/2026)

[Exploratory improvement review](IMPROVEMENT_REVIEW_2026-10-12.md) locates
forecast and selector weaknesses, reports two calibration/blending probes and
four selector probes on the already-opened fourth cohort, and sets out the
validation needed before adopting a new candidate. Small development gains
are not new confirmation; the selector probes include negative results.
Frozen candidates, predictions, protocols and verdicts are unchanged.

[Nested experiment results](NESTED_EXPERIMENT_RESULTS_2026-10-12.md) extend those
probes with inner candidate selection, matched-input joint/two-stage heads,
family-held-out adaptation checks, and exploratory simultaneous bounds. The
small grouped-CV gains are uncertain and reverse under the family stress test;
the two-stage head is worse in this setting. The current method is retained.

## M1 feature lineage and matched ablation (13/10/2026)

[M1 record](M1_FEATURE_LINEAGE_ABLATION_2026-10-13.md) assigns every one of the 108
sealed columns to a block, producer, dependency and measured cost, and runs the
five pre-defined arms (B2, full OEH, without direct OEH-B, without OEH-H/risk,
without legacy OEH3/LCOEH4 + context) with the sealed `matched_forecast`
procedure. The full and B2 arms reproduce the sealed seventh-cohort predictions
(≤8.9e-8); an audit hook blocks outcome files until predictions are hashed.
Result: exploratory (opened cohort); the blocks are largely redundant. Runner:
`scripts/analysis/run_m1_matched_ablation.py`; tests:
`tests/test_m1_matched_ablation.py`.

## Paper and thesis revision (13/10/2026)

[Writing plan and project assessment](KE_HOACH_VIET_PAPER_THESIS_2026-10-13.md).
The IEEE paper (`paper/main.tex`, 7 pages) follows the 13/10 IEEE outline; the
thesis v2 (six chapters, school template) is built from
`Drive/Thesis/v2_2026-10-13/`. All manuscript tables and figures are produced by
`scripts/analysis/export_manuscript_tables_20261013.py` into
`artifacts/week8/manuscript_tables_2026-10-13/`.
