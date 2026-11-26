# How the final method was reached

Every earlier version, sealed test and amendment stays in the repository (the evidence rules forbid deleting tried
candidates, failed runs or negative results). This page gives the route in order and points to the record of each
step; the records themselves are partly in Vietnamese.

| When (2026) | Step | Outcome | Record |
|---|---|---|---|
| August – September | A single offline effective horizon per pilot, computed from the pilot | Saturated on the first development data: every pilot looked sufficient, so problems could not be separated. The derivation behind it was partly withdrawn in an audit | [OEH_DEVELOPMENT_SMOKE](../research/OEH_DEVELOPMENT_SMOKE_2026-09-18.md), [DERIVATIONS](../DERIVATIONS.md) |
| Late September | Learner routing on the E1 dataset (OEH-v5 router) | Replaced; kept as the earlier study | [`artifacts/week7`](../../artifacts/week7/), [restored runtime](../RESTORED_RUNTIME.md) |
| 17 September | Pivot: forecast the measured, learner-specific data requirement instead of one number per MDP; keep the offline horizon as one input (the later lookahead-budget block) | Question defined in the T1 protocol | [FINAL_RESEARCH_DIRECTION](../FINAL_RESEARCH_DIRECTION_2026-09-17.md), [T1 protocol](../week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md) |
| 1 – 4 October | Five sealed tests of forecaster versions on 120 new MDPs each (cohorts 3–7) | Cohorts 3–6 NO-GO; cohort 7 GO. The opened cohorts later became meta-training data | `artifacts/oeh_difficulty_unseen_120_2026-10-01`, `oeh_pooled_successor_unseen_120_2026-10-01`, `oeh_cross_capacity_unseen_120_2026-10-01`, `oeh_final_unseen_120_2026-10-02`, `oeh_confirmatory_unseen_120_2026-10-04` |
| 4 – 13 October | Transfer to published environments: FourRooms (published data, transition component only) and BRIDGE (14 + 5 + 5 MDPs) | FourRooms NO-GO; BRIDGE mixed by learner | [BRIDGE results](../week8/BRIDGE_ALL_RUNS_RESULTS_2026-10-13.md), `artifacts/published_fourrooms_calibration_2026-10-04` |
| 13 – 14 October | Eighth cohort and external panels with the earlier forecaster: 168 generator MDPs, 72 Colosseum, 108 SPIBB | Generator steps 1–2 GO; Colosseum step 1 GO, step 2 NO-GO; SPIBB step 1 NO-GO | [COHORT8_SPIBB_DEVELOPMENT](../week8/COHORT8_SPIBB_DEVELOPMENT_2026-10-14.md), [COLOSSEUM_EXTERNAL_PANEL](../week8/COLOSSEUM_EXTERNAL_PANEL_2026-10-14.md) |
| 14 October | Development on the opened panels, with rules written before any fit: M1 and M4 (together, the shift-gated combination of three heads) selected; M6 (a simpler head), M2 (a new reward prior) and M3 (a cohort with new behavior policies) not adopted; the specification frozen | Final method (OEH) | [OEH_V2_DEVELOPMENT](../week8/OEH_V2_DEVELOPMENT_2026-10-14.md) sections 1–3 |
| 14 October | **Test 1** (four new panels) | GO in all three steps | [FINAL_TESTS](FINAL_TESTS.md), [OEH_V2_DEVELOPMENT](../week8/OEH_V2_DEVELOPMENT_2026-10-14.md) section 4 |
| 15 October | **Test 2** (fifth panel) | NO-GO against the robust head; the advantage over that baseline is therefore claimed as source-dependent | [OEH_V2_DEVELOPMENT](../week8/OEH_V2_DEVELOPMENT_2026-10-14.md) section 5.3 |
| 15 – 16 October | **Test 3** (panel 6), new questions only; rescored on a second machine (H4) | F1 GO, F2 GO; rescoring matched | [T8 protocol](../week8/T8_PANEL6_PROTOCOL_DRAFT_2026-10-15.md), [T9 sealing](../week8/T9_PANEL6_SEALED_2026-10-15.md), [T10 results](../week8/T10_PANEL6_RESULTS_2026-10-16.md), [H4](../week8/H4_PANEL6_RESCORE_2026-10-16.md) |
| 15 October | Post hoc checks on opened panels (extrapolation endpoint, prefix + FQE and TabPFN baselines, nested LOPO, threshold sensitivity) | Descriptive only | `docs/week8/T1_EXTRAPOLATION_ENDPOINT…`, `T2_PREFIX_FQE…`, `T3_TABPFN…`, `T5_NESTED_LOPO…`, `T6_THRESHOLD_SENSITIVITY…` |
| 28 October – 1 November | Post hoc ablation, net-cost, learner-level and calibration analyses for the thesis | Descriptive only; recomputed by `python -m scripts.reproduce` | `scripts/analysis/posthoc_20261028/`, `scripts/analysis/posthoc_20261101/` |

Names used in these records differ from the final ones: OEH-B is sampled-MDP training (earlier: learner simulation),
OEH-H the lookahead budget (earlier: effective horizon, planning risk), "legacy" diagnostics or D the pilot learning curve (earlier:
pilot diagnostics), B2 the Baseline, OEH-F or `oeh_final` the full head, B2 + legacy the robust head, and
OEH v2 the final OEH. The full mapping is in the [root README](../../README.md#names-in-the-code-and-in-the-writing).
