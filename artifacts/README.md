# Artifacts: which folders the final pipeline uses

Every experiment of the project wrote to its own dated folder here, and sealed files are never changed or deleted
(`tests/test_sealed_integrity.py` checks every hash recorded in a seal or manifest). This index says which folders
belong to the final method and its three final tests; everything else is development history.

## Final tests (read by `python -m scripts.reproduce`)

| Folder | Content |
|---|---|
| [`oeh_v2_sealed_test_2026-10-14`](oeh_v2_sealed_test_2026-10-14/) | **Test 1**: 168 MDPs of the project generator (`internal`), 72 Colosseum (`colosseum2029`), 108 SPIBB (`spibb2`), 48 forest (`forest`); protocol, pre-outcome amendment 1, seals, forecasts, labels, `result.json` of the three registered steps |
| [`oeh_fifth_panel_2026-10-15`](oeh_fifth_panel_2026-10-15/) | **Test 2**: 36 Gymnasium FrozenLake, 36 gambler's problem; one registered contrast, NO-GO |
| [`oeh_panel6_2026-10-15`](oeh_panel6_2026-10-15/) | **Test 3**: 96 MDP Playground (`mdpp`), 96 labmaze; amendment A1; registered families F1, F2 (GO) and the F3 estimate; TabPFN forecasts sealed before the labels |
| [`week8/oeh_v2_development_2026-10-14`](week8/oeh_v2_development_2026-10-14/) | Frozen specification of OEH, `oeh_v2_spec.json` (heads used by the gate, its three parameters) |
| [`week8/baseline_tabpfn_t3_2026-10-15`](week8/baseline_tabpfn_t3_2026-10-15/) | TabPFN v2 forecasts for Tests 1 and 2, scored after the outcomes were known (descriptive) |

Each test folder has the same layout: `protocol.json` and `protocol.sha256` (frozen before the data existed), one
folder per source with `query_preoutcome.csv.gz` (pilot-only inputs), `predictions_preoutcome.csv` (all forecasts),
`preoutcome_seal.json` (hashes written before any learner outcome), `outcomes.csv.gz` and `labels.csv` (opened
afterwards), `outcome_run.json` (links the labels to the seal) and `oeh_b_groups/` (learner-simulation checkpoints);
at the top, `scored.csv.gz`, `summary.csv` and `result.json` from the registered analysis.

## Meta-training inputs (read when the heads are fitted)

The heads are fitted on 720 MDPs (10,800 cases, 7,560 of them stochastic) assembled by
`scripts.analysis.oeh_pipeline.training_panel` from these folders:

`oeh_final_development_2026-10-01` (simulation and planning-risk inputs of the opened cohorts),
`oeh_final_unseen_120_2026-10-02`, `oeh_confirmatory_unseen_120_2026-10-04`, `oeh_difficulty_unseen_120_2026-10-01`,
`oeh_pooled_successor_unseen_120_2026-10-01`, `oeh_cross_capacity_unseen_120_2026-10-01`,
`oeh_difficulty_confirmation_2026-10-01`, `oeh_b2_eh_oracle_second_development_2026-10-01`,
`independent_difficulty_2026-09-30`, `independent_validation_2026-09-30`, `difficulty_e1_posthoc_2026-09-30`,
`e1_oracle_qvi_posthoc_2026-09-30`, `week6/oehv4e_confirmatory_v1_1`.

These cohorts were opened during development; their labels train the heads and are never used to score the final
tests.

## Development history

All other folders: the earlier sealed 120-MDP tests (including the ones that failed), the candidates tried on opened
cohorts (`*_development_*`, `*_posthoc_*`), the eighth cohort, the SPIBB and Colosseum panels, BRIDGE, the published
FourRooms calibration and the routing study of weeks 6–7 (`week6`, `week7`, `provenance`). They are kept unchanged:
the evidence rules forbid deleting tried candidates, failed runs or negative results. Their decisions are recorded
in [docs/week8](../docs/week8/README.md); the route from these candidates to the final method is in
[OEH_V2_DEVELOPMENT_2026-10-14.md](../docs/week8/OEH_V2_DEVELOPMENT_2026-10-14.md) and the ledger of what may be cited
in [CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md](../docs/week8/CURRENT_EVIDENCE_AND_RETIREMENT_2026-10-09.md).

An earlier version of this file described the routing evidence of September 2026 (OEH-v5 routing on E1); that
version is in the git history, and its bundle is
[`week7/oeh_v5_e1_reuse_r44_2026-09-29`](week7/oeh_v5_e1_reuse_r44_2026-09-29/manifest.json).
