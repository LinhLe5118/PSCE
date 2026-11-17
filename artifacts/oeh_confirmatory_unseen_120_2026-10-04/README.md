# Seventh sealed OEH cohort, 2026-10-04

This cohort has 120 kernel-disjoint MDPs: 84 stochastic primary and 36 deterministic controls. Its protocol, generated dataset manifest, forecasts, and learner choices were sealed before any seventh-cohort learner outcome. The protocol SHA-256 is `c7d6a9ec3900c2113a04c311e35ce294bf9ba6d0fa5250e07f514bfbae995a18`. The pre-outcome seal is `preoutcome_seal.json`. Archived protocol versions record the pre-outcome schema, kernel-seed, and reused-text-ID/selector checks.

## Forecast verdict

All 108,000 core learner jobs were verified. On 84 stochastic MDP clusters, OEH scored 0.045989 integrated Brier and 0.563079 seven-class log loss, versus 0.054727 and 0.627179 for B2 matched. The simultaneous one-sided 95% lower bounds on absolute improvement are +0.002749 and +0.029810. All 22 registered forecast contrasts have positive lower bounds; the smallest is +0.001006 Brier versus the privileged TQ/SACo augmentation. Tier 1, Tier 2, and the combined seventh-cohort forecast gate are **GO**. This protocol requires positive lower bounds with no 5% minimum. `result_manifest.json`, `summary.csv`, and `contrasts.csv` are the compact result sources.

The fourth cohort remains separate: its Tier 2 and combined gates failed under its own sealed protocol. The seventh cohort tests new kernels within known generator families, not a new environment family or a stochastic sample-complexity theorem.

## Selection and external data

The x10 learner selector was fixed from opened data before seventh-cohort outcomes. All 16,800 external jobs were audited: 4,200 each for BCQ, CQL, SPIBB, and VI-LCB. Against fixed CFQI, the registered two-endpoint gate is **GO** over 84 MDP clusters: success increases by 0.066667 (simultaneous lower +0.042619) and normalized return by 0.018656 (lower +0.010341). The chosen rule selects CFQI in 306, BCQ in 104, and CQL in 10 of 420 MDP-behavior groups. The supplementary v5 comparison is **NO-GO**: return difference −0.001817 and success difference +0.006190 have lower bounds crossing zero. See `external_execution_audit.json`, `confirmatory_verdict.json`, `selection_summary.csv`, and `selection_contrasts.csv`.

The independently published Farama Minari FourRooms dataset is tested separately in `../published_fourrooms_calibration_2026-10-04`. Its pre-registered Dirichlet transition-prediction component gate is **NO-GO**. Partial observations place that test outside the full tabular OEH requirement-forecast claim.

## Audit and reproduction

`protocol.json` and `dataset_seal.json` pin the inputs; `preoutcome_seal.json` pins forecast and choice files. `execution_complete.json` verifies the 108,000 core jobs. Raw buffers, expanded plans, and individual jobs are retained locally and omitted from Git because of size. The repository clone supports checking compact outputs but cannot rerun the full training grid without the local dataset.

## Current terminology clarification (2026-10-05)

The sealed `protocol.json` and its hashes remain unchanged. Its
`oeh_final.forecaster` metadata says the same fitted head was used for every
fitted comparator, which is broader than the actual runner. In
[`all_forecasts`](../../scripts/analysis/run_oeh_final_120.py), OEH, B2 matched,
B2+ERI, B2+BWD, and other `FIELD_SETS` rows use the matched three-part head;
generic B2 and coverage/quality adaptations use AFT only; `train_frequency`
is unfitted. This is a wording error in frozen metadata, not a change to scores
or verdicts. See the [current T1 protocol](../../docs/week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md)
and [claim map](../../docs/week8/T1_CLAIM_MAP_2026-10-05.md).

The final `x10` selector is a separate ExtraTrees study with a synthetic MDP
family indicator among its inputs; the older posterior-draw router is not the
final selection rule.
