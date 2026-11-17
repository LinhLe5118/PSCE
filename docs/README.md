**Start here (11/11/2026):** the final documentation is in [`final/`](final/README.md): the method, the data, the three final tests and the route that led to them. Everything else in this folder, including the index below, is the development record. It is kept unchanged because the evidence rules forbid deleting tried candidates, failed runs or negative results, and many of these files are hash-recorded in protocols and manifests.

# Project documentation

For the current difficulty-aware requirement-forecasting question, start with the
[one-page Week 8 T1 protocol](week8/T1_DIFFICULTY_AWARE_PROTOCOL_v1.1.0_2026-10-05.md)
and its [claim map](week8/T1_CLAIM_MAP_2026-10-05.md). Difficulty is an empirical,
learner-specific data requirement conditional on MDP, behavior, and threshold;
OEH-H is a modeled descriptor, not that outcome or a proved sample bound.
The x10 selector is a secondary study. Dated Week 1–6 and v3/v4E documents below
retain their original experimental meaning and are not the current claim source.


- [H1 formula/theory freeze v1.0.0](week6/H1_FORMULA_THEORY_FREEZE_v1.0.0.md):
  current v3/v4E equations, feature dictionary, proofs, support/fallback rules
  and checksum handover. H1 does not grant evaluation execution GO.

For Week 6, start with the [T1 statistical freeze](protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0_2026-09-21.md)
and [current project audit](research/PROJECT_READINESS_AUDIT_2026-09-21.md).
`PLAN_Final.md` belongs to an older repository snapshot and is not a current
source file; newer dated decisions supersede it within their stated scope. Documents are
grouped by purpose:

- [`design/`](design/) — benchmark, dataset, and environment design.
- [`protocol/`](protocol/) — dated protocol decisions and deviations.
- [`week1/`](week1/) — Week 1 validation and audit records.
- [`week2/`](week2/) — Week 2 protocol, contracts, plan, and backlog.
- [`week5/`](week5/) — Week 5 development handoff records.
- [`week6/`](week6/) — Week 6 protocol-linked execution handoffs. Start with
  [D1 baseline provenance and unified-runner preflight](week6/D1_BASELINE_PREFLIGHT.md)
  for the completed development-only baseline package and its E1 boundary.
- [`week8/`](week8/) — current T1 terminology protocol and claim-to-evidence map.
- [`research/`](research/) — research administration and exploratory method designs.
  [OEH redesign for stochastic tabular RL (18/09/2026)](research/OEH_REDESIGN_2026-09-18.md)
  defines a regret-based candidate, its confidence assumptions, prototype, and limitations.
  [Development smoke test](research/OEH_DEVELOPMENT_SMOKE_2026-09-18.md)
  records the integration and the negative result on 225 pilot cases.
  [KL confidence and separate policy proposals](research/OEH_REFINEMENTS_2026-09-18.md)
  records the completed four-way refinement experiment, literature, and remaining
  saturation and selection-utility limitations.
  [Selective pessimism and coupled policy comparisons](research/OEH_SELECTIVE_METHODS_2026-09-18.md)
  documents the SPVI adaptation, discounted difference bounds, and LP-MDP
  support audit after reading the method definitions and pseudocode.
  [EH-aligned stochastic-tabular OEH](research/OEH_EH_ALIGNED_2026-09-19.md)
  derives the gap and empirical-Bernstein formulas, handles unresolved ties by
  abstention, records the 225-pilot development audit, and links the complete
  mechanical-saturation audit plus delayed-reward positive control.
  [Final OEH formulation](research/OEH_FINAL_FORMULATION_2026-09-19.md)
  separates the universal information-theoretic target from its cross-fitted
  Bernstein estimator and keeps `(depth, data requirement)` as the primary
  profile instead of forcing one offline scalar.
  [OEH v2 final development audit](research/OEH_V2_FINAL_AUDIT_2026-09-20.md)
  records the target-aligned future-requirement estimator, the explicit
  `epsilon_trivial` status, the non-saturated 225-pilot and 810-cell results,
  and the still-negative B3 utility gate.
  [Final development results](research/FINAL_DEVELOPMENT_RESULTS_2026-09-20.md)
  consolidates the final OEH audit, B0--B3 gates, full tabular reruns, CUDA
  neural panel, and paired all-method comparison.
  [Repository-wide research completion audit](research/RESEARCH_COMPLETION_AUDIT_2026-09-19.md)
  records the 22,725-row cross-fitted profile, B0--B3 nested OOF and selection,
  the SPIBB/VI-LCB/BCQ/CQL panel, the 810-cell controlled factorial, its negative
  development gates, and the remaining scientific and submission work.
  [External baseline audit](research/EXTERNAL_BASELINE_AUDIT_2026-09-19.md)
  records repository commits, compatibility decisions, adapters and GPU scope.
  [Controlled factorial protocol](research/CONTROLLED_STOCHASTIC_FACTORIAL_PROTOCOL_2026-09-19.md)
  freezes the matched slip × coverage × budget probe.
- [`thesis/`](thesis/) — synchronized Markdown drafts for Chapters I–VII.
- [`archive/`](archive/) — historical context retained for provenance, not as a current source of truth.

Generated evidence and run outputs belong under [`../artifacts/`](../artifacts/).
