# Week 2 backlog — scientific evaluation

> **SUPERSEDED — 2026-08-25.** `WEEK2_DECISION_GATE.md` §11 was ratified: the
> full 120-MDP evaluation grid is **not** to be run under the current
> definition, so item 1 below is now explicitly forbidden rather than pending.
> The live plan is [`../../PHASE_B_PLAN.md`](../../PHASE_B_PLAN.md). This file is
> kept only because dated artifacts cite it; do not take instructions from it.

This backlog is not authorization to start the official benchmark. Begin only
from a reviewed commit whose protocol is frozen and whose
[`../week1/WEEK1_AUDIT.md`](../week1/WEEK1_AUDIT.md) records no blocker.

1. Run the official evaluation benchmark under the frozen config.
2. Generate offline datasets under the frozen protocol.
3. Run OEH and compute exact references.
4. Compare OEH with reference, including coverage-group and behavior-family
   dependence.
5. Evaluate the registered OOD environments.
6. Generate pre-specified main figures and tables.
7. Perform failure analysis and only pre-specified ablations.

Do not change benchmark composition, behavior-policy definitions, coverage
thresholds, adaptive-pilot selection, bootstrap method, `K_max`, or estimator
semantics while carrying out these tasks. Any newly discovered correctness bug
requires a recorded protocol deviation, version increase, and rerun of the
relevant Week 1 validation gate before evaluation resumes.
