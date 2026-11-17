# Controlled stochastic factorial protocol — 19 September 2026

This development-only experiment closes the design confound identified in the
repository audit. It does not access or replace the sealed 120-MDP evaluation
split.

Each matched block fixes corridor topology and reward using one of six
`chain_length × decoy_fraction` pairs. Inside every block the experiment varies
only:

- transition slip: `{0, 0.05, 0.20}`, including a deterministic control;
- behavior probability of the decoy action: `{0.05, 0.20, 0.45}`;
- nested transition budget: `{2, 10, 50}|S||A|`;
- five registered generation replicates.

The resulting design has 810 cells. The largest buffer is collected once for
each block/slip/coverage/replicate stream and smaller budgets are prefixes, so a
budget contrast does not change the underlying random stream. The block ID is
the required fold/bootstrap cluster in any later predictive use.

For each cell the runner reports oracle-side changes in optimal/random return,
dynamic range and optimal action gap only for interpretation; these quantities
are never pilot features. Data-only outputs are B1/B2 diagnostics, the
cross-fitted EB-OEH status/profile, and CFQI return under exact common scoring.
Primary mechanism contrasts are matched differences for high versus zero slip,
balanced versus narrow action coverage, and budget 50 versus 2. They are
descriptive because six base blocks are not enough for a broad causal claim.

The factor grid, seeds, outcome fields and source hashes are written to
`artifacts/research/controlled_stochastic_factorial_2026-09-19/plan.json` before
row generation. No cell is filtered based on its result.
