# Contributing

Use focused changes and conventional commits: `feat:`, `fix:`, `docs:` or `chore:`.
Describe the resulting behavior and validation in each PR. At least one reviewer
and passing CI are required before merging. Existing authorized work on `linh`
may continue on that branch.

## Code and checks

Core modules belong in `code/oeh/`. CLI entry points belong in
`scripts/analysis/` or `scripts/testing/` and run from the repository root with
`python -m`. Do not introduce root Python scripts or recreate historical runners.
New CLI behavior needs a fresh-process startup test and appropriate functional
checks. Use data-only covariates; never pass oracle dynamics or held-out labels
into a forecast. Keep all behaviors and bootstrap copies of an MDP together.

Run the commands in the root README before proposing a change. The full suite
must have no skips and must satisfy the 82% branch-aware coverage threshold.
Cover operational code rather than counting archived provenance as executable
code or lowering thresholds to obtain a green build.

## Scientific provenance

OEH is the public name of the final retained method. Its published bundles are
immutable. `artifacts/provenance/migration.json` maps original source paths to
byte-identical archived files and lists the unchanged operational definitions.
Do not edit original manifests to hide path changes or remove negative outcomes.
Do not change archived bytes, salts, tie rules, feature blocks, tuning losses,
objectives or seed schedules as a cleanup fix.

A scientific change requires a separately declared revision/protocol and honest
evidence scope. Never relabel adaptively reused E1 results as independent
confirmation. Keep minimal selection history so unsuccessful prior choices are
not silently erased. Store local raw inputs outside Git's published evidence;
uploading them needs explicit authorization for those payloads.

Use **OEH** in public descriptions and new figure copies. Historical identifiers
in sealed files stay unchanged. The verifier permits only three hardcoded display
aliases in the original figure renderer; it still rejects arithmetic changes and
does not permit presentation exceptions in the forecasting or routing functions.
