# Scientific provenance and availability

`migration.json` records the original Git commit, original-to-archived source
paths, SHA-256 values, retained function/class definitions, selection-history
hashes and unchanged R34/R44 bundle manifest hashes.

`sources/` contains 27 original files required by published source/export/input
seals. They are byte-identical provenance snapshots, not installed or executed
by CI. Historical runner and test snapshots are present only when a retained
scientific manifest directly seals them. `selection_history/` retains 34 minimal
attempt records so adaptive E1 reuse remains traceable. Discarded research
prototypes and failed-run data are not retained as active artifacts.

The active package rewires imports and extracts only the functions it needs.
`verify_oeh` compares each retained scientific definition's entire AST, including body,
signature and defaults, against the frozen original. Three fixed display aliases
allow the new figure copies to use the public name **OEH**; numerical rendering
logic remains checked, and the archived figures retain their original bytes. Independent numerical,
pairing and held-out-label controls also run in CI.

## Data available in Git

- R44 raw report, full bootstrap arrays, choice seals, log/config/manifest and
  exported group/MDP/seed metrics and figures.
- R34 reference evidence needed to verify the original identities and contrasts.
- Original E1 input seal, compact execution/diagnostic evidence, baseline
  provenance and corrected S1 verdict, including its negative conclusion.
- Archived T1/R34/R44 protocols and original mathematical source seals.
- `validation/` records previous validation at their stated commits and the
  packaging reproducibility check. Past CI results do not certify a later head.

The 17 original E1 inputs are declared in the unchanged R34 manifest. Large
raw CSVs, the dataset manifest and NPZ training buffers remain local under their
original paths in `artifacts/week6/oehv4e_confirmatory_v1_1/` and
`datasets/week6_e1_oehv4e_confirmatory_v1_1/`. The original T1 input resolves to
its archived byte-identical path through the migration record. They were not
regenerated, deleted or uploaded during cleanup.

## Audit levels

`python -m scripts.analysis.verify_oeh` audits all published evidence and
whichever original inputs are available, explicitly reporting the count.
`--full-inputs` additionally fails unless all 17 original input files are present
and match their frozen hashes. It does not claim to read every NPZ buffer;
selected seed records preserve the original buffer identity hashes.

A fresh clone therefore supports the complete published audit and software
tests, but cannot repeat original policy training without the local inputs and
pinned external implementations. The pre-cleanup implementation remains in Git
at `e715e3c3e8e0565ef4d747e74326c807a5cec0d5`. One-shot E1 is consumed;
packaging verification is not a new evaluation or an independent confirmation.
