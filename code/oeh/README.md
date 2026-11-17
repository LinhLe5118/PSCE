# OEH operational package

The [method analysis](README_METHOD.md) explains the design choices, mathematical
objective, empirical mechanisms and current generalization limits in Vietnamese.

`router.fit_routes(panel, mdps, multiplicities=None)` returns group choices,
paired forecasts and training metadata. It uses the supplied whole-MDP five outer
folds, four grouped inner folds, the original ridge grid, quadratic kernel and
equal ensemble of two feature blocks. Each head predicts mean paired return
and win/loss/tie frequencies versus CFQI; predicted success gain is `p(win)-p(loss)`.
The supported deterministic frontier maximizes the product of aggregate positive
predicted gains; if no such plan exists, the route falls back to CFQI.

The three core heads retain their original mathematical definitions. All additional
registered learners use the same target and forecaster, with concatenated pilot
diagnostics from the three core anchors. `core.py` and `reference.py` are internal
dependencies of the final system, not competing active research versions.

## Input JSON

The CLI accepts a JSON object with this structure (placeholder values below are
schema notation, not generated research data):

```json
{
  "schema_version": 1,
  "methods": ["cfqi_tabular", "iql_tabular", "behavior_cloning_tabular", "REGISTERED_ADDITIONAL_METHODS"],
  "return_ridges": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
  "groups": [{
    "mdp_id": "MDP_ID",
    "policy_family": "BEHAVIOR_ID",
    "panel": "PANEL_NAME",
    "outer_fold": 0,
    "v4e_algorithm": "cfqi_tabular",
    "features": {
      "cfqi_tabular": {"B2_OEH3": [0.0], "B2_OEH3_LCOEH4": [0.0]},
      "iql_tabular": {"B2_OEH3": [0.0], "B2_OEH3_LCOEH4": [0.0]},
      "behavior_cloning_tabular": {"B2_OEH3": [0.0], "B2_OEH3_LCOEH4": [0.0]}
    },
    "outcomes": {
      "EACH_REGISTERED_METHOD": [{"seed": 0, "normalized_return": 0.8, "success": false}]
    }
  }]
}
```

Each actual group must include all registered methods and exactly seeds 0–9.
Success must equal `normalized_return >= 0.95`. Each panel needs all five outer
folds, and all behaviors of an MDP must share its panel and fold. Feature vectors
must be finite and dimensionally consistent; their coordinate definitions are
the original [T1 feature contract](../../artifacts/provenance/sources/docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json).
The listed ridge grid is the complete original T1 grid, not a suggested sweep.
The original E1 library appends the four external methods in sorted registry
order. Group IDs are used for splits and joins, never as numerical features.

## Operational limits

This API performs batch cross-fitted routing with a labeled meta-training
library. It does not provide an independently validated pretrained router for
unseen tasks or an online, per-task deployment model. The feature extraction and
seven policy-training runs were performed in original E1; this package consumes
their existing outputs. Their configuration, implementations and availability
are recorded in provenance and Git history.

`score_routes` requires comparator decisions and complete matched outcomes.
Scoring is separate from fitting. Query labels in an outer fold cannot affect
that fold's forecasts. The CLI writes decisions and model metadata without
scoring test returns, and it does not overwrite immutable research evidence.
