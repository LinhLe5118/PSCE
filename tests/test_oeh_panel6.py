"""Panel-6 runner (task cards T8/T9): registered design and analysis helpers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_panel6_20261015 as S6


def test_design_follows_the_h1_decisions():
    draft = json.loads(S6.DRAFT.read_text(encoding="utf-8"))
    assert set(S6.PANELS) == set(draft["families"])
    assert all(S6.PANELS[f]["mdps"] == draft["families"][f]["mdps"] == 96 for f in S6.PANELS)
    gated = [c for v in S6.GATES.values() for c, _ in v]
    assert "robust" not in gated and list(S6.GATES) == ["F1", "F2"]  # panel-5 contrast not re-tested; F3 is not a gate
    assert [m for _, m in S6.REGISTERED_ESTIMATE["F3"]] == ["regret", "regret"]
    assert "s4" in S6.DESCRIPTIVE_METHODS and "lc_pfn" not in S6.DESCRIPTIVE_METHODS
    assert S6.BOOTSTRAP["seed"] == draft["gates"]["bootstrap"]["seed"]
    assert S6.CANDIDATE_SEEDS == {f: draft["families"][f]["candidate_seed"] for f in S6.PANELS}
    assert set(S6.DATA_SEED_BASE) == set(S6.PANELS)  # bases are checked (and were moved) at freeze


def test_family_test_detects_a_positive_effect_and_not_a_null_one():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([["mdpp"], [f"m{i}" for i in range(150)]], names=["source_panel", "mdp_id"])
    base = rng.normal(1.0, 0.2, 150)
    cols = {("ibs_ge", "oeh"): base, ("ll_ext", "oeh"): base, ("ibs_ge", "b2_matched"): base + 0.1 + rng.normal(0, 0.1, 150),
            ("ll_ext", "b2_matched"): base + rng.normal(0, 0.1, 150)}
    wide = pd.DataFrame(cols, index=idx)
    good = S6.family_test(wide, [("b2_matched", "ibs_ge")])
    assert good["all_lower_bounds_positive"] and good["mdps"] == 150
    assert not S6.family_test(wide, [("b2_matched", "ibs_ge"), ("b2_matched", "ll_ext")])["all_lower_bounds_positive"]


def test_label_rules_match_the_registered_event_definitions():
    from scripts.analysis.compare_difficulty import suffix_event

    assert S6.extended_event((10, 10, 10, 10, 10, 10, 10, 10)) == 0
    assert S6.extended_event((0, 9, 7, 9, 9, 9, 9, 9)) == 3
    assert S6.extended_event((10,) * 7 + (7,)) == 8
    rng = np.random.default_rng(1)
    rows = [{"mdp_id": f"m{m}", "policy_family": "b", "algorithm": "a", "budget": b, "training_seed": s,
             "normalized_return": float(rng.uniform(0.9, 1.0))}
            for m in range(20) for b in S6.EXTENDED_BUDGETS for s in range(10)]
    outcomes = pd.DataFrame(rows)
    strict = S6.strict_event(outcomes, 0.95).set_index("mdp_id").event
    for m, g in outcomes[outcomes.budget.isin(S6.EXTENDED_BUDGETS[:6])].groupby("mdp_id"):
        counts = tuple(int((g[g.budget == b].normalized_return >= 0.95).sum()) for b in S6.EXTENDED_BUDGETS[:6])
        assert strict[m] == suffix_event(counts)
    assert (S6.strict_event(outcomes, 0.99).event >= strict.reindex(S6.strict_event(outcomes, 0.99).mdp_id).to_numpy()).all()


def test_generated_seed_collection():
    manifest = {"instances": [{"seed": 5}], "datasets": [{"mdp_seed": 5, "policy_seed": 7, "stream_seed": 9}, {"stream_seed": None}]}
    assert S6.generated_seeds(manifest) == {5, 7, 9}
