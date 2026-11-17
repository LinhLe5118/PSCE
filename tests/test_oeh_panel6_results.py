"""Panel 6 registered analysis (after H3): the recorded gates are reproducible and outcomes postdate every seal."""
from __future__ import annotations

import json

import pandas as pd

from scripts.analysis import run_oeh_panel6_20261015 as S6

E = S6.E


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_outcomes_follow_the_seals_and_are_recorded():
    for panel, spec in S6.PANELS.items():
        target = S6.panel_dir(panel)
        run = _load(target / "outcome_run.json")
        assert run["preoutcome_seal_sha256"] == E.sha256(target / "preoutcome_seal.json")
        assert run["labels_sha256"] == E.sha256(target / "labels.csv")
        assert run["completed_at_utc"] > _load(target / "preoutcome_seal.json")["sealed_at_utc"]
        assert run["jobs"] == spec["mdps"] * 5 * len(S6.EXTENDED_BUDGETS) * 10 * 3
        assert len(pd.read_csv(target / "labels.csv")) == spec["mdps"] * 5 * 3


def test_recorded_gates_are_reproduced_from_the_scored_rows():
    result = _load(S6.OUTPUT / "result.json")
    assert result["protocol_sha256"] == E.sha256(S6.OUTPUT / "protocol.json")
    s = pd.read_csv(S6.OUTPUT / "scored.csv.gz")
    metrics = ["ibs", "ll", "ibs_ge", "ibs_gt", "ll_ext", "regret"]
    per = s.groupby(["source_panel", "method", "mdp_id"])[metrics].mean().reset_index()
    wide = per.pivot_table(index=["source_panel", "mdp_id"], columns="method", values=metrics)
    for name, contrasts in {**S6.GATES, **S6.REGISTERED_ESTIMATE}.items():
        recorded = result["gates"].get(name) or result["registered_estimates"][name]
        again = S6.family_test(wide, contrasts)
        assert again["mdps"] == recorded["mdps"] == 192
        for a, b in zip(again["contrasts"], recorded["contrasts"]):
            assert abs(a["mean_improvement"] - b["mean_improvement"]) < 1e-12
            assert abs(a["simultaneous_lower_95"] - b["simultaneous_lower_95"]) < 1e-12
    assert list(result["gates"]) == ["F1", "F2"] and "verdict" not in result["registered_estimates"]["F3"]
