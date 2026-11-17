"""Simplified candidates S1-S4 (task card T4): selection rule and recorded result."""
from __future__ import annotations

import json

import pandas as pd

from scripts.analysis import simplified_candidates_t4 as T


def _table(rel: dict[str, float]) -> dict[str, pd.DataFrame]:
    panels = [*T.INTERNAL, *T.EXTERNAL]
    base = pd.DataFrame({"panel": panels, "mdp": ["m"] * len(panels), "ibs": 1.0, "ll": 1.0})
    out = {"OEH": base}
    for c in T.ORDER:
        out[c] = base.assign(ibs=1 + rel[c], ll=1 + rel[c])
    return out


def test_rule_picks_the_simplest_acceptable_and_the_best_external():
    picks = T.rule(_table({"S1": 0.05, "S2": 0.005, "S3": 0.002, "S4": -0.01}), T.INTERNAL, T.EXTERNAL)
    assert picks == ["S2", "S4"]
    assert T.rule(_table({"S1": 0.02, "S2": 0.02, "S3": 0.02, "S4": 0.02}), T.INTERNAL, T.EXTERNAL) == []
    assert T.rule(_table({"S1": 0.0, "S2": 0.5, "S3": 0.5, "S4": 0.5}), T.INTERNAL, T.EXTERNAL) == ["S1"]


def test_recorded_result_and_manifest():
    out = T.OUTPUT
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["sealed_oeh_reproduction_max_abs_diff"] < 1e-9
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive"
    for name, digest in manifest["outputs_sha256"].items():
        assert T.sha(out / name) == digest
    lopo = pd.read_csv(out / "nested_lopo.csv")
    assert len(lopo) == len(T.INTERNAL) + len(T.EXTERNAL)
