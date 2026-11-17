"""Panel 6 sealed forecasts (checkpoint H2): every seal matches its files and precedes any outcome."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import run_oeh_panel6_20261015 as S6

E = S6.E


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_seals_match_the_frozen_protocol_and_their_files():
    protocol_sha = E.sha256(S6.OUTPUT / "protocol.json")
    assert protocol_sha == (S6.OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip()
    for panel, spec in S6.PANELS.items():
        target = S6.panel_dir(panel)
        dataset, seal = _load(target / "dataset_seal.json"), _load(target / "preoutcome_seal.json")
        assert dataset["protocol_sha256"] == seal["protocol_sha256"] == protocol_sha
        assert dataset["learner_outcomes_observed"] == seal["learner_outcomes_observed"] == 0
        assert dataset["instances"] == spec["mdps"]
        for name, digest in seal["artifact_sha256"].items():
            assert E.sha256(target / name) == digest
        selection = pd.read_csv(target / "selection.csv")
        assert selection.selected.sum() == spec["mdps"] and selection[selection.selected].signature.is_unique


def test_forecasts_are_complete_seven_class_laws_for_every_registered_method():
    for panel, spec in S6.PANELS.items():
        pred = pd.read_csv(S6.panel_dir(panel) / "predictions_preoutcome.csv")
        assert set(pred.method) == set(S6.DESCRIPTIVE_METHODS)
        assert (pred.groupby("method").size() == spec["mdps"] * 5 * 3).all()
        p = pred[S6.P].to_numpy(float)
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-9)


def test_tabpfn_is_sealed_after_the_main_forecasts_and_before_any_outcome():
    for panel in S6.PANELS:
        target = S6.panel_dir(panel)
        tab = _load(target / "tabpfn_seal.json")
        assert tab["preoutcome_seal_sha256"] == E.sha256(target / "preoutcome_seal.json")
        assert tab["learner_outcomes_observed"] == 0 and tab["tabpfn"]["model_version"] == "v2"
        for name, digest in tab["artifact_sha256"].items():
            assert E.sha256(target / name) == digest
        if (target / "outcome_run.json").exists():  # after H3: outcomes must postdate both seals
            run = _load(target / "outcome_run.json")
            assert run["completed_at_utc"] > tab["sealed_at_utc"] > _load(target / "preoutcome_seal.json")["sealed_at_utc"]


def test_amendment_a1_is_recorded_and_approved():
    a1 = _load(S6.OUTPUT / "amendments/A1_2026-10-15_guard_count.json")
    assert a1["status"].startswith("APPROVED") and a1["new_sha256"] == E.sha256(S6.ROOT / a1["file"])
