"""Nested LOPO of the v2 selection procedure (task card T5)."""
from __future__ import annotations

import json

import pandas as pd

from scripts.analysis import nested_lopo_v2_selection as N


def test_reimplemented_procedure_reproduces_the_recorded_selection():
    frame, laws, pi, d = N.load()
    full = N.select(frame, laws, pi, d, N.INTERNAL, N.EXTERNAL)
    recorded = json.loads((N.HEADS / "selection.json").read_text(encoding="utf-8"))
    assert full["base"] == recorded["m6"]["chosen_base"]
    assert full["config"] == {k: recorded["chosen"][k] for k in ("q_oeh", "q_robust", "lambda_max", "eps_max", "d0")}
    assert abs(full["inner_score"] - recorded["chosen"]["external_score"]) < 1e-12


def test_recorded_outer_folds_and_manifest():
    folds = pd.read_csv(N.OUTPUT / "outer_folds.csv")
    assert set(folds.held_out) == {*N.INTERNAL, *N.EXTERNAL}
    manifest = json.loads((N.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive" and manifest["summary"]["reproduces_recorded_selection"]
    for name, digest in manifest["outputs_sha256"].items():
        assert N.sha(N.OUTPUT / name) == digest
    log = pd.read_csv(N.OUTPUT / "development_log.csv")
    assert len(log) == len(folds)
