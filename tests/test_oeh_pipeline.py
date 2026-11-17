"""Single entry point of the final OEH method, generic baselines and the fifth-panel kernels."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.analysis import oeh_generic_baselines as G
from scripts.analysis import oeh_pipeline as O
from scripts.analysis import run_oeh_fifth_panel_20261015 as F


def test_entry_point_reproduces_the_sealed_forecasts_of_all_four_panels():
    report = json.loads((O.ROOT / "artifacts/week8/oeh_pipeline_reproduction_2026-10-15/reproduction.json").read_text(encoding="utf-8"))
    assert set(report) == {"forest", "spibb2", "colosseum2029", "internal"}
    assert max(v["max_abs_diff_law"] for v in report.values()) < 1e-9


def test_entry_point_command_line_is_documented():
    out = subprocess.run([sys.executable, "-m", "scripts.analysis.oeh_pipeline", "--help"], capture_output=True, text=True,
                         cwd=O.ROOT, check=True).stdout
    assert "--dataset" in out and "--query" in out and "oeh_forecast.csv" in out


def test_spec_loader_rejects_an_unfrozen_specification(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    try:
        O.load_spec(path)
    except RuntimeError:
        return
    raise AssertionError("an unfrozen specification was accepted")


def _toy(rng, n, learners=("a", "b")):
    from scripts.analysis.compare_difficulty import FEATURES_B2
    rows = pd.DataFrame(rng.normal(size=(n, len(FEATURES_B2))), columns=list(FEATURES_B2))
    rows["algorithm"] = rng.choice(learners, n)
    rows["panel"] = "primary"
    rows["event"] = rng.integers(0, 7, n)
    return rows


def test_generic_baselines_return_seven_class_distributions():
    rng = np.random.default_rng(0)
    train, query = _toy(rng, 400), _toy(rng, 50)
    for law in (G.knn_analog(train, query), G.logistic(train, query)):
        assert law.shape == (50, 7) and np.isfinite(law).all()
        np.testing.assert_allclose(law.sum(1), 1, atol=1e-12)
        assert (law > 0).all()


def test_fifth_panel_kernels_are_valid_continuing_mdps():
    for T, R, rho in (F.frozenlake_kernel(4, 0.9, 7), F.gambler_kernel(10, 0.4)):
        np.testing.assert_allclose(T.sum(-1), 1, atol=1e-12)
        assert R.min() >= 0 and R.max() <= 1 and abs(rho.sum() - 1) < 1e-12
    T, R, rho = F.gambler_kernel(10, 0.4)
    np.testing.assert_allclose(T[0], np.tile(rho, (T.shape[1], 1)))  # ruin restarts
    assert R[9, 0] == 0.4 and R[5, 4] == 0.4  # stake 1 from 9 and stake 5 from 5 reach the goal with p_h
    assert F.candidates("frozenlake") == F.candidates("frozenlake") and len(F.candidates("gambler")) == 60


def test_fifth_panel_artifacts_are_consistent_with_their_seals():
    out = F.OUTPUT
    assert F.E.sha256(out / "protocol.json") == (out / "protocol.sha256").read_text(encoding="ascii").strip()
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["oeh_spec_sha256"] == F.E.sha256(O.SPEC)
    for panel in F.PANELS:
        target = out / panel
        seal = json.loads((target / "preoutcome_seal.json").read_text(encoding="utf-8"))
        assert seal["learner_outcomes_observed"] == 0 and seal["protocol_sha256"] == F.E.sha256(out / "protocol.json")
        for name, digest in seal["artifact_sha256"].items():
            assert F.E.sha256(target / name) == digest, f"{panel}/{name}"
        run = json.loads((target / "outcome_run.json").read_text(encoding="utf-8"))
        assert run["preoutcome_seal_sha256"] == F.E.sha256(target / "preoutcome_seal.json")
        assert run["labels_sha256"] == F.E.sha256(target / "labels.csv")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    gate = result["gate"]
    assert gate["mdps"] == 72 and gate["verdict"] == ("GO" if all(c["simultaneous_lower_95"] > 0 for c in gate["contrasts"]) else "NO-GO")
    scored = pd.read_csv(out / "scored.csv.gz")
    per = scored[scored.method.isin(["oeh", "robust"])].groupby(["source_panel", "method", "mdp_id"]).brier.mean().unstack("method")
    assert abs((per.robust - per.oeh).mean() - gate["contrasts"][0]["mean_improvement"]) < 1e-12
