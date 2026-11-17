"""OEH v2 definition, development selection and sealed-test helpers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import oeh_v2_forecast as V
from scripts.analysis import run_oeh_v2_sealed_test_20261014 as T


def _law(rng, n):
    p = rng.dirichlet(np.ones(7), size=n)
    return p


def test_v2_law_is_a_distribution_and_reduces_to_the_two_stage_head():
    rng = np.random.default_rng(0)
    p_oeh, cond, robust = _law(rng, 20), _law(rng, 20), _law(rng, 20)
    cond[:, 6] = 0
    cond /= cond.sum(1, keepdims=True)
    d, prior = rng.uniform(0, 5, 20), np.full((20, 7), 1 / 7)
    for params in ({"lambda_max": 1.0, "eps_max": 0.02, "d0": 3.0}, {"lambda_max": 0.5, "eps_max": 0.0, "d0": "const"}):
        law = V.v2_law(p_oeh, cond, robust, d, prior, **params)
        assert np.allclose(law.sum(1), 1) and (law >= 0).all()
        if params["eps_max"] == 0:
            assert np.allclose(law[:, 6], p_oeh[:, 6])
    plain = V.v2_law(p_oeh, cond, robust, d, prior, lambda_max=0.0, eps_max=0.0, d0="const")
    np.testing.assert_allclose(plain[:, :6], cond[:, :6] * (1 - p_oeh[:, 6:7]))


def test_shift_weight_saturates_at_d0():
    d = np.array([0.0, 1.5, 3.0, 6.0])
    np.testing.assert_allclose(V.shift_weight(1.0, d, 3.0), [0, 0.5, 1, 1])
    np.testing.assert_allclose(V.shift_weight(0.2, d, "const"), [0.2] * 4)


def test_extended_event_matches_the_primary_rule_on_six_budgets():
    from scripts.analysis.compare_difficulty import suffix_event
    rng = np.random.default_rng(1)
    for _ in range(200):
        counts = tuple(int(x) for x in rng.integers(0, 11, 6))
        assert T.extended_event(counts) == suffix_event(counts)
    assert T.extended_event((0, 0, 0, 0, 0, 0, 9, 10)) == 6
    assert T.extended_event((0, 0, 0, 0, 0, 0, 0, 7)) == 8


def test_forest_candidates_are_deterministic_and_unique():
    a, b = T._forest_candidates(), T._forest_candidates()
    assert a == b and len({tuple(c.values()) for c in a}) == len(a) == 80
    assert all(2 * c["S"] <= 128 and 0.01 <= c["p"] <= 0.3 for c in a)


def test_selected_candidate_is_reproduced_by_the_module():
    from scripts.analysis import develop_oeh_v2_heads_20261014 as H
    out = H.OUTPUT
    if not (out / "predictions.csv.gz").exists():  # development artifacts are local; the selection file is versioned
        sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
        assert sel["m6"]["chosen_base"] == "oeh_ensemble"
        return
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))["chosen"]
    pred = pd.read_csv(out / "predictions.csv.gz")
    frame = pred[pred.method == "oeh_ensemble"][H.KEYS + ["source_panel"]].merge(H.labels(), on=H.KEYS + ["source_panel"])
    frame = frame.merge(pd.read_csv(out / "distances.csv"), on=H.KEYS + ["source_panel"]).reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(frame[H.KEYS])
    laws = {m: g.set_index(H.KEYS).loc[idx][H.P].to_numpy(float) for m, g in pred.groupby("method")}
    prior = json.loads((out / "fit_manifest.json").read_text(encoding="utf-8"))["class_prior"]
    pi = np.stack([np.asarray(prior[a]) for a in frame.algorithm])
    law = V.v2_law(laws["oeh_ensemble"], laws["cond_oeh_ensemble"], laws["legacy_ensemble"], frame.distance.to_numpy(), pi,
                   lambda_max=sel["lambda_max"], eps_max=sel["eps_max"], d0=sel["d0"])
    scores = H.panel_scores(law, frame)
    assert scores["colosseum2027"][1] == pytest.approx(sel["colosseum2027_ll"], rel=1e-9)
    assert scores["spibb"][0] == pytest.approx(sel["spibb_ibs"], rel=1e-9)


def test_sealed_analysis_runs_end_to_end_on_synthetic_panels(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "OUTPUT", tmp_path)
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    (tmp_path / "protocol.sha256").write_text(T.E.sha256(tmp_path / "protocol.json") + "\n", encoding="ascii")
    rng = np.random.default_rng(3)
    methods = ["v2", "oeh_final", "legacy", "b2_matched", "xgboost_cf_18", "lce_raw", "b2_plus_lce", "cb_only", "b2_plus_cb",
               "b2_plus_P", "b2_plus_H", "oeh_without_legacy_diagnostics", "oeh_without_oehh_risk", "oeh_without_direct_oehb"]
    for panel, n in {"internal": 9, "colosseum2029": 5, "spibb2": 4, "forest": 6}.items():
        target = tmp_path / panel
        target.mkdir()
        keys = pd.DataFrame([{"mdp_id": f"{panel}_{i}", "policy_family": "f", "algorithm": a} for i in range(n) for a in ("a", "b")])
        event = rng.integers(0, 7, len(keys))
        lab = keys.assign(event=event, event_extended=np.where(event == 6, rng.integers(6, 9, len(keys)), event))
        lab.to_csv(target / "labels.csv", index=False)
        frames = []
        for m in methods:
            law = rng.dirichlet(np.ones(7), len(keys))
            if m == "v2":
                law = 0.5 * law + 0.5 * np.eye(7)[event]
            frames.append(keys.assign(method=m, panel="primary", **{p: law[:, i] for i, p in enumerate(T.P)}))
        pd.concat(frames).to_csv(target / "predictions_preoutcome.csv", index=False)
        (target / "preoutcome_seal.json").write_text("{}", encoding="utf-8")
        T.E.write_json(target / "outcome_run.json", {"preoutcome_seal_sha256": T.E.sha256(target / "preoutcome_seal.json")})
    monkeypatch.setattr(T, "BOOTSTRAP", {**T.BOOTSTRAP, "draws": 200})
    result = T.analyze()
    assert result["steps"]["step1"]["verdict"] == "GO"
    assert len(result["steps"]["step3"]["contrasts"]) == 24
    labels = {(r["baseline"], r["group"], r["metric"]) for r in result["steps"]["step2"]["contrasts"]}
    assert labels == {("oeh_final", g, m) for g in ("internal_noninferiority", "external") for m in ("brier", "log_loss")}
    for name in ("summary.csv", "decision_regret.csv", "extended_endpoint.csv", "shapley.csv", "murphy.csv"):
        assert (tmp_path / name).exists()
    shap = pd.read_csv(tmp_path / "shapley.csv")
    assert set(shap.scope) == {"internal", "colosseum2029", "spibb2", "forest", "pooled_external"}
    from scripts.analysis import report_oeh_v2_sealed_test_20261014 as Rep
    monkeypatch.setattr(Rep, "OUT", tmp_path)
    Rep.main()
    assert "Pooled external" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (tmp_path / "table_v2.tex").read_text(encoding="utf-8").count(r"\\") >= 8


def test_frozen_spec_follows_the_selection_and_the_m3_rule():
    from scripts.analysis import develop_oeh_v2_heads_20261014 as H
    spec = json.loads(H.SPEC.read_text(encoding="utf-8"))
    assert spec["status"] == "frozen" and spec["base_head"] == "oeh_ensemble"
    m3 = spec["m3"]
    assert m3["adopted"] == all(m3["checks"].values())
    assert not spec["extra_training_cohorts"] if not m3["adopted"] else len(spec["extra_training_cohorts"]) == 1
    chosen = json.loads((H.ROOT / spec["selected_in"] / "selection.json").read_text(encoding="utf-8"))["chosen"]
    for key in ("q_oeh", "q_robust", "lambda_max", "eps_max"):
        assert spec["parameters"][key] == chosen[key]
    assert str(spec["parameters"]["d0"]) == chosen["d0"]


def test_v2_is_a_distance_weighted_mixture_and_obeys_the_row_bounds():
    rng = np.random.default_rng(5)
    n = 500
    p_oeh, cond, robust = _law(rng, n), _law(rng, n), _law(rng, n)
    d, prior = rng.uniform(0, 6, n), rng.dirichlet(np.ones(7), n)
    params = {"lambda_max": 1.0, "eps_max": 0.02, "d0": 3.0}
    law = V.v2_law(p_oeh, cond, robust, d, prior, **params)
    lam = V.shift_weight(1.0, d, 3.0)[:, None]
    eps = V.shift_weight(0.02, d, 3.0)[:, None]
    p6 = p_oeh[:, 6:7]
    h_o = np.column_stack([(1 - p6) * V.finite_part(cond), p6])
    h_r = np.column_stack([(1 - p6) * V.finite_part(robust), p6])
    mix = (1 - eps) * (1 - lam) * h_o + (1 - eps) * lam * h_r + eps * prior
    np.testing.assert_allclose(law, mix, atol=1e-12)
    y = rng.integers(0, 7, n)

    def ll(p):
        return -np.log(p[np.arange(n), y])

    def ibs(p):
        return ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)

    w = np.column_stack([(1 - eps) * (1 - lam), (1 - eps) * lam, eps])
    for loss in (ll, ibs):
        bound = w[:, 0] * loss(h_o) + w[:, 1] * loss(h_r) + w[:, 2] * loss(prior)
        assert (loss(law) <= bound + 1e-12).all()
    assert (ll(law) <= -np.log(eps[:, 0] * prior[np.arange(n), y]) + 1e-12)[d > 0].all()


def test_sealed_test_artifacts_are_consistent_with_their_seals():
    out = T.OUTPUT
    assert T.E.sha256(out / "protocol.json") == (out / "protocol.sha256").read_text(encoding="ascii").strip()
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["protocol_sha256"] == T.E.sha256(out / "protocol.json")
    for panel in T.PANELS:
        target = out / panel
        seal = json.loads((target / "preoutcome_seal.json").read_text(encoding="utf-8"))
        assert seal["learner_outcomes_observed"] == 0 and seal["protocol_sha256"] == result["protocol_sha256"]
        for name, digest in seal["artifact_sha256"].items():
            assert T.E.sha256(target / name) == digest, f"{panel}/{name}"
        run = json.loads((target / "outcome_run.json").read_text(encoding="utf-8"))
        assert run["preoutcome_seal_sha256"] == T.E.sha256(target / "preoutcome_seal.json")
        assert run["labels_sha256"] == T.E.sha256(target / "labels.csv")
    steps = result["steps"]
    assert result["adopt_v2"] == (steps["step1"]["verdict"] == "GO" and steps["step2"]["verdict"] == "GO")
