"""Colosseum post-hoc analysis: arms, Shapley, decision costs, max-t rule, outcome guard."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import colosseum_posthoc_20261014 as H
from scripts.analysis import m1_feature_lineage as L
from scripts.analysis.compare_difficulty import FEATURES_B2

ROOT = Path(__file__).resolve().parents[1]
needs_cohort = pytest.mark.skipif(not L.PROTOCOL.exists(), reason="sealed cohort-7 artifacts are not available")


@needs_cohort
def test_factorial_arms_extend_the_m1_arms():
    arms = H.factorial_arms()
    m1 = L.arms()
    assert {name: len(fields) for name, fields in arms.items()} == {
        "b2_matched": 18, "b2_plus_P": 56, "b2_plus_H": 42, "b2_plus_L": 46,
        "oeh_without_oehh_risk": 84, "oeh_without_legacy_diagnostics": 80, "oeh_without_direct_oehb": 70,
        "oeh_full_108": 108}
    for name in m1:
        assert arms[name] == m1[name]
    full = arms["oeh_full_108"]
    for fields in arms.values():
        assert set(FEATURES_B2) <= set(fields)
        assert list(fields) == [column for column in full if column in fields]
    assert H.subset_of("b2_plus_P") == ("P",) and H.subset_of("oeh_without_direct_oehb") == ("H", "L")


@needs_cohort
def test_shapley_recovers_additive_block_effects_and_is_efficient():
    rng = np.random.default_rng(0)
    ids = [f"m{i}" for i in range(6)]
    effect = {"P": rng.normal(size=6), "H": rng.normal(size=6), "L": rng.normal(size=6)}
    interaction = rng.normal(size=6)  # shared only by P and H
    rows = []
    for name in H.factorial_arms():
        subset = H.subset_of(name)
        gain = sum(effect[f] for f in subset) + (interaction if {"P", "H"} <= set(subset) else 0)
        for i, mdp in enumerate(ids):
            rows.append({"method": name, "mdp_id": mdp, "brier": 1.0 - gain[i] if np.ndim(gain) else 1.0})
    shap = H.shapley(pd.DataFrame(rows), "brier")
    np.testing.assert_allclose(shap["L"], effect["L"])
    np.testing.assert_allclose(shap["P"], effect["P"] + interaction / 2)
    np.testing.assert_allclose(shap["H"], effect["H"] + interaction / 2)
    full = effect["P"] + effect["H"] + effect["L"] + interaction
    np.testing.assert_allclose(shap.sum(axis=1), full)


def test_decision_costs_follow_the_frozen_rule():
    # rows: confident x1 & true x1; confident x1 & censored; never confident & true x5; never & censored
    cdf = np.array([[.9] * 6, [.9] * 6, [0.] * 6, [0.] * 6])
    event = np.array([0, 6, 2, 6])
    out = H.decision_costs(cdf, event, tau=.8, lam=50)
    np.testing.assert_allclose(out["cost"], [1, 51, 50, 0])
    np.testing.assert_allclose(out["regret"], [0, 51, 45, 0])
    np.testing.assert_allclose(out["failed"], [0, 1, 0, 0])
    np.testing.assert_allclose(out["missed"], [0, 0, 1, 0])
    late = H.decision_costs(np.array([[0, 0, 0, .85, 1, 1.]]), np.array([1]), tau=.8, lam=50)
    assert late["cost"][0] == 10 and late["regret"][0] == 8  # over-collection is regret, not failure
    x50 = H.decision_costs(H.fixed_cdf("always_x50", 1), np.array([6]), tau=.5, lam=100)
    assert x50["cost"][0] == 150


def test_team_max_t_uses_the_original_standard_error():
    rng = np.random.default_rng(1)
    effect = rng.normal(.1, 1, size=(40, 3))
    mean, lower, critical = H.max_t(effect, seed=5, draws=2000)
    se = effect.std(0, ddof=1) / np.sqrt(40)
    index = np.random.default_rng(5).integers(0, 40, size=(2000, 40))
    expected = float(np.quantile(np.max((effect[index].mean(axis=1) - mean) / se, axis=1), .95))
    assert critical == pytest.approx(expected)
    np.testing.assert_allclose(lower, mean - expected * se)
    _, _, studentized = H.max_t(effect, seed=5, draws=2000, studentize_each_draw=True)
    assert studentized != pytest.approx(critical)


def test_outcome_guard_blocks_colosseum_outcomes(tmp_path):
    code = f"""
import pandas as pd
from pathlib import Path
from scripts.analysis import colosseum_posthoc_20261014 as H
panel = Path({str(tmp_path)!r})
(panel / 'score').mkdir()
for name in ('colosseum_labels.csv', 'colosseum_pilot_features.csv', 'score/summary.csv'):
    (panel / name).write_text('a' + chr(10) + '1' + chr(10), encoding='utf-8')
H.FORBIDDEN_FILES = frozenset([H._norm(panel / 'colosseum_labels.csv')])
H.FORBIDDEN_DIRS = (H._norm(panel / 'score') + H.os.sep,)
with H.outcome_guard() as opened:
    assert len(pd.read_csv(panel / 'colosseum_pilot_features.csv')) == 1
    for name in ('colosseum_labels.csv', 'score/summary.csv'):
        try:
            pd.read_csv(panel / name)
        except H.OutcomeAccessError:
            continue
        raise SystemExit('guard did not block ' + name)
print('OK')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_forbidden_paths_cover_every_colosseum_outcome():
    names = {Path(p).name for p in H.OUTCOME_FILES}
    assert {"colosseum_labels.csv", "colosseum_outcomes.csv", "labels.csv"} <= names
    assert any(str(d).endswith("score_comparators") for d in H.OUTCOME_DIRS)


def test_output_namespace_refuses_sealed_directories(tmp_path):
    with pytest.raises(RuntimeError):
        H.prepare_output(H.CONFIRM / "posthoc", "freeze")
    out = H.prepare_output(tmp_path / "new", "freeze")
    (out / "analysis_plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        H.prepare_output(out, "freeze")
