"""Executable examples and traceability checks for the Week 8 H1 handover.

These checks verify documented conventions; they are not evidence of a
sample-complexity bound or validation of the full forecaster on new data.
"""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pytest

from oeh import horizon as H
from oeh import posterior as post
from oeh import predictive as PR
from oeh import tabular as T
from scripts.analysis.compare_difficulty import FEATURES_B2, GRID, suffix_event
from scripts.analysis.develop_oeh_difficulty import score_probabilities
from scripts.analysis.develop_oeh_horizon import _posterior_mean_rows

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/week8"
GLOSSARY = DOCS / "H1_THEORY_GLOSSARY_2026-10-08.md"
MAP = DOCS / "H1_FORMULA_CODE_MAP_2026-10-08.md"
PROTOCOL = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04/protocol.json"


def test_h1_relative_links_resolve():
    paths = sorted(DOCS.glob("H1_*_2026-10-08.md"))
    assert len(paths) >= 4
    for path in paths:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)


def test_formula_map_references_real_source_symbols():
    text = MAP.read_text(encoding="utf-8")
    ids = re.findall(r"^\| (F\d+) \|", text, flags=re.MULTILINE)
    assert ids == [f"F{i:02}" for i in range(1, 23)]
    references = re.findall(r"\[[^\]]+\]\(([^)]+\.py)\)::([\w, ]+)", text)
    assert len(references) >= 22
    for relative, names in references:
        path = (MAP.parent / relative).resolve()
        # Parse instead of importing operational runners: no global runner
        # configuration, XGBoost dependency or archive path mutation is needed.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        symbols = {node.name for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        symbols.update(node.id for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        assert set(names.strip().split(", ")) <= symbols, (path, names)


def test_method_tex_citations_references_and_environments_resolve():
    sources = [ROOT / "paper/main.tex", *sorted((ROOT / "paper/sections").glob("*.tex"))]
    all_tex = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    keys = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "paper/references.bib").read_text(encoding="utf-8")))
    citations = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", all_tex)
                 for key in group.split(",")}
    assert citations <= keys
    labels = re.findall(r"\\label\{([^}]+)\}", all_tex)
    assert len(labels) == len(set(labels))
    references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", all_tex))
    assert references <= set(labels)
    for name in ("02_related_work.tex", "03_difficulty.tex"):
        tex = (ROOT / "paper/sections" / name).read_text(encoding="utf-8")
        assert Counter(re.findall(r"\\begin\{([^}]+)\}", tex)) == Counter(
            re.findall(r"\\end\{([^}]+)\}", tex))
        # This is a source lint, not a substitute for a LaTeX build/page check.
        assert tex.count("{") == tex.count("}")


@pytest.mark.parametrize("counts,event", [
    ((2, 5, 8, 7, 9, 10), 4),
    ((8, 8, 8, 8, 8, 8), 0),
    ((10, 10, 10, 10, 10, 7), 6),
    ((0, 0, 0, 0, 0, 8), 5),
])
def test_documented_suffix_is_not_first_threshold_crossing(counts, event):
    assert suffix_event(counts) == event
    assert PR.suffix_event(np.asarray(counts)) == event
    assert len(GRID) == 6


def test_documented_scores_preserve_censoring_and_use_six_thresholds():
    p = np.asarray([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]] * 2)
    events = np.array([4, 6])
    brier, log_loss = score_probabilities(p, events)
    cdf = np.cumsum(p, axis=1)[:, :6]
    truth = np.array([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]])
    np.testing.assert_allclose(brier, ((cdf - truth) ** 2).sum(axis=1) / 6)
    np.testing.assert_allclose(log_loss, -np.log([0.1, 0.4]))
    _, floored = score_probabilities(np.eye(7)[[0]], [6])
    assert floored[0] == pytest.approx(-np.log(1e-12))


@pytest.mark.parametrize("returned", [0, 2])
def test_documented_draw_shortfall_and_half_counts(returned):
    nr = np.ones((returned, 10, 6, 1))
    outcomes = PR.Outcomes(("example",), PR.GRID, nr, {},
                           {"draws_requested": 64, "draws_returned": returned})
    value = PR.requirement_forecast(outcomes)["example"]
    events = np.full(64, 6) if returned == 0 else np.zeros(returned, dtype=int)
    assert np.array_equal(value["events"], events)
    expected = (np.bincount(events, minlength=7) + 0.5) / (len(events) + 3.5)
    np.testing.assert_allclose(value["probabilities"], expected)


def test_normalized_return_is_not_clipped_or_a_probability():
    P = np.zeros((1, 2, 2, 2))
    P[..., 1] = 1.0  # one decision then absorbing terminal
    r = np.array([[[0.5, 1.0], [0.0, 0.0]]])
    policies = np.array([[
        [[1, 0], [1, 0]], [[0.5, 0.5], [1, 0]], [[0, 1], [1, 0]],
    ]], dtype=float)
    nr = T.normalized_returns(P, r, np.array([[1., 0.]]), policies, 0.95)
    np.testing.assert_allclose(nr, [[-1., 0., 1.]])


def test_same_logged_support_cannot_identify_optimum_in_both_mdps():
    P = np.zeros((2, 2, 2))
    P[..., 1] = 1.
    optima = []
    statistics = []
    for unseen_reward in (0., 1.):
        r = np.array([[0.5, unseen_reward], [0., 0.]])
        _, policy = T.optimal_values(P, r, 0.95)
        optima.append(policy[0])
        # beta always logs action 0, so both pilots are exactly identical.
        arrays = {"state": [0] * 10, "action": [0] * 10,
                  "next_state": [1] * 10, "reward": [0.5] * 10,
                  "terminal": [True] * 10}
        statistics.append(post.pilot_statistics(arrays, [1., 0.],
                           num_states=2, num_actions=2, gamma=0.95))
    assert optima == [0, 1]
    np.testing.assert_array_equal(statistics[0].transition_counts, statistics[1].transition_counts)
    assert statistics[0].reward_table == statistics[1].reward_table
    assert statistics[0].action_counts[0, 1] == 0


def test_model_support_can_exceed_observed_support():
    # A positive smoothing concentration is an explicit modeling choice.
    counts = np.array([[10., 0.], [0., 0.]])
    mean = _posterior_mean_rows(counts, None, 1., np.array([0.5, 0.5]))
    assert not (counts[0] > 0)[1] and (mean[0] > 0)[1]
    np.testing.assert_allclose(mean.sum(axis=1), 1.)


def test_contraction_gap_example_and_uniform_rollout_identity():
    P = np.array([[[0.8, 0.2], [0.1, 0.9]], [[0.7, 0.3], [0.2, 0.8]]])
    r = np.array([[0.1, 0.6], [0.7, 0.2]])
    gamma = 0.7
    v_star, greedy_star = T.optimal_values(P, r, gamma)
    q_star = r + gamma * np.einsum("san,n->sa", P, v_star)
    gap = np.min(np.abs(q_star[:, 0] - q_star[:, 1]))
    assert gap > 0
    q = np.zeros_like(r)
    initial_error = np.max(np.abs(q - q_star))
    for k in range(1, 81):
        bound = gamma ** (k - 1) * initial_error
        assert np.max(np.abs(q - q_star)) <= bound + 1e-12
        if 2 * bound < gap:
            np.testing.assert_array_equal(q.argmax(axis=1), greedy_star)
        q = r + gamma * np.einsum("san,n->sa", P, q.max(axis=1))
    beta = np.full((2, 2), 0.5)
    for depth in range(1, 6):
        cost = H.lookahead_cost(P, beta, [greedy_star] * (depth - 1))
        np.testing.assert_allclose(cost, depth)


def test_review_partition_matches_frozen_108_columns_and_variant_counts():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))["oeh_final"]
    fields = spec["fields"]
    assert spec["candidate_variant"] == "plus_prior_profile"
    assert spec["candidate_law"] is False and len(spec["meta_training_cohorts"]) == 7
    assert len(fields) == len(set(fields)) == 108
    groups = {
        "G1": [f for f in fields if f in FEATURES_B2],
        "G2": [f for f in fields if f.startswith(("oehb_cdf_", "oehb_success_x", "oehb_return_x"))],
        "G3": [f for f in fields if f.startswith("oehb_log_")],
        "G4": [f for f in fields if f.startswith("oehh_") and not f.startswith(
               ("oehh_risk_x", "oehh_miss_x", "oehh_risk_slack"))],
        "G5": [f for f in fields if f.startswith(("oehh_risk_x", "oehh_miss_x", "oehh_risk_slack"))],
        "G6": [f for f in fields if f.startswith("other_") and "_oehb_cdf_" in f],
        "G7": [f for f in fields if f.startswith(("oeh3_", "lcoeh4_"))],
        "G8": [f for f in fields if f.startswith("other_") and "_lcoeh4_" in f],
    }
    assert [len(g) for g in groups.values()] == [18, 18, 2, 11, 13, 18, 10, 18]
    assert Counter(f for group in groups.values() for f in group) == Counter(fields)
    for drop, remaining in [(('G4', 'G5'), 84), (('G2', 'G3', 'G6'), 70),
                            (('G6', 'G8'), 72), (('G7', 'G8'), 80)]:
        removed = {f for name in drop for f in groups[name]}
        assert len([f for f in fields if f not in removed]) == remaining
