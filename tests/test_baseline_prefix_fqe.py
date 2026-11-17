"""Prefix + FQE baseline (task card T2)."""
from __future__ import annotations

import sys

import numpy as np

from scripts.analysis import baseline_prefix_fqe as B

sys.path.insert(0, str(B.ROOT / "code"))
from oeh.tabular import policy_values  # noqa: E402


def _toy(rng, S=6, A=3, n=400, terminal_rate=0.0):
    P = rng.dirichlet(np.ones(S), (S, A))
    R = rng.uniform(0, 1, (S, A))
    s = rng.integers(0, S, n)
    a = rng.integers(0, A, n)
    nxt = np.array([rng.choice(S, p=P[i, j]) for i, j in zip(s, a)])
    t = rng.uniform(size=n) < terminal_rate
    return s, a, R[s, a], nxt, t, S, A


def test_fqe_matches_exact_evaluation_of_the_empirical_model():
    rng = np.random.default_rng(0)
    for terminal_rate in (0.0, 0.1):
        s, a, r, n, t, S, A = _toy(rng, terminal_rate=terminal_rate)
        p_hat, r_hat, visited = B.empirical(s, a, r, n, t, S, A)
        for _ in range(5):
            policy = rng.dirichlet(np.ones(A), S)
            exact = policy_values(np.where(visited[..., None], p_hat, 0.0), np.where(visited, r_hat, 0.0), policy, B.GAMMA)
            np.testing.assert_allclose(B.fqe(p_hat, r_hat, visited, policy), exact, atol=1e-8)


def test_unvisited_pairs_are_worth_zero():
    rng = np.random.default_rng(1)
    s, a, r, n, t, S, A = _toy(rng, n=30)
    p_hat, r_hat, visited = B.empirical(s, a, r, n, t, S, A)
    assert not visited.all()
    policy = np.zeros((S, A))
    rows = np.flatnonzero(~visited.any(1))
    if len(rows):
        policy[:, 0] = 1.0
        policy[rows] = np.eye(A)[0]
        v = B.fqe(p_hat, r_hat, visited, policy)
        assert np.allclose(v[rows][~visited[rows, 0]], 0.0)


def test_curve_is_monotone_and_the_law_is_a_distribution():
    budgets = np.repeat(np.asarray(B.PREFIXES, float), 10)
    for outcomes in (np.zeros(40), np.ones(40), (budgets >= 4).astype(float), (budgets < 2).astype(float)):
        alpha, beta = B.fit_curve(budgets, outcomes)
        assert beta >= 0 and np.isfinite(alpha)
        law = B.raw_law(alpha, beta)
        assert law.shape == (7,) and np.isclose(law.sum(), 1) and (law > 0).all()
    low = B.raw_law(*B.fit_curve(budgets, np.zeros(40)))
    high = B.raw_law(*B.fit_curve(budgets, np.ones(40)))
    assert low[6] > 0.9 and high[0] > 0.5


def test_pilot_features_are_deterministic_and_valid():
    rng = np.random.default_rng(2)
    S, A, n = 5, 2, 10 * 5 * 2
    s = rng.integers(0, S, n)
    arrays = {"state": s, "action": rng.integers(0, A, n), "reward": rng.uniform(0, 1, n), "next_state": rng.integers(0, S, n),
              "terminal": np.zeros(n, bool), "trajectory_id": np.repeat(np.arange(10), n // 10), "timestep": np.tile(np.arange(n // 10), 10)}
    a = B.pilot_features(arrays, S, A, "toy", "behavior")
    b = B.pilot_features(arrays, S, A, "toy", "behavior")
    assert a == b and len(a) == 3
    for row in a:
        law = np.array([row[f"raw_p_{i}"] for i in range(7)])
        assert np.isclose(law.sum(), 1) and all(0 <= row[f"pfqe_success_b{b}"] <= 1 for b in B.PREFIXES)


def test_recorded_run_matches_its_manifest():
    import json

    manifest = json.loads((B.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "posthoc_descriptive" and manifest["meta_training_feature_coverage"] >= 0.999
    for name, digest in manifest["outputs_sha256"].items():
        assert B.hashlib.sha256((B.OUTPUT / name).read_bytes()).hexdigest() == digest
    for rel, digest in manifest["inputs_sha256"].items():
        assert B.hashlib.sha256((B.ROOT / rel).read_bytes()).hexdigest() == digest
