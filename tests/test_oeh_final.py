"""Mathematical and equivalence controls for the final OEH modules."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from oeh import horizon as H
from oeh import posterior as post
from oeh import predictive as PR
from oeh import selection as SEL
from oeh import tabular as T

FIXTURE = Path(__file__).resolve().parent / "fixtures/reference/empirical_requirement.py"
# Byte identity of the archived learner source at commit e715e3c.
REFERENCE_SHA256 = "cd44b6b3043e9e66ec45f26641ba7c0f504e8d5fdb11004f71d1d7f0a6480398"


def reference():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("reference_learners", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("reference_learners", module)
    spec.loader.exec_module(module)
    return module


def random_mdp(rng, S=7, A=3, terminal=(6,), sparse=True):
    P = rng.dirichlet(np.full(S, 0.4 if sparse else 2.0), size=(S, A))
    R = np.where(rng.random((S, A, S)) < 0.25, rng.random((S, A, S)), 0.0)
    for t in terminal:
        P[t] = 0.0
        P[t, :, t] = 1.0
        R[t] = 0.0
    rho = np.zeros(S)
    rho[0] = 1.0
    beta = rng.dirichlet(np.ones(A), size=S)
    term = np.zeros(S, dtype=bool)
    term[list(terminal)] = True
    return P, R, beta, rho, term


def simulate(rng, P, R, beta, rho, term, size, horizon=6):
    sample = post.MDPSample(P[None], R[None], beta[None], term, rho, 0.95)
    return {k: v[0, 0] for k, v in post.simulate_streams(sample, rng, streams=1, length=size,
                                                       horizon=horizon).items()}


def as_transitions(arrays):
    return [SimpleNamespace(state=int(s), action=int(a), reward=float(r), next_state=int(n),
                            terminal=bool(t))
            for s, a, r, n, t in zip(arrays["state"], arrays["action"], arrays["reward"],
                                     arrays["next_state"], arrays["terminal"])]


@pytest.mark.parametrize("size", [5, 40, 300])
def test_batched_learners_match_archived_reference(size):
    ref = reference()
    rng = np.random.default_rng(size)
    P, R, beta, rho, term = random_mdp(rng)
    datasets = [simulate(rng, P, R, beta, rho, term, size) for _ in range(6)]
    stacked = {k: np.stack([d[k] for d in datasets]) for k in datasets[0]}
    model = T.count_model(stacked["state"], stacked["action"], stacked["reward"],
                          stacked["next_state"], stacked["terminal"], num_states=7, num_actions=3)
    batch = {name: fn(model, 0.95) for name, fn in T.CORE_LEARNERS.items()}
    for index, data in enumerate(datasets):
        transitions = as_transitions(data)
        expected = {
            "cfqi_tabular": ref.cfqi_tabular(transitions, num_states=7, num_actions=3, gamma=0.95),
            "iql_tabular": ref.iql_tabular(transitions, num_states=7, num_actions=3, gamma=0.95),
            "behavior_cloning_tabular": ref.behavior_cloning_tabular(transitions, num_states=7, num_actions=3),
        }
        for name, policy in expected.items():
            if name == "iql_tabular":
                # The batched and archived reduction orders can round the
                # soft action probabilities differently across BLAS builds.
                np.testing.assert_allclose(batch[name][index], policy,
                                           rtol=1e-12, atol=1e-14, err_msg=name)
            else:
                assert np.array_equal(batch[name][index], policy), name


def test_expectile_matches_reference_including_ties():
    ref = reference()
    rng = np.random.default_rng(3)
    values = np.round(rng.random((50, 4)), 1)
    weights = rng.random((50, 4)) + 0.01
    mask = rng.random((50, 4)) < 0.8
    mask[:, 0] = True
    batched = T._expectile(values, weights, mask, 0.7)
    for row in range(50):
        keep = mask[row]
        assert batched[row] == pytest.approx(
            ref._expectile(values[row, keep], weights[row, keep], 0.7),
            rel=1e-14, abs=1e-15,
        )


def test_expectile_resolves_tie_where_archived_tolerance_fails():
    ref = reference()
    values, weights = np.array([17.002194919023687, 17.002194919023687]), np.array([0.9, 0.1])
    with pytest.raises(RuntimeError):
        ref._expectile(values, weights, 0.7)
    solved = T._expectile(values[None], weights[None], np.ones((1, 2), dtype=bool), 0.7)
    assert solved[0] == pytest.approx(values[0], rel=1e-15)


def test_count_model_and_validation():
    model = T.count_model([0, 0, 1], [1, 1, 0], [0.5, 0.5, 1.0], [1, 2, 2], [False, False, True],
                          num_states=3, num_actions=2)
    assert model.shape == (1, 3, 2)
    assert model.counts[0, 0, 1] == 2 and model.next_counts[0, 1, 0].sum() == 0
    assert model.terminal[0, 2] and not model.terminal[0, 1]
    p_hat, r_hat = model.empirical()
    assert np.isclose(p_hat[0, 0, 1].sum(), 1.0) and r_hat[0, 1, 0] == 1.0
    with pytest.raises(ValueError):
        T.count_model([0], [5], [0.0], [0], [False], num_states=3, num_actions=2)
    with pytest.raises(ValueError):
        T.count_model([0, 1], [0], [0.0], [0], [False], num_states=3, num_actions=2)
    with pytest.raises(ValueError):
        T.count_model([0], [0], [np.nan], [0], [False], num_states=3, num_actions=2)


def test_normalized_return_endpoints():
    rng = np.random.default_rng(5)
    P, R, _, rho, _ = random_mdp(rng, terminal=())
    r = np.einsum("san,san->sa", P, R)
    _, best = T.optimal_values(P, r, 0.95)
    policies = np.stack([np.eye(3)[best], np.full((7, 3), 1 / 3)])
    nr = T.normalized_returns(P[None], r[None], rho[None], policies[None], 0.95)
    assert np.allclose(nr, [[1.0, 0.0]])


def test_empirical_bayes_concentration_limits_and_recovery():
    rng = np.random.default_rng(11)
    truth = 3.0
    rows = np.stack([rng.multinomial(40, rng.dirichlet(np.full(5, truth / 5))) for _ in range(400)])
    estimate = post.dirichlet_multinomial_concentration(rows)
    assert 2.0 < estimate < 4.5
    deterministic = np.zeros((30, 4), dtype=int)
    deterministic[:, 1] = 9
    assert post.dirichlet_multinomial_concentration(deterministic) < 1e-5
    assert post.dirichlet_multinomial_concentration(np.full((50, 3), 30)) == float("inf")
    assert math.isnan(post.dirichlet_multinomial_concentration(np.array([[1, 0], [0, 1]])))
    pooled = post.dirichlet_multinomial_concentration(rows, np.full(5, 0.2))
    assert pooled == pytest.approx(estimate)
    with pytest.raises(ValueError):
        post.dirichlet_multinomial_concentration(rows, np.array([1.0, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        post.dirichlet_multinomial_concentration(np.ones(4))


def test_dirichlet_point_mass_limit_is_exact():
    rng = np.random.default_rng(0)
    draws = post.sample_dirichlet(rng, np.full((2000, 3), 1e-300))
    assert np.allclose(draws.sum(-1), 1.0)
    assert set(np.unique(draws)) <= {0.0, 1.0}
    assert 500 < (draws[:, 0] == 1).sum() < 850


@pytest.mark.parametrize("kind,rewards", [
    ("next_state", [0.0, 1.0, 0.0, 1.0]),
    ("state_action", [0.3, 0.3, 0.2, 0.2]),
    ("triple", [0.3, 0.4, 0.2, 0.2]),
])
def test_reward_key_is_coarsest_consistent_factorization(kind, rewards):
    arrays = {"state": [0, 0, 1, 1], "action": [0, 0, 1, 1], "next_state": [1, 2, 1, 2],
              "reward": rewards, "terminal": [False] * 4}
    if kind == "state_action":
        arrays["next_state"] = [1, 2, 2, 0]
    stats = post.pilot_statistics(arrays, [1, 0, 0], num_states=3, num_actions=2, gamma=0.9)
    assert stats.reward_key == kind


def test_pilot_statistics_rejects_invalid_inputs():
    good = {"state": [0], "action": [0], "next_state": [1], "reward": [0.0], "terminal": [False]}
    with pytest.raises(ValueError):
        post.pilot_statistics({**good, "reward": [2.0]}, [1, 0], num_states=2, num_actions=2, gamma=0.9)
    with pytest.raises(ValueError):
        post.pilot_statistics({**good, "state": []}, [1, 0], num_states=2, num_actions=2, gamma=0.9)
    with pytest.raises(ValueError):
        post.pilot_statistics(good, [0.5, 0.2], num_states=2, num_actions=2, gamma=0.9)
    stats = post.pilot_statistics(good, [1, 0], num_states=2, num_actions=2, gamma=0.9)
    with pytest.raises(ValueError):
        post.fit_posterior(stats, reward_prior="optimistic")


@pytest.mark.parametrize("reward_prior", post.REWARD_PRIORS)
@pytest.mark.parametrize("transition_base", post.TRANSITION_BASES)
def test_posterior_draws_respect_data_terminals_and_filter(reward_prior, transition_base):
    rng = np.random.default_rng(2)
    P, R, beta, rho, term = random_mdp(rng)
    data = simulate(rng, P, R, beta, rho, term, 400)
    stats = post.pilot_statistics(data, rho, num_states=7, num_actions=3, gamma=0.95)
    model = post.fit_posterior(stats, reward_prior=reward_prior, transition_base=transition_base)
    assert model.describe()["reward_prior"] == reward_prior
    sample, report = post.sample_mdps(model, rng, 12, minimum_dynamic_range=0.05)
    assert report["returned"] == len(sample.P) > 0
    assert np.allclose(sample.P.sum(-1), 1.0) and np.allclose(sample.beta.sum(-1), 1.0)
    observed = stats.terminal
    assert np.all(sample.P[:, observed][..., observed].sum(-1) > 0)
    for s, a, n, value in zip(data["state"], data["action"], data["next_state"], data["reward"]):
        assert np.all(sample.R[:, s, a, n] == value)
    r = sample.expected_rewards
    j_star, j_rand = T.reference_returns(sample.P, r, np.broadcast_to(rho, (len(r), 7)), 0.95)
    assert np.all(j_star - j_rand >= 0.05)


def test_posterior_shortfall_is_reported_not_hidden():
    arrays = {"state": [0, 0, 1], "action": [0, 1, 0], "next_state": [1, 1, 0],
              "reward": [0.0, 0.0, 0.0], "terminal": [False] * 3}
    stats = post.pilot_statistics(arrays, [1, 0], num_states=2, num_actions=2, gamma=0.9)
    model = post.fit_posterior(stats, reward_prior="unseen_zero")
    sample, report = post.sample_mdps(model, np.random.default_rng(0), 4,
                                      minimum_dynamic_range=0.1, max_rounds=2)
    assert len(sample.P) == 0 and report["returned"] == 0 and report["rejected_with_certainty"]
    exchangeable = post.fit_posterior(stats)
    sample, report = post.sample_mdps(exchangeable, np.random.default_rng(0), 4,
                                      minimum_dynamic_range=1e9, max_rounds=2)
    assert len(sample.P) == 0 and report["proposals"] == 8


def test_simulated_streams_follow_generator_semantics():
    rng = np.random.default_rng(4)
    P, R, beta, rho, term = random_mdp(rng)
    sample = post.MDPSample(P[None], R[None], beta[None], term, rho, 0.95)
    out = post.simulate_streams(sample, rng, streams=3, length=500, horizon=6)
    assert out["state"].shape == (1, 3, 500)
    for i in range(3):
        s, n, t = out["state"][0, i], out["next_state"][0, i], out["terminal"][0, i]
        tr, clock, ep = out["truncated"][0, i], out["timestep"][0, i], out["trajectory_id"][0, i]
        assert not np.any(term[s]) and np.array_equal(t, term[n])
        assert np.all(clock < 6) and not np.any(t & tr)
        assert np.array_equal(tr, (clock == 5) & ~t)
        ends = t | tr
        assert np.all(s[1:][~ends[:-1]] == n[:-1][~ends[:-1]])
        assert np.all(s[1:][ends[:-1]] == 0) and np.all(np.diff(ep) == ends[:-1])


def test_suffix_event_and_kt_estimator():
    successes = np.array([[10, 10, 10, 10, 10, 10], [0, 9, 7, 8, 9, 10],
                          [0, 0, 0, 0, 0, 8], [9, 9, 9, 9, 9, 7]])
    assert PR.suffix_event(successes).tolist() == [0, 3, 5, 6]
    probabilities = PR.kt_probabilities([6, 6, 6], 7)
    assert probabilities.sum() == pytest.approx(1.0) and probabilities.min() > 0
    assert probabilities[6] == pytest.approx(3.5 / 6.5)
    assert PR.offline_horizon(6, 8, 2) == math.inf
    assert PR.offline_horizon(0, 8, 2) == pytest.approx(math.log(16, 2))


def test_posterior_outcomes_replay_protocol_for_all_learner_kinds():
    rng = np.random.default_rng(9)
    P, R, beta, rho, term = random_mdp(rng, S=5, A=2, terminal=(4,))
    pilot = simulate(rng, P, R, beta, rho, term, 100)
    seen = []

    def constant(arrays, context):
        seen.append((len(arrays["state"]), context["seed"], context["behavior"].shape))
        return np.full((context["num_states"], context["num_actions"]), 0.5)

    outcomes = PR.posterior_outcomes(pilot, rho, num_states=5, num_actions=2, gamma=0.95,
                                     draws=3, seed=1, seeds=2, per_dataset={"uniform": constant},
                                     minimum_dynamic_range=0.01)
    assert outcomes.learners[-1] == "uniform"
    assert outcomes.normalized_return.shape == (3, 2, 6, 4)
    assert np.allclose(outcomes.normalized_return[..., -1], 0.0, atol=1e-9)
    assert sorted({size for size, _, _ in seen}) == [10 * b for b in PR.GRID]
    forecast = PR.requirement_forecast(outcomes)
    assert set(forecast) == set(outcomes.learners)
    assert forecast["uniform"]["censored_fraction"] == 1.0
    repeat = PR.posterior_outcomes(pilot, rho, num_states=5, num_actions=2, gamma=0.95,
                                   draws=3, seed=1, seeds=2, minimum_dynamic_range=0.01)
    assert np.array_equal(repeat.normalized_return, outcomes.normalized_return[..., :3])
    with pytest.raises(ValueError):
        PR.posterior_outcomes(pilot, rho, num_states=5, num_actions=2, gamma=0.95, draws=1,
                              seed=1, budgets=(2, 1))
    with pytest.raises(ValueError):
        PR.posterior_outcomes(pilot, rho, num_states=5, num_actions=2, gamma=0.95, draws=1,
                              seed=1, batched={}, per_dataset={})
    short = PR.posterior_outcomes(pilot, rho, num_states=5, num_actions=2, gamma=0.95, draws=1,
                                  seed=1, budgets=(10,), seeds=1)
    with pytest.raises(ValueError):
        PR.requirement_forecast(short)


def test_single_draw_memory_chunk_preserves_forecast_contract(monkeypatch):
    """The BRIDGE memory amendment keeps 64 total draws but uses one per chunk."""
    rng = np.random.default_rng(23)
    P, R, beta, rho, term = random_mdp(rng, S=5, A=2, terminal=(4,))
    pilot = simulate(rng, P, R, beta, rho, term, 100)
    monkeypatch.setattr(PR, "CHUNK_DRAWS", 1)
    kwargs = dict(num_states=5, num_actions=2, gamma=0.95,
                  draws=3, seed=17, seeds=2, minimum_dynamic_range=0.01)
    first = PR.posterior_outcomes(pilot, rho, **kwargs)
    repeat = PR.posterior_outcomes(pilot, rho, **kwargs)
    assert first.normalized_return.shape == (3, 2, 6, 3)
    assert np.array_equal(first.normalized_return, repeat.normalized_return)
    assert first.sampling["draws_requested"] == 3
    assert first.sampling["draws_returned"] == 3
    for item in PR.requirement_forecast(first).values():
        assert item["probabilities"].shape == (7,)
        assert item["probabilities"].sum() == pytest.approx(1.0)


def test_unscorable_posterior_counts_as_censored():
    empty = PR.Outcomes(("a",), PR.GRID, np.empty((0, 10, 6, 1)), {},
                        {"draws_requested": 4, "draws_returned": 0})
    forecast = PR.requirement_forecast(empty)["a"]
    assert forecast["censored_fraction"] == 1.0 and forecast["probabilities"][6] == pytest.approx(4.5 / 7.5)


def test_lookahead_cost_reduces_to_effective_horizon_depth_under_uniform_behavior():
    rng = np.random.default_rng(1)
    P, R, _, _, _ = random_mdp(rng, S=6, A=3, terminal=())
    uniform = np.full((6, 3), 1 / 3)
    for k in range(1, 6):
        policies = [rng.integers(0, 3, size=6) for _ in range(k - 1)]
        assert np.allclose(H.lookahead_cost(P, uniform, policies), k)
    expert = np.eye(3)[np.zeros(6, dtype=int)]
    cost = H.lookahead_cost(P, expert, [np.zeros(6, dtype=int)] * 3)
    assert np.allclose(cost[:, 0], 0.0) and np.all(np.isinf(cost[:, 1:]))


def test_stream_visitation_matches_simulated_frequencies():
    rng = np.random.default_rng(6)
    P, R, beta, rho, term = random_mdp(rng)
    exact = H.stream_visitation(P, beta, rho, term, 6)
    data = simulate(rng, P, R, beta, rho, term, 200_000)
    freq = np.zeros_like(exact)
    np.add.at(freq, (data["state"], data["action"]), 1.0)
    assert exact.sum() == pytest.approx(1.0)
    assert np.abs(freq / freq.sum() - exact).max() < 0.01


def test_deviation_returns_match_direct_evaluation():
    rng = np.random.default_rng(8)
    P, R, _, rho, _ = random_mdp(rng, terminal=())
    r = np.einsum("san,san->sa", P, R)
    policy = rng.integers(0, 3, size=7)
    values = T.policy_values(P, r, np.eye(3)[policy], 0.95)
    q = r + 0.95 * np.einsum("san,n->sa", P, values)
    fast = H._deviation_returns(P, r, rho, 0.95, policy, values, q)
    for s in range(7):
        for a in range(3):
            changed = policy.copy()
            changed[s] = a
            direct = rho @ T.policy_values(P, r, np.eye(3)[changed], 0.95)
            assert fast[s, a] == pytest.approx(direct, rel=1e-9, abs=1e-12)


def chain(length=5, slip=0.1):
    """Right moves toward a rewarding terminal state; left resets to the start."""
    S, A = length + 1, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    for s in range(length):
        P[s, 1, s + 1] = 1 - slip
        P[s, 1, 0] += slip
        P[s, 0, 0] = 1.0
    P[length, :, length] = 1.0
    R[length - 1, 1, length] = 1.0
    rho = np.zeros(S)
    rho[0] = 1.0
    term = np.zeros(S, dtype=bool)
    term[length] = True
    return P, R, rho, term


def test_offline_horizon_monotone_in_behavior_coverage_and_detects_trap():
    P, R, rho, term = chain()
    values, coverage = [], []
    for right in (0.5, 0.3, 0.1, 0.01):
        beta = np.tile([1 - right, right], (6, 1))
        result = H.offline_effective_horizon(P, R, beta, rho, 0.95, terminal=term, tau=0.95, horizon=50)
        assert result.status == "identified" and result.depth is not None
        assert math.isfinite(result.statistical) and math.isfinite(result.coverage)
        assert result.value == pytest.approx(result.statistical + result.coverage)
        values.append(result.value)
        coverage.append(result.coverage)
    # Starving the needed action of data raises the horizon through the coverage term.
    assert values == sorted(values) and coverage == sorted(coverage)
    trapped = np.tile([1.0, 0.0], (6, 1))
    result = H.offline_effective_horizon(P, R, trapped, rho, 0.95, terminal=term, tau=0.95, horizon=50)
    assert result.status == "coverage_trap" and result.value == math.inf
    assert result.as_dict()["oeh_h_status"] == "coverage_trap"


def test_offline_horizon_without_critical_rankings_needs_one_sample():
    P, R, rho, term = chain(length=2, slip=0.0)
    R[:] = 0.0
    R[0, :, 0] = 1.0
    R[0, :, 1] = 1.0
    beta = np.full((3, 2), 0.5)
    result = H.offline_effective_horizon(P, R, beta, rho, 0.95, terminal=term, tau=0.0,
                                         horizon=20, max_depth=3)
    assert result.value <= 0.0


def test_stationary_eh_on_chain():
    P, R, rho, term = chain(length=3, slip=0.0)
    value, depth = H.stationary_eh(P, R, 0.95, max_depth=20, terminal=term)
    assert depth is not None and math.isfinite(value) and value >= depth


def test_selection_rule_orders_success_then_return_then_registry():
    values = np.array([[0.96, 0.97, 0.5], [0.97, 0.97, 0.99], [0.99, 0.10, 0.20]])
    choice = SEL.select_learner(values, ["a", "b", "c"])
    assert choice.learner == "a" and choice.draws == 3
    assert choice.success_probability == {"a": 1.0, "b": pytest.approx(2 / 3), "c": pytest.approx(1 / 3)}
    tied = SEL.select_learner(np.array([[0.5, 0.6], [0.5, 0.6]]), ["x", "y"])
    assert tied.learner == "y"
    exact = SEL.select_learner(np.array([[0.5, 0.5]]), ["first", "second"])
    assert exact.learner == "first"
    for bad in (np.zeros((0, 2)), np.zeros(3), np.array([[np.nan, 0.0]])):
        with pytest.raises(ValueError):
            SEL.select_learner(bad, ["x", "y"])


def test_dirichlet_mixture_recovers_regimes_and_bic_keeps_single():
    rng = np.random.default_rng(0)
    det = np.zeros((60, 4), dtype=int)
    det[np.arange(60), rng.integers(0, 4, 60)] = 20
    uni = np.stack([rng.multinomial(20, rng.dirichlet(np.full(4, 50.0))) for _ in range(60)])
    mixture = post.fit_dirichlet_mixture(np.vstack([det, uni]))
    assert mixture["components"] == 2
    assert np.allclose(mixture["weights"], 0.5, atol=0.02)
    assert np.exp(mixture["log_alpha"][0]) < 1e-3 < 10 < np.exp(mixture["log_alpha"][1])
    resp = post.mixture_responsibilities(np.vstack([det[:3], uni[:3]]), mixture, np.full(4, 0.25))
    assert np.all(resp[:3, 0] > 0.99) and np.all(resp[3:, 1] > 0.99)
    single = post.fit_dirichlet_mixture(uni)
    assert single["components"] == 1
    assert np.allclose(post.mixture_responsibilities(uni, single, np.full(4, 0.25)), 1.0)
    tiny = post.fit_dirichlet_mixture(np.array([[3, 0], [0, 3]]))
    assert tiny["components"] == 1


def test_row_log_likelihood_matches_direct_formula():
    import math as m
    rows = np.array([[2, 0, 1], [0, 0, 4]])
    base = np.array([0.5, 0.25, 0.25])
    alpha = 1.7
    direct = [sum(m.lgamma(alpha * b + n) - m.lgamma(alpha * b) for b, n in zip(base, row))
              - (m.lgamma(alpha + sum(row)) - m.lgamma(alpha)) for row in rows]
    assert np.allclose(post.row_log_likelihood(rows, alpha, base), direct)


def test_mixture_posterior_draws_are_valid():
    rng = np.random.default_rng(12)
    P, R, beta, rho, term = random_mdp(rng)
    beta[:3] = np.eye(3)[[0, 1, 2]]
    data = simulate(rng, P, R, beta, rho, term, 600)
    stats = post.pilot_statistics(data, rho, num_states=7, num_actions=3, gamma=0.95)
    model = post.fit_posterior(stats, behavior_prior="mixture", transition_prior="mixture")
    described = model.describe()
    assert "behavior_mixture" in described and "transition_mixture" in described
    sample, report = post.sample_mdps(model, rng, 6, minimum_dynamic_range=0.01)
    assert report["returned"] == len(sample.P) > 0
    assert np.allclose(sample.beta.sum(-1), 1.0) and np.allclose(sample.P.sum(-1), 1.0)
    with pytest.raises(ValueError):
        post.fit_posterior(stats, behavior_prior="hierarchical")


def test_exact_transport_and_tabular_bwd_properties():
    from scripts.analysis.oeh_final_published import bellman_wasserstein, episode_returns, exact_transport
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert exact_transport(cost, np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert exact_transport(cost, np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    P, R, beta, rho, term = random_mdp(rng)
    data = simulate(rng, P, R, beta, rho, term, 300)
    value = bellman_wasserstein(data, 7, 3, 0.95)
    assert np.isfinite(value)
    returns = episode_returns(data)
    assert len(returns) == int((data["terminal"] | data["truncated"]).sum())


def test_offline_theory_measures_detect_support_and_scale():
    P, R, rho, term = chain()
    good = H.offline_theory_measures(P, R, np.tile([0.3, 0.7], (6, 1)), rho, 0.95, terminal=term, horizon=50)
    poor = H.offline_theory_measures(P, R, np.tile([0.9, 0.1], (6, 1)), rho, 0.95, terminal=term, horizon=50)
    trap = H.offline_theory_measures(P, R, np.tile([1.0, 0.0], (6, 1)), rho, 0.95, terminal=term, horizon=50)
    assert good["log_concentrability"] < poor["log_concentrability"]
    assert good["log_intrinsic_bound"] < poor["log_intrinsic_bound"]
    assert good["log_clipped_concentrability"] <= good["log_concentrability"]
    assert all(value == math.inf for value in trap.values())


def test_offline_horizon_reports_degenerate_reward_range():
    P, R, rho, term = chain()
    R[:] = 0.0
    result = H.offline_effective_horizon(P, R, np.full((6, 2), 0.5), rho, 0.95, terminal=term,
                                         tau=0.95, horizon=20, form="risk")
    assert result.status == "degenerate_range" and result.value == math.inf


@pytest.mark.parametrize("form", ["risk", "coverage_risk", "missing_mass_risk"])
def test_risk_forms_are_monotone_in_needed_coverage(form):
    P, R, rho, term = chain()
    values = [H.offline_effective_horizon(P, R, np.tile([1 - p, p], (6, 1)), rho, 0.95, terminal=term,
                                          tau=0.95, horizon=50, form=form).value for p in (0.6, 0.3, 0.1)]
    assert values == sorted(values) and all(math.isfinite(v) for v in values)
    converged = H.offline_effective_horizon(P, R, np.tile([0.4, 0.6], (6, 1)), rho, 0.95, terminal=term,
                                            tau=0.95, horizon=50, form=form, depth_rule="converged")
    assert converged.depth in (None, 50)
    with pytest.raises(ValueError):
        H.offline_effective_horizon(P, R, np.tile([0.4, 0.6], (6, 1)), rho, 0.95, terminal=term,
                                    tau=0.95, horizon=50, form="other")


def test_asymptotic_ceilings():
    P, R, rho, term = chain()
    full = H.asymptotic_ceilings(P, R, np.tile([0.5, 0.5], (6, 1)), rho, 0.95,
                                 supported=np.ones((6, 2), dtype=bool), terminal=term)
    assert full["support_ceiling"] == pytest.approx(1.0) and full["behavior_ceiling"] < 1.0
    blocked = np.ones((6, 2), dtype=bool)
    blocked[:, 1] = False
    assert H.asymptotic_ceilings(P, R, np.tile([0.5, 0.5], (6, 1)), rho, 0.95, supported=blocked,
                                 terminal=term)["support_ceiling"] < 0.5
    R0 = np.zeros_like(R)
    assert math.isnan(H.asymptotic_ceilings(P, R0, np.tile([0.5, 0.5], (6, 1)), rho, 0.95,
                                            supported=blocked, terminal=term)["behavior_ceiling"])


def test_risk_profile_curves_decrease_with_budget_and_coverage():
    P, R, rho, term = chain()
    budgets = np.array([12, 24, 60, 120, 240, 600])
    good = H.risk_profile(P, R, np.tile([0.1, 0.9], (6, 1)), rho, 0.95, terminal=term, tau=0.95,
                          horizon=50, budgets=budgets)
    poor = H.risk_profile(P, R, np.tile([0.7, 0.3], (6, 1)), rho, 0.95, terminal=term, tau=0.95,
                          horizon=50, budgets=budgets)
    for key in ("noise_regret", "missing_mass"):
        assert np.all(np.diff(good[key]) <= 1e-12) and np.all(good[key] <= poor[key] + 1e-12)
    assert good["slack"] == pytest.approx(0.05)
    flat = H.risk_profile(P, np.zeros_like(R), np.tile([0.5, 0.5], (6, 1)), rho, 0.95, terminal=term,
                          tau=0.95, horizon=50, budgets=budgets)
    assert np.all(np.isnan(flat["noise_regret"])) and math.isnan(flat["slack"])
