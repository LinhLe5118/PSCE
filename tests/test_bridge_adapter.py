"""Check sticky state memory and reward aggregation in the BRIDGE adapter."""
from __future__ import annotations

import numpy as np

from benchmarks.bridge import adapter


def test_sticky_transition_and_terminal_reward_mixture(tmp_path, monkeypatch):
    name = "MiniGrid-Fixture-v0"
    directory = tmp_path / name
    directory.mkdir()
    # Both actions terminate but have different rewards. Published sticky MDPs
    # retain separate terminal states for the previous executed action.
    np.savez(directory / "consolidated.npz",
             transitions=np.array([[1, 1], [-1, -1]]),
             rewards=np.array([[0.0, 0.0], [0.0, 1.0]]))
    monkeypatch.setattr(adapter, "DATA", tmp_path)
    mdp = adapter.load_sticky_mdp(name)
    assert mdp.num_states == 5  # reset, previous action 0/1, terminal 0/1
    assert mdp.initial_state_distribution[0] == 1
    np.testing.assert_allclose(mdp.P[0, 1, 2], 1)
    np.testing.assert_allclose(mdp.P[1, 1, 3], .25)
    np.testing.assert_allclose(mdp.P[1, 1, 4], .75)
    np.testing.assert_allclose(mdp.R[1, 1, 3], 0)
    np.testing.assert_allclose(mdp.R[1, 1, 4], 1)
