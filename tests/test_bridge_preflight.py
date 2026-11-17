"""A malformed published MDP must fail before any rollout or label creation."""

import numpy as np
import pytest

from benchmarks.bridge.preflight_npz import inspect


NAME = "MiniGrid-Empty-5x5-v0"  # 37 incl. terminal, 36 stored rows, 3 actions.


def test_base_archive_schema_accepts_valid_arrays(tmp_path):
    path = tmp_path / "consolidated.npz"
    transitions = np.zeros((36, 3), dtype=np.int64)
    transitions[1, 2] = -1
    np.savez(path, transitions=transitions, rewards=np.zeros((36, 3)))
    result = inspect(NAME, path)
    assert result["status"] == "BASE_ARRAY_SCHEMA_PASS_ONLY"
    assert result["terminal_edges"] == 1
    assert result["metadata_states_including_terminal"] == 37
    assert "sticky-state construction" in result["remaining"]


@pytest.mark.parametrize("bad_transition", [-2, 36])
def test_base_archive_schema_rejects_invalid_transition(tmp_path, bad_transition):
    path = tmp_path / "consolidated.npz"
    transitions = np.zeros((36, 3), dtype=np.int64)
    transitions[0, 0] = bad_transition
    np.savez(path, transitions=transitions, rewards=np.zeros((36, 3)))
    with pytest.raises(ValueError, match="Transition ID outside"):
        inspect(NAME, path)


def test_base_archive_schema_rejects_terminal_row(tmp_path):
    path = tmp_path / "consolidated.npz"
    np.savez(path, transitions=np.zeros((37, 3), dtype=np.int64), rewards=np.zeros((37, 3)))
    with pytest.raises(ValueError, match="Expected both arrays"):
        inspect(NAME, path)
