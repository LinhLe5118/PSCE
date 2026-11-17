"""Exact SACo pair count remains unchanged after tabular acceleration."""

import numpy as np

from benchmarks.bridge.expanded_original_scores_preoutcome import exact_tabular_pairs
from scripts.analysis.oeh_final_tqsaco_original import unique_state_actions_exact


def test_exact_tabular_pairs_matches_author_string_count() -> None:
    rng = np.random.default_rng(20261008)
    for states_count, actions_count, rows in ((3, 2, 50), (17, 6, 400), (27, 9, 800)):
        states = rng.integers(states_count, size=rows)
        actions = rng.integers(actions_count, size=rows)
        one_hot = np.eye(states_count, dtype=np.float32)[states]
        reference = unique_state_actions_exact(one_hot, actions)
        assert exact_tabular_pairs(one_hot, actions) == reference
        assert exact_tabular_pairs(states, actions) == reference
