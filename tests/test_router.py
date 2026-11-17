"""Software fixtures for a method-agnostic library extension and its pairing."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from oeh import router as candidate


def fixture_panel():
    panel = dict(runs={}, features={}, outer={}, v4e_decisions={}, return_ridges=(.1, 1.))
    for i in range(15):
        mdp = f'software{i:02}'; panel['outer'][mdp] = i%5; g = (mdp, 'a')
        panel['runs'][g] = {}; panel['v4e_decisions'][g] = candidate.CORE[0]
        for j, m in enumerate(candidate.CORE):
            value = .8+.03*j+.01*(i%3)
            panel['runs'][g][m] = {seed:(value, int(value >= .95)) for seed in range(10)}
            panel['features'][(*g, m)] = {f:(i/15, j, float(i%3)) for f in candidate.FEATURES}
    return panel


def test_additional_result_grid_and_buffer_hashes_are_strict():
    panel = fixture_panel(); records = {}; rows = []
    for g in panel['runs']:
        for seed in range(10):
            records[(*g, 10, seed)] = {'sha256':f'buffer-{g[0]}-{seed}'}
            rows.append(dict(mdp_id=g[0], policy_family=g[1], algorithm='new_learner', training_seed=str(seed),
                training_size_multiplier='10', normalized_return='.96', success='True', run_status='observed',
                buffer_sha256=records[(*g, 10, seed)]['sha256']))
    prepared = candidate.attach_results(panel, rows, records, ('new_learner',))
    assert 'new_learner' not in next(iter(panel['runs'].values()))
    assert prepared['candidate_methods'] == (*candidate.CORE, 'new_learner')
    with pytest.raises(ValueError, match='Incomplete'): candidate.attach_results(panel, rows[:-1], records, ('new_learner',))
    with pytest.raises(ValueError, match='Duplicate'): candidate.attach_results(panel, rows+[rows[0]], records, ('new_learner',))
    corrupted = copy.deepcopy(rows); corrupted[0]['buffer_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'): candidate.attach_results(panel, corrupted, records, ('new_learner',))
    corrupted = copy.deepcopy(rows); corrupted[0]['run_status'] = 'failed'
    with pytest.raises(ValueError, match='Failed'): candidate.attach_results(panel, corrupted, records, ('new_learner',))


def test_core_forecasts_exact_and_new_test_labels_cannot_enter_own_fit():
    panel = fixture_panel(); panel['candidate_methods'] = (*candidate.CORE, 'new_learner')
    for g in panel['runs']:
        panel['runs'][g]['new_learner'] = {seed:(.96 if seed < 3 else .8, int(seed < 3)) for seed in range(10)}
    mdps = tuple(panel['outer']); choices, forecasts, _ = candidate.fit_routes(panel, mdps)
    _, original, _ = candidate.reference.fit_routes(panel, mdps)
    for g in forecasts:
        for method in candidate.CORE:
            assert forecasts[g][method] == original[g][method]
    changed = copy.deepcopy(panel)
    for g in changed['runs']:
        if panel['outer'][g[0]] == 0:
            changed['runs'][g]['new_learner'] = {seed:(-3., 0) for seed in range(10)}
    other, predictions, _ = candidate.fit_routes(changed, mdps)
    for g in choices:
        if panel['outer'][g[0]] == 0:
            assert choices[g] == other[g] and forecasts[g] == predictions[g]
    mean, clusters = candidate.score_routes(panel, {g:'new_learner' for g in panel['runs']}, mdps)
    assert len(clusters) == 15
    np.testing.assert_allclose(mean[:2], [.848, .3])
    with pytest.raises(ValueError): candidate.score_routes(panel, {g:'unregistered' for g in panel['runs']}, mdps)
