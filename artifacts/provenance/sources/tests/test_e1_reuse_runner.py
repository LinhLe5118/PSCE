"""Durable logs/seals, exact route coverage and single-use/withdrawal controls.

All outcomes below are invented software fixtures, never research MDPs.
"""
from concurrent.futures import Future
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.io import sha256_file, write_json
from evaluation import withdrawal
from scripts.analysis import run_oeh_e1_reuse as runner


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    names = ['code/oeh_fixture.py', 'tests/test_oeh_fixture.py']
    for name in names:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# software fixture\n', 'utf-8')
    protocol = tmp_path/'docs/protocol/fixture.json'
    write_json(protocol, dict(attempt_id='r99', candidate_module='oeh_fixture', evaluation_reuse=True,
        new_mdps_generated=0, budget=10, sources=names, bootstrap_draws=2, bootstrap_seed=20260921, one_sided_alpha=.05))
    groups = {('primary', 'a'):{}, ('control', 'a'):{}}
    loaded = dict(runs=groups, features={}, outer={'primary':0, 'control':1},
        v4e_decisions={g:'cfqi_tabular' for g in groups}, return_ridges=(.1,), input_hashes={},
        manifest_sha256='fixture', panels={'stochastic_primary':('primary',), 'deterministic_control':('control',)})
    state = dict(mean=np.array([1., .6, .5, .2, .6, .2]), incomplete=False, scored=0)
    def fit(panel, mdps, **kwargs):
        choices = {g:'cfqi_tabular' for g in groups if g[0] in mdps}
        if state['incomplete']:
            choices = {}
        predictions = {g:{'cfqi_tabular':[0., 0.]} for g in choices}
        return choices, predictions, []
    def score(panel, choices, mdps, **kwargs):
        bundle = tmp_path/'tmp/oeh_e1_reuse/r99'
        name = 'stochastic_primary' if mdps == ('primary',) else 'deterministic_control'
        seal = json.loads((bundle/f'{name}_choice_seal.json').read_text('utf-8'))
        assert seal['sha256'] == sha256_file(bundle/f'{name}_choices.json')
        state['scored'] += 1
        return state['mean'], {}
    module = SimpleNamespace(fit_routes=fit, score_routes=score)
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, task):
            index, weights = task
            future = Future()
            future.set_result((index, np.array([.5, .4, .4, .4])))
            return future
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_e1_panel', lambda root:loaded)
    monkeypatch.setattr(runner.importlib, 'import_module', lambda name:module)
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(withdrawal.subprocess, 'run', lambda *a, **k:SimpleNamespace(returncode=1))
    return tmp_path, protocol, state


def test_success_preserves_log_config_results_and_manifest(invocation):
    root, protocol, state = invocation
    report = runner.run(protocol, 1)
    assert report['screen_pass'] and not report['confirmation'] and report['new_mdps_generated'] == 0
    assert state['scored'] == 2
    bundle = root/'tmp/oeh_e1_reuse/r99'
    manifest = json.loads((bundle/'artifact_manifest.json').read_text('utf-8'))
    for name, digest in manifest['artifact_sha256'].items():
        assert sha256_file(bundle/name) == digest
    assert 'source/config sealed' in (bundle/'run.log').read_text('utf-8')
    assert np.load(bundle/'bootstrap.npz')['draws'].shape == (2, 4)
    with pytest.raises(ValueError, match='Consumed'):
        runner.run(protocol, 1)


def test_negative_screen_deletes_owned_artifacts_and_retains_only_ledger(invocation):
    root, protocol, state = invocation
    state['mean'] = np.array([.2, .1, .5, .2, .6, .2])
    report = runner.run(protocol, 1)
    assert not report['screen_pass']
    assert not (root/'tmp/oeh_e1_reuse/r99').exists()
    assert not protocol.exists() and not (root/'code/oeh_fixture.py').exists()
    ledger = json.loads((root/'docs/reassessment/withdrawal_ledger/r99.json').read_text('utf-8'))
    assert ledger['status'] == 'withdrawn' and 'paired_effects' not in ledger


def test_partial_original_routes_fail_before_scoring_and_are_withdrawn(invocation):
    root, protocol, state = invocation
    state['incomplete'] = True
    with pytest.raises(ValueError, match='exact route reproduction'):
        runner.run(protocol, 1)
    assert state['scored'] == 0
    assert not (root/'tmp/oeh_e1_reuse/r99').exists()


def test_invalid_source_fails_before_reserving_bundle(invocation):
    root, protocol, state = invocation
    config = json.loads(protocol.read_text('utf-8')); config['sources'].append('missing.py')
    write_json(protocol, config)
    with pytest.raises(FileNotFoundError):
        runner.run(protocol, 1)
    assert not (root/'tmp/oeh_e1_reuse/r99').exists()


def test_cluster_draws_preserve_fixed_outer_strata():
    mdps = tuple(f'm{i}' for i in range(10)); outer = {mdp:i%5 for i, mdp in enumerate(mdps)}
    draws = list(runner.paired_draws(mdps, outer, 4, 20260921))
    assert draws == list(runner.paired_draws(mdps, outer, 4, 20260921))
    for index, weights in draws:
        assert sum(weights.values()) == 10
        for fold in range(5):
            assert sum(weights[m] for m in mdps if outer[m] == fold) == 2
