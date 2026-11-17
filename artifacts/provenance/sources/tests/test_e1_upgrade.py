"""Six paired effects, exact refits, source seals and single-use withdrawal.

Every outcome here is an invented software row, not a research environment.
"""
from concurrent.futures import Future
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import e1_upgrade as runner, withdrawal
from evaluation.e1_routing import METHODS, score_routes
from evaluation.io import sha256_file, write_csv_rows, write_json


def test_all_six_contrasts_include_r34():
    np.testing.assert_allclose(runner.contrast_values([1., .8, .4, .2, .5, .3, .7, .6]), [.6, .6, .5, .5, .3, .2])
    with pytest.raises(ValueError): runner.contrast_values([1.]*6)
    with pytest.raises(ValueError): runner.contrast_values([np.nan]*8)


def test_reject_hash_tampering_and_path_escape(tmp_path):
    path = tmp_path/'software.txt'; path.write_text('sealed', 'utf-8')
    digest = sha256_file(path); runner.verify_hashes(tmp_path, {'software.txt':digest})
    path.write_text('changed', 'utf-8')
    with pytest.raises(ValueError): runner.verify_hashes(tmp_path, {'software.txt':digest})
    with pytest.raises(ValueError): runner.verify_hashes(tmp_path, {'../other.txt':digest})


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    owned = ['code/oeh_fixtureupgrade.py', 'tests/test_oeh_fixtureupgrade.py']
    for name in owned:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n', 'utf-8')
    protocol = tmp_path/'docs/protocol/software.json'
    write_json(protocol, dict(attempt_id='r99', candidate_module='oeh_fixtureupgrade', sources=owned,
        evaluation_reuse=True, confirmation=False, new_mdps_generated=0, budget=10, bootstrap_draws=2000,
        bootstrap_seed=20260921, one_sided_alpha=.05, contrasts=runner.CONTRASTS, reference=runner.REFERENCE))
    groups = [('primary', 'a'), ('control', 'a')]; runs = {}
    for g in groups:
        runs[g] = {}
        for method, wins, remaining in zip(METHODS, (2, 4, 8), (.5, .6, .7)):
            runs[g][method] = {seed:(.98 if seed < wins else remaining, int(seed < wins)) for seed in range(10)}
    loaded = dict(runs=runs, features={}, outer={'primary':0, 'control':1},
        v4e_decisions={g:METHODS[0] for g in groups}, return_ridges=(.1,), input_hashes={},
        manifest_sha256='software', panels={'stochastic_primary':('primary',), 'deterministic_control':('control',)})
    state = dict(fits=0, incomplete=False, negative=False, uncertain=False, comparisons=0, execution_error=False,
                 submissions=0, library=False, prepares=0, library_scores=0)
    def reference_fit(panel, mdps, **kwargs):
        state['fits'] += 1
        choices = {g:METHODS[0] if kwargs.get('original_v4e') else METHODS[1] for g in groups if g[0] in mdps}
        return choices, {g:{m:[0., 0.] for m in METHODS} for g in choices}, []
    def candidate_fit(panel, mdps, **kwargs):
        state['fits'] += 1
        choices = {g:METHODS[0] if state['negative'] else 'software_extra' if state['library'] else METHODS[2]
                   for g in groups if g[0] in mdps}
        if state['incomplete']: choices = {}
        return choices, {g:{m:[0., 0.] for m in (*METHODS, 'software_extra')} for g in choices}, []
    def prepare(panel, root, original):
        state['prepares'] += 1
        assert root == tmp_path and original is loaded
        if not state['library']: return panel
        runs = {g:dict(values, software_extra=values[METHODS[2]]) for g, values in panel['runs'].items()}
        return dict(panel, runs=runs, candidate_methods=(*METHODS, 'software_extra'))
    def library_score(panel, choices, mdps, **kwargs):
        if not state['library']: return score_routes(panel, choices, mdps, **kwargs)
        from oeh_library_gains import score_routes as extended
        state['library_scores'] += 1
        return extended(panel, choices, mdps, **kwargs)
    candidate = SimpleNamespace(fit_routes=candidate_fit, prepare_panel=prepare, score_routes=library_score)
    reference = SimpleNamespace(fit_routes=reference_fit)
    reference_path = tmp_path/runner.REFERENCE
    write_json(reference_path/'manifest.json', dict(scientific_source_hashes={}, original_input_hashes={}, artifact_sha256={}))
    write_csv_rows(reference_path/'raw/decisions.csv', [dict(mdp_id=g[0], policy_family=g[1], selected_algorithm=METHODS[1]) for g in groups],
        ['mdp_id', 'policy_family', 'selected_algorithm'])
    ref_scores = {name:score_routes(loaded, {g:METHODS[1] for g in groups if g[0] in mdps}, mdps)[0].tolist()
                  for name, mdps in loaded['panels'].items()}
    write_json(reference_path/'raw/report.json', dict(panels=ref_scores))
    original_compare = runner.compare_routes
    def compared(panel, choices, v4e, r34, mdps, multiplicities=None, **kwargs):
        name = 'stochastic_primary' if mdps == ('primary',) else 'deterministic_control'
        bundle = tmp_path/'tmp/oeh_e1_reuse/r99'; choice_file = bundle/f'{name}_choices.json'
        seal = json.loads((bundle/f'{name}_choice_seal.json').read_text('utf-8'))
        assert seal['sha256'] == sha256_file(choice_file)
        state['comparisons'] += 1
        return original_compare(panel, choices, v4e, r34, mdps, multiplicities, **kwargs)
    class Pool:
        def __init__(self, **kwargs): kwargs['initializer'](*kwargs['initargs'])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, function, task):
            state['submissions'] += 1; future = Future()
            if state['execution_error'] and task[0] == 0:
                future.set_exception(ValueError('fixture draw failure')); return future
            index, values = function(task)
            if state['uncertain']: values[-2:] = [1., 1.]
            future.set_result((index, values)); return future
    monkeypatch.setattr(runner, 'load_e1_panel', lambda root:loaded)
    monkeypatch.setattr(runner.importlib, 'import_module', lambda name:reference if name == 'oeh_interaction_gains' else candidate)
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(runner, 'compare_routes', compared)
    monkeypatch.setattr(withdrawal.subprocess, 'run', lambda *a, **kw:SimpleNamespace(returncode=1))
    return tmp_path, protocol, state


def test_success_exact_refits_six_effects_and_no_rerun(invocation):
    root, protocol, state = invocation; report = runner.run(root, protocol, 1)
    assert report['screen_pass'] and not report['confirmation'] and report['new_mdps_generated'] == 0
    assert state['fits'] == 6006 and state['comparisons'] == 2002 and state['submissions'] == 2000
    assert report['bootstrap_completed'] == 2000
    bundle = root/'tmp/oeh_e1_reuse/r99'
    assert np.load(bundle/'bootstrap.npz')['draws'].shape == (2000, 6)
    manifest = json.loads((bundle/'artifact_manifest.json').read_text('utf-8'))
    runner.verify_hashes(bundle, manifest['artifact_sha256'])
    with pytest.raises(ValueError, match='Consumed'): runner.run(root, protocol, 1)


def test_registered_library_is_prepared_once_and_scored_in_every_refit(invocation):
    root, protocol, state = invocation; state['library'] = True
    report = runner.run(root, protocol, 1)
    assert report['screen_pass'] and report['bootstrap_completed'] == 2000
    assert state['prepares'] == 1 and state['fits'] == 6006 and state['library_scores'] == 4004
    assert state['comparisons'] == 2002 and state['submissions'] == 2000


@pytest.mark.parametrize('case', ['negative', 'uncertain', 'incomplete', 'execution_error'])
def test_failed_attempt_withdraws_without_metrics_or_reroll(invocation, case):
    root, protocol, state = invocation; state[case] = True
    if case in ('incomplete', 'execution_error'):
        with pytest.raises(ValueError): runner.run(root, protocol, 1)
        if case == 'incomplete': assert state['comparisons'] == 0
        else: assert state['submissions'] < 2000
    else:
        report = runner.run(root, protocol, 1); assert not report['screen_pass']
        if case == 'negative': assert state['fits'] == 6 and report['bootstrap_completed'] == 0
    assert not (root/'tmp/oeh_e1_reuse/r99').exists() and not protocol.exists()
    assert not (root/'code/oeh_fixtureupgrade.py').exists()
    ledger = json.loads((root/'docs/reassessment/withdrawal_ledger/r99.json').read_text('utf-8'))
    assert ledger['status'] == 'withdrawn' and 'paired_effects' not in ledger


def test_relaxed_gate_rejected_before_loading_or_consuming(invocation):
    root, protocol, state = invocation
    config = json.loads(protocol.read_text('utf-8')); config['bootstrap_draws'] = 20; write_json(protocol, config)
    with pytest.raises(ValueError, match='six-contrast'): runner.run(root, protocol, 1)
    assert state['fits'] == 0 and not (root/'tmp/oeh_e1_reuse/r99').exists()


def test_frozen_reference_mismatch_cannot_be_scored(invocation):
    root, protocol, state = invocation
    file = root/runner.REFERENCE/'raw/report.json'
    data = json.loads(file.read_text('utf-8')); data['panels']['stochastic_primary'][0] = 99.; write_json(file, data)
    with pytest.raises(ValueError, match='reference scores'): runner.run(root, protocol, 1)
    assert state['comparisons'] == 0


def test_cli_starts_in_fresh_process():
    result = subprocess.run([sys.executable, '-m', 'scripts.analysis.run_oeh_upgrade', '--help'], capture_output=True)
    assert result.returncode == 0 and b'--protocol' in result.stdout
