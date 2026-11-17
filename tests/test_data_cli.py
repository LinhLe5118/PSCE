"""Fresh-process routing and malformed input controls use software fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oeh.data import RIDGE_GRID, load_panel
from oeh.reference import METHODS
from scripts.analysis import route_oeh
from test_router import fixture_panel


def document():
    p = fixture_panel()
    result = dict(schema_version=1, methods=[*METHODS, 'new_learner'], return_ridges=list(RIDGE_GRID), groups=[])
    for g, outcomes in p['runs'].items():
        labels = {m:[dict(seed=s, normalized_return=v[0], success=bool(v[1])) for s,v in seeds.items()]
                  for m,seeds in outcomes.items()}
        labels['new_learner'] = [dict(seed=s, normalized_return=.96, success=True) for s in range(10)]
        result['groups'].append(dict(mdp_id=g[0], policy_family=g[1], outer_fold=p['outer'][g[0]], panel='software',
            features={m:p['features'][(*g,m)] for m in METHODS}, outcomes=labels))
    return result


def save(tmp_path, value):
    path = tmp_path/'input.json'; path.write_text(json.dumps(value), 'utf-8'); return path


def test_fresh_process_router_writes_once_and_covers_every_group(tmp_path):
    input_path = save(tmp_path, document()); output = tmp_path/'new/result.json'
    result = subprocess.run([sys.executable, '-m', 'scripts.analysis.route_oeh', '--input', str(input_path),
                             '--output', str(output)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    value = json.loads(output.read_text('utf-8'))
    assert len(value['decisions']) == 15 and value['new_mdps_generated'] == 0
    assert value['confirmation'] is False and value['policy_training_performed'] is False
    with pytest.raises(FileExistsError): route_oeh.run(input_path, output)
    with pytest.raises(ValueError, match='immutable'):
        route_oeh.run(input_path, route_oeh.ROOT/'artifacts/forbidden.json')


@pytest.mark.parametrize('module', ['route_oeh', 'verify_oeh', 'render_oeh'])
def test_cli_starts_without_optional_training_or_gpu_dependencies(module):
    result = subprocess.run([sys.executable, '-m', f'scripts.analysis.{module}', '--help'], capture_output=True)
    assert result.returncode == 0 and b'usage:' in result.stdout


@pytest.mark.parametrize('case', [
    'schema', 'methods', 'ridges', 'duplicate_group', 'fold', 'split_mdp', 'cross_panel',
    'missing_method', 'missing_seed', 'duplicate_seed', 'nonfinite', 'threshold',
    'anchors', 'blocks', 'dimensions', 'empty', 'fold_coverage',
])
def test_invalid_data_fails_before_fitting(tmp_path, case):
    value = document(); g = value['groups'][0]
    if case == 'schema': value['schema_version'] = 2
    if case == 'methods': value['methods'][0] = 'unregistered_reference'
    if case == 'ridges': value['return_ridges'] = [0]
    if case == 'duplicate_group': value['groups'].append(copy.deepcopy(g))
    if case == 'fold': g['outer_fold'] = 5
    if case in ('split_mdp', 'cross_panel'):
        extra = copy.deepcopy(g); extra['policy_family'] = 'b'
        extra['outer_fold' if case == 'split_mdp' else 'panel'] = 1 if case == 'split_mdp' else 'other'
        value['groups'].append(extra)
    if case == 'missing_method': del g['outcomes']['new_learner']
    if case == 'missing_seed': g['outcomes']['new_learner'].pop()
    if case == 'duplicate_seed': g['outcomes']['new_learner'][-1]['seed'] = 0
    if case == 'nonfinite': g['outcomes']['new_learner'][0]['normalized_return'] = float('nan')
    if case == 'threshold': g['outcomes']['new_learner'][0]['success'] = False
    if case == 'anchors': del g['features'][METHODS[0]]
    if case == 'blocks': g['features'][METHODS[0]] = {}
    if case == 'dimensions': value['groups'][1]['features'][METHODS[0]] = {k:[1.] for k in g['features'][METHODS[0]]}
    if case == 'empty': value['groups'] = []
    if case == 'fold_coverage': value['groups'] = [g for g in value['groups'] if g['outer_fold'] != 4]
    with pytest.raises(ValueError): load_panel(save(tmp_path, value))


def test_multiple_behaviors_keep_the_whole_mdp_together(tmp_path):
    value = document(); extra = copy.deepcopy(value['groups'][0]); extra['policy_family'] = 'b'
    value['groups'].append(extra)
    p = load_panel(save(tmp_path, value))
    assert len(p['runs']) == 16 and len(p['outer']) == 15


def test_figures_render_from_verified_evidence_without_overwriting(tmp_path):
    output = tmp_path/'figures'
    result = subprocess.run([sys.executable, '-m', 'scripts.analysis.render_oeh', '--output', str(output)],
                            capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert {p.name for p in output.iterdir()} == {'comparison_x10.png', 'comparison_x10.pdf',
                                                'paired_gains.png', 'paired_gains.pdf'}
    assert all(p.stat().st_size > 1000 for p in output.iterdir())
