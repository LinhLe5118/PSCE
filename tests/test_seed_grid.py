"""Full original 120-MDP, 600-group and 6,000-seed public outcome audit."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from oeh.evidence import REFERENCE_BUNDLE as REFERENCE, FINAL_BUNDLE, read_csv_rows
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/FINAL_BUNDLE

def test_published_seed_grid_reference_outcomes_and_independent_aggregation():
    reference_rows = read_csv_rows(ROOT/REFERENCE/'selected_seed_outcomes.csv')
    old = {(r['mdp_id'], r['policy_family'], int(r['training_seed'])):r for r in reference_rows}
    assert len(old) == len(reference_rows) == 6000
    original_decisions = {(r['mdp_id'], r['policy_family']):r for r in read_csv_rows(ROOT/REFERENCE/'raw/decisions.csv')}
    decisions = {(r['mdp_id'], r['policy_family']):r for r in read_csv_rows(OUTPUT/'raw/decisions.csv')}
    assert set(decisions) == set(original_decisions) and len(decisions) == 600
    original_table = read_csv_rows(ROOT/REFERENCE/'comparison_x10.csv')
    library = {r['algorithm'] for r in original_table if not r['algorithm'].startswith('oeh')}
    assert len(library) == 7
    seeds = read_csv_rows(OUTPUT/'selected_seed_outcomes.csv')
    assert len(seeds) == 6000
    seen, clusters, groups, panels = set(), defaultdict(list), defaultdict(list), defaultdict(set)
    for row in seeds:
        g = row['mdp_id'], row['policy_family']; seed = int(row['training_seed']); identity = *g, seed
        assert identity in old and identity not in seen and seed in range(10)
        seen.add(identity); ref = old[identity]; route = decisions[g]
        assert row['selected_algorithm'] == route['selected_algorithm'] and row['selected_algorithm'] in library
        assert row['r34_algorithm'] == original_decisions[g]['selected_algorithm'] == ref['selected_algorithm']
        assert route['outer_fold'] == original_decisions[g]['outer_fold']
        assert row['buffer_sha256'] == ref['buffer_sha256'] and row['panel'] == ref['panel'] == route['panel']
        for field in ('cfqi_return', 'cfqi_success', 'v4e_return', 'v4e_success'):
            assert row[field] == ref[field]
        assert row['r34_return'] == ref['normalized_return'] and row['r34_success'] == ref['success']
        values = []
        for nr, sr in (('normalized_return', 'success'), ('cfqi_return', 'cfqi_success'),
                       ('v4e_return', 'v4e_success'), ('r34_return', 'r34_success')):
            value = float(row[nr]); assert np.isfinite(value) and row[sr] in ('True', 'False')
            assert (row[sr] == 'True') == (value >= .95)
            values.extend((value, int(row[sr] == 'True')))
        if row['selected_algorithm'] == row['r34_algorithm']:
            assert row['normalized_return'] == ref['normalized_return'] and row['success'] == ref['success']
        clusters[row['panel'], g[0]].append(values); groups[g].append(values)
        panels[row['panel']].add(g[0])
    assert seen == set(old) and all(len(values) == 10 for values in groups.values())
    assert {name:len(mdps) for name, mdps in panels.items()} == {'stochastic_primary':84, 'deterministic_control':36}
    assert all(len(values) == 50 for values in clusters.values())
    report = json.loads((OUTPUT/'raw/report.json').read_text('utf-8'))
    comparison = read_csv_rows(OUTPUT/'comparison_x10.csv')
    assert len(comparison) == 20 and all(row in comparison for row in original_table)
    for name, mdps in panels.items():
        means = np.mean([np.mean(clusters[name, mdp], axis=0) for mdp in sorted(mdps)], axis=0)
        np.testing.assert_allclose(means, report['panels'][name], atol=1e-12, rtol=0)
        point = next(r for r in comparison if r['panel'] == name and r['algorithm'] == 'oeh_v5_r44')
        np.testing.assert_allclose([float(point['mean_normalized_return']), float(point['mean_success_rate'])],
                                  means[:2], atol=1e-12, rtol=0)
        assert int(point['base_mdps']) == len(mdps) and int(point['runs']) == len(mdps)*50
