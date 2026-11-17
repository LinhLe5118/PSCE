"""Verify refinement provenance and report fixed-depth comparison diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--baseline-dir', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.run_dir/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete' and not report['errors']
    for name, expected in report['source_sha256'].items():
        assert digest(ROOT/name) == expected, name
    for name, expected in report['artifact_sha256'].items():
        assert digest(args.run_dir/name) == expected, name
    assert digest(args.dataset_dir/'manifest.json') == report['manifest_sha256']
    assert digest(args.dataset_dir/'data_only_index.json') == report['index_sha256']
    for rec in report['inputs']:
        assert digest(args.dataset_dir/rec['file']) == rec['sha256'], rec['file']
    rows = read_rows(args.run_dir/'summary.csv')
    curves = read_rows(args.run_dir/'curves.csv')
    baseline = {r['dataset_id']: r for r in read_rows(args.baseline_dir/'summary.csv')}
    baseline_matches = 0
    for row in rows:
        if row['variant'] == 'bernstein_pessimistic':
            for current, old in [('min_upper', 'v2_min_upper'), ('max_upper', 'v2_max_upper')]:
                assert float(row[current]) == float(baseline[row['dataset_id']][old])
            baseline_matches += 1
    empirical = {(r['dataset_id'], r['depth']): float(r['oracle_regret'])
                 for r in curves if r['variant'] == 'bernstein_empirical'}
    empirical_matches = 0
    for row in curves:
        if row['variant'] == 'kl_empirical':
            assert float(row['oracle_regret']) == empirical[row['dataset_id'], row['depth']]
            empirical_matches += 1
    assert len(rows) == report['num_variant_cases'] == 4*len(report['inputs'])
    assert len(curves) == report['num_depth_rows'] == len(rows)*101
    comparison = {}
    for variant in report['summary']:
        group = [r for r in rows if r['variant'] == variant and r['structural_family'] != 'lock_structured']
        gain = np.array([float(r['depth100_regret'])-float(r['selected_regret']) for r in group])
        comparison[variant] = {
            'cases': len(group), 'better_than_depth100': int(sum(gain > 1e-8)),
            'worse_than_depth100': int(sum(gain < -1e-8)),
            'mean_gain_vs_depth100': float(gain.mean()),
            'mean_depth100_regret': float(np.mean([float(r['depth100_regret']) for r in group])),
        }
    families = {}
    empirical_rows = [r for r in rows if r['variant'] == 'kl_empirical']
    for family in sorted({r['structural_family'] for r in empirical_rows}):
        group = [r for r in empirical_rows if r['structural_family'] == family]
        families[family] = {
            'cases': len(group), 'nonflat': sum(float(r['upper_range']) > 1e-9 for r in group),
            'mean_gain_vs_depth0': float(np.mean([float(r['gain_vs_depth0']) for r in group])),
        }
    audit = {
        'status': 'passed', 'report_sha256': digest(args.run_dir/'report.json'),
        'audit_source_sha256': digest(Path(__file__)),
        'baseline_summary_sha256': digest(args.baseline_dir/'summary.csv'),
        'source_artifact_manifest_index_input_hashes': 'verified',
        'baseline_cases_matched': baseline_matches, 'empirical_depth_regrets_matched': empirical_matches,
        'stochastic_depth100_comparison': comparison, 'kl_empirical_family_diagnostics': families,
        'scope': 'Descriptive post-run audit, not independent confirmation or significance testing.',
    }
    (args.run_dir/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
