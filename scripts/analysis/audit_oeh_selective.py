"""Audit selective-method results against frozen inputs and prior KL baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
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
    for path, expected in report['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    for path, expected in report['artifact_sha256'].items():
        assert sha(args.run_dir/path) == expected, path
    for name, key in [('manifest.json','manifest_sha256'),('data_only_index.json','index_sha256')]:
        assert sha(args.dataset_dir/name) == report[key]
    for row in report['inputs']:
        assert sha(args.dataset_dir/row['file']) == row['sha256']
    curves, rows = read(args.run_dir/'curves.csv'), read(args.run_dir/'summary.csv')
    assert len(rows) == 3*len(report['inputs']) == report['num_variant_cases']
    assert len(curves) == 101*len(rows) == report['num_depth_rows']
    assert len({(r['dataset_id'],r['variant'],r['depth']) for r in curves}) == len(curves)
    baseline = {(r['dataset_id'],r['depth']):r for r in read(args.baseline_dir/'curves.csv') if r['variant']=='kl_empirical'}
    matches = 0
    for row in curves:
        assert float(row['lower']) <= float(row['oracle_difference'])+1e-8
        assert float(row['upper']) >= float(row['oracle_difference'])-1e-8
        assert float(row['lower']) >= float(row['independent_lower'])-1e-12
        assert float(row['upper']) <= float(row['independent_upper'])+1e-12
        if row['variant'] == 'empirical':
            old = baseline[row['dataset_id'],row['depth']]
            assert abs(float(row['global_upper'])-float(old['upper'])) < 1e-12
            assert abs(float(row['oracle_regret'])-float(old['oracle_regret'])) < 1e-12
            matches += 1
    additional = {}
    for name in ('empirical','pvi','spvi'):
        relevant = [r for r in curves if r['variant']==name and r['identical_policy']=='False']
        narrow = [r for r in relevant if float(r['upper'])-float(r['lower']) < .1]
        additional[name] = dict(
            cases_with_distinct_width_below_0_1_and_nonzero_true_effect=len({r['dataset_id'] for r in narrow if abs(float(r['oracle_difference']))>1e-8}),
            cases_with_narrower_interval_and_nonzero_true_effect=len({r['dataset_id'] for r in relevant
                if abs(float(r['oracle_difference']))>1e-8 and
                float(r['upper'])-float(r['lower']) < float(r['independent_upper'])-float(r['independent_lower'])-1e-9}))
    sto = {name:{r['dataset_id']:r for r in rows if r['variant']==name and r['structural_family']!='lock_structured'}
           for name in ('empirical','pvi','spvi')}
    gains = np.array([float(sto['pvi'][k]['fixed100_regret'])-float(sto['spvi'][k]['fixed100_regret']) for k in sto['pvi']])
    audit = dict(status='passed', source_input_artifact_hashes='verified',
        report_sha256=sha(args.run_dir/'report.json'), audit_source_sha256=sha(Path(__file__)),
        baseline_curves_sha256=sha(args.baseline_dir/'curves.csv'), empirical_depth_matches=matches,
        additional_diagnostics=additional,
        stochastic_spvi_vs_pvi_fixed100=dict(better=int(sum(gains>1e-8)), worse=int(sum(gains< -1e-8)),
            tied=int(sum(abs(gains)<=1e-8)), mean_gain=float(gains.mean())),
        scope='Descriptive audit; post-run diagnostics; no significance or generalization claim')
    (args.run_dir/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    # Standard exportable scientific figure, not an image-generation illustration.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Empirical','PVI','SPVI']
    names = ['empirical','pvi','spvi']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(3)
    vals = [report['stochastic_summary'][n]['global_nonflat'] for n in names]
    axes[0].bar(x, vals, color=['#527a9a','#a2a2a2','#33836c'])
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 165)
    axes[0].set_ylabel('Nonflat global-bound curves / 165 pilots')
    axes[0].set_title('Numerical variation (range > 1e-9)')
    for i, v in enumerate(vals): axes[0].text(i, v+3, str(v), ha='center')
    for offset, field, label in [(-.18,'mean_selected_regret','Minimum global bound'),(.18,'mean_fixed100_regret','Fixed depth 100')]:
        axes[1].bar(x+offset, [report['stochastic_summary'][n][field] for n in names], width=.36, label=label)
    axes[1].set_xticks(x, labels); axes[1].set_ylabel('Mean normalized true regret (lower is better)')
    axes[1].set_title('Actual policy quality'); axes[1].legend(fontsize=8)
    fig.suptitle('Exploratory stochastic development: SPVI adaptation vs controls')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(args.run_dir/f'comparison.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
