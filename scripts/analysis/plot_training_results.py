"""Export development training figures and an evidence-linked analysis.

Run with --run-dir <completed queue directory> --output-dir <new directory>.
Install requirements-figures.txt. Figures aggregate runs, not independent MDPs;
they deliberately do not invent confidence intervals from pooled seeds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = args.run_dir
    source = root / 'training'
    summary = json.loads((source / 'offline_rl_summary.json').read_text(encoding='utf-8'))
    if summary['invalid_registered_runs'] or summary['invalid_bc_runs']:
        raise ValueError('Invalid fits require investigation before plotting')
    curves = read_csv(source / 'learning_curve_summary.csv')
    labels = read_csv(source / 'empirical_sample_complexity.csv')
    comparisons = read_csv(root / 'calibrated/calibrated_comparison.csv')
    measurements = json.loads((root / 'measurements.json').read_text(encoding='utf-8'))
    status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
    oeh = json.loads((root / 'oeh/oeh_validation_report.json').read_text(encoding='utf-8'))
    # Validate byte identity of the raw training evidence before using summaries.
    manifest = json.loads((source / 'artifact_manifest.json').read_text(encoding='utf-8'))
    for name, info in manifest['outputs'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Training hash mismatch: {name}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    names = {'cfqi_tabular': 'CFQI (tabular)', 'iql_tabular': 'IQL (tabular)',
             'behavior_cloning_tabular': 'BC (sanity)'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for algorithm, label in names.items():
        rows = sorted((r for r in curves if r['algorithm'] == algorithm),
                      key=lambda r: int(r['training_size_multiplier']))
        x = [int(r['training_size_multiplier']) for r in rows]
        for ax, field in zip(axes[0], ('median_normalized_return', 'success_rate')):
            ax.plot(x, [float(r[field]) for r in rows], marker='o', label=label)
            ax.set_xscale('log')
            ax.set_xticks(x, labels=[str(v) for v in x])
            ax.set_xlabel('Training size / (states x actions)')
            ax.grid(alpha=.2)
    axes[0, 0].axhline(.95, linestyle='--', color='gray', label='Success threshold')
    axes[0, 0].set_ylabel('Median normalized return')
    axes[0, 1].set_ylabel('Fraction of individual fits with return >= 0.95')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    families = sorted({r['policy_family'] for r in labels})
    x = np.arange(len(families))
    for offset, algorithm in zip((-.18, .18), ('cfqi_tabular', 'iql_tabular')):
        rates = []
        for family in families:
            rows = [r for r in labels if r['algorithm'] == algorithm and r['policy_family'] == family]
            rates.append(sum(r['n_emp_censored'].lower() == 'true' for r in rows) / len(rows))
        axes[1, 0].bar(x + offset, rates, width=.36, label=names[algorithm])
    axes[1, 0].set_xticks(x, labels=[f.replace('_', '\n') for f in families], fontsize=8)
    axes[1, 0].set_ylabel('Right-censored group fraction')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(fontsize=8)
    stages = [r for r in measurements if r['wall_seconds'] > 10]
    axes[1, 1].barh([r['name'].replace('_', ' ') for r in stages],
                   [r['wall_seconds'] / 60 for r in stages])
    axes[1, 1].set_xlabel('Measured wall time (minutes)')
    fig.suptitle('Expanded development: 48 MDPs, 10 seeds, candidate hybrid backend')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(args.output_dir / f'training_overview.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, algorithm in zip(axes, ('cfqi_tabular', 'iql_tabular')):
        rows = [r for r in comparisons if r['algorithm'] == algorithm and
                r['metric'] == 'brier' and r['baseline_is_oracle'].lower() == 'false']
        for i, row in enumerate(rows):
            delta, lo, hi = (float(row[k]) for k in ('difference', 'ci_low', 'ci_high'))
            ax.plot([lo, hi], [i, i], color='gray')
            significant = row['significant_after_holm'].lower() == 'true'
            ax.scatter([delta], [i], color='tab:blue' if significant else 'gray')
        ax.set_yticks(range(len(rows)), labels=[r['baseline'] for r in rows])
        ax.axvline(0, linestyle='--', color='black', linewidth=.8)
        ax.set_title(names[algorithm])
        ax.set_xlabel('OEH Brier - baseline Brier (negative favors OEH)')
    fig.suptitle('Paired development comparisons: 95% cluster-bootstrap intervals\n'
                 'Blue markers: significant after Holm; eligible subset only')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(args.output_dir / f'primary_brier_comparison.{ext}', dpi=180)
    plt.close(fig)
    # Matplotlib emits trailing spaces in SVG path data; normalize exports.
    for path in args.output_dir.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in
                        path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    results = {
        'scope': 'Expanded development only; not external neural SOTA or sealed evaluation',
        'pipeline_status': status,
        'total_wall_minutes': sum(r['wall_seconds'] for r in measurements) / 60,
        'training': {a: summary[a] for a in ('cfqi_tabular', 'iql_tabular')},
        'learning_curves': summary['learning_curve_diagnostics'],
        'oracle_undefined': oeh['check_4_k_dagger_vs_exact'].get('oracle_undefined', {}),
        'oracle_failures': oeh['check_4_k_dagger_vs_exact']['oracle_failures'],
        'evidence_curves': oeh['check_3_evidence_curve'],
        'significant_fair_comparisons': [r for r in comparisons if
            r['baseline_is_oracle'].lower() == 'false' and r['significant_after_holm'].lower() == 'true'],
        'limitations': ['Candidate GPU backend; full-grid reference parity unverified',
                        '100-update budget sensitivity pending',
                        'Pooled learning curves have no independent-cluster confidence intervals',
                        'N_emp median uses finite groups only; censoring must be retained',
                        'Primary comparisons use eligible OEH subset, not all 240 groups per learner'],
        'input_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv')},
    }
    (args.output_dir / 'analysis.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(args.output_dir), 'wall_minutes': results['total_wall_minutes'],
                      'pipeline_passed': status['all_stages_passed']}))


if __name__ == '__main__':
    main()
