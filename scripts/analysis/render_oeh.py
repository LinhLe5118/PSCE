"""Render copies of the final published figures without refitting models."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"code"))
from oeh.evidence import FINAL_BUNDLE, read_json, read_csv_rows, verify_repository

def figures(output, comparison, identifier, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'behavior_cloning_tabular':'BC', 'cfqi_tabular':'CFQI', 'iql_tabular':'IQL',
        'discrete_bcq_author':'BCQ', 'discrete_cql_d3rlpy':'CQL', 'pib_spibb_n10_author':'Pi-b SPIBB',
        'vi_lcb_cb005_author_empirical':'VI-LCB', 'oehv4e_selector':'OEH-v4E', 'oeh_v5_r34':'Archived reference',
        identifier:'OEH'}
    rows = [r for r in comparison if r['panel'] == 'stochastic_primary']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, field, scale, title in zip(axes, ('mean_normalized_return', 'mean_success_rate'), (1, 100),
                                     ('Mean normalized return', 'Success rate (%)')):
        ax.barh([labels.get(r['algorithm'], r['algorithm']) for r in rows], [scale*float(r[field]) for r in rows],
            color=['#2171b5' if r['algorithm'] == identifier else '#969696' for r in rows])
        ax.set_xlabel(title); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Original E1 x10: 84 stochastic MDPs, 420 groups, 10 training seeds/group (post-hoc)')
    for extension in ('png', 'pdf'): fig.savefig(output/f'comparison_x10.{extension}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for ax, indices, scale, title in zip(axes, ((0, 2, 4), (1, 3, 5)), (1, 100),
        ('Paired normalized return gain', 'Paired success gain (percentage points)')):
        effects = np.asarray(report['paired_effects'])[list(indices)]*scale
        lower = np.asarray(report['basic_one_sided_lower'])[list(indices)]*scale
        for i in range(3):
            ax.plot([i, i], [lower[i], effects[i]], color='#2171b5')
            ax.plot(i, lower[i], marker='_', color='#2171b5', markersize=12)
        ax.scatter(range(3), effects, color='#2171b5')
        ax.axhline(0, color='black', linewidth=.8); ax.set_xticks(range(3), ['vs CFQI', 'vs OEH-v4E', 'vs archived reference'])
        ax.set_ylabel(title); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Point estimates and one-sided basic-bootstrap 95% lower bounds\n2,000 refits; diagnostic after adaptive E1 reuse')
    for extension in ('png', 'pdf'): fig.savefig(output/f'paired_gains.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Figure copies must stay outside immutable artifacts/')
    verify_repository(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    figures(output, read_csv_rows(ROOT/FINAL_BUNDLE/'comparison_x10.csv'), 'oeh_v5_r44',
            read_json(ROOT/FINAL_BUNDLE/'raw/report.json'))


if __name__ == '__main__':
    main()
