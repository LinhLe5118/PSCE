"""Render the paper's numerical tables from sealed, audited E1 evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from oeh.evidence import FINAL_BUNDLE, verify_repository

NAMES = (
    ('oeh_v5_r44', 'OEH'),
    ('cfqi_tabular', 'CFQI'),
    ('oehv4e_selector', 'Frozen E1 selector'),
    ('oeh_v5_r34', 'Earlier router'),
    ('iql_tabular', 'IQL'),
    ('behavior_cloning_tabular', 'BC'),
    ('discrete_bcq_author', 'BCQ'),
    ('discrete_cql_d3rlpy', 'CQL'),
    ('pib_spibb_n10_author', r'$\Pi_b$-SPIBB'),
    ('vi_lcb_cb005_author_empirical', 'VI-LCB'),
)
CONTRASTS = ('CFQI', 'Frozen E1 selector', 'Earlier router')


def render_tables(comparison, report):
    by_key = {(r['panel'], r['algorithm']): r for r in comparison}
    expected = {(panel, method) for panel in ('stochastic_primary', 'deterministic_control')
                for method, _ in NAMES}
    if len(comparison) != len(expected) or set(by_key) != expected:
        raise ValueError('Incomplete or duplicate comparison rows')
    if report['contrast_order'] != ['NR_VS_CFQI', 'SR_VS_CFQI', 'NR_VS_V4E', 'SR_VS_V4E',
                                    'NR_VS_R34', 'SR_VS_R34']:
        raise ValueError('Unexpected contrast order')
    lines = [
        '% Generated from sealed comparison_x10.csv and raw/report.json; do not edit by hand.',
        r'\begin{table*}[t]',
        r'\caption{Matched x10 outcomes on the original E1 cohort. Each method has 4,200 primary and 1,800 control seed outcomes. The final OEH row is post-hoc development evidence; the control panel is descriptive.}',
        r'\label{tab:all-methods}',
        r'\centering\small',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Method & Primary return & Primary success (\%) & Control return & Control success (\%) \\',
        r'\midrule',
    ]
    for method, name in NAMES:
        p, c = by_key['stochastic_primary', method], by_key['deterministic_control', method]
        numbers = [f"{float(p['mean_normalized_return']):.4f}",
                   f"{100*float(p['mean_success_rate']):.2f}",
                   f"{float(c['mean_normalized_return']):.4f}",
                   f"{100*float(c['mean_success_rate']):.2f}"]
        cells = [name, *numbers]
        if method == NAMES[0][0]:
            cells = [r'\textbf{' + cell + '}' for cell in cells]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '',
              r'\begin{table}[t]',
              r'\caption{Paired gains on 84 stochastic MDPs (post-hoc). LBs are individual basic one-sided 95\% whole-MDP bootstrap lower bounds, not simultaneous or independent-confirmatory limits.}',
              r'\label{tab:gains}', r'\centering\footnotesize',
              r'\begin{tabular}{lrrrr}', r'\toprule',
              r'Comparator & $\Delta R$ & LB & $\Delta S$ (pp) & LB (pp) \\', r'\midrule']
    effects, lower = report['paired_effects'], report['basic_one_sided_lower']
    for i, name in enumerate(CONTRASTS):
        values = (f'{effects[2*i]:.4f}', f'{lower[2*i]:.4f}',
                  f'{100*effects[2*i+1]:.2f}', f'{100*lower[2*i+1]:.2f}')
        lines.append(f'{name} & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the historical table fixture differs')
    args = parser.parse_args()
    verify_repository(ROOT)
    bundle = ROOT / FINAL_BUNDLE
    with (bundle / 'comparison_x10.csv').open(encoding='utf-8', newline='') as stream:
        comparison = list(csv.DictReader(stream))
    report = json.loads((bundle / 'raw/report.json').read_text('utf-8'))
    rendered = render_tables(comparison, report)
    destination = ROOT / 'tests/fixtures/paper_history/tables.tex'
    if args.check:
        if not destination.is_file() or destination.read_text('utf-8') != rendered:
            raise SystemExit('Paper tables differ from sealed artifact; rerun export_paper_tables')
        print('Paper tables match sealed evidence')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered.encode('utf-8'))
        print(f'Wrote {destination.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
