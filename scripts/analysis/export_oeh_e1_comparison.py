"""Export a passed E1 reuse study, preserving raw bytes and every selected seed.

This command never fits a model or generates data. It verifies the original
seals and saved routing before producing descriptive tables and figures.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from evaluation.e1_reuse import load_e1_panel
from evaluation.e1_routing import METHODS, score_routes
from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json


def verified_hashes(base, hashes):
    base = base.resolve()
    for name, digest in hashes.items():
        path = (base/name.replace('\\', '/')).resolve()
        if not path.is_relative_to(base) or sha256_file(path) != digest:
            raise ValueError(f'Changed or escaping sealed file: {name}')


def export(bundle, output):
    bundle, output = bundle.resolve(), output.resolve()
    if not bundle.is_relative_to(ROOT/'tmp/oeh_e1_reuse') or not output.is_relative_to(ROOT/'artifacts/week7'):
        raise ValueError('Export paths must stay in their registered repository directories')
    seal = json.loads((bundle/'artifact_manifest.json').read_text('utf-8'))
    verified_hashes(bundle, seal['artifact_sha256'])
    plan = json.loads((bundle/'plan.json').read_text('utf-8'))
    verified_hashes(ROOT, plan['source_hashes']); verified_hashes(ROOT, plan['input_hashes'])
    report = json.loads((bundle/'report.json').read_text('utf-8'))
    if (report.get('screen_pass') is not True or report.get('confirmation') is not False
            or report.get('new_mdps_generated') != 0
            or not np.all(np.asarray(report['paired_effects']) > 0)
            or not np.all(np.asarray(report['basic_one_sided_lower']) > 0)):
        raise ValueError('Only an unchanged passed post-hoc study can be exported')
    with np.load(bundle/'bootstrap.npz') as archive:
        if archive['draws'].shape != (plan['config']['bootstrap_draws'], 4) or not np.isfinite(archive['draws']).all():
            raise ValueError('Incomplete bootstrap evidence')
        np.testing.assert_allclose(archive['observed'], report['paired_effects'], rtol=0, atol=1e-12)
    loaded = load_e1_panel(ROOT)
    rows = read_csv_rows(bundle/'decisions.csv'); choices = {}
    for row in rows:
        g = (row['mdp_id'], row['policy_family'])
        if g in choices or g not in loaded['runs'] or row['selected_algorithm'] not in METHODS:
            raise ValueError('Duplicate, missing or invalid saved route')
        if int(row['outer_fold']) != loaded['outer'][g[0]] or g[0] not in loaded['panels'][row['panel']]:
            raise ValueError('Saved route does not match original E1 identity')
        choices[g] = row['selected_algorithm']
    if set(choices) != set(loaded['runs']):
        raise ValueError('Incomplete original 600-group route grid')
    baseline = read_csv_rows(ROOT/'artifacts/week6/oehv4e_confirmatory_v1_1/x10_method_comparison.csv')
    comparison = list(baseline); groups = []; clusters = []; seeds = []; mechanisms = {}
    for name, mdps in loaded['panels'].items():
        selected = {g:m for g, m in choices.items() if g[0] in mdps}
        mean, per_mdp = score_routes(loaded, selected, mdps)
        np.testing.assert_allclose(mean, report['panels'][name], rtol=0, atol=1e-12)
        counts = dict(base_mdps=len(mdps), behavior_groups=len(selected), runs=10*len(selected))
        for row in baseline:
            if row['panel'] == name:
                if any(int(row[field]) != value for field, value in counts.items()):
                    raise ValueError('Baseline comparison uses a different cohort/seed grid')
                if not np.isfinite([float(row['mean_normalized_return']), float(row['mean_success_rate'])]).all():
                    raise ValueError('Nonfinite baseline table')
        for method, source, nr, sr, confirmatory in (
                ('oehv4e_selector', 'frozen_e1_selector', mean[4], mean[5], True),
                ('oeh_v5_r34', 'posthoc_e1_matched_routing', mean[0], mean[1], False)):
            comparison.append(dict(panel=name, algorithm=method, source_type=source, **counts,
                mean_normalized_return=float(nr), mean_success_rate=float(sr), in_confirmatory_family=confirmatory))
        success_wins, success_losses = Counter(), Counter()
        for g, method in sorted(selected.items()):
            reference, v4e = METHODS[0], loaded['v4e_decisions'][g]
            values = []
            for seed in range(10):
                a, b, c = (loaded['runs'][g][m][seed] for m in (method, reference, v4e))
                values.append(np.array([*a, *b, *c]))
                for comparator, other in (('cfqi', b), ('v4e', c)):
                    success_wins[comparator] += int(a[1] > other[1]); success_losses[comparator] += int(a[1] < other[1])
                seeds.append(dict(panel=name, mdp_id=g[0], policy_family=g[1], training_seed=seed,
                    selected_algorithm=method, normalized_return=a[0], success=bool(a[1]),
                    cfqi_return=b[0], cfqi_success=bool(b[1]), v4e_return=c[0], v4e_success=bool(c[1]),
                    buffer_sha256=loaded['training_records'][(*g, 10, seed)]['sha256']))
            averages = np.mean(values, axis=0)
            groups.append(dict(panel=name, mdp_id=g[0], policy_family=g[1], selected_algorithm=method,
                normalized_return=averages[0], success_rate=averages[1],
                nr_vs_cfqi=averages[0]-averages[2], sr_vs_cfqi=averages[1]-averages[3],
                nr_vs_v4e=averages[0]-averages[4], sr_vs_v4e=averages[1]-averages[5]))
        for mdp, values in sorted(per_mdp.items()):
            clusters.append(dict(panel=name, mdp_id=mdp, normalized_return=values[0], success_rate=values[1],
                nr_vs_cfqi=values[0]-values[2], sr_vs_cfqi=values[1]-values[3],
                nr_vs_v4e=values[0]-values[4], sr_vs_v4e=values[1]-values[5]))
        mechanisms[name] = dict(selected_algorithms=dict(Counter(selected.values())),
            success_wins=dict(success_wins), success_losses=dict(success_losses),
            success_net_gain={key:success_wins[key]-success_losses[key] for key in ('cfqi', 'v4e')},
            mdp_effect_signs={field:dict(Counter('positive' if row[field] > 1e-12 else 'negative' if row[field] < -1e-12 else 'zero'
                for row in clusters if row['panel'] == name)) for field in ('nr_vs_cfqi','sr_vs_cfqi','nr_vs_v4e','sr_vs_v4e')})
    if len(seeds) != 6000 or len(groups) != 600 or len(clusters) != 120:
        raise ValueError('Export dropped original MDPs/groups/seeds')
    reserve_output_dir(output)
    shutil.copytree(bundle, output/'raw')
    for name, values in (('comparison_x10.csv', comparison), ('group_metrics.csv', groups),
                         ('mdp_metrics.csv', clusters), ('selected_seed_outcomes.csv', seeds)):
        write_csv_rows(output/name, values, list(values[0]))
    all_mean = np.mean([[r['normalized_return'], r['success_rate']] for r in clusters], axis=0)
    write_json(output/'mechanism_audit.json', dict(panels=mechanisms, all_120_mean=all_mean.tolist(),
        all_120_descriptive_only=True, posthoc_consumed_e1=True, confirmation=False,
        export_refits=0, new_mdps_generated=0, original_manifest_sha256=loaded['manifest_sha256']))
    figures(output, comparison, report)
    write_json(output/'manifest.json', dict(
        artifact_sha256={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()},
        export_source_sha256=sha256_file(Path(__file__)), original_input_hashes=plan['input_hashes'],
        scientific_source_hashes=plan['source_hashes'], original_bundle_manifest_sha256=sha256_file(bundle/'artifact_manifest.json'),
        scientific_runtime_seconds=report['runtime_seconds'], posthoc_consumed_e1=True, confirmation=False,
        export_refits=0, new_mdps_generated=0))
    print(json.dumps(dict(output=str(output), mdps=120, groups=600, selected_seeds=6000, export_refits=0, new_mdps_generated=0)))


def figures(output, comparison, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'behavior_cloning_tabular':'BC', 'cfqi_tabular':'CFQI', 'iql_tabular':'IQL',
        'discrete_bcq_author':'BCQ', 'discrete_cql_d3rlpy':'CQL', 'pib_spibb_n10_author':'Pi-b SPIBB',
        'vi_lcb_cb005_author_empirical':'VI-LCB', 'oehv4e_selector':'OEH-v4E', 'oeh_v5_r34':'OEH-v5 R34'}
    panel = [r for r in comparison if r['panel'] == 'stochastic_primary']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, field, scale, title in zip(axes, ('mean_normalized_return','mean_success_rate'), (1,100),
                                     ('Mean normalized return','Success rate (%)')):
        values = [scale*float(r[field]) for r in panel]
        colors = ['#2171b5' if r['algorithm'] == 'oeh_v5_r34' else '#969696' for r in panel]
        ax.barh([labels[r['algorithm']] for r in panel], values, color=colors)
        ax.set_xlabel(title); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Original E1 x10: 84 stochastic MDPs, 420 groups, 10 training seeds/group (post-hoc)')
    for extension in ('png','pdf'): fig.savefig(output/f'comparison_x10.{extension}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for ax, indices, scale, title in zip(axes, ((0,2),(1,3)), (1,100), ('Paired normalized return gain','Paired success gain (percentage points)')):
        effects = np.asarray(report['paired_effects'])[list(indices)]*scale
        lower = np.asarray(report['basic_one_sided_lower'])[list(indices)]*scale
        ax.errorbar([0,1], effects, yerr=[effects-lower, np.zeros(2)], fmt='o', capsize=5, color='#2171b5')
        ax.axhline(0, color='black', linewidth=.8); ax.set_xticks([0,1], ['vs CFQI','vs OEH-v4E'])
        ax.set_ylabel(title); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Point estimate and one-sided basic-bootstrap 95% lower bound\n2,000 refits; diagnostic after adaptive E1 reuse')
    for extension in ('png','pdf'): fig.savefig(output/f'paired_gains.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); export(args.bundle, args.output)


if __name__ == '__main__':
    main()
