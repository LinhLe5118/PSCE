"""Export a passed six-contrast upgrade without fitting or generating data."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from evaluation.e1_reuse import load_e1_panel
from evaluation.e1_upgrade import CONTRASTS, REFERENCE, verify_hashes
from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json


def validate_report(report, archive):
    if (report.get('screen_pass') is not True or report.get('confirmation') is not False
            or report.get('new_mdps_generated') != 0 or report.get('bootstrap_completed') != 2000
            or report.get('contrast_order') != list(CONTRASTS)
            or np.asarray(report.get('paired_effects')).shape != (6,)
            or np.asarray(report.get('basic_one_sided_lower')).shape != (6,)):
        raise ValueError('Only a complete passed six-contrast post-hoc study may be exported')
    draws, observed = archive['draws'], archive['observed']
    if draws.shape != (2000, 6) or not np.isfinite(draws).all():
        raise ValueError('Incomplete bootstrap evidence')
    np.testing.assert_allclose(observed, report['paired_effects'], rtol=0, atol=1e-12)
    lower = 2*observed-np.quantile(draws, .95, axis=0)
    np.testing.assert_allclose(lower, report['basic_one_sided_lower'], rtol=0, atol=1e-12)
    if not np.all(observed > 0) or not np.all(lower > 0):
        raise ValueError('Upgrade did not pass every registered contrast')


def export(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if not bundle.is_relative_to(ROOT/'tmp/oeh_e1_reuse') or not output.is_relative_to(ROOT/'artifacts/week7'):
        raise ValueError('Export must stay inside registered study/output directories')
    seal = json.loads((bundle/'artifact_manifest.json').read_text('utf-8'))
    verify_hashes(bundle, seal['artifact_sha256'])
    plan = json.loads((bundle/'plan.json').read_text('utf-8'))
    verify_hashes(ROOT, plan['source_hashes']); verify_hashes(ROOT, plan['input_hashes'])
    report = json.loads((bundle/'report.json').read_text('utf-8'))
    with np.load(bundle/'bootstrap.npz') as archive:
        validate_report(report, archive)
    loaded = load_e1_panel(ROOT)
    panel = {key:loaded[key] for key in ('runs', 'features', 'outer', 'v4e_decisions', 'return_ridges')}
    candidate = importlib.import_module(plan['config']['candidate_module'])
    if hasattr(candidate, 'prepare_panel'):
        panel = candidate.prepare_panel(panel, ROOT, loaded)
    old_rows = read_csv_rows(ROOT/REFERENCE/'raw/decisions.csv')
    r34 = {(r['mdp_id'], r['policy_family']):r['selected_algorithm'] for r in old_rows}
    rows = read_csv_rows(bundle/'decisions.csv'); choices = {}
    for row in rows:
        g = row['mdp_id'], row['policy_family']
        if (g in choices or g not in loaded['runs'] or int(row['outer_fold']) != loaded['outer'][g[0]]
                or g[0] not in loaded['panels'][row['panel']] or row['reference_algorithm'] != r34[g]):
            raise ValueError('Saved route does not match the original E1 / R34 grid')
        choices[g] = row['selected_algorithm']
    if set(choices) != set(loaded['runs']):
        raise ValueError('Incomplete original 600-group routing')
    comparison = read_csv_rows(ROOT/REFERENCE/'comparison_x10.csv')
    identifier = f'oeh_v5_{plan["config"]["attempt_id"]}'
    groups, clusters, seeds, mechanisms = [], [], [], {}
    for name, mdps in loaded['panels'].items():
        selected = {g:m for g, m in choices.items() if g[0] in mdps}
        first, per_mdp = candidate.score_routes(panel, selected, mdps)
        second, per_r34 = candidate.score_routes(dict(panel, v4e_decisions=r34), selected, mdps)
        mean = np.concatenate((first, second[4:6]))
        np.testing.assert_allclose(mean, report['panels'][name], rtol=0, atol=1e-12)
        counts = dict(base_mdps=len(mdps), behavior_groups=len(selected), runs=10*len(selected))
        for row in comparison:
            if row['panel'] == name and any(int(row[field]) != value for field, value in counts.items()):
                raise ValueError('Comparison table uses a different cohort or seed grid')
        comparison.append(dict(panel=name, algorithm=identifier, source_type='posthoc_e1_matched_routing', **counts,
            mean_normalized_return=float(mean[0]), mean_success_rate=float(mean[1]), in_confirmatory_family=False))
        wins, losses = Counter(), Counter()
        for g, method in sorted(selected.items()):
            values = []
            for seed in range(10):
                a, b, c, d = [panel['runs'][g][m][seed] for m in (method, 'cfqi_tabular', panel['v4e_decisions'][g], r34[g])]
                values.append([*a, *b, *c, *d])
                for comparator, other in (('cfqi', b), ('v4e', c), ('r34', d)):
                    wins[comparator] += int(a[1] > other[1]); losses[comparator] += int(a[1] < other[1])
                seeds.append(dict(panel=name, mdp_id=g[0], policy_family=g[1], training_seed=seed,
                    selected_algorithm=method, normalized_return=a[0], success=bool(a[1]),
                    cfqi_return=b[0], cfqi_success=bool(b[1]), v4e_return=c[0], v4e_success=bool(c[1]),
                    r34_algorithm=r34[g], r34_return=d[0], r34_success=bool(d[1]),
                    buffer_sha256=loaded['training_records'][(*g, 10, seed)]['sha256']))
            averages = np.mean(values, axis=0)
            record = dict(panel=name, mdp_id=g[0], policy_family=g[1], selected_algorithm=method,
                normalized_return=averages[0], success_rate=averages[1])
            for comparator, offset in (('cfqi', 2), ('v4e', 4), ('r34', 6)):
                record[f'nr_vs_{comparator}'] = averages[0]-averages[offset]
                record[f'sr_vs_{comparator}'] = averages[1]-averages[offset+1]
            groups.append(record)
        for mdp, values in sorted(per_mdp.items()):
            combined = np.concatenate((values, per_r34[mdp][4:6]))
            record = dict(panel=name, mdp_id=mdp, normalized_return=combined[0], success_rate=combined[1])
            for comparator, offset in (('cfqi', 2), ('v4e', 4), ('r34', 6)):
                record[f'nr_vs_{comparator}'] = combined[0]-combined[offset]
                record[f'sr_vs_{comparator}'] = combined[1]-combined[offset+1]
            clusters.append(record)
        fields = [f'{endpoint}_vs_{method}' for method in ('cfqi', 'v4e', 'r34') for endpoint in ('nr', 'sr')]
        mechanisms[name] = dict(selected_algorithms=dict(Counter(selected.values())),
            routing_changes_vs_r34={str(k):v for k, v in Counter((r34[g], selected[g]) for g in selected).items()},
            success_wins=dict(wins), success_losses=dict(losses),
            success_net_gain={key:wins[key]-losses[key] for key in ('cfqi', 'v4e', 'r34')},
            mdp_effect_signs={field:dict(Counter('positive' if r[field] > 1e-12 else 'negative' if r[field] < -1e-12 else 'zero'
                for r in clusters if r['panel'] == name)) for field in fields})
    if len(seeds) != 6000 or len(groups) != 600 or len(clusters) != 120:
        raise ValueError('Export dropped original MDPs/groups/seeds')
    reserve_output_dir(output); shutil.copytree(bundle, output/'raw')
    for name, values in (('comparison_x10.csv', comparison), ('group_metrics.csv', groups),
                         ('mdp_metrics.csv', clusters), ('selected_seed_outcomes.csv', seeds)):
        write_csv_rows(output/name, values, list(values[0]))
    write_json(output/'mechanism_audit.json', dict(panels=mechanisms,
        all_120_mean=np.mean([[r['normalized_return'], r['success_rate']] for r in clusters], axis=0).tolist(),
        all_120_descriptive_only=True, posthoc_consumed_e1=True, confirmation=False, export_refits=0,
        new_mdps_generated=0, original_manifest_sha256=loaded['manifest_sha256']))
    figures(output, comparison, identifier, report)
    write_json(output/'manifest.json', dict(
        artifact_sha256={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()},
        export_source_sha256=sha256_file(Path(__file__)), original_input_hashes=plan['input_hashes'],
        scientific_source_hashes=plan['source_hashes'], scientific_runtime_seconds=report['runtime_seconds'],
        environment=plan.get('environment'), original_bundle_manifest_sha256=sha256_file(bundle/'artifact_manifest.json'),
        posthoc_consumed_e1=True, confirmation=False, export_refits=0, new_mdps_generated=0))
    print(json.dumps(dict(output=str(output), mdps=120, groups=600, selected_seeds=6000, export_refits=0)))


def figures(output, comparison, identifier, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'behavior_cloning_tabular':'BC', 'cfqi_tabular':'CFQI', 'iql_tabular':'IQL',
        'discrete_bcq_author':'BCQ', 'discrete_cql_d3rlpy':'CQL', 'pib_spibb_n10_author':'Pi-b SPIBB',
        'vi_lcb_cb005_author_empirical':'VI-LCB', 'oehv4e_selector':'OEH-v4E', 'oeh_v5_r34':'OEH-v5 R34',
        identifier:identifier.replace('oeh_v5_', 'OEH-v5 ').upper().replace('OEH-V5', 'OEH-v5')}
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
        ax.axhline(0, color='black', linewidth=.8); ax.set_xticks(range(3), ['vs CFQI', 'vs OEH-v4E', 'vs R34'])
        ax.set_ylabel(title); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Point estimates and one-sided basic-bootstrap 95% lower bounds\n2,000 refits; diagnostic after adaptive E1 reuse')
    for extension in ('png', 'pdf'): fig.savefig(output/f'paired_gains.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); export(args.bundle, args.output)


if __name__ == '__main__':
    main()
