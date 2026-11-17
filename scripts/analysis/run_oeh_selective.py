"""Development-only SPVI proposals, coupled differences and LP support audit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'code'))
from oeh_selective import selective_profiles
from scripts.analysis.run_oeh_development_smoke import development_inputs
from evaluation.io import reserve_output_dir, verified_dataset_path, sha256_file, write_json, write_csv_rows


def summarize(rows):
    return dict(cases=len(rows),
        global_nonflat=sum(r['global_range'] > 1e-9 for r in rows),
        global_range_above_0_001=sum(r['global_range'] > .001 for r in rows),
        global_certified=sum(r['global_certified'] for r in rows),
        multiple_policies=sum(r['unique_policies'] > 1 for r in rows),
        mean_selected_regret=float(np.mean([r['selected_regret'] for r in rows])),
        mean_fixed100_regret=float(np.mean([r['fixed100_regret'] for r in rows])),
        mean_reference_regret=float(np.mean([r['reference_regret'] for r in rows])),
        mean_safe_selected_regret=float(np.mean([r['safe_selected_regret'] for r in rows])),
        cases_with_distinct_comparison=sum(r['distinct_comparisons'] > 0 for r in rows),
        cases_with_narrower_distinct_interval=sum(r['narrower_distinct_comparisons'] > 0 for r in rows),
        cases_with_distinct_width_below_0_1=sum(r['distinct_width_below_0_1'] > 0 for r in rows),
        certified_improvement_cases=sum(r['improvement_depth'] is not None for r in rows),
        min_distinct_width=min((r['min_distinct_width'] for r in rows if r['min_distinct_width'] is not None), default=None),
        global_bound_violations=sum(r['global_bound_violations'] for r in rows),
        difference_bound_violations=sum(r['difference_bound_violations'] for r in rows),
        false_improvements=sum(r['false_improvements'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-mdps', type=int)
    args = parser.parse_args()
    manifest, records, labels, index = development_inputs(args.dataset_dir, args.limit_mdps)
    bench = manifest['effective_protocol']['benchmark']
    if bench['gamma'] != .95 or bench['reward_range'] != [0., 1.]:
        raise ValueError('Protocol requires gamma .95 and rewards [0,1]')
    reserve_output_dir(args.output_dir)
    sources = ['code/oeh_selective.py', 'code/oeh_refinements.py', 'code/oeh_certificate.py',
        'code/oeh_certificate_adapter.py', 'code/exact_solver.py', 'tests/test_oeh_selective.py',
        'scripts/analysis/run_oeh_selective.py', 'scripts/analysis/run_oeh_development_smoke.py',
        'environments/benchmark_environments.py', 'evaluation/io.py']
    plan = dict(status='exploratory_development',
        source_sha256={p:sha256_file(ROOT/p) for p in sources},
        manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'), index_sha256=sha256_file(index),
        inputs=[{k:r[k] for k in ('dataset_id','file','sha256','slice')} for r in records],
        design=dict(gamma=.95, reward_bound=1, epsilon=.05, delta=.05, k_max=100,
            proposals=['empirical','pvi','spvi'], confidence='shared KL set for all methods and comparisons',
            proposal_bonus='sqrt(log(S*A*101/delta)/max(n,1)); heuristic, not a certificate',
            reference='empirical proposal depth 0',
            global_selector='earliest depth within 1e-9 of minimum global upper bound',
            safe_selector='maximize coupled lower bound if >1e-9, otherwise empirical reference',
            lp_audit=dict(xi=.25, K=[1,2,5,10], policy='empirical depth100')),
        runtime=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
            threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}),
        limitations=['SPVI is a discounted receding-horizon proposal adaptation, not full author-code replication.',
            'Coupled intervals concern relative policy value; no conversion to global regret.',
            'Identical policies are excluded from nontrivial narrowing counts.',
            'LP-MDP audit checks applicability only; no LP-MDP learner trained.',
            'Pilot cases are clustered by MDP; no independent-case significance claim.'])
    write_json(args.output_dir/'plan.json', plan)
    start = time.perf_counter()
    rows, curves, lookahead, errors, oracles = [], [], [], [], {}
    for i, rec in enumerate(records, 1):
        label = labels[rec['dataset_id']]
        identity = dict(dataset_id=rec['dataset_id'], mdp_id=label['instance_id'],
            policy_family=label['policy_family'], pilot_multiplier=rec['size_multiplier'])
        try:
            with np.load(verified_dataset_path(args.dataset_dir, rec), allow_pickle=False) as z:
                arrays = {k:z[k][:rec['size']] for k in z.files}
            if any(len(x) != rec['size'] for x in arrays.values()):
                raise ValueError('Short buffer')
            profiles, reference, lp = selective_profiles(arrays, rec['initial_state_distribution'],
                num_states=rec['num_states'], num_actions=rec['num_actions'])
            # Oracle scoring is deliberately after ALL data-only estimates.
            from environments.benchmark_environments import make_benchmark_mdp
            from exact_solver import value_iteration, evaluate_policy
            instance = identity['mdp_id']
            if instance not in oracles:
                spec = next(s for s in manifest['instances'] if s['instance_id'] == instance)
                mdp = make_benchmark_mdp(type('Spec', (), spec)())
                best = float(mdp.initial_state_distribution@value_iteration(mdp).values)
                if (mdp.num_states != rec['num_states'] or mdp.num_actions != rec['num_actions']
                        or not np.isclose(best, spec['optimal_return'], atol=1e-8, rtol=1e-8)):
                    raise ValueError('Oracle mismatch')
                oracles[instance] = mdp, best, spec['family']
            mdp, best, family = oracles[instance]
            ref_value = evaluate_policy(mdp, reference).expected_return
            ref_regret = .05*(best-ref_value)
            cache = {reference.tobytes():ref_value}
            for name, profile in profiles.items():
                regrets, differences, distinct = [], [], {}
                global_violations, diff_violations, false_improvements = 0, 0, 0
                for k, pi in enumerate(profile['policies']):
                    key = pi.tobytes()
                    if key not in cache:
                        cache[key] = evaluate_policy(mdp, pi).expected_return
                    reg, diff = .05*(best-cache[key]), .05*(cache[key]-ref_value)
                    ci, upper = profile['intervals'][k], profile['upper'][k]
                    global_violations += int(reg > upper+1e-8)
                    diff_violations += int(diff < ci['lower']-1e-8 or diff > ci['upper']+1e-8)
                    false_improvements += int(ci['lower'] > 1e-9 and diff < -1e-8)
                    if not ci['identical_policy']:
                        distinct[key] = ci
                    curves.append({**identity, 'variant':name, 'depth':k, 'global_upper':float(upper),
                        **ci, 'oracle_regret':reg, 'oracle_difference':diff})
                    regrets.append(reg); differences.append(diff)
                widths = [x['upper']-x['lower'] for x in distinct.values()]
                improved, selected = profile['improvement_depth'], profile['selected_depth']
                rows.append({**identity, 'structural_family':family, 'variant':name,
                    'unique_policies':len({p.tobytes() for p in profile['policies']}),
                    'global_range':float(np.ptp(profile['upper'])),
                    'global_min':float(profile['upper'].min()),
                    'global_certified':bool(profile['upper'].min() <= .05),
                    'selected_depth':selected, 'selected_regret':regrets[selected],
                    'fixed100_regret':regrets[-1], 'reference_regret':ref_regret,
                    'improvement_depth':improved,
                    'safe_selected_regret':ref_regret if improved is None else regrets[improved],
                    'distinct_comparisons':len(distinct),
                    'narrower_distinct_comparisons':sum(
                        x['upper']-x['lower'] < x['independent_upper']-x['independent_lower']-1e-9
                        for x in distinct.values()),
                    'distinct_width_below_0_1':sum(w < .1 for w in widths),
                    'min_distinct_width':min(widths) if widths else None,
                    'global_bound_violations':global_violations,
                    'difference_bound_violations':diff_violations, 'false_improvements':false_improvements})
            lookahead.append({**identity, 'structural_family':family, **lp})
            print(f'[{i}/{len(records)}] {instance} x{rec["size_multiplier"]} '
                  + ', '.join(f'{n}: policies={len({p.tobytes() for p in v["policies"]})}' for n,v in profiles.items()), flush=True)
        except Exception as exc:
            errors.append({**identity, 'error':repr(exc)})
            print(f'[{i}/{len(records)}] ERROR {exc!r}', flush=True)
        write_json(args.output_dir/'progress.json', dict(processed=i,total=len(records),errors=errors))
    for filename, data in [('summary.csv',rows),('curves.csv',curves),('lookahead.csv',lookahead)]:
        if data:
            write_csv_rows(args.output_dir/filename, data, list(data[0]))
    summary = {name:summarize([r for r in rows if r['variant']==name]) for name in ('empirical','pvi','spvi')}
    stochastic = {name:summarize([r for r in rows if r['variant']==name and r['structural_family']!='lock_structured'])
                  for name in ('empirical','pvi','spvi')}
    write_json(args.output_dir/'report.json', {**plan,
        'status':'complete' if not errors else 'completed_with_errors', 'summary':summary,
        'stochastic_summary':stochastic, 'errors':errors, 'num_variant_cases':len(rows),
        'num_depth_rows':len(curves), 'wall_seconds':time.perf_counter()-start,
        'lookahead_summary':{k:sum(r[k] for r in lookahead) for k in
            ('uncertain_rows','total_rows','nominal_rescued_k1','nominal_rescued_k10',
             'confidence_rescued_k1','confidence_rescued_k10')},
        'artifact_sha256':{p.name:sha256_file(p) for p in args.output_dir.glob('*.csv')}})
    return int(bool(errors) or any(s['global_bound_violations'] or s['difference_bound_violations']
                                  or s['false_improvements'] for s in summary.values()))


if __name__ == '__main__':
    raise SystemExit(main())
