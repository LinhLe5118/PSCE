"""Fixed development-only 2x2 OEH confidence/proposal ablation.

No threshold tuning, evaluator access during estimation, or new learner fits.
All candidate depths are scored after estimation, including unsuccessful ones.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'code'))
from oeh_refinements import refinement_profiles  # noqa: E402
from scripts.analysis.run_oeh_development_smoke import development_inputs  # noqa: E402
from evaluation.io import reserve_output_dir, verified_dataset_path, sha256_file, write_csv_rows, write_json  # noqa: E402


def summarize(rows):
    return {
        "cases": len(rows),
        "nonflat_curves": sum(r["upper_range"] > 1e-9 for r in rows),
        "range_above_0_001": sum(r["upper_range"] > .001 for r in rows),
        "all_ceiling_curves": sum(r["min_upper"] >= 1-1e-9 for r in rows),
        "certified_curves": sum(r["certified_depth"] is not None for r in rows),
        "multiple_policy_curves": sum(r["unique_policies"] > 1 for r in rows),
        "min_upper": min(r["min_upper"] for r in rows),
        "median_min_upper": float(np.median([r["min_upper"] for r in rows])),
        "max_upper_range": max(r["upper_range"] for r in rows),
        "oracle_variable_curves": sum(r["oracle_regret_range"] > 1e-8 for r in rows),
        "selected_better_than_depth0": sum(r["gain_vs_depth0"] > 1e-8 for r in rows),
        "selected_worse_than_depth0": sum(r["gain_vs_depth0"] < -1e-8 for r in rows),
        "mean_gain_vs_depth0": float(np.mean([r["gain_vs_depth0"] for r in rows])),
        "mean_selected_regret": float(np.mean([r["selected_regret"] for r in rows])),
        "median_selected_excess_regret": float(np.median([r["selected_excess_regret"] for r in rows])),
        "matches_best_on_variable_curves": sum(r["oracle_regret_range"] > 1e-8 and r["selected_excess_regret"] <= 1e-8 for r in rows),
        "false_certificates": sum(r["false_certificates"] for r in rows),
        "bound_violations": sum(r["bound_violations"] for r in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--limit-mdps', type=int)
    args = parser.parse_args(argv)
    if args.limit_mdps is not None and args.limit_mdps < 1:
        parser.error('--limit-mdps must be positive')
    manifest, records, labels, index_path = development_inputs(args.dataset_dir, args.limit_mdps)
    benchmark = manifest['effective_protocol']['benchmark']
    if benchmark['gamma'] != .95 or benchmark['reward_range'] != [0.,1.]:
        raise ValueError('This experiment is fixed to gamma .95 and rewards [0,1].')
    reserve_output_dir(args.output_dir)
    sources = ['code/oeh_refinements.py', 'code/oeh_certificate.py',
        'code/oeh_certificate_adapter.py', 'code/exact_solver.py',
        'scripts/analysis/run_oeh_refinements.py', 'scripts/analysis/run_oeh_development_smoke.py',
        'environments/benchmark_environments.py', 'evaluation/io.py',
        'tests/test_oeh_refinements.py']
    plan = {
        'status': 'exploratory_development_not_preregistered_confirmation',
        'manifest_sha256': sha256_file(args.dataset_dir/'manifest.json'),
        'index_sha256': sha256_file(index_path),
        'source_sha256': {f: sha256_file(ROOT/f) for f in sources},
        'design': {'confidence': ['bernstein','kl'], 'proposal': ['pessimistic','empirical'],
            'gamma': .95, 'k_max': 100, 'epsilon': .05, 'delta_per_confidence_method': .05,
            'reward_bound': 1., 'nonflat_tolerance': 1e-9, 'descriptive_range_cutoff': .001,
            'regret_comparison_tolerance': 1e-8, 'selection': 'earliest upper bound within 1e-9 of minimum'},
        'inputs': [{k:r[k] for k in ('dataset_id','group_id','file','sha256','slice')} for r in records],
        'runtime': {'python':sys.version, 'numpy':np.__version__, 'platform':platform.platform(),
            'threads': {k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}},
        'limitations': ['No joint confidence guarantee for post-hoc choice of confidence method.',
            'A smaller upper bound or nonflat curve does not establish difficulty prediction.',
            'Policy selection by minimum upper bound is exploratory; it may worsen true return.',
            'Known normalization Rmax/(1-gamma) differs from benchmark NR.',
            'No LP-MDP/SUP/IDP replication and no held-out performance claim.'],
    }
    write_json(args.output_dir/'plan.json',plan)
    rows, curves, errors, oracle_cache = [], [], [], {}
    start = time.perf_counter()
    for i, rec in enumerate(records,1):
        label = labels[rec['dataset_id']]
        identity = dict(dataset_id=rec['dataset_id'], mdp_id=label['instance_id'],
            policy_family=label['policy_family'], pilot_multiplier=rec['size_multiplier'])
        try:
            with np.load(verified_dataset_path(args.dataset_dir,rec),allow_pickle=False) as z:
                arrays = {k:z[k][:rec['size']] for k in z.files}
                if any(len(a)!=rec['size'] for a in arrays.values()):
                    raise ValueError('Short physical buffer')
            profiles = refinement_profiles(arrays, rec['initial_state_distribution'],
                num_states=rec['num_states'], num_actions=rec['num_actions'])
            # Only now read model/structural information for scoring.
            from environments.benchmark_environments import make_benchmark_mdp
            from exact_solver import value_iteration, evaluate_policy
            instance = label['instance_id']
            if instance not in oracle_cache:
                spec = next(s for s in manifest['instances'] if s['instance_id']==instance)
                mdp = make_benchmark_mdp(type('Spec',(),spec)())
                best = float(mdp.initial_state_distribution @ value_iteration(mdp).values)
                if (mdp.num_states != rec['num_states'] or mdp.num_actions != rec['num_actions']
                        or not np.isclose(best,spec['optimal_return'],atol=1e-8,rtol=1e-8)):
                    raise ValueError('Oracle disagrees with original model')
                oracle_cache[instance] = (mdp,best,spec['family'])
            mdp,best,family = oracle_cache[instance]
            evaluations = {}
            for name, profile in profiles.items():
                regrets = []
                for pi in profile['policies']:
                    key = pi.tobytes()
                    if key not in evaluations:
                        evaluations[key] = .05*(best-evaluate_policy(mdp,pi).expected_return)
                    regrets.append(evaluations[key])
                reg = np.array(regrets)
                u, selected = profile['upper'], profile['selected_depth']
                rho = (float(spearmanr(u,reg).statistic)
                       if np.ptp(u)>1e-9 and np.ptp(reg)>1e-8 else None)
                row = {**identity, 'structural_family':family, 'variant':name,
                    'min_upper':float(u.min()), 'max_upper':float(u.max()), 'upper_range':float(np.ptp(u)),
                    'certified_depth':profile['certified_depth'], 'unique_policies':profile['unique_policies'],
                    'selected_depth':selected, 'oracle_regret_range':float(np.ptp(reg)),
                    'selected_regret':float(reg[selected]), 'oracle_best_regret':float(reg.min()),
                    'depth0_regret':float(reg[0]), 'depth100_regret':float(reg[-1]),
                    'selected_excess_regret':float(reg[selected]-reg.min()),
                    'gain_vs_depth0':float(reg[0]-reg[selected]), 'depth_spearman':rho,
                    'false_certificates':int(np.sum((u<=.05)&(reg>.05+1e-8))),
                    'bound_violations':int(np.sum(reg>u+1e-8))}
                rows.append(row)
                for k in range(len(u)):
                    curves.append({**identity,'variant':name,'depth':k,'upper':float(u[k]),'oracle_regret':float(reg[k])})
            print(f'[{i}/{len(records)}] {instance} x{rec["size_multiplier"]} '
                  + ', '.join(f'{n}: range={np.ptp(p["upper"]):.5g}' for n,p in profiles.items()),flush=True)
        except Exception as exc:
            errors.append({**identity,'error':repr(exc)})
            print(f'[{i}/{len(records)}] ERROR {exc!r}',flush=True)
        write_json(args.output_dir/'progress.json',dict(processed=i,total=len(records),errors=errors))
    if rows:
        write_csv_rows(args.output_dir/'summary.csv',rows,list(rows[0]))
        write_csv_rows(args.output_dir/'curves.csv',curves,list(curves[0]))
    names = sorted({r['variant'] for r in rows})
    summary = {n:summarize([r for r in rows if r['variant']==n]) for n in names}
    stochastic = {n:summarize([r for r in rows if r['variant']==n and r['structural_family']!='lock_structured'])
                  for n in names if any(r['variant']==n and r['structural_family']!='lock_structured' for r in rows)}
    report = {**plan,'status':'complete' if not errors else 'completed_with_errors',
        'num_pilot_cases_requested':len(records), 'num_variant_cases':len(rows), 'num_depth_rows':len(curves),
        'summary':summary,'stochastic_summary':stochastic,'errors':errors,'wall_seconds':time.perf_counter()-start,
        'artifact_sha256':{p.name:sha256_file(p) for p in args.output_dir.glob('*.csv')}}
    write_json(args.output_dir/'report.json',report)
    return 1 if errors or any(s['false_certificates'] or s['bound_violations'] for s in summary.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
