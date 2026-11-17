"""Controlled stochastic example: identifiable improvement, unknown future actions.

One illustrative topology, four fixed budgets, three fixed seeds. This is not
a held-out benchmark or a coverage-rate validation experiment.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'code'))
from oeh_certificate import certify_depths
from oeh_certificate_adapter import trajectory_statistics
from oeh_refinements import kl_confidence_box
from oeh_selective import coupled_difference
from evaluation.io import reserve_output_dir, sha256_file, write_json, write_csv_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
        default=ROOT/'artifacts/research/oeh_shared_future_probe_2026-09-18')
    args = parser.parse_args(argv)
    root = args.output_dir
    reserve_output_dir(root)
    plan = dict(seeds=[811,812,813], visits_per_root_action=[20,200,2000,20000],
        states=3, actions=2, gamma=.95, delta=.05, epsilon=.05,
        mechanism='Root actions share stochastic successor law but differ in immediate reward; future actions are unobserved.',
        source_sha256={p:sha256_file(ROOT/p) for p in ['code/oeh_selective.py',
            'code/oeh_refinements.py','code/oeh_certificate.py','code/oeh_certificate_adapter.py',
            'scripts/analysis/run_oeh_shared_future_probe.py']},
        scope='One illustrative stochastic topology; 12 cases; no generalization or empirical coverage claim.')
    write_json(root/'plan.json',plan)
    rows = []
    pi, ref = np.array([1,0,0]), np.array([0,0,0])
    for seed in plan['seeds']:
        for n in plan['visits_per_root_action']:
            rng = np.random.default_rng(seed)
            action = np.tile([0,1],n)
            arrays = dict(state=np.zeros(2*n,dtype=int), action=action,
                next_state=rng.integers(1,3,2*n), reward=np.where(action==0,.2,.8),
                terminal=np.zeros(2*n,dtype=bool), truncated=np.ones(2*n,dtype=bool),
                trajectory_id=np.arange(2*n), timestep=np.zeros(2*n,dtype=int))
            counts, rs, rss, _ = trajectory_statistics(arrays,num_states=3,num_actions=2,reward_bound=1.)
            box = kl_confidence_box(counts,rs,rss)
            ci = coupled_difference(box,.95,[1,0,0],pi,ref)
            global_u = certify_depths(box,.95,[1,0,0],[pi],epsilon=.05)['normalized_regret_upper'][0]
            # True model is used only after construction of both estimates.
            p = np.zeros((3,2,3));p[:,:,1:] = .5
            r = np.array([[.2,.8],[.3,.9],[.4,.8]])
            def j(policy):
                idx=np.arange(3)
                return np.linalg.solve(np.eye(3)-.95*p[idx,policy],r[idx,policy])[0]
            actual_diff = .05*(j(pi)-j(ref))
            actual_regret = .05*(j(np.array([1,1,1]))-j(pi))
            assert ci['lower']-1e-8 <= actual_diff <= ci['upper']+1e-8
            assert actual_regret <= global_u+1e-8
            rows.append(dict(seed=seed,visits_per_action=n,**ci,
                certified_improvement=ci['lower']>1e-9, global_upper=global_u,
                global_certificate=global_u<=.05, true_improvement=actual_diff,true_regret=actual_regret))
    write_csv_rows(root/'summary.csv',rows,list(rows[0]))
    write_json(root/'report.json',{**plan,'status':'complete','cases':len(rows),
        'certified_improvements':sum(r['certified_improvement'] for r in rows),
        'global_certificates':sum(r['global_certificate'] for r in rows),
        'summary_sha256':sha256_file(root/'summary.csv')})
    for row in rows:
        print(f"seed={row['seed']} n={row['visits_per_action']}: difference=[{row['lower']:.5f},{row['upper']:.5f}], global U={row['global_upper']:.5f}")


if __name__ == '__main__':
    main()
