"""Experimental FP64 CFQI CPU/CUDA comparison; not a production learner.

Uses real development buffers and compares original NumPy, vectorized NumPy,
and the identical vectorized CuPy algorithm. Reports transfers and CPU model
construction in end-to-end fitting. No scientific labels are published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH
from evaluation.io import sha256_file, write_json
from scripts.training import run_empirical_requirement as runner
from empirical_requirement import _model, _greedy_policy, cfqi_tabular


def solve(model, xp, gamma, iterations=100, tolerance=1e-10):
    p, r, counts, terminal = (xp.asarray(a) for a in model)
    support = counts > 0
    active = xp.any(support, axis=1) & ~terminal
    q = xp.zeros_like(r)
    for step in range(iterations):
        v = xp.where(active, xp.max(xp.where(support, q, -xp.inf), axis=1), 0)
        updated = xp.where(support, r + gamma * xp.einsum('san,n->sa', p, v), 0)
        delta = float(xp.max(xp.abs(updated-q)).item())
        q = updated
        if delta <= tolerance:
            break
    q = q if xp is np else xp.asnumpy(q)
    return _greedy_policy(q, model[2], model[3]), q, step + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    manifest = json.loads((args.dataset_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol_config'] != PROTOCOL_CONFIG:
        raise ValueError('Select the matching profile before launching Python.')
    # Preselect one MDP per actual state count, all three first registered
    # training seeds at maximum prefix for q_star_softmax. No outcome selection.
    selected = {}
    for r in sorted(manifest['datasets'], key=lambda r:r['dataset_id']):
        if (r['stream']=='training' and r['policy_family']=='q_star_softmax'
                and r['size_multiplier']==50):
            selected.setdefault(r['num_states'], r['instance_id'])
    records = [r for r in manifest['datasets'] if r['stream']=='training'
        and r['policy_family']=='q_star_softmax' and r['size_multiplier']==50
        and selected.get(r['num_states'])==r['instance_id']
        and runner._training_seed_index(r) in (0,1,2)]
    write_json(args.output_dir/'plan.json', dict(scope=__doc__, repeats=3,
        profile_sha256=sha256_file(PROTOCOL_CONFIG_PATH),
        manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'),
        source_sha256=sha256_file(Path(__file__)),
        dataset_ids=[r['dataset_id'] for r in records]))
    start=time.perf_counter()
    toy=(np.ones((2,2,2))/2, np.ones((2,2)), np.ones((2,2)),np.zeros(2,dtype=bool))
    solve(toy,cp,.95)
    cp.cuda.Stream.null.synchronize()
    cold=time.perf_counter()-start
    rows=[]
    for record in records:
        start=time.perf_counter()
        transitions=runner._load_run(args.dataset_dir,record,{})
        load_seconds=time.perf_counter()-start
        states,actions=record['num_states'],record['num_actions']
        gamma=PROTOCOL_CONFIG['benchmark']['gamma']
        models=_model(transitions,states,actions)
        reference=cfqi_tabular(transitions,num_states=states,num_actions=actions,gamma=gamma)
        _, q_cpu, _=solve(models,np,gamma)
        _, q_gpu, _=solve(models,cp,gamma)
        np.testing.assert_allclose(q_gpu,q_cpu,rtol=1e-10,atol=1e-10)
        samples={k:[] for k in ('original_cpu','vectorized_cpu','cuda')}
        for repeat in range(3):
            order=list(samples)
            order=order[repeat:]+order[:repeat]
            for name in order:
                cp.cuda.Stream.null.synchronize()
                start=time.perf_counter()
                if name=='original_cpu':
                    policy=cfqi_tabular(transitions,num_states=states,num_actions=actions,gamma=gamma)
                else:
                    model=_model(transitions,states,actions)
                    policy,_,_=solve(model,np if name=='vectorized_cpu' else cp,gamma)
                cp.cuda.Stream.null.synchronize()
                samples[name].append(time.perf_counter()-start)
                np.testing.assert_array_equal(policy,reference)
        row=dict(dataset_id=record['dataset_id'],states=states,actions=actions,
            load_seconds=load_seconds,seconds=samples,
            median_seconds={k:statistics.median(v) for k,v in samples.items()},
            max_q_error=float(np.max(np.abs(q_cpu-q_gpu))),policies_identical=True)
        rows.append(row)
        write_json(args.output_dir/'measurements.json', rows)
        print(f'{len(rows)}/{len(records)} S={states}: {row["median_seconds"]}',flush=True)
    totals={k:sum(r['median_seconds'][k] for r in rows) for k in samples}
    write_json(args.output_dir/'summary.json',dict(status='complete',cases=len(rows),
        scope='CFQI only; not IQL, OEH/bootstrap or full-grid timing',dtype='float64',
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        numpy=np.__version__,cupy=cp.__version__,cold_start_seconds=cold,
        sum_case_median_seconds=totals,
        cuda_speedup_vs_original=totals['original_cpu']/totals['cuda'],
        cuda_speedup_vs_vectorized_cpu=totals['vectorized_cpu']/totals['cuda'],
        timing_scope='model building + fitting + host/device transfers + policy extraction; excludes disk load',
        all_policies_identical=True,max_q_error=max(r['max_q_error'] for r in rows)))


if __name__=='__main__':
    main()
