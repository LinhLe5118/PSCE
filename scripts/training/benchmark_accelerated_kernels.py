"""Compare FP64 batched CFQI and restricted OEH sequence on real buffers.

Kernel component timing only: does not measure raw bootstrap or full pipeline.
All policies/depth arrays must match NumPy before timings can be reported.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
import time

import numpy as np

from scripts.training import run_empirical_requirement as runner
from empirical_requirement import _model, _greedy_policy
from tabular_acceleration import cfqi_models, iql_models, q_sequence
from evaluation.io import write_json, sha256_file
from config.protocol_config import PROTOCOL_CONFIG_PATH


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.dataset_dir/'manifest.json').read_text(encoding='utf-8'))
    runner.validate_training_manifest(manifest)
    records=manifest['datasets']
    selected={}
    maximum=max(r['size_multiplier'] for r in records if r['stream']=='training')
    for r in sorted(records,key=lambda r:r['dataset_id']):
        if r['stream']=='training' and r['size_multiplier']==maximum and r['policy_family']=='q_star_softmax':
            selected.setdefault(r['num_states'],r['instance_id'])
    cases=[[r for r in records if r['stream']=='training' and r['size_multiplier']==maximum
            and r['policy_family']=='q_star_softmax' and r['instance_id']==instance]
           for states,instance in sorted(selected.items()) if states in (32,128,256,512)]
    write_json(args.output_dir/'plan.json',dict(scope=__doc__,manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'),
        profile_sha256=sha256_file(PROTOCOL_CONFIG_PATH),source_sha256=sha256_file(Path(__file__)),
        acceleration_sha256=sha256_file(runner.ROOT/'code/tabular_acceleration.py'),
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        cases=[[r['dataset_id'] for r in rs] for rs in cases],precision='float64',repeats=3))
    import cupy as cp
    rows=[]
    for records in cases:
        start=time.perf_counter()
        cache={}
        models=[_model(runner._load_run(args.dataset_dir,r,cache),r['num_states'],r['num_actions']) for r in records]
        build_seconds=time.perf_counter()-start
        p,r,counts,_=models[0]
        beta=np.divide(counts,counts.sum(axis=1,keepdims=True),out=np.zeros_like(counts),where=counts.sum(axis=1,keepdims=True)>0)
        supported=tuple(tuple(np.flatnonzero(row)) for row in counts)
        reference=cfqi_models(models,.95)
        actual=cfqi_models(models,.95,backend='cuda')
        iql_reference=iql_models(models,.95)
        iql_actual=iql_models(models,.95,backend='cuda')
        np.testing.assert_allclose(iql_actual[0],iql_reference[0],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(iql_actual[1],iql_reference[1])
        np.testing.assert_allclose(actual[0],reference[0],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(actual[1],reference[1])
        for q_gpu,q_cpu,model in zip(actual[0],reference[0],models):
            np.testing.assert_array_equal(_greedy_policy(q_gpu,model[2],model[3]),_greedy_policy(q_cpu,model[2],model[3]))
        seq_cpu=q_sequence(p,r,beta,supported,.95,100,backend='numpy')
        seq_gpu=q_sequence(p,r,beta,supported,.95,100,backend='cuda')
        if seq_cpu is None or seq_gpu is None:
            raise RuntimeError('Invalid restricted sequence')
        np.testing.assert_allclose(seq_gpu,seq_cpu,rtol=1e-10,atol=1e-10,equal_nan=True)
        timings={}
        for backend in ['numpy','cuda']:
            for method in ['batched_cfqi','batched_iql','oeh_sequence']:
                samples=[]
                for _ in range(3):
                    cp.cuda.Stream.null.synchronize()
                    start=time.perf_counter()
                    if method=='batched_cfqi':
                        cfqi_models(models,.95,backend=backend)
                    elif method=='batched_iql':
                        iql_models(models,.95,backend=backend)
                    else:
                        q_sequence(p,r,beta,supported,.95,100,backend=backend)
                    cp.cuda.Stream.null.synchronize()
                    samples.append(time.perf_counter()-start)
                timings[method+'_'+backend]=samples
        row=dict(states=p.shape[0],actions=r.shape[1],batch=len(models),load_and_model_seconds=build_seconds,
            seconds=timings,medians={k:float(np.median(v)) for k,v in timings.items()},equivalence_passed=True)
        rows.append(row)
        write_json(args.output_dir/'measurements.json',rows)
        print(row,flush=True)
    write_json(args.output_dir/'summary.json',dict(status='complete',rows=rows,
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,
        numpy=np.__version__, cuda_driver_api=cp.cuda.runtime.driverGetVersion(),
        note='Includes transfers and result extraction; model construction separate. GPU warmed by equivalence checks. Includes tabular IQL. Not raw bootstrap, full scientific OEH verdict equivalence, original neural learner or A100 timing.'))


if __name__=='__main__':
    main()
