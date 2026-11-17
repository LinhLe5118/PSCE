"""Measure raw adaptive OEH and alternative definitions on development data.

Default measures every pilot, preserving the registered bootstrap/depth budget.
--sample selects one pilot per (structural family,state count,behavior) stratum,
before timing; its weighted extrapolation is not a confidence interval.
Training runtime is imported only as a separately labelled previous estimate.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH
from evaluation.io import verified_dataset_path, sha256_file, write_json
from scripts.analysis.run_saturation_sweep import pilot_records
from scripts.analysis.run_alternative_definitions import variant_rows, value_scale
from oeh_estimator import OEHConfig
from oeh_validation import run_pilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sample',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue the unchanged saved case plan.')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    manifest=json.loads((args.dataset_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol_config'] != PROTOCOL_CONFIG:
        raise ValueError('Dataset profile differs from active configuration')
    family={r['instance_id']:r['family'] for r in manifest['instances']}
    strata=defaultdict(list)
    for r in pilot_records(manifest):
        strata[(family[r['instance_id']],r['num_states'],r['policy_family'])].append(r)
    selected=[]
    for key,rs in sorted(strata.items(),key=lambda p:(p[0][1],p[0])):
        rs=sorted(rs,key=lambda r:r['dataset_id'])
        selected.extend((key,r,len(rs) if args.sample else 1) for r in (rs[:1] if args.sample else rs))
    plan=dict(scope=__doc__,sample=args.sample,
        numerical_backend=os.environ.get('DAORL_OEH_BACKEND','reference'),
        acceleration_source_sha256=sha256_file(Path(__file__).resolve().parents[2]/'code/tabular_acceleration.py'),
        profile_sha256=sha256_file(PROTOCOL_CONFIG_PATH),
        manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'),
        source_sha256=sha256_file(Path(__file__)),config=PROTOCOL_CONFIG['estimator'],
        cases=[dict(dataset_id=r['dataset_id'],stratum=k,weight=w) for k,r,w in selected],
        exclusions=['full learner fits','join/statistical comparison','external methods','sealed evaluation'])
    rows=[]
    if args.resume:
        saved=json.loads((args.output_dir/'plan.json').read_text(encoding='utf-8'))
        if saved.get('numerical_backend','reference') != plan['numerical_backend']:
            raise ValueError('Cannot resume with a different numerical backend')
        if plan['numerical_backend'] != 'reference' and saved.get('acceleration_source_sha256') != plan['acceleration_source_sha256']:
            raise ValueError('Cannot resume accelerated measurements after backend source changes')
        for key in ['sample','profile_sha256','manifest_sha256','config','cases']:
            if saved[key] != json.loads(json.dumps(plan[key])):
                raise ValueError(f'Resume plan mismatch: {key}')
        rows=[json.loads(line) for line in (args.output_dir/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
        ids=[r['dataset_id'] for r in rows]
        if len(ids)!=len(set(ids)) or not set(ids).issubset({r['dataset_id'] for _,r,_ in selected}):
            raise ValueError('Invalid or duplicate saved measurements')
        with (args.output_dir/'resume_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(timestamp=time.time(),source_sha256=plan['source_sha256'],completed=len(rows)))+'\n')
    else:
        write_json(args.output_dir/'plan.json',plan)
    config=OEHConfig.load()
    completed={r['dataset_id'] for r in rows}
    for key,record,weight in selected:
        if record['dataset_id'] in completed:
            continue
        started=time.perf_counter()
        with np.load(verified_dataset_path(args.dataset_dir,record),allow_pickle=False) as f:
            arrays={k:f[k] for k in f.files}
        loaded=time.perf_counter()
        pilot=run_pilot(arrays,np.asarray(record['initial_state_distribution']),
            num_states=record['num_states'],num_actions=record['num_actions'],bootstrap_seed=20260824)
        estimated=time.perf_counter()
        result=pilot.selected_result
        variants=variant_rows(record['instance_id'],record['policy_family'],key[0],result,config,
            scale=value_scale(result,arrays,config.gamma),
            pilot_size=pilot.selection.n0_effective_size,materiality_fractions=[.005,.01,.02])
        analysed=time.perf_counter()
        row=dict(dataset_id=record['dataset_id'],stratum=key,weight=weight,
            load_seconds=loaded-started,raw_oeh_seconds=estimated-loaded,
            variant_seconds=analysed-estimated,attempts=len(pilot.selection.attempts),
            selected_multiplier=pilot.selection.n0_effective_multiplier,
            abstained=result.abstained,censored=result.censored_right)
        with (args.output_dir/'measurements.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        rows.append(row)
        print(f'{len(rows)}/{len(selected)} {key}: {estimated-loaded:.2f}s raw OEH',flush=True)
    write_json(args.output_dir/'summary.json',dict(status='complete',sample=args.sample,
        measured_cases=len(rows),target_cases=sum(r['weight'] for r in rows),
        measured_seconds={k:sum(r[k] for r in rows) for k in ['load_seconds','raw_oeh_seconds','variant_seconds']},
        weighted_estimate_seconds={k:sum(r[k]*r['weight'] for r in rows) for k in ['load_seconds','raw_oeh_seconds','variant_seconds']},
        note='Selected cases retain full bootstrap and adaptive prefix work. One representative per stratum does not quantify within-stratum uncertainty. Not a complete end-to-end pipeline measurement.'))


if __name__=='__main__':
    main()
