"""Single-use, post-hoc candidate routing comparison on existing E1 artifacts.

No MDP generator, collector or one-shot E1 runner is called. Inputs are
verified against the original E1 seals. Failed candidate evidence and owned
prototype are automatically withdrawn after writing a minimal audit ledger.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from evaluation.e1_reuse import load_e1_panel
from evaluation.io import reserve_output_dir,sha256_file,write_csv_rows,write_json
from evaluation.withdrawal import withdraw_attempt

WORKER = None


def initialize_worker(panel,mdps,module):
    global WORKER
    WORKER=(panel,mdps,importlib.import_module(module))


def bootstrap_one(task):
    index,weights=task
    panel,mdps,module=WORKER
    choices,_,_=module.fit_routes(panel,mdps,multiplicities=weights)
    original,_,_=module.fit_routes(panel,mdps,multiplicities=weights,original_v4e=True)
    both=dict(panel,v4e_decisions=original)
    mean,_=module.score_routes(both,choices,mdps,multiplicities=weights)
    return index,np.concatenate((mean[:2]-mean[2:4],mean[:2]-mean[4:6]))


def paired_draws(mdps,outer,draws,seed):
    rng=np.random.default_rng(seed)
    strata=[np.array(sorted(mdp for mdp in mdps if outer[mdp]==fold)) for fold in range(5)]
    for index in range(draws):
        weights={mdp:0 for mdp in mdps}
        for values in strata:
            weights.update(Counter(rng.choice(values,size=len(values),replace=True)))
        yield index,weights


def run(protocol,workers):
    protocol=protocol.resolve()
    if not protocol.is_relative_to(ROOT/'docs/protocol') or workers<1:
        raise ValueError('Invalid protocol path or worker count')
    config=json.loads(protocol.read_text('utf-8')); attempt=config['attempt_id']
    if not re.fullmatch(r'r[1-9][0-9]*',attempt) or config.get('evaluation_reuse') is not True:
        raise ValueError('Explicit original E1 reuse required')
    if config['budget']!=10 or config['new_mdps_generated']!=0:
        raise ValueError('Only the registered E1 x10 comparison is supported')
    ledger=ROOT/'docs/reassessment/withdrawal_ledger'/f'{attempt}.json'
    if ledger.exists():
        raise ValueError('Consumed attempt; no rerun')
    sources=list(config['sources'])+[str(protocol.relative_to(ROOT))]
    if any(not (ROOT/name).resolve().is_relative_to(ROOT) for name in sources):
        raise ValueError('Source path escapes repository')
    hashes={name:sha256_file(ROOT/name) for name in sources}
    bundle=ROOT/'tmp/oeh_e1_reuse'/attempt
    reserve_output_dir(bundle)
    def log(message):
        with (bundle/'run.log').open('a',encoding='utf-8') as stream:
            stream.write(str(message)+'\n')
        print(message,flush=True)
    started=time.perf_counter(); report=None; input_hashes={}; reasons=['EXECUTION_FAILURE']
    try:
        log(f'{attempt}: source/config sealed; reading ONLY original E1 artifacts')
        loaded=load_e1_panel(ROOT); input_hashes=loaded['input_hashes']
        # The fit stage receives no oracle, family, dataset spec or new buffer.
        panel={key:loaded[key] for key in ('runs','features','outer','v4e_decisions','return_ridges')}
        write_json(bundle/'plan.json',dict(config=config,source_hashes=hashes,input_hashes=input_hashes,
            manifest_sha256=loaded['manifest_sha256'],original_cohort_reused=True,new_mdps_generated=0,
            source_worktree_uncommitted=True,posthoc_consumed_e1=True,workers=workers))
        module=importlib.import_module(config['candidate_module'])
        decisions=[]; scores={}; primary_choices=None
        for name,mdps in loaded['panels'].items():
            original,_,_=module.fit_routes(panel,mdps,original_v4e=True)
            expected={g for g in panel['runs'] if g[0] in mdps}
            if set(original)!=expected or any(original[group]!=panel['v4e_decisions'][group] for group in original):
                raise ValueError('Original OEH-v4E adapter failed exact route reproduction')
            choices,predictions,tuning=module.fit_routes(panel,mdps)
            if set(choices)!=expected or set(predictions)!=expected:
                raise ValueError('Incomplete candidate choices or predictions')
            for group,method in sorted(choices.items()):
                decisions.append(dict(mdp_id=group[0],policy_family=group[1],panel=name,
                    outer_fold=panel['outer'][group[0]],selected_algorithm=method,
                    predicted_paired_return=float(predictions[group][method][0]),
                    predicted_paired_success=float(predictions[group][method][1])))
            # Durable choices contain predictions only; actual test values
            # are accessed by score_routes only AFTER this file is hashed.
            write_json(bundle/f'{name}_choices.json',decisions[-len(choices):])
            write_json(bundle/f'{name}_choice_seal.json',dict(sha256=sha256_file(bundle/f'{name}_choices.json')))
            write_json(bundle/f'{name}_training.json',tuning)
            scores[name],_=module.score_routes(panel,choices,mdps)
            if name=='stochastic_primary':
                primary_choices=choices
        primary=scores['stochastic_primary']
        observed=np.concatenate((primary[:2]-primary[2:4],primary[:2]-primary[4:6]))
        reasons=[f'NONPOSITIVE_{name}' for name,value in zip(
            ('NR_VS_CFQI','SR_VS_CFQI','NR_VS_V4E','SR_VS_V4E'),observed) if value<=0]
        write_csv_rows(bundle/'decisions.csv',decisions,list(decisions[0]))
        bootstrap=None; lower=None
        if not reasons:
            mdps=loaded['panels']['stochastic_primary']
            bootstrap=np.full((config['bootstrap_draws'],4),np.nan)
            log('Point effects positive on both endpoints vs both comparators; refitting nested procedures in every bootstrap draw')
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
                                     initargs=(panel,mdps,config['candidate_module'])) as pool:
                futures=[pool.submit(bootstrap_one,task) for task in paired_draws(
                    mdps,panel['outer'],config['bootstrap_draws'],config['bootstrap_seed'])]
                for count,future in enumerate(as_completed(futures),1):
                    index,values=future.result(); bootstrap[index]=values
                    if count%10==0:
                        log(f'BOOTSTRAP {count}/{config["bootstrap_draws"]}; no reroll/filtering')
                        write_json(bundle/'progress.json',dict(completed=count,total=config['bootstrap_draws'],
                            runtime_seconds=time.perf_counter()-started))
            if not np.isfinite(bootstrap).all():
                raise ValueError('Incomplete bootstrap grid')
            np.savez_compressed(bundle/'bootstrap.npz',draws=bootstrap,observed=observed)
            lower=2*observed-np.quantile(bootstrap,1-config['one_sided_alpha'],axis=0)
            reasons=[f'UNCERTAIN_{name}' for name,value in zip(
                ('NR_VS_CFQI','SR_VS_CFQI','NR_VS_V4E','SR_VS_V4E'),lower) if value<=0]
        if (any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or
                any(sha256_file(ROOT/name)!=digest for name,digest in input_hashes.items())):
            raise ValueError('Source or original E1 input changed after seal')
        report=dict(screen_pass=not reasons,reason_codes=reasons,paired_effects=observed.tolist(),
            basic_one_sided_lower=None if lower is None else lower.tolist(),
            panels={name:values.tolist() for name,values in scores.items()},
            posthoc_consumed_e1=True,confirmation=False,new_mdps_generated=0,
            normalized_return_unclipped=True,success_threshold=.95,
            runtime_seconds=time.perf_counter()-started)
        write_json(bundle/'report.json',report)
        log(f'{attempt}: screen={report["screen_pass"]}; reason codes={reasons}')
        write_json(bundle/'artifact_manifest.json',dict(
            artifact_sha256={str(path.relative_to(bundle)):sha256_file(path)
                             for path in sorted(bundle.iterdir()) if path.is_file()},
            runtime_seconds=report['runtime_seconds'],source_hashes=hashes,input_hashes=input_hashes))
    except Exception as exc:
        reasons=['EXECUTION_FAILURE']
        write_json(bundle/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        write_json(ledger,dict(attempt_id=attempt,status='screen_pass' if report and report['screen_pass'] else 'withdrawn',
            evaluation_reuse=True,consumed_seeds=[],new_mdps_generated=0,source_hashes=hashes,
            input_hashes=input_hashes,reason_codes=reasons,posthoc_consumed_e1=True,
            runtime_seconds=time.perf_counter()-started))
        if not report or not report['screen_pass']:
            print(json.dumps(withdraw_attempt(ROOT,attempt),sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=18)
    args=parser.parse_args(); run(args.protocol,args.workers)


if __name__=='__main__':
    main()
