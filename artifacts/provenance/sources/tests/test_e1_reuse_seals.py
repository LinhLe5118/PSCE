"""Load-stage seal and routing checks, isolated from grid checks in test_e1_reuse.

Synthetic files exercise the original-input reader without reading research
data. The grid verifier has independent complete/missing/tamper controls.
"""
import json
from pathlib import Path

import pytest

from evaluation import e1_reuse
from evaluation.io import sha256_file,write_csv_rows,write_json


@pytest.fixture
def sealed_inputs(tmp_path,monkeypatch):
    output=tmp_path/'artifacts/week6/oehv4e_confirmatory_v1_1'; output.mkdir(parents=True)
    instances={}; runs={}; features=[]; decisions={name:[] for name in ('stochastic_primary','deterministic_control')}
    for i in range(120):
        mdp=f'm{i:03}'; panel='stochastic_primary' if i<84 else 'deterministic_control'
        instances[mdp]=dict(instance_id=mdp,split='evaluation',dynamics_audit=dict(panel=panel))
        for j in range(5):
            group=(mdp,str(j)); runs[group]={}
            for method in e1_reuse.ALGORITHMS:
                runs[group][method]={seed:(.96,1) for seed in range(10)}
                features.append(dict(mdp_id=mdp,policy_family=str(j),algorithm=method,a='1.',b='2.'))
            decisions[panel].append(dict(mdp_id=mdp,policy_family=str(j),outer_fold=str(i%5),
                selected_algorithm='cfqi_tabular',selected_return='.96'))
    dataset=tmp_path/'datasets/e1/manifest.json'
    write_json(dataset,dict(instances=list(instances.values()),datasets=[]))
    write_json(tmp_path/'artifacts/week6/E1_INPUT_SEAL_2026-09-26.json',dict(
        dataset_dir='datasets/e1',manifest_sha256=sha256_file(dataset)))
    (output/'core_run_results.csv').write_text('algorithm\n','utf-8')
    artifacts=['core_run_results.csv']
    for panel,rows in decisions.items():
        name=f'{panel}_selector_decisions.csv'; artifacts.append(name)
        write_csv_rows(output/name,rows,list(rows[0]))
    summary=dict(status='complete',core_fits=108000,manifest_sha256=sha256_file(dataset),
        artifact_sha256={name:sha256_file(output/name) for name in artifacts})
    write_json(output/'e1_execution_summary.json',summary)
    write_csv_rows(output/'pilot_features/features.csv',features,list(features[0]))
    write_json(output/'pilot_features/feature_seal.json',dict(features_sha256=sha256_file(output/'pilot_features/features.csv'),
        scope='evaluation_pilot_only_before_training_outcomes',rows=1800,groups=600))
    write_json(tmp_path/'docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json',dict(
        feature_sets={'B2_OEH3':['a'],'B2_OEH3_LCOEH4':['a','b']},evaluation={'return_ridges':[.1,1.]}))
    monkeypatch.setattr(e1_reuse,'verify_run_grid',lambda manifest,rows:({},runs,instances))
    return tmp_path,output,features,decisions,instances


def rewrite_features(output,features):
    write_csv_rows(output/'pilot_features/features.csv',features,list(features[0]))
    seal=json.loads((output/'pilot_features/feature_seal.json').read_text('utf-8'))
    seal['features_sha256']=sha256_file(output/'pilot_features/features.csv')
    write_json(output/'pilot_features/feature_seal.json',seal)


def rewrite_decisions(output,panel,rows):
    name=f'{panel}_selector_decisions.csv'
    write_csv_rows(output/name,rows,list(rows[0]))
    summary=json.loads((output/'e1_execution_summary.json').read_text('utf-8'))
    summary['artifact_sha256'][name]=sha256_file(output/name)
    write_json(output/'e1_execution_summary.json',summary)


def test_complete_sealed_inputs_preserve_original_cohort_and_decisions(sealed_inputs):
    root,_,_,_,_=sealed_inputs
    panel=e1_reuse.load_e1_panel(root)
    assert panel['new_mdps_generated']==0 and panel['original_cohort_reused'] and panel['posthoc_consumed_e1']
    assert len(panel['features'])==1800 and len(panel['v4e_decisions'])==600
    assert len(panel['panels']['stochastic_primary'])==84 and len(panel['panels']['deterministic_control'])==36
    assert panel['features'][('m000','0','cfqi_tabular')]['B2_OEH3_LCOEH4']==(1.,2.)


def test_artifact_tamper_is_rejected_before_loading_features(sealed_inputs):
    root,output,_,_,_=sealed_inputs
    (output/'core_run_results.csv').write_text('tampered','utf-8')
    with pytest.raises(ValueError):
        e1_reuse.load_e1_panel(root)


@pytest.mark.parametrize('kind',['nonfinite','duplicate','unmatched','incomplete'])
def test_feature_values_and_grid_are_independently_checked(sealed_inputs,kind):
    root,output,features,_,_=sealed_inputs
    if kind=='nonfinite': features[0]['a']='nan'
    if kind=='duplicate': features[1]=dict(features[0])
    if kind=='unmatched': features[0]['mdp_id']='foreign'
    if kind=='incomplete': features.pop()
    rewrite_features(output,features)
    with pytest.raises(ValueError,match='feature'):
        e1_reuse.load_e1_panel(root)


@pytest.mark.parametrize('kind',['score','fold','duplicate','incomplete'])
def test_v4e_choices_must_match_the_original_scores_and_fold_groups(sealed_inputs,kind):
    root,output,_,decisions,_=sealed_inputs
    rows=decisions['stochastic_primary']
    if kind=='score': rows[0]['selected_return']='.90'
    if kind=='fold': rows[0]['outer_fold']='6'
    if kind=='duplicate': rows[1]=dict(rows[0])
    if kind=='incomplete': rows.pop()
    rewrite_decisions(output,'stochastic_primary',rows)
    with pytest.raises(ValueError):
        e1_reuse.load_e1_panel(root)


def test_wrong_source_execution_status_and_dynamics_panel_are_rejected(sealed_inputs):
    root,output,_,_,instances=sealed_inputs
    summary_path=output/'e1_execution_summary.json'
    summary=json.loads(summary_path.read_text('utf-8')); summary['status']='incomplete'
    write_json(summary_path,summary)
    with pytest.raises(ValueError,match='summary'):
        e1_reuse.load_e1_panel(root)
    summary['status']='complete'; write_json(summary_path,summary)
    instances['m000']['dynamics_audit']['panel']='deterministic_control'
    with pytest.raises(ValueError,match='panels'):
        e1_reuse.load_e1_panel(root)
