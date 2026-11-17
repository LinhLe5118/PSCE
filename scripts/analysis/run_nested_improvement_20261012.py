"""Nested, MDP-grouped development experiments; no frozen artifact mutation."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from scripts.analysis import export_week8_d1 as D
from scripts.analysis.compare_difficulty import FEATURES_B2, METHODS
from scripts.analysis.probe_forecast_calibration_20261012 import loss, temperature

CONFIG = D.ROOT / 'config/week8_nested_experiment_20261012.json'
DEV = D.ROOT / 'artifacts/oeh_final_unseen_120_2026-10-02'
OUT = D.ROOT / 'artifacts/week8/nested_experiment_2026-10-12'
P = [f'p_{i}' for i in range(7)]


def verified_data():
    seal, result = D.read_json(DEV/'preoutcome_seal.json'), D.read_json(DEV/'result_manifest.json')
    source_hashes = {}
    for name in ('pilot_features.csv','predictions_preoutcome.csv','labels.csv'):
        expected = result['artifact_sha256'].get(name, seal['artifact_sha256'].get(name))
        if not expected or D.digest(DEV/name) != expected:
            raise ValueError(f'Frozen input hash differs: {name}')
        source_hashes[(DEV/name).relative_to(D.ROOT).as_posix()] = expected
    label = pd.read_csv(DEV/'labels.csv')
    label = label[label.panel=='primary'].sort_values(list(D.KEYS)).reset_index(drop=True)
    if len(label)!=1260 or label.mdp_id.nunique()!=84 or label.duplicated(list(D.KEYS)).any():
        raise ValueError('Wrong primary development panel')
    predictions = pd.read_csv(DEV/'predictions_preoutcome.csv')
    arrays = {}
    for method in ('oeh_final','b2_matched'):
        joined = label[list(D.KEYS)].merge(predictions[predictions.method==method],
            on=list(D.KEYS),how='left',validate='one_to_one')
        arrays[method]=joined[P].to_numpy(float)
        scores(arrays[method],label.event.to_numpy(int))
    pilot = label[list(D.KEYS)].merge(pd.read_csv(DEV/'pilot_features.csv'),
        on=list(D.KEYS),how='left',validate='one_to_one')
    x = pilot[list(FEATURES_B2)].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite or missing pilot feature')
    x = np.column_stack((x,arrays['oeh_final']))
    if x.shape!=(1260,25):
        raise ValueError('Unexpected head input shape')
    return label,x,arrays,source_hashes


def scores(p,y):
    return np.array([D.forecast_score(list(row),int(e)) for row,e in zip(p,y)])


def assert_split(groups,fit,val):
    if not len(fit) or not len(val) or set(groups[fit]) & set(groups[val]):
        raise ValueError('Empty or overlapping MDP split')
    if set(fit) & set(val):
        raise ValueError('Overlapping row split')


def prior(y,k):
    counts = np.bincount(y,minlength=k).astype(float)+.5
    return counts/counts.sum()


def multiclass(x,y,q,k,cfg):
    if len(np.unique(y))<2:
        return np.tile(prior(y,k),(len(q),1))
    params={**cfg['xgboost'],'objective':'multi:softprob','num_class':k,'eval_metric':'mlogloss'}
    model=xgb.train(params,xgb.DMatrix(x,label=y),num_boost_round=cfg['boosting_rounds'])
    return model.predict(xgb.DMatrix(q))


def two_stage(x,y,q,cfg):
    censored=(y==6).astype(int)
    if len(np.unique(censored))<2:
        tail=np.full(len(q),(censored.sum()+.5)/(len(y)+1))
    else:
        params={**cfg['xgboost'],'objective':'binary:logistic','eval_metric':'logloss'}
        model=xgb.train(params,xgb.DMatrix(x,label=censored),num_boost_round=cfg['boosting_rounds'])
        tail=model.predict(xgb.DMatrix(q))
    finite=y<6
    conditional=multiclass(x[finite],y[finite],q,6,cfg)
    return np.column_stack(((1-tail[:,None])*conditional,tail))


def heads(x,y,alg,fit,val,cfg):
    predictions={name:np.full((len(val),7),np.nan) for name in ('joint_7','two_stage')}
    for learner in METHODS:
        f=fit[alg[fit]==learner]
        v=np.flatnonzero(alg[val]==learner)
        if not len(v):
            continue
        if not len(f):
            raise ValueError(f'Missing learner training stratum: {learner}')
        for name,p in [('joint_7',multiclass(x[f],y[f],x[val[v]],7,cfg)),
                       ('two_stage',two_stage(x[f],y[f],x[val[v]],cfg))]:
            p=np.maximum(p,cfg['new_head_probability_floor'])
            p=p/p.sum(axis=1,keepdims=True)
            predictions[name][v]=p
    for p in predictions.values():
        scores(p,y[val])
    return predictions


def calibrated(a,b,y,fit,val,cfg):
    tfit=minimize_scalar(lambda t:loss(temperature(a[fit],t),y[fit]),
        bounds=cfg['temperature_bounds'],method='bounded')
    wfit=minimize_scalar(lambda w:loss(w*a[fit]+(1-w)*b[fit],y[fit]),
        bounds=cfg['blend_oeh_weight_bounds'],method='bounded')
    if not tfit.success or not wfit.success:
        raise ValueError('Calibration optimization failed')
    t=min((*cfg['temperature_bounds'],1.,float(tfit.x)),
        key=lambda t:loss(temperature(a[fit],t),y[fit]))
    w=min((0.,1.,float(wfit.x)),key=lambda w:loss(w*a[fit]+(1-w)*b[fit],y[fit]))
    return {'current':a[val], 'temperature':temperature(a[val],t),
            'blend_b2':w*a[val]+(1-w)*b[val]}, {'temperature':t,'oeh_weight':w}


def fit_candidates(x,y,alg,a,b,fit,val,cfg):
    p,params=calibrated(a,b,y,fit,val,cfg)
    p.update(heads(x,y,alg,fit,val,cfg))
    return p,params


def grouped_loss(p,y,groups):
    frame=pd.DataFrame({'mdp':groups,'loss':scores(p,y)[:,1]})
    return float(frame.groupby('mdp').loss.mean().mean())


def evaluate_splits(label,x,a,b,splits,cfg,mode):
    y=label.event.to_numpy(int)
    groups=label.mdp_id.to_numpy()
    alg=label.algorithm.to_numpy()
    candidates=cfg['candidates']
    output={c:np.full((len(y),7),np.nan) for c in (*candidates,'nested_selected')}
    records=[]
    for outer,(fit,val) in enumerate(splits):
        assert_split(groups,fit,val)
        inner_predictions={c:np.full((len(fit),7),np.nan) for c in candidates}
        inner_records=[]
        for inner,(f,v) in enumerate(GroupKFold(cfg['inner_folds']).split(x[fit],groups=groups[fit])):
            ff,vv=fit[f],fit[v]
            assert_split(groups,ff,vv)
            pred,params=fit_candidates(x,y,alg,a,b,ff,vv,cfg)
            for name in candidates:
                inner_predictions[name][v]=pred[name]
            inner_records.append(dict(fold=inner,train_mdps=sorted(set(groups[ff])),
                validation_mdps=sorted(set(groups[vv])),parameters=params))
        losses={c:grouped_loss(p,y[fit],groups[fit]) for c,p in inner_predictions.items()}
        selected=min(candidates,key=lambda c:losses[c])
        tick=time.perf_counter()
        predictions,params=fit_candidates(x,y,alg,a,b,fit,val,cfg)
        elapsed=time.perf_counter()-tick
        for name in candidates:
            output[name][val]=predictions[name]
        output['nested_selected'][val]=predictions[selected]
        records.append(dict(outer_fold=outer,train_mdps=sorted(set(groups[fit])),
            validation_mdps=sorted(set(groups[val])),inner_folds=inner_records,
            inner_oof_log_loss=losses,selected=selected,parameters=params,
            all_candidates_outer_fit_predict_seconds=elapsed))
        print(f'{mode} fold {outer+1}: selected={selected}; inner losses={losses}',flush=True)
    for p in output.values():
        scores(p,y)
    rows=[]
    for name,p in output.items():
        frame=label[list(D.KEYS)].copy()
        frame['candidate']=name
        frame[P]=p
        frame[['brier','log_loss']]=scores(p,y)
        frame['event']=y
        frame['family']=label.mdp_id.str.removeprefix('oehfinal120_').str.rsplit('_',n=1).str[0]
        frame['event_status']=np.where(y==6,'censored','observed')
        rows.append(frame)
    return pd.concat(rows,ignore_index=True),records


def aggregate(detail):
    mdp=detail.groupby(['candidate','mdp_id'],as_index=False)[['brier','log_loss']].mean()
    summary=mdp.groupby('candidate',as_index=False).agg(mdps=('mdp_id','size'),
        brier=('brier','mean'),log_loss=('log_loss','mean'))
    subgroup=[]
    for dimension in ('family','algorithm','event_status'):
        within=detail.groupby(['candidate',dimension,'mdp_id'],as_index=False)[['brier','log_loss']].mean()
        grouped=within.groupby(['candidate',dimension],as_index=False).agg(
            mdps=('mdp_id','size'),brier=('brier','mean'),log_loss=('log_loss','mean'))
        grouped=grouped.rename(columns={dimension:'value'})
        grouped['dimension']=dimension
        subgroup.append(grouped)
    return mdp,summary,pd.concat(subgroup,ignore_index=True)


def bounds(mdp,cfg):
    matrix=mdp.pivot(index='mdp_id',columns='candidate',values=['brier','log_loss'])
    descriptors=[]
    vectors=[]
    for baseline,candidate in cfg['contrasts']:
        for metric in cfg['metrics']:
            vector=(matrix[(metric,baseline)]-matrix[(metric,candidate)]).to_numpy()
            vectors.append(vector)
            descriptors.append(dict(baseline=baseline,candidate=candidate,metric=metric,
                mean_error_reduction=float(vector.mean()),mdps_improved=int((vector>0).sum()),
                mdps_worse=int((vector<0).sum())))
    effects=np.column_stack(vectors)
    mean=effects.mean(0)
    se=effects.std(0,ddof=1)/np.sqrt(len(effects))
    nonzero=se>1e-15
    index=np.random.default_rng(cfg['seed']).integers(0,len(effects),
        size=(cfg['conditional_bootstrap_draws'],len(effects)))
    z=np.zeros((len(index),effects.shape[1]))
    z[:,nonzero]=(effects[index].mean(1)[:,nonzero]-mean[nonzero])/se[nonzero]
    critical=float(np.quantile(z.max(1),.95))
    for row,m,s in zip(descriptors,mean,se):
        row['conditional_simultaneous_lower_95']=float(m-critical*s)
        row['interpretation']='cached OOF exploratory only; excludes refit uncertainty and historical selection'
    return pd.DataFrame(descriptors),critical


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    cfg=D.read_json(CONFIG)
    label,x,arrays,hashes=verified_data()
    for p in (CONFIG,Path(__file__),D.ROOT/'scripts/analysis/probe_forecast_calibration_20261012.py',
              D.ROOT/'scripts/analysis/compare_difficulty.py',DEV/'preoutcome_seal.json',DEV/'result_manifest.json'):
        hashes[p.relative_to(D.ROOT).as_posix()]=D.digest(p)
    folds=list(GroupKFold(cfg['outer_folds']).split(x,groups=label.mdp_id))
    families=label.mdp_id.str.removeprefix('oehfinal120_').str.rsplit('_',n=1).str[0].to_numpy()
    lofo=[(np.flatnonzero(families!=f),np.flatnonzero(families==f)) for f in sorted(set(families))]
    OUT.mkdir(parents=True)
    # Persist the full candidate/contrast family before any experiment fitting.
    (OUT/'plan.json').write_text(json.dumps(dict(config=cfg,input_source_sha256=hashes,
        outer_validation_mdps=[sorted(set(label.mdp_id.iloc[v])) for _,v in folds],
        lofo_validation_mdps=[sorted(set(label.mdp_id.iloc[v])) for _,v in lofo]),indent=2)+'\n',
        encoding='utf-8',newline='\n')
    tick=time.perf_counter()
    records={}
    critical=None
    for mode,split in [('grouped_nested',folds),('family_holdout_nested',lofo)]:
        detail,records[mode]=evaluate_splits(label,x,arrays['oeh_final'],arrays['b2_matched'],split,cfg,mode)
        mdp,summary,subgroup=aggregate(detail)
        for name,frame in [('predictions',detail),('mdp_scores',mdp),('summary',summary),('subgroups',subgroup)]:
            frame.to_csv(OUT/f'{mode}_{name}.csv',index=False,lineterminator='\n')
        if mode=='grouped_nested':
            contrast,critical=bounds(mdp,cfg)
            contrast.to_csv(OUT/'conditional_contrasts.csv',index=False,lineterminator='\n')
        print(mode+' summary\n'+summary.to_string(index=False),flush=True)
    import scipy,sklearn
    result=dict(status='complete_exploratory_not_confirmatory',configuration_sha256=D.digest(CONFIG),
        candidate_promoted=False,frozen_artifacts_changed=False,
        fit_confirmatory_cohort7=False,elapsed_seconds=time.perf_counter()-tick,
        nominal_head_features=25,nominal_output_trees_per_learner_per_head_candidate=1400,
        conditional_max_t_critical=critical,fold_records=records,
        versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                      sklearn=sklearn.__version__,xgboost=xgb.__version__),
        caveat='Opened fourth cohort was historically reused. Grouped OOF and family holdout are exploratory, not fresh independent confirmation. Cached-score bounds omit model-refitting dependence.',
        artifact_sha256={p.name:D.digest(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    for p,h in hashes.items():
        if D.digest(D.ROOT/p)!=h:
            raise ValueError(f'Input/source changed during experiment: {p}')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Completed exploratory experiments: {OUT}',flush=True)


if __name__=='__main__':
    main()
