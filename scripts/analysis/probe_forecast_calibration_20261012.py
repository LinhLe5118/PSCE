"""Fixed grouped-OOF calibration probes on the already-opened fourth cohort.

Exploratory, not confirmatory; no modification of frozen forecasts or verdicts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

from scripts.analysis import export_week8_d1 as D

DEV = D.ROOT / 'artifacts/oeh_final_unseen_120_2026-10-02'
OUT = D.ROOT / 'artifacts/week8/improvement_forecast_2026-10-12'


def temperature(p, t):
    logp = np.log(np.maximum(p, 1e-12))/t
    q = np.exp(logp-logp.max(axis=1,keepdims=True))
    return q/q.sum(axis=1,keepdims=True)


def loss(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    manifest = D.read_json(DEV/'result_manifest.json')
    seal = D.read_json(DEV/'preoutcome_seal.json')
    for name in ('labels.csv','predictions_preoutcome.csv'):
        expected = manifest['artifact_sha256'].get(name, seal['artifact_sha256'].get(name))
        if not expected or D.digest(DEV/name) != expected:
            raise ValueError(f'Unverified development input {name}')
    label = pd.read_csv(DEV/'labels.csv')
    label = label[label.panel=='primary'].sort_values(list(D.KEYS)).reset_index(drop=True)
    pred = pd.read_csv(DEV/'predictions_preoutcome.csv')
    arrays = {}
    for method in ('oeh_final','b2_matched'):
        frame = label[list(D.KEYS)].merge(pred[pred.method==method], on=list(D.KEYS),
                                          how='left',validate='one_to_one')
        arrays[method] = frame[[f'p_{j}' for j in range(7)]].to_numpy()
        if not np.isfinite(arrays[method]).all():
            raise ValueError('Incomplete development forecasts')
    y = label.event.to_numpy(dtype=int)
    if len(y)!=1260 or label.mdp_id.nunique()!=84:
        raise ValueError('Unexpected development panel')
    a,b = arrays['oeh_final'],arrays['b2_matched']
    candidates={'current':a.copy(),'blend_b2':np.full_like(a,np.nan),
                'temperature':np.full_like(a,np.nan)}
    parameters=[]
    for fold,(fit,val) in enumerate(GroupKFold(5).split(a,groups=label.mdp_id)):
        assert not set(label.mdp_id.iloc[fit]) & set(label.mdp_id.iloc[val])
        wfit=minimize_scalar(lambda w:loss(w*a[fit]+(1-w)*b[fit],y[fit]),
                             bounds=(0,1),method='bounded')
        if not wfit.success:
            raise ValueError('Blend optimization failed')
        # Include endpoints so a failed-to-help blend can retain the original.
        w=min((0.,float(wfit.x),1.),key=lambda w:loss(w*a[fit]+(1-w)*b[fit],y[fit]))
        tfit=minimize_scalar(lambda t:loss(temperature(a[fit],t),y[fit]),
                             bounds=(.5,2.),method='bounded')
        if not tfit.success:
            raise ValueError('Temperature optimization failed')
        t=min((.5,float(tfit.x),1.,2.),key=lambda t:loss(temperature(a[fit],t),y[fit]))
        candidates['blend_b2'][val]=w*a[val]+(1-w)*b[val]
        candidates['temperature'][val]=temperature(a[val],t)
        parameters.append(dict(fold=fold,oeh_weight=w,temperature=t,
                               fit_mdps=sorted(set(label.mdp_id.iloc[fit])),
                               validation_mdps=sorted(set(label.mdp_id.iloc[val]))))
    rows,summary=[],[]
    for name,p in candidates.items():
        scores=np.array([D.forecast_score(list(q),int(e)) for q,e in zip(p,y)])
        frame=pd.DataFrame(dict(mdp_id=label.mdp_id,brier=scores[:,0],log_loss=scores[:,1]))
        grouped=frame.groupby('mdp_id')[['brier','log_loss']].mean()
        summary.append(dict(candidate=name,mdps=len(grouped),brier=float(grouped.brier.mean()),
                            log_loss=float(grouped.log_loss.mean())))
        rows.extend(dict(candidate=name,mdp_id=mdp,brier=float(r.brier),log_loss=float(r.log_loss))
                    for mdp,r in grouped.iterrows())
    OUT.mkdir(parents=True)
    (OUT/'summary.csv').write_text(D.csv_text(summary),encoding='utf-8',newline='\n')
    (OUT/'mdp_scores.csv').write_text(D.csv_text(rows),encoding='utf-8',newline='\n')
    (OUT/'scope.json').write_text(json.dumps(dict(status='exploratory_only',
        confirmatory_fitting=False,statistical_approval=False,multiplicity_inference=False,
        method='Three fixed candidates; parameters fitted only on each MDP-grouped training fold, log loss objective',
        caveat='Fourth cohort was reused historically; grouped OOF is not fresh confirmation or nested candidate selection.',
        interpretation='This changes the forecast procedure, so it is not a feature-only matched-head effect.',
        temperature_range=[.5,2.],folds=parameters,
        source_sha256={p.relative_to(D.ROOT).as_posix():D.digest(p) for p in
                       (Path(__file__),DEV/'labels.csv',DEV/'predictions_preoutcome.csv',
                        DEV/'result_manifest.json',DEV/'preoutcome_seal.json')}
        ),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
