import sys, numpy as np, pandas as pd
d=sys.argv[1]
lab=pd.read_csv(f'{d}/labels.csv'); pr=pd.read_csv(f'{d}/predictions_preoutcome.csv')
key=['mdp_id','policy_family','algorithm']
m=pr.merge(lab[key+['event','panel']],on=key,suffixes=('','_l'))
assert len(m)==len(pr)
P=m[[f'p_{i}' for i in range(7)]].to_numpy(); e=m.event.to_numpy().astype(int)
print('prob sum range',P.sum(1).min(),P.sum(1).max(),'min p',P.min())
F=np.cumsum(P,1)[:,:6]; Y=(e[:,None]<=np.arange(6)[None,:]).astype(float)
m['ibs']=((F-Y)**2).mean(1); m['ll']=-np.log(np.clip(P[np.arange(len(e)),e],1e-15,1))
g=m.groupby(['panel','method','mdp_id'])[['ibs','ll']].mean().reset_index()
s=g.groupby(['panel','method'])[['ibs','ll']].mean()
print(s.loc['primary'].sort_values('ibs'))
prim=g[g.panel=='primary'].pivot(index='mdp_id',columns='method')
# label dist
L=lab[lab.panel=='primary']; print(L.event.value_counts().sort_index(), len(L))
print(pd.crosstab(L.algorithm,L.event))
# bootstrap max-t simultaneous lower bounds, for oeh vs registered comparators
comps=['b2_matched','b2_plus_eri','b2_plus_bwd','train_frequency','coverage_adaptation','quality_adaptation','coverage_quality_adaptation','b2_generic','b2_plus_eh_oracle_matched','b2_plus_tq_saco_original','published_metrics_only']
D=np.column_stack([prim[(met,c)]-prim[(met,'oeh_final')] for c in comps for met in ['ibs','ll']])
n=len(D); mu=D.mean(0); se=D.std(0,ddof=1)/np.sqrt(n)
rng=np.random.default_rng(1); T=[]
for _ in range(10000):
    idx=rng.integers(0,n,n); Db=D[idx]; T.append(((Db.mean(0)-mu)/(Db.std(0,ddof=1)/np.sqrt(n))).max())
c=np.quantile(T,0.95); lb=mu-c*se
print('crit',c)
for i,(cc,met) in enumerate([(c_,mt) for c_ in comps for mt in ['ibs','ll']]): print(f'{cc:28s} {met:3s} gain={mu[i]:.6f} lb={lb[i]:.6f} min_lb_t? ')
# per-family
fam=prim.index.str.extract(r'oehfinal120_(.*)_\d+')[0].values
for f in np.unique(fam):
    sel=fam==f
    print(f, sel.sum(), 'ibs b2 %.5f oeh %.5f | ll b2 %.4f oeh %.4f'%(prim[('ibs','b2_matched')][sel].mean(),prim[('ibs','oeh_final')][sel].mean(),prim[('ll','b2_matched')][sel].mean(),prim[('ll','oeh_final')][sel].mean()))
# per learner
mm=m[m.panel=='primary']
print(mm[mm.method.isin(['b2_matched','oeh_final','b2_plus_tq_saco_original'])].groupby(['algorithm','method'])[['ibs','ll']].mean().unstack())
# censored vs finite
mm=mm.assign(cens=mm.event==6)
print(mm[mm.method.isin(['b2_matched','oeh_final'])].groupby(['cens','method'])[['ibs','ll']].agg(['mean','sum']))
# wins per MDP
w=(prim[('ll','b2_matched')]>prim[('ll','oeh_final')]).mean(); w2=(prim[('ibs','b2_matched')]>prim[('ibs','oeh_final')]).mean(); print('frac MDPs OEH better ll',w,'ibs',w2)
