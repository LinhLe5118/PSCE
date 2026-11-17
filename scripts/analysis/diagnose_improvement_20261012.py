"""Post-outcome diagnostics and fixed development-only selector probes.

Never changes a frozen candidate, seal, or verdict. No confirmatory fitting.
The opened fourth cohort is consumed development data, not a new test set.
"""
from collections import defaultdict
import json
from pathlib import Path

from scripts.analysis import export_week8_d1 as D

ROOT = D.ROOT
OUT = ROOT / 'artifacts/week8/improvement_2026-10-12'


def family(mdp):
    return mdp.removeprefix('oehfinal120_').rsplit('_', 1)[0]


def diagnostic():
    D.audit()
    labels = D.unique(D.read_csv(D.BUNDLE / 'labels.csv'), D.KEYS)
    groups = defaultdict(list)
    for row in D.read_csv(D.BUNDLE / 'predictions_preoutcome.csv'):
        key = tuple(row[k] for k in D.KEYS)
        label = labels[key]
        if label['panel'] != 'primary' or row['method'] not in ('oeh_final', 'b2_matched'):
            continue
        p = [float(row[f'p_{j}']) for j in range(7)]
        event = int(label['event'])
        score = D.forecast_score(p, event)
        for dimension, value in [('learner', row['algorithm']), ('family', family(row['mdp_id'])),
                                 ('event_status', 'censored' if event == 6 else 'observed')]:
            groups[(dimension, value, row['method'])].append((score[0], score[1], p[6], event == 6))
    forecast = []
    for (dimension, value, method), rows in sorted(groups.items()):
        forecast.append(dict(dimension=dimension, value=value, method=method, tasks=len(rows),
                             brier=sum(r[0] for r in rows)/len(rows),
                             log_loss=sum(r[1] for r in rows)/len(rows),
                             predicted_censored=sum(r[2] for r in rows)/len(rows),
                             actual_censored=sum(r[3] for r in rows)/len(rows)))
    selection = D.read_csv(D.BUNDLE / 'selection_group_outcomes.csv')
    choices = D.read_json(D.BUNDLE / 'oeh_choices_preoutcome.json')
    switches = defaultdict(list)
    fixed = 'fixed:cfqi_tabular'
    for row in selection:
        key = f"{row['mdp_id']}|{row['policy_family']}"
        choice = choices[key]
        switches[(family(row['mdp_id']), choice)].append((
            float(row['oeh_final_selection|ret']) - float(row[fixed+'|ret']),
            float(row['oeh_final_selection|suc']) - float(row[fixed+'|suc'])))
    selection_rows = [dict(family=f, chosen=c, groups=len(rows),
                           return_delta=sum(r[0] for r in rows)/len(rows),
                           success_delta=sum(r[1] for r in rows)/len(rows),
                           return_worse=sum(r[0] < -1e-12 for r in rows),
                           success_worse=sum(r[1] < -1e-12 for r in rows))
                      for (f,c), rows in sorted(switches.items())]
    return forecast, selection_rows


def development_probe():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from scripts.analysis.develop_fixed_selector_20261004 import design, model, OUT as DEV
    key, x = design(DEV)
    y = pd.read_csv(DEV / 'selection_group_outcomes.csv')
    frame = key.merge(y, on=['mdp_id', 'policy_family'], validate='one_to_one', how='inner')
    pos = pd.MultiIndex.from_frame(key).get_indexer(pd.MultiIndex.from_frame(frame[['mdp_id', 'policy_family']]))
    x = x.iloc[pos].reset_index(drop=True)
    if len(frame) != 420 or frame.mdp_id.nunique() != 84:
        raise ValueError('Wrong development panel')
    methods = [c[6:-4] for c in frame if c.startswith('fixed:') and c.endswith('|suc')]
    suc = frame[[f'fixed:{m}|suc' for m in methods]].to_numpy()
    ret = frame[[f'fixed:{m}|ret' for m in methods]].to_numpy()
    cfqi = methods.index('cfqi_tabular')
    predictions = {}
    fold_rows = []
    for mode in ('absolute', 'paired'):
        targets = [suc, ret] if mode == 'absolute' else [suc-suc[:,cfqi,None], ret-ret[:,cfqi,None]]
        oof = [np.full_like(suc, np.nan), np.full_like(ret, np.nan)]
        for fold, (fit, val) in enumerate(GroupKFold(5).split(x, groups=frame.mdp_id)):
            assert not set(frame.mdp_id.iloc[fit]) & set(frame.mdp_id.iloc[val])
            if mode == 'absolute':
                fold_rows.append(dict(fold=fold, train_mdps=sorted(set(frame.mdp_id.iloc[fit])),
                                      validation_mdps=sorted(set(frame.mdp_id.iloc[val]))))
            for target, table in zip(targets, oof):
                table[val] = model().fit(x.iloc[fit], target[fit]).predict(x.iloc[val])
        assert all(np.isfinite(p).all() for p in oof)
        ds, dr = oof if mode == 'paired' else [a-a[:,cfqi,None] for a in oof]
        predictions[mode] = (ds, dr)
    rules = [('absolute_current_rule', 'absolute', False),
             ('absolute_both_positive', 'absolute', True),
             ('paired_same_rule', 'paired', False),
             ('paired_both_positive', 'paired', True)]
    result, group_rows = [], []
    for name, mode, veto in rules:
        ds, dr = predictions[mode]
        score = ds + .15*dr
        if veto:
            score = np.where((ds > 0) & (dr > 0), score, -np.inf)
            score[:,cfqi] = 0
        chosen = score.argmax(1)
        chosen = np.where(score[np.arange(len(chosen)), chosen] > .1, chosen, cfqi)
        selected_s = suc[np.arange(len(chosen)), chosen]
        selected_r = ret[np.arange(len(chosen)), chosen]
        result.append(dict(rule=name, groups=len(frame), switches=int((chosen!=cfqi).sum()),
                           mean_return=float(selected_r.mean()), success_rate=float(selected_s.mean()),
                           return_vs_cfqi=float((selected_r-ret[:,cfqi]).mean()),
                           success_vs_cfqi=float((selected_s-suc[:,cfqi]).mean()),
                           return_vs_v5=float((selected_r-frame['oeh_v5_router|ret'].to_numpy()).mean()),
                           success_vs_v5=float((selected_s-frame['oeh_v5_router|suc'].to_numpy()).mean())))
        for i, row in frame.iterrows():
            group_rows.append(dict(rule=name, mdp_id=row.mdp_id, policy_family=row.policy_family,
                                   chosen=methods[chosen[i]], return_delta=float(selected_r[i]-ret[i,cfqi]),
                                   success_delta=float(selected_s[i]-suc[i,cfqi])))
    return result, group_rows, fold_rows


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    forecast, switches = diagnostic()
    probe, groups, folds = development_probe()
    OUT.mkdir(parents=True)
    for name, rows in [('forecast_diagnostics.csv',forecast), ('selection_switches.csv',switches),
                       ('development_probes.csv',probe), ('development_group_effects.csv',groups)]:
        (OUT / name).write_text(D.csv_text(rows), encoding='utf-8', newline='\n')
    (OUT / 'scope.json').write_text(json.dumps(dict(
        status='exploratory_only_not_confirmatory', new_learner_training=False,
        selector_meta_models_fitted_on_opened_fourth_cohort_only=True,
        confirmatory_outcomes_used_for_diagnostics_not_fitting=True,
        candidate_rules_fixed_before_this_run=True, new_statistical_approval=False,
        multiplicity_corrected_inference_performed=False,
        feature_access='same CFQI pilot + preoutcome forecast probabilities + family label as existing development design',
        probes='four fixed rules, floor .1, success + .15 return, ExtraTrees 150 leaf7 max_features.8 seed42',
        note='Group OOF is exploratory after prior reuse; needs nested candidate selection and fresh sealed cohort.',
        development_folds=folds,
        source_sha256={p.relative_to(ROOT).as_posix(): D.digest(p) for p in [
            Path(__file__), ROOT/'scripts/analysis/develop_fixed_selector_20261004.py',
            ROOT/'artifacts/oeh_final_unseen_120_2026-10-02/pilot_features.csv',
            ROOT/'artifacts/oeh_final_unseen_120_2026-10-02/predictions_preoutcome.csv',
            ROOT/'artifacts/oeh_final_unseen_120_2026-10-02/selection_group_outcomes.csv',
            D.BUNDLE/'labels.csv', D.BUNDLE/'predictions_preoutcome.csv',
            D.BUNDLE/'selection_group_outcomes.csv', D.BUNDLE/'oeh_choices_preoutcome.json']}
        ), ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(forecast=forecast, switches=switches, development=probe), indent=2))


if __name__ == '__main__':
    main()
