"""Reassess the original sealed selector using existing E1 outcomes only.

All oracle summaries and alternative component scores are posthoc diagnostics.
This command never trains an RL learner or updates the original E1 directory.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json
from environments.benchmark_environments import benchmark_specs

ALGORITHMS = ('cfqi_tabular', 'iql_tabular', 'behavior_cloning_tabular')
COMPONENTS = ('B2_OEH3', 'B2_OEH3_LCOEH4')


def paired_summary(selected, reference):
    a, b = np.asarray(selected, float), np.asarray(reference, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Complete finite paired scores required')
    sa, sb = a >= .95, b >= .95
    return dict(units=len(a), mean_return=float(a.mean()), reference_return=float(b.mean()),
                delta_return=float((a-b).mean()), success_rate=float(sa.mean()),
                reference_success=float(sb.mean()), delta_success=float((sa.astype(int)-sb.astype(int)).mean()),
                success_gained=int((sa & ~sb).sum()), success_lost=int((sb & ~sa).sum()),
                return_improved_units=int((a>b).sum()), return_harmed_units=int((a<b).sum()))


def seed_cells(rows):
    cells = defaultdict(dict)
    for r in rows:
        if r['run_status'] != 'observed':
            raise ValueError('Failure cannot be removed from diagnosis')
        key = (r['mdp_id'], r['policy_family'], r['algorithm'], int(r['training_size_multiplier']))
        seed = int(r['training_seed'])
        if seed in cells[key]:
            raise ValueError('Duplicate seed')
        value = float(r['normalized_return'])
        if (str(r['success']).lower() == 'true') != (value >= .95):
            raise ValueError('Success metric mismatch')
        cells[key][seed] = value
    if any(set(v) != set(range(10)) for v in cells.values()):
        raise ValueError('Missing seed')
    return {k: np.array([v[i] for i in range(10)]) for k, v in cells.items()}


def run(e1, output):
    reserve_output_dir(output)
    execution = json.loads((e1 / 'e1_execution_summary.json').read_text('utf-8'))
    if execution['status'] != 'complete':
        raise ValueError('Completed E1 required')
    names = ('stochastic_primary_core_run_results.csv', 'stochastic_primary_selector_decisions.csv',
             'stochastic_primary_return_oof_predictions.csv')
    hashes = {n: sha256_file(e1 / n) for n in names}
    if any(execution['artifact_sha256'][n] != h for n, h in hashes.items()):
        raise ValueError('Original E1 result seal mismatch')
    feature_path = e1 / 'pilot_features/features.csv'
    prior = json.loads((e1 / 'e1_diagnostics.json').read_text('utf-8'))
    hashes['pilot_features/features.csv'] = sha256_file(feature_path)
    if hashes['pilot_features/features.csv'] != prior['input_sha256']['features.csv']:
        raise ValueError('Original pilot feature seal mismatch')
    write_json(output / 'plan.json', dict(role='posthoc_root_cause_diagnostic_not_new_evaluation',
        original_run_id=execution['run_id'], original_manifest_sha256=execution['manifest_sha256'],
        input_sha256=hashes, source_sha256=sha256_file(Path(__file__)), E1_rerun=False,
        evaluation_outcomes_used_for_new_model_fitting=False))
    cells = seed_cells(read_csv_rows(e1 / names[0]))
    decisions = read_csv_rows(e1 / names[1]); predictions = read_csv_rows(e1 / names[2])
    features = {(r['mdp_id'], r['policy_family'], r['algorithm']): r for r in read_csv_rows(feature_path)}
    pred = {(r['mdp_id'], r['policy_family'], r['algorithm'], r['feature_set']): r for r in predictions}
    primary_keys = {k[:3] for k in pred}
    features = {k:v for k,v in features.items() if k in primary_keys}
    if len(decisions) != 420 or len({(r['mdp_id'], r['policy_family']) for r in decisions}) != 420:
        raise ValueError('Expected all 84 x 5 groups')
    if len(cells) != 84*5*3*6 or len(pred) != 84*5*3*2:
        raise ValueError('Incomplete original grid')
    families = {s.instance_id: s.family for s in benchmark_specs()}
    grouped = []; chosen = []; reference = []; oracle_return = []; oracle_success = []
    component_selected = {n: [] for n in COMPONENTS}
    component_residuals = {n: [] for n in COMPONENTS}
    switch_by_learner = defaultdict(lambda: [[], []]); strata = defaultdict(lambda: [[], []])
    switched_wrong_sign = switched_positive_return_negative_success = 0
    for d in decisions:
        group = (d['mdp_id'], d['policy_family']); fixed = d['fold_best_fixed_algorithm']
        values = {a: cells[(*group, a, 10)] for a in ALGORITHMS}
        means = {a: float(v.mean()) for a, v in values.items()}
        success = {a: float((v >= .95).mean()) for a, v in values.items()}
        scores = {a: float(np.mean([float(pred[(*group, a, n)]['predicted_return']) for n in COMPONENTS])) for a in ALGORITHMS}
        selected = max(ALGORITHMS, key=lambda a: (scores[a], -ALGORITHMS.index(a)))
        if selected != d['selected_algorithm'] or any(abs(scores[a]-float(d['score_'+a])) > 1e-12 for a in ALGORITHMS):
            raise ValueError('Sealed selector does not reproduce from saved predictions')
        switched = selected != fixed
        delta = means[selected]-means[fixed]; delta_success = success[selected]-success[fixed]
        switched_wrong_sign += int(switched and delta < 0)
        switched_positive_return_negative_success += int(switched and delta > 0 and delta_success < 0)
        oracle = max(ALGORITHMS, key=lambda a: (means[a], -ALGORITHMS.index(a)))
        oracle_return.extend(values[oracle]); oracle_success.append(max(success.values()))
        chosen.extend(values[selected]); reference.extend(values[fixed])
        if switched:
            switch_by_learner[selected][0].extend(values[selected]); switch_by_learner[selected][1].extend(values[fixed])
        family = families[group[0]]
        strata[(family, group[1])][0].extend(values[selected]); strata[(family, group[1])][1].extend(values[fixed])
        for n in COMPONENTS:
            cp = {a: float(pred[(*group, a, n)]['predicted_return']) for a in ALGORITHMS}
            ca = max(ALGORITHMS, key=lambda a: (cp[a], -ALGORITHMS.index(a)))
            component_selected[n].extend(values[ca])
            component_residuals[n].extend([(cp[a]-cp[fixed])-(means[a]-means[fixed]) for a in ALGORITHMS if a != fixed])
        paired = paired_summary(values[selected], values[fixed])
        grouped.append(dict(mdp_id=group[0], policy_family=group[1], environment_family=family,
            selected=selected, reference=fixed, switched=switched, predicted_gain=scores[selected]-scores[fixed],
            delta_return=delta, delta_success=delta_success, oracle_return_regret=means[oracle]-means[selected],
            reference_dominates_all_in_return=means[fixed] >= max(means.values()),
            fallback=str(d['fallback']).lower() == 'true',
            success_gained=paired['success_gained'], success_lost=paired['success_lost']))
    selected_summary = paired_summary(chosen, reference)
    oracle = paired_summary(oracle_return, reference)
    oracle['success_max_per_group_upper_envelope'] = float(np.mean(oracle_success))
    oracle['deployable'] = False
    stratum_rows = [dict(environment_family=k[0], policy_family=k[1], **paired_summary(*v)) for k, v in sorted(strata.items())]
    component_rows = {n: paired_summary(v, reference) for n, v in component_selected.items()}
    errors = np.array([component_residuals[n] for n in COMPONENTS])
    contract = json.loads((ROOT/'docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json').read_text('utf-8'))
    extreme = []
    for algorithm in ALGORITHMS:
        for fold in range(5):
            records = [r for r in predictions if r['algorithm'] == algorithm and r['feature_set'] == COMPONENTS[1]]
            train = [r for r in records if int(r['outer_fold']) != fold]
            test = [r for r in records if int(r['outer_fold']) == fold]
            def matrix(rows):
                return np.array([[float(features[(r['mdp_id'],r['policy_family'],algorithm)][f]) for f in contract['feature_sets'][COMPONENTS[1]]] for r in rows])
            x, y = matrix(train), matrix(test); sd = x.std(axis=0); sd[sd <= 1e-12] = 1
            extreme.extend(np.max(np.abs((y-x.mean(axis=0))/sd), axis=1))
    censor = {a: dict(rows=sum(r['algorithm']==a for r in features.values()),
        censored=sum(float(r['lcoeh4_requirement_censored']) > .5 for r in features.values() if r['algorithm']==a),
        mean_independence_fraction=float(np.mean([float(r['lcoeh4_trajectory_independence_fraction']) for r in features.values() if r['algorithm']==a]))) for a in ALGORITHMS}
    # Applying the x10 decision to other recorded budgets is diagnostic only.
    budget_replays = []
    for budget in (1,2,5,10,20,50):
        a = np.concatenate([cells[(d['mdp_id'],d['policy_family'],d['selected_algorithm'],budget)] for d in decisions])
        b = np.concatenate([cells[(d['mdp_id'],d['policy_family'],d['fold_best_fixed_algorithm'],budget)] for d in decisions])
        budget_replays.append(dict(budget=budget, role='posthoc_fixed_x10_decision_replay', **paired_summary(a,b)))
    report = dict(status='complete', role='posthoc_root_cause_diagnostic_not_confirmation',
        groups=420, base_mdps=84, original_selector_reproduced_groups=420, original=selected_summary,
        selection_counts=dict(Counter(d['selected_algorithm'] for d in decisions)),
        reference_counts=dict(Counter(d['fold_best_fixed_algorithm'] for d in decisions)),
        switched_groups=sum(r['switched'] for r in grouped), wrong_sign_switches=switched_wrong_sign,
        positive_return_negative_success_switches=switched_positive_return_negative_success,
        reference_return_dominant_groups=sum(r['reference_dominates_all_in_return'] for r in grouped),
        fallback_groups=sum(r['fallback'] for r in grouped), oracle_learner_envelope=oracle,
        switch_by_learner={a: paired_summary(*v) for a, v in switch_by_learner.items()},
        component_counterfactuals=component_rows, paired_error_component_correlation=float(np.corrcoef(errors)[0,1]),
        paired_prediction_rmse={n: float(np.sqrt(np.mean(np.square(component_residuals[n])))) for n in COMPONENTS},
        LC_censoring=censor, max_abs_outer_standardized_feature=float(max(extreme)),
        feature_rows_above_5_training_sd=int(sum(z>5 for z in extreme)),
        E1_rerun=False, new_model_fitted_on_E1=False)
    write_csv_rows(output/'group_decomposition.csv', grouped, list(grouped[0]))
    write_csv_rows(output/'strata.csv', stratum_rows, list(stratum_rows[0]))
    write_csv_rows(output/'budget_replay.csv', budget_replays, list(budget_replays[0]))
    write_json(output/'report.json', report)
    write_json(output/'manifest.json', {p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e1', type=Path, default=ROOT/'artifacts/week6/oehv4e_confirmatory_v1_1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.e1, args.output)


if __name__ == '__main__':
    main()
