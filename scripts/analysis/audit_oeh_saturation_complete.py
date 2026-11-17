"""Audit whether EH-aligned OEH has any remaining mechanical saturation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'code'))
from oeh_eh_aligned import _profile
from evaluation.io import reserve_output_dir, sha256_file, write_json


def _read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _classify(summary, curves):
    by_dataset = {}
    for row in curves:
        by_dataset.setdefault(row['dataset_id'], []).append(row)
    counts = {}
    violations = []
    details = []
    for row in summary:
        case = by_dataset[row['dataset_id']]
        observed = [x for x in case if x['est_eb_raw'] != '']
        selectable = [x for x in case if x['est_eb_score'] != '']
        positive = [x for x in case if x['est_minimum_gap'] != '' and float(x['est_minimum_gap']) > 0]
        false_floor = [x for x in positive if x['est_eb_sample_requirement'] != ''
                       and float(x['est_eb_sample_requirement']) <= 1+1e-12]
        if false_floor:
            violations.append(dict(dataset_id=row['dataset_id'], kind='positive_gap_requirement_at_floor',
                                   depths=[int(x['est_depth']) for x in false_floor]))
        if not observed:
            category = 'no_identified_depth'
        elif len(observed) == 1:
            category = 'one_identified_depth'
        else:
            spread = np.ptp([float(x['est_eb_raw']) for x in observed])
            if spread <= 1e-9:
                category = 'multi_identified_flat_total_score'
                violations.append(dict(dataset_id=row['dataset_id'], kind=category))
            elif not selectable:
                category = 'identified_no_admissible_depth'
            else:
                category = 'finite_nonflat_selector'
        counts[category] = counts.get(category, 0)+1
        details.append(dict(dataset_id=row['dataset_id'], structural_family=row['structural_family'],
                            category=category, identified_depths=len(observed),
                            selectable_depths=len(selectable), positive_gap_depths=len(positive)))
    return counts, violations, details


def _delayed_reward_control():
    """A shallow behavior leaf is wrong; one backup exposes delayed reward."""
    states, actions, gamma = 3, 2, .9
    p = np.zeros((states, actions, states)); r = np.zeros((states, actions))
    p[0, 0, 2], r[0, 0] = 1, .2
    p[0, 1, 1] = 1
    p[1, :, 2], r[1] = 1, [1., 0.]
    p[2, :, 2] = 1
    counts = np.array([[100., 100.], [10., 90.], [0., 0.]])
    beta = np.divide(counts, np.maximum(counts.sum(1, keepdims=True), 1))
    beta[2] = .5
    result = _profile(p, r, beta, counts, np.array([1., 0., 0.]), [2],
                      lambda leaf: np.zeros((states, actions)), gamma=gamma,
                      k_max=4, epsilon=.05, exact_model=True)
    rows = result['records']
    return dict(point_selected_depth=result['point_selected_depth'],
                eb_selected_depth=result['eb_selected_depth'],
                depth0_admissible=rows[0]['admissible'], depth1_admissible=rows[1]['admissible'],
                depth0_gap=rows[0]['minimum_gap'], depth1_gap=rows[1]['minimum_gap'],
                depth0_point_raw=rows[0]['point_raw'], depth1_point_raw=rows[1]['point_raw'],
                pass_control=(result['point_selected_depth'] == 1
                              and result['eb_selected_depth'] == 1
                              and not rows[0]['admissible'] and rows[1]['admissible']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-artifact', type=Path,
        default=ROOT/'artifacts/research/oeh_eh_aligned_final_v2_2026-09-19')
    parser.add_argument('--old-artifact', type=Path,
        default=ROOT/'artifacts/research/oeh_development_smoke_numpy_2026-09-18')
    parser.add_argument('--output-dir', type=Path,
        default=ROOT/'artifacts/research/oeh_saturation_completion_2026-09-19')
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)
    summary = _read_csv(args.new_artifact/'summary.csv')
    curves = _read_csv(args.new_artifact/'curves.csv')
    if len(summary) != 225 or len(curves) != 225*101:
        raise ValueError('Incomplete EH-aligned artifact')
    new_report = json.loads((args.new_artifact/'report.json').read_text(encoding='utf-8'))
    old_report = json.loads((args.old_artifact/'report.json').read_text(encoding='utf-8'))
    all_counts, all_violations, details = _classify(summary, curves)
    stochastic_rows = [r for r in summary if r['structural_family'] != 'lock_structured']
    ids = {r['dataset_id'] for r in stochastic_rows}
    stochastic_counts, stochastic_violations, _ = _classify(
        stochastic_rows, [r for r in curves if r['dataset_id'] in ids])
    control = _delayed_reward_control()
    violations = all_violations+stochastic_violations
    passed = not violations and control['pass_control']
    report = dict(
        status='PASS' if passed else 'FAIL',
        operational_definition={
            'mechanical_saturation': 'two or more identified depths have the same finite total EH score, or a positive gap is assigned sample requirement <= 1',
            'not_saturation': 'zero/one identified depth, no admissible depth, or explicit abstention; these are non-identification/censoring',
            'positive_control': 'a delayed-reward MDP must make both point and EB selectors choose depth 1'},
        old_v1=dict(assessable_curves=old_report['v1_assessable_curves'],
                    all_floor_curves=old_report['v1_all_floor_curves']),
        new_all=dict(cases=len(summary), classifications=all_counts,
                     mechanical_saturation_cases=len(all_violations)),
        new_stochastic=dict(cases=len(stochastic_rows), classifications=stochastic_counts,
                            mechanical_saturation_cases=len(stochastic_violations),
                            evidence_nonflat=new_report['stochastic']['eb_evidence_nonflat'],
                            total_score_nonflat=new_report['stochastic']['eb_score_nonflat'],
                            selector_available=new_report['stochastic']['eb_selector_available']),
        delayed_reward_positive_control=control, violations=violations,
        interpretation='Complete escape means no artificial finite floor remains. It does not require fabricating variation when the pilot is uninformative; such cases abstain.',
        source_sha256={
            'code/oeh_eh_aligned.py':sha256_file(ROOT/'code/oeh_eh_aligned.py'),
            'scripts/analysis/audit_oeh_saturation_complete.py':sha256_file(ROOT/'scripts/analysis/audit_oeh_saturation_complete.py'),
            'new_report.json':sha256_file(args.new_artifact/'report.json'),
            'new_summary.csv':sha256_file(args.new_artifact/'summary.csv'),
            'new_curves.csv':sha256_file(args.new_artifact/'curves.csv'),
            'old_report.json':sha256_file(args.old_artifact/'report.json')})
    write_json(args.output_dir/'report.json', report)
    with (args.output_dir/'classification.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    md = f"""# OEH saturation completion audit (2026-09-19)

Status: **{report['status']}**.

For this post-hoc development audit, mechanical saturation is defined as either (i) two or more
identified depths having an identical finite total EH score, or (ii) a positive
gap receiving sample requirement at the artificial floor 1. Zero/one identified
depth and no admissible depth are reported as non-identification/censoring.

- OEH-v1: {old_report['v1_all_floor_curves']}/{old_report['v1_assessable_curves']} assessable curves at the all-one multiplier floor.
- EH-aligned, all: {all_counts}; mechanical violations: {len(all_violations)}.
- EH-aligned, stochastic: {stochastic_counts}; mechanical violations: {len(stochastic_violations)}.
- Delayed-reward control: point depth {control['point_selected_depth']}, EB depth {control['eb_selected_depth']}, pass={control['pass_control']}.

The result establishes removal of the old mechanical floor under this explicit
definition. It does not establish strong predictive utility or a finite-sample
certificate; those remain separate validation questions.
"""
    (args.output_dir/'REPORT.md').write_text(md, encoding='utf-8')
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
