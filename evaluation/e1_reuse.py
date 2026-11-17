"""Verified, read-only reuse of the consumed E1 panel; never generates data."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from evaluation.io import read_csv_rows, sha256_file, verified_dataset_path
from scripts.training.run_empirical_requirement import _training_seed_index

ALGORITHMS = ('cfqi_tabular', 'iql_tabular', 'behavior_cloning_tabular')
GRID = (1, 2, 5, 10, 20, 50)


def verify_run_grid(manifest, rows, *, seeds=tuple(range(10))):
    """Verify pairing to registered buffers and completeness before fitting."""
    indexed = {}
    instances = {x['instance_id']:x for x in manifest['instances'] if x.get('split') == 'evaluation'}
    for record in manifest['datasets']:
        if record.get('benchmark_split') != 'evaluation' or record.get('stream') != 'training':
            continue
        key = (record['instance_id'], record['policy_family'], int(record['size_multiplier']), _training_seed_index(record))
        if key in indexed:
            raise ValueError('Duplicate E1 training identity')
        if record.get('horizon') != 100:
            raise ValueError('E1 collection horizon changed')
        indexed[key] = record
    groups = {(key[0],key[1]) for key in indexed}
    expected = {(mdp,behavior,budget,seed) for mdp,behavior in groups for budget in GRID for seed in seeds}
    if set(indexed) != expected or {key[0] for key in expected} != set(instances):
        raise ValueError('Incomplete registered E1 training grid')
    if any(len({behavior for mdp,behavior in groups if mdp == instance}) != 5 for instance in instances):
        raise ValueError('E1 requires five behavior groups per MDP')
    seen = set(); x10 = defaultdict(dict)
    for row in rows:
        key = (row['mdp_id'],row['policy_family'],int(row['training_size_multiplier']),int(row['training_seed']))
        method = row['algorithm']; identity = (*key,method)
        if identity in seen or method not in ALGORITHMS or key not in indexed:
            raise ValueError('Duplicate or unregistered E1 result')
        seen.add(identity)
        value = float(row['normalized_return'])
        if row['run_status'] != 'observed' or not np.isfinite(value):
            raise ValueError('Failed or nonfinite E1 core result')
        if row['buffer_sha256'] != indexed[key]['sha256']:
            raise ValueError('E1 result/training buffer hash mismatch')
        if row['success'] not in ('True','False') or (row['success'] == 'True') != (value >= .95):
            raise ValueError('E1 success is inconsistent with the registered threshold')
        if key[2] == 10:
            x10[(key[0],key[1])].setdefault(method,{})[key[3]] = (value,int(value >= .95))
    if seen != {(*key,method) for key in expected for method in ALGORITHMS}:
        raise ValueError('Incomplete E1 method/budget/seed grid')
    return indexed, dict(x10), instances


def load_e1_panel(root):
    root = Path(root).resolve()
    seal_path = root/'artifacts/week6/E1_INPUT_SEAL_2026-09-26.json'
    seal = json.loads(seal_path.read_text('utf-8'))
    dataset = verified_dataset_path(root, dict(file=seal['dataset_dir']+'/manifest.json',sha256=seal['manifest_sha256']))
    manifest = json.loads(dataset.read_text('utf-8'))
    output = root/'artifacts/week6/oehv4e_confirmatory_v1_1'
    summary_path = output/'e1_execution_summary.json'
    summary = json.loads(summary_path.read_text('utf-8'))
    if (summary.get('status') != 'complete' or summary.get('core_fits') != 108000
            or summary.get('manifest_sha256') != seal['manifest_sha256']):
        raise ValueError('E1 summary does not match the consumed input seal')
    hashes = {str(dataset.relative_to(root)):sha256_file(dataset), str(seal_path.relative_to(root)):sha256_file(seal_path),
              str(summary_path.relative_to(root)):sha256_file(summary_path)}
    for name,digest in summary['artifact_sha256'].items():
        path = verified_dataset_path(output, dict(file=name,sha256=digest))
        hashes[str(path.relative_to(root))] = digest
    feature_seal_path = output/'pilot_features/feature_seal.json'
    feature_seal = json.loads(feature_seal_path.read_text('utf-8'))
    if (feature_seal.get('scope') != 'evaluation_pilot_only_before_training_outcomes'
            or feature_seal.get('rows') != 1800 or feature_seal.get('groups') != 600):
        raise ValueError('The registered E1 pilot feature scope/count changed')
    feature_path = verified_dataset_path(output,dict(file='pilot_features/features.csv',sha256=feature_seal['features_sha256']))
    hashes[str(feature_path.relative_to(root))] = feature_seal['features_sha256']
    hashes[str(feature_seal_path.relative_to(root))] = sha256_file(feature_seal_path)
    indexed, runs, instances = verify_run_grid(manifest,read_csv_rows(output/'core_run_results.csv'))
    if len(instances) != 120 or len(runs) != 600:
        raise ValueError('The original E1 cohort is not 120 MDPs / 600 groups')
    panels = {name:tuple(sorted(mdp for mdp,item in instances.items() if item['dynamics_audit']['panel'] == name))
              for name in ('stochastic_primary','deterministic_control')}
    if len(panels['stochastic_primary']) != 84 or len(panels['deterministic_control']) != 36:
        raise ValueError('E1 dynamics panels changed')
    contract_path = root/'docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json'
    contract = json.loads(contract_path.read_text('utf-8'))
    hashes[str(contract_path.relative_to(root))] = sha256_file(contract_path)
    feature_names = {key:tuple(contract['feature_sets'][key]) for key in ('B2_OEH3','B2_OEH3_LCOEH4')}
    features = {}
    for row in read_csv_rows(feature_path):
        key = (row['mdp_id'],row['policy_family'],row['algorithm'])
        if key in features or key[:2] not in runs or key[2] not in ALGORITHMS:
            raise ValueError('Duplicate or unmatched sealed E1 feature')
        values = {name:tuple(float(row[field]) for field in fields) for name,fields in feature_names.items()}
        if not all(np.isfinite(x) for array in values.values() for x in array):
            raise ValueError('Nonfinite sealed E1 feature')
        features[key] = values
    if len(features) != 1800:
        raise ValueError('Incomplete sealed E1 feature panel')
    decisions = {}; outer = {}
    for panel,mdps in panels.items():
        for row in read_csv_rows(output/f'{panel}_selector_decisions.csv'):
            key = (row['mdp_id'],row['policy_family'])
            if key in decisions or key not in runs or key[0] not in mdps or row['selected_algorithm'] not in ALGORITHMS:
                raise ValueError('Invalid frozen OEH-v4E selection')
            selected = np.mean([x[0] for x in runs[key][row['selected_algorithm']].values()])
            if not np.isclose(selected,float(row['selected_return']),rtol=0,atol=1e-12):
                raise ValueError('OEH-v4E decision/score mismatch')
            fold = int(row['outer_fold'])
            if not 0 <= fold < 5 or (key[0] in outer and outer[key[0]] != fold):
                raise ValueError('Frozen E1 outer-fold assignment mismatch')
            outer[key[0]] = fold; decisions[key] = row['selected_algorithm']
    if set(decisions) != set(runs):
        raise ValueError('Incomplete frozen OEH-v4E decision panel')
    return dict(runs=runs,features=features,outer=outer,panels=panels,v4e_decisions=decisions,
                input_hashes=hashes,training_records=indexed,manifest_sha256=seal['manifest_sha256'],
                return_ridges=tuple(float(x) for x in contract['evaluation']['return_ridges']),
                original_cohort_reused=True,new_mdps_generated=0,posthoc_consumed_e1=True)
