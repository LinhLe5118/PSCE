"""Final OEH library router. Scientific function bodies are unchanged."""
from collections import defaultdict
import numpy as np
from .reference import FEATURES, METHODS as CORE, grouped_split, original_routes
from . import core as reference

def attach_results(panel, rows, training_records, additional):
    additional = tuple(additional)
    if not additional or len(set(additional)) != len(additional) or set(additional)&set(CORE):
        raise ValueError('Invalid additional registered methods')
    runs = {g:dict(methods) for g, methods in panel['runs'].items()}
    expected = {(g, method, seed) for g in runs for method in additional for seed in range(10)}
    seen = set()
    for row in rows:
        g = row['mdp_id'], row['policy_family']; method = row['algorithm']; seed = int(row['training_seed'])
        identity = g, method, seed; budget = int(row['training_size_multiplier'])
        if identity not in expected or identity in seen or budget != 10:
            raise ValueError('Duplicate/unregistered additional E1 result')
        value = float(row['normalized_return']); success = row['success']
        if (row['run_status'] != 'observed' or not np.isfinite(value)
                or success not in ('True', 'False') or (success == 'True') != (value >= .95)):
            raise ValueError('Failed/nonfinite additional E1 result or inconsistent success')
        record = training_records[(*g, 10, seed)]
        if row['buffer_sha256'] != record['sha256']:
            raise ValueError('Additional learner/training buffer hash mismatch')
        seen.add(identity); runs[g].setdefault(method, {})[seed] = value, int(success == 'True')
    if seen != expected:
        raise ValueError('Incomplete additional learner/group/seed grid')
    return dict(panel, runs=runs, candidate_methods=CORE+additional)

def score_routes(panel, choices, mdps, *, multiplicities=None):
    weights, expected = grouped_split(panel, mdps, multiplicities)
    methods = panel['candidate_methods']
    if set(choices) != set(expected) or any(m not in methods for m in choices.values()):
        raise ValueError('Incomplete or unregistered library routing')
    clusters = defaultdict(list)
    for g in sorted(choices):
        selected = choices[g], CORE[0], panel['v4e_decisions'][g]
        clusters[g[0]].append(np.concatenate([np.mean(list(panel['runs'][g][m].values()), axis=0) for m in selected]))
    values = {mdp:np.mean(rows, axis=0) for mdp, rows in clusters.items()}
    active = [mdp for mdp in mdps if weights[mdp] > 0]
    if set(active) != set(values):
        raise ValueError('Incomplete scored MDP library panel')
    return np.average([values[m] for m in active], axis=0, weights=[weights[m] for m in active]), values

def fit_routes(panel, mdps, *, multiplicities=None, original_v4e=False):
    if original_v4e:
        return original_routes(panel, mdps, multiplicities)
    methods = panel['candidate_methods']
    weights, groups = grouped_split(panel, mdps, multiplicities)
    _, forecasts, tuning = reference.fit_routes(panel, mdps, multiplicities=multiplicities)
    for fold in range(5):
        train = [g for g in groups if panel['outer'][g[0]] != fold]
        test = [g for g in groups if panel['outer'][g[0]] == fold]
        if not test: continue
        train_weights = np.array([weights[g[0]] for g in train], dtype=int)
        for method in methods[len(CORE):]:
            targets = np.array([reference.paired_targets(panel['runs'][g][CORE[0]], panel['runs'][g][method]) for g in train])
            blocks = []
            for feature in FEATURES:
                def row(g):
                    return tuple(value for anchor in CORE for value in panel['features'][(*g, anchor)][feature])
                x, query = np.array([row(g) for g in train]), np.array([row(g) for g in test])
                values, model = reference.forecast_block(x, query, targets, train_weights, [g[0] for g in train],
                    panel['return_ridges'], f'paired-quadratic|{fold}|{method}|{feature}')
                blocks.append(values); tuning.append(dict(fold=fold, method=method, feature=feature,
                    input='all_registered_core_pilot_diagnostics', **model))
            for g, values in zip(test, np.mean(blocks, axis=0)):
                forecasts[g][method] = values.tolist()
    choices = {}
    for fold in range(5):
        test = [g for g in groups if panel['outer'][g[0]] == fold]
        if not test: continue
        values = np.array([[forecasts[g][m] for m in methods] for g in test])
        route, evidence = reference.joint_routes(values[:, :, 0], values[:, :, 1], [weights[g[0]] for g in test])
        tuning.append(dict(fold=fold, library=methods, **evidence))
        choices.update({g:methods[index] for g, index in zip(test, route)})
    return choices, forecasts, tuning
