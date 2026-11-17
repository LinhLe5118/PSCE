"""Single-use post-hoc candidate gate against CFQI, frozen v4E and R34.

Only original E1 outcomes/features are reused. All procedures are refitted
under the same whole-MDP draw. Failed prototypes are withdrawn, retaining
only a minimal audit ledger. An execution error never triggers a reroll.
"""
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from evaluation.e1_reuse import load_e1_panel
from evaluation.e1_routing import score_routes
from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json
from evaluation.withdrawal import withdraw_attempt
from scripts.analysis.run_oeh_e1_reuse import paired_draws

REFERENCE = 'artifacts/week7/oeh_v5_e1_reuse_r34_2026-09-29'
CONTRASTS = ('NR_VS_CFQI', 'SR_VS_CFQI', 'NR_VS_V4E', 'SR_VS_V4E', 'NR_VS_R34', 'SR_VS_R34')
WORKER = None


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for name, digest in hashes.items():
        path = (root/name.replace('\\', '/')).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError(f'Sealed input/source checksum mismatch: {name}')


def contrast_values(means):
    means = np.asarray(means, float)
    if means.shape != (8,) or not np.isfinite(means).all():
        raise ValueError('Expected four paired methods / two endpoints')
    return np.concatenate([means[:2]-means[i:i+2] for i in (2, 4, 6)])


def compare_routes(panel, choices, v4e, r34, mdps, multiplicities=None, *, scorer=score_routes):
    first, _ = scorer(dict(panel, v4e_decisions=v4e), choices, mdps, multiplicities=multiplicities)
    second, _ = scorer(dict(panel, v4e_decisions=r34), choices, mdps, multiplicities=multiplicities)
    if not np.array_equal(first[:4], second[:4]):
        raise ValueError('Unpaired candidate/reference scores')
    return np.concatenate((first, second[4:6]))


def initialize_worker(panel, mdps, module):
    global WORKER
    WORKER = panel, mdps, importlib.import_module(module), importlib.import_module('oeh_interaction_gains')


def bootstrap_one(task):
    index, multiplicities = task
    panel, mdps, candidate, reference = WORKER
    choices, _, _ = candidate.fit_routes(panel, mdps, multiplicities=multiplicities)
    r34, _, _ = reference.fit_routes(panel, mdps, multiplicities=multiplicities)
    v4e, _, _ = reference.fit_routes(panel, mdps, multiplicities=multiplicities, original_v4e=True)
    return index, contrast_values(compare_routes(panel, choices, v4e, r34, mdps, multiplicities,
        scorer=getattr(candidate, 'score_routes', score_routes)))


def execute_draws(pool, tasks, total, workers, progress):
    """Bound pending work so a failed draw cancels the unstarted queue."""
    iterator = iter(tasks); pending = {}; completed = 0
    result = np.full((total, 6), np.nan)
    def submit_next():
        task = next(iterator, None)
        if task is not None:
            pending[pool.submit(bootstrap_one, task)] = task[0]
    for _ in range(min(total, 2*workers)):
        submit_next()
    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                expected = pending.pop(future); index, values = future.result()
                values = np.asarray(values, float)
                if (index != expected or not 0 <= index < total or values.shape != (6,)
                        or not np.isfinite(values).all() or np.isfinite(result[index]).any()):
                    raise ValueError('Invalid or duplicate bootstrap result')
                result[index] = values; completed += 1
                progress(completed)
                submit_next()
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    if completed != total or not np.isfinite(result).all():
        raise ValueError('Incomplete bootstrap grid')
    return result


def run(root, protocol, workers):
    root = Path(root).resolve(); protocol = Path(protocol).resolve()
    if not protocol.is_relative_to(root/'docs/protocol') or workers < 1:
        raise ValueError('Invalid protocol path or worker count')
    config = json.loads(protocol.read_text('utf-8')); attempt = config['attempt_id']
    if (not re.fullmatch(r'r[1-9][0-9]*', attempt) or config.get('evaluation_reuse') is not True
            or config.get('budget') != 10 or config.get('new_mdps_generated') != 0
            or config.get('bootstrap_draws') != 2000 or config.get('bootstrap_seed') != 20260921
            or config.get('one_sided_alpha') != .05 or config.get('contrasts') != list(CONTRASTS)
            or config.get('reference') != REFERENCE or config.get('confirmation') is not False):
        raise ValueError('Original E1 / six-contrast registered post-hoc gate required')
    ledger = root/'docs/reassessment/withdrawal_ledger'/f'{attempt}.json'
    if ledger.exists():
        raise ValueError('Consumed attempt; no rerun')
    sources = list(config['sources'])+[protocol.relative_to(root).as_posix()]
    if any(not (root/name).resolve().is_relative_to(root) for name in sources):
        raise ValueError('Source escapes repository')
    hashes = {name:sha256_file(root/name) for name in sources}
    bundle = root/'tmp/oeh_e1_reuse'/attempt; reserve_output_dir(bundle)
    started = time.perf_counter(); report = None; inputs = {}; reasons = ['EXECUTION_FAILURE']; completed = 0
    def log(message):
        with (bundle/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(str(message)+'\n')
        print(message, flush=True)
    try:
        log(f'{attempt}: source/config sealed; original E1 and R34 inputs only')
        reference_path = root/REFERENCE; manifest_path = reference_path/'manifest.json'
        manifest = json.loads(manifest_path.read_text('utf-8'))
        for section in ('scientific_source_hashes', 'original_input_hashes'):
            inputs.update({name.replace('\\', '/'):digest for name, digest in manifest[section].items()})
        inputs.update({f'{REFERENCE}/{name}':digest for name, digest in manifest['artifact_sha256'].items()})
        inputs[f'{REFERENCE}/manifest.json'] = sha256_file(manifest_path)
        verify_hashes(root, inputs)
        loaded = load_e1_panel(root)
        inputs.update({name.replace('\\', '/'):digest for name, digest in loaded['input_hashes'].items()})
        panel = {key:loaded[key] for key in ('runs', 'features', 'outer', 'v4e_decisions', 'return_ridges')}
        candidate = importlib.import_module(config['candidate_module'])
        if hasattr(candidate, 'prepare_panel'):
            panel = candidate.prepare_panel(panel, root, loaded)
        environment = dict(python=sys.version, packages={name:importlib.metadata.version(name)
            for name in ('numpy', 'scipy', *config.get('extra_packages', []))})
        write_json(bundle/'plan.json', dict(config=config, source_hashes=hashes, input_hashes=inputs,
            manifest_sha256=loaded['manifest_sha256'], workers=workers, environment=environment,
            confirmation=False, posthoc_consumed_e1=True, new_mdps_generated=0, original_cohort_reused=True))
        expected_r34 = {(row['mdp_id'], row['policy_family']):row['selected_algorithm']
                        for row in read_csv_rows(reference_path/'raw/decisions.csv')}
        reference_report = json.loads((reference_path/'raw/report.json').read_text('utf-8'))
        reference = importlib.import_module('oeh_interaction_gains')
        decisions, scores = [], {}
        for name, mdps in loaded['panels'].items():
            r34, _, reference_models = reference.fit_routes(panel, mdps)
            v4e, _, _ = reference.fit_routes(panel, mdps, original_v4e=True)
            expected = {g for g in panel['runs'] if g[0] in mdps}
            if (set(r34) != expected or set(v4e) != expected
                    or any(r34[g] != expected_r34[g] or v4e[g] != panel['v4e_decisions'][g] for g in expected)):
                raise ValueError('Frozen reference route reproduction failed')
            reproduced, _ = score_routes(dict(panel, v4e_decisions=v4e), r34, mdps)
            if not np.allclose(reproduced, reference_report['panels'][name], atol=1e-12, rtol=0):
                raise ValueError('Frozen R34 reference scores mismatch')
            choices, forecasts, models = candidate.fit_routes(panel, mdps)
            if set(choices) != expected or set(forecasts) != expected:
                raise ValueError('Incomplete candidate routes/forecasts')
            rows = [dict(mdp_id=g[0], policy_family=g[1], panel=name, outer_fold=panel['outer'][g[0]],
                selected_algorithm=method, reference_algorithm=r34[g],
                predicted_paired_return=float(forecasts[g][method][0]),
                predicted_paired_success=float(forecasts[g][method][1])) for g, method in sorted(choices.items())]
            choice_file = bundle/f'{name}_choices.json'
            write_json(choice_file, rows)
            write_json(bundle/f'{name}_choice_seal.json', dict(sha256=sha256_file(choice_file)))
            write_json(bundle/f'{name}_training.json', dict(reference_models=reference_models, candidate_models=models))
            # Actual candidate outer-test scores are read only after sealing.
            scores[name] = compare_routes(panel, choices, v4e, r34, mdps,
                scorer=getattr(candidate, 'score_routes', score_routes))
            decisions.extend(rows)
            log(f'{name}: choices sealed; changed {sum(choices[g]!=r34[g] for g in expected)}/{len(expected)} groups')
        write_csv_rows(bundle/'decisions.csv', decisions, list(decisions[0]))
        observed = contrast_values(scores['stochastic_primary'])
        log('POINT_EFFECTS '+json.dumps(dict(zip(CONTRASTS, observed.tolist()))))
        reasons = [f'NONPOSITIVE_{name}' for name, value in zip(CONTRASTS, observed) if value <= 0]
        lower = None
        if not reasons:
            mdps = loaded['panels']['stochastic_primary']
            log('Point gate passed; 2000 whole-MDP refits of candidate, R34 and original v4E')
            def progress(count):
                nonlocal completed
                completed = count
                if count % 10 == 0:
                    log(f'BOOTSTRAP {count}/2000; no reroll/filtering')
                    write_json(bundle/'progress.json', dict(completed=count, total=2000, runtime_seconds=time.perf_counter()-started))
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                     initargs=(panel, mdps, config['candidate_module'])) as pool:
                bootstrap = execute_draws(pool, paired_draws(mdps, panel['outer'], 2000, 20260921), 2000, workers, progress)
            np.savez_compressed(bundle/'bootstrap.npz', draws=bootstrap, observed=observed)
            lower = 2*observed-np.quantile(bootstrap, .95, axis=0)
            log('BASIC_LOWER '+json.dumps(dict(zip(CONTRASTS, lower.tolist()))))
            reasons = [f'UNCERTAIN_{name}' for name, value in zip(CONTRASTS, lower) if value <= 0]
        verify_hashes(root, hashes); verify_hashes(root, inputs)
        report = dict(screen_pass=not reasons, reason_codes=reasons, contrast_order=CONTRASTS,
            paired_effects=observed.tolist(), basic_one_sided_lower=None if lower is None else lower.tolist(),
            panels={name:value.tolist() for name, value in scores.items()}, bootstrap_completed=completed,
            panel_order=['candidate_NR', 'candidate_SR', 'CFQI_NR', 'CFQI_SR', 'v4E_NR', 'v4E_SR', 'R34_NR', 'R34_SR'],
            normalized_return_unclipped=True, success_threshold=.95, new_mdps_generated=0,
            confirmation=False, posthoc_consumed_e1=True, runtime_seconds=time.perf_counter()-started)
        write_json(bundle/'report.json', report); log(f'{attempt}: screen={not reasons}; reasons={reasons}')
        write_json(bundle/'artifact_manifest.json', dict(source_hashes=hashes, input_hashes=inputs,
            artifact_sha256={path.name:sha256_file(path) for path in sorted(bundle.iterdir()) if path.is_file()},
            runtime_seconds=report['runtime_seconds']))
    except Exception as exc:
        reasons = ['EXECUTION_FAILURE']
        write_json(bundle/'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        write_json(ledger, dict(attempt_id=attempt, status='screen_pass' if report and report['screen_pass'] else 'withdrawn',
            evaluation_reuse=True, consumed_seeds=[], new_mdps_generated=0, source_hashes=hashes,
            input_hashes=inputs, reason_codes=reasons, posthoc_consumed_e1=True, bootstrap_completed=completed,
            runtime_seconds=time.perf_counter()-started))
        if not report or not report['screen_pass']:
            print(json.dumps(withdraw_attempt(root, attempt), sort_keys=True), flush=True)
    return report
