"""Compare complete adaptive pilot outputs against reference, without tuning budgets."""
import argparse
from dataclasses import fields, is_dataclass
import json
import os
import platform
import sys
from pathlib import Path
import time

import numpy as np
from scripts.training import benchmark_pipeline_runtime as pipeline
from evaluation.io import verified_dataset_path, write_json, sha256_file


def compare(actual, expected):
    """Discrete verdicts must match exactly; floating outputs tolerate rounding."""
    if is_dataclass(expected):
        for field in fields(expected):
            compare(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('Assessed prefixes differ')
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError('Adaptive trace lengths differ')
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(expected, np.ndarray):
        if np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True)
    else:
        if actual != expected:
            raise AssertionError((actual, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--states', type=int, nargs='+', default=[32, 512])
    parser.add_argument('--backend', choices=['numpy', 'cuda', 'hybrid'], default='hybrid')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol_config'] != pipeline.PROTOCOL_CONFIG:
        raise ValueError('Dataset differs from active protocol')
    candidates = sorted(pipeline.pilot_records(manifest), key=lambda r: r['dataset_id'])
    selected = [next(r for r in candidates if r['num_states'] == states
                     and r['policy_family'] == 'q_star_softmax') for states in args.states]
    write_json(args.output_dir/'plan.json', dict(
        manifest_sha256=sha256_file(args.dataset_dir/'manifest.json'),
        profile_sha256=sha256_file(pipeline.PROTOCOL_CONFIG_PATH),
        acceleration_sha256=sha256_file(Path(__file__).resolve().parents[2]/'code/tabular_acceleration.py'),
        source_sha256=sha256_file(Path(__file__)), backend=args.backend,
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        dataset_ids=[r['dataset_id'] for r in selected],
        config=pipeline.PROTOCOL_CONFIG['estimator'], bootstrap_seed=20260824))
    rows = []
    previous = os.environ.get('DAORL_OEH_BACKEND')
    try:
        for record in selected:
            with np.load(verified_dataset_path(args.dataset_dir, record), allow_pickle=False) as f:
                arrays = {k: f[k] for k in f.files}
            results, seconds = {}, {}
            for backend in ['reference', args.backend]:
                os.environ['DAORL_OEH_BACKEND'] = backend
                start = time.perf_counter()
                results[backend] = pipeline.run_pilot(arrays, record['initial_state_distribution'],
                    num_states=record['num_states'], num_actions=record['num_actions'],
                    bootstrap_seed=20260824)
                seconds[backend] = time.perf_counter()-start
                print(record['dataset_id'], backend, seconds[backend], flush=True)
            # Includes every assessed prefix and the complete adaptive decision trace.
            compare(results[args.backend].results, results['reference'].results)
            compare(results[args.backend].selection, results['reference'].selection)
            rows.append(dict(dataset_id=record['dataset_id'], states=record['num_states'],
                seconds=seconds, equivalence_passed=True,
                attempts=len(results['reference'].selection.attempts)))
            write_json(args.output_dir/'measurements.json', rows)
        write_json(args.output_dir/'summary.json', dict(status='complete', rows=rows,
            note='Full registered bootstrap/adaptive work for selected pilots only. Does not validate all development groups or measure A100.'))
    except Exception as exc:
        write_json(args.output_dir/'failure.json', dict(status='failed',
            error_type=type(exc).__name__, error=str(exc), completed_cases=len(rows),
            note='Failed comparisons do not establish scientific equivalence.'))
        raise
    finally:
        if previous is None:
            os.environ.pop('DAORL_OEH_BACKEND', None)
        else:
            os.environ['DAORL_OEH_BACKEND'] = previous


if __name__ == '__main__':
    main()
