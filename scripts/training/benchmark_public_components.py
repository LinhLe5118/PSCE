"""Time unmodified author routines; this is NOT a full paper reproduction.

OfflineRL's return/HLL routines run on our real pilots. Effective Horizon's
finite-horizon value iteration runs on the author's public test MDPs. Neither
timing is advertised as neural CQL/IQL or as the complete EH bound algorithm.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from evaluation.io import sha256_file, verified_dataset_path, write_json


def resolve_git_link(path, root):
    """Resolve Git symlinks checked out as text on Windows, inside the clone."""
    path, root = path.resolve(), root.resolve()
    for _ in range(8):
        if not path.is_relative_to(root):
            raise ValueError('Public data link leaves its repository')
        with path.open('rb') as f:
            header = f.read(4)
        if header[:2] == b'PK':
            return path
        if path.stat().st_size > 1024:
            raise ValueError(f'Not an NPZ or Git text link: {path}')
        path = (path.parent / path.read_text().strip()).resolve()
    raise ValueError('Git data link cycle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--effective-horizon-repo', type=Path, required=True)
    parser.add_argument('--offlinerl-repo', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    commits = {name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
               for name, path in [('effective_horizon', args.effective_horizon_repo), ('offlinerl', args.offlinerl_repo)]}
    source = args.effective_horizon_repo / 'effective_horizon/mdp_utils.py'
    write_json(args.output_dir / 'plan.json', dict(scope=__doc__, commits=commits,
        source_sha256=sha256_file(source), wrapper_sha256=sha256_file(Path(__file__)),
        manifest_sha256=sha256_file(args.dataset_dir / 'manifest.json'),
        python=sys.version, platform=platform.platform(), repetitions=3,
        compatibility='Original source; current Python/NumPy/PyTorch, not the archived paper environment',
        exclusions=['EH Julia bound computation', 'SQIRL/GORP training', 'neural CQL/IQL',
                    'TQ/reference normalization', 'public OfflineRL buffer generation', 'BC entropy training']))
    spec = importlib.util.spec_from_file_location('author_mdp_utils', source)
    mdp_utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mdp_utils
    spec.loader.exec_module(mdp_utils)
    rows = []
    for name, horizon in [('MiniGrid-KeyCorridorS3R1-v0', 100), ('freeway_10_fs30', 10), ('maze_easy_l0_30_fs1', 30)]:
        path = resolve_git_link(args.effective_horizon_repo / 'data/mdps' / name / 'consolidated.npz', args.effective_horizon_repo)
        t0 = time.perf_counter()
        transitions, rewards = mdp_utils.load_mdp(str(path))
        sparse, reward_vector = mdp_utils.get_sparse_mdp(transitions, rewards)
        load_seconds = time.perf_counter() - t0
        states, actions = transitions.shape
        # Lists plus returned stacks coexist during conversion; bound memory.
        estimated_bytes = 2 * 3 * horizon * states * (actions + 1) * 4
        if estimated_bytes > 1024**3:
            row = dict(method='author_value_iteration', dataset=name, status='memory_guard',
                       estimated_array_bytes=estimated_bytes)
        else:
            timings = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = mdp_utils.run_value_iteration(sparse, reward_vector, horizon)
                timings.append(time.perf_counter() - t0)
                assert np.isfinite(result.optimal_values).all()
                del result
                gc.collect()
            row = dict(method='author_value_iteration', dataset=name, status='measured',
                states=states, actions=actions, horizon=horizon, load_seconds=load_seconds,
                seconds=timings, median_seconds=float(np.median(timings)), data_sha256=sha256_file(path))
        rows.append(row)
        write_json(args.output_dir / 'partial.json', rows)
    sys.path.insert(0, str(args.offlinerl_repo.resolve()))
    from source.offline_ds_evaluation.evaluator import Evaluator
    import torch
    manifest = json.loads((args.dataset_dir / 'manifest.json').read_text(encoding='utf-8'))
    if any(r.get('benchmark_split') == 'evaluation' for r in manifest['datasets']):
        raise ValueError('Refusing sealed evaluation data')
    records = [r for r in manifest['datasets'] if r['stream'] == 'pilot' and r['size_multiplier'] == 10]
    for index, record in enumerate(records):
        t0 = time.perf_counter()
        with np.load(verified_dataset_path(args.dataset_dir, record), allow_pickle=False) as f:
            evaluator = Evaluator(record['instance_id'], record['policy_family'],
                f['state'].reshape(-1, 1).astype(float), f['action'], f['reward'], f['done'],
                num_actions=record['num_actions'])
        setup_seconds = time.perf_counter() - t0
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            returns = evaluator.get_returns()
            unique = evaluator.get_unique_state_actions()
            timings.append(time.perf_counter() - t0)
        rows.append(dict(method='author_return_and_hll_components', dataset=record['dataset_id'],
            status='measured', transitions=record['size'], setup_seconds=setup_seconds,
            seconds=timings, median_seconds=float(np.median(timings)), complete_episodes=len(returns),
            hll_count=unique,
            caveat='Original 100-bin discretization of scalar state IDs is a runtime probe, not validated tabular SACo. No normalized TQ/SACo scores.'))
        with (args.output_dir / 'measurements.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(rows[-1]) + '\n')
        print(f'Pilot metrics {index + 1}/{len(records)}', flush=True)
    write_json(args.output_dir / 'summary.json', dict(status='complete', rows=rows,
        torch_version=torch.__version__, cuda=torch.cuda.is_available(),
        pilot_component_median_sum_seconds=sum(r['median_seconds'] for r in rows if r['method']=='author_return_and_hll_components'),
        note=__doc__))


if __name__ == '__main__':
    main()
