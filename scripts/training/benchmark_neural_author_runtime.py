"""Benchmark author CQL/IQL update routines on a real D4RL HDF5 buffer.

This measures learner components, not MuJoCo evaluation or a paper reproduction.
Imports selected author definitions unchanged to avoid registering unavailable
simulator environments. Source and compatibility changes are recorded explicitly.
"""
import argparse
import ast
import collections
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np

from evaluation.io import sha256_file, write_json


def author_definitions(path, names, namespace):
    """Execute original AST nodes, retaining their filename and line numbers."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'Missing author definitions in {path}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['cql', 'iql'], required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--d4rl-repo', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps-per-block', type=int, default=100)
    parser.add_argument('--compatibility-shim', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps_per_block < 1:
        parser.error('steps-per-block must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = list(args.repo.rglob('*.py'))
    plan = dict(scope=__doc__, method=args.method, python=sys.version,
        repo_commit=subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
        d4rl_commit=subprocess.check_output(['git', '-C', str(args.d4rl_repo), 'rev-parse', 'HEAD'], text=True).strip(),
        data_sha256=sha256_file(args.dataset), wrapper_sha256=sha256_file(Path(__file__)),
        source_hashes={str(p.relative_to(args.repo)):sha256_file(p) for p in sources},
        batch_size=256, blocks=3, steps_per_block=args.steps_per_block, seed=42,
        device=args.device, compatibility_shim=args.compatibility_shim,
        exclusions=['MuJoCo rollouts/evaluation', 'checkpoint/log export', 'other D4RL environments',
                    'whole-paper replication'] + (['GPU runtime'] if args.device == 'cpu' else []))
    write_json(args.output_dir/'plan.json', plan)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output_dir/'failure.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc),
            note='No full-run estimate is valid from a failed author-code probe.'))
        raise


def run(args):
    import h5py
    import gym
    from tqdm import tqdm
    t0 = time.perf_counter()
    with h5py.File(args.dataset, 'r') as f:
        raw = {key:f[key][:] for key in ['observations','actions','rewards','terminals','timeouts']}
    definitions = author_definitions(args.d4rl_repo/'d4rl/__init__.py', ['qlearning_dataset'], {'np':np})
    data = definitions['qlearning_dataset'](None, dataset=raw)
    load_seconds = time.perf_counter()-t0
    np.random.seed(42)
    synchronize = lambda: None
    if args.method == 'cql':
        import torch
        sys.path.insert(0, str((args.repo/'d4rl').resolve()))
        import rlkit.torch.pytorch_util as ptu
        from rlkit.torch.networks import FlattenMlp
        from rlkit.torch.sac.policies import TanhGaussianPolicy
        from rlkit.torch.sac.cql import CQLTrainer
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but CUDA-enabled PyTorch/GPU is unavailable')
        ptu.set_gpu_mode(args.device == 'cuda')
        if args.device == 'cuda':
            synchronize = torch.cuda.synchronize
        torch.manual_seed(42)
        obs_dim, action_dim = data['observations'].shape[1], data['actions'].shape[1]
        qnets = [FlattenMlp(input_size=obs_dim+action_dim,output_size=1,hidden_sizes=[256]*3) for _ in range(4)]
        policy = TanhGaussianPolicy(obs_dim=obs_dim,action_dim=action_dim,hidden_sizes=[256]*3)
        # Only the action-space shape is read by CQLTrainer. No fake env rollouts.
        env = SimpleNamespace(action_space=gym.spaces.Box(-1,1,shape=(action_dim,)))
        learner = CQLTrainer(env=env,policy=policy,qf1=qnets[0],qf2=qnets[1],target_qf1=qnets[2],target_qf2=qnets[3],
            discount=.99,soft_target_tau=.005,policy_lr=1e-4,qf_lr=3e-4,reward_scale=1,
            use_automatic_entropy_tuning=True,policy_eval_start=40000,num_qs=2,temp=1,
            min_q_version=3,min_q_weight=5,with_lagrange=False,num_random=10,
            max_q_backup=False,deterministic_backup=True)
        for network in learner.networks:
            network.to(ptu.device)
        def update():
            indices=np.random.randint(len(data['rewards']),size=256)
            batch={k:torch.as_tensor(data[k][indices],dtype=torch.float32,device=ptu.device) for k in data}
            batch['rewards']=batch['rewards'].reshape(-1,1)
            batch['terminals']=batch['terminals'].reshape(-1,1)
            if args.compatibility_shim:
                # Preserve saved forward tensors across the author's optimizer
                # step before policy backward; do not reorder author updates.
                with torch.autograd.graph.allow_mutation_on_saved_tensors():
                    learner.train_from_torch(batch)
            else:
                learner.train_from_torch(batch)
        metadata=dict(torch=torch.__version__,threads=torch.get_num_threads(),device=str(ptu.device),
            gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            compatibility='copy-on-write saved autograd tensors' if args.compatibility_shim else None,
            configuration='Author README CQL(H): weight=5, no Lagrange, policy_lr=1e-4; early BC phase and post-40000 phase')
        phases=[('early_bc',0),('post_bc',40000)]
    else:
        import jax
        jax.config.update('jax_platforms', 'cuda' if args.device == 'cuda' else 'cpu')
        devices = jax.devices()
        if args.device == 'cuda' and not any(d.platform == 'gpu' for d in devices):
            raise RuntimeError('CUDA requested but CUDA-enabled JAX/GPU is unavailable')
        import flax
        if args.compatibility_shim:
            # Renamed tree API and legacy FrozenDict return convention only.
            jax.tree_multimap=jax.tree_util.tree_map
            flax.config.update('flax_return_frozendict', True)
            original_jit=jax.jit
            def legacy_jit(fun, *jit_args, **jit_kwargs):
                # The author decorator lists an absent 'distribution' argument.
                # Older JAX ignored it; new JAX refuses the import.
                if 'static_argnames' in jit_kwargs:
                    parameters=inspect.signature(fun).parameters
                    jit_kwargs['static_argnames']=tuple(n for n in jit_kwargs['static_argnames'] if n in parameters)
                return original_jit(fun,*jit_args,**jit_kwargs)
            jax.jit=legacy_jit
        sys.path.insert(0,str(args.repo.resolve()))
        from learner import Learner
        batch_type=collections.namedtuple('Batch',['observations','actions','rewards','masks','next_observations'])
        ds_scope={'np':np,'gym':gym,'tqdm':tqdm,'Optional':Optional,'Batch':batch_type,
                  'd4rl':SimpleNamespace(qlearning_dataset=lambda env:data)}
        author_definitions(args.repo/'dataset_utils.py',['Dataset','D4RLDataset','split_into_trajectories'],ds_scope)
        dataset=ds_scope['D4RLDataset'](None)
        author_definitions(args.repo/'train_offline.py',['normalize'],ds_scope)
        ds_scope['normalize'](dataset)
        learner=Learner(42,dataset.observations[:1],dataset.actions[:1],actor_lr=3e-4,value_lr=3e-4,
            critic_lr=3e-4,hidden_dims=(256,256),discount=.99,expectile=.7,temperature=3.,dropout_rate=None,
            tau=.005,max_steps=1_000_000)
        def update():
            info=learner.update(dataset.sample(256))
            jax.block_until_ready(info)
        metadata=dict(jax=jax.__version__,flax=flax.__version__,devices=[str(d) for d in jax.devices()],
            compatibility='tree_multimap alias; FrozenDict legacy return; ignore absent static argument names' if args.compatibility_shim else None,
            configuration='Author configs/mujoco_config.py, cosine schedule max_steps=1000000')
        phases=[('updates',0)]
    setup_seconds=time.perf_counter()-t0-load_seconds
    rows=[]
    for phase, start in phases:
        if args.method=='cql':
            learner._current_epoch=start
        synchronize()
        cold=time.perf_counter()
        update()
        synchronize()
        cold_seconds=time.perf_counter()-cold
        blocks=[]
        for block in range(3):
            synchronize()
            started=time.perf_counter()
            for _ in range(args.steps_per_block):
                update()
            synchronize()
            elapsed=time.perf_counter()-started
            blocks.append(elapsed)
            print(f'{args.method} {phase} block {block+1}: {elapsed:.3f}s',flush=True)
        row=dict(phase=phase,cold_update_seconds=cold_seconds,block_seconds=blocks,
            seconds_per_update=float(np.median(blocks))/args.steps_per_block)
        rows.append(row)
        write_json(args.output_dir/'partial.json',rows)
    write_json(args.output_dir/'summary.json',dict(status='complete',load_seconds=load_seconds,
        setup_seconds=setup_seconds,metadata=metadata,phases=rows,
        note='Update-component timings on the recorded device only. Do not infer complete pipeline, other environments or other hardware timings.'))


if __name__=='__main__':
    main()
