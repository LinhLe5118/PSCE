#!/usr/bin/env python
"""Run compatible neural discrete offline-RL baselines on development data.

BCQ imports the authors' unmodified discrete implementation. Discrete CQL is
provided by the pinned d3rlpy checkout, whose implementation is documented and
benchmarked in the accompanying JMLR library paper. The adapter changes only
data representation, batching, and experiment orchestration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from evaluation.io import (  # noqa: E402
    reserve_output_dir,
    sha256_file,
    verified_dataset_path,
    write_csv_rows,
    write_json,
)
from evaluation.training_protocol import verify_training_buffers  # noqa: E402
from scripts.training.run_empirical_requirement import (  # noqa: E402
    _score_policy,
    _training_seed_index,
)


BCQ_COMMIT = "4876f7e5afa9eb2981feec5daf67202514477518"
D3RLPY_COMMIT = "38c34b6f99fde82875e2ad81d1ba1f7a7da34137"
ALGORITHMS = ("discrete_bcq_author", "discrete_cql_d3rlpy")
FIELDS = (
    "mdp_id", "policy_family", "algorithm", "training_size_multiplier",
    "training_seed", "neural_seed", "normalized_return", "success",
    "run_status", "fit_seconds", "steps", "device", "peak_vram_mib",
    "external_commit",
)
SUMMARY_FIELDS = (
    "algorithm", "training_size_multiplier", "valid_runs", "invalid_runs",
    "mean_normalized_return", "median_normalized_return", "success_rate",
    "mean_fit_seconds", "max_peak_vram_mib",
)


def _git_commit_clean(path: Path, expected: str) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != expected:
        raise ValueError(f"external commit mismatch for {path}: {commit} != {expected}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError(f"tracked external source is dirty: {path}")
    return commit


def _one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= size):
        raise ValueError("state identifiers are outside the declared state space")
    return np.eye(size, dtype=np.float32)[values]


def _load_arrays(dataset_dir: Path, record: dict, cache: dict) -> dict[str, np.ndarray]:
    file_name = str(record["file"])
    if file_name not in cache:
        path = verified_dataset_path(dataset_dir, record)
        with np.load(path, allow_pickle=False) as loaded:
            cache[file_name] = {name: loaded[name] for name in loaded.files}
    size = int(record["size"])
    return {name: values[:size] for name, values in cache[file_name].items()}


class _ResidentReplay:
    """Sampling-compatible BCQ replay with the full small buffer on CUDA."""

    def __init__(self, arrays: dict[str, np.ndarray], num_states: int, device: torch.device):
        self.state = torch.as_tensor(_one_hot(arrays["state"], num_states), device=device)
        self.action = torch.as_tensor(arrays["action"], dtype=torch.long, device=device).reshape(-1, 1)
        self.next_state = torch.as_tensor(_one_hot(arrays["next_state"], num_states), device=device)
        self.reward = torch.as_tensor(arrays["reward"], dtype=torch.float32, device=device).reshape(-1, 1)
        self.not_done = 1.0 - torch.as_tensor(
            arrays["terminal"], dtype=torch.float32, device=device
        ).reshape(-1, 1)
        self.batch_size = 64

    def sample(self):
        index = torch.randint(self.state.shape[0], (self.batch_size,), device=self.state.device)
        return (
            self.state[index], self.action[index], self.next_state[index],
            self.reward[index], self.not_done[index],
        )


def _deterministic_policy(actions: np.ndarray, num_actions: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    policy = np.zeros((actions.size, num_actions), dtype=np.float64)
    policy[np.arange(actions.size), actions] = 1.0
    return policy


def _fit_bcq(
    arrays: dict[str, np.ndarray], num_states: int, num_actions: int,
    gamma: float, steps: int, seed: int, device: torch.device, bcq_dir: Path,
) -> tuple[np.ndarray, float, float]:
    module_dir = bcq_dir / "discrete_BCQ"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    author_module = importlib.import_module("discrete_BCQ")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    replay = _ResidentReplay(arrays, num_states, device)
    learner = author_module.discrete_BCQ(
        False, num_actions, num_states, device,
        BCQ_threshold=0.3, discount=gamma,
        optimizer="Adam", optimizer_parameters={"lr": 3e-4},
        polyak_target_update=True, target_update_frequency=1,
        tau=0.005, initial_eps=0.1, end_eps=0.1,
        eps_decay_period=1, eval_eps=0.0,
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        learner.train(replay)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        states = torch.eye(num_states, dtype=torch.float32, device=device)
        q_values, log_imitation, _ = learner.Q(states)
        ratios = log_imitation.exp()
        allowed = (ratios / ratios.max(dim=1, keepdim=True).values > learner.threshold).float()
        actions = (allowed * q_values + (1.0 - allowed) * -1e8).argmax(dim=1)
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    return _deterministic_policy(actions.cpu().numpy(), num_actions), elapsed, peak


def _d3rlpy_dataset(arrays: dict[str, np.ndarray], num_states: int, num_actions: int):
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer

    states = _one_hot(arrays["state"], num_states)
    next_states = _one_hot(arrays["next_state"], num_states)
    episodes = []
    for state, action, reward, next_state, terminal in zip(
        states, arrays["action"], arrays["reward"], next_states, arrays["terminal"]
    ):
        if terminal:
            observations = state[None, :]
            actions = np.asarray([[action]], dtype=np.int64)
            rewards = np.asarray([[reward]], dtype=np.float32)
            terminated = True
        else:
            observations = np.stack((state, next_state)).astype(np.float32, copy=False)
            actions = np.asarray([[action], [0]], dtype=np.int64)
            rewards = np.asarray([[reward], [0.0]], dtype=np.float32)
            terminated = False
        episodes.append(Episode(observations, actions, rewards, terminated))
    return ReplayBuffer(
        InfiniteBuffer(), episodes=episodes, action_space=ActionSpace.DISCRETE,
        action_size=num_actions,
    )


def _fit_cql(
    arrays: dict[str, np.ndarray], num_states: int, num_actions: int,
    gamma: float, steps: int, seed: int, device: torch.device,
) -> tuple[np.ndarray, float, float]:
    import d3rlpy
    from d3rlpy.algos import DiscreteCQLConfig
    from d3rlpy.logging import NoopAdapterFactory
    from d3rlpy.models.encoders import VectorEncoderFactory

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    d3rlpy.seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    dataset = _d3rlpy_dataset(arrays, num_states, num_actions)
    learner = DiscreteCQLConfig(
        learning_rate=6.25e-5,
        encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
        batch_size=64,
        gamma=gamma,
        target_update_interval=50,
        alpha=1.0,
    ).create(device="cuda:0")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    learner.fit(
        dataset, n_steps=steps, n_steps_per_epoch=steps,
        logger_adapter=NoopAdapterFactory(), show_progress=False,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    actions = learner.predict(np.eye(num_states, dtype=np.float32))
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    return _deterministic_policy(actions, num_actions), elapsed, peak


def _summaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["training_size_multiplier"])].append(row)
    output = []
    for (algorithm, budget), group in sorted(grouped.items()):
        valid = [row for row in group if row["run_status"] == "observed"]
        scores = [float(row["normalized_return"]) for row in valid]
        output.append({
            "algorithm": algorithm, "training_size_multiplier": budget,
            "valid_runs": len(valid), "invalid_runs": len(group) - len(valid),
            "mean_normalized_return": float(np.mean(scores)) if scores else math.nan,
            "median_normalized_return": float(np.median(scores)) if scores else math.nan,
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in group])),
            "max_peak_vram_mib": float(max(row["peak_vram_mib"] for row in group)),
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/week1_development_validation")
    parser.add_argument("--bcq-dir", type=Path, default=ROOT / "tmp/external_bcq")
    parser.add_argument("--d3rlpy-dir", type=Path, default=ROOT / "tmp/external_d3rlpy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args(argv)
    global torch
    try:
        import torch as torch_module
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Neural baselines require PyTorch; install requirements-gpu.txt."
        ) from error
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this GPU baseline protocol")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    selected_seeds = tuple(sorted(set(args.training_seeds)))
    registered_seeds = tuple(PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    if not selected_seeds or not set(selected_seeds).issubset(registered_seeds):
        raise ValueError("training seeds must be registered seed indices")
    registered_budgets = tuple(PROTOCOL_CONFIG["collection"]["size_multipliers"])
    if args.budget not in registered_budgets:
        raise ValueError("budget must be a registered size multiplier")

    commits = {}
    if "discrete_bcq_author" in args.algorithms:
        commits["discrete_bcq_author"] = _git_commit_clean(args.bcq_dir, BCQ_COMMIT)
    if "discrete_cql_d3rlpy" in args.algorithms:
        commits["discrete_cql_d3rlpy"] = _git_commit_clean(args.d3rlpy_dir, D3RLPY_COMMIT)
        sys.path.insert(0, str(args.d3rlpy_dir.resolve()))
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = verify_training_buffers(args.dataset_dir, manifest)
    if manifest.get("run_mode") != "review":
        raise ValueError("development review bundle required")
    records = [
        row for row in manifest["datasets"]
        if row["stream"] == "training" and int(row["size_multiplier"]) == args.budget
        and _training_seed_index(row) in selected_seeds
    ]
    if any(row["benchmark_split"] != "development" for row in records):
        raise ValueError("evaluation record refused")
    records.sort(key=lambda row: (row["instance_id"], row["policy_family"], _training_seed_index(row)))
    groups = sorted({(row["instance_id"], row["policy_family"]) for row in records})
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
        allowed = set(groups)
        records = [row for row in records if (row["instance_id"], row["policy_family"]) in allowed]
    instances = {row["instance_id"]: row for row in manifest["instances"]}
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    plan = {
        "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "sealed_evaluation_accessed": False,
        "algorithms": args.algorithms, "budget": args.budget,
        "training_seed_indices": selected_seeds, "groups": len(groups),
        "fits": len(records) * len(args.algorithms), "gradient_steps_per_fit": args.steps,
        "representation": "one-hot state; exact recorded next state",
        "neural_seed": "20260919 + 1000 * training_seed_index + development instance ordinal",
        "hyperparameters": {
            "common": {"network": [256, 256], "batch_size": 64, "gamma": PROTOCOL_CONFIG["benchmark"]["gamma"]},
            "discrete_bcq_author": {"threshold": 0.3, "learning_rate": 3e-4, "polyak_tau": 0.005},
            "discrete_cql_d3rlpy": {"alpha": 1.0, "learning_rate": 6.25e-5, "target_update_interval": 50},
        },
        "external_repositories": {
            "discrete_bcq_author": "https://github.com/sfujim/BCQ",
            "discrete_cql_d3rlpy": "https://github.com/takuseno/d3rlpy",
        },
        "external_commits": commits,
        "external_source_sha256": {
            "bcq_algorithm": sha256_file(args.bcq_dir / "discrete_BCQ/discrete_BCQ.py") if "discrete_bcq_author" in args.algorithms else None,
            "d3rlpy_cql": sha256_file(args.d3rlpy_dir / "d3rlpy/algos/qlearning/cql.py") if "discrete_cql_d3rlpy" in args.algorithms else None,
        },
        "adapter_sha256": sha256_file(Path(__file__)),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "verified_training_buffers": verified,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    write_json(args.output_dir / "plan.json", plan)
    threshold = float(PROTOCOL_CONFIG["learning"]["success_normalized_return"])
    cache = {}
    output = []
    total = len(records) * len(args.algorithms)
    completed = 0
    for record in records:
        instance = instances[record["instance_id"]]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        arrays = _load_arrays(args.dataset_dir, record, cache)
        training_seed = _training_seed_index(record)
        neural_seed = 20260919 + 1000 * training_seed + int(instance["ordinal"])
        for algorithm in args.algorithms:
            if algorithm == "discrete_bcq_author":
                policy, fit_seconds, peak = _fit_bcq(
                    arrays, mdp.num_states, mdp.num_actions, mdp.gamma,
                    args.steps, neural_seed, device, args.bcq_dir,
                )
            else:
                policy, fit_seconds, peak = _fit_cql(
                    arrays, mdp.num_states, mdp.num_actions, mdp.gamma,
                    args.steps, neural_seed, device,
                )
            normalized_return, status = _score_policy(
                mdp, policy, random_return=float(instance["random_return"]),
                dynamic_range=float(instance["dynamic_range"]),
            )
            output.append({
                "mdp_id": record["instance_id"], "policy_family": record["policy_family"],
                "algorithm": algorithm, "training_size_multiplier": args.budget,
                "training_seed": training_seed, "neural_seed": neural_seed,
                "normalized_return": normalized_return,
                "success": status == "observed" and normalized_return >= threshold,
                "run_status": status, "fit_seconds": fit_seconds, "steps": args.steps,
                "device": str(device), "peak_vram_mib": peak,
                "external_commit": commits[algorithm],
            })
            completed += 1
            write_json(args.output_dir / "progress.json", {
                "completed_fits": completed, "total_fits": total,
                "last_algorithm": algorithm, "last_mdp_id": record["instance_id"],
                "last_policy_family": record["policy_family"], "last_training_seed": training_seed,
            })
            print(f"fits {completed}/{total}", flush=True)
    summaries = _summaries(output)
    write_csv_rows(args.output_dir / "run_results.csv", output, FIELDS)
    write_csv_rows(args.output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    report = {
        **plan, "status": "complete", "runs": len(output),
        "invalid_runs": sum(row["run_status"] != "observed" for row in output),
        "wall_seconds": time.perf_counter() - started, "summaries": summaries,
    }
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name) for name in ("run_results.csv", "summary.csv")
    }
    write_json(args.output_dir / "report.json", report)
    lines = [
        "# Neural discrete offline-RL development comparison", "",
        f"- CUDA device: `{plan['environment']['gpu']}`; budget: `{args.budget}|S||A|`.",
        f"- Groups/seeds/fits: {len(groups)}/{len(selected_seeds)}/{len(output)}.",
        f"- Fixed gradient steps per fit: {args.steps}; evaluation split accessed: no.", "",
        "| Algorithm | Mean return | Median return | Success | Mean fit (s) | Peak VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['algorithm']} | {row['mean_normalized_return']:.4f} | "
            f"{row['median_normalized_return']:.4f} | {row['success_rate']:.4f} | "
            f"{row['mean_fit_seconds']:.3f} | {row['max_peak_vram_mib']:.1f} |"
        )
    lines += ["", "This is a development-only neural robustness panel, not confirmatory evidence.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
