"""Fixed x10 external paper-method panel for the sealed E1 runner."""

from __future__ import annotations

import importlib
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np

from environments.behavior_policies import behavior_policy_suite
from environments.benchmark_environments import make_benchmark_mdp
from evaluation.e1_pipeline import EXTERNAL_METHODS
from evaluation.io import sha256_file
from scripts.external import run_neural_discrete_development as neural
from scripts.external import run_spibb_development as spibb_adapter
from scripts.external import run_vi_lcb_development as vi_adapter
from scripts.training.run_empirical_requirement import _load_run, _score_policy, _training_seed_index


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from vi_lcb import vi_lcb_discounted  # noqa: E402


def _require_module_origin(module: Any, root: Path, label: str) -> None:
    origin = getattr(module, "__file__", None)
    if not origin or not Path(origin).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"E1 {label} import is outside the pinned checkout")


def verify_external_components(external_root: Path) -> dict[str, Any]:
    """Pin the real implementations and CUDA before sealed input is opened."""

    spibb_dir = external_root / "external_spibb"
    vi_dir = external_root / "external_vi_lcb"
    bcq_dir = external_root / "external_bcq"
    cql_dir = external_root / "external_d3rlpy"
    spibb, transitions, spibb_commit = spibb_adapter._load_author_modules(spibb_dir)
    _require_module_origin(spibb, spibb_dir, "SPIBB")
    _require_module_origin(transitions, spibb_dir, "SPIBB transition model")
    vi_commit = vi_adapter._check_author_repo(vi_dir)
    bcq_commit = neural._git_commit_clean(bcq_dir, neural.BCQ_COMMIT)
    cql_commit = neural._git_commit_clean(cql_dir, neural.D3RLPY_COMMIT)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("E1 BCQ/CQL require installed PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("E1 BCQ/CQL require CUDA for the frozen neural panel")
    bcq_module_dir = bcq_dir / "discrete_BCQ"
    sys.path.insert(0, str(bcq_module_dir.resolve()))
    _require_module_origin(importlib.import_module("discrete_BCQ"), bcq_module_dir, "BCQ")
    sys.path.insert(0, str(cql_dir.resolve()))
    d3rlpy = importlib.import_module("d3rlpy")
    _require_module_origin(d3rlpy, cql_dir, "d3rlpy CQL")
    return {
        "methods": list(EXTERNAL_METHODS),
        "source_type": {
            "pib_spibb_n10_author": "author_repository_unmodified_core",
            "vi_lcb_cb005_author_empirical": "discounted_paper_spec_port",
            "discrete_bcq_author": "author_repository_unmodified_learner",
            "discrete_cql_d3rlpy": "pinned_d3rlpy_implementation",
        },
        "commits": {
            "pib_spibb_n10_author": spibb_commit,
            "vi_lcb_cb005_author_empirical": vi_commit,
            "discrete_bcq_author": bcq_commit,
            "discrete_cql_d3rlpy": cql_commit,
        },
        "source_sha256": {
            "spibb.py": sha256_file(spibb_dir / "spibb.py"),
            "modelTransitions.py": sha256_file(spibb_dir / "modelTransitions.py"),
            "vi_lcb.py": sha256_file(ROOT / "code/vi_lcb.py"),
            "discrete_BCQ.py": sha256_file(bcq_dir / "discrete_BCQ/discrete_BCQ.py"),
            "d3rlpy_cql.py": sha256_file(cql_dir / "d3rlpy/algos/qlearning/cql.py"),
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "d3rlpy": getattr(d3rlpy, "__version__", "unknown"),
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


def build_external_train_score_callback(
    dataset_dir: Path, manifest: Mapping[str, Any], external_root: Path,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Run the four T1 external methods on the same x10 buffer and seed cell."""

    import torch

    neural.torch = torch
    spibb, model_transitions, _ = spibb_adapter._load_author_modules(external_root / "external_spibb")
    indexed = {
        (str(row["instance_id"]), str(row["policy_family"]), _training_seed_index(row)): row
        for row in manifest["datasets"]
        if row.get("benchmark_split") == "evaluation" and row.get("stream") == "training"
        and int(row["size_multiplier"]) == 10
    }
    instances = {str(row["instance_id"]): row for row in manifest["instances"] if row.get("split") == "evaluation"}
    mdp_cache: dict[str, Any] = {}
    transition_cache: dict[str, dict[str, np.ndarray]] = {}
    arrays_cache: dict[str, dict[str, np.ndarray]] = {}
    policy_cache: dict[tuple[str, str], np.ndarray] = {}
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    def run(job: Mapping[str, Any]) -> Mapping[str, Any]:
        method = str(job["method"])
        if method not in EXTERNAL_METHODS or int(job["size_multiplier"]) != 10:
            raise ValueError(f"Unregistered E1 external job: {method}")
        key = (str(job["instance_id"]), str(job["policy_family"]), int(job["training_seed"]))
        record = indexed[key]
        if str(record["file"]) != str(job["buffer_file"]) or str(record["sha256"]).lower() != str(job["buffer_sha256"]).lower():
            raise ValueError(f"External E1 job/buffer mismatch: {key}")
        instance = instances[key[0]]
        if key[0] not in mdp_cache:
            mdp_cache.clear()
            policy_cache.clear()
            mdp_cache[key[0]] = make_benchmark_mdp(type("Spec", (), instance)())
        mdp = mdp_cache[key[0]]
        fit_started = time.perf_counter()
        peak_vram = None
        if method == "pib_spibb_n10_author":
            if str(record["file"]) not in transition_cache:
                transition_cache.clear()
            transitions = _load_run(dataset_dir, dict(record), transition_cache)
            batch = [[item.action, item.state, item.next_state, item.reward] for item in transitions]
            if key[:2] not in policy_cache:
                policies = behavior_policy_suite(mdp, seed=int(instance["seed"]) + 3_000_000,
                                                 variant_index=int(instance["ordinal"]))
                policy_cache[key[:2]] = next(item.probabilities for item in policies if item.family == key[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                model = model_transitions.ModelTransitions(batch, mdp.num_states, mdp.num_actions)
            mask = spibb.compute_mask_N_wedge(mdp.num_states, mdp.num_actions, 10, batch)
            learner = spibb.spibb(
                mdp.gamma, mdp.num_states, mdp.num_actions, policy_cache[key[:2]],
                mask, model.transitions,
                spibb_adapter._reward_model(batch, mdp.num_states, mdp.num_actions), "Pi_b_SPIBB",
            )
            learner.fit()
            policy = learner.pi
        elif method == "vi_lcb_cb005_author_empirical":
            if str(record["file"]) not in transition_cache:
                transition_cache.clear()
            transitions = _load_run(dataset_dir, dict(record), transition_cache)
            policy = vi_lcb_discounted(
                transitions, num_states=mdp.num_states, num_actions=mdp.num_actions,
                gamma=mdp.gamma, cb=.05, delta=.05, iterations=100,
                terminal_states=mdp.terminal_states,
            ).policy
        else:
            if str(record["file"]) not in arrays_cache:
                arrays_cache.clear()
            arrays = neural._load_arrays(dataset_dir, dict(record), arrays_cache)
            neural_seed = 20260919 + 1000 * key[2] + int(instance["ordinal"])
            if method == "discrete_bcq_author":
                policy, _, peak_vram = neural._fit_bcq(
                    arrays, mdp.num_states, mdp.num_actions, mdp.gamma,
                    500, neural_seed, device, external_root / "external_bcq",
                )
            else:
                policy, _, peak_vram = neural._fit_cql(
                    arrays, mdp.num_states, mdp.num_actions, mdp.gamma,
                    500, neural_seed, device,
                )
        fit_seconds = time.perf_counter() - fit_started
        score, status = _score_policy(
            mdp, np.asarray(policy, dtype=float),
            random_return=float(instance["random_return"]),
            dynamic_range=float(instance["dynamic_range"]),
        )
        if status != "observed":
            raise ValueError(f"E1 external method produced invalid score: {key}/{method}")
        return {
            "status": status, "method": method, "normalized_return": score,
            "success": bool(score >= .95), "buffer_sha256": job["buffer_sha256"],
            "fit_seconds": fit_seconds, "peak_vram_mib": peak_vram,
        }

    return run


__all__ = ["verify_external_components", "build_external_train_score_callback"]
