#!/usr/bin/env python
"""Run the authors' finite-MDP Pi_b-SPIBB core on the development grid.

The external repository is pinned and imported without editing its algorithm.
This adapter supplies the project's verified training prefixes, exact known
behavior policy (an assumption of SPIBB), empirical transition/reward model,
and the repository's common exact-policy scoring contract.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
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

from empirical_requirement import empirical_requirement  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from environments.behavior_policies import behavior_policy_suite  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from evaluation.io import reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from evaluation.training_protocol import verify_training_buffers  # noqa: E402
from scripts.training.run_empirical_requirement import (  # noqa: E402
    _load_run, _score_policy, _training_seed_index,
)


EXPECTED_COMMIT = "833f32b642759111907d2c4d086d1ae1ce9d84cb"
ALGORITHM = "pib_spibb_n10_author"
N_WEDGE = 10
RUN_FIELDS = (
    "mdp_id", "policy_family", "algorithm", "training_size_multiplier",
    "training_seed", "normalized_return", "success", "run_status",
    "fit_seconds", "n_wedge", "external_commit",
)
N_EMP_FIELDS = (
    "mdp_id", "policy_family", "algorithm", "n_emp_multiplier",
    "n_emp_transitions", "n_emp_censored", "success_counts", "success_rates",
)
CURVE_FIELDS = (
    "algorithm", "training_size_multiplier", "valid_runs", "invalid_runs",
    "mean_normalized_return", "median_normalized_return", "success_rate",
    "mean_fit_seconds",
)


def _external_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load_author_modules(path: Path):
    commit = _external_commit(path)
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"SPIBB commit mismatch: {commit}; expected {EXPECTED_COMMIT}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError("SPIBB checkout is dirty; author source must be unmodified")
    sys.path.insert(0, str(path.resolve()))
    import modelTransitions  # type: ignore
    import spibb  # type: ignore
    return spibb, modelTransitions, commit


def _reward_model(batch, num_states: int, num_actions: int) -> np.ndarray:
    counts = np.zeros((num_states, num_actions), dtype=float)
    rewards = np.zeros_like(counts)
    for action, state, _next_state, reward in batch:
        counts[int(state), int(action)] += 1
        rewards[int(state), int(action)] += float(reward)
    return np.divide(rewards, counts, out=np.zeros_like(rewards), where=counts > 0)


def _curve_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["training_size_multiplier"])].append(row)
    output = []
    for multiplier, group in sorted(grouped.items()):
        valid = [r for r in group if r["run_status"] == "observed"]
        output.append({
            "algorithm": ALGORITHM, "training_size_multiplier": multiplier,
            "valid_runs": len(valid), "invalid_runs": len(group) - len(valid),
            "mean_normalized_return": float(np.mean([r["normalized_return"] for r in valid])),
            "median_normalized_return": float(np.median([r["normalized_return"] for r in valid])),
            "success_rate": sum(bool(r["success"]) for r in group) / len(group),
            "mean_fit_seconds": float(np.mean([r["fit_seconds"] for r in group])),
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/week1_development_validation")
    parser.add_argument("--spibb-dir", type=Path, default=ROOT / "tmp/external_spibb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/spibb_development_run")
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args(argv)
    spibb, model_transitions, commit = _load_author_modules(args.spibb_dir)
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = verify_training_buffers(args.dataset_dir, manifest)
    if manifest.get("run_mode") != "review":
        raise ValueError("development review bundle required")
    records = [r for r in manifest["datasets"] if r["stream"] == "training"]
    if any(r["benchmark_split"] != "development" for r in records):
        raise ValueError("evaluation record refused")
    by_pair = defaultdict(list)
    for record in records:
        by_pair[(record["instance_id"], record["policy_family"])].append(record)
    pair_items = sorted(by_pair.items())
    counts = {
        "instance_policy_groups": len(pair_items),
        "training_prefixes": len(records),
    }
    if args.limit_groups is not None:
        pair_items = pair_items[:args.limit_groups]
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    source_files = ("spibb.py", "modelTransitions.py", "LICENSE", "README.md")
    plan = {
        "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "sealed_evaluation_accessed": False,
        "algorithm": ALGORITHM, "space": "Pi_b_SPIBB", "n_wedge": N_WEDGE,
        "n_wedge_basis": "fixed before run; value used as the fixed Soft-SPIBB setting in the author repository",
        "external_repository": "https://github.com/RomainLaroche/SPIBB",
        "external_commit": commit,
        "external_source_sha256": {name: sha256_file(args.spibb_dir / name) for name in source_files},
        "adapter_sha256": sha256_file(Path(__file__)),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "validated_manifest_counts": counts, "verified_training_buffers": verified,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "behavior_policy": "exact registered collection policy reconstructed from public generator seed/spec",
        "model": "author ModelTransitions MLE; empirical mean reward; unseen rows zero as in author default",
        "scoring": "project exact policy evaluation and registered normalized-return threshold",
    }
    write_json(args.output_dir / "plan.json", plan)

    multipliers = tuple(int(x) for x in PROTOCOL_CONFIG["collection"]["size_multipliers"])
    seeds = tuple(int(x) for x in PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    threshold = float(PROTOCOL_CONFIG["learning"]["success_normalized_return"])
    required = int(PROTOCOL_CONFIG["learning"]["success_runs_required"])
    run_rows = []
    quality = defaultdict(lambda: [[] for _ in multipliers])
    cache = {}
    for pair_position, ((instance_id, family), group) in enumerate(pair_items, 1):
        instance = instances[instance_id]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        policies = behavior_policy_suite(
            mdp, seed=int(instance["seed"]) + 3_000_000,
            variant_index=int(instance["ordinal"]),
        )
        pi_b = next(policy.probabilities for policy in policies if policy.family == family)
        indexed = {
            (int(record["size_multiplier"]), _training_seed_index(record)): record
            for record in group if record.get("size_multiplier") is not None
        }
        expected = {(multiplier, seed) for multiplier in multipliers for seed in seeds}
        if set(indexed) != expected:
            raise ValueError(f"incomplete training grid for {(instance_id, family)}")
        for multiplier_index, multiplier in enumerate(multipliers):
            for seed in seeds:
                transitions = _load_run(args.dataset_dir, indexed[(multiplier, seed)], cache)
                batch = [[t.action, t.state, t.next_state, t.reward] for t in transitions]
                with np.errstate(divide="ignore", invalid="ignore"):
                    model = model_transitions.ModelTransitions(
                        batch, mdp.num_states, mdp.num_actions
                    )
                mask = spibb.compute_mask_N_wedge(
                    mdp.num_states, mdp.num_actions, N_WEDGE, batch
                )
                learner = spibb.spibb(
                    mdp.gamma, mdp.num_states, mdp.num_actions, pi_b, mask,
                    model.transitions,
                    _reward_model(batch, mdp.num_states, mdp.num_actions),
                    "Pi_b_SPIBB",
                )
                fit_started = time.perf_counter()
                learner.fit()
                fit_seconds = time.perf_counter() - fit_started
                normalized_return, status = _score_policy(
                    mdp, learner.pi, random_return=float(instance["random_return"]),
                    dynamic_range=float(instance["dynamic_range"]),
                )
                success = status == "observed" and normalized_return >= threshold
                run_rows.append({
                    "mdp_id": instance_id, "policy_family": family,
                    "algorithm": ALGORITHM, "training_size_multiplier": multiplier,
                    "training_seed": seed, "normalized_return": normalized_return,
                    "success": success, "run_status": status,
                    "fit_seconds": fit_seconds, "n_wedge": N_WEDGE,
                    "external_commit": commit,
                })
                quality[(instance_id, family)][multiplier_index].append(normalized_return)
        write_json(args.output_dir / "progress.json", {
            "completed_groups": pair_position, "total_groups": len(pair_items),
            "run_rows": len(run_rows),
        })
        print(f"groups {pair_position}/{len(pair_items)}", flush=True)

    n_emp_rows = []
    for (instance_id, family), scores_by_multiplier in sorted(quality.items()):
        sa = int(instances[instance_id]["num_states"] * instances[instance_id]["num_actions"])
        requirement = empirical_requirement(
            [(multiplier * sa, tuple(scores)) for multiplier, scores in zip(multipliers, scores_by_multiplier)],
            learner=float, evaluate=float, success_threshold=threshold,
            required_successes=required,
        )
        multiplier_requirement = None if requirement.requirement is None else requirement.requirement // sa
        n_emp_rows.append({
            "mdp_id": instance_id, "policy_family": family, "algorithm": ALGORITHM,
            "n_emp_multiplier": multiplier_requirement,
            "n_emp_transitions": requirement.requirement,
            "n_emp_censored": requirement.right_censored,
            "success_counts": json.dumps(list(requirement.success_counts), separators=(",", ":")),
            "success_rates": json.dumps(list(requirement.success_rates), separators=(",", ":")),
        })
    curves = _curve_summary(run_rows)
    write_csv_rows(args.output_dir / "run_results.csv", run_rows, RUN_FIELDS)
    write_csv_rows(args.output_dir / "empirical_sample_complexity.csv", n_emp_rows, N_EMP_FIELDS)
    write_csv_rows(args.output_dir / "learning_curve_summary.csv", curves, CURVE_FIELDS)
    report = {
        **plan, "status": "complete", "groups": len(pair_items),
        "runs": len(run_rows), "invalid_runs": sum(r["run_status"] != "observed" for r in run_rows),
        "observed_requirements": sum(not r["n_emp_censored"] for r in n_emp_rows),
        "censored_requirements": sum(bool(r["n_emp_censored"]) for r in n_emp_rows),
        "budget_10_mean_return": next(r["mean_normalized_return"] for r in curves if r["training_size_multiplier"] == 10),
        "budget_10_success_rate": next(r["success_rate"] for r in curves if r["training_size_multiplier"] == 10),
        "wall_seconds": time.perf_counter() - started,
    }
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name) for name in
        ("run_results.csv", "empirical_sample_complexity.csv", "learning_curve_summary.csv")
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "REPORT.md").write_text(
        "\n".join([
            "# External Pi_b-SPIBB development replication", "",
            f"- Author commit: `{commit}`; author core unmodified.",
            f"- Fixed `N_wedge={N_WEDGE}` before this run.",
            f"- Groups/runs: {len(pair_items)}/{len(run_rows)}; invalid: {report['invalid_runs']}.",
            f"- Observed/right-censored requirements: {report['observed_requirements']}/{report['censored_requirements']}.",
            f"- Budget 10 mean normalized return: {report['budget_10_mean_return']:.6f}.",
            f"- Budget 10 success rate: {report['budget_10_success_rate']:.6f}.",
            "- Development-only compatibility replication; not a confirmatory or strongest-hyperparameter claim.",
            "- Evaluation split was not accessed.", "",
        ]), encoding="utf-8"
    )
    print(f"complete: {len(run_rows)} runs in {report['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
