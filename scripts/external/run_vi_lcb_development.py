#!/usr/bin/env python
"""Run the discounted VI-LCB paper-spec port on the full development grid."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from empirical_requirement import empirical_requirement  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from evaluation.io import (  # noqa: E402
    reserve_output_dir, sha256_file, verified_dataset_path, write_csv_rows, write_json,
)
from evaluation.training_protocol import verify_training_buffers  # noqa: E402
from scripts.training.run_empirical_requirement import _score_policy, _training_seed_index  # noqa: E402
from vi_lcb import vi_lcb_discounted  # noqa: E402


AUTHOR_COMMIT = "37858d392cf11a01ff5af4ef7815672ec0a36ca9"
VARIANTS = {
    "vi_lcb_cb005_author_empirical": 0.05,
    "vi_lcb_cb144_theory": 144.0,
}
RUN_FIELDS = (
    "mdp_id", "policy_family", "algorithm", "training_size_multiplier",
    "training_seed", "normalized_return", "success", "run_status",
    "fit_seconds", "cb", "delta", "iterations", "external_commit",
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


def _check_author_repo(path: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if commit != AUTHOR_COMMIT:
        raise ValueError(f"VI-LCB author commit mismatch: {commit}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError("tracked VI-LCB author source is dirty")
    return commit


def _load(dataset_dir: Path, record: dict, cache: dict) -> tuple:
    file_name = str(record["file"])
    if file_name not in cache:
        path = verified_dataset_path(dataset_dir, record)
        with np.load(path, allow_pickle=False) as loaded:
            cache[file_name] = {name: loaded[name] for name in loaded.files}
    size = int(record["size"])
    return transitions_from_arrays({name: values[:size] for name, values in cache[file_name].items()})


def _curve(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], int(row["training_size_multiplier"]))].append(row)
    output = []
    for (algorithm, budget), group in sorted(grouped.items()):
        valid = [row for row in group if row["run_status"] == "observed"]
        values = [float(row["normalized_return"]) for row in valid]
        output.append({
            "algorithm": algorithm, "training_size_multiplier": budget,
            "valid_runs": len(valid), "invalid_runs": len(group) - len(valid),
            "mean_normalized_return": float(np.mean(values)),
            "median_normalized_return": float(np.median(values)),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in group])),
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets/week1_development_validation")
    parser.add_argument("--author-dir", type=Path, default=ROOT / "tmp/external_vi_lcb")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args(argv)
    commit = _check_author_repo(args.author_dir)
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = verify_training_buffers(args.dataset_dir, manifest)
    if manifest.get("run_mode") != "review":
        raise ValueError("development review bundle required")
    records = [row for row in manifest["datasets"] if row["stream"] == "training"]
    if any(row["benchmark_split"] != "development" for row in records):
        raise ValueError("evaluation record refused")
    by_group = defaultdict(list)
    for record in records:
        by_group[(record["instance_id"], record["policy_family"])].append(record)
    groups = sorted(by_group.items())
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]
    instances = {row["instance_id"]: row for row in manifest["instances"]}
    delta, iterations = 0.05, int(PROTOCOL_CONFIG["learning"]["iterations"])
    plan = {
        "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "sealed_evaluation_accessed": False,
        "algorithm": "VI-LCB discounted Algorithm 1 / equation (28)",
        "implementation_status": "local paper-spec port; author repo only supplies finite-horizon Gambler notebook",
        "variants": VARIANTS, "delta": delta, "iterations": iterations,
        "external_repository": "https://github.com/Laixishi/Model-based-VI-LCB",
        "external_commit": commit,
        "external_notebook_sha256": sha256_file(args.author_dir / "gambler-offline-model-based.ipynb"),
        "adapter_sha256": sha256_file(Path(__file__)),
        "implementation_sha256": sha256_file(ROOT / "code/vi_lcb.py"),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "verified_training_buffers": verified,
        "reward_adapter": "empirical mean reward; true reward function is not exposed to learners",
        "terminal_adapter": "known terminal semantics enforced as absorbing zero-reward rows",
        "unseen_transition_rows": "uniform as equation (25)",
        "cb_provenance": {
            "0.05": "fixed in the official author's finite-horizon Gambler notebook",
            "144": "example sufficient numerical constant stated after equation (28)",
        },
    }
    write_json(args.output_dir / "plan.json", plan)
    multipliers = tuple(int(x) for x in PROTOCOL_CONFIG["collection"]["size_multipliers"])
    seeds = tuple(int(x) for x in PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    success_threshold = float(PROTOCOL_CONFIG["learning"]["success_normalized_return"])
    required = int(PROTOCOL_CONFIG["learning"]["success_runs_required"])
    cache, rows = {}, []
    quality = defaultdict(lambda: [[] for _ in multipliers])
    for position, ((instance_id, family), group) in enumerate(groups, 1):
        instance = instances[instance_id]
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        indexed = {
            (int(record["size_multiplier"]), _training_seed_index(record)): record
            for record in group
        }
        expected = {(budget, seed) for budget in multipliers for seed in seeds}
        if set(indexed) != expected:
            raise ValueError(f"incomplete training grid for {(instance_id, family)}")
        for budget_index, budget in enumerate(multipliers):
            for seed in seeds:
                transitions = _load(args.dataset_dir, indexed[(budget, seed)], cache)
                for algorithm, cb in VARIANTS.items():
                    fit_started = time.perf_counter()
                    result = vi_lcb_discounted(
                        transitions, num_states=mdp.num_states, num_actions=mdp.num_actions,
                        gamma=mdp.gamma, cb=cb, delta=delta, iterations=iterations,
                        terminal_states=mdp.terminal_states,
                    )
                    fit_seconds = time.perf_counter() - fit_started
                    normalized_return, status = _score_policy(
                        mdp, result.policy, random_return=float(instance["random_return"]),
                        dynamic_range=float(instance["dynamic_range"]),
                    )
                    success = status == "observed" and normalized_return >= success_threshold
                    rows.append({
                        "mdp_id": instance_id, "policy_family": family,
                        "algorithm": algorithm, "training_size_multiplier": budget,
                        "training_seed": seed, "normalized_return": normalized_return,
                        "success": success, "run_status": status,
                        "fit_seconds": fit_seconds, "cb": cb, "delta": delta,
                        "iterations": iterations, "external_commit": commit,
                    })
                    quality[(instance_id, family, algorithm)][budget_index].append(normalized_return)
        write_json(args.output_dir / "progress.json", {
            "completed_groups": position, "total_groups": len(groups), "run_rows": len(rows),
        })
        print(f"groups {position}/{len(groups)}", flush=True)

    n_emp_rows = []
    for (instance_id, family, algorithm), scores in sorted(quality.items()):
        sa = int(instances[instance_id]["num_states"] * instances[instance_id]["num_actions"])
        requirement = empirical_requirement(
            [(budget * sa, tuple(values)) for budget, values in zip(multipliers, scores)],
            learner=float, evaluate=float, success_threshold=success_threshold,
            required_successes=required,
        )
        n_emp_rows.append({
            "mdp_id": instance_id, "policy_family": family, "algorithm": algorithm,
            "n_emp_multiplier": None if requirement.requirement is None else requirement.requirement // sa,
            "n_emp_transitions": requirement.requirement,
            "n_emp_censored": requirement.right_censored,
            "success_counts": json.dumps(list(requirement.success_counts), separators=(",", ":")),
            "success_rates": json.dumps(list(requirement.success_rates), separators=(",", ":")),
        })
    curves = _curve(rows)
    write_csv_rows(args.output_dir / "run_results.csv", rows, RUN_FIELDS)
    write_csv_rows(args.output_dir / "empirical_sample_complexity.csv", n_emp_rows, N_EMP_FIELDS)
    write_csv_rows(args.output_dir / "learning_curve_summary.csv", curves, CURVE_FIELDS)
    report = {
        **plan, "status": "complete", "groups": len(groups), "runs": len(rows),
        "invalid_runs": sum(row["run_status"] != "observed" for row in rows),
        "observed_requirements": {
            algorithm: sum(row["algorithm"] == algorithm and not row["n_emp_censored"] for row in n_emp_rows)
            for algorithm in VARIANTS
        },
        "wall_seconds": time.perf_counter() - started, "curves": curves,
    }
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name) for name in
        ("run_results.csv", "empirical_sample_complexity.csv", "learning_curve_summary.csv")
    }
    write_json(args.output_dir / "report.json", report)
    budget10 = [row for row in curves if row["training_size_multiplier"] == 10]
    lines = [
        "# VI-LCB paper-spec development replication", "",
        f"- Author repository commit: `{commit}`.",
        "- Discounted implementation: local port of Algorithm 1/equation (28); the author repo contains only the finite-horizon Gambler notebook.",
        f"- Groups/runs: {len(groups)}/{len(rows)}; invalid: {report['invalid_runs']}.", "",
        "| Variant | Budget-10 mean return | Success | Observed N_emp |",
        "|---|---:|---:|---:|",
    ]
    for row in budget10:
        lines.append(
            f"| {row['algorithm']} | {row['mean_normalized_return']:.4f} | "
            f"{row['success_rate']:.4f} | {report['observed_requirements'][row['algorithm']]} |"
        )
    lines += ["", "This is development evidence; the sealed evaluation split was not accessed.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
