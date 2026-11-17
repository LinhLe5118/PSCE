#!/usr/bin/env python
"""Compare all compatible external baselines on the paired development slice."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.aggregation.correlation import holm_adjust  # noqa: E402
from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from scripts.analysis.run_calibrated_comparison import cluster_bootstrap_difference  # noqa: E402


EXTERNAL = (
    "pib_spibb_n10_author",
    "vi_lcb_cb005_author_empirical",
    "vi_lcb_cb144_theory",
    "discrete_bcq_author",
    "discrete_cql_d3rlpy",
)
COMPARATORS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
FIELDS = (
    "budget", "training_seed", "external_algorithm", "comparator", "n_groups",
    "num_mdp_clusters", "external_mean_return", "comparator_mean_return",
    "external_minus_comparator", "ci_low", "ci_high", "p_value", "p_holm",
    "significant_after_holm",
)
MEAN_FIELDS = ("algorithm", "n_groups", "mean_return", "median_return", "success_rate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=ROOT / "artifacts/research/final_local_methods_development_v2_2026-09-20")
    parser.add_argument("--spibb-dir", type=Path, default=ROOT / "artifacts/research/final_spibb_development_2026-09-20")
    parser.add_argument("--vi-lcb-dir", type=Path, default=ROOT / "artifacts/research/final_vi_lcb_development_2026-09-20")
    parser.add_argument("--bcq-dir", type=Path, default=ROOT / "artifacts/research/final_neural_bcq_development_2026-09-20")
    parser.add_argument("--cql-dir", type=Path, default=ROOT / "artifacts/research/final_neural_cql_development_2026-09-20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--training-seed", type=int, default=0)
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)
    inputs = {
        "local_rl": args.local_dir / "rl_run_results.csv",
        "local_bc": args.local_dir / "bc_sanity_results.csv",
        "spibb": args.spibb_dir / "run_results.csv",
        "vi_lcb": args.vi_lcb_dir / "run_results.csv",
        "bcq": args.bcq_dir / "run_results.csv",
        "cql": args.cql_dir / "run_results.csv",
    }
    grouped = defaultdict(list)
    success = defaultdict(list)
    for path in inputs.values():
        for row in read_csv_rows(path):
            if (
                int(float(row["training_size_multiplier"])) == args.budget
                and int(float(row["training_seed"])) == args.training_seed
            ):
                key = (row["mdp_id"], row["policy_family"], row["algorithm"])
                grouped[key].append(float(row["normalized_return"]))
                success[key].append(str(row["success"]).lower() == "true")
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    algorithms = EXTERNAL + COMPARATORS
    pairs_by_algorithm = {
        algorithm: sorted((key[0], key[1]) for key in means if key[2] == algorithm)
        for algorithm in algorithms
    }
    for algorithm, pairs in pairs_by_algorithm.items():
        if len(pairs) != 75:
            raise ValueError(f"expected 75 groups for {algorithm}, got {len(pairs)}")
    pairs = pairs_by_algorithm[EXTERNAL[0]]
    if any(pairs_by_algorithm[algorithm] != pairs for algorithm in algorithms):
        raise ValueError("algorithm panels do not contain identical MDP-policy pairs")

    output = []
    for external_index, external in enumerate(EXTERNAL):
        for comparator_index, comparator in enumerate(COMPARATORS):
            external_values = [means[pair + (external,)] for pair in pairs]
            comparator_values = [means[pair + (comparator,)] for pair in pairs]
            difference, low, high, p_value = cluster_bootstrap_difference(
                [pair[0] for pair in pairs], external_values, comparator_values,
                num_resamples=10_000, confidence=0.95,
                seed=20260919 + 10 * external_index + comparator_index,
            )
            output.append({
                "budget": args.budget, "training_seed": args.training_seed,
                "external_algorithm": external, "comparator": comparator,
                "n_groups": len(pairs), "num_mdp_clusters": len({pair[0] for pair in pairs}),
                "external_mean_return": float(np.mean(external_values)),
                "comparator_mean_return": float(np.mean(comparator_values)),
                "external_minus_comparator": difference, "ci_low": low, "ci_high": high,
                "p_value": p_value, "p_holm": np.nan, "significant_after_holm": False,
            })
    adjusted, significant = holm_adjust([row["p_value"] for row in output])
    for row, p_holm, is_significant in zip(output, adjusted, significant):
        row["p_holm"] = p_holm
        row["significant_after_holm"] = is_significant

    mean_rows = []
    for algorithm in algorithms:
        values = [means[pair + (algorithm,)] for pair in pairs]
        flags = [flag for pair in pairs for flag in success[pair + (algorithm,)]]
        mean_rows.append({
            "algorithm": algorithm, "n_groups": len(pairs),
            "mean_return": float(np.mean(values)), "median_return": float(np.median(values)),
            "success_rate": sum(flags) / len(flags),
        })
    write_csv_rows(args.output_dir / "paired_comparisons.csv", output, FIELDS)
    write_csv_rows(args.output_dir / "method_summary.csv", mean_rows, MEAN_FIELDS)
    report = {
        "status": "complete", "development_only": True,
        "sealed_evaluation_accessed": False, "budget": args.budget,
        "training_seed": args.training_seed, "groups": len(pairs),
        "clusters": len({pair[0] for pair in pairs}), "bootstrap_draws": 10_000,
        "holm_family": "all 15 external-versus-local comparisons",
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "source_sha256": sha256_file(Path(__file__)), "method_summary": mean_rows,
        "comparisons": output,
    }
    write_json(args.output_dir / "report.json", report)
    lines = [
        "# Compatible external baseline comparison", "",
        f"Budget `{args.budget}|S||A|`, registered training seed {args.training_seed}; 75 groups clustered by 15 base MDPs.",
        "All methods use the same logged prefix and exact normalized-return scorer.", "",
        "| Method | Mean | Median | Success |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(mean_rows, key=lambda item: -item["mean_return"]):
        lines.append(
            f"| {row['algorithm']} | {row['mean_return']:.4f} | "
            f"{row['median_return']:.4f} | {row['success_rate']:.4f} |"
        )
    lines += ["", "| External | Comparator | Difference (95% cluster CI) | Holm p |", "|---|---|---:|---:|"]
    for row in output:
        lines.append(
            f"| {row['external_algorithm']} | {row['comparator']} | "
            f"{row['external_minus_comparator']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['p_holm']:.4g} |"
        )
    lines += ["", "These are development results; the evaluation split remains sealed.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
