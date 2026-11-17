#!/usr/bin/env python
"""Compare the pinned external SPIBB run with the local learner panel."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.io import read_csv_rows, reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from scripts.analysis.run_calibrated_comparison import cluster_bootstrap_difference  # noqa: E402


EXTERNAL = "pib_spibb_n10_author"
COMPARATORS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
FIELDS = (
    "budget", "external_algorithm", "comparator", "n_groups", "num_mdp_clusters",
    "external_mean_return", "comparator_mean_return",
    "external_minus_comparator", "ci_low", "ci_high", "p_value",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=ROOT / "artifacts/research/final_local_methods_development_v2_2026-09-20")
    parser.add_argument("--external-dir", type=Path, default=ROOT / "artifacts/research/final_spibb_development_2026-09-20")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/spibb_comparison_run")
    parser.add_argument("--budget", type=int, default=10)
    args = parser.parse_args(argv)
    reserve_output_dir(args.output_dir)
    inputs = {
        "rl": args.local_dir / "rl_run_results.csv",
        "bc": args.local_dir / "bc_sanity_results.csv",
        "spibb": args.external_dir / "run_results.csv",
    }
    rows = []
    for path in inputs.values():
        rows.extend(read_csv_rows(path))
    grouped = defaultdict(list)
    for row in rows:
        if int(float(row["training_size_multiplier"])) == args.budget:
            grouped[(row["mdp_id"], row["policy_family"], row["algorithm"])].append(
                float(row["normalized_return"])
            )
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    pairs = sorted({(key[0], key[1]) for key in means if key[2] == EXTERNAL})
    if len(pairs) != 75:
        raise ValueError(f"expected 75 external groups, got {len(pairs)}")
    output = []
    for index, comparator in enumerate(COMPARATORS):
        external = [means[pair + (EXTERNAL,)] for pair in pairs]
        baseline = [means[pair + (comparator,)] for pair in pairs]
        difference, low, high, p_value = cluster_bootstrap_difference(
            [pair[0] for pair in pairs], external, baseline,
            num_resamples=10_000, confidence=.95, seed=20260919 + index,
        )
        output.append({
            "budget": args.budget, "external_algorithm": EXTERNAL,
            "comparator": comparator, "n_groups": len(pairs),
            "num_mdp_clusters": len({pair[0] for pair in pairs}),
            "external_mean_return": float(np.mean(external)),
            "comparator_mean_return": float(np.mean(baseline)),
            "external_minus_comparator": difference,
            "ci_low": low, "ci_high": high, "p_value": p_value,
        })
    write_csv_rows(args.output_dir / "comparison.csv", output, FIELDS)
    report = {
        "status": "complete", "development_only": True,
        "sealed_evaluation_accessed": False, "budget": args.budget,
        "bootstrap_draws": 10_000, "cluster": "base MDP",
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "source_sha256": sha256_file(Path(__file__)),
        "comparison_sha256": sha256_file(args.output_dir / "comparison.csv"),
        "comparisons": output,
    }
    write_json(args.output_dir / "report.json", report)
    labels = {"cfqi_tabular": "CFQI", "iql_tabular": "IQL", "behavior_cloning_tabular": "BC"}
    lines = [
        "# External SPIBB comparison", "",
        "Budget: `10|S||A|`; 75 groups; CI resamples 15 base MDPs.", "",
        "| Comparator | SPIBB return | Comparator return | SPIBB − comparator (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            f"| {labels[row['comparator']]} | {row['external_mean_return']:.4f} | "
            f"{row['comparator_mean_return']:.4f} | {row['external_minus_comparator']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )
    lines += ["", "SPIBB N10 beats BC on this development cohort, but is below IQL and CFQI.",
              "The fixed N10 run is a compatible author-code replication, not a tuned strongest-SPIBB claim.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("external comparison complete", flush=True)


if __name__ == "__main__":
    main()
