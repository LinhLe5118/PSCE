"""Audit a complete development bundle without fitting or reading test data.

Coverage is reported separately at each pilot budget, avoiding inflated sample
counts from treating nested prefixes as independent MDPs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from config.protocol_config import PROTOCOL_CONFIG_PATH
from evaluation.io import sha256_file, write_json
from evaluation.training_protocol import validate_training_manifest, verify_training_buffers


def summarize(manifest: dict) -> dict:
    instances = manifest["instances"]
    physical = {row["file"]: row for row in manifest["datasets"]}
    pilots = [row for row in manifest["datasets"] if row["stream"] == "pilot"]
    thresholds = manifest["protocol_config"]["coverage"]["thresholds"]
    labels = manifest["protocol_config"]["coverage"]["labels"]
    coverage = {}
    for multiplier in sorted({row["size_multiplier"] for row in pilots}):
        rows = [row for row in pilots if row["size_multiplier"] == multiplier]
        values = sorted(row["coverage"]["observed_reachable_sa_fraction"] for row in rows)
        # Same boundary convention as generator: threshold values remain in
        # the lower band. These are prefix measurements, not max-buffer strata.
        bands = Counter(labels[sum(value > t for t in thresholds)] for value in values)
        coverage[str(multiplier)] = {
            "rows": len(rows), "min": values[0], "max": values[-1],
            "median": (values[(len(values)-1)//2] + values[len(values)//2]) / 2,
            "bands": {label: bands[label] for label in labels},
        }
    return {
        "scope": "development distribution audit; no training or generalization result",
        "independent_mdp_count": len(instances),
        "families": dict(Counter(row["family"] for row in instances)),
        "states": sorted({row["num_states"] for row in instances}),
        "actions": sorted({row["num_actions"] for row in instances}),
        "physical_buffers": len(physical),
        "logical_prefixes": len(manifest["datasets"]),
        "physical_transitions": sum(row["max_size"] for row in physical.values()),
        "max_dense_P_R_bytes": max(16 * row["num_states"]**2 * row["num_actions"] for row in instances),
        "memory_note": "P and R only; excludes solver copies, bootstrap, Python and buffers",
        "pilot_coverage_by_multiplier": coverage,
        "dynamic_range_rejections": manifest["dynamic_range_rejection_count"],
        "pending_validations": manifest["pending_validations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = validate_training_manifest(manifest)
    verified = verify_training_buffers(args.dataset_dir, manifest)
    result = summarize(manifest)
    result.update(training_grid=counts, verified_buffers=verified,
                  manifest_sha256=sha256_file(manifest_path),
                  profile_sha256=sha256_file(PROTOCOL_CONFIG_PATH),
                  audit_source_sha256=sha256_file(Path(__file__)),
                  stored_buffer_bytes=sum((args.dataset_dir / name).stat().st_size
                                          for name in {r["file"] for r in manifest["datasets"]}))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
