#!/usr/bin/env python
"""Generate the post-freeze OEH-v3 development holdout without evaluation IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environments.benchmark_environments import (
    BENCHMARK_SEED, DEVELOPMENT_FAMILY_COUNTS, STRUCTURAL_FAMILIES,
    BenchmarkSpec,
)
from scripts.data.generate_offline_datasets import generate_benchmark_bundle


COHORT_SIZE = 60
SEED_OFFSET = 4_000_000


def fresh_specs():
    counts = {
        family: COHORT_SIZE * DEVELOPMENT_FAMILY_COUNTS[family]
        // sum(DEVELOPMENT_FAMILY_COUNTS.values())
        for family in STRUCTURAL_FAMILIES
    }
    if sum(counts.values()) != COHORT_SIZE:
        raise RuntimeError("Fresh cohort allocation must be exact")
    positions = [
        ((index + .5) / count, STRUCTURAL_FAMILIES.index(family), family, index)
        for family, count in counts.items() for index in range(count)
    ]
    family_ordinals = {family: 100 for family in STRUCTURAL_FAMILIES}
    specs = []
    for ordinal, (_position, _family_index, family, _index) in enumerate(sorted(positions)):
        family_ordinal = family_ordinals[family]
        specs.append(BenchmarkSpec(
            instance_id=f"oeh3_holdout_{family}_{ordinal:03d}",
            split="development", family=family,
            seed=BENCHMARK_SEED + SEED_OFFSET + ordinal,
            ordinal=1000 + ordinal, family_ordinal=family_ordinal,
        ))
        family_ordinals[family] += 1
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    if status:
        raise RuntimeError("Fresh holdout generation requires a clean committed tree")
    manifest = generate_benchmark_bundle(
        args.output, specs=fresh_specs(), run_mode="review",
        deviation_reason=(
            "OEH-v3 fresh post-freeze development holdout; formula and analysis "
            "protocol were committed before any buffer or learner outcome existed."
        ),
    )
    manifest["review_cohort"] = {
        "role": "oeh_v3_post_freeze_holdout",
        "formula_commit": commit,
        "cohort_size": COHORT_SIZE,
        "seed_namespace": f"BENCHMARK_SEED+{SEED_OFFSET}+ordinal",
        "evaluation_instance_overlap": False,
        "single_use": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: generated {COHORT_SIZE} post-freeze development MDPs at {args.output}")


if __name__ == "__main__":
    main()
