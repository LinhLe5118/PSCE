#!/usr/bin/env python
"""Build learner-conditional OEH-v4 diagnostics from pilot buffers only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import (  # noqa: E402
    reserve_output_dir, sha256_file, verified_dataset_path, write_csv_rows,
    write_json,
)
from lcoeh_v4 import ALGORITHMS, LC_OEH_V4_FEATURES, estimate_lcoeh_v4  # noqa: E402
from scripts.analysis.run_oeh_development_smoke import development_inputs  # noqa: E402


PREFIXES = (2, 5, 10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-mdps", type=int)
    args = parser.parse_args(argv)
    if args.limit_mdps is not None and args.limit_mdps < 1:
        parser.error("--limit-mdps must be positive")

    manifest, records, labels, index = development_inputs(
        args.dataset_dir, args.limit_mdps)
    grouped = defaultdict(dict)
    for record in records:
        multiplier = int(record["size_multiplier"])
        if multiplier not in PREFIXES:
            continue
        label = labels[record["dataset_id"]]
        key = (str(label["instance_id"]), str(label["policy_family"]))
        if multiplier in grouped[key]:
            raise ValueError(f"duplicate pilot prefix for {key}/{multiplier}")
        grouped[key][multiplier] = record
    incomplete = {key: sorted(set(PREFIXES) - set(by_size))
                  for key, by_size in grouped.items() if set(by_size) != set(PREFIXES)}
    if incomplete:
        raise ValueError(f"incomplete LC-OEH pilot grid: {incomplete}")

    reserve_output_dir(args.output_dir)
    sources = (
        "code/lcoeh_v4.py", "code/empirical_requirement.py",
        "code/adaptive_pilot.py", "scripts/analysis/run_lcoeh_v4.py",
        "scripts/analysis/run_oeh_development_smoke.py",
    )
    plan = {
        "status": "started",
        "method": "learner_conditional_oeh_v4_exploratory",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
        "index_sha256": sha256_file(index),
        "source_sha256": {name: sha256_file(ROOT / name) for name in sources},
        "registered_prefix_multipliers": list(PREFIXES),
        "learners": list(ALGORITHMS),
        "trajectory_folds": 5,
        "epsilon": .05,
        "confidence": .95,
        "outcome_labels_accessed": False,
        "training_buffers_accessed": False,
        "sealed_evaluation_accessed": False,
    }
    write_json(args.output_dir / "plan.json", plan)
    feature_rows, curve_rows, errors = [], [], []
    started = time.perf_counter()
    for position, (key, by_size) in enumerate(sorted(grouped.items()), 1):
        try:
            arrays = {}
            for multiplier, record in sorted(by_size.items()):
                with np.load(verified_dataset_path(args.dataset_dir, record),
                             allow_pickle=False) as loaded:
                    arrays[multiplier] = {
                        name: loaded[name][:int(record["size"])]
                        for name in loaded.files
                    }
            reference = by_size[PREFIXES[-1]]
            result = estimate_lcoeh_v4(
                arrays, reference["initial_state_distribution"],
                num_states=int(reference["num_states"]),
                num_actions=int(reference["num_actions"]),
                gamma=float(manifest["effective_protocol"]["benchmark"]["gamma"]),
                seed_key=f"{key[0]}|{key[1]}",
            )
            for algorithm, values in result.items():
                identity = {
                    "mdp_id": key[0], "policy_family": key[1],
                    "algorithm": algorithm,
                }
                feature_rows.append({
                    **identity,
                    **{name: values[name] for name in LC_OEH_V4_FEATURES},
                    "requirement": "" if values["requirement"] is None
                    else values["requirement"],
                })
                for item in values["curve"]:
                    curve_rows.append({**identity, **item})
        except Exception as exc:  # retain a complete forensic error record
            errors.append({"mdp_id": key[0], "policy_family": key[1],
                           "error": repr(exc)})
        if position % 10 == 0 or errors:
            print(f"LC-OEH {position}/{len(grouped)} errors={len(errors)}", flush=True)
            write_json(args.output_dir / "progress.json", {
                "processed": position, "total": len(grouped), "errors": errors})

    if feature_rows:
        write_csv_rows(args.output_dir / "features.csv", feature_rows,
                       list(feature_rows[0]))
    if curve_rows:
        write_csv_rows(args.output_dir / "curves.csv", curve_rows,
                       list(curve_rows[0]))
    report = {
        **plan,
        "status": "complete" if not errors else "completed_with_errors",
        "groups": len(grouped), "feature_rows": len(feature_rows),
        "curve_rows": len(curve_rows), "errors": errors,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "report.json", report)
    (args.output_dir / "REPORT.md").write_text(
        "# Exploratory learner-conditional OEH-v4\n\n"
        f"Status: **{report['status']}**; groups: {len(grouped)}; learner rows: "
        f"{len(feature_rows)}; errors: {len(errors)}.\n\n"
        "Every covariate was computed from pilot transitions. Learner outcome "
        "labels, independent training buffers, and the sealed evaluation split "
        "were not accessed. This cohort is exploratory because it was already "
        "used to diagnose OEH-v3.\n",
        encoding="utf-8",
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
