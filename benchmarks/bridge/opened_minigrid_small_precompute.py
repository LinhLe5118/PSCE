"""Checkpoint metadata-small, already opened MiniGrid groups while fresh OEH-H runs.

This is an exploratory scheduling wrapper around the unchanged full-OEH-B
``run_one`` implementation. It never reads learner labels or writes a panel
manifest. The queued 70-group runner later verifies and reuses these files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import opened_minigrid_full_exploratory as full
from oeh import predictive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pilot_file = full.DATA / "manifest.json"
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    addendum = json.loads(full.ADDENDUM.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or
            len(pilot["pilot_groups"]) != 70 or
            len({row["mdp_id"] for row in pilot["pilot_groups"]}) != 14):
        raise RuntimeError("Unexpected opened MiniGrid pilot manifest")
    rows = [row for row in pilot["pilot_groups"] if int(row["num_states"]) <= 335]
    print(f"Metadata-small groups: {len(rows)} from "
          f"{len({row['mdp_id'] for row in rows})} published tasks", flush=True)
    if args.dry_run:
        print("\n".join(sorted({row["mdp_id"] for row in rows})))
        return
    full.OUTPUT.mkdir(parents=True, exist_ok=True)
    (full.OUTPUT / "groups").mkdir(exist_ok=True)
    full.install()
    source_hash = full.sha256(Path(full.__file__))
    backend_hash = full.sha256(full.ROOT / "benchmarks/bridge/fast_backend.py")
    predictive_hash = full.sha256(Path(predictive.__file__))
    for index, row in enumerate(rows, 1):
        item = full.run_one(row, addendum["posterior_structure"], source_hash,
                            backend_hash, predictive_hash)
        print(f"Small MiniGrid {index}/{len(rows)}: {row['mdp_id']} / "
              f"{row['policy_family']} {item['status']} "
              f"{item['runtime_seconds']:.1f}s", flush=True)
        if item["status"] != "EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN":
            raise RuntimeError(f"Exploratory OEH-B failure: {row['mdp_id']}")


if __name__ == "__main__":
    main()
