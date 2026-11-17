"""Compute one metadata-small opened MiniGrid checkpoint with the sealed runner.

This schedules existing exploratory tasks while the independent prospective
panel is running. It does not read learner labels or create an analysis result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import opened_minigrid_full_exploratory as full
from oeh import predictive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-groups", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_groups <= 5:
        raise ValueError("One to five groups per guarded invocation")
    pilot = json.loads((full.DATA / "manifest.json").read_text(encoding="utf-8"))
    addendum = json.loads(full.ADDENDUM.read_text(encoding="utf-8"))
    if (pilot["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS" or
            len(pilot["pilot_groups"]) != 70):
        raise RuntimeError("Unexpected opened MiniGrid pilot")
    candidates = (("MiniGrid-LavaCrossingS11N5-v0", 469),
                  ("MiniGrid-LavaGapS7-v0", 481),
                  ("MiniGrid-SimpleCrossingS11N5-v0", 493))
    rows = []
    for mdp_id, states in candidates:
        group = [row for row in pilot["pilot_groups"]
                 if row["mdp_id"] == mdp_id and int(row["num_states"]) == states]
        if len(group) != 5:
            raise RuntimeError(f"Unexpected published task group count: {mdp_id}")
        missing = [row for row in group if not (full.OUTPUT / "groups" / (
            hashlib.sha256(f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
            + ".json")).exists()]
        if missing:
            rows = group
            break
    if not rows:
        print("All three medium MiniGrid tasks already checkpointed", flush=True)
        return
    print("Selected", rows[0]["mdp_id"], "missing", len(missing), flush=True)
    if args.dry_run:
        return
    full.OUTPUT.mkdir(parents=True, exist_ok=True)
    (full.OUTPUT / "groups").mkdir(exist_ok=True)
    full.install()
    source_hash = full.sha256(Path(full.__file__))
    backend_hash = full.sha256(full.ROOT / "benchmarks/bridge/fast_backend.py")
    predictive_hash = full.sha256(Path(predictive.__file__))
    completed = 0
    for row in rows:
        path = full.OUTPUT / "groups" / (
            hashlib.sha256(
                f"{row['mdp_id']}|{row['policy_family']}".encode()).hexdigest()[:24]
            + ".json")
        if path.exists():
            continue
        result = full.run_one(row, addendum["posterior_structure"],
                              source_hash, backend_hash, predictive_hash)
        print(result["mdp_id"], result["policy_family"], result["status"],
              result["runtime_seconds"], flush=True)
        if result["status"] != "EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN":
            raise RuntimeError("Checkpoint failed; inspect saved failure JSON")
        completed += 1
        if completed >= args.max_groups:
            break


if __name__ == "__main__":
    main()
