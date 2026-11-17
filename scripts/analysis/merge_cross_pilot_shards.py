"""Merge 600 verified third-cohort pilot groups in the original sorted order."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.compare_difficulty import METHODS
from scripts.analysis.develop_b2_eh_oracle_aft import KEYS
from scripts.analysis.run_oeh_120_cross_capacity import DATASET, OUTPUT
from scripts.analysis.run_cross_pilot_shard import groups
from scripts.analysis.run_unseen_validation import load_json, write_json


base.OUTPUT = OUTPUT
base.DATASET = DATASET


def main() -> None:
    if (OUTPUT / "preoutcome_seal.json").exists():
        raise FileExistsError("Outcome-independent forecasts already sealed")
    if (OUTPUT / "pilot_features.csv").exists():
        raise FileExistsError("Original pilot feature file already exists")
    base.protocol()
    manifest = base.dataset()
    grouped = groups(manifest)
    root = OUTPUT / "pilot_group_shards"
    protocol_sha = base.sha256(OUTPUT / "protocol.json")
    for shard in range(8):
        record = load_json(root / f"complete_{shard}.json")
        if record != {"shard": shard, "shards": 8,
                      "groups": 75, "protocol_sha256": protocol_sha}:
            raise RuntimeError(f"Pilot shard {shard} incomplete")
    rows = []
    for key in sorted(grouped):
        identity = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()
        record = load_json(root / f"{identity}.json")
        if (record["group"] != list(key) or record["protocol_sha256"] != protocol_sha
                or len(record["rows"]) != 3
                or {row["algorithm"] for row in record["rows"]} != set(METHODS)):
            raise RuntimeError(f"Invalid pilot group: {key}")
        rows.extend(record["rows"])
    frame = pd.DataFrame(rows)
    if len(frame) != 1800 or frame[KEYS].duplicated().any():
        raise RuntimeError("Merged pilot feature grid incomplete")
    path = OUTPUT / "pilot_features.csv"
    frame.to_csv(path, index=False)
    write_json(root / "merge_seal.json", {
        "groups": len(grouped), "rows": len(frame),
        "protocol_sha256": protocol_sha,
        "dataset_manifest_sha256": base.sha256(DATASET / "manifest.json"),
        "pilot_features_sha256": base.sha256(path),
        "merge_script_sha256": base.sha256(Path(__file__)),
        "learner_outcomes_observed": 0,
    })
    print("Merged and sealed 600 pilot groups into 1,800 rows", flush=True)


if __name__ == "__main__":
    main()
