"""Compute a disjoint third-cohort pilot feature shard before any outcome."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.run_oeh_120_cross_capacity import DATASET, OUTPUT
from scripts.analysis.run_unseen_validation import archive_imports, load_json, write_json


base.OUTPUT = OUTPUT
base.DATASET = DATASET


def groups(manifest: dict) -> dict:
    indexed = load_json(DATASET / "data_only_index.json")["datasets"]
    records = {row["data_only_dataset_id"]: row for row in manifest["datasets"]}
    grouped = defaultdict(dict)
    for item in indexed:
        if item.get("stream") != "pilot" or int(item.get("size_multiplier", -1)) not in (2, 5, 10):
            continue
        row = records[item["dataset_id"]]
        if row["sha256"] != item["sha256"] or row["file"] != item["file"]:
            raise RuntimeError("Pilot index differs from manifest")
        key = row["instance_id"], row["policy_family"]
        size = int(row["size_multiplier"])
        if size in grouped[key]:
            raise RuntimeError("Duplicate pilot dataset")
        grouped[key][size] = row
    if len(grouped) != 600 or any(set(group) != {2, 5, 10} for group in grouped.values()):
        raise RuntimeError("Incomplete third-cohort pilot grid")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards != 8 or not 0 <= args.shard < args.shards:
        raise ValueError("Eight disjoint pilot shards are required")
    doc = base.protocol()
    manifest = base.dataset()
    pipeline, _, _, _ = archive_imports()
    grouped = groups(manifest)
    keys = sorted(grouped)
    root = OUTPUT / "pilot_group_shards"
    root.mkdir(parents=True, exist_ok=True)
    protocol_sha = base.sha256(OUTPUT / "protocol.json")
    done = 0
    for index, key in enumerate(keys):
        if index % args.shards != args.shard:
            continue
        identity = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()
        path = root / f"{identity}.json"
        if path.exists():
            record = load_json(path)
            if (record["group"] != list(key) or record["protocol_sha256"] != protocol_sha
                    or len(record["rows"]) != 3):
                raise RuntimeError(f"Invalid saved pilot group: {path}")
        else:
            rows = pipeline._group_features(DATASET, key, grouped[key])
            if len(rows) != 3 or any(row["mdp_id"] != key[0] or row["policy_family"] != key[1]
                                     for row in rows):
                raise RuntimeError(f"Incomplete pilot group: {key}")
            write_json(path, {"group": list(key), "protocol_sha256": protocol_sha,
                              "rows": rows})
        done += 1
        if done % 10 == 0:
            print(f"Pilot shard {args.shard}: {done}/75 groups", flush=True)
    if done != 75:
        raise RuntimeError("Pilot shard size differs from frozen 600-group grid")
    write_json(root / f"complete_{args.shard}.json", {
        "shard": args.shard, "shards": args.shards,
        "groups": done, "protocol_sha256": protocol_sha,
    })
    print(f"Pilot shard {args.shard} complete", flush=True)


if __name__ == "__main__":
    main()
