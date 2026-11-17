"""Recompute pilot features on regenerated pilots and compare to the sealed CSV.

Usage: python driver/verify_features.py c7 [n_groups]
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from common import COHORTS, DATA, load_json, use_workspace, write_json

name, limit = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
use_workspace()
from scripts.analysis.run_unseen_validation import archive_imports  # noqa: E402

archive_imports()
from evaluation import e1_pipeline  # noqa: E402

dataset = DATA / name
manifest = load_json(dataset / "manifest.json")
records = {row["data_only_dataset_id"]: row for row in manifest["datasets"]}
grouped = defaultdict(dict)
for item in load_json(dataset / "data_only_index.json")["datasets"]:
    if item.get("stream") == "pilot" and int(item.get("size_multiplier", -1)) in (2, 5, 10):
        row = records[item["dataset_id"]]
        grouped[(row["instance_id"], row["policy_family"])][int(row["size_multiplier"])] = item
sealed = pd.read_csv(COHORTS[name] / "pilot_features.csv").set_index(["mdp_id", "policy_family", "algorithm"])
keys = sorted(grouped)[:: max(1, len(grouped) // min(limit, len(grouped)))][:limit]
worst = 0.0
for key in keys:
    for row in e1_pipeline._group_features(dataset, key, grouped[key]):
        ref = sealed.loc[(row["mdp_id"], row["policy_family"], row["algorithm"])]
        diff = max(abs(float(row[c]) - float(ref[c])) for c in ref.index)
        worst = max(worst, diff)
write_json(dataset / "feature_check.json", {"groups_checked": len(keys), "max_abs_diff": worst})
print(f"{name}: checked {len(keys)} groups, max |diff| = {worst:.3e}")
