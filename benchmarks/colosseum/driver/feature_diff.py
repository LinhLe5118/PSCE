import sys
from collections import defaultdict
import pandas as pd
from common import COHORTS, DATA, load_json, use_workspace
name = sys.argv[1]
use_workspace()
from scripts.analysis.run_unseen_validation import archive_imports
archive_imports()
from evaluation import e1_pipeline
dataset = DATA / name
manifest = load_json(dataset / "manifest.json")
records = {r["data_only_dataset_id"]: r for r in manifest["datasets"]}
grouped = defaultdict(dict)
for it in load_json(dataset / "data_only_index.json")["datasets"]:
    if it.get("stream") == "pilot" and int(it.get("size_multiplier", -1)) in (2, 5, 10):
        r = records[it["dataset_id"]]; grouped[(r["instance_id"], r["policy_family"])][int(r["size_multiplier"])] = it
sealed = pd.read_csv(COHORTS[name] / "pilot_features.csv").set_index(["mdp_id", "policy_family", "algorithm"])
key = sorted(grouped)[0]
rows = e1_pipeline._group_features(dataset, key, grouped[key])
for row in rows:
    ref = sealed.loc[(row["mdp_id"], row["policy_family"], row["algorithm"])]
    d = {c: float(row[c]) - float(ref[c]) for c in ref.index if abs(float(row[c]) - float(ref[c])) > 1e-9}
    print(row["algorithm"], d)
