"""Pilot-only features for the Colosseum bundle (no outcome is touched).

Outputs results/colosseum_pilot_features.csv (B2 + OEH3/LCOEH4 columns, the
same function as the sealed cohorts) and results/colosseum_horizon.csv
(data-only OEH-H on the posterior-mean model; oracle panel not used).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import COHORTS, RES, RUN_BUNDLE, load_json, use_workspace

DATASET = RUN_BUNDLE


def _features(task):
    use_workspace()
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    import expectile_fix
    expectile_fix.install()
    expectile_fix.CALLS["fallback"] = 0
    from evaluation import e1_pipeline
    rows = e1_pipeline._group_features(DATASET, tuple(task["key"]), task["prefixes"])
    for row in rows:
        row["_expectile_fallbacks"] = expectile_fix.CALLS["fallback"]
    return rows


def _horizon(task):
    use_workspace()
    from scripts.analysis import run_oeh_final_120 as R
    return R.horizon_group(task)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    use_workspace()
    from scripts.analysis import run_oeh_final_120 as R

    manifest = load_json(DATASET / "manifest.json")
    records = {r["data_only_dataset_id"]: r for r in manifest["datasets"]}
    grouped = defaultdict(dict)
    for item in load_json(DATASET / "data_only_index.json")["datasets"]:
        if item.get("stream") == "pilot" and int(item.get("size_multiplier", -1)) in (2, 5, 10):
            r = records[item["dataset_id"]]
            grouped[(r["instance_id"], r["policy_family"])][int(r["size_multiplier"])] = item
    structure = load_json(COHORTS["c7"] / "protocol.json")["oeh_final"]["structure"]
    feature_tasks = [{"key": list(k), "prefixes": v} for k, v in sorted(grouped.items())]
    horizon_tasks = [{"row": row, "dataset": str(DATASET), "structure": structure, "oracle": False}
                     for row in R.pilot_rows(DATASET)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for result in pool.map(_features, feature_tasks) for row in result]
        horizon = list(pool.map(_horizon, horizon_tasks))
    RES.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    print("groups that needed the expectile fallback:",
          int((frame.groupby(["mdp_id", "policy_family"])._expectile_fallbacks.max() > 0).sum()))
    frame.drop(columns="_expectile_fallbacks").to_csv(RES / "colosseum_pilot_features.csv", index=False)
    pd.DataFrame(horizon).to_csv(RES / "colosseum_horizon.csv", index=False)
    print(f"features {len(rows)} rows, horizon {len(horizon)} groups")


if __name__ == "__main__":
    main()
