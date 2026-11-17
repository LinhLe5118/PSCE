"""Train the three core learners on Colosseum training prefixes -> labels.

Refuses to run unless the pre-outcome prediction seal exists and matches.
Learners, hyper-parameters, exact evaluation and the >=8/10 full-suffix
label rule are the project's own (e1_pipeline / compare_difficulty).
Usage: python driver/labels_colosseum.py <workers>
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import RES as OUT, RUN_BUNDLE, load_json, sha256, use_workspace, write_json

DATASET = RUN_BUNDLE
BUDGETS = (1, 2, 5, 10, 20, 50)


def _mdp_jobs(task):
    """All learner jobs of one MDP (keeps one kernel resident per process)."""
    use_workspace()
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from config.protocol_config import PROTOCOL_CONFIG
    from scripts.training.run_empirical_requirement import _load_run, _score_policy
    import expectile_fix
    expectile_fix.install()
    expectile_fix.CALLS["fallback"] = 0
    from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular
    from environments.tabular_environments import TabularMDP
    from colosseum_mdp import to_mdp

    learning = PROTOCOL_CONFIG["learning"]
    instance = task["instance"]
    mdp = to_mdp(instance["instance_id"], TabularMDP)
    cache, rows = {}, []
    for record in task["records"]:
        transitions = _load_run(DATASET, record, cache)
        for method in ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular"):
            if method == "cfqi_tabular":
                policy = cfqi_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions,
                                      gamma=mdp.gamma, iterations=int(learning["iterations"]),
                                      tolerance=float(learning["tolerance"]))
            elif method == "iql_tabular":
                policy = iql_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions,
                                     gamma=mdp.gamma, expectile=float(learning["iql_expectile"]),
                                     inverse_temperature=float(learning["iql_inverse_temperature"]),
                                     iterations=int(learning["iterations"]), tolerance=float(learning["tolerance"]))
            else:
                policy = behavior_cloning_tabular(transitions, num_states=mdp.num_states,
                                                  num_actions=mdp.num_actions)
            value, status = _score_policy(mdp, np.asarray(policy, dtype=float),
                                          random_return=float(instance["random_return"]),
                                          dynamic_range=float(instance["dynamic_range"]))
            if status != "observed":
                raise RuntimeError(f"invalid score {instance['instance_id']}/{method}")
            rows.append({"mdp_id": instance["instance_id"], "policy_family": record["policy_family"],
                         "algorithm": method, "budget": int(record["size_multiplier"]),
                         "training_seed": int(record["seed_index"]), "normalized_return": value,
                         "success": bool(value >= 0.95),
                         "expectile_fallbacks_so_far": expectile_fix.CALLS["fallback"]})
    return rows


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seal = load_json(OUT / "colosseum_preoutcome_seal.json")
    if seal["predictions_sha256"] != sha256(OUT / "colosseum_predictions_preoutcome.csv"):
        raise SystemExit("Prediction seal mismatch; refusing to compute outcomes")
    use_workspace()
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, _, seed_index = archive_imports()
    from scripts.analysis.compare_difficulty import suffix_event

    manifest = load_json(DATASET / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    by_mdp = defaultdict(list)
    seen = set()
    for row in manifest["datasets"]:
        if row.get("stream") != "training":
            continue
        key = (row["instance_id"], row["policy_family"], int(row["size_multiplier"]), seed_index(row))
        if key in seen:
            raise RuntimeError("duplicate training record")
        seen.add(key)
        by_mdp[row["instance_id"]].append({**row, "seed_index": key[3]})
    if len(seen) != len(instances) * 5 * 6 * 10:
        raise RuntimeError(f"incomplete training grid: {len(seen)}")
    tasks = [{"instance": instances[m], "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    tick = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for result in pool.map(_mdp_jobs, tasks) for row in result]
    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(OUT / "colosseum_outcomes.csv", index=False)
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("every group/learner/budget needs ten seeds")
    wide = counts["sum"].unstack("budget").reindex(columns=list(BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    labels = labels.rename(columns={b: f"successes_x{b}" for b in BUDGETS})
    labels["panel"] = "primary"
    labels.to_csv(OUT / "colosseum_labels.csv", index=False)
    write_json(OUT / "colosseum_outcome_run.json", {"jobs": len(outcomes), "seconds": time.time() - tick,
                                                    "predictions_sha256_at_start": seal["predictions_sha256"]})
    print(f"{len(outcomes)} learner jobs, {len(labels)} labels, {time.time() - tick:.0f}s")


if __name__ == "__main__":
    main()
