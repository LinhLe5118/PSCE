"""Cohort-7 comparator inputs for the Colosseum MDPs, with the team's code.

  published  oeh_final_published._task (TQ/SACo reference, ERI, BWD)
  eh         e1_oracle_qvi_diagnostic.finite_horizon_online_bounds (EH oracle)

The team's tasks build the MDP with ``make_benchmark_mdp``; here it returns
the Colosseum kernel instead (``colosseum_mdp.to_mdp``). Nothing else changes.
Writes RES/published_preoutcome.csv and RES/oracle_descriptors_preoutcome.csv.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import DATA, RES, RUN_BUNDLE, load_json, use_workspace


def _patch():
    use_workspace()
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    import environments.benchmark_environments as B
    from environments.tabular_environments import TabularMDP
    from colosseum_mdp import to_mdp
    B.make_benchmark_mdp = lambda spec: to_mdp(spec.instance_id, TabularMDP)
    return to_mdp, TabularMDP


def published_task(task: dict) -> dict:
    _patch()
    from scripts.analysis import oeh_final_published as P
    return P._task(task)


def eh_task(mdp_id: str) -> dict:
    to_mdp, TabularMDP = _patch()
    from scripts.analysis.develop_b2_eh_oracle_aft import EH
    from scripts.analysis.e1_oracle_qvi_diagnostic import finite_horizon_online_bounds
    d = finite_horizon_online_bounds(to_mdp(mdp_id, TabularMDP))
    return {"mdp_id": mdp_id, **{name: d[name] for name in EH}}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    manifest = load_json(RUN_BUNDLE / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot"
                   and int(r.get("size_multiplier", -1)) == 10
                   and r.get("benchmark_split", "evaluation") == "evaluation"),
                  key=lambda r: (r["instance_id"], r["policy_family"]))
    selected = load_json(DATA / "confirm_selection.json")["selected"] if RES.name == "confirm" else None
    if selected is not None and (len(instances) != selected or len(rows) != selected * 5):
        raise SystemExit(f"unexpected bundle: {len(instances)} MDPs, {len(rows)} pilots")
    tick = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        eh = pd.DataFrame(list(pool.map(eh_task, sorted(instances))))
        print(f"EH oracle: {len(eh)} MDPs, {time.time() - tick:.0f}s", flush=True)
        tasks = [{"instance": instances[r["instance_id"]], "row": r, "dataset": str(RUN_BUNDLE)} for r in rows]
        published = pd.DataFrame(list(pool.map(published_task, tasks, chunksize=4)))
        print(f"published: {len(published)} pilots, {time.time() - tick:.0f}s", flush=True)
    eh.to_csv(RES / "oracle_descriptors_preoutcome.csv", index=False)
    published.to_csv(RES / "published_preoutcome.csv", index=False)
    print(published.describe().round(3).to_string())
    print(eh.describe().round(3).to_string())


if __name__ == "__main__":
    main()
