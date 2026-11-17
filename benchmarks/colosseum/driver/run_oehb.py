"""OEH-B posterior-predictive groups with the frozen final structure.

Usage: python driver/run_oehb.py <dataset> <workers> [limit] [--pick a,b,c]
Seeds use each cohort's original namespace, so opened-cohort groups are the
same computation as in the sealed runs. Resumable: finished groups are skipped.
"""
from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import COHORTS, DATA, OUT, load_json, use_workspace

NAMESPACES = {"c4": "oeh-final-fourth-new-120-v1-20261002", "c7": None,
              "colosseum_bundle": "external-colosseum-trial-20261013",
              "confirm_bundle": "external-colosseum-confirm-20261014",
              # meta-training cohorts: the namespace of the original ``meta`` phase
              **{name: f"oeh-final-meta|{name}" for name in
                 ("first60", "second60", "first120", "second120", "third120")}}


def main() -> None:
    use_workspace()
    from scripts.analysis import run_oeh_final_120 as R

    name, workers = sys.argv[1], int(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    doc = load_json(COHORTS["c7"] / "protocol.json")
    structure, draws = doc["oeh_final"]["structure"], doc["oeh_final"]["requirement_draws"]
    namespace = NAMESPACES[name] or doc["namespace"]
    dataset = DATA / name
    target = OUT / "oehb" / name
    target.mkdir(parents=True, exist_ok=True)
    rows = R.pilot_rows(dataset)
    if "--pick" in sys.argv:
        wanted = sys.argv[sys.argv.index("--pick") + 1].split(",")
        rows = [r for r in rows if any(w in r["instance_id"] for w in wanted)]
        seen, picked = set(), []
        for r in rows:  # one behavior per requested MDP pattern
            tag = next(w for w in wanted if w in r["instance_id"])
            if tag not in seen:
                seen.add(tag)
                picked.append(r)
        rows = picked
    tasks = []
    for row in rows:
        path = target / f"{R._group_name(row)}.json"
        if path.exists():
            continue
        path.with_suffix(".claim").unlink(missing_ok=True)
        digest = hashlib.sha256(f"{namespace}|{row['instance_id']}|{row['policy_family']}".encode()).digest()
        tasks.append({"row": row, "path": str(path), "dataset": str(dataset), "structure": structure,
                      "draws": draws, "seed": [int.from_bytes(digest[:8], "little")], "backend": "cpu"})
    tasks = tasks[:limit] if limit else tasks
    print(f"OEH-B {name}: {len(tasks)} pending, {workers} workers", flush=True)
    tick = time.time()
    R._repo_first()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(R.oeh_b_group, tasks), 1):
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  {time.time() - tick:.0f}s", flush=True)


if __name__ == "__main__":
    main()
