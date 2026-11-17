"""Data-only OEH-H (posterior-mean model) for a regenerated cohort.

Same function as the sealed ``meta`` phase (``horizon_group`` with oracle=False).
Usage: python driver/run_horizon.py <cohort> <workers>
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from common import COHORTS, DATA, OUT, load_json, use_workspace


def _one(task):
    use_workspace()
    from scripts.analysis import run_oeh_final_120 as R
    return R.horizon_group(task)


def main() -> None:
    name, workers = sys.argv[1], int(sys.argv[2])
    use_workspace()
    from scripts.analysis import run_oeh_final_120 as R

    out = OUT / "horizon" / f"{name}.csv"
    if out.exists():
        print("exists", out)
        return
    structure = load_json(COHORTS["c7"] / "protocol.json")["oeh_final"]["structure"]
    tasks = [{"row": row, "dataset": str(DATA / name), "structure": structure, "oracle": False}
             for row in R.pilot_rows(DATA / name)]
    tick = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(_one, tasks, chunksize=4)))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{name}: {len(frame)} groups, {time.time() - tick:.0f}s")


if __name__ == "__main__":
    main()
