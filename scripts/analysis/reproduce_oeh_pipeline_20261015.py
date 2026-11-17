"""The single entry point ``oeh_pipeline`` reproduces the sealed OEH v2 forecasts of all four test panels."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis import oeh_pipeline as O

ROOT = O.ROOT
SEALED = ROOT / "artifacts/oeh_v2_sealed_test_2026-10-14"
OUT = ROOT / "artifacts/week8/oeh_pipeline_reproduction_2026-10-15"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for panel in ("forest", "spibb2", "colosseum2029", "internal"):
        query = pd.read_csv(SEALED / panel / "query_preoutcome.csv.gz")
        mine = O.forecast(query, 6, OUT / "_work")
        sealed = pd.read_csv(SEALED / panel / "predictions_preoutcome.csv")
        sealed = sealed[sealed.method == "v2"][O.KEYS + O.P + ["distance"]]
        joined = mine.merge(sealed, on=O.KEYS, suffixes=("", "_sealed"), validate="one_to_one")
        report[panel] = {"rows": len(joined),
                         "max_abs_diff_law": float(np.abs(joined[O.P].to_numpy() - joined[[f"{p}_sealed" for p in O.P]].to_numpy()).max()),
                         "max_abs_diff_distance": float(np.abs(joined.distance - joined.distance_sealed).max())}
        print(panel, report[panel], flush=True)
    (OUT / "reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if max(v["max_abs_diff_law"] for v in report.values()) > 1e-9:
        raise RuntimeError("the entry point does not reproduce the sealed forecasts")


if __name__ == "__main__":
    main()
