"""Post hoc gated ablation of OEH on the three final tests (no learner runs).

Refits the sealed heads on the unchanged meta-training panel, first reproduces the sealed
OEH forecast as a check, then forms the full OEH law (shift gate included) with one block
removed from the full and finite-only heads. The robust head of the gate is kept as sealed.
Reads the sealed artifacts and never modifies them; writes to ./out/gated_ablation.

Needs the pinned forecasting environment (requirements-difficulty.txt, xgboost 3.3.0):

    python scripts/analysis/posthoc_20261028/gated_ablation.py
    python scripts/analysis/posthoc_20261028/score_gated_ablation.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
OUT = Path(__file__).resolve().parent / "out" / "gated_ablation"
P = [f"p_{i}" for i in range(7)]
KEYS = ["mdp_id", "policy_family", "algorithm"]
TESTS = {
    "Test 1": ("oeh_v2_sealed_test_2026-10-14", ["internal", "colosseum2029", "spibb2", "forest"], "v2"),
    "Test 2": ("oeh_fifth_panel_2026-10-15", ["frozenlake", "gambler"], "oeh"),
    "Test 3": ("oeh_panel6_2026-10-15", ["mdpp", "labmaze"], "oeh"),
}
BLOCKS = {"no_P": "oeh_without_direct_oehb", "no_H": "oeh_without_oehh_risk", "no_L": "oeh_without_legacy_diagnostics"}


def main() -> None:
    from scripts.analysis import oeh_pipeline as O
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E

    OUT.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    spec = O.load_spec()
    train = O.training_panel(spec)
    queries = []
    for test, (folder, panels, _) in TESTS.items():
        for panel in panels:
            q = pd.read_csv(ROOT / "artifacts" / folder / panel / "query_preoutcome.csv.gz")
            q["panel"] = "primary"
            queries.append(q.assign(test=test, source_panel=panel))
    columns = set.intersection(*(set(q.columns) for q in queries))
    query = pd.concat([q[[c for c in queries[0].columns if c in columns]] for q in queries], ignore_index=True)
    print(f"train {train.shape}, query {query.shape}, {time.perf_counter() - tick:.0f}s", flush=True)

    sets = E.forecast_sets()
    extra = {}
    for name, field_set in BLOCKS.items():
        extra[name] = (sets[field_set], False)
        extra[name + "_finite"] = (sets[field_set], True)
    tables = O.fit_heads(train, query, workers=9, work=OUT / "work", extra=extra)
    print(f"heads fitted, {time.perf_counter() - tick:.0f}s", flush=True)

    laws = {"OEH": O.final_law(tables, train, query, spec)[0]}
    for name in BLOCKS:
        laws[name] = O.final_law(tables, train, query, spec, full=name, full_finite=name + "_finite")[0]

    frames = []
    for method, law in laws.items():
        frame = query[KEYS + ["test", "source_panel"]].copy()
        frame["method"] = method
        frame[P] = law
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "predictions.csv.gz", index=False)

    # Reproduction check against the sealed OEH forecasts.
    for test, (folder, panels, sealed_name) in TESTS.items():
        for panel in panels:
            sealed = pd.read_csv(ROOT / "artifacts" / folder / panel / "predictions_preoutcome.csv")
            sealed = sealed[sealed.method == sealed_name]
            mine = frames[0][(frames[0].test == test) & (frames[0].source_panel == panel)]
            both = mine.merge(sealed[KEYS + P], on=KEYS, suffixes=("", "_sealed"), validate="one_to_one")
            diff = np.abs(both[P].to_numpy() - both[[p + "_sealed" for p in P]].to_numpy()).max()
            print(f"reproduction {test}/{panel}: {len(both)} rows, max |diff| = {diff:.2e}", flush=True)
    print(f"done, {time.perf_counter() - tick:.0f}s", flush=True)


if __name__ == "__main__":
    main()
