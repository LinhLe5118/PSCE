"""Step 2: refit the heads on the meta-training data and regenerate the forecasts of the three final tests.

The heads are fitted on the 720 meta-training MDPs in their sealed row order (the boosted heads depend on it) with
the frozen specification of OEH; the inputs of each test are its sealed pilot-only table ``query_preoutcome.csv.gz``.
Regenerated: OEH, the full head, the robust head, B2, and OEH without each evidence block (Table 16).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce.common import ART, BLOCKS, KEYS, P, TESTS, write_json

TOLERANCE = 1e-9


def regenerate(workers: int, work: Path, log) -> pd.DataFrame:
    from scripts.analysis import oeh_pipeline as O
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E

    tick = time.perf_counter()
    spec = O.load_spec()
    train = O.training_panel(spec)
    queries = []
    for test, t in TESTS.items():
        for panel in t["panels"]:
            q = pd.read_csv(ART / t["folder"] / panel / "query_preoutcome.csv.gz")
            queries.append(q.assign(panel="primary", test=test, source_panel=panel))
    common = set.intersection(*(set(q.columns) for q in queries))
    query = pd.concat([q[[c for c in queries[0].columns if c in common]] for q in queries], ignore_index=True)
    log(f"  meta-training {train.shape[0]} cases ({train.mdp_id.nunique()} MDPs); "
        f"final tests {query.shape[0]} cases ({query.groupby(['test', 'source_panel']).mdp_id.nunique().sum()} MDPs); "
        f"{time.perf_counter() - tick:.0f} s")

    sets = E.forecast_sets()
    extra = {"b2": (sets["b2_matched"], False)}
    for name, field_set in BLOCKS.items():
        extra[name] = (sets[field_set], False)
        extra[name + "_finite"] = (sets[field_set], True)
    log(f"  fitting {3 + len(extra)} heads with {workers} worker processes (about two minutes) ...")
    tables = O.fit_heads(train, query, workers, work, extra=extra)
    log(f"  heads fitted, {time.perf_counter() - tick:.0f} s")

    oeh, distance = O.final_law(tables, train, query, spec)
    laws = {"oeh": oeh, "full": tables["full"], "robust": tables["robust"], "b2": tables["b2"]}
    for name in BLOCKS:
        laws["oeh_" + name] = O.final_law(tables, train, query, spec, full=name, full_finite=name + "_finite")[0]
    frames = []
    for method, law in laws.items():
        frame = query[["test", "source_panel", *KEYS]].copy()
        frame["method"] = method
        frame[P] = law
        frame["distance"] = distance
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if not np.allclose(out[P].sum(1), 1, atol=1e-9) or (out[P].to_numpy() < 0).any():
        raise RuntimeError("a regenerated forecast is not a probability vector")
    return out


def compare(forecasts: pd.DataFrame) -> dict:
    """Largest absolute difference between each regenerated forecast and its sealed counterpart."""
    report = {}
    for test, t in TESTS.items():
        pairs = {kind: code for kind, code in t["codes"].items()}
        if test != "Test 1":
            pairs["oeh_no_planning_risk"] = "oeh_without_horizon"   # sealed as a descriptive variant of Tests 2 and 3
        for panel in t["panels"]:
            sealed = pd.read_csv(ART / t["folder"] / panel / "predictions_preoutcome.csv")
            mine = forecasts[(forecasts.test == test) & (forecasts.source_panel == panel)]
            for kind, code in pairs.items():
                a = mine[mine.method == kind]
                b = sealed[sealed.method == code]
                both = a.merge(b[KEYS + P + ["distance"]], on=KEYS, suffixes=("", "_sealed"), validate="one_to_one")
                if len(both) != len(a) or len(both) != len(b):
                    raise RuntimeError(f"{test}/{panel}/{kind}: rows do not match the sealed forecast")
                entry = {"rows": len(both), "sealed_as": code,
                         "max_abs_diff": float(np.abs(both[P].to_numpy() - both[[p + "_sealed" for p in P]].to_numpy()).max())}
                if kind == "oeh":
                    entry["max_abs_diff_distance"] = float(np.abs(both.distance - both.distance_sealed).max())
                report[f"{test}/{panel}/{kind}"] = entry
    return report


def run(out: Path, workers: int, log) -> dict:
    target, check = out / "2_forecasts.csv.gz", out / "2_forecast_check.json"
    if target.exists() and check.exists():
        log("  already done, skipped")
        forecasts = pd.read_csv(target)
    else:
        forecasts = regenerate(workers, out / "_work", log)
        forecasts.to_csv(target, index=False)
    report = compare(forecasts)
    worst = max(v["max_abs_diff"] for v in report.values())
    write_json(check, {"tolerance": TOLERANCE, "max_abs_diff_any": worst, "by_panel_and_forecast": report})
    for test in TESTS:
        rows = {k: v for k, v in report.items() if k.startswith(test + "/")}
        log(f"  {test}: {len(rows)} sealed forecasts compared, max |difference| {max(v['max_abs_diff'] for v in rows.values()):.1e}")
    if worst > TOLERANCE:
        raise SystemExit(f"STOP: a regenerated forecast differs from the sealed one by {worst:.2e} (see {check.name})")
    return report
