"""Step 3: rerun the registered analysis of each final test on a copy of its files.

Each test folder is copied without its simulation checkpoints (``oeh_b_groups``). In the copy, the sealed rows of
OEH, the full head, the robust head and B2 are replaced by the forecasts regenerated in step 2, and the test's own
``analyze`` runs with its output folder pointed at the copy. Every number it writes is then compared with the
committed file. The committed artifacts are only read.
"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce.common import ART, KEYS, P, TESTS, load_json, write_json

TOLERANCE = 1e-9
OUTPUTS = ("result.json", "summary.csv", "decision_regret.csv", "murphy.csv", "by_learner.csv", "shapley.csv",
           "extended_endpoint.csv", "gain_by_distance.csv", "label_variant_tau099.csv", "scored.csv.gz")


def _numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "completed_at_utc":
                yield from _numbers(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}/{i}")
    else:
        yield path, obj


def _diff_json(a: Path, b: Path) -> float:
    x, y = dict(_numbers(load_json(a))), dict(_numbers(load_json(b)))
    if x.keys() != y.keys():
        return float("inf")
    worst = 0.0
    for k in x:
        if isinstance(x[k], (int, float)) and not isinstance(x[k], bool):
            worst = max(worst, abs(float(x[k]) - float(y[k])))
        elif x[k] != y[k]:
            return float("inf")
    return worst


def _diff_csv(a: Path, b: Path) -> float:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return float("inf")
    worst = 0.0
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            same_nan = np.isnan(u) == np.isnan(v)
            if not same_nan.all():
                return float("inf")
            if (~np.isnan(u)).any():
                worst = max(worst, float(np.nanmax(np.abs(u - v))))
        elif not x[c].astype(str).equals(y[c].astype(str)):
            return float("inf")
    return worst


def prepare_copy(test: str, forecasts: pd.DataFrame, target: Path) -> None:
    t = TESTS[test]
    shutil.copytree(ART / t["folder"], target, ignore=shutil.ignore_patterns("oeh_b_groups"))
    mine = forecasts[forecasts.test == test]
    for panel in t["panels"]:
        path = target / panel / "predictions_preoutcome.csv"
        sealed = pd.read_csv(path)
        for kind, code in t["codes"].items():
            new = mine[(mine.source_panel == panel) & (mine.method == kind)].set_index(KEYS)
            rows = sealed.method == code
            index = pd.MultiIndex.from_frame(sealed.loc[rows, KEYS])
            sealed.loc[rows, P] = new.loc[index, P].to_numpy()
        sealed.to_csv(path, index=False)


def run(out: Path, forecasts: pd.DataFrame, log) -> dict:
    report = {}
    for test, t in TESTS.items():
        target = out / "3_registered" / t["folder"]
        done = target / "_analyzed"
        if not done.exists():
            if target.exists():
                shutil.rmtree(target)
            prepare_copy(test, forecasts, target)
            module = importlib.import_module(t["runner"])
            original = module.OUTPUT
            module.OUTPUT = target
            try:
                module.analyze()
            finally:
                module.OUTPUT = original
            done.write_text("analysis rerun on this copy\n", encoding="utf-8")
        else:
            log(f"  {test}: already done, skipped")
        diffs = {}
        for name in OUTPUTS:
            a, b = ART / t["folder"] / name, target / name
            if a.exists():
                diffs[name] = _diff_json(a, b) if name.endswith(".json") else _diff_csv(a, b)
        result = load_json(target / "result.json")
        report[test] = {"max_abs_diff": diffs, "result": {k: v for k, v in result.items() if k != "events"}}
        worst = max(diffs.values())
        log(f"  {test}: {len(diffs)} outputs compared with the committed ones, max |difference| {worst:.1e}")
    write_json(out / "3_registered_check.json", {"tolerance": TOLERANCE, "tests": report})
    bad = {f"{test}/{name}": d for test, r in report.items() for name, d in r["max_abs_diff"].items() if d > TOLERANCE}
    if bad:
        raise SystemExit(f"STOP: registered outputs differ from the committed ones: {bad}")
    return report
