"""Command line of the reproduction; see ``scripts/reproduce/__init__.py`` and the README.

    python -m scripts.reproduce --out reproduce_out [--workers 8] [--steps seals,forecast,registered,tables]
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PINS = {"numpy": "2.5.1", "pandas": "2.3.3", "scipy": "1.18.0", "sklearn": "1.9.1", "xgboost": "3.3.0"}
STEPS = ("seals", "forecast", "registered", "tables")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # a Windows console may not be UTF-8
        stream.reconfigure(encoding="utf-8", errors="replace")
    # One BLAS thread per process: multi-threaded BLAS inside the worker processes made fits thousands of times slower.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("reproduce_out"), help="output folder (default: reproduce_out)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of " + ", ".join(STEPS))
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    from scripts.reproduce import common as C

    out = args.out.resolve()
    if C.ART in out.parents or out == C.ART:
        parser.error("the output folder must not be inside artifacts/")
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "run_log.txt").open("a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    import importlib
    versions = {name: importlib.import_module(name).__version__ for name in PINS}
    log(f"=== reproduction started {datetime.now(timezone.utc).isoformat()} in {out}")
    log(f"Python {platform.python_version()} on {platform.platform()}; " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    off = {k: v for k, v in versions.items() if v != PINS[k]}
    if off:
        log(f"WARNING: versions differ from the pins {PINS}: {off}. Forecasts may then differ from the sealed ones.")

    timings, forecasts = {}, None
    for number, step in enumerate(STEPS, 1):
        if step not in steps:
            continue
        log(f"--- step {number}/4: {step}")
        tick = time.perf_counter()
        if step == "seals":
            from scripts.reproduce import seals
            seals.run(out, log)
        elif step == "forecast":
            from scripts.reproduce import forecast
            forecast.run(out, args.workers, log)
        elif step == "registered":
            import pandas as pd
            from scripts.reproduce import registered
            if not (out / "2_forecasts.csv.gz").exists():
                raise SystemExit("STOP: run the forecast step first")
            forecasts = forecasts if forecasts is not None else pd.read_csv(out / "2_forecasts.csv.gz")
            registered.run(out, forecasts, log)
        elif step == "tables":
            from scripts.reproduce import tables
            if not (out / "3_registered").exists():
                raise SystemExit("STOP: run the forecast and registered steps first")
            result = tables.run(out, log)
            if result["mismatches"]:
                log("STOP: some printed numbers were not reproduced (see 4_tables.md)")
        timings[step] = round(time.perf_counter() - tick, 1)
        log(f"    done in {timings[step]} s")

    manifest = {
        "status": "reproduction check of the final results; post hoc (all labels were opened before); no learner is run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.executable).name, "-m", "scripts.reproduce", *sys.argv[1:]]),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), **versions},
        "seconds": timings,
        "sources_sha256": {p.relative_to(C.ROOT).as_posix(): C.sha256(p) for p in sorted((C.ROOT / "scripts/reproduce").glob("*"))
                           if p.is_file()},
        "outputs_sha256": {p.relative_to(out).as_posix(): C.sha256(p) for p in sorted(out.glob("*"))
                           if p.is_file() and p.name not in ("manifest.json", "run_log.txt")},
        "inputs": "the committed artifacts of the three final tests, the meta-training artifacts read by "
                  "scripts.analysis.oeh_pipeline.training_panel, and the TabPFN forecasts; step 1 checks their seals",
    }
    C.write_json(out / "manifest.json", manifest)
    log(f"=== finished; results in {out} (start with 4_tables.md)")
    log_file.close()


if __name__ == "__main__":
    main()
