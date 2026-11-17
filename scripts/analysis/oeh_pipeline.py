"""OEH (final method): pilot-only forecast of a learner's data requirement, single entry point.

From a dataset bundle in the project's format (pilot streams of every MDP-behavior
group; training streams are never read) the pipeline computes the pilot-only inputs
and the final OEH law over the seven requirement classes for BC, CFQI and IQL:

1. pilot inputs: B2 statistics, OEH-B posterior simulation, OEH-H planning risk,
   pilot learning curve (LCE), the cross-fitted pilot diagnostics and concentrability
   quantities (``pilot_inputs``);
2. three heads fitted by ``oeh_final_forecast.matched_forecast`` on the 720-MDP
   meta-training panel: the full 108-field head, the same fields refit on finite rows
   only, and the B2 + diagnostics head (``fit_heads``);
3. the shift-gated combination of ``oeh_v2_forecast.v2_law`` with the frozen
   parameters of ``oeh_v2_spec.json`` (``final_law``).

Command line::

    python -m scripts.analysis.oeh_pipeline --dataset <bundle> --out <dir> \
        [--kernels <dir> --kind colosseum|spibb] [--workers 8]

writes ``query.csv.gz`` (pilot-only inputs) and ``oeh_forecast.csv`` (one row per
MDP, behavior and learner with p_0..p_6 = P(E = x1, x2, x5, x10, x20, x50, >x50)
and the shift distance d). ``--query`` skips step 1 for an existing input table.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "artifacts/week8/oeh_v2_development_2026-10-14/oeh_v2_spec.json"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
CLASSES = ("x1", "x2", "x5", "x10", "x20", "x50", ">x50")


def load_spec(path: Path = SPEC) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("status") != "frozen" or spec.get("base_head") != "oeh_ensemble":
        raise RuntimeError("the OEH specification is not frozen on the full head")
    return spec


def pilot_inputs(dataset: Path, target: Path, workers: int, kernels: Path | None = None, kind: str = "internal",
                 namespace: str = "oeh-pipeline") -> pd.DataFrame:
    """Pilot-only input table (one row per MDP, behavior and learner); intermediate files go to ``target``."""
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import run_spibb_panel_20261014 as S
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields
    from scripts.analysis.run_unseen_validation import archive_imports

    archive_imports()
    R._repo_first()
    dataset, target = Path(dataset), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    seven = E.load_json(E.COHORT7 / "protocol.json")["oeh_final"]
    tick = {}
    t0 = time.perf_counter()
    features = E.pilot_features(dataset, target / "pilot_features.csv", workers)
    tick["pilot_features"] = time.perf_counter() - t0
    source = str(kernels) if kernels is not None else None
    out = target / "published.csv"
    t0 = time.perf_counter()
    if kind == "spibb":
        if not out.exists():
            manifest = E.load_json(dataset / "manifest.json")
            inst = {i["instance_id"]: i for i in manifest["instances"]}
            rows = sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot" and int(r.get("size_multiplier", -1)) == 10),
                          key=lambda r: r["instance_id"])
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pd.DataFrame(list(pool.map(S._published_task, [{"instance": inst[r["instance_id"]], "row": r, "dataset": str(dataset),
                                                                  "kernels": source} for r in rows], chunksize=4))).to_csv(out, index=False)
        pub = pd.read_csv(out)
    else:
        pub = E.published(dataset, out, workers, source)
    tick["published"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R.run_oeh_b(dataset, target / "oeh_b_groups", seven["structure"], seven["requirement_draws"], namespace, workers, "cpu")
    tick["oeh_b"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R.run_horizon(dataset, target / "horizon.csv", seven["structure"], workers, oracle=False)
    tick["horizon"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    curve = R.run_curve(dataset, target / "curve.csv", seven["structure"], seven["requirement_draws"], workers)
    tick["curve"] = time.perf_counter() - t0
    query = features.merge(F.oeh_feature_rows(target / "oeh_b_groups", target / "horizon.csv"), on=KEYS, validate="one_to_one")
    query = query.merge(R.curve_features(curve), on=KEYS, validate="one_to_one")
    query = query.merge(pub, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = add_fields(with_context(query))
    query["panel"] = "primary"
    if query[KEYS].duplicated().any():
        raise RuntimeError("duplicate query rows")
    (target / "input_runtime_seconds.json").write_text(json.dumps(tick, indent=2), encoding="utf-8")
    return query.sort_values(KEYS).reset_index(drop=True)


def training_panel(spec: dict) -> pd.DataFrame:
    from scripts.analysis.develop_cohort8_candidates_20261014 import meta_panel
    if spec.get("extra_training_cohorts"):
        raise RuntimeError("this entry point serves the frozen specification without extra cohorts")
    return meta_panel(sort=False)  # sealed row order: the boosted heads are not invariant to it


def fit_heads(train: pd.DataFrame, query: pd.DataFrame, workers: int, work: Path, extra: dict | None = None) -> dict:
    """The three heads of OEH (plus optional extra field sets: {name: (fields, finite_only)})."""
    from scripts.analysis import run_oeh_eighth_cohort_20261014 as E

    sets = E.forecast_sets()
    jobs = {"full": (sets["oeh_final"], False), "full_finite": (sets["oeh_final"], True),
            "robust": (sets["prior_oeh_matched"], False), **(extra or {})}
    work.mkdir(parents=True, exist_ok=True)
    tpath, qpath = work / "_train.pkl", work / "_query.pkl"
    train.to_pickle(tpath)
    query.to_pickle(qpath)
    tasks = [{"method": ("cond_full" if finite else (name if name.startswith("xgboost") else "matched")), "label": name,
              "fields": list(fields), "train": str(tpath), "query": str(qpath), "finite": finite}
             for name, (fields, finite) in sorted(jobs.items(), key=lambda kv: -len(kv[1][0]))]
    tables, tuning = {}, {}
    try:
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
            for task, (_, probability, weights, _) in zip(tasks, pool.map(E._forecast_task, tasks)):
                tables[task["label"]], tuning[task["label"]] = probability, weights
    finally:
        tpath.unlink(missing_ok=True)
        qpath.unlink(missing_ok=True)
    tables["_tuning"] = tuning
    return tables


def final_law(tables: dict, train: pd.DataFrame, query: pd.DataFrame, spec: dict,
              full: str = "full", full_finite: str = "full_finite") -> tuple[np.ndarray, np.ndarray]:
    from scripts.analysis import oeh_v2_forecast as V

    params = spec["parameters"]
    d = V.knn_distance(train, query)
    prior = V.class_prior(train)
    pi = np.stack([prior[a] for a in query.algorithm])
    own = tables[full_finite] if params["q_oeh"] == "finite_only_head" else tables[full]
    if params["q_robust"] != "legacy_finite":
        raise RuntimeError("the frozen specification uses the finite part of the B2 + diagnostics head")
    law = V.v2_law(tables[full], own, tables["robust"], d, pi,
                   lambda_max=params["lambda_max"], eps_max=params["eps_max"], d0=params["d0"])
    return law, d


def forecast(query: pd.DataFrame, workers: int, work: Path, spec_path: Path = SPEC) -> pd.DataFrame:
    spec = load_spec(spec_path)
    train = training_panel(spec)
    tables = fit_heads(train, query, workers, work)
    law, d = final_law(tables, train, query, spec)
    out = query[KEYS].copy()
    out[P] = law
    out["distance"] = d
    out["p_censored_full_head"] = tables["full"][:, 6]
    if not np.allclose(law.sum(1), 1, atol=1e-9) or (law < 0).any():
        raise RuntimeError("invalid forecast")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="dataset bundle (manifest.json with pilot streams)")
    parser.add_argument("--query", type=Path, help="existing pilot-input table (skips the input step)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, help="kernel directory for external MDPs (npz with T, R, rho)")
    parser.add_argument("--kind", choices=("internal", "colosseum", "spibb"), default="internal")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.query is not None:
        query = pd.read_csv(args.query)
    elif args.dataset is not None:
        query = pilot_inputs(args.dataset, args.out, args.workers, args.kernels, args.kind)
        query.to_csv(args.out / "query.csv.gz", index=False)
    else:
        parser.error("--dataset or --query is required")
    result = forecast(query, args.workers, args.out)
    result.to_csv(args.out / "oeh_forecast.csv", index=False)
    print(f"wrote {len(result)} forecasts to {args.out / 'oeh_forecast.csv'}")


if __name__ == "__main__":
    main()
