"""Eighth cohort: sealed replication with fixed-sequence (gatekeeping) contrasts.

Two separately reported panels share one protocol, one forecaster and one gate:

* ``internal``  168 new stochastic MDPs from the registered recipe (two fresh
  kernels per primary template of the seventh cohort; no deterministic
  controls), five behaviors, pilot x10, ten seeds, BC/CFQI/IQL;
* ``colosseum`` 72 new Colosseum kernels (authors' sampler, seed 2028, 18 per
  family, every kernel of the two earlier Colosseum runs excluded).

The forecaster is the sealed 720-MDP meta-training fit of cohort 7 (no refit
choice). OEH-S and OEH-2T and the new LCE/CB comparators were fixed on that
panel by ``develop_cohort8_candidates_20261014`` before this protocol.

Phases (each refuses to run out of order):
``freeze``                 write protocol.json (+ sha256) before any buffer;
``prepare-internal``       generate the internal dataset, pilot-only inputs,
                           forecasts, and seal them;
``prepare-colosseum``      same for a Colosseum bundle built by the driver;
``evaluate``               learner jobs (only after the panel's seal);
``analyze``                labels, scores, gatekeeping verdicts, descriptive
                           analyses.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/oeh_eighth_cohort_2026-10-14"
DATASET = ROOT / "datasets/oeh_eighth_cohort_2026-10-14"
NAMESPACE = "oeh-eighth-gatekeeping-v1-20261014"
DEVELOPMENT = ROOT / "artifacts/week8/cohort8_development_2026-10-14"
COHORT7 = ROOT / "artifacts/oeh_confirmatory_unseen_120_2026-10-04"
COLOSSEUM_CONFIRM = ROOT / "artifacts/week8/colosseum_confirm_2026-10-14"
KEYS = ["mdp_id", "policy_family", "algorithm"]
P = [f"p_{i}" for i in range(7)]
BUDGETS = (1, 2, 5, 10, 20, 50)
PRIMARY_COUNTS = {"uniform_garnet": 30, "forward_layered": 36, "bottleneck": 18}
REPLICAS = 2
INTERNAL_MDPS = REPLICAS * sum(PRIMARY_COUNTS.values())  # 168
COLOSSEUM_MDPS = 72
BOOTSTRAP = {"draws": 10_000, "seed": 20261015,
             "rule": "max-t with the original standard error (run_oeh_120_unseen.analyze)"}
STEPS = {
    "step1": {"candidate": "oeh_final", "baselines": ["b2_matched", "xgboost_cf_18"]},
    "step2": {"candidate": "oeh_final", "baselines": ["b2_plus_eri", "prior_oeh_matched", "lce_raw",
                                                      "b2_plus_lce", "b2_plus_cb", "cb_only"]},
    "step3": {"candidates": ["oeh_s", "oeh_2t"], "baseline": "oeh_final"},
}
DECISION = {"tau": [0.5, 0.8, 0.9], "lambda": [50, 100], "headline": {"tau": 0.8, "lambda": 100},
            "methods": ["oeh_final", "b2_matched", "xgboost_cf_18", "b2_plus_eri", "prior_oeh_matched",
                        "b2_plus_lce", "oeh_s", "oeh_2t"],
            "fixed_rules": ["always_x10", "always_x50", "always_skip"]}
FACTORIAL = ["b2_matched", "b2_plus_P", "b2_plus_H", "prior_oeh_matched", "oeh_without_legacy_diagnostics",
             "oeh_without_oehh_risk", "oeh_without_direct_oehb", "oeh_final"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def rel(path) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def panel_dir(panel: str) -> Path:
    return OUTPUT / panel


def protocol() -> dict:
    path = OUTPUT / "protocol.json"
    if sha256(path) != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Protocol changed after freeze")
    return load_json(path)


# --------------------------------------------------------------------------- #
# Phase: freeze
# --------------------------------------------------------------------------- #

def _fresh_specs() -> tuple[list, dict]:
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import run_oeh_120_unseen as base

    C.configure()
    seen = set(R.seen_kernels())
    seen |= set(load_json(COHORT7 / "actual_kernel_audit_preoutcome.json")["actual_kernel_fingerprints"].values())
    seen |= set(load_json(COHORT7 / "protocol.json")["kernel_fingerprints"].values())
    used = set()
    for spec in R.OPENED.values():
        used |= {int(row["seed"]) for row in load_json(spec["dataset"] / "manifest.json")["instances"]}
    used |= {int(item["seed"]) for item in load_json(COHORT7 / "protocol.json")["requested_specs"]}
    templates = [row for row in benchmark_specs() if row.split == "evaluation"]
    specs, kernels = [], {}
    for family, count in PRIMARY_COUNTS.items():
        selected = sorted((row for row in templates if row.family == family), key=lambda row: row.ordinal)
        if len(selected) != count:
            raise RuntimeError("Registered recipe counts changed")
        for replica in range(REPLICAS):
            for ordinal, template in enumerate(selected):
                instance_id = f"c8_{family}_{replica * count + ordinal:03d}"
                for retry in range(5000):
                    digest = hashlib.sha256(f"{NAMESPACE}|{template.instance_id}|{replica}|{retry}".encode()).digest()
                    seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
                    if seed in used:
                        continue
                    spec = replace(template, instance_id=instance_id, seed=seed)
                    kernel = base.fingerprint(make_benchmark_mdp(spec))
                    if kernel not in seen:
                        break
                else:
                    raise RuntimeError(f"No fresh kernel for {instance_id}")
                used.add(seed)
                seen.add(kernel)
                specs.append(spec)
                kernels[instance_id] = kernel
    if len(specs) != INTERNAL_MDPS:
        raise RuntimeError("Incomplete internal panel")
    return specs, kernels


def freeze() -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("Eighth-cohort namespace already exists")
    selection = load_json(DEVELOPMENT / "selection.json")
    if selection.get("status") != "frozen":
        raise RuntimeError("Development selection is not frozen")
    specs, kernels = _fresh_specs()
    sources = [Path(__file__), ROOT / "scripts/analysis/develop_cohort8_candidates_20261014.py",
               ROOT / "scripts/analysis/colosseum_posthoc_20261014.py", ROOT / "scripts/analysis/oeh_final_forecast.py",
               ROOT / "scripts/analysis/run_oeh_final_120.py", ROOT / "benchmarks/colosseum/driver/colosseum_export_fast.py",
               ROOT / "benchmarks/colosseum/driver/build_colosseum.py", DEVELOPMENT / "selection.json",
               DEVELOPMENT / "rules.json", DEVELOPMENT / "oof_predictions.csv.gz"]
    doc = {
        "schema": "oeh-eighth-gatekeeping-v1", "namespace": NAMESPACE, "frozen_at_utc": now(),
        "status": "frozen_before_dataset_buffers_or_learner_outcomes",
        "panels": {
            "internal": {"mdps": INTERNAL_MDPS, "families": {k: REPLICAS * v for k, v in PRIMARY_COUNTS.items()},
                         "recipe": "two fresh kernels per primary evaluation template of the registered recipe "
                                   "(seed retry hash with the namespace); kernels distinct from every opened "
                                   "cohort and from cohort 7; no deterministic controls",
                         "requested_specs": [spec.to_dict() for spec in specs], "kernel_fingerprints": kernels},
            "colosseum": {"mdps": COLOSSEUM_MDPS, "source": "Colosseum (Conserva & Rauber 2022), authors' "
                          "sample_mdp_parameters, repo 981b344, continuous setting, mean rewards",
                          "families": ["riverswim", "deepsea", "frozenlake", "simplegrid"],
                          "sampler_seed": 2028, "draws_per_family": 600, "export_cap_per_family": 30,
                          "filters": ["stochastic kernel", "16 <= |S||A| <= 256", "project dynamic-range filter"],
                          "exclusion": "every kernel of the exploratory (seed 2026) and confirmatory (seed 2027) runs",
                          "selection": "first 18 per family passing the filter, sampler order", "data_seed_base": 9_800_000,
                          "export": "benchmarks/colosseum/driver/colosseum_export_fast.py"},
        },
        "data": {"behaviors": 5, "pilot_multipliers": [2, 5, 10], "budgets": list(BUDGETS),
                 "training_seed_indices": list(range(10)), "core_learners": ["behavior_cloning_tabular",
                                                                             "cfqi_tabular", "iql_tabular"],
                 "target": "first complete suffix with >=8/10 seeds normalized return >=0.95; else >x50",
                 "expectile": "team's documented IQL expectile fallback, counted, used only where the archived code raises"},
        "forecaster": {"meta_training": "the sealed seven-cohort panel of cohort 7 (720 MDPs, 10,800 rows), unchanged",
                       "procedure": "oeh_final_forecast.matched_forecast for every fitted field set; XGBoost-CF as "
                                    "published_survival_120_*",
                       "oeh_final_fields": "protocol.json of cohort 7 (108 columns)"},
        "candidates": {
            "oeh_final": "OEH current (108 columns)",
            "oeh_s": {"rule": "p' = (1-eps) * normalize(p^(1/T)) + eps * pi_learner", **selection["oeh_s"]["chosen"],
                      "pi_learner": selection["oeh_s"]["class_prior"]},
            "oeh_2t": {"rule": "w * [(1-p6) * P(k | finite; OEH 108 fields fitted on finite meta rows), p6] "
                               "+ (1-w) * OEH current; p6 from OEH current", "w": selection["oeh_2t"]["chosen"]["w"],
                       "conditional": selection["oeh_2t"]["chosen"]["conditional"]},
        },
        "comparators": {"b2_matched": "B2 18 generic statistics, matched head",
                        "xgboost_cf_18": "adapted published XGBoost-CF survival model on B2 statistics",
                        "b2_plus_eri": "B2 + ERI adaptation (Swazinna et al.)",
                        "prior_oeh_matched": "B2 + legacy OEH3/LCOEH4 and context (no posterior)",
                        "lce_raw": "learning-curve extrapolation from posterior prefix success, no meta-training",
                        "b2_plus_lce": "B2 + LCE quantities, matched head",
                        "b2_plus_cb": "B2 + concentrability / sample-complexity quantities (Li et al. 2022), matched head",
                        "cb_only": "concentrability quantities alone, matched head",
                        "xgboost_cf_108": "architecture control with the OEH inputs (descriptive)"},
        "gate": {
            "unit": "MDP; integrated Brier (six thresholds) and seven-class log loss (1e-12 floor), averaged within MDP",
            "rule": "each step: max-t simultaneous one-sided 95% lower bounds of the absolute improvement, all > 0",
            "bootstrap": BOOTSTRAP, "steps": STEPS,
            "sequence": "step 2 is tested only if step 1 is GO and step 3 only if step 2 is GO; the sequence is "
                        "applied to each panel separately; a closed step is reported descriptively and labelled",
            "step3_reading": "a candidate replaces OEH current on a panel only if both of its contrasts pass",
            "sample_size": "internal 168 MDPs exceeds the ~82 MDPs needed for the Colosseum log-loss effect at "
                           "critical 1.9; the Colosseum tier repeats the earlier design (72)",
        },
        "descriptive": {"decision_regret": DECISION, "factorial_blocks": FACTORIAL,
                        "by_learner_family": True, "finite_vs_censored": True, "zero_draw_groups": True},
        "development_sha256": {rel(path): sha256(path) for path in sources if path.exists()},
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"Frozen eighth-cohort protocol {sha256(OUTPUT / 'protocol.json')}", flush=True)


# --------------------------------------------------------------------------- #
# Pilot-only inputs
# --------------------------------------------------------------------------- #

def _generate_internal(doc: dict) -> dict:
    from scripts.analysis.run_unseen_validation import archive_imports
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import BenchmarkSpec, make_benchmark_mdp
    from scripts.analysis import run_oeh_120_unseen as base

    target = panel_dir("internal")
    if (target / "dataset_seal.json").exists():
        return load_json(DATASET / "manifest.json")
    specs = [BenchmarkSpec(**item) for item in doc["panels"]["internal"]["requested_specs"]]
    manifest = generator.generate_benchmark_bundle(
        DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None, horizon=100,
        run_mode="test", deviation_reason="Eighth cohort: sealed gatekeeping replication",
        pilot_size_multipliers=(2, 5, 10))
    if manifest["instance_count"] != INTERNAL_MDPS or manifest["physical_buffer_count"] != INTERNAL_MDPS * 55:
        raise RuntimeError("Incomplete internal dataset")
    actual = {item["instance_id"]: base.fingerprint(make_benchmark_mdp(type("Spec", (), item)()))
              for item in manifest["instances"]}
    requested = doc["panels"]["internal"]["kernel_fingerprints"]
    changed = {k: [requested[k], v] for k, v in actual.items() if requested[k] != v}
    from scripts.analysis import run_oeh_confirmatory_20261004 as C
    from scripts.analysis import run_oeh_final_120 as R
    C.configure()
    seen = set(R.seen_kernels()) | set(load_json(COHORT7 / "actual_kernel_audit_preoutcome.json")["actual_kernel_fingerprints"].values())
    if len(set(actual.values())) != INTERNAL_MDPS or set(actual.values()) & seen:
        raise RuntimeError("Resolved internal kernels are not all distinct and new")
    write_json(target / "dataset_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"), "sealed_at_utc": now(), "learner_outcomes_observed": 0,
        "resolved_kernel_changes": changed, "actual_kernel_fingerprints": actual,
        "sha256": {rel(p): sha256(p) for p in (DATASET / "manifest.json", DATASET / "data_only_index.json")}})
    return manifest


def _features_task(task):
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    sys.path.insert(0, str(ROOT / "benchmarks/colosseum/driver"))
    import expectile_fix
    expectile_fix.install()
    expectile_fix.CALLS["fallback"] = 0
    from evaluation import e1_pipeline
    rows = e1_pipeline._group_features(Path(task["dataset"]), tuple(task["group"]), task["prefixes"])
    for row in rows:
        row["_expectile_fallbacks"] = expectile_fix.CALLS["fallback"]
    return rows


def pilot_features(dataset: Path, out: Path, workers: int) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    manifest = load_json(dataset / "manifest.json")
    records = {row["data_only_dataset_id"]: row for row in manifest["datasets"]}
    grouped = defaultdict(dict)
    for item in load_json(dataset / "data_only_index.json")["datasets"]:
        if item.get("stream") == "pilot" and int(item.get("size_multiplier", -1)) in (2, 5, 10):
            row = records[item["dataset_id"]]
            grouped[(row["instance_id"], row["policy_family"])][int(row["size_multiplier"])] = item
    tasks = [{"dataset": str(dataset), "group": list(k), "prefixes": v} for k, v in sorted(grouped.items())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for result in pool.map(_features_task, tasks, chunksize=2) for row in result]
    frame = pd.DataFrame(rows)
    fallbacks = int((frame.groupby(["mdp_id", "policy_family"])._expectile_fallbacks.max() > 0).sum())
    frame = frame.drop(columns="_expectile_fallbacks")
    frame.to_csv(out, index=False)
    write_json(out.with_suffix(".json"), {"groups": len(tasks), "groups_with_expectile_fallback": fallbacks})
    return frame


def _published_task(task):
    if task.get("colosseum"):
        sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
        from scripts.analysis.run_unseen_validation import archive_imports
        archive_imports()
        import environments.benchmark_environments as B
        from environments.tabular_environments import TabularMDP
        B.make_benchmark_mdp = lambda spec: colosseum_mdp(task["colosseum"], spec.instance_id, TabularMDP)
    from scripts.analysis import oeh_final_published as Pub
    return Pub._task(task)


def published(dataset: Path, out: Path, workers: int, colosseum: str | None) -> pd.DataFrame:
    if out.exists():
        return pd.read_csv(out)
    manifest = load_json(dataset / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    rows = sorted((r for r in manifest["datasets"] if r.get("stream") == "pilot" and int(r.get("size_multiplier", -1)) == 10
                   and r.get("benchmark_split", "evaluation") == "evaluation"),
                  key=lambda r: (r["instance_id"], r["policy_family"]))
    tasks = [{"instance": instances[r["instance_id"]], "row": r, "dataset": str(dataset), "colosseum": colosseum}
             for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame(list(pool.map(_published_task, tasks, chunksize=4)))
    frame.to_csv(out, index=False)
    return frame


def colosseum_mdp(source: str, mdp_id: str, TabularMDP):
    with np.load(Path(source) / f"{mdp_id}.npz") as z:
        T, R, rho = z["T"], z["R"], z["rho"]
    T = T / T.sum(-1, keepdims=True)
    S, _ = R.shape
    mdp = TabularMDP(environment_id=mdp_id, P=T.astype(np.float64), R=np.repeat(R[:, :, None], S, axis=2).astype(np.float64),
                     initial_state_distribution=(rho / rho.sum()).astype(np.float64), terminal_states=(), gamma=0.95,
                     metadata={})
    mdp.validate()
    return mdp


def build_query(panel: str, dataset: Path, workers: int, colosseum: str | None = None) -> pd.DataFrame:
    """Pilot-only query rows: B2/legacy, OEH-B, OEH-H, curve, ERI, LCE/CB fields."""
    from scripts.analysis import run_oeh_final_120 as R
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.develop_cross_learner_context import with_context
    from scripts.analysis.develop_cohort8_candidates_20261014 import add_fields

    doc = protocol()
    target = panel_dir(panel)
    target.mkdir(parents=True, exist_ok=True)
    seven = load_json(COHORT7 / "protocol.json")["oeh_final"]
    structure, draws = seven["structure"], seven["requirement_draws"]
    tick = {}
    t0 = time.perf_counter()
    features = pilot_features(dataset, target / "pilot_features.csv", workers)
    tick["pilot_features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pub = published(dataset, target / "published_preoutcome.csv", workers, colosseum)
    tick["published"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R._repo_first()
    R.run_oeh_b(dataset, target / "oeh_b_groups", structure, draws, f"{NAMESPACE}|{panel}", workers, "cpu")
    tick["oeh_b"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    R.run_horizon(dataset, target / "horizon_preoutcome.csv", structure, workers, oracle=False)
    tick["horizon"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    curve = R.run_curve(dataset, target / "curve_preoutcome.csv", structure, draws, workers)
    tick["curve"] = time.perf_counter() - t0
    query = features.merge(F.oeh_feature_rows(target / "oeh_b_groups", target / "horizon_preoutcome.csv"),
                           on=KEYS, validate="one_to_one")
    query = query.merge(R.curve_features(curve), on=KEYS, validate="one_to_one")
    query = query.merge(pub, on=["mdp_id", "policy_family"], validate="many_to_one")
    query = add_fields(with_context(query))
    query["panel"] = "primary"
    expected = doc["panels"][panel]["mdps"] * 15
    if len(query) != expected or query[KEYS].duplicated().any():
        raise RuntimeError(f"Incomplete {panel} query: {len(query)} rows")
    write_json(target / "input_runtime_seconds.json", tick)
    return query.sort_values(KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #

_CACHE: dict = {}


def _forecast_task(task):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis import oeh_final_forecast as F
    from scripts.analysis.colosseum_posthoc_20261014 import xgboost_cf

    if task["train"] not in _CACHE:
        _CACHE[task["train"]] = pd.read_pickle(task["train"])
    if task["query"] not in _CACHE:
        _CACHE[task["query"]] = pd.read_pickle(task["query"])
    train, query = _CACHE[task["train"]], _CACHE[task["query"]]
    fields = tuple(task["fields"])
    tick = time.perf_counter()
    if task["method"].startswith("xgboost"):
        probability, weights = xgboost_cf(train, query, fields)
    elif task["method"] == "cond_full":
        probability, weights = F.matched_forecast(train[train.event < 6], query, fields)
        probability = probability[:, :6] / probability[:, :6].sum(1, keepdims=True)
        probability = np.column_stack([probability, np.zeros(len(probability))])
    else:
        probability, weights = F.matched_forecast(train, query, fields)
    return task["method"], probability, weights, time.perf_counter() - tick


def forecast_sets() -> dict[str, tuple[str, ...]]:
    from scripts.analysis.colosseum_posthoc_20261014 import factorial_arms
    from scripts.analysis.develop_cohort8_candidates_20261014 import field_sets

    sets = field_sets()
    arms = factorial_arms()
    for name in ("b2_plus_P", "b2_plus_H", "oeh_without_legacy_diagnostics", "oeh_without_oehh_risk",
                 "oeh_without_direct_oehb"):
        sets[name] = arms[name]
    if tuple(arms["oeh_full_108"]) != tuple(sets["oeh_final"]) or set(arms["b2_plus_L"]) != set(sets["prior_oeh_matched"]):
        raise RuntimeError("Arm definitions disagree with the sealed field sets")
    sets["xgboost_cf_18"] = sets["b2_matched"]
    sets["xgboost_cf_108"] = sets["oeh_final"]
    sets["cond_full"] = sets["oeh_final"]
    return sets


def check_sealed_tuning(tuning: dict, target: Path) -> None:
    """The ensemble weights depend only on the training panel: they must equal cohort 7's."""
    sealed = load_json(COHORT7 / "forecast_tuning_preoutcome.json")
    report = {}
    for method in ("oeh_final", "b2_matched", "b2_plus_eri", "prior_oeh_matched"):
        for stratum, value in tuning[method].items():
            ref = sealed[method][stratum]
            report[f"{method}/{stratum}"] = {
                "weights_max_abs_diff": float(np.abs(np.subtract(value["aft_direct_multi"], ref["aft_direct_multi"])).max()),
                "oof_logloss_abs_diff": abs(value["oof_logloss"] - ref["oof_logloss"])}
    write_json(target / "sealed_fit_check.json", report)
    bad = {k: v for k, v in report.items() if v["weights_max_abs_diff"] > 1e-6 or v["oof_logloss_abs_diff"] > 1e-8}
    if bad:
        raise RuntimeError(f"The fit differs from the sealed cohort-7 forecaster: {bad}")


def forecast(panel: str, query: pd.DataFrame, workers: int) -> pd.DataFrame:
    from scripts.analysis.develop_cohort8_candidates_20261014 import lce_law, meta_panel, soften

    doc = protocol()
    target = panel_dir(panel)
    train = meta_panel(sort=False)  # R.meta_training order: the sealed cohort-7 fit
    train_path, query_path = target / "_train.pkl", target / "_query.pkl"
    train.to_pickle(train_path)
    query.to_pickle(query_path)
    sets = forecast_sets()
    tasks = [{"method": m, "fields": list(f), "train": str(train_path), "query": str(query_path)}
             for m, f in sorted(sets.items(), key=lambda item: -len(item[1]))]
    tables, tuning, runtime = {}, {}, {}
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        for method, probability, weights, seconds in pool.map(_forecast_task, tasks):
            tables[method], tuning[method], runtime[method] = probability, weights, seconds
            print(f"forecast {panel}/{method}: {seconds:.0f}s", flush=True)
    train_path.unlink()
    query_path.unlink()
    check_sealed_tuning(tuning, target)
    oeh = tables["oeh_final"]
    cand = doc["candidates"]
    prior = np.stack([np.asarray(cand["oeh_s"]["pi_learner"][a]) for a in query.algorithm])
    tables["oeh_s"] = soften(oeh, prior, cand["oeh_s"]["T"], cand["oeh_s"]["eps"])
    two = np.column_stack([tables.pop("cond_full")[:, :6] * (1 - oeh[:, 6:7]), oeh[:, 6]])
    tables["oeh_2t"] = cand["oeh_2t"]["w"] * two + (1 - cand["oeh_2t"]["w"]) * oeh
    tables["lce_raw"] = lce_law(query)
    frames = []
    for method, probability in tables.items():
        frame = query[[*KEYS, "panel"]].copy()
        frame["method"] = method
        frame[P] = probability
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    p = predictions[P].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9):
        raise RuntimeError("Invalid seven-class forecast")
    write_json(target / "forecast_tuning_preoutcome.json", tuning)
    write_json(target / "forecast_runtime_seconds.json", runtime)
    return predictions


def seal(panel: str, query: pd.DataFrame, predictions: pd.DataFrame) -> None:
    target = panel_dir(panel)
    predictions.to_csv(target / "predictions_preoutcome.csv", index=False)
    query.to_csv(target / "query_preoutcome.csv.gz", index=False)
    names = ["pilot_features.csv", "published_preoutcome.csv", "horizon_preoutcome.csv", "curve_preoutcome.csv",
             "query_preoutcome.csv.gz", "predictions_preoutcome.csv", "forecast_tuning_preoutcome.json"]
    groups = sorted((target / "oeh_b_groups").glob("*.json"))
    import sklearn, xgboost
    write_json(target / "preoutcome_seal.json", {
        "panel": panel, "protocol_sha256": sha256(OUTPUT / "protocol.json"), "learner_outcomes_observed": 0,
        "sealed_at_utc": now(), "artifact_sha256": {name: sha256(target / name) for name in names},
        "oeh_b_groups": len(groups),
        "oeh_b_groups_sha256": hashlib.sha256("".join(sha256(p) for p in groups).encode()).hexdigest(),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}})
    print(f"Sealed {panel}: {len(predictions)} predictions", flush=True)


def prepare_internal(workers: int) -> None:
    target = panel_dir("internal")
    if (target / "preoutcome_seal.json").exists():
        raise FileExistsError("Internal panel already sealed")
    doc = protocol()
    _generate_internal(doc)
    from scripts.analysis import run_oeh_final_120 as R
    R._repo_first()
    query = build_query("internal", DATASET, workers)
    seal("internal", query, forecast("internal", query, workers))


def prepare_colosseum(workers: int, bundle: Path, source: Path) -> None:
    target = panel_dir("colosseum")
    if (target / "preoutcome_seal.json").exists():
        raise FileExistsError("Colosseum panel already sealed")
    protocol()
    selection = bundle.parent / f"{bundle.name.removesuffix('_bundle')}_selection.json"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "dataset_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"), "sealed_at_utc": now(), "learner_outcomes_observed": 0,
        "bundle": str(bundle), "kernel_source": str(source),
        "sha256": {"manifest.json": sha256(bundle / "manifest.json"), "data_only_index.json": sha256(bundle / "data_only_index.json"),
                   "selection.json": sha256(selection), "candidates.json": sha256(source / "candidates.json")},
        "kernels_sha256": {p.stem: sha256(p) for p in sorted(source.glob("*.npz"))}})
    for name in ("candidates.json",):
        (target / f"colosseum_{name}").write_bytes((source / name).read_bytes())
    (target / "colosseum_selection.json").write_bytes(selection.read_bytes())
    query = build_query("colosseum", bundle, workers, colosseum=str(source))
    seal("colosseum", query, forecast("colosseum", query, workers))


# --------------------------------------------------------------------------- #
# Phase: evaluate (learner jobs)
# --------------------------------------------------------------------------- #

def _mdp_jobs(task):
    sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
    from scripts.analysis.run_unseen_validation import archive_imports
    archive_imports()
    sys.path.insert(0, str(ROOT / "benchmarks/colosseum/driver"))
    import expectile_fix
    expectile_fix.install()
    expectile_fix.CALLS["fallback"] = 0
    from config.protocol_config import PROTOCOL_CONFIG
    from scripts.training.run_empirical_requirement import _load_run, _score_policy
    from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular
    from environments.benchmark_environments import make_benchmark_mdp
    from environments.tabular_environments import TabularMDP

    learning = PROTOCOL_CONFIG["learning"]
    instance = task["instance"]
    if task["colosseum"]:
        mdp = colosseum_mdp(task["colosseum"], instance["instance_id"], TabularMDP)
    else:
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
    cache, rows = {}, []
    for record in task["records"]:
        transitions = _load_run(Path(task["dataset"]), record, cache)
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
                policy = behavior_cloning_tabular(transitions, num_states=mdp.num_states, num_actions=mdp.num_actions)
            value, status = _score_policy(mdp, np.asarray(policy, dtype=float),
                                          random_return=float(instance["random_return"]),
                                          dynamic_range=float(instance["dynamic_range"]))
            if status != "observed":
                raise RuntimeError(f"invalid score {instance['instance_id']}/{method}")
            rows.append({"mdp_id": instance["instance_id"], "policy_family": record["policy_family"],
                         "algorithm": method, "budget": int(record["size_multiplier"]),
                         "training_seed": int(record["seed_index"]), "buffer_sha256": record["sha256"],
                         "normalized_return": value, "success": bool(value >= 0.95),
                         "expectile_fallbacks_so_far": expectile_fix.CALLS["fallback"]})
    return rows


def evaluate(panel: str, workers: int, bundle: Path | None, source: Path | None) -> None:
    from scripts.analysis.run_unseen_validation import archive_imports
    from scripts.analysis.compare_difficulty import suffix_event

    target = panel_dir(panel)
    seal_doc = load_json(target / "preoutcome_seal.json")
    for name, digest in seal_doc["artifact_sha256"].items():
        if sha256(target / name) != digest:
            raise RuntimeError(f"Sealed file changed: {name}")
    if seal_doc["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise RuntimeError("Protocol changed after the seal")
    if (target / "outcomes.csv.gz").exists():
        raise FileExistsError("Outcomes already computed")
    _, _, _, seed_index = archive_imports()
    from scripts.analysis import run_oeh_final_120 as R
    R._repo_first()  # spawned workers must import this runner, not the archived package
    dataset = DATASET if panel == "internal" else bundle
    manifest = load_json(dataset / "manifest.json")
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    by_mdp, seen = defaultdict(list), set()
    for row in manifest["datasets"]:
        if row.get("stream") != "training" or row.get("benchmark_split", "evaluation") != "evaluation":
            continue
        key = (row["instance_id"], row["policy_family"], int(row["size_multiplier"]), seed_index(row))
        if key in seen:
            raise RuntimeError("duplicate training record")
        seen.add(key)
        by_mdp[row["instance_id"]].append({**row, "seed_index": key[3]})
    if len(seen) != len(instances) * 5 * 6 * 10:
        raise RuntimeError(f"incomplete training grid: {len(seen)}")
    tasks = [{"instance": instances[m], "dataset": str(dataset), "colosseum": str(source) if source else None,
              "records": sorted(r, key=lambda x: (x["policy_family"], x["size_multiplier"], x["seed_index"]))}
             for m, r in sorted(by_mdp.items())]
    tick = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(_mdp_jobs, tasks), 1):
            rows.extend(result)
            if done % 10 == 0 or done == len(tasks):
                print(f"labels {panel}: {done}/{len(tasks)} MDPs, {time.perf_counter() - tick:.0f}s", flush=True)
    outcomes = pd.DataFrame(rows)
    counts = outcomes.groupby(["mdp_id", "policy_family", "algorithm", "budget"]).success.agg(["sum", "size"])
    if (counts["size"] != 10).any():
        raise RuntimeError("every group/learner/budget needs ten seeds")
    wide = counts["sum"].unstack("budget").reindex(columns=list(BUDGETS))
    labels = wide.reset_index()
    labels["event"] = [suffix_event(tuple(int(x) for x in r)) for r in wide.to_numpy()]
    labels = labels.rename(columns={b: f"successes_x{b}" for b in BUDGETS})
    outcomes.to_csv(target / "outcomes.csv.gz", index=False)
    labels.to_csv(target / "labels.csv", index=False)
    write_json(target / "outcome_run.json", {
        "jobs": len(outcomes), "seconds": time.perf_counter() - tick, "completed_at_utc": now(),
        "preoutcome_seal_sha256": sha256(target / "preoutcome_seal.json"),
        "mdps_with_expectile_fallback": int((outcomes.groupby("mdp_id").expectile_fallbacks_so_far.max() > 0).sum()),
        "outcomes_sha256": sha256(target / "outcomes.csv.gz"), "labels_sha256": sha256(target / "labels.csv")})
    print(f"{panel}: {len(outcomes)} jobs, {len(labels)} labels", flush=True)


# --------------------------------------------------------------------------- #
# Phase: analyze
# --------------------------------------------------------------------------- #

def analyze(panel: str) -> dict:
    from scripts.analysis.colosseum_posthoc_20261014 import decision_costs, fixed_cdf, max_t, percentile_ci, shapley

    doc = protocol()
    target = panel_dir(panel)
    run = load_json(target / "outcome_run.json")
    if run["preoutcome_seal_sha256"] != sha256(target / "preoutcome_seal.json"):
        raise RuntimeError("Seal changed after outcomes")
    pred = pd.read_csv(target / "predictions_preoutcome.csv")
    labels = pd.read_csv(target / "labels.csv")
    m = pred.merge(labels[KEYS + ["event"]], on=KEYS, validate="many_to_one")
    p, y = m[P].to_numpy(float), m.event.to_numpy(int)
    m["brier"] = ((np.cumsum(p, 1)[:, :6] - (y[:, None] <= np.arange(6))) ** 2).mean(1)
    m["log_loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    m["family"] = m.mdp_id.str.replace(r"^(c8_|c8_colosseum_)", "", regex=True).str.replace(r"_\d+$", "", regex=True)
    m.to_csv(target / "scored.csv.gz", index=False)
    per_mdp = m.groupby(["method", "mdp_id"])[["brier", "log_loss"]].mean().reset_index()
    wide = per_mdp.pivot(index="mdp_id", columns="method")
    steps, open_gate = {}, True
    for name, step in STEPS.items():
        if name == "step3":
            pairs = [(step["baseline"], c) for c in step["candidates"]]
        else:
            pairs = [(b, step["candidate"]) for b in step["baselines"]]
        names = [(b, c, metric) for b, c in pairs for metric in ("brier", "log_loss")]
        effect = np.column_stack([wide[(metric, b)] - wide[(metric, c)] for b, c, metric in names])
        mean, lower, critical = max_t(effect, BOOTSTRAP["seed"], BOOTSTRAP["draws"])
        rows = [{"baseline": b, "candidate": c, "metric": metric, "baseline_mean": wide[(metric, b)].mean(),
                 "candidate_mean": wide[(metric, c)].mean(), "relative_gain": mean[i] / wide[(metric, b)].mean(),
                 "mean_improvement": mean[i], "simultaneous_lower_95": lower[i],
                 "mdps_better": int((effect[:, i] > 0).sum()), "mdps_worse": int((effect[:, i] < 0).sum())}
                for i, (b, c, metric) in enumerate(names)]
        passed = bool(all(r["simultaneous_lower_95"] > 0 for r in rows))
        if name == "step3":
            verdict = {c: all(r["simultaneous_lower_95"] > 0 for r in rows if r["candidate"] == c)
                       for c in step["candidates"]}
        else:
            verdict = "GO" if passed else "NO-GO"
        steps[name] = {"tested": open_gate, "critical": critical, "contrasts": rows, "verdict": verdict
                       if open_gate else f"not tested (gate closed); descriptive {verdict}"}
        open_gate = open_gate and passed
    # descriptive: summary, learner, family, finite/censored, decision regret, Shapley
    summaries = []
    for scope, by in (("pooled", []), ("learner", ["algorithm"]), ("family", ["family"])):
        t = m.groupby([*by, "method", "mdp_id"])[["brier", "log_loss"]].mean().groupby([*by, "method"]).mean().reset_index()
        t.insert(0, "scope", scope)
        summaries.append(t)
    pd.concat(summaries).to_csv(target / "summary.csv", index=False)
    ids = sorted(wide.index)
    index = np.random.default_rng(BOOTSTRAP["seed"]).integers(0, len(ids), size=(BOOTSTRAP["draws"], len(ids)))
    regret_rows, regret_contrasts = [], []
    base = m[m.method == "oeh_final"].sort_values(KEYS).reset_index(drop=True)
    for tau in DECISION["tau"]:
        for lam in DECISION["lambda"]:
            by = {}
            for name in DECISION["methods"] + DECISION["fixed_rules"]:
                if name in DECISION["fixed_rules"]:
                    frame, cdf = base, fixed_cdf(name, len(base))
                else:
                    frame = m[m.method == name].sort_values(KEYS).reset_index(drop=True)
                    cdf = np.cumsum(frame[P].to_numpy(float), 1)[:, :6]
                parts = pd.DataFrame(decision_costs(cdf, frame.event.to_numpy(int), tau, lam))
                parts["mdp_id"] = frame.mdp_id.to_numpy()
                by[name] = parts.groupby("mdp_id").mean().reindex(ids)
                regret_rows.append({"tau": tau, "lambda": lam, "method": name,
                                    **{f"mean_{c}": float(by[name][c].mean()) for c in by[name].columns}})
            for name, frame in by.items():
                if name != "oeh_final":
                    delta = (by["oeh_final"].regret - frame.regret).to_numpy(float)
                    low, high = percentile_ci(delta, index)
                    regret_contrasts.append({"tau": tau, "lambda": lam, "comparator": name,
                                             "oeh_minus_comparator": float(delta.mean()), "ci95_low": low, "ci95_high": high})
    pd.DataFrame(regret_rows).to_csv(target / "decision_regret.csv", index=False)
    pd.DataFrame(regret_contrasts).to_csv(target / "decision_contrasts.csv", index=False)
    shap_rows = []
    factor = per_mdp[per_mdp.method.isin(FACTORIAL)].replace(
        {"method": {"oeh_final": "oeh_full_108", "prior_oeh_matched": "b2_plus_L"}})
    for metric in ("brier", "log_loss"):
        shap = shapley(factor, metric).reindex(ids)
        for block in shap.columns:
            values = shap[block].to_numpy(float)
            low, high = percentile_ci(values, index)
            shap_rows.append({"metric": metric, "block": block, "shapley_mean": values.mean(), "ci95_low": low,
                              "ci95_high": high, "full_minus_b2": shap.sum(axis=1).mean()})
    pd.DataFrame(shap_rows).to_csv(target / "block_shapley.csv", index=False)
    finite = m[m.event < 6]
    fin = []
    for method, g in finite.groupby("method"):
        q = g[P].to_numpy(float)
        cond = q[:, :6] / np.clip(q[:, :6].sum(1, keepdims=True), 1e-300, None)
        fin.append({"method": method, "finite_rows": len(g),
                    "within_one_level": float((np.abs(cond.argmax(1) - g.event.to_numpy()) <= 1).mean())})
    from sklearn.metrics import roc_auc_score
    auc = {method: float(roc_auc_score(g.event == 6, g.p_6)) for method, g in m.groupby("method") if g.event.nunique() > 1}
    groups = []
    for path in (target / "oeh_b_groups").glob("*.json"):
        d = load_json(path)
        groups.append(d["sampling"].get("draws_returned"))
    result = {"panel": panel, "completed_at_utc": now(), "mdps": int(m.mdp_id.nunique()), "rows": int(len(labels)),
              "censored_fraction": float((labels.event == 6).mean()),
              "censored_by_learner": labels.groupby("algorithm").event.apply(lambda e: float((e == 6).mean())).to_dict(),
              "steps": steps, "auc_censored": auc, "finite": fin,
              "zero_draw_groups": int(sum(g == 0 for g in groups)), "oeh_b_groups": len(groups),
              "labels_sha256": sha256(target / "labels.csv"), "protocol_sha256": sha256(OUTPUT / "protocol.json")}
    write_json(target / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("freeze", "prepare-internal", "prepare-colosseum", "evaluate", "analyze"))
    parser.add_argument("--panel", choices=("internal", "colosseum"), default="internal")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "prepare-internal":
        prepare_internal(args.workers)
    elif args.phase == "prepare-colosseum":
        prepare_colosseum(args.workers, args.bundle, args.source)
    elif args.phase == "evaluate":
        evaluate(args.panel, args.workers, args.bundle, args.source)
    else:
        result = analyze(args.panel)
        print(json.dumps({k: v for k, v in result.items() if k in ("panel", "mdps", "censored_fraction")}, indent=1))
        for name, step in result["steps"].items():
            print(name, step["verdict"], round(step["critical"], 3))
            for r in step["contrasts"]:
                print(f"  {r['baseline']:>20s} -> {r['candidate']:<10s} {r['metric']:<8s} gain {r['relative_gain']:+.3f} "
                      f"LB {r['simultaneous_lower_95']:+.4f}")


if __name__ == "__main__":
    main()
