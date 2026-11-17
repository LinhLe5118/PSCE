"""Sealed fresh-MDP confirmation of the development-only OEH difficulty revision.

Run freeze, prepare, evaluate, analyze in that order. The complete six-budget
grid is scored once. Historical E1 and the open first 60 MDPs are never query
labels. No interim endpoint is used to revise the registered candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.analysis.compare_difficulty import GRID, METHODS, aft_interval, sha256, suffix_event
from scripts.analysis.compare_e1_dataset_signals import SIGNALS
from scripts.analysis.develop_oeh_blend import FEATURES as DIRECT_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.independent_difficulty import XGB_PARAMS
from scripts.analysis.run_unseen_validation import ROOT, ARCHIVE, archive_imports, load_json, write_json


OUTPUT = ROOT / "artifacts/oeh_difficulty_confirmation_2026-10-01"
DATASET = ROOT / "datasets/oeh_difficulty_confirmation_2026-10-01"
E1 = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"
E1_FEATURES = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv"
DEVELOPMENT = ROOT / "artifacts/oeh_difficulty_blend_development_2026-10-01/manifest.json"
FAMILY_COUNTS = {"uniform_garnet": 15, "forward_layered": 18,
                 "bottleneck": 9, "lock_structured": 18}
NAMESPACE = "oeh-difficulty-independent-v2-20261001"
KEYS = ["mdp_id", "policy_family", "algorithm"]
BASELINES = ("train_frequency", "coverage_adaptation", "quality_adaptation",
             "coverage_quality_adaptation", "b2_generic")
METHODS_SCORED = (*BASELINES, "b2_plus_oeh", "oeh_blend")
PROBABILITY_COLUMNS = [f"p_{index}" for index in range(7)]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_specs():
    from environments.benchmark_environments import benchmark_specs

    specs = benchmark_specs()
    selected = []
    for family, count in FAMILY_COUNTS.items():
        available = [spec for spec in specs if spec.split == "evaluation" and spec.family == family]
        if len(available) != 2 * count:
            raise RuntimeError(f"Original recipe family size changed: {family}")
        ordered = sorted(available, key=lambda spec: hashlib.sha256(
            f"{NAMESPACE}|select|{spec.instance_id}".encode()).digest())
        for index, template in enumerate(sorted(ordered[:count], key=lambda spec: spec.ordinal)):
            digest = hashlib.sha256(f"{NAMESPACE}|seed|{template.instance_id}".encode()).digest()
            seed = 100_000_000 + int.from_bytes(digest[:8], "little") % 1_000_000_000
            selected.append(replace(template, instance_id=f"difficulty2_{family}_{index:03d}", seed=seed))
    if len(selected) != 60 or len({spec.seed for spec in selected}) != 60:
        raise RuntimeError("Invalid fresh cohort count/seeds")
    return selected


def freeze():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace confirmation protocol: {OUTPUT}")
    _, _, _, _ = archive_imports()
    from scripts.analysis.e1_oracle_qvi_diagnostic import archived_source_hash
    from environments.benchmark_environments import BenchmarkSpec

    source_manifest = ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1/manifest.json"
    e1_manifest = load_json(source_manifest)
    e1_seal = load_json(ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json")
    if sha256(source_manifest) != e1_seal["manifest_sha256"]:
        raise RuntimeError("Original E1 dataset seal changed")
    if archived_source_hash(ARCHIVE, e1_manifest["generator_provenance"]["source_files"]) != e1_manifest["generator_provenance"]["source_sha256"]:
        raise RuntimeError("Pinned dataset generator changed")
    development = load_json(DEVELOPMENT)
    if development["status"] != "development_only_open_60_mdp_not_independent_confirmation":
        raise RuntimeError("Development selection history missing")
    specs = choose_specs()
    first_protocol = load_json(ROOT / "artifacts/independent_validation_2026-09-30/protocol.json")
    old_seeds = {row["seed"] for row in e1_manifest["instances"]}
    old_seeds |= {row["seed"] for row in first_protocol["requested_specs"]}
    if old_seeds & {spec.seed for spec in specs}:
        raise RuntimeError("Fresh seeds overlap E1 or inspected MDPs")
    source_files = [Path(__file__), ROOT / "scripts/analysis/develop_oeh_blend.py",
                    ROOT / "scripts/analysis/develop_oeh_difficulty.py",
                    ROOT / "scripts/analysis/independent_difficulty.py",
                    ROOT / "scripts/analysis/compare_difficulty.py",
                    ROOT / "scripts/analysis/compare_e1_dataset_signals.py",
                    ROOT / "scripts/analysis/run_unseen_validation.py", E1, E1_FEATURES,
                    DEVELOPMENT, source_manifest,
                    ARCHIVE / "evaluation/e1_pipeline.py",
                    ARCHIVE / "scripts/data/generate_offline_datasets.py",
                    ARCHIVE / "environments/benchmark_environments.py"]
    doc = {
        "schema": "oeh-difficulty-confirmation-v2", "status": "frozen_before_new_data_or_outcomes",
        "namespace": NAMESPACE, "requested_specs": [spec.to_dict() for spec in specs],
        "family_counts": FAMILY_COUNTS, "mdp_count": 60, "primary_count": 42,
        "pilot_multipliers": [2, 5, 10], "grid": GRID,
        "training_seed_indices": list(range(10)), "behavior_families": 5,
        "core_methods": METHODS, "expected_jobs": 60 * 5 * 3 * 6 * 10,
        "target": "first complete suffix with at least 8/10 seeds at normalized return >=0.95; else >x50",
        "candidate": "global E1-out-of-fold logloss-weighted blend of direct six-threshold OEH and original OEH+AFT",
        "direct_weight": float(development["weight_on_direct"]),
        "direct_features": DIRECT_FEATURES, "aft_features": SIGNALS,
        "fit": {"direct": "same frozen binary XGBoost for six thresholds",
                "aft": XGB_PARAMS, "rounds": 200},
        "comparators": METHODS_SCORED,
        "primary_gate": "positive simultaneous one-sided 95% MDP-bootstrap lower bounds for five data-only baselines by two proper scores",
        "bootstrap": {"draws": 2000, "seed": 20261001,
                      "method": "max studentized centered error over ten contrasts"},
        "controls": "reported separately; no control-based candidate selection",
        "scope": "tabular adaptations of published data-quality ideas plus generic B2; no exact EH/TQ/SACo or literature-wide SOTA claim",
        "e1_reuse": "meta-training/development only; any later 120-MDP analysis is descriptive",
        "original_generator_sha256": e1_manifest["generator_provenance"]["source_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in source_files},
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "protocol.json", doc)
    (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"Frozen new 60-MDP difficulty protocol: {sha256(OUTPUT / 'protocol.json')}", flush=True)


def protocol():
    path = OUTPUT / "protocol.json"
    if sha256(path) != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Confirmation protocol hash changed")
    doc = load_json(path)
    if doc["schema"] != "oeh-difficulty-confirmation-v2":
        raise RuntimeError("Unknown confirmation protocol")
    for relative, expected in doc["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source/input changed: {relative}")
    return doc


def dataset():
    seal = load_json(OUTPUT / "dataset_seal.json")
    for relative, expected in seal["sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Dataset seal changed: {relative}")
    manifest = load_json(DATASET / "manifest.json")
    if manifest["instance_count"] != 60 or manifest["physical_buffer_count"] != 3300:
        raise RuntimeError("Incomplete confirmation dataset")
    if manifest["generator_provenance"]["source_sha256"] != protocol()["original_generator_sha256"]:
        raise RuntimeError("Confirmation generator provenance mismatch")
    return manifest


def pilot_features(manifest):
    from evaluation import e1_pipeline

    indexed = load_json(DATASET / "data_only_index.json")["datasets"]
    records = {row["data_only_dataset_id"]: row for row in manifest["datasets"]}
    grouped = defaultdict(dict)
    for item in indexed:
        if item.get("stream") != "pilot" or int(item.get("size_multiplier", -1)) not in (2, 5, 10):
            continue
        row = records[item["dataset_id"]]
        if row["sha256"] != item["sha256"] or row["file"] != item["file"]:
            raise RuntimeError("Pilot index differs from manifest")
        key = row["instance_id"], row["policy_family"]
        size = int(row["size_multiplier"])
        if size in grouped[key]:
            raise RuntimeError("Duplicate pilot budget")
        grouped[key][size] = row
    if len(grouped) != 300 or any(set(value) != {2, 5, 10} for value in grouped.values()):
        raise RuntimeError("Incomplete data-only pilot grid")
    rows = []
    for group, prefixes in sorted(grouped.items()):
        rows.extend(e1_pipeline._group_features(DATASET, group, prefixes))
    if len(rows) != 900:
        raise RuntimeError("Incomplete learner-specific pilot features")
    write_csv(OUTPUT / "pilot_features.csv", rows)


def prepare():
    from scipy.special import ndtr
    import pandas as pd
    import xgboost as xgb

    doc = protocol()
    if (OUTPUT / "preoutcome_seal.json").exists() or DATASET.exists():
        raise FileExistsError("Confirmation dataset/predictions already exist")
    pipeline, _, generator, _ = archive_imports()
    from environments.benchmark_environments import BenchmarkSpec

    specs = [BenchmarkSpec(**row) for row in doc["requested_specs"]]
    manifest = generator.generate_benchmark_bundle(
        DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)),
        max_size=None, horizon=100, run_mode="test",
        deviation_reason="Predeclared new-MDP OEH difficulty confirmation",
        pilot_size_multipliers=(2, 5, 10))
    if manifest["instance_count"] != 60 or manifest["physical_buffer_count"] != 3300:
        raise RuntimeError("Unexpected new dataset count")
    paths = (DATASET / "manifest.json", DATASET / "data_only_index.json")
    write_json(OUTPUT / "dataset_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "sealed_at_utc": now(),
        "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}})
    pilot_features(manifest)
    keys = KEYS
    train = pd.read_csv(E1).merge(pd.read_csv(E1_FEATURES), on=keys, validate="one_to_one")
    query = pd.read_csv(OUTPUT / "pilot_features.csv")
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    query["panel"] = query.mdp_id.map(lambda mdp: "control" if family[mdp] == "lock_structured" else "primary")
    if (len(train) != 1800 or len(query) != 900 or train[keys].duplicated().any()
            or query[keys].duplicated().any()):
        raise RuntimeError("Invalid E1/query feature identity")
    methods = {**SIGNALS}
    rows = []
    for panel in ("primary", "control"):
        for algorithm in METHODS:
            tr = train[(train.panel == panel) & (train.algorithm == algorithm)]
            qu = query[(query.panel == panel) & (query.algorithm == algorithm)]
            if len(tr) != (420 if panel == "primary" else 180) or len(qu) != (210 if panel == "primary" else 90):
                raise RuntimeError("Incomplete panel/learner forecast stratum")
            event = tr.event.to_numpy(dtype=int)
            direct = fit_thresholds(tr[list(DIRECT_FEATURES)].to_numpy(), event,
                                    qu[list(DIRECT_FEATURES)].to_numpy())
            forecasts = {"train_frequency": np.repeat(
                (np.bincount(event, minlength=7) / len(event))[None, :], len(qu), axis=0)}
            lower, upper = np.asarray([aft_interval(int(value)) for value in event], dtype=float).T
            for name, fields in methods.items():
                x = tr[list(fields)].to_numpy(dtype=float)
                q = qu[list(fields)].to_numpy(dtype=float)
                if not np.isfinite(x).all() or not np.isfinite(q).all():
                    raise RuntimeError(f"Nonfinite pilot feature: {name}")
                fitted = xgb.DMatrix(x)
                fitted.set_float_info("label_lower_bound", lower)
                fitted.set_float_info("label_upper_bound", upper)
                model = xgb.train(XGB_PARAMS, fitted, num_boost_round=200)
                median = model.predict(xgb.DMatrix(q))
                if not np.isfinite(median).all() or (median <= 0).any():
                    raise RuntimeError(f"Invalid AFT forecast: {name}")
                cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
                cats = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1. - cdf[:, -1])), 0.)
                forecasts[name] = cats / cats.sum(axis=1, keepdims=True)
            weight = doc["direct_weight"]
            forecasts["oeh_blend"] = weight * direct + (1. - weight) * forecasts["b2_plus_oeh"]
            for index, (_, item) in enumerate(qu.iterrows()):
                for name in METHODS_SCORED:
                    p = forecasts[name][index]
                    if not np.isfinite(p).all() or (p < -1e-12).any() or not np.isclose(p.sum(), 1., atol=1e-10):
                        raise RuntimeError("Invalid pre-outcome event probabilities")
                    rows.append({"mdp_id": item.mdp_id, "policy_family": item.policy_family,
                                 "algorithm": algorithm, "panel": panel, "method": name,
                                 **{f"p_{category}": float(p[category]) for category in range(7)}})
            print(f"Sealed forecasts: {panel}/{algorithm}", flush=True)
    if len(rows) != 900 * len(METHODS_SCORED):
        raise RuntimeError("Incomplete pre-outcome forecasts")
    write_csv(OUTPUT / "predictions_preoutcome.csv", rows)
    write_json(OUTPUT / "preoutcome_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "dataset_seal_sha256": sha256(OUTPUT / "dataset_seal.json"),
        "validation_outcomes_observed": 0, "sealed_at_utc": now(),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("pilot_features.csv", "predictions_preoutcome.csv")}})
    print(f"Sealed {len(rows)} predictions before learner outcomes", flush=True)


def verified_preoutcome():
    doc = protocol()
    manifest = dataset()
    seal = load_json(OUTPUT / "preoutcome_seal.json")
    if (seal["protocol_sha256"] != sha256(OUTPUT / "protocol.json")
            or seal["dataset_seal_sha256"] != sha256(OUTPUT / "dataset_seal.json")
            or seal["validation_outcomes_observed"] != 0):
        raise RuntimeError("Invalid pre-outcome seal")
    for name, expected in seal["artifact_sha256"].items():
        if sha256(OUTPUT / name) != expected:
            raise RuntimeError(f"Pre-outcome artifact changed: {name}")
    return doc, manifest


def jobs(manifest, seed_index):
    records = {}
    for row in manifest["datasets"]:
        if row.get("benchmark_split") != "evaluation" or row.get("stream") != "training":
            continue
        key = row["instance_id"], row["policy_family"], int(row["size_multiplier"]), seed_index(row)
        if key in records:
            raise RuntimeError("Duplicate training dataset identity")
        records[key] = row
    if len(records) != 60 * 5 * 6 * 10:
        raise RuntimeError("Incomplete training data grid")
    for (mdp, behavior, budget, seed), record in sorted(records.items()):
        for method in METHODS:
            yield {"instance_id": mdp, "policy_family": behavior, "size_multiplier": budget,
                   "training_seed": seed, "method": method,
                   "buffer_file": record["file"], "buffer_sha256": record["sha256"]}


def job_path(job):
    identity = "|".join(str(job[key]) for key in
                        ("instance_id", "policy_family", "size_multiplier", "training_seed", "method"))
    return OUTPUT / "execution/jobs" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")


def verify_job(job, path, plan_sha):
    row = load_json(path)
    outcome = row["outcome"]
    value = float(outcome["normalized_return"])
    if (row["job"] != job or row["execution_plan_sha256"] != plan_sha
            or outcome["status"] != "observed" or outcome["method"] != job["method"]
            or outcome["buffer_sha256"] != job["buffer_sha256"]
            or not np.isfinite(value) or bool(outcome["success"]) != (value >= .95)):
        raise RuntimeError(f"Invalid/mismatched job outcome: {path}")
    return row


def evaluate(resume):
    doc, manifest = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Recorded learner failure: retain it for audit")
    pipeline, _, _, seed_index = archive_imports()
    planned = list(jobs(manifest, seed_index))
    if len(planned) != doc["expected_jobs"]:
        raise RuntimeError("Job plan count differs from protocol")
    plan = {"protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
            "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
            "jobs": planned, "job_count": len(planned)}
    path = OUTPUT / "execution_plan.json"
    if path.exists():
        if not resume or load_json(path) != plan:
            raise RuntimeError("Resume requires identical sealed execution plan")
    else:
        write_json(path, plan)
    plan_sha = sha256(path)
    (OUTPUT / "execution/jobs").mkdir(parents=True, exist_ok=True)
    callback = pipeline.build_core_train_score_callback(DATASET, manifest)
    started, skipped = time.perf_counter(), 0
    for index, job in enumerate(planned, 1):
        file = job_path(job)
        if file.exists():
            if not resume:
                raise FileExistsError(f"Existing job without --resume: {file}")
            verify_job(job, file, plan_sha)
            skipped += 1
            continue
        try:
            tick = time.perf_counter()
            outcome = callback(job)
            write_json(file, {"job": job, "execution_plan_sha256": plan_sha,
                              "outcome": dict(outcome), "runtime_seconds": time.perf_counter() - tick,
                              "completed_at_utc": now()})
            verify_job(job, file, plan_sha)
        except Exception as error:
            write_json(OUTPUT / "failure_evaluate.json", {
                "job_index": index, "job": job, "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(),
                "plan_sha256": plan_sha, "failed_at_utc": now()})
            raise
        if index % 500 == 0 or index == len(planned):
            write_json(OUTPUT / "execution/progress.json", {
                "completed": index, "total": len(planned), "resumed": skipped,
                "elapsed_seconds_this_process": time.perf_counter() - started,
                "updated_at_utc": now()})
            print(f"Difficulty confirmation {index}/{len(planned)}", flush=True)
    write_json(OUTPUT / "execution_complete.json", {
        "completed_at_utc": now(), "plan_sha256": plan_sha, "job_count": len(planned)})


def simultaneous_lower(differences, seed=20261001, draws=2000):
    """One shared cluster bootstrap with a studentized max-error critical value."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite MDP-by-contrast differences")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    standardized = np.divide(sampled - mean, scale, out=np.zeros_like(sampled), where=scale > 0)
    critical = float(np.quantile(standardized.max(axis=1), .95))
    return mean, mean - critical * scale, critical


def analyze():
    doc, manifest = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Failed learner panel cannot be selectively analyzed")
    plan_file = OUTPUT / "execution_plan.json"
    plan = load_json(plan_file)
    complete = load_json(OUTPUT / "execution_complete.json")
    plan_sha = sha256(plan_file)
    if (complete["plan_sha256"] != plan_sha or complete["job_count"] != doc["expected_jobs"]
            or plan["preoutcome_seal_sha256"] != sha256(OUTPUT / "preoutcome_seal.json")
            or len(plan["jobs"]) != doc["expected_jobs"]):
        raise RuntimeError("Execution incomplete or mismatched")
    outcomes = []
    grouped = defaultdict(dict)
    for job in plan["jobs"]:
        payload = verify_job(job, job_path(job), plan_sha)
        key = job["instance_id"], job["policy_family"], job["method"]
        budget = job["size_multiplier"]
        grouped[key].setdefault(budget, []).append(int(payload["outcome"]["success"]))
        outcomes.append({"mdp_id": key[0], "policy_family": key[1], "algorithm": key[2],
                         "budget": budget, "training_seed": job["training_seed"],
                         "normalized_return": float(payload["outcome"]["normalized_return"]),
                         "success": int(payload["outcome"]["success"]),
                         "runtime_seconds": payload["runtime_seconds"]})
    write_csv(OUTPUT / "all_core_budget_seed_outcomes.csv", outcomes)
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    labels = []
    for (mdp, behavior, algorithm), budgets in sorted(grouped.items()):
        if set(budgets) != set(GRID) or any(len(budgets[b]) != 10 for b in GRID):
            raise RuntimeError("Incomplete budget/seed grid")
        counts = tuple(sum(budgets[b]) for b in GRID)
        labels.append({"mdp_id": mdp, "policy_family": behavior, "algorithm": algorithm,
                       "panel": "control" if family[mdp] == "lock_structured" else "primary",
                       "event": suffix_event(counts),
                       **{f"successes_x{b}": count for b, count in zip(GRID, counts)}})
    if len(labels) != 900:
        raise RuntimeError("Incomplete learner-specific event labels")
    write_csv(OUTPUT / "labels.csv", labels)
    lookup = {(row["mdp_id"], row["policy_family"], row["algorithm"]): row for row in labels}
    forecasts = read_csv(OUTPUT / "predictions_preoutcome.csv")
    if len(forecasts) != 900 * len(METHODS_SCORED):
        raise RuntimeError("Incomplete frozen forecasts")
    seen, scored = set(), []
    for row in forecasts:
        key = row["mdp_id"], row["policy_family"], row["algorithm"]
        identity = (*key, row["method"])
        if identity in seen or key not in lookup or row["method"] not in METHODS_SCORED:
            raise RuntimeError("Duplicate/unregistered prediction")
        seen.add(identity)
        event = int(lookup[key]["event"])
        if row["panel"] != lookup[key]["panel"]:
            raise RuntimeError("Forecast panel mismatch")
        p = np.asarray([float(row[name]) for name in PROBABILITY_COLUMNS])[None, :]
        brier, logloss = score_probabilities(p, np.asarray([event]))
        scored.append({"mdp_id": key[0], "policy_family": key[1], "algorithm": key[2],
                       "panel": row["panel"], "method": row["method"], "event": event,
                       "integrated_brier": float(brier[0]),
                       "seven_class_log_loss": float(logloss[0])})
    if len(seen) != 900 * len(METHODS_SCORED):
        raise RuntimeError("Prediction identities incomplete")
    write_csv(OUTPUT / "scored_predictions.csv", scored)
    by_mdp = defaultdict(list)
    for row in scored:
        by_mdp[(row["mdp_id"], row["method"])].append(row)
    mdp_scores = []
    for (mdp, method), rows in sorted(by_mdp.items()):
        if len(rows) != 15:
            raise RuntimeError("Incomplete within-MDP task scores")
        mdp_scores.append({"mdp_id": mdp, "panel": rows[0]["panel"], "method": method,
                           **{metric: float(np.mean([item[metric] for item in rows]))
                              for metric in ("integrated_brier", "seven_class_log_loss")}})
    write_csv(OUTPUT / "mdp_scores.csv", mdp_scores)
    summary = []
    for panel, expected in (("primary", 42), ("control", 18)):
        for method in METHODS_SCORED:
            selected = [row for row in mdp_scores if row["panel"] == panel and row["method"] == method]
            if len(selected) != expected:
                raise RuntimeError("Incomplete panel/MDP scores")
            summary.append({"panel": panel, "method": method, "mdps": expected,
                            **{metric: float(np.mean([row[metric] for row in selected]))
                               for metric in ("integrated_brier", "seven_class_log_loss")}})
    write_csv(OUTPUT / "summary.csv", summary)
    mdp_ids = sorted(item["instance_id"] for item in manifest["instances"]
                     if item["family"] != "lock_structured")
    lookup_score = {(row["mdp_id"], row["method"]): row for row in mdp_scores}
    contrasts = [(metric, baseline) for metric in ("integrated_brier", "seven_class_log_loss")
                 for baseline in BASELINES]
    differences = np.asarray([[lookup_score[(mdp, baseline)][metric]
                               - lookup_score[(mdp, "oeh_blend")][metric]
                               for metric, baseline in contrasts] for mdp in mdp_ids])
    point, lower, critical = simultaneous_lower(differences)
    contrast_rows = [{"panel": "primary", "metric": metric, "baseline": baseline,
                      "candidate": "oeh_blend", "mdps": len(mdp_ids),
                      "mean_improvement": float(point[index]),
                      "simultaneous_one_sided_lower_95": float(lower[index])}
                     for index, (metric, baseline) in enumerate(contrasts)]
    write_csv(OUTPUT / "contrasts.csv", contrast_rows)
    status = "GO" if np.all(lower > 0) else "NO-GO"
    write_json(OUTPUT / "result_manifest.json", {
        "status": "complete", "completed_at_utc": now(),
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
        "execution_plan_sha256": plan_sha, "jobs": len(outcomes),
        "mdps": 60, "stochastic_primary_mdps": 42,
        "simultaneous_critical_value": critical, "difficulty_gate": status,
        "scope": doc["scope"],
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("all_core_budget_seed_outcomes.csv", "labels.csv",
                             "scored_predictions.csv", "mdp_scores.csv", "summary.csv", "contrasts.csv")}})
    print(f"Independent difficulty gate: {status}", flush=True)
    print(json.dumps([row for row in summary if row["panel"] == "primary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--resume", action="store_true", help="Verify and reuse only matching completed jobs")
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare,
     "evaluate": lambda: evaluate(args.resume), "analyze": analyze}[args.phase]()


if __name__ == "__main__":
    main()
