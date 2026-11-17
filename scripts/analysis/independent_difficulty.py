"""Independent offline-difficulty comparison on the post-E1 60-MDP cohort.

The prepare phase seals every data-only forecast and oracle descriptor before
new learner outcomes. Evaluate runs the original core learners on the original
six-budget, ten-seed grid. Analyze opens outcomes only after all 54,000 cells
exist. Published online EH quantities remain oracle diagnostics, not claims of
an offline EH implementation or directly comparable sample-complexity bounds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from scripts.analysis.compare_difficulty import GRID, METHODS, aft_interval, sha256, suffix_event
from scripts.analysis.compare_e1_dataset_signals import SIGNALS
from scripts.analysis.run_unseen_validation import (
    ROOT, ARCHIVE, DATASET, OUTPUT as ROUTING_OUTPUT,
    archive_imports, load_json, verified_dataset, write_json,
)

OUTPUT = ROOT / "artifacts/independent_difficulty_2026-09-30"
SOURCE = ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1"
LABELS = ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"
PANELS = ("primary", "control")
ENDPOINTS = ("integrated_brier", "seven_class_log_loss")
MODEL_METHODS = ("train_frequency", *SIGNALS, "oeh3_raw_proxy", "lcoeh4_raw_proxy")
XGB_PARAMS = {
    "objective": "survival:aft", "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
    "tree_method": "hist", "learning_rate": .05, "max_depth": 2,
    "min_child_weight": 5, "lambda": 1, "seed": 20260930,
    "nthread": 1, "verbosity": 0,
}
SOURCES = {
    "stochastic_eh": "https://proceedings.iclr.cc/paper_files/paper/2024/file/3e18f7152b5e1696e73722dc5121b6f2-Paper-Conference.pdf",
    "epw_covering": "https://papers.nips.cc/paper_files/paper/2023/file/b8be628bf719550b560de8bec9456e0b-Paper-Conference.pdf",
    "dataset_quality": "https://proceedings.mlr.press/v199/schweighofer22a.html",
    "eri_eas_coi": "https://arxiv.org/abs/2111.13461",
    "xgboost_aft": "https://arxiv.org/abs/2006.04920",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _inputs():
    manifest, dataset_seal = verified_dataset()
    sources = {
        "scripts/analysis/independent_difficulty.py": sha256(Path(__file__)),
        "scripts/analysis/run_unseen_validation.py": sha256(ROOT / "scripts/analysis/run_unseen_validation.py"),
        "scripts/analysis/compare_difficulty.py": sha256(ROOT / "scripts/analysis/compare_difficulty.py"),
        "scripts/analysis/compare_e1_dataset_signals.py": sha256(ROOT / "scripts/analysis/compare_e1_dataset_signals.py"),
        "scripts/analysis/e1_oracle_qvi_diagnostic.py": sha256(ROOT / "scripts/analysis/e1_oracle_qvi_diagnostic.py"),
        "code/oeh/unseen.py": sha256(ROOT / "code/oeh/unseen.py"),
        "artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv": sha256(LABELS),
        "artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv": sha256(SOURCE / "pilot_features/features.csv"),
        "artifacts/e1_oracle_qvi_posthoc_2026-09-30/oracle_diagnostics.csv":
            sha256(ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30/oracle_diagnostics.csv"),
        "artifacts/independent_validation_2026-09-30/pilot_features.csv":
            sha256(ROUTING_OUTPUT / "pilot_features.csv"),
        "artifacts/independent_validation_2026-09-30/dataset_seal.json":
            sha256(ROUTING_OUTPUT / "dataset_seal.json"),
    }
    return manifest, sources


def protocol():
    path = OUTPUT / "protocol.json"
    if sha256(path) != (OUTPUT / "protocol.sha256").read_text("ascii").strip():
        raise RuntimeError("Difficulty protocol seal changed")
    doc = load_json(path)
    if doc["schema"] != "oeh-independent-difficulty-v1":
        raise RuntimeError("Unknown difficulty protocol")
    for relative, digest in doc["input_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Difficulty source/input hash changed: {relative}")
    return doc


def _panel(family):
    return "control" if family == "lock_structured" else "primary"


def prepare():
    if OUTPUT.exists():
        raise FileExistsError("Independent difficulty output already exists")
    manifest, sources = _inputs()
    if (ROUTING_OUTPUT / "execution_plan.json").exists() or (ROUTING_OUTPUT / "execution/jobs").exists():
        raise RuntimeError("Prepare difficulty predictions before all new learner outcomes")
    e1_manifest = load_json(ROOT / "artifacts/difficulty_e1_posthoc_2026-09-30/manifest.json")
    if (sources["artifacts/difficulty_e1_posthoc_2026-09-30/labels.csv"] !=
            e1_manifest["output_hashes"]["labels.csv"]):
        raise RuntimeError("E1 requirement labels differ from historical seal")
    feature_seal = load_json(SOURCE / "pilot_features/feature_seal.json")
    if sources["artifacts/week6/oehv4e_confirmatory_v1_1/pilot_features/features.csv"] != feature_seal["features_sha256"]:
        raise RuntimeError("E1 pilot features differ from historical seal")
    OUTPUT.mkdir(parents=True)
    doc = {
        "schema": "oeh-independent-difficulty-v1", "status": "frozen_before_new_learner_outcomes",
        "question": "Can pilot-only OEH features predict the learner-specific offline data requirement better than published dataset-quality signals and a strong AFT baseline?",
        "cohort": "same 60 newly seeded MDPs already sealed for independent routing validation",
        "mdp_count": 60, "primary_stochastic_mdps": 42, "control_mdps": 18,
        "training_cohort": "original E1 120 MDPs; all meta-training labels but no validation outcome",
        "method_library": list(METHODS), "budget_multipliers": list(GRID),
        "training_seed_indices": list(range(10)), "behavior_families": 5,
        "expected_core_jobs": 60 * 5 * 3 * 6 * 10,
        "target": "first budget whose complete suffix has >=8/10 seeds at normalized return >=0.95; otherwise >x50 right-censored",
        "primary_metric": "integrated Brier over six budget thresholds, lower better; mean per MDP then over 42 stochastic MDPs",
        "secondary_metrics": ["seven-class log loss", "event-category accuracy", "Spearman with MDP-level success-by-x50 fraction"],
        "inference": "2000 whole-MDP bootstrap resamples, seed 20260930; simultaneous one-sided lower bounds for five Brier contrasts",
        "data_only_models": list(MODEL_METHODS),
        "raw_proxy_mapping": "OEH3/LCOEH4 log1p requirement to first grid budget >= expm1(proxy), >x50 censored; LCOEH4 censor flag forces >x50; one-hot uncalibrated diagnostic",
        "data_only_features": {name: list(fields) for name, fields in SIGNALS.items()},
        "fit": "XGBoost-AFT per panel and core learner on E1 only; 200 rounds, fixed parameters; train-frequency reference",
        "xgboost_params": XGB_PARAMS,
        "oracle_comparators": ["2024 stochastic EH/QVI adaptation", "EH oracle features calibrated by the same AFT model",
                               "effective planning window adaptation",
                               "covering-length upper-bound adaptation", "SAT reference"],
        "oracle_scope": "full true MDP; online RL published bounds adapted to E1 gamma/horizon, reported as rank diagnostics only",
        "not_exact_published_metrics": "coverage/quality are pilot adaptations, not exact TQ/SACo without matched references; ERI may be undefined for nonpositive mean returns and EAS continuous-action Gaussian estimator is not directly applicable",
        "sources": SOURCES, "input_sha256": sources,
        "original_routing_dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
        "no_validation_outcome_access_for_fits": True,
    }
    path = OUTPUT / "protocol.json"
    write_json(path, doc)
    (OUTPUT / "protocol.sha256").write_text(sha256(path) + "\n", encoding="ascii")
    _oracle_preoutcome(doc, manifest)
    _predict_preoutcome(doc)
    seals = {name: sha256(OUTPUT / name) for name in ("predictions_preoutcome.csv", "oracle_preoutcome.csv")}
    write_json(OUTPUT / "preoutcome_seal.json", {
        "sealed_at_utc": now(), "protocol_sha256": sha256(path), "artifact_sha256": seals,
        "validation_outcomes_observed": 0,
    })
    print(f"Sealed 60-MDP difficulty predictions/oracles: {sha256(OUTPUT / 'preoutcome_seal.json')}", flush=True)


def _oracle_preoutcome(doc, manifest):
    from scripts.analysis.e1_oracle_qvi_diagnostic import diagnostics, finite_horizon_online_bounds
    _, _, generator, _ = archive_imports()
    from environments.benchmark_environments import make_benchmark_mdp
    prior = read_csv(ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30/oracle_diagnostics.csv")
    fingerprints = {row["kernel_fingerprint_reconstructed"] for row in prior}
    if len(fingerprints) != 120:
        raise RuntimeError("Original E1 oracle fingerprints are incomplete")
    rows = []
    for instance in sorted(manifest["instances"], key=lambda item: item["instance_id"]):
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        dynamic, optimum, random = generator.dynamic_range(mdp)
        for key, actual in (("dynamic_range", dynamic), ("optimal_return", optimum), ("random_return", random)):
            if not np.isclose(actual, float(instance[key]), rtol=0, atol=1e-10):
                raise RuntimeError(f"Fresh MDP oracle audit mismatch: {instance['instance_id']}/{key}")
        fingerprint = hashlib.sha256()
        for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
            fingerprint.update(array.tobytes())
        fingerprint.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
        identity = fingerprint.hexdigest()
        if identity in fingerprints:
            raise RuntimeError("New MDP aliases original E1 or another new MDP")
        fingerprints.add(identity)
        rows.append({"mdp_id": instance["instance_id"], "family": instance["family"],
                     "panel": _panel(instance["family"]), "kernel_fingerprint": identity,
                     **diagnostics(mdp), **finite_horizon_online_bounds(mdp)})
        if len(rows) % 20 == 0:
            print(f"Sealed oracle descriptors {len(rows)}/60", flush=True)
    if len(rows) != 60 or Counter(row["panel"] for row in rows) != {"primary": 42, "control": 18}:
        raise RuntimeError("Incomplete oracle stratum")
    write_csv(OUTPUT / "oracle_preoutcome.csv", rows, tuple(rows[0]))


def _predict_preoutcome(doc):
    import pandas as pd
    from scipy.special import ndtr
    import xgboost as xgb

    keys = ["mdp_id", "policy_family", "algorithm"]
    labels = pd.read_csv(LABELS)
    e1_features = pd.read_csv(SOURCE / "pilot_features/features.csv")
    query_features = pd.read_csv(ROUTING_OUTPUT / "pilot_features.csv")
    e1_oracle = pd.read_csv(ROOT / "artifacts/e1_oracle_qvi_posthoc_2026-09-30/oracle_diagnostics.csv")
    query_oracle = pd.read_csv(OUTPUT / "oracle_preoutcome.csv")
    train = labels.merge(e1_features, on=keys, validate="one_to_one").merge(
        e1_oracle[["mdp_id", "stochastic_eh_hbar_h100", "stochastic_eh_min_qvi_k_h100",
                   "stochastic_eh_delta_h100"]], on="mdp_id", validate="many_to_one")
    families = {row["mdp_id"]: row["panel"] for row in read_csv(OUTPUT / "oracle_preoutcome.csv")}
    query_features["panel"] = query_features["mdp_id"].map(families)
    query = query_features.merge(query_oracle[["mdp_id", "stochastic_eh_hbar_h100",
                                               "stochastic_eh_min_qvi_k_h100", "stochastic_eh_delta_h100"]],
                                 on="mdp_id", validate="many_to_one")
    if (len(train) != 1800 or len(query) != 900 or train[keys].duplicated().any()
            or query[keys].duplicated().any() or query.panel.isna().any()):
        raise RuntimeError("Incomplete E1-train/fresh-query difficulty feature grid")
    if set(train["event"]) - set(range(7)) or set(train["algorithm"]) != set(METHODS):
        raise RuntimeError("Invalid censored E1 training labels")
    oracle_columns = ("stochastic_eh_hbar_h100", "stochastic_eh_min_qvi_k_h100",
                      "stochastic_eh_delta_h100")
    signals = {**SIGNALS, "eh_oracle_aft": oracle_columns}
    output_rows = []
    for panel in PANELS:
        for algorithm in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == algorithm)]
            target = query[(query.panel == panel) & (query.algorithm == algorithm)]
            if len(source) != (84 * 5 if panel == "primary" else 36 * 5):
                raise RuntimeError("Incomplete E1 learner/panel training grid")
            if len(target) != (42 * 5 if panel == "primary" else 18 * 5):
                raise RuntimeError("Incomplete fresh learner/panel pilot grid")
            event = source["event"].to_numpy(dtype=int)
            counts = np.bincount(event, minlength=7).astype(float) / len(event)
            forecasts = {"train_frequency": np.repeat(counts[None, :], len(target), axis=0)}
            lower, upper = np.asarray([aft_interval(int(value)) for value in event], dtype=float).T
            for name, columns in signals.items():
                x = source[list(columns)].to_numpy(dtype=float)
                q = target[list(columns)].to_numpy(dtype=float)
                if name == "eh_oracle_aft":
                    x = np.where(np.isfinite(x), x, np.nan)
                    q = np.where(np.isfinite(q), q, np.nan)
                elif not np.isfinite(x).all() or not np.isfinite(q).all():
                    raise RuntimeError(f"Nonfinite pilot signal: {name}")
                fitted = xgb.DMatrix(x)
                fitted.set_float_info("label_lower_bound", lower)
                fitted.set_float_info("label_upper_bound", upper)
                model = xgb.train(XGB_PARAMS, fitted, num_boost_round=200)
                median = model.predict(xgb.DMatrix(q))
                if not np.isfinite(median).all() or np.any(median <= 0):
                    raise RuntimeError(f"Invalid AFT predictions: {name}")
                cdf = ndtr((np.log(np.asarray(GRID))[None, :] - np.log(median)[:, None]) / 1.2)
                probabilities = np.maximum(np.column_stack((cdf[:, 0], np.diff(cdf, axis=1), 1 - cdf[:, -1])), 0)
                forecasts[name] = probabilities / probabilities.sum(axis=1, keepdims=True)
            for index, (_, item) in enumerate(target.iterrows()):
                for name, column in (("oeh3_raw_proxy", "oeh3_log1p_requirement"),
                                     ("lcoeh4_raw_proxy", "lcoeh4_log1p_requirement")):
                    requirement = float(np.expm1(float(item[column])))
                    category = int(np.searchsorted(np.asarray(GRID), max(requirement, 0.), side="left"))
                    if name == "lcoeh4_raw_proxy" and float(item["lcoeh4_requirement_censored"]) > .5:
                        category = 6
                    raw = np.zeros(7)
                    raw[category] = 1.
                    forecasts[name] = forecasts.get(name, np.zeros((len(target), 7)))
                    forecasts[name][index] = raw
                for method, values in forecasts.items():
                    p = values[index]
                    if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-10):
                        raise RuntimeError("Invalid pre-outcome probability vector")
                    output_rows.append({"mdp_id": item.mdp_id, "policy_family": item.policy_family,
                                        "algorithm": algorithm, "panel": panel, "method": method,
                                        **{f"p_{category}": float(p[category]) for category in range(7)}})
    if len(output_rows) != 900 * (len(SIGNALS) + 4):
        raise RuntimeError("Incomplete independent difficulty predictions")
    write_csv(OUTPUT / "predictions_preoutcome.csv", output_rows, tuple(output_rows[0]))


def verified_preoutcome():
    doc = protocol()
    manifest, _ = verified_dataset()
    seal = load_json(OUTPUT / "preoutcome_seal.json")
    if seal["protocol_sha256"] != sha256(OUTPUT / "protocol.json") or seal["validation_outcomes_observed"] != 0:
        raise RuntimeError("Invalid pre-outcome difficulty seal")
    for name, digest in seal["artifact_sha256"].items():
        if sha256(OUTPUT / name) != digest:
            raise RuntimeError(f"Pre-outcome prediction/oracle artifact changed: {name}")
    return doc, manifest, seal


def _jobs(manifest, seed_index):
    records = {}
    for row in manifest["datasets"]:
        if row.get("benchmark_split") != "evaluation" or row.get("stream") != "training":
            continue
        key = (row["instance_id"], row["policy_family"],
               int(row["size_multiplier"]), seed_index(row))
        if key in records:
            raise RuntimeError("Duplicate difficulty training identity")
        records[key] = row
    if len(records) != 60 * 5 * 6 * 10:
        raise RuntimeError("Incomplete six-budget validation data grid")
    for (mdp, behavior, budget, seed), record in sorted(
            records.items(), key=lambda item: (item[0][0], item[0][1], item[0][3], item[0][2])):
        for method in METHODS:
            yield {"instance_id": mdp, "policy_family": behavior,
                   "size_multiplier": budget, "training_seed": seed,
                   "method": method, "buffer_file": record["file"],
                   "buffer_sha256": record["sha256"]}


def _job_file(job):
    identity = "|".join(str(job[key]) for key in
                        ("instance_id", "policy_family", "size_multiplier", "training_seed", "method"))
    return OUTPUT / "execution/jobs" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")


def _verify_job(job, path, plan_sha):
    payload = load_json(path)
    if payload["job"] != job or payload["execution_plan_sha256"] != plan_sha:
        raise RuntimeError(f"Difficulty job/plan identity changed: {path}")
    outcome = payload["outcome"]
    value = float(outcome["normalized_return"])
    if (outcome["status"] != "observed" or outcome["method"] != job["method"]
            or outcome["buffer_sha256"] != job["buffer_sha256"]
            or not np.isfinite(value) or bool(outcome["success"]) != (value >= .95)):
        raise RuntimeError(f"Invalid difficulty learner outcome: {path}")
    return payload


def evaluate(resume):
    doc, manifest, _ = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Recorded learner failure: no silent retry")
    pipeline, _, _, seed_index = archive_imports()
    jobs = list(_jobs(manifest, seed_index))
    if len(jobs) != doc["expected_core_jobs"]:
        raise RuntimeError("Difficulty job count differs from frozen protocol")
    plan = {"protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
            "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
            "jobs": jobs, "job_count": len(jobs)}
    path = OUTPUT / "execution_plan.json"
    if path.exists():
        if not resume or load_json(path) != plan:
            raise RuntimeError("Existing execution plan differs; verified --resume required")
    else:
        write_json(path, plan)
    plan_sha = sha256(path)
    (OUTPUT / "execution/jobs").mkdir(parents=True, exist_ok=True)
    callback = pipeline.build_core_train_score_callback(DATASET, manifest)
    resumed = 0
    start = time.perf_counter()
    for index, job in enumerate(jobs, 1):
        output = _job_file(job)
        if output.exists():
            if not resume:
                raise FileExistsError(f"Existing job without --resume: {output}")
            _verify_job(job, output, plan_sha)
            resumed += 1
            continue
        try:
            tick = time.perf_counter()
            outcome = callback(job)
            payload = {"job": job, "execution_plan_sha256": plan_sha,
                       "outcome": dict(outcome), "runtime_seconds": time.perf_counter() - tick,
                       "completed_at_utc": now()}
            write_json(output, payload)
            _verify_job(job, output, plan_sha)
        except Exception as exc:
            write_json(OUTPUT / "failure_evaluate.json", {
                "job_index": index, "job": job, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "plan_sha256": plan_sha, "failed_at_utc": now(),
            })
            raise
        if index % 500 == 0 or index == len(jobs):
            write_json(OUTPUT / "execution/progress.json", {
                "completed": index, "total": len(jobs), "resumed": resumed,
                "elapsed_seconds_this_process": time.perf_counter() - start,
                "updated_at_utc": now(),
            })
            print(f"Completed {index}/{len(jobs)} difficulty method-budget-seed jobs", flush=True)
    write_json(OUTPUT / "execution_complete.json", {
        "completed_at_utc": now(), "plan_sha256": plan_sha, "job_count": len(jobs),
    })


def analyze():
    from scipy.stats import spearmanr

    doc, manifest, seal = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Cannot analyze a failed learner panel")
    plan_path = OUTPUT / "execution_plan.json"
    plan = load_json(plan_path)
    complete = load_json(OUTPUT / "execution_complete.json")
    plan_sha = sha256(plan_path)
    if (complete["plan_sha256"] != plan_sha or plan["preoutcome_seal_sha256"] != sha256(OUTPUT / "preoutcome_seal.json")
            or plan["protocol_sha256"] != sha256(OUTPUT / "protocol.json")
            or len(plan["jobs"]) != doc["expected_core_jobs"]):
        raise RuntimeError("Complete difficulty execution seal mismatch")
    rows = []
    for job in plan["jobs"]:
        payload = _verify_job(job, _job_file(job), plan_sha)
        rows.append({"mdp_id": job["instance_id"], "policy_family": job["policy_family"],
                     "algorithm": job["method"], "budget": job["size_multiplier"],
                     "training_seed": job["training_seed"],
                     "normalized_return": float(payload["outcome"]["normalized_return"]),
                     "success": int(payload["outcome"]["success"]),
                     "buffer_sha256": job["buffer_sha256"],
                     "runtime_seconds": payload["runtime_seconds"]})
    if len(rows) != doc["expected_core_jobs"]:
        raise RuntimeError("Incomplete independent difficulty outcomes")
    write_csv(OUTPUT / "all_core_budget_seed_outcomes.csv", rows, tuple(rows[0]))
    counts = defaultdict(dict)
    for row in rows:
        key = row["mdp_id"], row["policy_family"], row["algorithm"]
        counts[key].setdefault(int(row["budget"]), []).append(row)
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    labels = []
    for key, by_budget in sorted(counts.items()):
        if set(by_budget) != set(GRID):
            raise RuntimeError(f"Missing budget for {key}")
        successes = []
        for budget in GRID:
            group = by_budget[budget]
            if len(group) != 10 or {int(item["training_seed"]) for item in group} != set(range(10)):
                raise RuntimeError(f"Missing/duplicate seed for {key}/{budget}")
            successes.append(sum(item["success"] for item in group))
        labels.append({"mdp_id": key[0], "policy_family": key[1], "algorithm": key[2],
                       "panel": _panel(instances[key[0]]["family"]),
                       "event": suffix_event(tuple(successes)),
                       **{f"successes_x{budget}": count for budget, count in zip(GRID, successes)}})
    if len(labels) != 900 or {item["mdp_id"] for item in labels} != set(instances):
        raise RuntimeError("Incomplete new requirement labels")
    write_csv(OUTPUT / "labels.csv", labels, tuple(labels[0]))
    label_by_key = {(item["mdp_id"], item["policy_family"], item["algorithm"]): item for item in labels}
    predictions = read_csv(OUTPUT / "predictions_preoutcome.csv")
    expected_methods = set(MODEL_METHODS) | {"eh_oracle_aft"}
    if len(predictions) != 900 * len(expected_methods):
        raise RuntimeError("Incomplete pre-outcome forecast table")
    seen, scored = set(), []
    for row in predictions:
        key = row["mdp_id"], row["policy_family"], row["algorithm"]
        identity = (*key, row["method"])
        if identity in seen or row["method"] not in expected_methods or key not in label_by_key:
            raise RuntimeError("Duplicate/unregistered prediction identity")
        seen.add(identity)
        label = label_by_key[key]
        if row["panel"] != label["panel"]:
            raise RuntimeError("Prediction/label panel mismatch")
        p = np.asarray([float(row[f"p_{category}"]) for category in range(7)])
        if np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-10):
            raise RuntimeError("Invalid pre-outcome probability vector")
        event = int(label["event"])
        actual = (event <= np.arange(6)).astype(float)
        brier = float(np.mean((np.cumsum(p)[:6] - actual) ** 2))
        scored.append({"mdp_id": key[0], "policy_family": key[1], "algorithm": key[2],
                       "panel": label["panel"], "method": row["method"], "event": event,
                       "integrated_brier": brier,
                       "seven_class_log_loss": float(-np.log(max(p[event], 1e-12))),
                       "event_category_correct": int(np.argmax(p) == event),
                       "predicted_success_by_x50": float(1 - p[6])})
    if len(seen) != 900 * len(expected_methods):
        raise RuntimeError("Incomplete scored prediction identities")
    write_csv(OUTPUT / "scored_predictions.csv", scored, tuple(scored[0]))
    grouped = defaultdict(list)
    for row in scored:
        grouped[(row["mdp_id"], row["method"])].append(row)
    mdp_rows = []
    for (mdp, method), group in sorted(grouped.items()):
        if len(group) != 15 or len({(item["policy_family"], item["algorithm"]) for item in group}) != 15:
            raise RuntimeError("Incomplete MDP/behavior/learner score block")
        mdp_rows.append({"mdp_id": mdp, "panel": group[0]["panel"], "method": method,
                         "integrated_brier": float(np.mean([item["integrated_brier"] for item in group])),
                         "seven_class_log_loss": float(np.mean([item["seven_class_log_loss"] for item in group])),
                         "event_category_accuracy": float(np.mean([item["event_category_correct"] for item in group])),
                         "predicted_success_by_x50": float(np.mean([item["predicted_success_by_x50"] for item in group]))})
    write_csv(OUTPUT / "mdp_scores.csv", mdp_rows, tuple(mdp_rows[0]))
    summary = []
    for panel in PANELS:
        for method in sorted(expected_methods):
            selected = [row for row in mdp_rows if row["panel"] == panel and row["method"] == method]
            if len(selected) != (42 if panel == "primary" else 18):
                raise RuntimeError("Incomplete panel/method MDP score block")
            summary.append({"panel": panel, "method": method, "mdps": len(selected),
                            "information_access": "oracle_full_mdp" if method == "eh_oracle_aft" else "pilot_only",
                            **{metric: float(np.mean([row[metric] for row in selected]))
                               for metric in ("integrated_brier", "seven_class_log_loss", "event_category_accuracy")}})
    write_csv(OUTPUT / "summary.csv", summary, tuple(summary[0]))
    # One common whole-MDP bootstrap for five predeclared data-only contrasts.
    ids = sorted(item["instance_id"] for item in manifest["instances"] if _panel(item["family"]) == "primary")
    by_mdp = {(row["mdp_id"], row["method"]): row for row in mdp_rows}
    baselines = ("train_frequency", "coverage_adaptation", "quality_adaptation",
                 "coverage_quality_adaptation", "b2_generic")
    differences = np.asarray([[by_mdp[(mdp, baseline)]["integrated_brier"]
                               - by_mdp[(mdp, "b2_plus_oeh")]["integrated_brier"]
                               for baseline in baselines] for mdp in ids])
    point = differences.mean(axis=0)
    rng = np.random.default_rng(20260930)
    bootstrap = np.asarray([differences[rng.integers(0, len(ids), len(ids))].mean(axis=0)
                            for _ in range(2000)])
    critical = float(np.quantile(np.max(bootstrap - point, axis=1), .95))
    contrasts = [{"panel": "primary", "baseline": baseline, "candidate": "b2_plus_oeh",
                  "mdps": len(ids), "mean_brier_improvement": float(point[index]),
                  "simultaneous_one_sided_lower_95": float(point[index] - critical)}
                 for index, baseline in enumerate(baselines)]
    write_csv(OUTPUT / "contrasts.csv", contrasts, tuple(contrasts[0]))
    oracle = read_csv(OUTPUT / "oracle_preoutcome.csv")
    labels_by_mdp = defaultdict(list)
    for item in labels:
        labels_by_mdp[item["mdp_id"]].append(int(item["event"]) < 6)
    oracle_by_mdp = {row["mdp_id"]: row for row in oracle}
    rank_rows = []
    oracle_fields = (("stochastic_eh_hbar_h100", -1), ("stochastic_eh_min_qvi_k_h100", -1),
                     ("epw_window_h100", -1), ("log_cover_online_bound_tL", -1),
                     ("log_ucb_sat_reference", -1))
    for panel in PANELS:
        panel_ids = sorted(mdp for mdp in instances if _panel(instances[mdp]["family"]) == panel)
        target = np.asarray([np.mean(labels_by_mdp[mdp]) for mdp in panel_ids])
        for method in sorted(expected_methods):
            predictor = np.asarray([by_mdp[(mdp, method)]["predicted_success_by_x50"] for mdp in panel_ids])
            coefficient = float(spearmanr(predictor, target).statistic) if len(set(predictor)) > 1 and len(set(target)) > 1 else float("nan")
            rank_rows.append({"panel": panel, "method": method, "information_access":
                              "oracle_full_mdp" if method == "eh_oracle_aft" else "pilot_only",
                              "mdps": len(panel_ids), "spearman_with_success_by_x50": coefficient})
        for field, polarity in oracle_fields:
            predictor = np.asarray([float(oracle_by_mdp[mdp][field]) for mdp in panel_ids]) * polarity
            valid = ~np.isnan(predictor)
            coefficient = float(spearmanr(predictor[valid], target[valid]).statistic) if len(set(predictor[valid])) > 1 and len(set(target[valid])) > 1 else float("nan")
            rank_rows.append({"panel": panel, "method": field, "information_access": "oracle_full_mdp",
                              "mdps": int(valid.sum()), "spearman_with_success_by_x50": coefficient})
    write_csv(OUTPUT / "rank_comparison.csv", rank_rows, tuple(rank_rows[0]))
    result = {
        "status": "complete", "completed_at_utc": now(),
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
        "execution_plan_sha256": plan_sha, "core_method_budget_seed_jobs": len(rows),
        "new_mdp_count": len(instances), "stochastic_primary_mdps": len(ids),
        "data_only_brier_gate": "GO" if all(item["simultaneous_one_sided_lower_95"] > 0 for item in contrasts) else "NO-GO",
        "scope": "Independent within-generator test of learner-specific offline requirement; published EH quantities are full-MDP online-oracle adaptations; no literature-wide SOTA claim",
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("all_core_budget_seed_outcomes.csv", "labels.csv", "scored_predictions.csv",
                             "mdp_scores.csv", "summary.csv", "contrasts.csv", "rank_comparison.csv")},
    }
    write_json(OUTPUT / "result_manifest.json", result)
    print(json.dumps({"data_only_brier_gate": result["data_only_brier_gate"],
                      "primary": [row for row in summary if row["panel"] == "primary"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "analyze"))
    parser.add_argument("--resume", action="store_true", help="Only for verified interrupted execution")
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": lambda: evaluate(args.resume), "analyze": analyze}[args.phase]()


if __name__ == "__main__":
    main()
