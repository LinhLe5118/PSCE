"""One-shot 120-MDP OEH offline-difficulty evaluation.

The candidate and all forecasts are sealed before learner outcomes. E1 is
meta-training only. Published EH is evaluated as an oracle descriptor transfer
to the offline endpoint, not as its original online theorem.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from scripts.analysis.compare_difficulty import FEATURES_B2, GRID, METHODS, suffix_event
from scripts.analysis.compare_e1_dataset_signals import SIGNALS
from scripts.analysis.develop_b2_eh_oracle_aft import E1_FEATURES, E1_LABELS, E1_ORACLE, EH, FIELDS, KEYS, P, ROOT, sha256
from scripts.analysis.develop_eh_oracle_three_way import aft_probabilities, numeric
from scripts.analysis.develop_multiclass_oeh import fit_multiclass
from scripts.analysis.develop_oeh_blend import FEATURES as OEH_FEATURES
from scripts.analysis.develop_oeh_difficulty import fit_thresholds, score_probabilities
from scripts.analysis.e1_oracle_qvi_diagnostic import archived_source_hash, finite_horizon_online_bounds
from scripts.analysis.run_unseen_validation import ARCHIVE, archive_imports, load_json, write_json


OUTPUT = ROOT / "artifacts/oeh_difficulty_unseen_120_2026-10-01"
DATASET = ROOT / "datasets/oeh_difficulty_unseen_120_2026-10-01"
E1_MANIFEST = ROOT / "datasets/week6_e1_oehv4e_confirmatory_v1_1/manifest.json"
DEV_GATE = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/joint_development_gate.json"
THREE_WAY = ROOT / "artifacts/oeh_difficulty_three_way_development_2026-10-01/manifest.json"
EH_THREE_WAY = ROOT / "artifacts/oeh_eh_oracle_three_way_development_2026-10-01/manifest.json"
NAMESPACE = "oeh-difficulty-unseen-120-v1-20261001"
FAMILY_COUNTS = {"uniform_garnet": 30, "forward_layered": 36,
                 "bottleneck": 18, "lock_structured": 36}
PRIMARY = 84
COMPARATORS = ("b2_three_way", "b2_plus_eh_oracle_three_way", "train_frequency",
               "coverage_adaptation", "quality_adaptation",
               "coverage_quality_adaptation", "b2_generic")
CANDIDATE = "oeh_three_way"
FORECAST_METHODS = (*COMPARATORS, CANDIDATE)
EXPECTED_JOBS = 120 * 5 * 3 * 6 * 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(mdp) -> str:
    digest = hashlib.sha256()
    for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
        digest.update(array.tobytes())
    digest.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
    return digest.hexdigest()


def prior_kernels() -> set[str]:
    first = ROOT / "artifacts/independent_difficulty_2026-09-30/oracle_preoutcome.csv"
    second = ROOT / "artifacts/oeh_b2_eh_oracle_second_development_2026-10-01/oracle_descriptors.csv"
    return (set(pd.read_csv(E1_ORACLE).kernel_fingerprint_reconstructed)
            | set(pd.read_csv(first).kernel_fingerprint)
            | set(pd.read_csv(second).kernel_fingerprint))


def freeze() -> None:
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError("A final evaluation artifact or dataset already exists")
    archive_imports()
    from environments.benchmark_environments import benchmark_specs, make_benchmark_mdp

    e1 = load_json(E1_MANIFEST)
    seal = load_json(ROOT / "artifacts/week6/E1_INPUT_SEAL_2026-09-26.json")
    if sha256(E1_MANIFEST) != seal["manifest_sha256"]:
        raise RuntimeError("Historical E1 manifest seal mismatch")
    if archived_source_hash(ARCHIVE, e1["generator_provenance"]["source_files"]) != e1["generator_provenance"]["source_sha256"]:
        raise RuntimeError("Archived benchmark generator changed")
    gate = load_json(DEV_GATE)
    if gate["development_gate"] != "GO" or gate["stochastic_mdps"] != 84:
        raise RuntimeError("Matched development gate did not pass")
    seen = prior_kernels()
    old_seeds = {int(row["seed"]) for row in e1["instances"]}
    for name in ("independent_validation_2026-09-30", "oeh_difficulty_confirmation_2026-10-01"):
        path = ROOT / "datasets" / name / "manifest.json"
        if path.exists():
            old_seeds.update(int(row["seed"]) for row in load_json(path)["instances"])
    specs, kernels = [], {}
    templates = [spec for spec in benchmark_specs() if spec.split == "evaluation"]
    for family, count in FAMILY_COUNTS.items():
        selected = sorted((spec for spec in templates if spec.family == family),
                          key=lambda spec: spec.ordinal)
        if len(selected) != count:
            raise RuntimeError(f"Unexpected E1 recipe count: {family}")
        for ordinal, template in enumerate(selected):
            name = f"oeh120_{family}_{ordinal:03d}"
            for retry in range(100):
                seed_bytes = hashlib.sha256(f"{NAMESPACE}|{template.instance_id}|{retry}".encode()).digest()
                seed = 100_000_000 + int.from_bytes(seed_bytes[:8], "little") % 1_000_000_000
                if seed in old_seeds:
                    continue
                spec = replace(template, instance_id=name, seed=seed)
                kernel = fingerprint(make_benchmark_mdp(spec))
                if kernel not in seen:
                    break
            else:
                raise RuntimeError(f"Could not find unseen kernel: {name}")
            seen.add(kernel)
            old_seeds.add(seed)
            specs.append(spec)
            kernels[name] = kernel
    if len(specs) != 120 or len(kernels) != 120:
        raise RuntimeError("Incomplete untouched 120-MDP cohort")
    source_files = (Path(__file__), E1_LABELS, E1_FEATURES, E1_ORACLE,
                    E1_MANIFEST, DEV_GATE, THREE_WAY, EH_THREE_WAY,
                    ROOT / "scripts/analysis/develop_three_way_ensemble.py",
                    ROOT / "scripts/analysis/develop_eh_oracle_three_way.py",
                    ROOT / "scripts/analysis/develop_oeh_difficulty.py",
                    ROOT / "scripts/analysis/develop_multiclass_oeh.py",
                    ARCHIVE / "evaluation/e1_pipeline.py",
                    ARCHIVE / "scripts/data/generate_offline_datasets.py",
                    ARCHIVE / "environments/benchmark_environments.py")
    protocol = {
        "schema": "oeh-unseen-120-difficulty-v1",
        "status": "frozen_before_dataset_buffers_or_learner_outcomes",
        "namespace": NAMESPACE, "requested_specs": [spec.to_dict() for spec in specs],
        "kernel_fingerprints": kernels, "family_counts": FAMILY_COUNTS,
        "stochastic_primary_mdps": PRIMARY, "control_mdps": 36,
        "pilot_multipliers": [2, 5, 10], "budgets": GRID,
        "training_seed_indices": list(range(10)), "behavior_families": 5,
        "core_learners": METHODS, "expected_jobs": EXPECTED_JOBS,
        "target": "first complete suffix with >=8/10 seeds normalized return >=0.95; else >x50",
        "candidate": CANDIDATE, "comparators": COMPARATORS,
        "forecast_architecture": "same E1-trained AFT/direct-threshold/multiclass convex ensemble for B2, B2+oracle-EH, and B2+pilot-OEH",
        "weights": {"b2_three_way": load_json(THREE_WAY)["weights"]["b2_three_way"]["aft_direct_multi"],
                    CANDIDATE: load_json(THREE_WAY)["weights"][CANDIDATE]["aft_direct_multi"],
                    "b2_plus_eh_oracle_three_way": load_json(EH_THREE_WAY)["e1_oof_weights_aft_direct_multi"]},
        "decision_rule": "on 84 stochastic MDPs: OEH point relative gains >=5% in integrated Brier and seven-class log loss versus every comparator, and all simultaneous one-sided 95% MDP-cluster absolute-improvement lower bounds >0",
        "bootstrap": {"draws": 10000, "seed": 20261001, "joint_contrasts": 2 * len(COMPARATORS)},
        "control_rule": "report 36 control MDPs separately; never select candidate on them",
        "scope": "EH is a full-MDP oracle descriptor transfer to offline difficulty, not the original online theorem; data-quality comparators are tabular adaptations, not exact original implementations",
        "original_generator_sha256": e1["generator_provenance"]["source_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in source_files},
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "protocol.sha256").write_text(sha256(OUTPUT / "protocol.json") + "\n", encoding="ascii")
    print(f"Frozen 120 distinct unseen MDP kernels: {sha256(OUTPUT / 'protocol.json')}", flush=True)


def protocol() -> dict:
    path = OUTPUT / "protocol.json"
    if sha256(path) != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Protocol seal mismatch")
    doc = load_json(path)
    if doc["schema"] != "oeh-unseen-120-difficulty-v1":
        raise RuntimeError("Wrong protocol schema")
    for relative, expected in doc["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    return doc


def dataset() -> dict:
    doc = protocol()
    seal = load_json(OUTPUT / "dataset_seal.json")
    if seal["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise RuntimeError("Dataset/protocol seal mismatch")
    for relative, expected in seal["sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Dataset seal mismatch: {relative}")
    manifest = load_json(DATASET / "manifest.json")
    if manifest["instance_count"] != 120 or manifest["physical_buffer_count"] != 6600:
        raise RuntimeError("Incomplete 120-MDP dataset")
    if manifest["generator_provenance"]["source_sha256"] != doc["original_generator_sha256"]:
        raise RuntimeError("Generator source changed")
    return manifest


def pilot_features(manifest: dict) -> pd.DataFrame:
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
            raise RuntimeError("Duplicate pilot dataset")
        grouped[key][size] = row
    if len(grouped) != 120 * 5 or any(set(group) != {2, 5, 10} for group in grouped.values()):
        raise RuntimeError("Incomplete pilot grid")
    rows = []
    for group, prefixes in sorted(grouped.items()):
        rows.extend(e1_pipeline._group_features(DATASET, group, prefixes))
    frame = pd.DataFrame(rows)
    if len(frame) != 1800 or frame[KEYS].duplicated().any():
        raise RuntimeError("Incomplete pilot features")
    frame.to_csv(OUTPUT / "pilot_features.csv", index=False)
    return frame


def oracle_descriptors(manifest: dict, doc: dict) -> pd.DataFrame:
    from environments.benchmark_environments import make_benchmark_mdp

    rows = []
    for instance in manifest["instances"]:
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        digest = fingerprint(mdp)
        if digest != doc["kernel_fingerprints"][instance["instance_id"]]:
            raise RuntimeError(f"Frozen kernel changed: {instance['instance_id']}")
        descriptors = finite_horizon_online_bounds(mdp)
        rows.append({"mdp_id": instance["instance_id"],
                     "kernel_fingerprint": digest,
                     **{name: descriptors[name] for name in EH}})
    frame = pd.DataFrame(rows)
    if len(frame) != 120 or frame.kernel_fingerprint.nunique() != 120:
        raise RuntimeError("Final MDP kernels are not all distinct")
    frame.to_csv(OUTPUT / "oracle_descriptors_preoutcome.csv", index=False)
    return frame


def prepare() -> None:
    if (OUTPUT / "preoutcome_seal.json").exists():
        raise FileExistsError("Sealed forecasts already exist")
    doc = protocol()
    if DATASET.exists():
        manifest = dataset()
    else:
        _, _, generator, _ = archive_imports()
        from environments.benchmark_environments import BenchmarkSpec

        specs = [BenchmarkSpec(**item) for item in doc["requested_specs"]]
        manifest = generator.generate_benchmark_bundle(
            DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)),
            max_size=None, horizon=100, run_mode="test",
            deviation_reason="Frozen untouched 120-MDP OEH difficulty evaluation",
            pilot_size_multipliers=(2, 5, 10))
        if manifest["instance_count"] != 120 or manifest["physical_buffer_count"] != 6600:
            raise RuntimeError("Unexpected final dataset count")
        paths = (DATASET / "manifest.json", DATASET / "data_only_index.json")
        write_json(OUTPUT / "dataset_seal.json", {
            "protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "sealed_at_utc": now(),
            "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                       for path in paths},
        })
    query = pilot_features(manifest)
    oracle = oracle_descriptors(manifest, doc)
    train = pd.read_csv(E1_LABELS).merge(pd.read_csv(E1_FEATURES), on=KEYS,
                                         validate="one_to_one")
    train = train.merge(pd.read_csv(E1_ORACLE)[["mdp_id", *EH]], on="mdp_id",
                        validate="many_to_one")
    query = query.merge(oracle[["mdp_id", *EH]], on="mdp_id", validate="many_to_one")
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    query["panel"] = query.mdp_id.map(lambda item: "control" if family[item] == "lock_structured" else "primary")
    if len(train) != 1800 or len(query) != 1800 or set(train.mdp_id) & set(query.mdp_id):
        raise RuntimeError("Training and final evaluation grids overlap or are incomplete")
    field_sets = {"b2_three_way": FEATURES_B2,
                  CANDIDATE: OEH_FEATURES,
                  "b2_plus_eh_oracle_three_way": FIELDS}
    forecasts = []
    for panel in ("primary", "control"):
        for learner in METHODS:
            source = train[(train.panel == panel) & (train.algorithm == learner)]
            selected = query[(query.panel == panel) & (query.algorithm == learner)]
            if len(source) != (420 if panel == "primary" else 180) or len(selected) != (420 if panel == "primary" else 180):
                raise RuntimeError("Wrong learner/panel forecast size")
            y = source.event.to_numpy(dtype=int)
            probability = {"train_frequency": np.repeat(
                (np.bincount(y, minlength=7) / len(y))[None, :], len(selected), axis=0)}
            for name, fields in field_sets.items():
                x = numeric(source) if name == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = numeric(selected) if name == "b2_plus_eh_oracle_three_way" else np.nan_to_num(
                    selected[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                parts = np.asarray((aft_probabilities(x, y, q),
                                    fit_thresholds(x, y, q), fit_multiclass(x, y, q)))
                probability[name] = np.einsum("i,ijk->jk", doc["weights"][name], parts)
                if name == "b2_three_way":
                    probability["b2_generic"] = parts[0]
            for name in ("coverage_adaptation", "quality_adaptation", "coverage_quality_adaptation"):
                fields = SIGNALS[name]
                x = np.nan_to_num(source[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                q = np.nan_to_num(selected[list(fields)].to_numpy(dtype=float), nan=0., posinf=1e6, neginf=-1e6)
                probability[name] = aft_probabilities(x, y, q)
            for name in FORECAST_METHODS:
                p = probability[name]
                if p.shape != (len(selected), 7) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.):
                    raise RuntimeError(f"Invalid {name} forecast")
                frame = selected[KEYS + ["panel"]].copy()
                frame["method"] = name
                for i, col in enumerate(P):
                    frame[col] = p[:, i]
                forecasts.append(frame)
            print(f"Sealed 120-MDP forecasts: {panel}/{learner}", flush=True)
    predictions = pd.concat(forecasts, ignore_index=True)
    if len(predictions) != 1800 * len(FORECAST_METHODS) or predictions.duplicated(KEYS + ["method"]).any():
        raise RuntimeError("Incomplete final preoutcome forecast panel")
    predictions.to_csv(OUTPUT / "predictions_preoutcome.csv", index=False)
    write_json(OUTPUT / "preoutcome_seal.json", {
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "dataset_seal_sha256": sha256(OUTPUT / "dataset_seal.json"),
        "learner_outcomes_observed": 0, "sealed_at_utc": now(),
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("pilot_features.csv", "oracle_descriptors_preoutcome.csv",
                             "predictions_preoutcome.csv")},
    })
    print(f"Sealed {len(predictions)} forecasts before any final learner outcome", flush=True)


def verified_preoutcome() -> tuple[dict, dict]:
    doc, manifest = protocol(), dataset()
    seal = load_json(OUTPUT / "preoutcome_seal.json")
    if (seal["protocol_sha256"] != sha256(OUTPUT / "protocol.json")
            or seal["dataset_seal_sha256"] != sha256(OUTPUT / "dataset_seal.json")
            or seal["learner_outcomes_observed"] != 0):
        raise RuntimeError("Preoutcome seal mismatch")
    for name, expected in seal["artifact_sha256"].items():
        if sha256(OUTPUT / name) != expected:
            raise RuntimeError(f"Preoutcome artifact changed: {name}")
    return doc, manifest


def jobs(manifest: dict, seed_index) -> list[dict]:
    records = {}
    for row in manifest["datasets"]:
        if row.get("benchmark_split") != "evaluation" or row.get("stream") != "training":
            continue
        key = (row["instance_id"], row["policy_family"],
               int(row["size_multiplier"]), seed_index(row))
        if key in records:
            raise RuntimeError("Duplicate final training dataset")
        records[key] = row
    if len(records) != 120 * 5 * 6 * 10:
        raise RuntimeError("Incomplete final training data grid")
    planned = []
    for (mdp, behavior, budget, seed), record in sorted(records.items()):
        for learner in METHODS:
            planned.append({"instance_id": mdp, "policy_family": behavior,
                            "size_multiplier": budget, "training_seed": seed,
                            "method": learner, "buffer_file": record["file"],
                            "buffer_sha256": record["sha256"]})
    return planned


def job_path(job: dict) -> Path:
    identity = "|".join(str(job[name]) for name in
                        ("instance_id", "policy_family", "size_multiplier",
                         "training_seed", "method"))
    return OUTPUT / "execution/jobs" / (hashlib.sha256(identity.encode()).hexdigest() + ".json")


def verify_job(job: dict, path: Path, plan_sha: str) -> dict:
    row = load_json(path)
    outcome = row["outcome"]
    value = float(outcome["normalized_return"])
    if (row["job"] != job or row["execution_plan_sha256"] != plan_sha
            or outcome["status"] != "observed" or outcome["method"] != job["method"]
            or outcome["buffer_sha256"] != job["buffer_sha256"]
            or not np.isfinite(value) or bool(outcome["success"]) != (value >= .95)):
        raise RuntimeError(f"Invalid final evaluation job: {path}")
    return row


def evaluate(resume: bool) -> None:
    doc, manifest = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("A recorded learner failure requires audit")
    pipeline, _, _, seed_index = archive_imports()
    planned = jobs(manifest, seed_index)
    if len(planned) != doc["expected_jobs"]:
        raise RuntimeError("Final job count differs from frozen protocol")
    plan = {"protocol_sha256": sha256(OUTPUT / "protocol.json"),
            "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
            "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
            "jobs": planned, "job_count": len(planned)}
    plan_path = OUTPUT / "execution_plan.json"
    if plan_path.exists():
        if not resume or load_json(plan_path) != plan:
            raise RuntimeError("Resume requires the identical sealed execution plan")
    else:
        write_json(plan_path, plan)
    plan_sha = sha256(plan_path)
    (OUTPUT / "execution/jobs").mkdir(parents=True, exist_ok=True)
    callback = pipeline.build_core_train_score_callback(DATASET, manifest)
    started, skipped = time.perf_counter(), 0
    for index, job in enumerate(planned, 1):
        path = job_path(job)
        if path.exists():
            if not resume:
                raise FileExistsError(f"Existing job without --resume: {path}")
            verify_job(job, path, plan_sha)
            skipped += 1
            continue
        try:
            tick = time.perf_counter()
            outcome = callback(job)
            write_json(path, {"job": job, "execution_plan_sha256": plan_sha,
                              "outcome": dict(outcome),
                              "runtime_seconds": time.perf_counter() - tick,
                              "completed_at_utc": now()})
            verify_job(job, path, plan_sha)
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
            print(f"Unseen 120-MDP learner jobs {index}/{len(planned)}", flush=True)
    write_json(OUTPUT / "execution_complete.json", {
        "completed_at_utc": now(), "plan_sha256": plan_sha,
        "job_count": len(planned)})


def analyze() -> None:
    doc, manifest = verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Failed learner panel cannot be selectively analyzed")
    plan_path = OUTPUT / "execution_plan.json"
    plan = load_json(plan_path)
    complete = load_json(OUTPUT / "execution_complete.json")
    plan_sha = sha256(plan_path)
    if (complete["plan_sha256"] != plan_sha or complete["job_count"] != EXPECTED_JOBS
            or plan["preoutcome_seal_sha256"] != sha256(OUTPUT / "preoutcome_seal.json")
            or len(plan["jobs"]) != EXPECTED_JOBS):
        raise RuntimeError("Final learner outcome panel incomplete")
    outcomes, grouped = [], defaultdict(dict)
    for job in plan["jobs"]:
        payload = verify_job(job, job_path(job), plan_sha)
        outcome = payload["outcome"]
        key = job["instance_id"], job["policy_family"], job["method"]
        budget = int(job["size_multiplier"])
        grouped[key].setdefault(budget, []).append(int(outcome["success"]))
        outcomes.append({"mdp_id": key[0], "policy_family": key[1],
                         "algorithm": key[2], "budget": budget,
                         "training_seed": job["training_seed"],
                         "normalized_return": float(outcome["normalized_return"]),
                         "success": int(outcome["success"]),
                         "runtime_seconds": payload["runtime_seconds"]})
    pd.DataFrame(outcomes).to_csv(OUTPUT / "all_core_budget_seed_outcomes.csv", index=False)
    family = {item["instance_id"]: item["family"] for item in manifest["instances"]}
    labels = []
    for (mdp, behavior, learner), budgets in sorted(grouped.items()):
        if set(budgets) != set(GRID) or any(len(budgets[b]) != 10 for b in GRID):
            raise RuntimeError("Incomplete final budget/seed grid")
        counts = tuple(sum(budgets[b]) for b in GRID)
        labels.append({"mdp_id": mdp, "policy_family": behavior,
                       "algorithm": learner,
                       "panel": "control" if family[mdp] == "lock_structured" else "primary",
                       "event": suffix_event(counts),
                       **{f"successes_x{budget}": n for budget, n in zip(GRID, counts)}})
    labels = pd.DataFrame(labels)
    if len(labels) != 1800 or labels[KEYS].duplicated().any():
        raise RuntimeError("Incomplete final difficulty labels")
    labels.to_csv(OUTPUT / "labels.csv", index=False)
    predictions = pd.read_csv(OUTPUT / "predictions_preoutcome.csv")
    scored = predictions.merge(labels[KEYS + ["panel", "event"]], on=KEYS,
                               suffixes=("_forecast", ""), validate="many_to_one")
    if (len(scored) != 1800 * len(FORECAST_METHODS)
            or (scored.panel_forecast != scored.panel).any()):
        raise RuntimeError("Final predictions and labels differ")
    brier, logloss = score_probabilities(scored[P].to_numpy(dtype=float),
                                         scored.event.to_numpy(dtype=int))
    scored["integrated_brier"] = brier
    scored["seven_class_log_loss"] = logloss
    scored.drop(columns="panel_forecast").to_csv(OUTPUT / "scored_predictions.csv", index=False)
    mdp = scored.groupby(["mdp_id", "panel", "method"], as_index=False).agg(
        tasks=("event", "size"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    if len(mdp) != 120 * len(FORECAST_METHODS) or not (mdp.tasks == 15).all():
        raise RuntimeError("Incomplete within-MDP forecast scores")
    mdp.to_csv(OUTPUT / "mdp_scores.csv", index=False)
    summary = mdp.groupby(["panel", "method"], as_index=False).agg(
        mdps=("mdp_id", "nunique"), integrated_brier=("integrated_brier", "mean"),
        seven_class_log_loss=("seven_class_log_loss", "mean"))
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    primary = mdp[mdp.panel == "primary"]
    ids = sorted(primary.mdp_id.unique())
    if len(ids) != PRIMARY:
        raise RuntimeError("Expected 84 stochastic primary MDPs")
    indexed = primary.set_index(["mdp_id", "method"])
    effects, contrasts = [], []
    for baseline in COMPARATORS:
        for metric in ("integrated_brier", "seven_class_log_loss"):
            before = np.asarray([indexed.loc[(mdp_id, baseline), metric] for mdp_id in ids])
            after = np.asarray([indexed.loc[(mdp_id, CANDIDATE), metric] for mdp_id in ids])
            effects.append(before - after)
            contrasts.append({"baseline": baseline, "candidate": CANDIDATE,
                              "metric": metric, "mdps": PRIMARY,
                              "baseline_mean": float(before.mean()),
                              "candidate_mean": float(after.mean()),
                              "relative_gain": float(1 - after.mean() / before.mean()),
                              "mean_absolute_improvement": float((before - after).mean())})
    effect = np.column_stack(effects)
    mean = effect.mean(axis=0)
    se = effect.std(axis=0, ddof=1) / np.sqrt(PRIMARY)
    draws = np.random.default_rng(doc["bootstrap"]["seed"]).integers(
        0, PRIMARY, size=(doc["bootstrap"]["draws"], PRIMARY))
    critical = float(np.quantile(np.max((effect[draws].mean(axis=1) - mean) / se,
                                        axis=1), .95))
    lower = mean - critical * se
    for i, row in enumerate(contrasts):
        row["simultaneous_one_sided_lower_95"] = float(lower[i])
    pd.DataFrame(contrasts).to_csv(OUTPUT / "contrasts.csv", index=False)
    go = all(row["relative_gain"] >= .05 and row["simultaneous_one_sided_lower_95"] > 0
             for row in contrasts)
    artifact_names = ("all_core_budget_seed_outcomes.csv", "labels.csv",
                      "scored_predictions.csv", "mdp_scores.csv", "summary.csv", "contrasts.csv")
    write_json(OUTPUT / "result_manifest.json", {
        "status": "complete", "completed_at_utc": now(),
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "preoutcome_seal_sha256": sha256(OUTPUT / "preoutcome_seal.json"),
        "execution_plan_sha256": plan_sha, "jobs": EXPECTED_JOBS,
        "mdps": 120, "stochastic_primary_mdps": PRIMARY,
        "simultaneous_critical_value": critical,
        "difficulty_gate": "GO" if go else "NO-GO",
        "scope": doc["scope"],
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in artifact_names},
    })
    print(summary.to_string(index=False), flush=True)
    print(f"Unseen 120-MDP difficulty gate: {'GO' if go else 'NO-GO'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "analyze"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"freeze": freeze, "prepare": prepare,
     "evaluate": lambda: evaluate(args.resume), "analyze": analyze}[args.phase]()


if __name__ == "__main__":
    main()
