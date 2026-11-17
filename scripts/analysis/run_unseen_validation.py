"""Audited, single-protocol independent validation after E1 development.

Run prepare, evaluate, analyze in order. Never inspect validation outcomes to
revise the frozen route, cohort, baselines, or endpoints. All per-job records
are retained, including failures and wall-clock runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from scripts.analysis.compare_difficulty import sha256


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tmp/oeh_development_2026-09-30"
OUTPUT = ROOT / "artifacts/independent_validation_2026-09-30"
DATASET = ROOT / "datasets/independent_validation_2026-09-30"
CORE = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
EXTERNAL = ("pib_spibb_n10_author", "vi_lcb_cb005_author_empirical",
            "discrete_bcq_author", "discrete_cql_d3rlpy")
GROUP_FIELDS = ("mdp_id", "policy_family")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive_imports():
    from scripts.restore_frozen_runtime import restore_runtime
    restore_runtime(ARCHIVE)
    sys.path.insert(0, str(ARCHIVE))
    sys.path.insert(0, str(ARCHIVE / "code"))
    sys.path.insert(0, str(ROOT / "code"))
    import scripts
    archived_scripts = str(ARCHIVE / "scripts")
    if archived_scripts not in scripts.__path__:
        scripts.__path__.insert(0, archived_scripts)
    from evaluation import e1_pipeline, e1_external
    from scripts.data import generate_offline_datasets as generator
    from scripts.training.run_empirical_requirement import _training_seed_index
    for module in (e1_pipeline, e1_external, generator):
        if not Path(module.__file__).resolve().is_relative_to(ARCHIVE.resolve()):
            raise RuntimeError("Frozen runner must start in a fresh Python process; a root runtime module was already imported")
    return e1_pipeline, e1_external, generator, _training_seed_index


def protocol():
    path = OUTPUT / "protocol.json"
    expected = (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip()
    if sha256(path) != expected:
        raise RuntimeError("Frozen protocol hash changed")
    document = load_json(path)
    if document["schema"] != "oeh-independent-validation-v1":
        raise RuntimeError("Unexpected protocol schema")
    for relative, digest in document["input_source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Frozen source/input changed: {relative}")
    return document, expected


def verified_dataset():
    seal = load_json(OUTPUT / "dataset_seal.json")
    for relative, digest in seal["sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Validation dataset seal changed: {relative}")
    manifest = load_json(DATASET / "manifest.json")
    if manifest["instance_count"] != 60 or manifest["physical_buffer_count"] != 3300:
        raise RuntimeError("Incomplete fresh MDP cohort")
    if manifest["generator_provenance"]["source_sha256"] != load_json(OUTPUT / "protocol.json")["original_generator_source_sha256"]:
        raise RuntimeError("Historical generator source mismatch")
    return manifest, seal


def _pilot_groups(manifest):
    index = load_json(DATASET / "data_only_index.json")
    labels = {row["data_only_dataset_id"]: row for row in manifest["datasets"]
              if row.get("benchmark_split") == "evaluation"}
    grouped = defaultdict(dict)
    for row in index["datasets"]:
        if row.get("stream") != "pilot" or int(row.get("size_multiplier", -1)) not in (2, 5, 10):
            continue
        label = labels[row["dataset_id"]]
        for key in ("file", "sha256", "size", "slice", "num_states", "num_actions", "initial_state_distribution"):
            if row[key] != label[key]:
                raise RuntimeError(f"Data-only/manifest pilot mismatch: {key}")
        group = label["instance_id"], label["policy_family"]
        size = int(row["size_multiplier"])
        if size in grouped[group]:
            raise RuntimeError("Duplicate pilot prefix")
        grouped[group][size] = row
    if len(grouped) != 300 or any(set(rows) != {2, 5, 10} for rows in grouped.values()):
        raise RuntimeError("Incomplete 60-MDP pilot grid")
    return grouped


def _load_e1_training():
    path = ROOT / "artifacts/provenance/sources/evaluation/e1_reuse.py"
    spec = importlib.util.spec_from_file_location("e1_reuse_pinned", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from evaluation.io import read_csv_rows
    from oeh.router import attach_results
    panel = module.load_e1_panel(ROOT)
    external = read_csv_rows(ROOT / "artifacts/week6/oehv4e_confirmatory_v1_1/external_run_results.csv")
    return attach_results(panel, external, panel["training_records"], EXTERNAL)


def prepare():
    frozen, digest = protocol()
    if (OUTPUT / "choices_seal.json").exists():
        raise FileExistsError("Validation choices already sealed")
    e1_pipeline, _, generator, _ = archive_imports()
    if DATASET.exists() and any(DATASET.iterdir()):
        manifest, _ = verified_dataset()
    else:
        from environments.benchmark_environments import BenchmarkSpec
        specs = [BenchmarkSpec(**row) for row in frozen["requested_specs"]]
        manifest = generator.generate_benchmark_bundle(
            DATASET, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)),
            max_size=None, horizon=100, run_mode="test",
            deviation_reason="Fresh post-E1 independent validation with frozen new MDP seeds",
            pilot_size_multipliers=(2, 5, 10),
        )
        if manifest["instance_count"] != 60 or manifest["physical_buffer_count"] != 3300:
            raise RuntimeError("Unexpected generated cohort size")
        paths = (DATASET / "manifest.json", DATASET / "data_only_index.json")
        write_json(OUTPUT / "dataset_seal.json", {
            "sealed_at_utc": now(), "protocol_sha256": digest,
            "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths},
        })
    groups = _pilot_groups(manifest)
    feature_path = OUTPUT / "pilot_features.csv"
    names = e1_pipeline._feature_names()
    if frozen.get("pilot_feature_sha256"):
        if sha256(feature_path) != frozen["pilot_feature_sha256"]:
            raise RuntimeError("Pre-outcome pilot feature checkpoint hash changed")
        with feature_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for group, prefixes in sorted(groups.items()):
            rows.extend(e1_pipeline._group_features(DATASET, group, prefixes))
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("mdp_id", "policy_family", "algorithm", *names["B2_OEH3_LCOEH4"]))
            writer.writeheader()
            writer.writerows(rows)
    if len(rows) != 900:
        raise RuntimeError("Incomplete pilot-only feature rows")
    features = {
        (row["mdp_id"], row["policy_family"], row["algorithm"]):
            {block: tuple(float(row[field]) for field in names[block])
             for block in ("B2_OEH3", "B2_OEH3_LCOEH4")}
        for row in rows
    }
    if len(features) != 900 or {key[:2] for key in features} != set(groups):
        raise RuntimeError("Pilot feature identities do not match all new groups")
    # Panel is the predeclared benchmark recipe stratum, never inferred from
    # held-out returns or inspected transition-kernel/oracle audit fields.
    panels = {row["instance_id"]: ("deterministic_control" if row["family"] == "lock_structured"
                                   else "stochastic_primary") for row in manifest["instances"]}
    if Counter(panels.values()) != {"stochastic_primary": 42, "deterministic_control": 18}:
        raise RuntimeError("Fresh panel allocation differs from frozen protocol")
    from oeh.unseen import fit_unseen_routes
    panel = _load_e1_training()
    choices, forecasts, tuning = fit_unseen_routes(panel, features, panels)
    if len(choices) != 300 or set(choices.values()) - set(frozen["candidate_methods"]):
        raise RuntimeError("Incomplete unseen route")
    choice_rows = [{"mdp_id": mdp, "policy_family": behavior, "selected_method": choice,
                    "forecasts": forecasts[(mdp, behavior)]}
                   for (mdp, behavior), choice in sorted(choices.items())]
    write_json(OUTPUT / "choices.json", choice_rows)
    write_json(OUTPUT / "model_tuning.json", tuning)
    paths = (feature_path, OUTPUT / "choices.json", OUTPUT / "model_tuning.json")
    write_json(OUTPUT / "choices_seal.json", {
        "sealed_before_new_training_outcomes_at_utc": now(),
        "protocol_sha256": digest,
        "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
        "sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths},
        "groups": len(choices), "method_counts": dict(Counter(choices.values())),
    })
    print(f"Sealed {len(choices)} unlabeled choices, SHA-256 {sha256(OUTPUT / 'choices.json')}", flush=True)


def verified_choices():
    seal = load_json(OUTPUT / "choices_seal.json")
    if seal["dataset_manifest_sha256"] != sha256(DATASET / "manifest.json"):
        raise RuntimeError("Choice/dataset seal mismatch")
    for relative, digest in seal["sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Sealed choice input changed: {relative}")
    rows = load_json(OUTPUT / "choices.json")
    choices = {(row["mdp_id"], row["policy_family"]): row["selected_method"] for row in rows}
    if len(rows) != 300 or len(choices) != 300:
        raise RuntimeError("Invalid sealed choice grid")
    return choices, seal


def _jobs(manifest, methods, training_seed_index):
    records = {}
    for row in manifest["datasets"]:
        if row.get("benchmark_split") != "evaluation" or row.get("stream") != "training" or int(row["size_multiplier"]) != 10:
            continue
        key = (row["instance_id"], row["policy_family"], training_seed_index(row))
        if key in records:
            raise RuntimeError("Duplicate x10 training identity")
        records[key] = row
    if len(records) != 3000:
        raise RuntimeError("Incomplete x10 training grid")
    for (mdp, behavior, seed), record in sorted(records.items()):
        for method in methods:
            yield {"instance_id": mdp, "policy_family": behavior, "training_seed": seed,
                   "size_multiplier": 10, "method": method,
                   "buffer_file": record["file"], "buffer_sha256": record["sha256"]}


def _job_name(job):
    identity = "|".join(str(job[field]) for field in
                        ("instance_id", "policy_family", "training_seed", "method"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"


def _verify_job(path, job, plan_sha):
    payload = load_json(path)
    if payload["job"] != job or payload["execution_plan_sha256"] != plan_sha:
        raise RuntimeError(f"Job identity/seal mismatch: {path}")
    outcome = payload["outcome"]
    if (outcome["status"] != "observed" or outcome["method"] != job["method"]
            or outcome["buffer_sha256"] != job["buffer_sha256"]
            or not np.isfinite(float(outcome["normalized_return"]))
            or bool(outcome["success"]) != (float(outcome["normalized_return"]) >= .95)):
        raise RuntimeError(f"Invalid observed job result: {path}")
    return payload


def evaluate(resume):
    frozen, protocol_sha = protocol()
    manifest, dataset_seal = verified_dataset()
    choices, choice_seal = verified_choices()
    if choice_seal["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Choices were sealed under a different protocol")
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("A learner failure was recorded; retain it for audit")
    pipeline, external, _, seed_index = archive_imports()
    provenance = external.verify_external_components(ROOT / "tmp")
    if tuple(frozen["candidate_methods"]) != CORE + EXTERNAL:
        raise RuntimeError("Frozen learner library differs from E1")
    jobs = list(_jobs(manifest, frozen["candidate_methods"], seed_index))
    if len(jobs) != 21000 or {job["instance_id"] for job in jobs} != set(choices_mdp for choices_mdp, _ in choices):
        raise RuntimeError("Incomplete method/seed/MDP plan")
    plan_path = OUTPUT / "execution_plan.json"
    plan = {"protocol_sha256": protocol_sha, "dataset_seal_sha256": sha256(OUTPUT / "dataset_seal.json"),
            "choices_seal_sha256": sha256(OUTPUT / "choices_seal.json"), "external_provenance": provenance,
            "jobs": jobs, "job_count": len(jobs)}
    if plan_path.exists():
        if not resume or load_json(plan_path) != plan:
            raise RuntimeError("Existing execution plan differs; resume requires identical inputs")
    else:
        write_json(plan_path, plan)
    plan_sha = sha256(plan_path)
    directory = OUTPUT / "execution/jobs"
    directory.mkdir(parents=True, exist_ok=True)
    core_run = pipeline.build_core_train_score_callback(DATASET, manifest)
    external_run = external.build_external_train_score_callback(DATASET, manifest, ROOT / "tmp")
    total, skipped, started = len(jobs), 0, time.perf_counter()
    for index, job in enumerate(jobs, 1):
        path = directory / _job_name(job)
        if path.exists():
            if not resume:
                raise FileExistsError(f"Job exists without --resume: {path}")
            _verify_job(path, job, plan_sha)
            skipped += 1
            continue
        try:
            begin = time.perf_counter()
            outcome = (core_run if job["method"] in CORE else external_run)(job)
            runtime = time.perf_counter() - begin
            payload = {"job": job, "execution_plan_sha256": plan_sha,
                       "outcome": dict(outcome), "runtime_seconds": runtime,
                       "completed_at_utc": now()}
            write_json(path, payload)
            _verify_job(path, job, plan_sha)
        except Exception as exc:
            failure = {"job_index": index, "job": job, "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc(),
                       "completed_jobs": index - 1, "execution_plan_sha256": plan_sha,
                       "failed_at_utc": now()}
            write_json(OUTPUT / "failure_evaluate.json", failure)
            raise
        if index % 100 == 0 or index == total:
            progress = {"completed": index, "total": total, "resumed": skipped,
                        "elapsed_seconds_this_process": time.perf_counter() - started,
                        "updated_at_utc": now()}
            write_json(OUTPUT / "execution/progress.json", progress)
            print(f"Completed {index}/{total} method-seed jobs", flush=True)
    write_json(OUTPUT / "execution_complete.json", {
        "completed_at_utc": now(), "execution_plan_sha256": plan_sha,
        "job_count": total, "jobs_dir": str(directory.relative_to(ROOT)).replace("\\", "/"),
    })


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _simultaneous_bounds(matrix, draws=2000, seed=20260921):
    """One-sided max-error bounds for all contrasts on whole-MDP resamples."""
    point = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, matrix.shape[1]))
    for index in range(draws):
        chosen = rng.integers(0, len(matrix), len(matrix))
        boot[index] = matrix[chosen].mean(axis=0)
    max_error = np.max(boot - point, axis=1)
    critical = float(np.quantile(max_error, .95))
    return point, point - critical, critical


def analyze():
    frozen, protocol_sha = protocol()
    manifest, _ = verified_dataset()
    choices, choice_seal = verified_choices()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Cannot analyze with a recorded method failure")
    plan_path = OUTPUT / "execution_plan.json"
    plan = load_json(plan_path)
    plan_sha = sha256(plan_path)
    complete = load_json(OUTPUT / "execution_complete.json")
    if (complete["execution_plan_sha256"] != plan_sha or plan["protocol_sha256"] != protocol_sha
            or plan["choices_seal_sha256"] != sha256(OUTPUT / "choices_seal.json")):
        raise RuntimeError("Execution/choice/protocol seals do not agree")
    if len(plan["jobs"]) != 21000:
        raise RuntimeError("Incomplete frozen job plan")
    instances = {row["instance_id"]: row for row in manifest["instances"]}
    rows = []
    for job in plan["jobs"]:
        path = OUTPUT / "execution/jobs" / _job_name(job)
        payload = _verify_job(path, job, plan_sha)
        rows.append({"mdp_id": job["instance_id"], "policy_family": job["policy_family"],
                     "training_seed": job["training_seed"], "method": job["method"],
                     "panel": instances[job["instance_id"]]["dynamics_audit"]["panel"],
                     "normalized_return": float(payload["outcome"]["normalized_return"]),
                     "success": int(payload["outcome"]["success"]),
                     "buffer_sha256": job["buffer_sha256"],
                     "runtime_seconds": payload["runtime_seconds"], "job_file_sha256": sha256(path)})
    if len(rows) != 21000:
        raise RuntimeError("Incomplete outcome panel")
    selected = []
    for row in rows:
        if row["method"] == choices[(row["mdp_id"], row["policy_family"])]:
            selected.append({**row, "method": "OEH"})
    if len(selected) != 3000:
        raise RuntimeError("Incomplete matched OEH outcome panel")
    all_rows = rows + selected
    _csv(OUTPUT / "all_method_seed_outcomes.csv", all_rows, tuple(all_rows[0]))
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["mdp_id"], row["method"])].append(row)
    mdp_rows = []
    for (mdp, method), group in sorted(grouped.items()):
        if len(group) != 50 or len({(r["policy_family"], r["training_seed"]) for r in group}) != 50:
            raise RuntimeError("Incomplete matched MDP behavior/seed grid")
        mdp_rows.append({"mdp_id": mdp, "panel": group[0]["panel"], "method": method,
                         "mean_normalized_return": float(np.mean([r["normalized_return"] for r in group])),
                         "seed_success_rate": float(np.mean([r["success"] for r in group]))})
    _csv(OUTPUT / "mdp_metrics.csv", mdp_rows, tuple(mdp_rows[0]))
    methods = ("OEH", *frozen["candidate_methods"])
    lookup = {(row["mdp_id"], row["method"]): row for row in mdp_rows}
    summary, contrasts = [], []
    for panel_name in ("stochastic_primary", "deterministic_control"):
        ids = sorted(mdp for mdp, record in instances.items() if record["dynamics_audit"]["panel"] == panel_name)
        if len(ids) != (42 if panel_name == "stochastic_primary" else 18):
            raise RuntimeError("Unexpected panel size")
        for endpoint, field in (("mean_normalized_return", "mean_normalized_return"),
                                ("seed_success_rate", "seed_success_rate")):
            values = np.asarray([[lookup[(mdp, method)][field] for method in methods] for mdp in ids])
            for index, method in enumerate(methods):
                summary.append({"panel": panel_name, "endpoint": endpoint, "method": method,
                                "mean": float(values[:, index].mean()), "mdp_count": len(ids)})
            if panel_name == "stochastic_primary":
                for index, method in enumerate(methods[1:], 1):
                    contrasts.append({"panel": panel_name, "endpoint": endpoint, "baseline": method,
                                      "delta": float(np.mean(values[:, 0] - values[:, index])),
                                      "mdp_count": len(ids)})
    # One family of 14 contrasts, jointly calibrated across both endpoints.
    ids = sorted(mdp for mdp, record in instances.items()
                 if record["dynamics_audit"]["panel"] == "stochastic_primary")
    matrix = np.asarray([
        [lookup[(mdp, "OEH")][field] - lookup[(mdp, baseline)][field]
         for field in ("mean_normalized_return", "seed_success_rate")
         for baseline in methods[1:]]
        for mdp in ids
    ])
    point, lower, critical = _simultaneous_bounds(matrix)
    for endpoint_index, endpoint in enumerate(("mean_normalized_return", "seed_success_rate")):
        for baseline_index, baseline in enumerate(methods[1:]):
            index = endpoint_index * 7 + baseline_index
            item = next(row for row in contrasts if row["endpoint"] == endpoint and row["baseline"] == baseline)
            if not np.isclose(item["delta"], point[index], atol=1e-12):
                raise RuntimeError("Bootstrap contrast ordering error")
            item["simultaneous_one_sided_lower_95"] = float(lower[index])
    _csv(OUTPUT / "summary.csv", summary, tuple(summary[0]))
    _csv(OUTPUT / "contrasts.csv", contrasts, tuple(contrasts[0]))
    superiority = all(row["simultaneous_one_sided_lower_95"] > 0 for row in contrasts)
    oeh_primary = {row["endpoint"]: row["mean"] for row in summary
                   if row["panel"] == "stochastic_primary" and row["method"] == "OEH"}
    result = {
        "status": "complete", "completed_at_utc": now(), "protocol_sha256": protocol_sha,
        "dataset_manifest_sha256": sha256(DATASET / "manifest.json"),
        "choices_seal_sha256": sha256(OUTPUT / "choices_seal.json"),
        "execution_plan_sha256": plan_sha, "method_seed_jobs": len(rows),
        "oeh_selected_seed_outcomes": len(selected), "mdp_count": len(instances),
        "primary_mdps": len(ids), "simultaneous_bootstrap_critical": critical,
        "superiority_gate": "GO" if superiority else "NO-GO",
        "descriptive_targets_met": bool(oeh_primary["mean_normalized_return"] > .8 and
                                        oeh_primary["seed_success_rate"] > .5),
        "scope": "Independent new seeds/MDPs under the same generator families; E1-trained router vs seven fixed learners, not proof of literature-wide SOTA",
        "artifact_sha256": {name: sha256(OUTPUT / name) for name in
                            ("all_method_seed_outcomes.csv", "mdp_metrics.csv", "summary.csv", "contrasts.csv")},
    }
    write_json(OUTPUT / "result_manifest.json", result)
    print(json.dumps({"superiority_gate": result["superiority_gate"],
                      "oeh_primary": oeh_primary, "descriptive_targets_met": result["descriptive_targets_met"]},
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "analyze"))
    parser.add_argument("--resume", action="store_true", help="Only for verified interrupted execution")
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": lambda: evaluate(args.resume), "analyze": analyze}[args.phase]()


if __name__ == "__main__":
    main()
