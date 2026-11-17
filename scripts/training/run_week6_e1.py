#!/usr/bin/env python
"""Execute the frozen E1 grid after a separate implementation review.

The default ``--check`` reads no evaluation input.  ``--run`` requires a
reviewed implementation hash set, an independently supplied manifest seal and
the explicit authorization reference.  Development runners remain sealed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from evaluation.aggregation.folds import assign_folds  # noqa: E402
from evaluation.confirmatory import (  # noqa: E402
    CONFIRMATORY_OUTPUT_DIR_NAME, CONFIRMATORY_RUN_ID,
    build_job_plan, json_sha256, run_missing_jobs_only,
)
from evaluation.e1_access import open_evaluation_bundle  # noqa: E402
from evaluation.e1_external import build_external_train_score_callback, verify_external_components  # noqa: E402
from evaluation.e1_pipeline import (  # noqa: E402
    CORE_METHODS, EXTERNAL_METHODS, build_core_train_score_callback,
    feature_rows, seal_pilot_features,
)
from evaluation.io import sha256_file, write_csv_rows, write_json  # noqa: E402
from return_selection import ReturnInstance, _fit_ridge, tune_return_ridge  # noqa: E402
from scripts.training.run_empirical_requirement import _training_seed_index  # noqa: E402
from scripts.validation.verify_week6_h1_freeze import verify as verify_h1  # noqa: E402
from scripts.validation.verify_week6_t1_freeze import verify as verify_t1  # noqa: E402


T1_MANIFEST = "docs/protocol/WEEK6_T1_FREEZE_v1.1.0-r1.sha256.json"
B2_DIR = ROOT / "artifacts/week6/viettq_b2_final_review_2026-09-25"
B2_REPORT = "b2_final_independent_preflight_review_v3.json"
IMPLEMENTATION_FILES = (
    "evaluation/e1_access.py", "evaluation/e1_pipeline.py", "evaluation/e1_external.py",
    "scripts/data/generate_week6_e1_bundle.py",
    "scripts/training/run_week6_e1.py", "tests/test_e1_access.py",
    "tests/test_e1_pipeline_adapter.py", "scripts/testing/probe_e1_external_components.py",
    "scripts/testing/probe_e1_external_adapter.py", "tests/test_e1_selector_partition.py",
    "tests/test_e1_generator_gate.py", "tests/test_e1_failure_partition.py",
    "tests/test_e1_external_provenance.py",
)
REVIEW_PATH = ROOT / "docs/week6/E1_EXECUTION_IMPLEMENTATION_REVIEW.json"
METHOD_ORDER = CORE_METHODS
E1_LOCK_PATH = ROOT / "artifacts/week6" / f".{CONFIRMATORY_RUN_ID}.execution.lock"


@contextmanager
def _exclusive_e1_claim(lock_path: Path = E1_LOCK_PATH):
    """Hold a process-exclusive claim for the one-shot run ID.

    The lock file deliberately remains after release. Removing a lock file can
    create two independently lockable inodes while another process is waiting.
    Its contents are diagnostic only; the operating-system byte lock is the
    authority.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"E1 run ID is already claimed: {CONFIRMATORY_RUN_ID}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"E1 run ID is already claimed: {CONFIRMATORY_RUN_ID}") from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write((json.dumps({
            "run_id": CONFIRMATORY_RUN_ID,
            "pid": os.getpid(),
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in (T1_MANIFEST, *IMPLEMENTATION_FILES)}


def source_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("E1 requires a full Git source commit")
    return commit


def _committed_clean(paths: tuple[str, ...]) -> bool:
    try:
        for path in paths:
            subprocess.run(["git", "ls-files", "--error-unmatch", "--", path], cwd=ROOT,
                           check=True, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        return not status.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _verify_resume_policy(output: Path, *, source_commit_value: str) -> None:
    """Permit one verified infrastructure retry, never a numerical rerun."""

    if not output.is_dir() or not (output / "first_access_log.json").is_file():
        raise FileNotFoundError("E1 resume requires an existing first-access log")
    access = json.loads((output / "first_access_log.json").read_text(encoding="utf-8"))
    if (access.get("status") == "rejected"
            and access.get("error_type") not in {"OSError", "TimeoutError",
                                                 "ConnectionError", "InterruptedError"}):
        raise ValueError("E1 first-access seal, schema, or algorithm rejection cannot be resumed")
    marker = output / "resume_attempt.json"
    if marker.exists():
        raise ValueError("E1 permits at most one infrastructure resume attempt")
    failures = sorted(output.glob("e1_failure_*.json"))
    if failures:
        latest = json.loads(failures[-1].read_text(encoding="utf-8"))
        if latest.get("error_type") not in {"OSError", "TimeoutError", "ConnectionError",
                                             "InterruptedError"}:
            raise ValueError("E1 numerical, schema, or algorithm failure cannot be resumed")
    write_json(marker, {
        "source_commit": source_commit_value,
        "resume_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_failure_count": len(failures),
    })


def check_release_gate(*, review_path: Path = REVIEW_PATH) -> dict[str, Any]:
    """Check the tracked freezes and B2 evidence without touching evaluation."""

    t1 = verify_t1(ROOT, ROOT / T1_MANIFEST)
    h1 = verify_h1(ROOT)
    b2 = json.loads((B2_DIR / B2_REPORT).read_text(encoding="utf-8"))
    checksum = json.loads((B2_DIR / "checksum_manifest.json").read_text(encoding="utf-8"))
    if (b2.get("status") != "GO" or checksum.get("status") != "GO"
            or checksum.get("files", {}).get(B2_REPORT) != sha256_file(B2_DIR / B2_REPORT)):
        raise ValueError("B2 GO report or detached checksum is invalid")
    if t1["status"] != "PASS" or h1["status"] != "PASS":
        raise ValueError("T1/H1 freeze verification failed")
    current_sources = source_hashes()
    implementation_review = "missing"
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        reviewer = review.get("reviewer")
        if (review.get("status") == "GO"
                and review.get("run_id") == CONFIRMATORY_RUN_ID
                and review.get("source_sha256") == current_sources
                and isinstance(reviewer, str) and reviewer.strip()
                and reviewer.strip().casefold() != "codex"):
            implementation_review = "GO"
        else:
            implementation_review = "mismatch_or_no_go"
    try:
        review_relative = review_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        review_relative = ""
    source_committed = bool(review_relative) and _committed_clean((*IMPLEMENTATION_FILES, review_relative))
    return {
        "run_id": CONFIRMATORY_RUN_ID,
        "t1": t1["status"], "h1": h1["status"], "b2": b2["status"],
        "implementation_review": implementation_review,
        "source_and_review_committed_clean": source_committed,
        "generator_protocol_status": PROTOCOL_CONFIG["protocol_status"],
        "source_commit": source_commit(),
        "source_sha256": current_sources,
        "evaluation_opened": False,
    }


def _job_rows(stage_dir: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    ledger = json.loads((stage_dir / "checkpoint_ledger.json").read_text(encoding="utf-8"))
    if set(ledger["jobs"]) != set(plan["job_keys"]):
        raise ValueError("E1 job ledger is incomplete")
    rows = []
    for job in plan["jobs"]:
        entry = ledger["jobs"][job["job_key"]]
        path = stage_dir / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"E1 job output hash mismatch: {job['job_key']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        rows.append({
            "mdp_id": job["instance_id"], "policy_family": job["policy_family"],
            "algorithm": job["method"], "training_size_multiplier": job["size_multiplier"],
            "training_seed": job["training_seed"],
            "normalized_return": result["normalized_return"], "success": result["success"],
            "run_status": result["status"], "buffer_sha256": job["buffer_sha256"],
            "runtime_seconds": payload["runtime_seconds"],
            "peak_tracemalloc_bytes": payload["peak_tracemalloc_bytes"],
            "job_file_sha256": entry["sha256"],
        })
    return rows


def _write_or_verify_csv(path: Path, rows: list[dict[str, Any]], *, resume: bool) -> None:
    if not rows:
        raise ValueError(f"Refusing empty E1 CSV: {path.name}")
    fields = tuple(rows[0])
    if path.exists():
        if not resume:
            raise FileExistsError(f"Existing E1 CSV without resume: {path}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        expected = hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()
        if sha256_file(path) != expected:
            raise ValueError(f"E1 completed CSV hash/content mismatch on resume: {path}")
    else:
        write_csv_rows(path, rows, fields)


def _descriptive_comparison(core_rows: list[dict[str, Any]],
                            external_rows: list[dict[str, Any]],
                            external_provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Report x10 paper-method outcomes with base MDP as the aggregation unit."""

    from environments.benchmark_environments import benchmark_specs

    registered = {spec.instance_id: spec.family for spec in benchmark_specs() if spec.split == "evaluation"}
    primary_families = {"uniform_garnet", "forward_layered", "bottleneck"}
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in (*core_rows, *external_rows):
        if int(row["training_size_multiplier"]) != 10:
            continue
        panel = "stochastic_primary" if registered[row["mdp_id"]] in primary_families else "deterministic_control"
        cells[(panel, row["algorithm"], row["mdp_id"], row["policy_family"])].append(row)
    by_method: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (panel, method, mdp_id, _family), records in cells.items():
        if len(records) != 10 or {int(row["training_seed"]) for row in records} != set(range(10)):
            raise ValueError(f"Incomplete E1 x10 method/seed cell: {panel}/{method}/{mdp_id}")
        by_method[(panel, method)][mdp_id].append((
            sum(float(row["normalized_return"]) for row in records) / 10.0,
            sum(bool(row["success"]) for row in records) / 10.0,
        ))
    result = []
    for (panel, method), by_mdp in sorted(by_method.items()):
        expected_mdps = 84 if panel == "stochastic_primary" else 36
        if len(by_mdp) != expected_mdps or any(len(values) != 5 for values in by_mdp.values()):
            raise ValueError(f"Incomplete E1 behavior/MDP panel: {panel}/{method}")
        mdp_return = [sum(pair[0] for pair in values) / 5.0 for values in by_mdp.values()]
        mdp_success = [sum(pair[1] for pair in values) / 5.0 for values in by_mdp.values()]
        result.append({
            "panel": panel, "algorithm": method,
            "source_type": "registered_project_implementation" if method in CORE_METHODS
            else external_provenance["source_type"][method],
            "base_mdps": len(by_mdp), "behavior_groups": len(by_mdp) * 5,
            "runs": len(by_mdp) * 5 * 10,
            "mean_normalized_return": sum(mdp_return) / len(mdp_return),
            "mean_success_rate": sum(mdp_success) / len(mdp_success),
            "in_confirmatory_family": method in CORE_METHODS,
        })
    if len(result) != 2 * (len(CORE_METHODS) + len(EXTERNAL_METHODS)):
        raise ValueError("E1 descriptive method panel is incomplete")
    return result


def _choose_e1_method(component_scores: Mapping[str, list[float]],
                      fold_best_fixed: str) -> tuple[str, bool, str, dict[str, float]]:
    """Apply the frozen equal-weight rule or abstain for the entire group."""

    def finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    scores = {}
    unavailable = False
    for method in METHOD_ORDER:
        components = component_scores.get(method, [])
        if len(components) != 2 or not all(finite(value) for value in components):
            scores[method] = math.nan
            unavailable = True
        else:
            scores[method] = (float(components[0]) + float(components[1])) / 2.0
    if unavailable:
        return fold_best_fixed, True, "component_score_unavailable", scores
    chosen = max(METHOD_ORDER, key=lambda method: (scores[method], -METHOD_ORDER.index(method)))
    return chosen, False, "", scores


def _nested_selector(features: list[dict[str, Any]], runs: list[dict[str, Any]],
                     *, primary: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = json.loads((ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json").read_text(encoding="utf-8"))
    names = contract["feature_sets"]
    from environments.benchmark_environments import benchmark_specs
    registered = {spec.instance_id: spec.family for spec in benchmark_specs() if spec.split == "evaluation"}
    primary_families = {"uniform_garnet", "forward_layered", "bottleneck"}
    belongs = lambda mdp_id: (registered[mdp_id] in primary_families) == primary
    source = [row for row in runs if int(row["training_size_multiplier"]) == 10
              and belongs(row["mdp_id"])]
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in source:
        by_key[(row["mdp_id"], row["policy_family"], row["algorithm"])].append(float(row["normalized_return"]))
    expected_clusters = 84 if primary else 36
    if len(by_key) != expected_clusters * 5 * len(CORE_METHODS) or any(len(values) != 10 for values in by_key.values()):
        raise ValueError("E1 x10 core outcome grid is incomplete")
    mean = {key: sum(values) / 10.0 for key, values in by_key.items()}
    feature_by_key = {(row["mdp_id"], row["policy_family"], row["algorithm"]): row
                      for row in features if belongs(row["mdp_id"])}
    if set(mean) != set(feature_by_key):
        raise ValueError("E1 feature/outcome identity mismatch")
    instances: list[ReturnInstance] = []
    for key, outcome in mean.items():
        row = feature_by_key[key]
        instances.append(ReturnInstance(
            mdp_id=key[0], policy_family=key[1], algorithm=key[2], normalized_return=outcome,
            features={name: tuple(float(row[field]) for field in names[name])
                      for name in ("B2_OEH3", "B2_OEH3_LCOEH4")},
        ))
    # The registered MDP family is supplied by the caller through the recipe
    # identity in the feature rows; no outcome-derived stochasticity split.
    panel = [row for row in instances if (registered[row.mdp_id] in primary_families) == primary]
    if len({row.mdp_id for row in panel}) != expected_clusters:
        raise ValueError("E1 stochastic/control partition quota mismatch")
    outer = assign_folds({row.mdp_id for row in panel}, num_folds=5)
    ridges = tuple(float(value) for value in contract["evaluation"]["return_ridges"])
    predictions: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    for algorithm in METHOD_ORDER:
        learner_panel = [row for row in panel if row.algorithm == algorithm]
        for feature_set in ("B2_OEH3", "B2_OEH3_LCOEH4"):
            for fold in range(5):
                train = [row for row in learner_panel if outer[row.mdp_id] != fold]
                test = [row for row in learner_panel if outer[row.mdp_id] == fold]
                if not train or not test:
                    raise ValueError("E1 outer fold is empty")
                chosen, scores = tune_return_ridge(
                    train, feature_set=feature_set, ridge_grid=ridges, num_folds=4,
                    salt=f"return|{fold}|{algorithm}|{feature_set}",
                )
                for ridge, mse in scores.items():
                    inner_scores.append({"algorithm": algorithm, "feature_set": feature_set,
                                         "outer_fold": fold, "ridge": ridge, "inner_mse": mse,
                                         "selected": ridge == chosen})
                for row, value in zip(test, _fit_ridge(train, test, feature_set=feature_set, ridge=chosen)):
                    predictions.append({"mdp_id": row.mdp_id, "policy_family": row.policy_family,
                                        "algorithm": algorithm, "feature_set": feature_set,
                                        "outer_fold": fold, "selected_ridge": chosen,
                                        "predicted_return": float(value), "normalized_return": row.normalized_return})
    if len(predictions) != len(panel) * 2:
        raise ValueError("E1 return OOF predictions are incomplete")
    pred_by_key = {(row["mdp_id"], row["policy_family"], row["algorithm"], row["feature_set"]): row
                   for row in predictions}
    decisions = []
    groups = sorted({(row.mdp_id, row.policy_family) for row in panel})
    for mdp_id, policy_family in groups:
        fold = outer[mdp_id]
        train_mdp_ids = sorted({row.mdp_id for row in panel if outer[row.mdp_id] != fold})
        fixed_scores = {}
        for algorithm in METHOD_ORDER:
            per_mdp = []
            for train_id in train_mdp_ids:
                values = [row.normalized_return for row in panel
                          if row.mdp_id == train_id and row.algorithm == algorithm]
                if len(values) != 5:
                    raise ValueError("E1 best-fixed training behavior group incomplete")
                per_mdp.append(sum(values) / 5.0)
            fixed_scores[algorithm] = sum(per_mdp) / len(per_mdp)
        fixed = max(METHOD_ORDER, key=lambda method: (fixed_scores[method], -METHOD_ORDER.index(method)))
        component_scores = {}
        for algorithm in METHOD_ORDER:
            component_scores[algorithm] = [
                pred_by_key[(mdp_id, policy_family, algorithm, feature)]["predicted_return"]
                for feature in ("B2_OEH3", "B2_OEH3_LCOEH4")
            ]
        selected, fallback, fallback_reason, scores = _choose_e1_method(
            component_scores, fixed)
        decisions.append({
            "mdp_id": mdp_id, "policy_family": policy_family, "outer_fold": fold,
            "selected_algorithm": selected, "fold_best_fixed_algorithm": fixed,
            "selected_return": mean[(mdp_id, policy_family, selected)],
            "fold_best_fixed_return": mean[(mdp_id, policy_family, fixed)],
            "fallback": fallback, "fallback_reason": fallback_reason,
            **{f"score_{method}": scores[method] for method in METHOD_ORDER},
        })
    return predictions, inner_scores, decisions


def _run_e1_claimed(dataset_dir: Path, expected_manifest_sha256: str, authorization_reference: str,
                    *, external_root: Path, resume: bool) -> dict[str, Any]:
    gate = check_release_gate()
    if gate["implementation_review"] != "GO" or not gate["source_and_review_committed_clean"]:
        raise ValueError("E1 implementation requires committed source and independent review of its hashes")
    external = verify_external_components(external_root)
    output = ROOT / "artifacts/week6" / CONFIRMATORY_OUTPUT_DIR_NAME
    if resume:
        _verify_resume_policy(output, source_commit_value=gate["source_commit"])
    manifest, counts = open_evaluation_bundle(
        dataset_dir, output, expected_manifest_sha256=expected_manifest_sha256,
        authorization_reference=authorization_reference,
        source_commit=gate["source_commit"], source_sha256=gate["source_sha256"],
        resume=resume,
    )
    phase = "pilot_features"
    try:
        sources = {**gate["source_sha256"],
                   **{f"external:{name}": digest for name, digest in external["source_sha256"].items()},
                   "external_runtime": json_sha256(external["runtime"])}
        features_path = seal_pilot_features(dataset_dir, manifest, output, resume=resume, source_sha256=sources)
        feature_digest = sha256_file(features_path)
        records = [row for row in manifest["datasets"] if row.get("benchmark_split") == "evaluation"]
        input_hashes = {"manifest.json": expected_manifest_sha256.lower(), "pilot_features.csv": feature_digest,
                        "data_only_index": sha256_file(dataset_dir / manifest["estimator_input_index"])}
        common = {"training_seeds": PROTOCOL_CONFIG["collection"]["training_seed_indices"],
                  "seed_index_fn": _training_seed_index}
        from environments.benchmark_environments import benchmark_specs
        registered = {spec.instance_id: spec.family for spec in benchmark_specs() if spec.split == "evaluation"}
        primary_families = {"uniform_garnet", "forward_layered", "bottleneck"}
        primary_records = [row for row in records if registered[row["instance_id"]] in primary_families]
        control_records = [row for row in records if registered[row["instance_id"]] not in primary_families]
        size_grid = PROTOCOL_CONFIG["collection"]["size_multipliers"]
        primary_plan = build_job_plan(primary_records, methods=CORE_METHODS,
                                      size_multipliers=size_grid, **common)
        control_plan = build_job_plan(control_records, methods=CORE_METHODS,
                                      size_multipliers=size_grid, **common)
        if (primary_plan["counts"]["jobs"] != 75_600
                or control_plan["counts"]["jobs"] != 32_400):
            raise ValueError("E1 primary/control core fit quotas differ from T1")
        callback = build_core_train_score_callback(dataset_dir, manifest)
        phase = "primary_core"
        primary_dir = output / "core" / "stochastic_primary" / CONFIRMATORY_OUTPUT_DIR_NAME
        run_missing_jobs_only(primary_dir, primary_plan, callback,
                              source_hashes=sources, input_hashes=input_hashes, resume=resume,
                              checkpoint_interval=5_000)
        primary_rows = _job_rows(primary_dir, primary_plan)
        _write_or_verify_csv(output / "stochastic_primary_core_run_results.csv", primary_rows,
                             resume=resume)
        features = feature_rows(features_path)
        phase = "primary_selector"
        predictions, inner, decisions = _nested_selector(features, primary_rows, primary=True)
        for name, rows in (("return_oof_predictions", predictions), ("return_inner_scores", inner),
                           ("selector_decisions", decisions)):
            _write_or_verify_csv(output / f"stochastic_primary_{name}.csv", rows, resume=resume)
        phase = "control_core"
        control_dir = output / "core" / "deterministic_control" / CONFIRMATORY_OUTPUT_DIR_NAME
        run_missing_jobs_only(control_dir, control_plan, callback,
                              source_hashes=sources, input_hashes=input_hashes, resume=resume,
                              checkpoint_interval=5_000)
        control_rows = _job_rows(control_dir, control_plan)
        _write_or_verify_csv(output / "deterministic_control_core_run_results.csv", control_rows,
                             resume=resume)
        core_rows = primary_rows + control_rows
        _write_or_verify_csv(output / "core_run_results.csv", core_rows, resume=resume)
        phase = "control_selector"
        predictions, inner, decisions = _nested_selector(features, control_rows, primary=False)
        for name, rows in (("return_oof_predictions", predictions), ("return_inner_scores", inner),
                           ("selector_decisions", decisions)):
            _write_or_verify_csv(output / f"deterministic_control_{name}.csv", rows,
                                 resume=resume)
        phase = "external_core"
        external_records = [row for row in records if row.get("stream") == "training"
                            and int(row["size_multiplier"]) == 10]
        external_plan = build_job_plan(external_records, methods=EXTERNAL_METHODS,
                                       size_multipliers=(10,), **common)
        if external_plan["counts"]["jobs"] != 24_000:
            raise ValueError("E1 external fit quota mismatch")
        external_dir = output / "external" / CONFIRMATORY_OUTPUT_DIR_NAME
        run_missing_jobs_only(external_dir, external_plan,
                              build_external_train_score_callback(dataset_dir, manifest, external_root),
                              source_hashes=sources, input_hashes=input_hashes, resume=resume,
                              checkpoint_interval=2_000)
        external_rows = _job_rows(external_dir, external_plan)
        external_csv = output / "external_run_results.csv"
        _write_or_verify_csv(external_csv, external_rows, resume=resume)
        phase = "comparison"
        comparison = _descriptive_comparison(core_rows, external_rows, external)
        _write_or_verify_csv(output / "x10_method_comparison.csv", comparison, resume=resume)
        summary = {
            "status": "complete", "run_id": CONFIRMATORY_RUN_ID,
            "source_commit": gate["source_commit"],
            "manifest_sha256": expected_manifest_sha256.lower(), "counts": counts,
            "core_fits": len(core_rows), "external_fits": len(external_rows),
            "external_provenance": external,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in output.glob("*.csv")
            },
            "raw_outputs_preserved": True,
            "statistical_analysis": "S1 separate; no interim endpoint used for retuning",
        }
        summary_path = output / "e1_execution_summary.json"
        if summary_path.exists():
            if not resume or json.loads(summary_path.read_text(encoding="utf-8")) != summary:
                raise ValueError("E1 completed summary differs on resume")
        else:
            write_json(summary_path, summary)
        write_json(output / "e1_status.json", {"status": "complete", "run_id": CONFIRMATORY_RUN_ID,
                                               "source_commit": gate["source_commit"]})
        return summary
    except Exception as exc:
        failure_status = {"primary_core": "incomplete_primary",
                          "control_core": "incomplete_controls",
                          "external_core": "incomplete_external"}.get(phase, "failed_stop_on_error")
        number = 1
        while (output / f"e1_failure_{number:03d}.json").exists():
            number += 1
        write_json(output / f"e1_failure_{number:03d}.json", {
            "status": failure_status, "phase": phase, "run_id": CONFIRMATORY_RUN_ID,
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        })
        write_json(output / "e1_status.json", {"status": failure_status, "phase": phase,
                                               "run_id": CONFIRMATORY_RUN_ID,
                                               "source_commit": gate["source_commit"]})
        raise


def run_e1(dataset_dir: Path, expected_manifest_sha256: str, authorization_reference: str,
           *, external_root: Path, resume: bool) -> dict[str, Any]:
    """Execute E1 while exclusively owning the frozen run ID."""

    with _exclusive_e1_claim():
        return _run_e1_claimed(
            dataset_dir, expected_manifest_sha256, authorization_reference,
            external_root=external_root, resume=resume,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read only freezes and local readiness; never open evaluation")
    parser.add_argument("--run", action="store_true", help="Open sealed evaluation and execute the full frozen E1 grid")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--authorization-reference")
    parser.add_argument("--external-root", type=Path, default=ROOT / "tmp")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.check == args.run:
        parser.error("Choose exactly one of --check or --run")
    if args.check:
        report = check_release_gate()
        report["dataset_dir_exists"] = bool(args.dataset_dir and args.dataset_dir.is_dir())
        report["e1_ready"] = (report["implementation_review"] == "GO"
                              and report["source_and_review_committed_clean"]
                              and report["dataset_dir_exists"])
        print(json.dumps(report, indent=2))
        return 0 if report["e1_ready"] else 2
    if args.dataset_dir is None or not args.expected_manifest_sha256 or not args.authorization_reference:
        parser.error("--run requires --dataset-dir, --expected-manifest-sha256 and --authorization-reference")
    result = run_e1(args.dataset_dir, args.expected_manifest_sha256, args.authorization_reference,
                    external_root=args.external_root, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
