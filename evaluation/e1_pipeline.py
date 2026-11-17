"""Frozen-method adapters for a sealed Week 6 evaluation bundle.

This module is called only after ``e1_access.open_evaluation_bundle``.  Pilot
features are durably sealed before any learner outcome is scored.  The learners
are the same pinned implementations used by the development capacity run.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from config.protocol_config import PROTOCOL_CONFIG
from evaluation.confirmatory import json_sha256
from evaluation.io import read_csv_rows, sha256_file, verified_dataset_path, write_csv_rows, write_json
from scripts.training.run_empirical_requirement import _load_run, _score_policy, _training_seed_index


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from empirical_requirement import behavior_cloning_tabular, cfqi_tabular, iql_tabular  # noqa: E402
from environments.benchmark_environments import make_benchmark_mdp  # noqa: E402
from lcoeh_v4 import LC_OEH_V4_FEATURES, estimate_lcoeh_v4  # noqa: E402
from oeh_risk_v3 import estimate_oeh_risk_v3  # noqa: E402
from stochastic_features import aggregate_oeh_risk_v3_profile, extract_stochastic_features  # noqa: E402


T1_CONTRACT = ROOT / "docs/protocol/WEEK6_T1_STATISTICAL_FREEZE_v1.1.0.json"
CORE_METHODS = ("cfqi_tabular", "iql_tabular", "behavior_cloning_tabular")
EXTERNAL_METHODS = (
    "pib_spibb_n10_author", "vi_lcb_cb005_author_empirical",
    "discrete_bcq_author", "discrete_cql_d3rlpy",
)


def _feature_names() -> dict[str, tuple[str, ...]]:
    contract = json.loads(T1_CONTRACT.read_text(encoding="utf-8"))
    return {key: tuple(values) for key, values in contract["feature_sets"].items()}


def _pilot_groups(manifest: Mapping[str, Any], dataset_dir: Path):
    index_path = dataset_dir / str(manifest["estimator_input_index"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    allowed_index_fields = {
        "dataset_id", "group_id", "stream", "max_size", "horizon",
        "num_states", "num_actions", "initial_state_distribution", "size",
        "size_multiplier", "budget_role", "is_reference_n0", "slice",
        "file", "sha256", "content_sha256",
    }
    identity_fields = (
        "instance_id", "policy_family", "file", "sha256", "size", "slice",
        "num_states", "num_actions", "initial_state_distribution",
    )
    labels = {
        str(item["data_only_dataset_id"]): {field: item[field] for field in identity_fields}
        for item in manifest["datasets"] if item.get("benchmark_split") == "evaluation"
    }
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in index.get("datasets", ()):
        if set(row) - allowed_index_fields:
            raise ValueError("Data-only pilot index contains oracle or unregistered fields")
        if row.get("stream") != "pilot" or int(row.get("size_multiplier", -1)) not in (2, 5, 10):
            continue
        label = labels.get(str(row.get("dataset_id")))
        if label is None:
            continue
        for field in ("file", "sha256", "size", "slice", "num_states", "num_actions", "initial_state_distribution"):
            if row.get(field) != label.get(field):
                raise ValueError(f"Data-only index/manifest mismatch: {field}")
        key = (str(label["instance_id"]), str(label["policy_family"]))
        size = int(row["size_multiplier"])
        if size in grouped[key]:
            raise ValueError(f"Duplicate pilot prefix: {key}/{size}")
        grouped[key][size] = dict(row)
    if len(grouped) != 600 or any(set(group) != {2, 5, 10} for group in grouped.values()):
        raise ValueError("E1 pilot feature grid is incomplete")
    return grouped


def _group_features(dataset_dir: Path, key: tuple[str, str], prefixes: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for multiplier, record in sorted(prefixes.items()):
        with np.load(verified_dataset_path(dataset_dir, record), allow_pickle=False) as loaded:
            arrays[multiplier] = {name: loaded[name][:int(record["size"])] for name in loaded.files}
    record = prefixes[10]
    base = extract_stochastic_features(
        arrays[10], num_states=int(record["num_states"]), num_actions=int(record["num_actions"]),
        gamma=.95, bootstrap_replicates=200, seed_key=str(record["dataset_id"]),
    )
    risk = estimate_oeh_risk_v3(
        arrays[10], record["initial_state_distribution"],
        num_states=int(record["num_states"]), num_actions=int(record["num_actions"]),
        gamma=.95, k_max=100,
    )
    by_depth = {int(item["depth"]): item for item in risk["records"]}
    profile = [
        {**item, "risk_at_pilot": by_depth[int(item["depth"])]["risk_at_pilot"],
         "policy_instability": by_depth[int(item["depth"])]["policy_instability"]}
        for item in risk["profile"]
    ]
    oeh = aggregate_oeh_risk_v3_profile(profile)
    learner_features = estimate_lcoeh_v4(
        arrays, record["initial_state_distribution"],
        num_states=int(record["num_states"]), num_actions=int(record["num_actions"]),
        gamma=.95, seed_key=f"{key[0]}|{key[1]}",
    )
    names = _feature_names()
    rows = []
    for algorithm in CORE_METHODS:
        values = {**base, **oeh, **{name: learner_features[algorithm][name] for name in LC_OEH_V4_FEATURES}}
        if any(name not in values for fields in names.values() for name in fields):
            raise ValueError(f"Frozen E1 feature is missing for {key}/{algorithm}")
        if not all(math.isfinite(float(values[name])) for fields in names.values() for name in fields):
            raise ValueError(f"Non-finite E1 feature for {key}/{algorithm}")
        rows.append({"mdp_id": key[0], "policy_family": key[1], "algorithm": algorithm,
                     **{name: values[name] for name in names["B2_OEH3_LCOEH4"]}})
    return rows


def seal_pilot_features(dataset_dir: Path, manifest: Mapping[str, Any], output_dir: Path,
                        *, resume: bool, source_sha256: Mapping[str, str]) -> Path:
    """Compute or verify all 1,800 pilot-only learner rows with per-group resume."""

    grouped = _pilot_groups(manifest, dataset_dir)
    feature_dir = output_dir / "pilot_features"
    group_dir = feature_dir / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for key, prefixes in sorted(grouped.items()):
        label = "|".join(key)
        path = group_dir / f"{hashlib.sha256(label.encode('utf-8')).hexdigest()}.json"
        identity = json_sha256({"group": key, "records": prefixes, "source_sha256": source_sha256})
        if path.exists():
            if not resume:
                raise FileExistsError(f"Existing pilot feature checkpoint without resume: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            body = {field: value for field, value in payload.items() if field != "payload_sha256"}
            if (payload.get("identity_sha256") != identity
                    or payload.get("payload_sha256") != json_sha256(body)
                    or len(payload.get("rows", ())) != 3):
                raise ValueError(f"Pilot feature checkpoint hash/identity mismatch: {path}")
            rows = payload["rows"]
        else:
            rows = _group_features(dataset_dir, key, prefixes)
            body = {"group": list(key), "identity_sha256": identity, "rows": rows}
            write_json(path, {**body, "payload_sha256": json_sha256(body)})
        all_rows.extend(rows)
    if len(all_rows) != 1_800:
        raise ValueError("E1 pilot feature row count differs from 600 groups x 3 learners")
    names = _feature_names()["B2_OEH3_LCOEH4"]
    path = feature_dir / "features.csv"
    seal_path = feature_dir / "feature_seal.json"
    seal_payload = {
        "scope": "evaluation_pilot_only_before_training_outcomes",
        "rows": len(all_rows), "groups": len(grouped),
        "source_sha256": dict(source_sha256),
    }
    if path.exists() and resume:
        if seal_path.exists():
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            if seal != {**seal_payload, "features_sha256": sha256_file(path)}:
                raise ValueError("Sealed E1 pilot feature CSV differs on resume")
        else:
            existing = read_csv_rows(path)
            if (len(existing) != len(all_rows)
                    or any(existing_row != {name: str(value) for name, value in row.items()}
                           for existing_row, row in zip(existing, all_rows))):
                raise ValueError("Interrupted E1 pilot feature CSV differs from verified group checkpoints")
            write_json(seal_path, {**seal_payload, "features_sha256": sha256_file(path)})
    elif path.exists():
        raise FileExistsError("Existing E1 pilot feature CSV without verified resume")
    else:
        write_csv_rows(path, all_rows, ("mdp_id", "policy_family", "algorithm", *names))
        write_json(seal_path, {**seal_payload, "features_sha256": sha256_file(path)})
    return path


def build_core_train_score_callback(dataset_dir: Path, manifest: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Fit registered tabular learners on evaluation training prefixes."""

    indexed = {
        (str(row["instance_id"]), str(row["policy_family"]), int(row["size_multiplier"]), _training_seed_index(row)): row
        for row in manifest["datasets"] if row.get("benchmark_split") == "evaluation" and row.get("stream") == "training"
    }
    instances = {str(item["instance_id"]): item for item in manifest["instances"] if item.get("split") == "evaluation"}
    learning = PROTOCOL_CONFIG["learning"]
    mdp_cache: dict[str, Any] = {}
    array_cache: dict[str, dict[str, np.ndarray]] = {}

    def run(job: Mapping[str, Any]) -> Mapping[str, Any]:
        method = str(job["method"])
        if method not in CORE_METHODS:
            raise ValueError(f"Unregistered E1 core method: {method}")
        key = (str(job["instance_id"]), str(job["policy_family"]), int(job["size_multiplier"]), int(job["training_seed"]))
        record = indexed[key]
        if str(record["sha256"]).lower() != str(job["buffer_sha256"]).lower() or str(record["file"]) != job["buffer_file"]:
            raise ValueError(f"E1 job/buffer identity mismatch: {key}")
        # Bound resident memory to one MDP and one physical trajectory buffer.
        if key[0] not in mdp_cache:
            mdp_cache.clear()
            mdp_cache[key[0]] = make_benchmark_mdp(type("Spec", (), instances[key[0]])())
        mdp = mdp_cache[key[0]]
        if str(record["file"]) not in array_cache:
            array_cache.clear()
        transitions = _load_run(dataset_dir, dict(record), array_cache)
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
        score, status = _score_policy(mdp, np.asarray(policy, dtype=float),
                                      random_return=float(instances[key[0]]["random_return"]),
                                      dynamic_range=float(instances[key[0]]["dynamic_range"]))
        if status != "observed":
            raise ValueError(f"E1 core learner produced invalid score: {key}/{method}")
        return {"status": status, "method": method, "normalized_return": score,
                "success": bool(score >= .95), "buffer_sha256": job["buffer_sha256"]}

    return run


def feature_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    if len(rows) != 1_800:
        raise ValueError("Incomplete sealed E1 feature rows")
    return rows


__all__ = ["CORE_METHODS", "EXTERNAL_METHODS", "build_core_train_score_callback",
           "feature_rows", "seal_pilot_features"]
