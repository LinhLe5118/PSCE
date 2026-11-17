"""Preflight contract for the full development reproduction (no fitting)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config.protocol_config import PROTOCOL_CONFIG
from environments.benchmark_environments import benchmark_specs
from evaluation.io import verified_dataset_path


def validate_training_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    """Reject incomplete grids and scientific overrides before any learner runs."""
    if manifest.get("run_mode") != "review":
        raise ValueError("Training requires a development review bundle.")
    # The dataset manifest freezes only data-generation sections.  Learner
    # settings are frozen by the training plan/artifact and were intentionally
    # never embedded in this manifest schema; requiring a missing `learning`
    # section made every valid review bundle fail preflight.
    for section in ("benchmark", "behavior_policies", "collection", "coverage", "estimator"):
        for source in ("protocol_config", "effective_protocol"):
            if manifest.get(source, {}).get(section) != PROTOCOL_CONFIG[section]:
                raise ValueError(f"{source}.{section} differs from the training protocol.")
    instances = manifest["instances"]
    registered = benchmark_specs()
    registered_development = {
        s.instance_id for s in registered if s.split == "development"
    }
    registered_evaluation = {
        s.instance_id for s in registered if s.split == "evaluation"
    }
    by_id = {item["instance_id"]: item for item in instances}
    cohort = manifest.get("review_cohort", {})
    fresh_holdout = cohort.get("role") == "oeh_v3_post_freeze_holdout"
    if fresh_holdout:
        expected_ids = set(by_id)
        if (len(by_id) != len(instances) or len(expected_ids) < 15
                or expected_ids & registered_evaluation
                or any(item.get("split") != "development" for item in instances)
                or not cohort.get("formula_commit")):
            raise ValueError("Invalid OEH-v3 post-freeze development holdout.")
    else:
        expected_ids = registered_development
        if len(by_id) != len(instances) or set(by_id) != expected_ids:
            raise ValueError(f"Expected each of the {len(expected_ids)} development instances exactly once.")
    collection = PROTOCOL_CONFIG["collection"]
    families = PROTOCOL_CONFIG["behavior_policies"]["families"]
    expected = set()
    for instance_id in expected_ids:
        for family in families:
            for stream in ("pilot", "training"):
                sizes = collection["pilot_size_multipliers" if stream == "pilot" else "size_multipliers"]
                for seed in collection[f"{stream}_seed_indices"]:
                    stream_seed = (int(by_id[instance_id]["seed"]) + collection[f"{stream}_seed_offset"] + seed) % (2**32 - 1)
                    expected.update((instance_id, family, stream, stream_seed, n) for n in sizes)
    actual: Counter = Counter()
    physical_streams: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    stream_seeds: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(
        lambda: {"pilot": set(), "training": set()}
    )
    for record in manifest["datasets"]:
        if record.get("benchmark_split") != "development":
            raise ValueError("Only development records are permitted, including pilot records.")
        key = (record["instance_id"], record["policy_family"], record["stream"], record["stream_seed"], record["size_multiplier"])
        actual[key] += 1
        instance = by_id.get(record["instance_id"])
        if instance is None or record["mdp_seed"] != instance["seed"]:
            raise ValueError("Record seed does not match its instance.")
        multiplier = record["size_multiplier"]
        if multiplier is None or record["size"] != multiplier * record["num_states"] * record["num_actions"]:
            raise ValueError("Training/pilot prefix is capped or has an inconsistent size.")
        physical_streams[key[:4]].append(record)
        stream_seeds[key[:2]][key[2]].add(int(key[3]))
    if set(actual) != expected or any(count != 1 for count in actual.values()):
        raise ValueError(f"Dataset grid mismatch: missing={len(expected - set(actual))}, extra={len(set(actual) - expected)}, duplicates={sum(n > 1 for n in actual.values())}.")
    for stream_key, records in physical_streams.items():
        expected_multipliers = collection[
            "pilot_size_multipliers" if stream_key[2] == "pilot" else "size_multipliers"
        ]
        files = {str(record.get("file", "")) for record in records}
        hashes = {str(record.get("sha256", "")) for record in records}
        expected_max = max(expected_multipliers) * int(records[0]["num_states"]) * int(records[0]["num_actions"])
        if len(files) != 1 or "" in files or len(hashes) != 1 or "" in hashes:
            raise ValueError("Every seed stream must use one checksummed physical buffer for all nested prefixes.")
        if any(
            record.get("slice") != [0, record["size"]]
            or record.get("max_size") != expected_max
            for record in records
        ):
            raise ValueError("Every logical dataset must be a zero-based nested prefix of the registered maximum buffer.")
    overlapping = {
        group: sorted(seeds["pilot"] & seeds["training"])
        for group, seeds in stream_seeds.items()
        if seeds["pilot"] & seeds["training"]
    }
    if overlapping:
        raise ValueError(f"Pilot and training seed streams overlap within a group: {overlapping}.")
    training_count = sum(key[2] == "training" for key in actual)
    return {"instance_policy_groups": len(expected_ids) * len(families),
            "training_prefixes": training_count,
            "registered_runs": training_count * len(PROTOCOL_CONFIG["learning"]["algorithms"]),
            "bc_sanity_runs": training_count,
            "n_emp_rows": len(expected_ids) * len(families) * len(PROTOCOL_CONFIG["learning"]["algorithms"])}


def verify_training_buffers(dataset_dir: Path, manifest: dict[str, Any]) -> int:
    """Verify every distinct pilot/train buffer and reject conflicting hashes."""
    verified: dict[str, str] = {}
    for record in manifest["datasets"]:
        name, digest = record["file"], record["sha256"]
        if name in verified:
            if verified[name] != digest:
                raise ValueError(f"Conflicting checksums for {name}.")
        else:
            verified_dataset_path(dataset_dir, record)
            verified[name] = digest
    return len(verified)
