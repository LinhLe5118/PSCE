#!/usr/bin/env python
"""Materialize only the registered 120-MDP E1 cohort after execution review.

The historical generator's ``protocol`` switch is intentionally locked and
would also regenerate development.  This E1-only path calls its pinned recipe,
collection and serialization functions with all registered settings unchanged.
It never creates a development or OOD instance.  A partial output is retained
for audit and cannot be silently restarted under the same path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402
from environments.benchmark_environments import benchmark_specs  # noqa: E402
from evaluation.confirmatory import CONFIRMATORY_RUN_ID  # noqa: E402
from evaluation.io import sha256_file, write_json  # noqa: E402
from scripts.data import generate_offline_datasets as original  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registered_evaluation_specs():
    specs = [spec for spec in benchmark_specs() if spec.split == "evaluation"]
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.family] = counts.get(spec.family, 0) + 1
    if len(specs) != 120 or counts != PROTOCOL_CONFIG["benchmark"]["evaluation_family_counts"]:
        raise ValueError("Registered E1 generation recipe differs from T1 120-MDP quota")
    return specs


def generate_e1_bundle(output: Path, *, authorization_reference: str) -> dict[str, Any]:
    """Generate exact evaluation trajectories with no parameter overrides."""

    from scripts.training.run_week6_e1 import check_release_gate

    gate = check_release_gate()
    if gate["implementation_review"] != "GO" or not gate["source_and_review_committed_clean"]:
        raise ValueError("E1 generation requires committed source and independent execution review")
    if not authorization_reference.strip():
        raise ValueError("E1 generation requires explicit authorization reference")
    specs = registered_evaluation_specs()
    if output.exists():
        raise FileExistsError(f"E1 generation output already exists: {output}")
    output.mkdir(parents=True)
    log_path = output / "generation_access_log.json"
    log = {
        "run_id": CONFIRMATORY_RUN_ID,
        "status": "generation_started",
        "started_at_utc": _now(),
        "authorization_reference": authorization_reference,
        "source_commit": gate["source_commit"],
        "source_sha256": gate["source_sha256"],
        "registered_evaluation_mdps": len(specs),
        "development_mdps_generated": 0,
    }
    write_json(log_path, log)
    try:
        collection = PROTOCOL_CONFIG["collection"]
        pilot_seeds = tuple(int(value) for value in collection["pilot_seed_indices"])
        training_seeds = tuple(int(value) for value in collection["training_seed_indices"])
        pilot_sizes = tuple(int(value) for value in collection["pilot_size_multipliers"])
        horizon = int(collection["horizon"])
        records: list[dict[str, Any]] = []
        instance_records: list[dict[str, Any]] = []
        total_rejections = 0
        seen_mdps: dict[str, str] = {}
        for requested in specs:
            spec, mdp, rejections = original.resolve_dynamic_range(requested)
            fingerprint = hashlib.sha256()
            for array in (mdp.P, mdp.R, mdp.initial_state_distribution):
                fingerprint.update(array.tobytes())
            fingerprint.update(json.dumps(list(mdp.terminal_states)).encode("ascii"))
            identity = fingerprint.hexdigest()
            if identity in seen_mdps:
                raise ValueError(f"Exact base-MDP alias: {seen_mdps[identity]} and {spec.instance_id}")
            seen_mdps[identity] = spec.instance_id
            total_rejections += rejections
            range_value, optimal_return, random_return = original.dynamic_range(mdp)
            dynamics_audit = original.audit_dynamics(mdp, spec.family)
            instance_records.append({
                **spec.to_dict(), "requested_seed": requested.seed,
                "dynamic_range_rejections": rejections,
                "num_states": mdp.num_states, "num_actions": mdp.num_actions,
                "gamma": mdp.gamma, "terminal_states": list(mdp.terminal_states),
                "reward_min": float(mdp.R.min()), "reward_max": float(mdp.R.max()),
                "dynamic_range": range_value,
                "optimal_return": optimal_return, "random_return": random_return,
                "metadata": original._json_ready(mdp.metadata),
                "dynamics_audit": dynamics_audit,
            })
            policies = original.behavior_policy_suite(
                mdp, seed=spec.seed + 3_000_000, variant_index=spec.ordinal)
            for policy in policies:
                reachability = (
                    original._reachable_states(mdp, np.ones_like(policy.probabilities, dtype=bool)),
                    original._reachable_states(mdp, policy.probabilities > 0.0),
                )
                policy_return = original.evaluate_policy(mdp, policy.probabilities).expected_return
                policy_normalized_return = (policy_return - random_return) / range_value
                for stream in ("pilot", "training"):
                    seed_indices = pilot_seeds if stream == "pilot" else training_seeds
                    sizes = original._logical_sizes(
                        mdp, None, stream=stream,
                        multipliers=pilot_sizes if stream == "pilot" else original.SIZE_MULTIPLIERS)
                    collection_size = max(sizes)
                    for seed_index in seed_indices:
                        stream_seed = original._stream_seed(spec.seed, stream, int(seed_index))
                        config = original.CollectionConfig(
                            instance_id=spec.instance_id, benchmark_split="evaluation",
                            mdp_seed=spec.seed, policy_family=policy.family,
                            policy_seed=policy.seed, stream=stream,
                            stream_seed=stream_seed, max_size=collection_size,
                            horizon=horizon,
                        )
                        data = original.collect_trajectories(
                            mdp, policy, seed=stream_seed, size=collection_size, horizon=horizon)
                        original.validate_buffer(mdp, data, horizon=horizon)
                        buffer_id = original._stable_identifier(
                            "buffer", spec.instance_id, spec.seed, policy.family,
                            stream, int(seed_index))
                        path = output / f"buffer-{buffer_id}.npz"
                        sha256 = original._write_buffer(path, data)
                        for size in sizes:
                            records.append(original._collection_record(
                                spec=spec, mdp=mdp, policy=policy, config=config,
                                data=data, size=size, file_name=path.name,
                                file_sha256=sha256,
                                policy_expected_return=policy_return,
                                policy_normalized_return=policy_normalized_return,
                                reachability=reachability,
                            ))
        stratification = original.assign_coverage_strata(records, thresholds=None)
        effective = deepcopy(PROTOCOL_CONFIG)
        effective["run_scope"] = {
            "mode": "e1_confirmatory", "instances": [spec.to_dict() for spec in specs],
            "max_size_cap": None, "streams": ["pilot", "training"],
        }
        manifest = {
            "schema_version": original.SCHEMA_VERSION,
            "protocol_config": original._json_ready(PROTOCOL_CONFIG),
            "protocol_status": PROTOCOL_CONFIG["protocol_status"],
            "pending_validations": list(PROTOCOL_CONFIG["pending_validations"]),
            "effective_protocol": original._json_ready(effective),
            "run_mode": "e1_confirmatory", "protocol_deviations": [],
            "deviation_reason": None,
            "e1_authorization_reference": authorization_reference,
            "e1_source_review": "docs/week6/E1_EXECUTION_IMPLEMENTATION_REVIEW.json",
            "estimator_input_index": "data_only_index.json",
            "access_policy": {
                "data_only_estimator": "NPZ transitions plus data_only_index.json",
                "oracle_audit_only": "manifest.json fields for exact MDP, policy quality, and coverage labels",
            },
            "generator_provenance": original._generator_provenance(),
            "e1_generator_sha256": sha256_file(Path(__file__)),
            "benchmark_design": {
                "evaluation_count": 120,
                "development_count": int(PROTOCOL_CONFIG["benchmark"]["development_count"]),
                "ood_count": len(PROTOCOL_CONFIG["benchmark"]["ood_representatives"]),
            },
            "benchmark_seed": original.BENCHMARK_SEED,
            "storage": "natural trajectory buffers with nested prefixes",
            "transition_fields": list(original.TRANSITION_FIELDS),
            "size_multipliers": list(original.SIZE_MULTIPLIERS),
            "training_size_multipliers": list(original.SIZE_MULTIPLIERS),
            "pilot_size_multipliers": list(pilot_sizes),
            "reference_pilot_n0_coefficient": original.DEFAULT_N0_COEFFICIENT,
            "pilot_training_independence": "disjoint deterministic seed namespaces",
            "coverage_rule": "generate_then_measure_then_stratify; no post-hoc filtering",
            "coverage_stratification": stratification,
            "dynamic_range_rejection_count": total_rejections,
            "instance_count": len(instance_records),
            "physical_buffer_count": len({(row["file"], row["sha256"]) for row in records}),
            "logical_dataset_count": len(records),
            "instances": instance_records, "datasets": records,
        }
        index_path = output / "data_only_index.json"
        index_path.write_text(
            json.dumps(original._json_ready(original._data_only_index(records)), indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["estimator_input_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(original._json_ready(manifest), indent=2) + "\n", encoding="utf-8")
        log.update(status="sealed_bundle_generated", completed_at_utc=_now(),
                   manifest_sha256=sha256_file(manifest_path),
                   data_only_index_sha256=sha256_file(index_path),
                   physical_buffers=manifest["physical_buffer_count"])
        write_json(log_path, log)
        return {"output": str(output.resolve()), "manifest_sha256": log["manifest_sha256"],
                "base_mdps": len(instance_records), "logical_datasets": len(records),
                "physical_buffers": manifest["physical_buffer_count"]}
    except Exception as exc:
        log.update(status="generation_failed_stop_on_error", failed_at_utc=_now(),
                   error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(log_path, log)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/oehv4e_evaluation_v1_1")
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate_e1_bundle(args.output, authorization_reference=args.authorization_reference), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
