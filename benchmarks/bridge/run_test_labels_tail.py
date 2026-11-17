"""Compute disjoint tail BRIDGE test-label groups with the frozen label kernel.

This is an execution-only accelerator for the already sealed reduced comparison.
It shares the label checkpoints with run_test_labels.py; no predictor is fitted here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

from benchmarks.bridge.adapter import load_sticky_mdp
from benchmarks.bridge.run_test_labels import (
    PILOT, ROOT, prelabel_gate, seed, sha256,
)
from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import policy_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    if not (8 <= args.start < args.stop <= 14):
        raise ValueError("Tail workers are restricted to frozen task indices 8..13")

    groups, tasks, seal = prelabel_gate("reduced")
    output = ROOT / "artifacts/week8/bridge_reduced_test_labels_2026-10-07"
    seal_path = output / "prelabel_protocol_seal.json"
    if not seal_path.exists() or json.loads(seal_path.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("Existing prelabel seal does not match the frozen runner")
    if (output / "manifest.json").exists():
        raise RuntimeError("Label manifest is already complete")

    from benchmarks.bridge.labels import one_group  # Only after the prelabel gate.

    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    candidates = [row for row in registry if row["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    ordinal = {row["mdp_name"]: index for index, row in enumerate(candidates)}
    lookup = {(row["mdp_id"], row["policy_family"]): row for row in groups}
    group_dir = output / "groups"

    for task in tasks[args.start:args.stop]:
        mdp = load_sticky_mdp(task)
        policies = behavior_policy_suite(mdp, seed=seed("bridge-behavior-v1", task),
                                         variant_index=ordinal[task])
        for policy in policies:
            source = lookup[(task, policy.family)]
            if policy_sha256(policy) != source["behavior_policy_sha256"]:
                raise RuntimeError(f"Behavior changed: {task}/{policy.family}")
            stem = hashlib.sha256(f"{task}|{policy.family}".encode()).hexdigest()[:24]
            path = group_dir / f"{stem}.json"
            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
                if (item.get("pilot_sha256") != source["sha256"] or
                        item.get("prelabel_protocol_sha256") != sha256(seal_path)):
                    raise RuntimeError(f"Stale label checkpoint: {path.name}")
            else:
                start = time.perf_counter()
                try:
                    result = one_group(mdp, policy)
                    item = {"status": "BRIDGE_TEST_LABEL_COMPLETE", **result,
                            "pilot_sha256": source["sha256"],
                            "prelabel_protocol_sha256": sha256(seal_path)}
                except Exception as exc:
                    item = {"status": "BRIDGE_TEST_LABEL_FAILURE", "task": task,
                            "behavior": policy.family, "pilot_sha256": source["sha256"],
                            "prelabel_protocol_sha256": sha256(seal_path),
                            "runtime_seconds": time.perf_counter() - start,
                            "error": repr(exc), "traceback": traceback.format_exc()}
                # Publish a complete checkpoint in one rename so readers never see partial JSON.
                temporary = path.with_suffix(f".{args.start}.tmp")
                temporary.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
                temporary.replace(path)
            print(f"BRIDGE tail label {task}/{policy.family}: {item['status']}", flush=True)
            if item["status"] != "BRIDGE_TEST_LABEL_COMPLETE":
                raise RuntimeError(f"Label group failed: {task}/{policy.family}")


if __name__ == "__main__":
    main()
