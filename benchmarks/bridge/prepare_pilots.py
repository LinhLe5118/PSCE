"""Collect sealed, label-free pilot streams for frozen BRIDGE candidate tasks.

This phase reads published MDP kernels to generate behavior policies and pilot
trajectories. It never trains a learner or opens test difficulty labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.bridge.adapter import load_sticky_mdp
from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import (
    collect_trajectories, dataset_content_sha256, policy_sha256, validate_buffer,
)

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "artifacts/week8/bridge_transfer_preflight_freeze_2026-10-07.json"
OUTPUT = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> None:
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Pilot manifest already sealed; refusing to overwrite")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["status"] != "PREFLIGHT_SPLIT_FROZEN_NOT_BENCHMARK_GO":
        raise RuntimeError("BRIDGE split not frozen")
    if sha256(ROOT / "benchmarks/bridge/adapter.py") != frozen["source_sha256"]["benchmarks/bridge/adapter.py"]:
        raise RuntimeError("Adapter changed after preflight freeze")
    registry = list(csv.DictReader((ROOT / "benchmarks/bridge/candidate_registry.csv").open(
        encoding="utf-8", newline="")))
    candidates = [r for r in registry if r["resource_screen"] == "CANDIDATE_METADATA_ONLY"]
    task_ordinal = {r["mdp_name"]: i for i, r in enumerate(candidates)}
    tests = frozen["candidate_test_tasks"]
    if len(tests) != 14 or any(t not in task_ordinal for t in tests):
        raise RuntimeError("Candidate test task registry changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tests:
        mdp = load_sticky_mdp(task)
        policies = behavior_policy_suite(mdp, seed=seed("bridge-behavior-v1", task),
                                         variant_index=task_ordinal[task])
        if len(policies) != 5:
            raise RuntimeError("Incomplete behavior suite")
        for policy in policies:
            name = hashlib.sha256(f"{task}|{policy.family}".encode()).hexdigest()[:24]
            path = OUTPUT / f"{name}.npz"
            checkpoint = OUTPUT / f"{name}.json"
            expected = {"mdp_id": task, "policy_family": policy.family,
                        "behavior_policy_sha256": policy_sha256(policy),
                        "source_npz_sha256": mdp.metadata["source_sha256"],
                        "num_states": mdp.num_states, "num_actions": mdp.num_actions,
                        "initial_state_distribution": mdp.initial_state_distribution.tolist(),
                        "max_size": 10 * mdp.num_states * mdp.num_actions,
                        "horizon": 100, "stream": "pilot", "file": path.name}
            if path.exists() or checkpoint.exists():
                if not path.exists() or not checkpoint.exists():
                    raise RuntimeError(f"Incomplete pilot checkpoint: {name}")
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if any(saved.get(k) != v for k, v in expected.items()) or sha256(path) != saved.get("sha256"):
                    raise RuntimeError(f"Stale pilot checkpoint: {name}")
            else:
                data = collect_trajectories(
                    mdp, policy, seed=seed("bridge-pilot-v1", task, policy.family),
                    size=expected["max_size"], horizon=100)
                validate_buffer(mdp, data, horizon=100)
                np.savez_compressed(path, **data)
                saved = {**expected, "sha256": sha256(path),
                         "content_sha256": dataset_content_sha256(data),
                         "pilot_nonzero_rewards": int(np.count_nonzero(data["reward"])),
                         "pilot_terminal_count": int(np.count_nonzero(data["terminal"]))}
                checkpoint.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
            rows.append(saved)
        print(f"BRIDGE label-free pilot: {task} ({len(rows)}/{len(tests)*5} groups)", flush=True)
    if len(rows) != 70:
        raise RuntimeError("Incomplete BRIDGE pilot grid")
    manifest = {"status": "SEALED_PILOT_ONLY_NO_TEST_LABELS", "freeze_sha256": sha256(FREEZE),
                "source_script_sha256": sha256(Path(__file__)), "tasks": tests,
                "pilot_groups": rows, "training_streams_created": 0, "test_labels_read": False}
    path = OUTPUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.sha256").write_text(sha256(path) + "\n", encoding="ascii")
    print(f"Sealed {len(rows)} label-free BRIDGE pilots: {path}", flush=True)


if __name__ == "__main__":
    main()
