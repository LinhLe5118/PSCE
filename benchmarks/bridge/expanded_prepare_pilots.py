# Generated from the sealed fresh pipeline by materialize_expanded_pipeline.py.
"""Generate independent, label-free pilots for the frozen fresh BRIDGE panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from environments.behavior_policies import behavior_policy_suite
from scripts.data.generate_offline_datasets import (
    collect_trajectories, dataset_content_sha256, policy_sha256, validate_buffer)
from .expanded_adapter import PANEL, registry
from .expanded_preflight import load_eligible_mdp

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "datasets/bridge_v1_1_1/offline_pilot_expanded_2026-10-08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


def main() -> None:
    protocol_file = PANEL / "protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if protocol["status"] != "FRESH_BRIDGE_PROTOCOL_FROZEN_BEFORE_LABELS" or protocol["test_labels_read"]:
        raise RuntimeError("Fresh BRIDGE protocol not frozen")
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Fresh pilot manifest already sealed")
    ordinal = {name: i for i, name in enumerate(registry())}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in protocol["evaluation_tasks"]:
        mdp = load_eligible_mdp(task)
        horizon = int(protocol["episode_horizon"][task])
        policies = behavior_policy_suite(mdp, seed=seed("bridge-fresh-behavior-v1", task),
                                         variant_index=ordinal[task])
        if len(policies) != 5:
            raise RuntimeError("Expected exactly five behavior families")
        for policy in policies:
            stem = hashlib.sha256(f"{task}|{policy.family}".encode()).hexdigest()[:24]
            path = OUTPUT / f"{stem}.npz"
            checkpoint = OUTPUT / f"{stem}.json"
            expected = {"mdp_id": task, "policy_family": policy.family,
                        "behavior_policy_sha256": policy_sha256(policy),
                        "source_npz_sha256": mdp.metadata["source_sha256"],
                        "num_states": mdp.num_states, "num_actions": mdp.num_actions,
                        "initial_state_distribution": mdp.initial_state_distribution.tolist(),
                        "max_size": 10 * mdp.num_states * mdp.num_actions,
                        "horizon": horizon, "stream": "pilot", "file": path.name,
                        "reward_scale_positive": mdp.metadata["reward_scale_positive"]}
            if path.exists() or checkpoint.exists():
                if not path.exists() or not checkpoint.exists():
                    raise RuntimeError(f"Incomplete pilot checkpoint: {stem}")
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if any(saved.get(k) != v for k, v in expected.items()) or sha256(path) != saved.get("sha256"):
                    raise RuntimeError(f"Stale pilot checkpoint: {stem}")
            else:
                data = collect_trajectories(mdp, policy,
                                            seed=seed("bridge-fresh-pilot-v1", task, policy.family),
                                            size=expected["max_size"], horizon=horizon)
                validate_buffer(mdp, data, horizon=horizon)
                np.savez_compressed(path, **data)
                saved = {**expected, "sha256": sha256(path),
                         "content_sha256": dataset_content_sha256(data),
                         "pilot_nonzero_rewards": int(np.count_nonzero(data["reward"])),
                         "pilot_terminal_count": int(np.count_nonzero(data["terminal"])),
                         "test_labels_read": False}
                checkpoint.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
            rows.append(saved)
        print(f"Fresh BRIDGE pilot {task}: {len(rows)}/25 groups", flush=True)
    if len(rows) != 25:
        raise RuntimeError("Incomplete fresh BRIDGE pilots")
    manifest = {"status": "SEALED_FRESH_PILOT_ONLY_NO_TEST_LABELS",
                "protocol_sha256": sha256(protocol_file),
                "source_script_sha256": sha256(Path(__file__)),
                "tasks": protocol["evaluation_tasks"], "pilot_groups": rows,
                "training_streams_created": 0, "test_labels_read": False}
    output = OUTPUT / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.sha256").write_text(sha256(output) + "\n", encoding="ascii")
    print("Sealed 25 fresh BRIDGE pilots; no learner labels read", flush=True)


if __name__ == "__main__":
    main()
