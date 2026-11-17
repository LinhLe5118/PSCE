"""Validate new BRIDGE arrays, reward domain, and sticky adapter before labels."""
from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .expanded_adapter import PANEL, load_expanded_sticky_mdp, registry

DATA = Path(__file__).resolve().parents[2] / "datasets/bridge_v1_1_1/mdps"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eligible_mdp(name: str):
    gate = json.loads((PANEL / "adapter_reward_preflight.json").read_text(encoding="utf-8"))
    row = next((x for x in gate["tasks"] if x["mdp_name"] == name), None)
    if row is None or row["status"] != "ELIGIBLE_NONNEGATIVE":
        raise ValueError("Task not eligible in frozen expanded panel")
    path = DATA / name / "consolidated.npz"
    if sha256(path) != row["source_npz_sha256"]:
        raise ValueError("Published MDP file changed")
    mdp = load_expanded_sticky_mdp(name)
    scale = float(row["reward_scale_positive"])
    normalized = replace(mdp, R=mdp.R / scale,
                         metadata={**mdp.metadata, "reward_scale_positive": scale,
                                   "reward_normalization": "positive reward scaling; no shift"})
    if np.any(normalized.R < -1e-12) or np.any(normalized.R > 1 + 1e-12):
        raise ValueError("Scaled reward outside current OEH posterior bound")
    normalized.validate()
    return normalized


def main() -> None:
    output = PANEL / "adapter_reward_preflight.json"
    if output.exists():
        raise FileExistsError("Expanded adapter preflight already recorded")
    frozen_file = PANEL / "manifest.json"
    download_file = PANEL / "download_manifest.json"
    frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
    downloaded = json.loads(download_file.read_text(encoding="utf-8"))
    if (frozen["status"] != "EXPANDED_BRIDGE_METADATA_SCREEN_FROZEN_BEFORE_NEW_LABELS" or
            downloaded["status"] != "EXPANDED_BRIDGE_MEMBERS_VERIFIED_NO_LABELS" or
            downloaded["frozen_registry_sha256"] != frozen["source_sha256"]["candidate_registry"]):
        raise ValueError("Expanded source gate failed")
    files = {Path(item["local_path"]).parent.name: item for item in downloaded["members"]}
    rows = []
    for name, spec in registry().items():
        started = time.perf_counter()
        path = DATA / name / "consolidated.npz"
        entry = files.get(name)
        if entry is None or sha256(path) != entry["file_sha256"]:
            raise ValueError(f"Missing/unverified source file: {name}")
        with np.load(path, allow_pickle=False) as raw:
            transitions, rewards = raw["transitions"], raw["rewards"]
            reward_min, reward_max = float(np.min(rewards)), float(np.max(rewards))
            if transitions.shape != rewards.shape:
                raise ValueError(f"Transition/reward shape mismatch: {name}")
        status = "ELIGIBLE_NONNEGATIVE" if reward_min >= 0 and reward_max > 0 else "EXCLUDED_REWARD_DOMAIN"
        mdp = load_expanded_sticky_mdp(name) if status == "ELIGIBLE_NONNEGATIVE" else None
        if mdp is not None and (mdp.num_actions != int(spec["actions"]) or
                                not np.allclose(mdp.P.sum(axis=2), 1.0, atol=1e-10)):
            raise ValueError(f"Sticky adapter parity/basic stochasticity failed: {name}")
        rows.append({"mdp_name": name, "origin": spec["origin"], "status": status,
                     "source_npz_sha256": entry["file_sha256"],
                     "reward_min_published": reward_min, "reward_max_published": reward_max,
                     "reward_scale_positive": reward_max if status == "ELIGIBLE_NONNEGATIVE" else None,
                     "base_states": int(spec["base_states"]),
                     "published_sticky_states": int(spec["sticky_states"]),
                     "adapter_states": int(mdp.num_states) if mdp is not None else None,
                     "actions": int(spec["actions"]), "horizon": int(spec["horizon"]),
                     "adapter_seconds": time.perf_counter() - started})
        print(f"Preflight {name}: {status}, S={rows[-1]['adapter_states']}, {rows[-1]['adapter_seconds']:.1f}s", flush=True)
        del mdp
        gc.collect()
    output.write_text(json.dumps({
        "status": "EXPANDED_BRIDGE_ADAPTER_REWARD_PREFLIGHT_NO_LABELS",
        "frozen_manifest_sha256": sha256(frozen_file),
        "download_manifest_sha256": sha256(download_file),
        "source_script_sha256": sha256(Path(__file__)),
        "adapter_script_sha256": sha256(Path(__file__).with_name("expanded_adapter.py")),
        "tasks": rows, "test_labels_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("Expanded adapter/reward preflight complete; no learner labels read", flush=True)


if __name__ == "__main__":
    main()
