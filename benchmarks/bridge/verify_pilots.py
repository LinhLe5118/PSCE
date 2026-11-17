"""Verify the sealed 70 BRIDGE pilot NPZ files without opening labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bridge_v1_1_1/offline_pilot_preoutcome_v1"
OUTPUT = ROOT / "artifacts/week8/bridge_pilot_verification_2026-10-07.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    manifest = DATA / "manifest.json"
    if manifest.with_suffix(".sha256").read_text(encoding="ascii").strip() != sha256(manifest):
        raise RuntimeError("Pilot seal hash mismatch")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if doc["status"] != "SEALED_PILOT_ONLY_NO_TEST_LABELS":
        raise RuntimeError("Pilot manifest not sealed")
    if sha256(ROOT / "benchmarks/bridge/prepare_pilots.py") != doc["source_script_sha256"]:
        raise RuntimeError("Pilot generator changed since seal")
    counts = Counter(row["mdp_id"] for row in doc["pilot_groups"])
    if len(counts) != 14 or set(counts.values()) != {5} or len(doc["pilot_groups"]) != 70:
        raise RuntimeError("Incomplete pilot task/behavior grid")
    if len({(r["mdp_id"], r["policy_family"]) for r in doc["pilot_groups"]}) != 70:
        raise RuntimeError("Duplicate pilot group")
    total_bytes = 0
    for row in doc["pilot_groups"]:
        path = DATA / row["file"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Pilot file hash mismatch: {path.name}")
        if row["max_size"] != 10 * row["num_states"] * row["num_actions"]:
            raise RuntimeError("Pilot x10 budget mismatch")
        total_bytes += path.stat().st_size
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"status": "SEALED_PILOT_VERIFIED_NO_TEST_LABELS",
        "manifest_sha256": sha256(manifest), "tasks": len(counts),
        "pilot_groups": len(doc["pilot_groups"]), "npz_total_bytes": total_bytes,
        "training_streams_created": 0, "test_labels_read": False}, indent=2) + "\n",
                      encoding="utf-8")
    print(f"Verified {len(doc['pilot_groups'])} label-free BRIDGE pilot files", flush=True)


if __name__ == "__main__":
    main()
