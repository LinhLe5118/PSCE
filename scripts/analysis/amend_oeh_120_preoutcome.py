"""One-time auditable repair of frozen 120-MDP preparation before outcomes."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.run_oeh_120_unseen import DATASET, OUTPUT, ROOT, now, sha256


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    protocol_path = OUTPUT / "protocol.json"
    dataset_seal_path = OUTPUT / "dataset_seal.json"
    amendment_path = OUTPUT / "amendment_001_preoutcome.json"
    if (amendment_path.exists() or (OUTPUT / "preoutcome_seal.json").exists()
            or (OUTPUT / "execution_plan.json").exists()
            or (OUTPUT / "execution").exists()
            or (OUTPUT / "labels.csv").exists()):
        raise RuntimeError("Amendment is only valid once before any learner outcome")
    original_protocol = protocol_path.read_bytes()
    original_seal = dataset_seal_path.read_bytes()
    doc = json.loads(original_protocol)
    dataset_seal = json.loads(original_seal)
    old_protocol_sha = sha256(protocol_path)
    if (old_protocol_sha != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip()
            or dataset_seal["protocol_sha256"] != old_protocol_sha):
        raise RuntimeError("Original protocol/dataset seal mismatch")
    for relative, expected in dataset_seal["sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dataset file changed: {relative}")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    if manifest["instance_count"] != 120 or manifest["physical_buffer_count"] != 6600:
        raise RuntimeError("Frozen dataset incomplete")
    runner_relative = "scripts/analysis/run_oeh_120_unseen.py"
    old_runner_sha = doc["source_sha256"][runner_relative]
    new_runner_sha = sha256(ROOT / runner_relative)
    if old_runner_sha == new_runner_sha:
        raise RuntimeError("Runner source did not change")
    for relative, expected in doc["source_sha256"].items():
        if relative != runner_relative and sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Other frozen source changed: {relative}")
    (OUTPUT / "original_protocol_before_amendment.json").write_bytes(original_protocol)
    (OUTPUT / "original_dataset_seal_before_amendment.json").write_bytes(original_seal)
    doc["source_sha256"][runner_relative] = new_runner_sha
    doc["preoutcome_amendment"] = {
        "number": 1,
        "reason": "Pilot feature extraction passed a file string instead of its dataset record; recovery now reuses verified generated buffers",
        "old_protocol_sha256": old_protocol_sha,
        "old_runner_sha256": old_runner_sha,
        "learner_outcomes_observed": 0,
        "dataset_manifest_unchanged": True,
    }
    save(protocol_path, doc)
    new_protocol_sha = sha256(protocol_path)
    (OUTPUT / "protocol.sha256").write_text(new_protocol_sha + "\n", encoding="ascii")
    dataset_seal["protocol_sha256"] = new_protocol_sha
    save(dataset_seal_path, dataset_seal)
    amendment = {
        "status": "preoutcome_repair_only", "at_utc": now(),
        "reason": doc["preoutcome_amendment"]["reason"],
        "old_protocol_sha256": old_protocol_sha,
        "new_protocol_sha256": new_protocol_sha,
        "old_runner_sha256": old_runner_sha,
        "new_runner_sha256": new_runner_sha,
        "original_protocol_copy_sha256": sha256(OUTPUT / "original_protocol_before_amendment.json"),
        "original_dataset_seal_copy_sha256": sha256(OUTPUT / "original_dataset_seal_before_amendment.json"),
        "amended_dataset_seal_sha256": sha256(dataset_seal_path),
        "learner_outcomes_observed": 0,
    }
    save(amendment_path, amendment)
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
