"""One-time pre-outcome protocol repair for generator-resolved MDP seeds."""
from __future__ import annotations

import json

from scripts.analysis.run_oeh_120_unseen import OUTPUT, ROOT, now, sha256


def save(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    protocol_path = OUTPUT / "protocol.json"
    seal_path = OUTPUT / "dataset_seal.json"
    audit_path = OUTPUT / "actual_kernel_audit_preoutcome.json"
    amendment_path = OUTPUT / "amendment_002_resolved_kernels.json"
    if (amendment_path.exists() or (OUTPUT / "preoutcome_seal.json").exists()
            or (OUTPUT / "execution_plan.json").exists()
            or (OUTPUT / "execution").exists()):
        raise RuntimeError("Kernel amendment is allowed once before outcomes")
    doc = json.loads(protocol_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    old_hash = sha256(protocol_path)
    if (old_hash != (OUTPUT / "protocol.sha256").read_text(encoding="ascii").strip()
            or seal["protocol_sha256"] != old_hash
            or audit["protocol_sha256"] != old_hash
            or not audit["all_actual_kernels_new_and_distinct"]
            or audit["conflicts"] or audit["mdps"] != 120):
        raise RuntimeError("Original protocol, dataset seal or actual kernel audit mismatch")
    for relative, expected in doc["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for relative, expected in seal["sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dataset changed: {relative}")
    actual = {row["mdp_id"]: row["actual_kernel"] for row in audit["rows"]}
    requested = doc["kernel_fingerprints"]
    if (set(actual) != set(requested)
            or any(requested[name] != row["requested_kernel"]
                   for name, row in ((item["mdp_id"], item) for item in audit["rows"]))):
        raise RuntimeError("Actual kernel audit does not match frozen requested kernels")
    original_protocol = protocol_path.read_bytes()
    original_seal = seal_path.read_bytes()
    (OUTPUT / "protocol_before_amendment_002.json").write_bytes(original_protocol)
    (OUTPUT / "dataset_seal_before_amendment_002.json").write_bytes(original_seal)
    doc["requested_kernel_fingerprints"] = requested
    doc["kernel_fingerprints"] = actual
    doc["preoutcome_amendment_2"] = {
        "number": 2,
        "reason": "Benchmark generator resolved five seeds through its registered dynamic-range filter; fingerprint seal now records the actual materialized kernels",
        "old_protocol_sha256": old_hash,
        "actual_kernel_audit_sha256": sha256(audit_path),
        "learner_outcomes_observed": 0,
    }
    save(protocol_path, doc)
    new_hash = sha256(protocol_path)
    (OUTPUT / "protocol.sha256").write_text(new_hash + "\n", encoding="ascii")
    seal["protocol_sha256"] = new_hash
    save(seal_path, seal)
    amendment = {
        "status": "preoutcome_resolved_seed_repair", "at_utc": now(),
        "old_protocol_sha256": old_hash,
        "new_protocol_sha256": new_hash,
        "resolved_seed_count": audit["generator_resolved_seed_count"],
        "actual_kernels_new_and_distinct": True,
        "actual_kernel_audit_sha256": sha256(audit_path),
        "original_protocol_copy_sha256": sha256(OUTPUT / "protocol_before_amendment_002.json"),
        "original_dataset_seal_copy_sha256": sha256(OUTPUT / "dataset_seal_before_amendment_002.json"),
        "new_dataset_seal_sha256": sha256(seal_path),
        "learner_outcomes_observed": 0,
    }
    save(amendment_path, amendment)
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
