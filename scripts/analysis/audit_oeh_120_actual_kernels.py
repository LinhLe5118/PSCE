"""Audit generator-resolved final MDP kernels before any learner outcomes."""
from __future__ import annotations

import json

from scripts.analysis.run_oeh_120_unseen import (
    DATASET, OUTPUT, fingerprint, now, prior_kernels, protocol, sha256,
)
from scripts.analysis.run_unseen_validation import archive_imports


def main() -> None:
    output = OUTPUT / "actual_kernel_audit_preoutcome.json"
    if (output.exists() or (OUTPUT / "preoutcome_seal.json").exists()
            or (OUTPUT / "execution_plan.json").exists()):
        raise RuntimeError("Kernel audit must run exactly once before outcomes")
    doc = protocol()
    archive_imports()
    from environments.benchmark_environments import make_benchmark_mdp

    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen = prior_kernels()
    prior = set(seen)
    rows, conflicts = [], []
    for instance in manifest["instances"]:
        mdp = make_benchmark_mdp(type("Spec", (), instance)())
        digest = fingerprint(mdp)
        if digest in seen:
            conflicts.append({"mdp_id": instance["instance_id"],
                              "prior_cohort": digest in prior, "fingerprint": digest})
        seen.add(digest)
        rows.append({"mdp_id": instance["instance_id"],
                     "requested_seed": instance.get("requested_seed", instance["seed"]),
                     "resolved_seed": instance["seed"],
                     "dynamic_range_rejections": instance.get("dynamic_range_rejections", 0),
                     "requested_kernel": doc["kernel_fingerprints"][instance["instance_id"]],
                     "actual_kernel": digest})
    if len(rows) != 120 or len({row["mdp_id"] for row in rows}) != 120:
        raise RuntimeError("Incomplete 120-MDP actual kernel audit")
    result = {
        "status": "preoutcome_actual_kernel_audit",
        "at_utc": now(), "mdps": 120,
        "generator_resolved_seed_count": sum(row["requested_seed"] != row["resolved_seed"] for row in rows),
        "requested_to_actual_kernel_changes": sum(row["requested_kernel"] != row["actual_kernel"] for row in rows),
        "conflicts": conflicts,
        "all_actual_kernels_new_and_distinct": len(conflicts) == 0,
        "dataset_manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(OUTPUT / "protocol.json"),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "mdps", "generator_resolved_seed_count", "requested_to_actual_kernel_changes",
        "conflicts", "all_actual_kernels_new_and_distinct")}, indent=2))


if __name__ == "__main__":
    main()
