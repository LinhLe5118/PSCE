"""Write the original sealed execution plan before third-cohort outcomes.

This only changes scheduling. The original runner later verifies all jobs
with ``evaluate --resume`` and its frozen decision rule remains unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.run_oeh_120_cross_capacity import DATASET, OUTPUT
from scripts.analysis.run_unseen_validation import archive_imports, write_json


base.OUTPUT = OUTPUT
base.DATASET = DATASET


def main() -> None:
    doc, manifest = base.verified_preoutcome()
    execution = OUTPUT / "execution"
    plan_path = OUTPUT / "execution_plan.json"
    amendment = OUTPUT / "execution_schedule_amendment.json"
    if (plan_path.exists() or amendment.exists() or (execution / "jobs").exists()
            or (OUTPUT / "result_manifest.json").exists()):
        raise FileExistsError("Third-cohort execution has already started")
    _, _, _, seed_index = archive_imports()
    planned = base.jobs(manifest, seed_index)
    if len(planned) != doc["expected_jobs"]:
        raise RuntimeError("Sealed expected job count differs")
    plan = {
        "protocol_sha256": base.sha256(OUTPUT / "protocol.json"),
        "preoutcome_seal_sha256": base.sha256(OUTPUT / "preoutcome_seal.json"),
        "dataset_manifest_sha256": base.sha256(DATASET / "manifest.json"),
        "jobs": planned,
        "job_count": len(planned),
    }
    write_json(plan_path, plan)
    (execution / "jobs").mkdir(parents=True)
    worker = Path(__file__).with_name("run_sealed_cross_learner_shard.py")
    write_json(amendment, {
        "schema": "execution_schedule_amendment_v1",
        "reason": "Execute disjoint sealed learner jobs on eight CPU shards",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shards": 8,
        "learner_outcomes_observed": 0,
        "protocol_sha256": plan["protocol_sha256"],
        "preoutcome_seal_sha256": plan["preoutcome_seal_sha256"],
        "execution_plan_sha256": base.sha256(plan_path),
        "plan_builder_sha256": base.sha256(Path(__file__)),
        "shard_worker_sha256": base.sha256(worker),
        "method_data_forecasts_and_decision_rule_changed": False,
        "final_verification": "Original runner evaluate --resume checks all 108000 jobs against frozen plan before analysis",
    })
    print(f"Sealed {len(planned)} third-cohort learner jobs for eight disjoint shards", flush=True)


if __name__ == "__main__":
    main()
