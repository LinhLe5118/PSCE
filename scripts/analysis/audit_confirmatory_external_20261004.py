"""Read-only integrity audit of the seventh cohort's 16,800 external jobs."""

from __future__ import annotations

import math

from scripts.analysis import run_oeh_confirmatory_20261004 as C
from scripts.analysis.run_unseen_validation import load_json, write_json


def main() -> None:
    C.configure()
    R = C.R
    R.base.verified_preoutcome()
    root = R.OUTPUT
    plan_path = root / "external_execution_plan.json"
    plan = load_json(plan_path)
    jobs = plan["jobs"]
    if len(jobs) != 16800 or plan["job_count"] != len(jobs):
        raise RuntimeError("External plan does not contain 16,800 jobs")
    if plan["preoutcome_seal_sha256"] != R.sha256(root / "preoutcome_seal.json"):
        raise RuntimeError("External plan uses a different pre-outcome seal")
    plan_hash = R.sha256(plan_path)
    target = root / "external_execution" / "jobs"
    if len(list(target.glob("*.json"))) != len(jobs):
        raise RuntimeError("External job-file count differs from plan")
    methods: dict[str, int] = {}
    for job in jobs:
        path = target / R.base.job_path(job).name
        record = load_json(path)
        outcome = record["outcome"]
        score = float(outcome["normalized_return"])
        if (record["job"] != job or record["execution_plan_sha256"] != plan_hash
                or outcome["buffer_sha256"] != job["buffer_sha256"]
                or outcome["status"] != "observed" or not math.isfinite(score)
                or bool(outcome["success"]) != (score >= 0.95)):
            raise RuntimeError(f"Invalid external job: {path}")
        methods[job["method"]] = methods.get(job["method"], 0) + 1
    result = {
        "status": "verified",
        "job_count": len(jobs),
        "methods": methods,
        "external_execution_plan_sha256": plan_hash,
        "preoutcome_seal_sha256": plan["preoutcome_seal_sha256"],
    }
    write_json(root / "external_execution_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
