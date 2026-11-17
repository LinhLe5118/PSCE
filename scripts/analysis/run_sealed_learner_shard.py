"""Execute a disjoint shard of an existing sealed 120-MDP learner plan.

Only the execution schedule changes. Each job is scored by the original
callback, written in its original schema, and immediately verified against
the immutable execution-plan SHA-256 and buffer SHA-256. The original runner
must subsequently be invoked with ``evaluate --resume`` to verify all jobs
and create ``execution_complete.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from scripts.analysis import run_oeh_120_unseen as base
from scripts.analysis.run_unseen_validation import archive_imports, load_json, write_json
from scripts.analysis.run_oeh_120_successor import DATASET, OUTPUT


base.OUTPUT = OUTPUT
base.DATASET = DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard identity")
    doc, manifest = base.verified_preoutcome()
    if (OUTPUT / "failure_evaluate.json").exists():
        raise RuntimeError("Recorded original-runner failure requires audit")
    pipeline, _, _, seed_index = archive_imports()
    planned = base.jobs(manifest, seed_index)
    plan_path = OUTPUT / "execution_plan.json"
    frozen_plan = load_json(plan_path)
    plan_sha = base.sha256(plan_path)
    if (len(planned) != doc["expected_jobs"] or frozen_plan["jobs"] != planned
            or frozen_plan["preoutcome_seal_sha256"] != base.sha256(OUTPUT / "preoutcome_seal.json")):
        raise RuntimeError("Shard differs from original sealed execution plan")
    callback = pipeline.build_core_train_score_callback(DATASET, manifest)
    started = time.perf_counter()
    done, skipped = 0, 0
    for index, job in enumerate(planned):
        if index % args.shards != args.shard:
            continue
        path = base.job_path(job)
        if path.exists():
            base.verify_job(job, path, plan_sha)
            skipped += 1
            continue
        try:
            tick = time.perf_counter()
            outcome = callback(job)
            write_json(path, {"job": job, "execution_plan_sha256": plan_sha,
                              "outcome": dict(outcome),
                              "runtime_seconds": time.perf_counter() - tick,
                              "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            base.verify_job(job, path, plan_sha)
            done += 1
        except Exception as error:
            failure = OUTPUT / "execution" / f"failure_shard_{args.shard}.json"
            write_json(failure, {"job_index_zero_based": index, "job": job,
                                 "error_type": type(error).__name__, "error": str(error),
                                 "traceback": traceback.format_exc(),
                                 "plan_sha256": plan_sha})
            raise
        if (done + skipped) % 500 == 0:
            write_json(OUTPUT / "execution" / f"progress_shard_{args.shard}.json", {
                "shard": args.shard, "shards": args.shards,
                "processed": done + skipped, "new": done, "skipped": skipped,
                "target": sum(i % args.shards == args.shard for i in range(len(planned))),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"Shard {args.shard}: {done + skipped} verified", flush=True)
    write_json(OUTPUT / "execution" / f"complete_shard_{args.shard}.json", {
        "shard": args.shard, "shards": args.shards, "plan_sha256": plan_sha,
        "processed": done + skipped, "new": done, "skipped": skipped,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Shard {args.shard} complete: {done + skipped} verified", flush=True)


if __name__ == "__main__":
    main()
