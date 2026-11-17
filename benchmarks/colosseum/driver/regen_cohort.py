"""Regenerate the pilot streams of an opened cohort and verify them.

Usage: python driver/regen_cohort.py c7
Pilot buffers only (x2/x5/x10). Verification: every resolved kernel must
match the sealed fingerprint of that cohort.
"""
from __future__ import annotations

import sys
import time

from common import COHORTS, DATA, load_json, use_workspace, write_json

name = sys.argv[1]
use_workspace()
from scripts.analysis.run_unseen_validation import archive_imports  # noqa: E402
from scripts.analysis import run_oeh_120_unseen as base  # noqa: E402

_, _, generator, _ = archive_imports()
from environments.benchmark_environments import BenchmarkSpec, make_benchmark_mdp  # noqa: E402

artifact = COHORTS[name]
doc = load_json(artifact / "protocol.json")
target = DATA / name
tick = time.time()
specs = [BenchmarkSpec(**item) for item in doc["requested_specs"]]
manifest = generator.generate_benchmark_bundle(
    target, specs=specs, pilot_seeds=(0,), training_seeds=tuple(range(10)), max_size=None,
    horizon=100, run_mode="test", deviation_reason=f"external trial: regenerate {name} pilots",
    pilot_size_multipliers=(2, 5, 10), streams=("pilot",))
audit = artifact / "actual_kernel_audit_preoutcome.json"
audit_doc = load_json(audit) if audit.exists() else {}
if "actual_kernel_fingerprints" in audit_doc:
    expected = audit_doc["actual_kernel_fingerprints"]
elif "rows" in audit_doc:
    expected = {row["mdp_id"]: row["actual_kernel"] for row in audit_doc["rows"]}
else:
    expected = doc.get("kernel_fingerprints") or {}
actual = {item["instance_id"]: base.fingerprint(make_benchmark_mdp(type("Spec", (), item)()))
          for item in manifest["instances"]}
mismatch = sorted(k for k in expected if actual.get(k) != expected[k])
write_json(target / "regen_check.json", {"cohort": name, "instances": len(actual), "fingerprints_checked": len(expected),
                                         "kernel_mismatch": mismatch, "seconds": time.time() - tick})
print(f"{name}: {len(actual)} kernels, checked={len(expected)}, mismatches={len(mismatch)}, {time.time() - tick:.0f}s")
if mismatch:
    raise SystemExit(f"kernel mismatch: {mismatch[:5]}")
