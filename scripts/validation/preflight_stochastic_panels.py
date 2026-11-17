"""Check registered panel quotas and existing development kernels, no buffers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from config.protocol_config import PROTOCOL_CONFIG
from environments.benchmark_environments import benchmark_specs, resolve_dynamic_range
from environments.benchmark_panels import audit_dynamics, inference_panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-development", action="store_true",
                        help="Reconstruct existing 60-MDP development recipe, not new data")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; preserve previous evidence")
    quotas = Counter()
    for family, count in PROTOCOL_CONFIG["benchmark"]["evaluation_family_counts"].items():
        quotas[inference_panel(family)] += count
    if dict(quotas) != {"stochastic_primary": 84, "deterministic_control": 36}:
        raise ValueError("Evaluation quota differs from T1 v1.1.0")
    if args.fresh_development:
        from scripts.data.generate_oeh_v3_holdout import fresh_specs
        specs = fresh_specs()
    else:
        specs = [s for s in benchmark_specs() if s.split == "development"]
    rows = []
    for spec in specs:
        if spec.split != "development":
            raise ValueError("Development kernels only")
        actual, mdp, rejections = resolve_dynamic_range(spec)
        rows.append({"mdp_id": actual.instance_id, "family": actual.family,
                     "dynamic_range_rejections": rejections,
                     **audit_dynamics(mdp, actual.family)})
    report = {"status": "PASS", "version": "1.1.0",
              "evaluation_recipe_quotas": dict(quotas),
              "development_counts": dict(Counter(r["panel"] for r in rows)),
              "evaluation_kernels_materialized": False,
              "evaluation_manifest_or_buffers_read": False,
              "new_development_dataset_generated": False,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
