"""Synthetic development probes of the OEH regret-certificate redesign.

Uses generated per-(s,a) outcome counts, NOT project evaluation datasets.
The oracle stays on the scoring side. This is neither a learner comparison
nor a validation of future sample-requirement predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from oeh_certificate import (  # noqa: E402
    ModelBox, confidence_box, certify_depths, pessimistic_depth_policies,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    p = np.zeros((3, 2, 3))
    p[:, :, 2] = 1
    p[0, 0] = [0, .8, .2]
    r = np.array([[0, .6], [1, 0], [0, 0]])
    exact = ModelBox(p, p, r, r, 1)
    gamma, epsilon, delta = .9, .005, .05
    oracle_policies = pessimistic_depth_policies(exact, gamma, [1, 1, 0], 3)
    oracle = certify_depths(exact, gamma, [1, 0, 0], oracle_policies, epsilon=epsilon)
    rows = []
    for method in ("bernstein", "hoeffding"):
        for n in (20, 200, 2000, 20000, 200000):
            for seed in range(args.seeds):
                # Identical counts across confidence methods; no rejection sampling.
                rng = np.random.default_rng(seed)
                counts = np.array([[rng.multinomial(n, p[s, a]) for a in range(2)] for s in range(3)])
                box = confidence_box(counts, n*r, n*r*r, reward_bound=1, delta=delta, method=method)
                policies = pessimistic_depth_policies(box, gamma, [1, 1, 0], 3)
                certificate = certify_depths(box, gamma, [1, 0, 0], policies, epsilon=epsilon)
                scoring = certify_depths(exact, gamma, [1, 0, 0], policies, epsilon=epsilon)
                contains_true_model = bool(
                    np.all(box.p_lower <= p) and np.all(p <= box.p_upper)
                    and np.all(box.r_lower <= r) and np.all(r <= box.r_upper))
                for k, (upper, regret) in enumerate(zip(certificate["normalized_regret_upper"],
                                                       scoring["normalized_regret_upper"])):
                    rows.append({
                        "method": method, "seed": seed, "visits_per_pair": n,
                        "total_transitions": 6*n, "depth": k,
                        "start_action": int(policies[k, 0]),
                        "normalized_regret_upper": upper, "oracle_regret_numerical": regret,
                        "certified": upper <= epsilon,
                        "false_certificate": upper <= epsilon and regret > epsilon + 1e-8,
                        "contains_true_model": contains_true_model,
                        "bound_violation_over_1e_8": regret > upper + 1e-8,
                    })
    csv_path = args.output_dir / "probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [ROOT / "code/oeh_certificate.py", Path(__file__).resolve(),
               ROOT / "tests/test_oeh_certificate.py"]
    report = {
        "status": "exploratory_synthetic_only",
        "sampling": "fixed visits per state-action pair; deterministic rewards; stochastic transition",
        "project_datasets_read": [],
        "settings": {"gamma": gamma, "epsilon": epsilon, "delta": delta, "seeds": args.seeds,
                     "k_max": 3, "normalization": "reward_bound/(1-gamma)"},
        "oracle_limit": oracle,
        "num_model_boxes": len(rows)//4, "num_depth_rows": len(rows),
        "false_certificate_rows": sum(x["false_certificate"] for x in rows),
        "bound_violation_rows": sum(x["bound_violation_over_1e_8"] for x in rows),
        "true_model_exclusion_rows": sum(not x["contains_true_model"] for x in rows),
        "certified_rows": sum(x["certified"] for x in rows),
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        "artifact_sha256": {"probe.csv": hashlib.sha256(csv_path.read_bytes()).hexdigest()},
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "limitations": [
            "One constructed topology; no general difficulty-prediction validation.",
            "Repeated seeds and depths are not independent environment evidence.",
            "No certified additional-data forecast and no learner-specific guarantee.",
            "Numerical residual allowances are not formally verified floating-point intervals.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("num_model_boxes", "num_depth_rows",
                     "false_certificate_rows", "bound_violation_rows", "certified_rows")}, indent=2))
    return 0 if not report["false_certificate_rows"] and not report["bound_violation_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
