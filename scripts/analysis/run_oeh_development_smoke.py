"""Compare OEH-v1 and the experimental regret certificate on development pilots.

Reuses the repository data-only index, raw NPZ prefixes and registered OEH-v1
configuration. Runs every declared prefix; no adaptive early stop. Oracle
metadata is used only for split gating, final labels and post-prediction scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
from adaptive_pilot import transitions_from_arrays  # noqa: E402
from oeh_estimator import OEHConfig, estimate_oeh  # noqa: E402
from oeh_certificate_adapter import estimate_oeh_certificate  # noqa: E402
from evaluation.io import reserve_output_dir, sha256_file, verified_dataset_path, write_csv_rows, write_json  # noqa: E402


def development_inputs(root: Path, limit_mdps=None):
    manifest_path = root / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("run_mode") != "review" or any(r.get("benchmark_split") != "development" for r in m["datasets"]):
        raise ValueError("Development-only review bundle required; evaluation records refused.")
    if any(r.get("split") != "development" for r in m["instances"]):
        raise ValueError("Non-development instance refused.")
    ip = (root / m["estimator_input_index"]).resolve()
    if not ip.is_relative_to(root.resolve()) or sha256_file(ip) != m["estimator_input_index_sha256"]:
        raise ValueError("Data-only index path/checksum mismatch.")
    index = json.loads(ip.read_text(encoding="utf-8"))
    labels = {r["data_only_dataset_id"]: r for r in m["datasets"]}
    instances = m["instances"] if limit_mdps is None else m["instances"][:limit_mdps]
    ids = {r["instance_id"] for r in instances}
    records = []
    for r in index["datasets"]:
        if r["stream"] != "pilot" or r["budget_role"] != "adaptive_pilot_candidate":
            continue
        label = labels[r["dataset_id"]]
        if label["instance_id"] not in ids:
            continue
        for name in ("file", "sha256", "size", "slice", "num_states", "num_actions", "initial_state_distribution"):
            if r[name] != label[name]:
                raise ValueError(f"Index/manifest mismatch: {name}")
        if r["slice"] != [0, r["size"]]:
            raise ValueError("Only natural pilot prefixes are supported.")
        records.append(r)
    if not records:
        raise ValueError("No development pilots.")
    return m, records, labels, ip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-mdps", type=int)
    parser.add_argument("--v1-backend", choices=("reference", "numpy"), default="numpy")
    args = parser.parse_args(argv)
    if args.limit_mdps is not None and args.limit_mdps < 1:
        parser.error("--limit-mdps must be positive")
    manifest, records, labels, index_path = development_inputs(args.dataset_dir, args.limit_mdps)
    os.environ["DAORL_OEH_BACKEND"] = args.v1_backend
    config = OEHConfig.load()
    benchmark = manifest["effective_protocol"]["benchmark"]
    if benchmark["gamma"] != config.gamma or benchmark["reward_range"] != [0., 1.]:
        raise ValueError("This smoke protocol requires gamma matching v1 and public rewards [0,1].")
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    sources = ["code/oeh_estimator.py", "code/oeh_certificate.py", "code/oeh_certificate_adapter.py",
               "code/adaptive_pilot.py", "code/exact_solver.py", "code/tabular_acceleration.py", "evaluation/io.py",
               "environments/benchmark_environments.py", "protocol_config.json",
               "scripts/analysis/run_oeh_development_smoke.py"]
    provenance = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": sha256_file(args.dataset_dir / "manifest.json"),
        "index_sha256": sha256_file(index_path),
        "source_sha256": {p: sha256_file(ROOT / p) for p in sources},
        "v1_config": asdict(config),
        "v1_backend": args.v1_backend,
        "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "v2_config": {"epsilon": .05, "delta": .05, "method": "bernstein", "reward_bound": 1,
                      "gamma": config.gamma, "k_max": config.k_max},
        "bootstrap_seed": 20260817,
        "prefix_policy": "all 2/5/10 SA pilot prefixes; no adaptive early stopping",
        "normalization_caveat": "v2 epsilon uses R_max/(1-gamma); not v1 proxy or benchmark NR",
        "input_records": [{k: r[k] for k in ("dataset_id", "group_id", "file", "sha256", "slice")} for r in records],
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
    }
    write_json(args.output_dir / "plan.json", provenance)
    rows, curves, errors = [], [], []
    oracle_cache = {}
    for i, rec in enumerate(records, 1):
        label = labels[rec["dataset_id"]]
        identity = {"dataset_id": rec["dataset_id"], "group_id": rec["group_id"],
                    "mdp_id": label["instance_id"], "policy_family": label["policy_family"],
                    "pilot_multiplier": rec["size_multiplier"], "num_transitions": rec["size"]}
        try:
            with np.load(verified_dataset_path(args.dataset_dir, rec), allow_pickle=False) as z:
                # Slice BEFORE statistics/terminal detection/behavior estimation.
                arrays = {k: z[k][:rec["size"]] for k in z.files}
                if any(len(a) != rec["size"] for a in arrays.values()):
                    raise ValueError("Physical buffer is shorter than declared pilot.")
            dims = {"num_states": rec["num_states"], "num_actions": rec["num_actions"]}
            t = time.perf_counter()
            new, policies = estimate_oeh_certificate(arrays, rec["initial_state_distribution"],
                gamma=config.gamma, k_max=config.k_max, epsilon=.05, delta=.05, **dims)
            new_seconds = time.perf_counter() - t
            seed = np.random.SeedSequence([20260817, rec["size"], rec["num_states"], rec["num_actions"]])
            t = time.perf_counter()
            old = estimate_oeh(transitions_from_arrays(arrays), rec["initial_state_distribution"],
                               config=config, rng=np.random.default_rng(seed), **dims)
            old_seconds = time.perf_counter() - t
            u = np.asarray(new["normalized_regret_upper"])
            row = {**identity,
                "v1_status": "abstained" if old.abstained else ("censored" if old.censored_right else "observed"),
                "v1_reason": old.reason, "v1_depth": old.k_dagger,
                "v1_all_floor": None if old.multipliers is None else bool(np.all(np.abs(old.multipliers - 1) <= 1e-9)),
                "v1_min_multiplier": None if old.multipliers is None else float(np.min(old.multipliers)),
                "v1_max_multiplier": None if old.multipliers is None else float(np.max(old.multipliers)),
                "v2_status": new["status"], "v2_certified_depth": new["certified_depth"],
                "v2_min_upper": float(u.min()), "v2_max_upper": float(u.max()),
                "v2_depth_range": float(np.ptp(u)), "v2_flat": bool(np.ptp(u) <= 1e-9),
                "v2_all_ceiling": bool(np.all(u >= 1-1e-9)),
                "v2_unique_policies": new["unique_policy_count"],
                **{k: v for k, v in new["diagnostics"].items() if k != "num_transitions"},
                "v1_seconds": old_seconds, "v2_seconds": new_seconds,
                "oracle_status": "pending", "oracle_min_regret": None, "oracle_max_regret": None,
                "bound_violation_count": None, "false_certificate_count": None,
            }
            # Oracle import/access only after both estimators have produced results.
            from environments.benchmark_environments import make_benchmark_mdp
            from exact_solver import evaluate_policy, value_iteration
            try:
                key = label["instance_id"]
                if key not in oracle_cache:
                    spec = next(s for s in manifest["instances"] if s["instance_id"] == key)
                    mdp = make_benchmark_mdp(type("Spec", (), spec)())
                    opt = value_iteration(mdp)
                    best = float(mdp.initial_state_distribution @ opt.values)
                    if (mdp.num_states != rec["num_states"] or mdp.num_actions != rec["num_actions"]
                            or not np.isclose(best, spec["optimal_return"], atol=1e-8, rtol=1e-8)):
                        raise ValueError("Reconstructed oracle disagrees with recorded model.")
                    oracle_cache[key] = mdp, best
                mdp, best = oracle_cache[key]
                cache = {}
                regrets = []
                for policy in policies:
                    key = policy.tobytes()
                    if key not in cache:
                        cache[key] = (1-config.gamma) * (best - evaluate_policy(mdp, policy).expected_return)
                    regrets.append(cache[key])
                regret = np.array(regrets)
                row.update(oracle_status="scored", oracle_min_regret=float(regret.min()),
                    oracle_max_regret=float(regret.max()),
                    bound_violation_count=int(np.sum(regret > u + 1e-8)),
                    false_certificate_count=int(np.sum((u <= .05) & (regret > .05+1e-8))))
            except Exception as exc:
                regrets = [None]*len(u)
                row["oracle_status"] = "error"
                errors.append({**identity, "stage": "oracle", "error": repr(exc)})
            rows.append(row)
            for k, bound in enumerate(u):
                curves.append({**identity, "depth": k, "v2_regret_upper": float(bound),
                    "oracle_regret": regrets[k], "v2_certified": bool(bound <= .05),
                    "v1_multiplier": None if old.multipliers is None else float(old.multipliers[k]),
                    "v1_loss": None if old.losses_at_one is None else float(old.losses_at_one[k])})
            print(f"[{i}/{len(records)}] {identity['mdp_id']} {identity['policy_family']} x{rec['size_multiplier']}: "
                  f"v1={row['v1_status']}, v2 U=[{u.min():.6f},{u.max():.6f}], "
                  f"policies={new['unique_policy_count']}, {new_seconds+old_seconds:.2f}s", flush=True)
        except Exception as exc:
            errors.append({**identity, "stage": "estimator_or_input", "error": repr(exc)})
            print(f"[{i}/{len(records)}] ERROR {exc!r}", flush=True)
        write_json(args.output_dir / "progress.json", {"processed": i, "total": len(records), "errors": errors})
    if rows:
        write_csv_rows(args.output_dir / "summary.csv", rows, list(rows[0]))
        write_csv_rows(args.output_dir / "curves.csv", curves, list(curves[0]))
    report = {**provenance, "status": "complete" if not errors else "completed_with_errors",
        "num_requested": len(records), "num_completed": len(rows),
        "mdp_count": len({r['mdp_id'] for r in rows}),
        "v1_status_counts": dict(Counter(r["v1_status"] for r in rows)),
        "v1_assessable_curves": sum(r["v1_all_floor"] is not None for r in rows),
        "v1_all_floor_curves": sum(r["v1_all_floor"] is True for r in rows),
        "v2_certified_curves": sum(r["v2_certified_depth"] is not None for r in rows),
        "v2_flat_curves": sum(r["v2_flat"] for r in rows),
        "v2_all_ceiling_curves": sum(r["v2_all_ceiling"] for r in rows),
        "v2_single_policy_curves": sum(r["v2_unique_policies"] == 1 for r in rows),
        "v2_min_upper_overall": min((r["v2_min_upper"] for r in rows), default=None),
        "oracle_scored_curves": sum(r["oracle_status"] == "scored" for r in rows),
        "bound_violation_count": sum(r["bound_violation_count"] or 0 for r in rows),
        "false_certificate_count": sum(r["false_certificate_count"] or 0 for r in rows),
        "total_seconds": time.perf_counter()-started, "errors": errors,
        "artifact_sha256": {p.name: sha256_file(p) for p in args.output_dir.glob('*.csv')},
        "limitations": ["Development smoke only; thresholds not tuned from these outputs.",
            "Different mathematical estimands; equal numeric epsilon does not equal target quality.",
            "All prefixes rerun including those original adaptive selection might not visit.",
            "No sample-requirement forecast; constant upper bounds do not establish predictive utility."]}
    write_json(args.output_dir / "report.json", report)
    return 1 if errors or report["bound_violation_count"] or report["false_certificate_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
