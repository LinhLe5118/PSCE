"""Measure tabular training runtime without publishing scientific run results.

Three registered seed slices cover every development instance, behavior and
prefix. A separate three-update calibration is a runtime-only deviation; it
never changes protocol_config.json or produces N_emp labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from config.protocol_config import PROTOCOL_CONFIG, PROTOCOL_CONFIG_PATH
from evaluation.io import reserve_output_dir, sha256_file, write_csv_rows, write_json
from scripts.training import run_empirical_requirement as runner


def summarize_runtime(rows: list[dict], *, preflight_seconds: float,
                      mdp_seconds: float) -> dict:
    """Scale complete measured seed slices, not update count, to the full grid."""
    full = [r for r in rows if r["phase"] == "registered_budget"]
    seeds = sorted({r["seed_index"] for r in full})
    if not seeds:
        raise ValueError("No registered-budget measurements.")
    seed_costs = []
    for seed in seeds:
        seed_rows = [r for r in full if r["seed_index"] == seed]
        seed_costs.append(sum(r["fit_seconds"] + r["score_seconds"] +
                              r["load_seconds"] for r in seed_rows))
    scale = len(PROTOCOL_CONFIG["collection"]["training_seed_indices"])
    overhead = preflight_seconds + mdp_seconds
    algorithms = {}
    for algorithm in sorted({r["algorithm"] for r in full}):
        selected = [r for r in full if r["algorithm"] == algorithm]
        algorithms[algorithm] = {
            "measured_jobs": len(selected),
            "fit_seconds": sum(r["fit_seconds"] for r in selected),
            "score_seconds": sum(r["score_seconds"] for r in selected),
            "invalid_policies": sum(r["run_status"] != "observed" for r in selected),
            "converged_jobs": sum(r["converged"] is True for r in selected),
            "iteration_cap_jobs": sum(r["converged"] is False for r in selected),
        }
    return {
        "seed_indices": seeds,
        "seconds_per_complete_seed_slice": dict(zip(map(str, seeds), seed_costs)),
        "full_grid_estimate_seconds": overhead + statistics.mean(seed_costs) * scale,
        "full_grid_seed_sensitivity_seconds": [overhead + min(seed_costs) * scale,
                                                overhead + max(seed_costs) * scale],
        "estimate_method": "One-time preflight and MDP construction plus mean complete seed slice times registered seed count; range uses fastest/slowest slice, not a confidence interval.",
        "excludes": "Dataset generation, Python startup, full-grid artifact export/schema validation, long-run thermal variation and competing system workloads.",
        "algorithms": algorithms,
    }


def benchmark(dataset_dir: Path, output_dir: Path, seeds: tuple[int, ...]) -> dict:
    collection, learning = PROTOCOL_CONFIG["collection"], PROTOCOL_CONFIG["learning"]
    if (not seeds or len(set(seeds)) != len(seeds) or
            not set(seeds) <= set(collection["training_seed_indices"])):
        raise ValueError("Benchmark seeds must be unique registered training seed indices.")
    reserve_output_dir(output_dir)
    started = time.perf_counter()
    # Record selection before any fitting; failed runs retain this planned entry.
    write_json(output_dir / "benchmark_plan.json", {
        "status": "planned_runtime_diagnostic_not_scientific_result",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_indices": list(seeds), "all_development_groups_and_prefixes": True,
        "phases": {"short_calibration": {"iterations": 3, "seeds": [seeds[0]]},
                   "registered_budget": {"iterations": learning["iterations"], "seeds": list(seeds)}},
        "stop_rule": "Complete these phases, no adaptive hyperparameter/seed changes; abort on exception.",
        "dataset_dir": str(dataset_dir.resolve()),
        "protocol_path": str(PROTOCOL_CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_CONFIG_PATH),
        "benchmark_source_sha256": sha256_file(Path(__file__)),
        "provenance": runner._source_provenance(),
    })
    preflight_started = time.perf_counter()
    plan = runner.prepare_training_plan(dataset_dir)
    preflight_seconds = time.perf_counter() - preflight_started
    write_json(output_dir / "training_preflight.json", plan)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    instances = {item["instance_id"]: item for item in manifest["instances"]}
    groups = defaultdict(dict)
    for record in manifest["datasets"]:
        if record["stream"] == "training":
            groups[(record["instance_id"], record["policy_family"])][
                (record["size_multiplier"], runner._training_seed_index(record))] = record
    rows = []
    mdp_seconds = 0.0
    for group_index, ((instance_id, family), records) in enumerate(sorted(groups.items()), 1):
        instance = instances[instance_id]
        build_started = time.perf_counter()
        mdp = runner.make_benchmark_mdp(type("Spec", (), instance)())
        mdp_seconds += time.perf_counter() - build_started
        group_rows = []
        for phase, iterations, phase_seeds in (
            ("short_calibration", 3, seeds[:1]),
            ("registered_budget", int(learning["iterations"]), seeds),
        ):
            cache = {}
            for multiplier in collection["size_multipliers"]:
                for seed in phase_seeds:
                    load_started = time.perf_counter()
                    transitions = runner._load_run(dataset_dir, records[(multiplier, seed)], cache)
                    load_seconds = time.perf_counter() - load_started
                    for algorithm in (*learning["algorithms"], "behavior_cloning_tabular"):
                        kwargs = dict(num_states=mdp.num_states, num_actions=mdp.num_actions)
                        diagnostics = {}
                        if algorithm != "behavior_cloning_tabular":
                            kwargs.update(gamma=mdp.gamma, iterations=iterations,
                                          tolerance=float(learning["tolerance"]), diagnostics=diagnostics)
                        if algorithm == "iql_tabular":
                            kwargs.update(expectile=float(learning["iql_expectile"]),
                                          inverse_temperature=float(learning["iql_inverse_temperature"]))
                        fit_started = time.perf_counter()
                        policy = getattr(runner, algorithm)(transitions, **kwargs)
                        fit_seconds = time.perf_counter() - fit_started
                        score_started = time.perf_counter()
                        _, status = runner._score_policy(mdp, policy,
                            random_return=float(instance["random_return"]),
                            dynamic_range=float(instance["dynamic_range"]))
                        score_seconds = time.perf_counter() - score_started
                        row = dict(phase=phase, instance_id=instance_id,
                            structural_family=instance["family"], policy_family=family,
                            multiplier=multiplier, seed_index=seed, algorithm=algorithm,
                            iteration_budget=iterations if algorithm != "behavior_cloning_tabular" else None,
                            iterations_used=diagnostics.get("iterations_used"),
                            converged=diagnostics.get("converged"),
                            fit_seconds=fit_seconds, score_seconds=score_seconds,
                            # Loading happens once per combo, not once per learner.
                            load_seconds=load_seconds if algorithm == learning["algorithms"][0] else 0.0,
                            run_status=status)
                        rows.append(row)
                        group_rows.append(row)
        with (output_dir / "measurements.jsonl").open("a", encoding="utf-8") as handle:
            for row in group_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"{group_index}/{len(groups)} groups complete; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    summary = summarize_runtime(rows, preflight_seconds=preflight_seconds, mdp_seconds=mdp_seconds)
    summary.update(status="runtime_benchmark_complete_not_full_training",
                   measured_total_jobs=len(rows), preflight_seconds=preflight_seconds,
                   mdp_construction_seconds=mdp_seconds,
                   benchmark_wall_seconds=time.perf_counter() - started,
                   preflight=plan,
                   measurement_note="Sequential CPU runs. Short calibration precedes registered fits per group; filesystem cache can be warm. BC has no iterative epochs. No N_emp/scientific artifacts are emitted.")
    write_csv_rows(output_dir / "runtime_measurements.csv", rows, tuple(rows[0]))
    write_json(output_dir / "runtime_summary.json", summary)
    write_json(output_dir / "artifact_manifest.json", {
        "scope": "runtime diagnostic only",
        "outputs": {name: sha256_file(output_dir / name) for name in (
            "benchmark_plan.json", "training_preflight.json", "measurements.jsonl",
            "runtime_measurements.csv", "runtime_summary.json")}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-indices", default="0,1,2")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seed_indices.split(","))
    summary = benchmark(args.dataset_dir, args.output_dir, seeds)
    print(json.dumps({k: summary[k] for k in (
        "status", "measured_total_jobs", "benchmark_wall_seconds", "full_grid_estimate_seconds",
        "full_grid_seed_sensitivity_seconds")}, indent=2))


if __name__ == "__main__":
    main()
