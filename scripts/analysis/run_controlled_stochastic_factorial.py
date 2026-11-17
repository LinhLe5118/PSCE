#!/usr/bin/env python
"""Run a matched development-only stochasticity x coverage x budget factorial.

The topology and reward of each corridor base are fixed inside a block.  Slip,
behavior coverage, and nested data budget are then varied independently.  This
runner is descriptive mechanism evidence; it never reads the sealed benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from decoy_corridor import CorridorSpec, build_corridor, collect_buffer  # noqa: E402
from empirical_requirement import cfqi_tabular  # noqa: E402
from environments.tabular_environments import TabularMDP  # noqa: E402
from evaluation.io import reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from exact_solver import action_gaps_from_q_values, dynamic_range, normalized_return, value_iteration  # noqa: E402
from oeh_crossfit import estimate_crossfit_oeh  # noqa: E402
from stochastic_features import extract_stochastic_features  # noqa: E402


CHAIN_LENGTHS = (4, 6, 8)
DECOY_FRACTIONS = (0.4, 0.7)
SLIP_LEVELS = (0.0, 0.05, 0.2)
COVERAGE_LEVELS = (0.05, 0.2, 0.45)  # probability of the decoy action
BUDGET_LEVELS = (2, 10, 50)
REPLICATES = tuple(range(5))
SUCCESS_THRESHOLD = 0.95

ROW_FIELDS = (
    "base_id", "chain_length", "decoy_fraction", "slip_probability",
    "behavior_decoy_probability", "budget_multiplier", "replicate", "seed",
    "num_transitions", "dynamic_range", "optimal_return", "random_return",
    "minimum_positive_optimal_gap", "cfqi_normalized_return", "cfqi_success",
    "oeh_status", "oeh_selected_qvi", "oeh_requirement",
    "observed_sa_fraction", "normalized_visitation_entropy",
    "weighted_transition_entropy", "empirical_stochastic_cell_fraction",
    "low_count_cell_fraction", "bootstrap_return_mean_std",
)
CELL_FIELDS = (
    "slip_probability", "behavior_decoy_probability", "budget_multiplier",
    "n", "cfqi_mean_return", "cfqi_success_rate", "oeh_identified_rate",
    "oeh_unsupported_rate", "mean_observed_sa_fraction",
    "mean_transition_entropy", "mean_stochastic_cell_fraction",
)
EFFECT_FIELDS = ("factor", "contrast", "outcome", "matched_pairs", "mean_difference")


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _mdp(spec: CorridorSpec) -> TabularMDP:
    p, r, initial, terminals = build_corridor(spec)
    result = TabularMDP(
        environment_id=spec.label, P=p, R=r,
        initial_state_distribution=initial, terminal_states=terminals,
        gamma=spec.gamma, metadata={"controlled_factorial": True},
    )
    result.validate()
    return result


def _status(profile: dict) -> tuple[str, int | None, float | None]:
    selected = profile["primary_data_selection"]
    if selected["status"] == "selected":
        return "identified", selected["qvi_index"], selected["n_hat_eb"]
    statuses = [row["status"] for row in profile["profile"]]
    for candidate in ("unsupported", "unresolved", "not_epsilon_solvable"):
        if candidate in statuses:
            return candidate, None, None
    raise ValueError("OEH profile has no recognized status")


def _cell_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["slip_probability"], row["behavior_decoy_probability"],
                 row["budget_multiplier"])].append(row)
    output = []
    for (slip, coverage, budget), group in sorted(grouped.items()):
        output.append({
            "slip_probability": slip, "behavior_decoy_probability": coverage,
            "budget_multiplier": budget, "n": len(group),
            "cfqi_mean_return": float(np.mean([r["cfqi_normalized_return"] for r in group])),
            "cfqi_success_rate": float(np.mean([r["cfqi_success"] for r in group])),
            "oeh_identified_rate": float(np.mean([r["oeh_status"] == "identified" for r in group])),
            "oeh_unsupported_rate": float(np.mean([r["oeh_status"] == "unsupported" for r in group])),
            "mean_observed_sa_fraction": float(np.mean([r["observed_sa_fraction"] for r in group])),
            "mean_transition_entropy": float(np.mean([r["weighted_transition_entropy"] for r in group])),
            "mean_stochastic_cell_fraction": float(np.mean([r["empirical_stochastic_cell_fraction"] for r in group])),
        })
    return output


def _main_effects(rows: list[dict]) -> list[dict]:
    outcomes = (
        "cfqi_normalized_return", "cfqi_success", "observed_sa_fraction",
        "weighted_transition_entropy", "empirical_stochastic_cell_fraction",
    )
    definitions = (
        ("slip", "0.20 - 0.00", "slip_probability", 0.0, 0.2,
         ("base_id", "behavior_decoy_probability", "budget_multiplier", "replicate")),
        ("coverage", "0.45 - 0.05", "behavior_decoy_probability", 0.05, 0.45,
         ("base_id", "slip_probability", "budget_multiplier", "replicate")),
        ("budget", "50 - 2", "budget_multiplier", 2, 50,
         ("base_id", "slip_probability", "behavior_decoy_probability", "replicate")),
    )
    result = []
    for factor, contrast, field, low, high, keys in definitions:
        indexed = {tuple(row[k] for k in keys) + (row[field],): row for row in rows}
        pairs = []
        for key in sorted({tuple(row[k] for k in keys) for row in rows}):
            if key + (low,) in indexed and key + (high,) in indexed:
                pairs.append((indexed[key + (low,)], indexed[key + (high,)]))
        for outcome in outcomes:
            differences = [float(high_row[outcome]) - float(low_row[outcome])
                           for low_row, high_row in pairs]
            result.append({
                "factor": factor, "contrast": contrast, "outcome": outcome,
                "matched_pairs": len(differences),
                "mean_difference": float(np.mean(differences)),
            })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=len(REPLICATES))
    parser.add_argument("--k-max", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.replicates <= len(REPLICATES):
        raise ValueError(f"replicates must be in [1,{len(REPLICATES)}]")
    reserve_output_dir(args.output_dir)
    started = time.perf_counter()
    plan = {
        "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "sealed_evaluation_accessed": False,
        "matched_block": "chain_length x decoy_fraction; topology and reward fixed within block",
        "slip_levels": SLIP_LEVELS, "coverage_levels": COVERAGE_LEVELS,
        "budget_levels": BUDGET_LEVELS, "replicates": args.replicates,
        "nested_budget_prefixes": True, "k_max": args.k_max,
        "success_threshold": SUCCESS_THRESHOLD,
        "source_sha256": sha256_file(Path(__file__)),
        "generator_sha256": sha256_file(ROOT / "code/decoy_corridor.py"),
        "oeh_sha256": sha256_file(ROOT / "code/oeh_crossfit.py"),
    }
    write_json(args.output_dir / "plan.json", plan)
    rows = []
    total_streams = len(CHAIN_LENGTHS) * len(DECOY_FRACTIONS) * len(SLIP_LEVELS) * len(COVERAGE_LEVELS) * args.replicates
    position = 0
    for length in CHAIN_LENGTHS:
        for fraction in DECOY_FRACTIONS:
            base_id = f"corridor_L{length}_f{fraction:g}"
            for slip in SLIP_LEVELS:
                spec = CorridorSpec(length, fraction, slip)
                mdp = _mdp(spec)
                solution = value_iteration(mdp)
                range_value, optimal_return, random_return = dynamic_range(mdp, solution=solution)
                gaps = action_gaps_from_q_values(solution.q_values)
                positive_gaps = gaps[gaps > 1e-12]
                minimum_gap = float(np.min(positive_gaps)) if len(positive_gaps) else 0.0
                for coverage in COVERAGE_LEVELS:
                    for replicate in REPLICATES[:args.replicates]:
                        seed = _seed("factorial-v1", base_id, slip, coverage, replicate)
                        maximum = max(BUDGET_LEVELS) * mdp.num_states * mdp.num_actions
                        buffer = collect_buffer(
                            spec, num_transitions=maximum, epsilon=coverage,
                            gamma=mdp.gamma, seed=seed, horizon=100,
                        )
                        for budget in BUDGET_LEVELS:
                            size = budget * mdp.num_states * mdp.num_actions
                            arrays = {name: np.asarray(value)[:size] for name, value in buffer.items()
                                      if isinstance(value, np.ndarray) and np.asarray(value).ndim == 1
                                      and len(np.asarray(value)) == maximum}
                            features = extract_stochastic_features(
                                arrays, num_states=mdp.num_states,
                                num_actions=mdp.num_actions, gamma=mdp.gamma,
                                bootstrap_replicates=200,
                                seed_key=f"{base_id}|{slip}|{coverage}|{replicate}|{budget}",
                            )
                            transitions = transitions_from_arrays(arrays)
                            policy = cfqi_tabular(
                                transitions, num_states=mdp.num_states,
                                num_actions=mdp.num_actions, gamma=mdp.gamma,
                            )
                            score = float(normalized_return(mdp, policy, solution=solution))
                            profile = estimate_crossfit_oeh(
                                arrays, mdp.initial_state_distribution,
                                num_states=mdp.num_states, num_actions=mdp.num_actions,
                                gamma=mdp.gamma, k_max=args.k_max, reward_bound=1.0,
                                epsilon=0.05, delta=0.05, seed=seed,
                            )
                            status, selected_qvi, requirement = _status(profile)
                            rows.append({
                                "base_id": base_id, "chain_length": length,
                                "decoy_fraction": fraction, "slip_probability": slip,
                                "behavior_decoy_probability": coverage,
                                "budget_multiplier": budget, "replicate": replicate,
                                "seed": seed, "num_transitions": size,
                                "dynamic_range": range_value, "optimal_return": optimal_return,
                                "random_return": random_return,
                                "minimum_positive_optimal_gap": minimum_gap,
                                "cfqi_normalized_return": score,
                                "cfqi_success": score >= SUCCESS_THRESHOLD,
                                "oeh_status": status, "oeh_selected_qvi": selected_qvi,
                                "oeh_requirement": requirement,
                                **{name: features[name] for name in (
                                    "observed_sa_fraction", "normalized_visitation_entropy",
                                    "weighted_transition_entropy",
                                    "empirical_stochastic_cell_fraction",
                                    "low_count_cell_fraction", "bootstrap_return_mean_std",
                                )},
                            })
                        position += 1
                        if position % 15 == 0 or position == total_streams:
                            print(f"streams {position}/{total_streams}", flush=True)
    cells = _cell_summary(rows)
    effects = _main_effects(rows)
    write_csv_rows(args.output_dir / "factorial_rows.csv", rows, ROW_FIELDS)
    write_csv_rows(args.output_dir / "cell_summary.csv", cells, CELL_FIELDS)
    write_csv_rows(args.output_dir / "matched_main_effects.csv", effects, EFFECT_FIELDS)
    counts = {status: sum(row["oeh_status"] == status for row in rows)
              for status in ("identified", "unsupported", "unresolved", "not_epsilon_solvable")}
    report = {
        **plan, "status": "complete", "rows": len(rows),
        "streams": total_streams, "oeh_status_counts": counts,
        "mean_cfqi_return": float(np.mean([row["cfqi_normalized_return"] for row in rows])),
        "cfqi_success_rate": float(np.mean([row["cfqi_success"] for row in rows])),
        "main_effects": effects, "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "report.json", report)
    lines = [
        "# Controlled stochastic factorial", "",
        f"{len(rows)} cells from six matched topology/reward blocks; evaluation split accessed: no.", "",
        f"OEH status counts: `{counts}`.",
        f"Mean CFQI normalized return: `{report['mean_cfqi_return']:.4f}`; success: `{report['cfqi_success_rate']:.4f}`.", "",
        "| Factor | Contrast | Outcome | Matched pairs | Mean difference |",
        "|---|---|---|---:|---:|",
    ]
    for row in effects:
        lines.append(f"| {row['factor']} | {row['contrast']} | {row['outcome']} | {row['matched_pairs']} | {row['mean_difference']:.6f} |")
    lines += ["", "These are descriptive matched development effects; six base blocks are too few for a general causal claim.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
