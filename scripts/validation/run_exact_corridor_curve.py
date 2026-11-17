#!/usr/bin/env python
"""Week 4 B1: exact evidence curves on the 12 registered decoy corridors.

The true finite-depth gaps come from the known MDP.  Dispersion comes from the
registered 200 independently sampled datasets, with one fixed support mask
shared by the true curve and every sampled empirical model.  This is an oracle
scoring probe, never an input to the data-only estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from decoy_corridor import (  # noqa: E402
    ADVANCE,
    DECOY,
    REGISTERED_GRID,
    CorridorSpec,
    build_corridor,
    collect_buffer,
)
from evaluation.io import reserve_output_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from exact_evidence_curve import ExactEvidenceCurve, exact_evidence_curve  # noqa: E402
from oeh_estimator import OEHConfig  # noqa: E402
from positive_control import assert_registered  # noqa: E402
from scripts.analysis.run_depth_probe import BEHAVIOR_EPSILON, BUFFER_SEED  # noqa: E402
from environments.tabular_environments import TabularMDP  # noqa: E402


ORACLE_SEED = 20260907

CURVE_FIELDS = (
    "instance",
    "chain_length",
    "decoy_fraction",
    "slip_probability",
    "depth",
    "mean_delta",
    "max_delta",
    "mean_sigma",
    "max_sigma",
    "paired_states",
    "m_k",
    "k_dagger_exact",
    "valid_sample_fraction",
)

STATE_FIELDS = (
    "instance",
    "depth",
    "state",
    "is_terminal",
    "action_mask_advance",
    "action_mask_decoy",
    "delta_k",
    "sigma_k",
    "m_k",
    "k_dagger_exact",
)

SUMMARY_FIELDS = (
    "instance",
    "chain_length",
    "decoy_fraction",
    "slip_probability",
    "sample_size",
    "num_exact_replicates",
    "valid_sample_fraction",
    "curve_status",
    "paired_states",
    "k_dagger_exact",
    "exact_interior_argmin",
    "k_dagger_registered_data_only",
    "k_settle_exact",
    "exact_minus_data_only",
    "exact_minus_settle",
)


def corridor_mdp(spec: CorridorSpec) -> TabularMDP:
    p, rewards, initial, terminals = build_corridor(spec)
    mdp = TabularMDP(spec.label, p, rewards, initial, terminals, spec.gamma, {
        "probe": "decoy_corridor",
        "chain_length": spec.chain_length,
        "decoy_fraction": spec.decoy_fraction,
        "slip_probability": spec.slip_probability,
    })
    mdp.validate()
    return mdp


def fixed_beta_and_mask(
    spec: CorridorSpec, mdp: TabularMDP, config: OEHConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Collection beta plus the data-only H1 mask, frozen before the oracle.

    This mirrors ``run_oeh_validation``: support is selected once from the
    existing data-only pilot, then the exact curve and all independent samples
    share it.  Nothing computed by the oracle can change the mask.
    """

    beta = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.float64)
    beta[:, ADVANCE] = 1.0 - BEHAVIOR_EPSILON
    beta[:, DECOY] = BEHAVIOR_EPSILON
    pilot_size = config.n0_coefficient * mdp.num_states * mdp.num_actions
    buffer = collect_buffer(
        spec,
        num_transitions=pilot_size,
        epsilon=BEHAVIOR_EPSILON,
        gamma=spec.gamma,
        seed=BUFFER_SEED + 17 * spec.chain_length + config.n0_coefficient,
        horizon=config.horizon,
    )
    counts = np.zeros((mdp.num_states, mdp.num_actions), dtype=np.int64)
    np.add.at(counts, (buffer["state"], buffer["action"]), 1)
    trajectory_counts = np.zeros_like(counts)
    observed = set(
        zip(
            np.asarray(buffer["trajectory_id"], dtype=np.int64),
            np.asarray(buffer["state"], dtype=np.int64),
            np.asarray(buffer["action"], dtype=np.int64),
        )
    )
    for _, state, action in observed:
        trajectory_counts[state, action] += 1
    mask = (counts >= config.n_min) & (
        trajectory_counts >= config.min_trajectory_support
    )
    mask[np.asarray(mdp.terminal_states, dtype=np.int64)] = False
    return beta, mask


def instance_seed(label: str, root_seed: int = ORACLE_SEED) -> int:
    digest = hashlib.sha256(f"{root_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _finite_summary(values: np.ndarray) -> tuple[float | str, float | str]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "", ""
    return float(np.mean(finite)), float(np.max(finite))


def reported_exact_depth(curve: ExactEvidenceCurve) -> int | None:
    """Return the exact argmin only when the curve has an action gap.

    ``exact_evidence_curve`` uses an internal all-one multiplier vector for an
    empty action-pair set.  Its numerical tie-break is zero, but that value is
    not an estimand and must never cross the artifact boundary as a result.
    """

    if not np.isfinite(curve.deltas).any():
        return None
    return int(curve.k_dagger)


def curve_rows(
    spec: CorridorSpec,
    mdp: TabularMDP,
    mask: np.ndarray,
    curve: ExactEvidenceCurve,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    depths: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    terminal_set = set(mdp.terminal_states)
    exact_depth = reported_exact_depth(curve)
    scored = exact_depth is not None
    reported_k_dagger: int | str = exact_depth if exact_depth is not None else ""
    for depth, multiplier in enumerate(curve.multipliers):
        mean_delta, max_delta = _finite_summary(curve.deltas[depth])
        mean_sigma, max_sigma = _finite_summary(curve.sigmas[depth])
        paired_states = int(np.isfinite(curve.deltas[depth]).sum())
        depths.append({
            "instance": spec.label,
            "chain_length": spec.chain_length,
            "decoy_fraction": spec.decoy_fraction,
            "slip_probability": spec.slip_probability,
            "depth": depth,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "mean_sigma": mean_sigma,
            "max_sigma": max_sigma,
            "paired_states": paired_states,
            "m_k": float(multiplier) if scored else "",
            "k_dagger_exact": reported_k_dagger,
            "valid_sample_fraction": float(curve.valid_sample_fraction),
        })
        for state in range(mdp.num_states):
            delta = float(curve.deltas[depth, state])
            sigma = float(curve.sigmas[depth, state])
            states.append({
                "instance": spec.label,
                "depth": depth,
                "state": state,
                "is_terminal": state in terminal_set,
                "action_mask_advance": bool(mask[state, ADVANCE]),
                "action_mask_decoy": bool(mask[state, DECOY]),
                "delta_k": "" if not np.isfinite(delta) else delta,
                "sigma_k": "" if not np.isfinite(sigma) else sigma,
                "m_k": float(multiplier) if scored else "",
                "k_dagger_exact": reported_k_dagger,
            })
    return depths, states


def h1_reference(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | None]]:
    """Select the registered 5|S||A| row without using oracle values as inputs."""

    result: dict[str, dict[str, int | None]] = {}
    for row in rows:
        if str(row.get("variant")) != "registered" or int(row.get("pilot_multiplier", -1)) != 5:
            continue
        label = str(row["mdp_id"])
        suffix = f"_x{int(row['pilot_multiplier'])}"
        if label.endswith(suffix):
            label = label[: -len(suffix)]
        data_depth = None if str(row.get("k_dagger", "")).strip() == "" else int(row["k_dagger"])
        settle = None if str(row.get("k_settle_exact", "")).strip() == "" else int(row["k_settle_exact"])
        candidate = {"data_only": data_depth, "settle": settle}
        if label in result and result[label] != candidate:
            raise ValueError(f"Conflicting H1 reference rows for {label}.")
        result[label] = candidate
    return result


def run(
    output_dir: Path,
    *,
    h1_path: Path,
    config: OEHConfig | None = None,
    specs: Sequence[CorridorSpec] = REGISTERED_GRID,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    cfg = config or OEHConfig.load()
    if config is None:
        assert_registered(cfg)
    references = h1_reference(read_csv_rows(h1_path))
    expected_labels = {spec.label for spec in specs}
    missing_references = sorted(expected_labels - set(references))
    if missing_references:
        raise ValueError(
            f"H1 reference is missing registered corridors: {missing_references}"
        )
    depth_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    started = time.time()
    for index, spec in enumerate(specs, 1):
        mdp = corridor_mdp(spec)
        beta, mask = fixed_beta_and_mask(spec, mdp, cfg)
        curve = exact_evidence_curve(
            mdp,
            beta,
            action_mask=mask,
            config=cfg,
            rng=np.random.default_rng(instance_seed(spec.label)),
        )
        depths, states = curve_rows(spec, mdp, mask, curve)
        depth_rows.extend(depths)
        state_rows.extend(states)
        reference = references[spec.label]
        data_only = reference["data_only"]
        settle = reference["settle"]
        paired_states = int(np.isfinite(curve.deltas[0]).sum())
        reported_depth = reported_exact_depth(curve)
        is_scored = reported_depth is not None
        curve_status = "scored" if is_scored else "undefined_no_two_action_state"
        exact_depth: int | str = reported_depth if reported_depth is not None else ""
        summaries.append({
            "instance": spec.label,
            "chain_length": spec.chain_length,
            "decoy_fraction": spec.decoy_fraction,
            "slip_probability": spec.slip_probability,
            "sample_size": cfg.n0_coefficient * mdp.num_states * mdp.num_actions,
            "num_exact_replicates": cfg.num_exact_replicates,
            "valid_sample_fraction": curve.valid_sample_fraction,
            "curve_status": curve_status,
            "paired_states": paired_states,
            "k_dagger_exact": exact_depth,
            "exact_interior_argmin": (
                0 < int(exact_depth) < cfg.k_max if is_scored else ""
            ),
            "k_dagger_registered_data_only": "" if data_only is None else data_only,
            "k_settle_exact": "" if settle is None else settle,
            "exact_minus_data_only": (
                int(exact_depth) - data_only
                if is_scored and data_only is not None
                else ""
            ),
            "exact_minus_settle": (
                int(exact_depth) - settle
                if is_scored and settle is not None
                else ""
            ),
        })
        shown_depth = exact_depth if is_scored else "undefined"
        print(f"[{index}/{len(specs)}] {spec.label}: k_exact={shown_depth}, valid={curve.valid_sample_fraction:.3f} ({time.time()-started:.0f}s)", flush=True)

    scored = [row for row in summaries if row["curve_status"] == "scored"]
    exact_depths = [int(row["k_dagger_exact"]) for row in scored]
    valid = [float(row["valid_sample_fraction"]) for row in summaries]
    data_pairs = [row for row in summaries if row["k_dagger_registered_data_only"] != ""]
    undefined = len(summaries) - len(scored)
    if undefined and all(value == 0 for value in exact_depths):
        outcome = "exact_flat_at_zero_with_support_undefined_cells"
    elif all(value == 0 for value in exact_depths):
        outcome = "exact_flat_at_zero"
    elif any(0 < value < cfg.k_max for value in exact_depths):
        outcome = "exact_has_interior_minima"
    else:
        outcome = "exact_boundary_only"
    report = {
        "scope": "12 registered decoy-corridor probes; not benchmark data",
        "registry_date": "2026-09-07",
        "root_seed": ORACLE_SEED,
        "seed_derivation": "little-endian first 64 bits of SHA256(f'{root_seed}:{instance_label}')",
        "behavior_policy": {"advance": 1.0 - BEHAVIOR_EPSILON, "decoy": BEHAVIOR_EPSILON},
        "action_mask": (
            "frozen from the existing H1 5|S||A| pilot using n_min and "
            "min_trajectory_support; shared by the true curve and all replicas"
        ),
        "config": {
            "n0_coefficient": cfg.n0_coefficient,
            "k_max": cfg.k_max,
            "epsilon": cfg.epsilon,
            "num_exact_replicates": cfg.num_exact_replicates,
            "horizon": cfg.horizon,
            "estimator_min_valid_bootstrap_fraction_not_applied": cfg.min_valid_bootstrap_fraction,
        },
        "instances": len(summaries),
        "instances_scored": len(scored),
        "undefined_no_two_action_state": undefined,
        "outcome": outcome,
        "k_dagger_exact_counts": dict(sorted((str(k), exact_depths.count(k)) for k in set(exact_depths))),
        "undefined_k_dagger_exact": undefined,
        "exact_interior_instances": sum(0 < value < cfg.k_max for value in exact_depths),
        "depth_necessary_scored_instances": sum(
            row["k_settle_exact"] != "" and int(row["k_settle_exact"]) > 0
            for row in scored
        ),
        "depth_necessary_scored_with_exact_zero": sum(
            row["k_settle_exact"] != ""
            and int(row["k_settle_exact"]) > 0
            and int(row["k_dagger_exact"]) == 0
            for row in scored
        ),
        "registered_data_only_pairs": len(data_pairs),
        "registered_data_only_zero": sum(int(row["k_dagger_registered_data_only"]) == 0 for row in data_pairs),
        "minimum_valid_sample_fraction": min(valid) if valid else None,
        "interpretation": (
            "Among curves with at least one supported action pair, if exact curves are all tied at k=0, "
            "the registered criterion itself lacks a depth axis on this probe. Empty-pair curves are undefined, not zero. "
            "If exact curves have interior minima while registered data-only rows stay at zero, estimation error and definition failure separate."
        ),
        "h1_source_sha256": sha256_file(h1_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "exact_corridor_curve.csv", depth_rows, CURVE_FIELDS)
    write_csv_rows(output_dir / "exact_corridor_curve_states.csv", state_rows, STATE_FIELDS)
    write_csv_rows(output_dir / "exact_corridor_summary.csv", summaries, SUMMARY_FIELDS)
    report["artifact_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in ("exact_corridor_curve.csv", "exact_corridor_curve_states.csv", "exact_corridor_summary.csv")
    }
    write_json(output_dir / "exact_corridor_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--h1-path",
        type=Path,
        default=Path("artifacts/week2/corridor_definitions_2026-09-06/corridor_definitions.csv"),
    )
    args = parser.parse_args()
    reserve_output_dir(args.output_dir)
    report = run(args.output_dir, h1_path=args.h1_path)
    print(json.dumps({key: report[key] for key in ("outcome", "k_dagger_exact_counts", "minimum_valid_sample_fraction")}, indent=2))
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
