#!/usr/bin/env python
"""Reproduce the paper's OEH mechanism figure from the registered corridor probe.

The representative cell is selected post hoc by a declared descriptive rule:
the largest finite weighted-gap swing among scored rows in the frozen probe
artifact. The script reruns that cell through the registered estimator, checks
the recomputed summaries against the artifact, and exports both plot and data.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "code"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from decoy_corridor import REGISTERED_GRID, CorridorSpec, build_corridor, collect_buffer  # noqa: E402
from evaluation.io import (  # noqa: E402
    read_csv_rows,
    reserve_output_dir,
    sha256_file,
    write_csv_rows,
    write_json,
)
from oeh_estimator import OEHConfig, OEHResult, estimate_oeh  # noqa: E402
from oeh_validation import prefix_bootstrap_rng  # noqa: E402
from positive_control import assert_registered  # noqa: E402
from scripts.analysis.run_depth_probe import (  # noqa: E402
    BEHAVIOR_EPSILON,
    BUFFER_SEED,
)

G_REGISTERED = 0.169971207479904
DEFAULT_SOURCE = ROOT / "artifacts" / "week2" / "depth_probe_2026-09-06" / "depth_probe.csv"
CURVE_FIELDS = (
    "depth",
    "weighted_gap",
    "weighted_sigma",
    "loss_at_one",
    "theorem_ceiling",
    "epsilon",
    "multiplier",
)


def select_representative(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the scored row with the largest finite weighted-gap swing."""

    eligible: list[tuple[float, str, int, Mapping[str, Any]]] = []
    for row in rows:
        if str(row.get("oeh_status", "")) == "abstained":
            continue
        try:
            swing = float(row["weighted_gap_swing"])
            multiplier = int(row["pilot_multiplier"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(swing):
            eligible.append((swing, str(row.get("instance", "")), multiplier, row))
    if not eligible:
        raise ValueError("No scored row has a finite weighted-gap swing.")
    # The secondary keys make an otherwise unlikely tie deterministic.
    return max(eligible, key=lambda item: (item[0], item[1], item[2]))[3]


def weighted_curve_rows(result: OEHResult, epsilon: float) -> list[dict[str, float | int]]:
    """Reduce the estimator's state-level arrays to the plotted depth curves."""

    if result.abstained:
        raise ValueError(f"Representative estimator run abstained: {result.reason}")
    required = (
        result.occupancy_weights,
        result.deltas,
        result.sigmas,
        result.losses_at_one,
        result.multipliers,
    )
    if any(value is None for value in required):
        raise ValueError("Observed OEH result is missing diagnostic arrays.")

    weights = np.asarray(result.occupancy_weights, dtype=np.float64)
    deltas = np.nan_to_num(np.asarray(result.deltas, dtype=np.float64), nan=0.0)
    sigmas = np.nan_to_num(np.asarray(result.sigmas, dtype=np.float64), nan=0.0)
    losses = np.asarray(result.losses_at_one, dtype=np.float64)
    multipliers = np.asarray(result.multipliers, dtype=np.float64)
    weighted_gap = np.sum(weights * deltas, axis=1)
    weighted_sigma = np.sum(weights * sigmas, axis=1)

    return [
        {
            "depth": depth,
            "weighted_gap": float(weighted_gap[depth]),
            "weighted_sigma": float(weighted_sigma[depth]),
            "loss_at_one": float(losses[depth]),
            "theorem_ceiling": float(G_REGISTERED * weighted_sigma[depth]),
            "epsilon": float(epsilon),
            "multiplier": float(multipliers[depth]),
        }
        for depth in range(len(losses))
    ]


def summarize_curve(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    gaps = np.asarray([float(row["weighted_gap"]) for row in rows])
    sigmas = np.asarray([float(row["weighted_sigma"]) for row in rows])
    losses = np.asarray([float(row["loss_at_one"]) for row in rows])
    peak = int(np.argmax(gaps))
    smallest = float(np.min(gaps))
    return {
        "weighted_gap_at_zero": float(gaps[0]),
        "max_weighted_gap": float(gaps[peak]),
        "argmax_weighted_gap": peak,
        "weighted_gap_swing": float(gaps[peak] / smallest) if smallest > 0 else float("inf"),
        "loss_at_gap_peak": float(losses[peak]),
        "max_loss_at_one": float(np.max(losses)),
        "max_weighted_sigma": float(np.max(sigmas)),
        "max_theorem_ceiling": float(G_REGISTERED * np.max(sigmas)),
    }


def _assert_matches_source(summary: Mapping[str, float | int], source: Mapping[str, Any]) -> None:
    fields = (
        "weighted_gap_at_zero",
        "max_weighted_gap",
        "weighted_gap_swing",
        "loss_at_gap_peak",
        "max_loss_at_one",
    )
    for field in fields:
        if not np.isclose(float(summary[field]), float(source[field]), rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Recomputed {field}={summary[field]!r} does not match source {source[field]!r}."
            )
    if int(summary["argmax_weighted_gap"]) != int(source["argmax_weighted_gap"]):
        raise ValueError("Recomputed gap-peak depth does not match the frozen artifact.")
    # The historical column name says mean, but run_depth_probe stores max_k bar-sigma.
    if not np.isclose(
        float(summary["max_weighted_sigma"]),
        float(source["sigma_weighted_mean"]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Recomputed maximum weighted dispersion does not match the source artifact.")


def _find_spec(label: str) -> CorridorSpec:
    matches = [spec for spec in REGISTERED_GRID if spec.label == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered corridor spec for {label!r}, found {len(matches)}.")
    return matches[0]


def _run_selected(source: Mapping[str, Any], config: OEHConfig) -> OEHResult:
    spec = _find_spec(str(source["instance"]))
    multiplier = int(source["pilot_multiplier"])
    transitions_count = multiplier * (spec.chain_length + 1) * 2
    _, _, initial, _ = build_corridor(spec)
    buffer = collect_buffer(
        spec,
        num_transitions=transitions_count,
        epsilon=BEHAVIOR_EPSILON,
        gamma=spec.gamma,
        seed=BUFFER_SEED + 17 * spec.chain_length + multiplier,
    )
    transitions = transitions_from_arrays(buffer)
    return estimate_oeh(
        transitions,
        initial,
        num_states=spec.chain_length + 1,
        num_actions=2,
        config=config,
        rng=prefix_bootstrap_rng(BUFFER_SEED, len(transitions), spec.chain_length + 1, 2),
    )


def _plot(rows: Sequence[Mapping[str, Any]], output_dir: Path, max_plot_depth: int, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    displayed = [row for row in rows if int(row["depth"]) <= max_plot_depth]
    if not displayed:
        raise ValueError("max-plot-depth excludes every curve point.")
    depth = np.asarray([int(row["depth"]) for row in displayed])
    gap = np.asarray([float(row["weighted_gap"]) for row in displayed])
    sigma = np.asarray([float(row["weighted_sigma"]) for row in displayed])
    loss = np.asarray([float(row["loss_at_one"]) for row in displayed])
    ceiling = np.asarray([float(row["theorem_ceiling"]) for row in displayed])
    epsilon = float(displayed[0]["epsilon"])

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(depth, gap, color="tab:blue", marker="o", markersize=3)
    axes[0].set_ylabel(r"Weighted gap $\sum_s w_k\widehat\Delta_k$")
    axes[1].plot(depth, sigma, color="tab:orange", marker="o", markersize=3)
    axes[1].set_ylabel(r"Weighted dispersion $\bar\sigma_k$")
    floor = np.finfo(float).tiny
    axes[2].plot(depth, np.maximum(loss, floor), color="tab:red", marker="o", markersize=3, label=r"$L_k(1)$")
    axes[2].plot(depth, np.maximum(ceiling, floor), color="tab:purple", linestyle="--", label="Theorem ceiling")
    axes[2].axhline(epsilon, color="black", linestyle=":", label=r"$\varepsilon=0.05$")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Loss / ceiling (log scale)")
    axes[2].set_xlabel(r"Lookahead depth $k$")
    axes[2].legend(loc="best", fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"oeh_mechanism.{extension}", dpi=200)
    plt.close(fig)
    svg = output_dir / "oeh_mechanism.svg"
    normalized_svg = "\n".join(
        line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
    ) + "\n"
    with svg.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized_svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-plot-depth", type=int, default=15)
    args = parser.parse_args()
    if args.max_plot_depth < 0:
        raise ValueError("max-plot-depth must be non-negative.")

    config = OEHConfig.load()
    assert_registered(config)
    source_rows = read_csv_rows(args.source)
    selected = select_representative(source_rows)
    result = _run_selected(selected, config)
    curves = weighted_curve_rows(result, config.epsilon)
    summary = summarize_curve(curves)
    _assert_matches_source(summary, selected)

    reserve_output_dir(args.output_dir)
    write_csv_rows(args.output_dir / "oeh_mechanism.csv", curves, CURVE_FIELDS)
    _plot(
        curves,
        args.output_dir,
        args.max_plot_depth,
        f"Registered OEH mechanism: {selected['instance']}, pilot multiplier {selected['pilot_multiplier']}",
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "oeh_mechanism_report.json"
    }
    source_path = args.source.resolve()
    try:
        source_label = source_path.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = str(source_path)
    report = {
        "scope": "Post-hoc mechanism visualization on development probe; not confirmatory or held-out.",
        "selection_rule": "Largest finite weighted_gap_swing among scored frozen depth-probe rows.",
        "source": {"path": source_label, "sha256": sha256_file(source_path)},
        "selected_source_row": dict(selected),
        "registered_config": asdict(config),
        "displayed_depth_max": args.max_plot_depth,
        "full_curve_depth_max": len(curves) - 1,
        "summary": summary,
        "output_sha256": output_hashes,
    }
    write_json(args.output_dir / "oeh_mechanism_report.json", report)
    print(f"Wrote reproducible OEH mechanism figure to {args.output_dir}")


if __name__ == "__main__":
    main()
