#!/usr/bin/env python
"""Round 4b: does ``k_dagger = 0`` survive a benchmark built to break it?

Pre-registered in ``ARM_REGISTRY.md`` on 2026-09-06, hypotheses and stop rule
written before the first estimator call (``EXPERIMENT_RULE.md`` rule 7).

**The question.** The development split reports ``k_dagger = 0`` at all 12,019
evidence-curve points, and the project reads that as a property of the
definition. Two objections have never been separated from that reading:

* the MDPs have 8-64 states at ``gamma = 0.95``, where one or two backups
  already give the optimal action ranking, so ``k = 0`` as the argmin is close
  to forced by the instance rather than informative about the definition;
* 12 of 20 ``lock_structured`` pilots contain no non-zero reward at all, and no
  data-only statistic can work on a buffer with no signal in it.

**The design.** ``code/decoy_corridor.py`` builds instances where neither
objection applies: an epsilon-greedy-advance behavior policy reaches the goal on
83-100% of trajectories instead of the combination lock's ``2^-L``, and a
locally attractive but globally suboptimal decoy keeps the action gap alive on
the states that carry occupancy weight.

.. note:: **Correction, 2026-09-06.** This docstring previously claimed the
   design makes the depth-0 ranking *provably wrong*. Only ``pi*(s_0) =
   advance`` was verified (12/12); the depth-0 half was argued from the design
   and never measured, and the argument is unsound because ``Q^(0)`` is the
   behavior-policy value rather than a greedy rule on immediate reward. See
   ``code/decoy_corridor.py`` and ``WEEK2_DECISION_GATE.md`` sec 10j. The probe
   is cited as removing reward starvation and occupancy dilution, **not** as
   establishing that depth is required.

**The two outcomes, both registered as usable:**

* **H1** -- ``k_dagger`` moves interior on a substantial share of instances.
  Then the current negative result is about *this benchmark*, and the C2
  conclusion must be narrowed to say so.
* **H2** -- ``k_dagger = 0`` persists even here. Then reward starvation and
  occupancy dilution do not explain the result on this probe. This narrows the
  plausible explanations but does not by itself prove a benchmark-independent
  property of the definition.

**Nothing is tuned to make either happen.** The estimator runs unmodified;
``assert_registered`` raises on any deviation from ``protocol_config.json``.
The escape criterion is the one in ``EXPERIMENT_RULE.md`` sec 2, verbatim:
``exists k: L_k(1) >= epsilon`` and ``exists k: m_k > 1 + 1e-9`` and
``0 < k_dagger < K_max``. Only the environment is new, and it is a probe for
testing the definition, not a benchmark instance (``PLAN_Final.md`` sec 5).

Usage::

    python -m scripts.analysis.run_depth_probe --output-dir artifacts/week2/depth_probe_<date>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from decoy_corridor import (  # noqa: E402
    ADVANCE,
    REGISTERED_GRID,
    CorridorSpec,
    build_corridor,
    collect_buffer,
    reward_density,
    terminal_reach_rate,
)
from evaluation.io import sha256_file, write_csv_rows, write_json  # noqa: E402
from exact_solver import value_iteration  # noqa: E402
from oeh_estimator import OEHConfig, estimate_oeh  # noqa: E402
from oeh_validation import prefix_bootstrap_rng  # noqa: E402
from positive_control import assert_registered  # noqa: E402

#: The registered pilot budget axis (``ctrl-pilot``, round 1). No new value.
PILOT_MULTIPLIERS: tuple[int, ...] = (2, 5, 10)

#: ``EXPERIMENT_RULE.md`` sec 2: "actually moved" rather than float noise.
MULTIPLIER_ESCAPE_ATOL = 1e-9

#: Registered buffer seed, declared in the round 4b entry before any run.
BUFFER_SEED = 20260906

#: Probability the behavior policy takes the decoy at a corridor state. Chosen
#: to keep the corridor completable; the resulting reward density and terminal
#: reach rate are reported on every row so the premise is measured, not assumed.
BEHAVIOR_EPSILON = 0.2

ROW_FIELDS = (
    "instance",
    "chain_length",
    "decoy_fraction",
    "slip_probability",
    "decoy_reward",
    "pilot_multiplier",
    "pilot_size",
    "num_trajectories",
    "reward_density",
    "terminal_reach_rate",
    "optimal_action_is_advance",
    "value_at_start",
    "oeh_status",
    "abstain_reason",
    "max_loss_at_one",
    "epsilon",
    "loss_reaches_epsilon",
    "max_multiplier",
    "multiplier_above_one",
    "k_dagger",
    "interior_argmin",
    "escaped",
    "sigma_weighted_mean",
    "sigma_headroom",
    "weighted_gap_at_zero",
    "max_weighted_gap",
    "argmax_weighted_gap",
    "weighted_gap_swing",
    "loss_at_gap_peak",
    "theorem_1_holds",
    "num_supported_states",
)


def _finite_max(values: Any) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("inf") if np.any(np.isinf(array)) else None
    return float(np.max(finite))


def evaluate_instance(spec: CorridorSpec, multiplier: int, config: OEHConfig) -> dict[str, Any]:
    P, R, initial, _ = build_corridor(spec)
    num_states, num_actions = P.shape[0], P.shape[1]
    pilot_size = multiplier * num_states * num_actions

    # Oracle side, used only for scoring the instance's design (sec 11 rule 5:
    # never inside the estimator). It answers "is the decoy actually a decoy?".
    class _View:
        pass

    view = _View()
    view.P, view.R, view.gamma, view.rho0 = P, R, spec.gamma, initial  # type: ignore[attr-defined]
    solution = value_iteration(view)
    advance_optimal = bool(solution.optimal_action_mask[0][ADVANCE])

    buffer = collect_buffer(
        spec,
        num_transitions=pilot_size,
        epsilon=BEHAVIOR_EPSILON,
        gamma=spec.gamma,
        seed=BUFFER_SEED + 17 * spec.chain_length + multiplier,
    )
    transitions = transitions_from_arrays(buffer)
    result = estimate_oeh(
        transitions,
        initial,
        num_states=num_states,
        num_actions=num_actions,
        config=config,
        rng=prefix_bootstrap_rng(BUFFER_SEED, len(transitions), num_states, num_actions),
    )

    gap_at_zero = max_gap = argmax_gap = gap_swing = loss_at_peak = None
    bound_holds = None
    if result.abstained:
        sigma_bar = None
        max_loss = None
        max_multiplier = None
        k_dagger = None
    else:
        weights = np.asarray(result.occupancy_weights, dtype=np.float64)
        sigmas = np.nan_to_num(np.asarray(result.sigmas, dtype=np.float64), nan=0.0)
        sigma_bar = float(np.max((weights * sigmas).sum(axis=1)))
        max_loss = _finite_max(result.losses_at_one)
        max_multiplier = _finite_max(result.multipliers)
        k_dagger = result.k_dagger

        # The diagnostic this probe exists for. If the action gap does NOT
        # vary with depth, the instance failed to implement its own design and
        # nothing about the definition follows. If it DOES vary and the loss
        # still never moves, these two benchmark confounds do not explain the
        # flat curve -- and which of the two happened has to be readable from
        # the artifact, not re-derived by the reader.
        deltas = np.nan_to_num(np.asarray(result.deltas, dtype=np.float64), nan=0.0)
        losses = np.asarray(result.losses_at_one, dtype=np.float64)
        weighted_gap = (weights * deltas).sum(axis=1)
        weighted_sigma = (weights * sigmas).sum(axis=1)
        gap_at_zero = float(weighted_gap[0])
        max_gap = float(np.max(weighted_gap))
        argmax_gap = int(np.argmax(weighted_gap))
        smallest_gap = float(np.min(weighted_gap))
        gap_swing = max_gap / smallest_gap if smallest_gap > 0 else float("inf")
        loss_at_peak = float(losses[argmax_gap])
        bound_holds = bool(np.all(losses <= 0.16997 * weighted_sigma + 1e-12))

    epsilon = config.epsilon
    loss_reaches = bool(max_loss is not None and max_loss >= epsilon)
    multiplier_above = bool(max_multiplier is not None and max_multiplier > 1.0 + MULTIPLIER_ESCAPE_ATOL)
    interior = bool(k_dagger is not None and 0 < k_dagger < config.k_max)
    required_sigma = epsilon / 0.16997

    return {
        "instance": spec.label,
        "chain_length": spec.chain_length,
        "decoy_fraction": spec.decoy_fraction,
        "slip_probability": spec.slip_probability,
        "decoy_reward": spec.decoy_reward,
        "pilot_multiplier": multiplier,
        "pilot_size": pilot_size,
        "num_trajectories": int(len(np.unique(buffer["trajectory_id"]))),
        "reward_density": reward_density(buffer),
        "terminal_reach_rate": terminal_reach_rate(buffer),
        "optimal_action_is_advance": advance_optimal,
        "value_at_start": float(solution.values[0]),
        "oeh_status": "abstained" if result.abstained else ("censored" if result.censored_right else "observed"),
        "abstain_reason": result.reason or "",
        "max_loss_at_one": "" if max_loss is None else max_loss,
        "epsilon": epsilon,
        "loss_reaches_epsilon": loss_reaches,
        "max_multiplier": "" if max_multiplier is None else max_multiplier,
        "multiplier_above_one": multiplier_above,
        "k_dagger": "" if k_dagger is None else k_dagger,
        "interior_argmin": interior,
        "escaped": bool(loss_reaches and multiplier_above and interior),
        "sigma_weighted_mean": "" if sigma_bar is None else sigma_bar,
        "sigma_headroom": "" if sigma_bar is None else sigma_bar / required_sigma,
        "weighted_gap_at_zero": "" if gap_at_zero is None else gap_at_zero,
        "max_weighted_gap": "" if max_gap is None else max_gap,
        "argmax_weighted_gap": "" if argmax_gap is None else argmax_gap,
        "weighted_gap_swing": "" if gap_swing is None else gap_swing,
        "loss_at_gap_peak": "" if loss_at_peak is None else loss_at_peak,
        "theorem_1_holds": "" if bound_holds is None else bound_holds,
        "num_supported_states": sum(1 for actions in result.supported_actions if len(actions) >= 2),
    }


def run(config: OEHConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_registered(config)
    rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(REGISTERED_GRID) * len(PILOT_MULTIPLIERS)
    done = 0
    for spec in REGISTERED_GRID:
        for multiplier in PILOT_MULTIPLIERS:
            row = evaluate_instance(spec, multiplier, config)
            rows.append(row)
            done += 1
            print(
                f"[{done}/{total}] {row['instance']} N0={multiplier}x: "
                f"status={row['oeh_status']} k+={row['k_dagger']} "
                f"maxL={row['max_loss_at_one']} escaped={row['escaped']} "
                f"rho={row['reward_density']:.3f} ({time.time() - started:.0f}s)",
                flush=True,
            )

    scored = [row for row in rows if row["oeh_status"] != "abstained"]
    interior = [row for row in scored if row["interior_argmin"]]
    escaped = [row for row in rows if row["escaped"]]
    depths = [int(row["k_dagger"]) for row in scored if row["k_dagger"] != ""]
    sigmas = [float(row["sigma_weighted_mean"]) for row in scored if row["sigma_weighted_mean"] != ""]
    swings = [float(row["weighted_gap_swing"]) for row in scored if row["weighted_gap_swing"] not in ("", float("inf"))]
    peaks = [int(row["argmax_weighted_gap"]) for row in scored if row["argmax_weighted_gap"] != ""]
    interior_gap_peaks = [depth for depth in peaks if depth > 0]

    if interior:
        verdict = "H1"
        reading = (
            "k_dagger moves interior on instances built to exclude reward starvation and "
            "occupancy dilution. The saturation observed on the development split is "
            "therefore at least partly a property of THAT benchmark, and the C2 conclusion "
            "has to be narrowed to say so."
        )
    else:
        verdict = "H2"
        reading = (
            "k_dagger stays at 0 even on instances that exclude reward starvation (the "
            "pilot reaches reward on most trajectories) and occupancy dilution (the gap "
            "peaks at an interior depth). Those two objections to the null are therefore "
            "removed. This does not prove a universal property of the definition. NOT "
            "established here: that lookahead depth is required on these "
            "instances -- k_settle is unmeasured, see WEEK2_DECISION_GATE.md sec 10j."
        )

    report = {
        "round": "4b, pre-registered in ARM_REGISTRY.md on 2026-09-06 before the first run",
        "purpose": (
            "Separate 'k_dagger = 0 is a fact about the definition' from 'k_dagger = 0 is a "
            "fact about a benchmark that is too shallow and reward-starved to show anything "
            "else'."
        ),
        "escape_criterion": (
            "EXPERIMENT_RULE.md sec 2, verbatim: exists k with L_k(1) >= epsilon, and exists k "
            "with m_k > 1 + 1e-9, and 0 < k_dagger < K_max."
        ),
        "estimator": "registered, unmodified; assert_registered passed",
        "buffer_seed": BUFFER_SEED,
        "behavior_epsilon": BEHAVIOR_EPSILON,
        "pilot_multipliers": list(PILOT_MULTIPLIERS),
        "instances": len(REGISTERED_GRID),
        "cells": len(rows),
        "scored": len(scored),
        "abstained": len(rows) - len(scored),
        "design_checks": {
            "advance_is_optimal_everywhere": all(row["optimal_action_is_advance"] for row in rows),
            "min_reward_density": min(row["reward_density"] for row in rows),
            "min_terminal_reach_rate": min(row["terminal_reach_rate"] for row in rows),
            "note": (
                "Both are the premise the probe rests on. On the development split 12 of 20 "
                "lock_structured pilots have a reward density of exactly 0."
            ),
        },
        "k_dagger_distribution": {str(depth): depths.count(depth) for depth in sorted(set(depths))},
        "interior_argmin_cells": len(interior),
        "escaped_cells": len(escaped),
        "sigma_weighted_mean": (
            None
            if not sigmas
            else {"min": min(sigmas), "median": float(np.median(sigmas)), "max": max(sigmas)}
        ),
        "required_sigma_for_escape": config.epsilon / 0.16997,
        "gap_diagnostic": {
            "weighted_gap_swing": (
                None
                if not swings
                else {"min": min(swings), "median": float(np.median(swings)), "max": max(swings)}
            ),
            "cells_with_interior_gap_peak": len(interior_gap_peaks),
            "cells_scored": len(scored),
            "gap_peak_depths": {str(d): peaks.count(d) for d in sorted(set(peaks))},
            "note": (
                "This separates two very different failures. A gap that does not move with depth "
                "would mean the instance never implemented its design, and nothing about the "
                "definition would follow. A gap that swings widely while the loss stays pinned "
                "shows that reward starvation and occupancy dilution do not explain the flat curve."
            ),
        },
        "theorem_1_holds_everywhere": all(
            row["theorem_1_holds"] is True for row in scored if row["theorem_1_holds"] != ""
        ),
        "verdict": verdict,
        "reading": reading,
        "runtime_seconds": round(time.time() - started, 1),
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, report = run(OEHConfig.load())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "depth_probe.csv", rows, ROW_FIELDS)
    report["artifact_sha256"] = {"depth_probe.csv": sha256_file(args.output_dir / "depth_probe.csv")}
    write_json(args.output_dir / "depth_probe_report.json", report)

    print()
    print(f"  {report['cells']} cells, {report['scored']} scored, {report['abstained']} abstained")
    print(f"  k_dagger distribution: {report['k_dagger_distribution']}")
    print(f"  interior argmin: {report['interior_argmin_cells']}  |  full escapes: {report['escaped_cells']}")
    if report["sigma_weighted_mean"]:
        sigma = report["sigma_weighted_mean"]
        print(
            f"  sigma_bar: median {sigma['median']:.4f} (escape needs "
            f"{report['required_sigma_for_escape']:.4f})"
        )
    print(f"\n  VERDICT: {report['verdict']}")
    print(f"  {report['reading']}")
    print(f"\n  -> {args.output_dir}")


if __name__ == "__main__":
    main()
