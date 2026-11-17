#!/usr/bin/env python
"""Round 4c: do the *admissible* definitions recover depth where it demonstrably exists?

Pre-registered in ``ARM_REGISTRY.md`` on 2026-09-06, hypotheses, ground truth
and stop rule written before the first estimator call (``EXPERIMENT_RULE.md``
rules 2 and 5).

**Why the obvious question is the wrong one.** "Can any definition escape
saturation?" is already answered on disk: on the 75 development pilots at
``f = 0.005``, ``max_state`` escapes all three conditions on 27 of 62 scored
pilots, ``probability`` on 16, ``materiality`` on 15, and ``registered`` on
none (``alternative_definitions_2026-09-01``).  Re-running that on new
instances spends a registered run to confirm a number already recorded.

**What only the corridor can give: ground truth for depth.** On a development
MDP, when ``max_state`` reports ``k_dagger = 3`` nobody knows whether 3 is
right, so "the statistic runs" and "the statistic measures depth" cannot be
told apart.  The decoy corridor separates them, because the depth at which
lookahead stops changing the greedy choice is fixed by construction and
computable exactly from the true model::

    k_settle_exact = min{k : greedy(Q^(k))(s) is optimal at every corridor state s}

with ``Q^(0)`` the restricted behaviour evaluation and
``Q^(k+1) = r + gamma P max_a Q^(k)`` -- the convention
``oeh_estimator._q_sequence`` uses, borrowed here through
``exact_evidence_curve._true_q_sequence`` so the two cannot drift apart.  This
is the offline analogue of the quantity the interactive effective horizon
measures: how much lookahead is needed before the greedy choice is correct.

This is an **oracle design check**, in the same category as round 4b's
``optimal_action_is_advance`` and ``value_at_start``.  It never enters the
estimator (``PLAN_Final.md`` sec 11 rule 5), and it is **not** the exact
evidence curve of B1 -- there is no ``R=200`` resampling and no ``sigma`` here.

**It also supplies a check round 4b was missing.** Round 4b states that on the
corridor "the depth-0 action ranking is provably wrong"; what it verified is
``pi*(s_0) = advance`` on 12/12, which is the *other* half of that claim.
``k_settle_exact`` tests the half that was asserted, and the answer is that
depth 0 is already optimal at every state on **8 of the 12 instances**: the
behaviour policy is epsilon-greedy-*advance* at ``epsilon = 0.2``, so ``Q^(0)``
inherits a policy that already completes the corridor, and inherits its
competence with it.  Reward density and depth-necessity pull against each
other, and the grid resolved that tension towards reward density.  Every row
therefore carries ``depth_is_necessary``, and the escape counts are reported
separately on the cells where the hypothesis is testable at all.

**The three registered outcomes** (``ARM_REGISTRY.md`` round 4c):

* **O1** -- a tested scale-invariant definition escapes *and* its ``k_dagger``
  tracks ``k_flip_exact``.  That definition works on this probe; the result does
  not establish the whole construction class.
* **O2** -- it escapes but ``k_dagger`` does not track ``k_flip_exact``.  Scale
  invariance and the other design requirements are then not sufficient: a
  statistic can satisfy A1-A3, be non-degenerate, and still not measure depth.
* **O3** -- ``k_dagger = 0`` for all four here too.  C1 in its strongest form.

**Nothing is tuned.** The estimator is unmodified and ``assert_registered``
raises on any deviation.  The instance grid, seeds, pilot budgets and
behaviour policy are round 4b's verbatim, so the ``registered`` variant must
reproduce ``depth_probe_2026-09-06/depth_probe.csv`` bit for bit; a mismatch is
a bug and stops the run.  The materiality grid is
``alternative_definitions``' registered ``{0.005, 0.01, 0.02}``, reported in
full -- no fourth value, and no choosing one after seeing results.

Usage::

    python -m scripts.analysis.run_corridor_definitions \\
        --output-dir artifacts/week2/corridor_definitions_<date>
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from decoy_corridor import (  # noqa: E402
    ADVANCE,
    DECOY,
    REGISTERED_GRID,
    CorridorSpec,
    build_corridor,
    collect_buffer,
    reward_density,
    terminal_reach_rate,
)
from evaluation.io import read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from exact_evidence_curve import _true_q_sequence  # noqa: E402
from exact_solver import value_iteration  # noqa: E402
from oeh_estimator import OEHConfig, OEHResult, estimate_oeh  # noqa: E402
from oeh_validation import prefix_bootstrap_rng  # noqa: E402
from positive_control import assert_registered  # noqa: E402
from scripts.analysis.run_alternative_definitions import (  # noqa: E402
    CSV_FIELDS,
    MATERIALITY_VARIANTS,
    VARIANTS,
    _check_registered_matches_estimator,
    value_scale,
    variant_rows,
)
from scripts.analysis.run_depth_probe import (  # noqa: E402
    BEHAVIOR_EPSILON,
    BUFFER_SEED,
    PILOT_MULTIPLIERS,
)

#: Registered materiality grid, taken verbatim from
#: ``alternative_definitions_2026-09-01``.  Reported in full; a fourth value
#: would be a new constant and choosing among these after seeing results would
#: be ``EXPERIMENT_RULE.md`` rule 2.
MATERIALITY_FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.02)

#: Extra columns this round adds on top of ``run_alternative_definitions``'
#: schema: the corridor's design parameters, round 4b's design checks, and the
#: depth-alignment columns that are the point of the run.
CORRIDOR_FIELDS: tuple[str, ...] = CSV_FIELDS + (
    "chain_length",
    "decoy_fraction",
    "slip_probability",
    "pilot_multiplier",
    "pilot_size",
    "num_trajectories",
    "reward_density",
    "terminal_reach_rate",
    "optimal_action_is_advance",
    "multiplier_above_one",
    "k_settle_exact",
    "k_flip_exact_at_start",
    "depth_is_necessary",
    "argmax_exact_gap_at_start",
    "argmax_weighted_gap",
    "occupancy_at_start",
    "depth_error_vs_settle",
    "abs_depth_error_vs_settle",
    "depth_error_vs_gap_peak",
    "material_sum_empty",
)


class _MDPView:
    """Minimal duck-typed MDP for the oracle helpers, which read attributes."""

    def __init__(self, P: np.ndarray, R: np.ndarray, gamma: float, rho0: np.ndarray) -> None:
        self.P = P
        self.R = R
        self.gamma = gamma
        self.rho0 = rho0


def exact_depth_reference(
    spec: CorridorSpec, k_max: int, behavior_epsilon: float
) -> dict[str, Any]:
    """Oracle design checks: where the corridor's mechanism actually lives.

    Three references, all from the true model and the true behaviour policy, so
    none is available to the estimator.  They exist to score whether a
    data-only ``k_dagger`` landed in the right place.

    * ``k_settle_exact`` -- the depth at which the greedy action becomes
      optimal at *every* corridor state and stays there.  This is the primary
      reference: the offline analogue of the lookahead the interactive
      effective horizon measures.  ``None`` if it never settles within
      ``k_max``.
    * ``k_flip_exact_at_start`` -- the same, restricted to ``s_0``.
    * ``argmax_exact_gap_at_start`` -- where the exact action gap at ``s_0`` is
      widest, which is a different question and reported separately.

    Ties matter here: a state where several actions are optimal is *not* a
    state that needs more lookahead, so the test is membership in the optimal
    set, not equality with one designated action.
    """

    P, R, initial, _ = build_corridor(spec)
    num_states, num_actions = P.shape[0], P.shape[1]

    # The true behaviour policy, not an empirical estimate of it: this is the
    # oracle side.  epsilon-greedy-advance, exactly what collect_buffer rolls.
    beta = np.zeros((num_states, num_actions), dtype=np.float64)
    beta[:, ADVANCE] = 1.0 - behavior_epsilon
    beta[:, DECOY] = behavior_epsilon
    mask = np.ones((num_states, num_actions), dtype=bool)

    view = _MDPView(P, R, spec.gamma, initial)
    sequence = _true_q_sequence(view, beta, mask, k_max)
    solution = value_iteration(view)
    optimal = np.asarray(solution.optimal_action_mask)

    # The terminal is absorbing with identical actions, so it is excluded: it
    # would be "settled" at every depth and would only dilute the check.
    corridor = range(spec.chain_length)

    def greedy_is_optimal(q_values: np.ndarray) -> bool:
        return all(bool(optimal[s][int(np.argmax(q_values[s]))]) for s in corridor)

    settle = next(
        (k for k, q in enumerate(sequence) if greedy_is_optimal(q)),
        None,
    )
    flip = next(
        (
            k
            for k, q in enumerate(sequence)
            if bool(optimal[0][int(np.argmax(q[0]))])
        ),
        None,
    )
    gaps = np.array(
        [abs(float(q[0, ADVANCE]) - float(q[0, DECOY])) for q in sequence],
        dtype=np.float64,
    )
    return {
        "k_settle_exact": settle,
        "k_flip_exact_at_start": flip,
        "argmax_exact_gap_at_start": int(np.argmax(gaps)),
        "optimal_action_is_advance": bool(optimal[0][ADVANCE]),
        "value_at_start": float(solution.values[0]),
    }


def _weighted_gap_peak(result: OEHResult) -> int | None:
    """Data-only depth where ``sum_s w_k(s) Delta_hat_k(s)`` is largest.

    Round 4b's diagnostic, recomputed here so the two artifacts can be joined
    on it without trusting a copy.
    """

    if result.occupancy_weights is None or result.deltas is None:
        return None
    weights = np.asarray(result.occupancy_weights, dtype=np.float64)
    deltas = np.nan_to_num(np.asarray(result.deltas, dtype=np.float64), nan=0.0)
    return int(np.argmax((weights * deltas).sum(axis=1)))


def evaluate_instance(
    spec: CorridorSpec, multiplier: int, config: OEHConfig
) -> list[dict[str, Any]]:
    """All four definitions on one corridor cell, plus the alignment columns."""

    P, _, initial, _ = build_corridor(spec)
    num_states, num_actions = P.shape[0], P.shape[1]
    pilot_target = multiplier * num_states * num_actions

    reference = exact_depth_reference(spec, config.k_max, BEHAVIOR_EPSILON)

    # Round 4b's seeds, verbatim.  This is what makes `registered` reproduce
    # depth_probe.csv, which is the run's own drift check.
    buffer = collect_buffer(
        spec,
        num_transitions=pilot_target,
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

    rows = variant_rows(
        f"{spec.label}_x{multiplier}",
        "epsilon_greedy_advance",
        "decoy_corridor",
        result,
        config,
        scale=value_scale(result, buffer, spec.gamma),
        pilot_size=len(transitions),
        materiality_fractions=MATERIALITY_FRACTIONS,
    )
    if not _check_registered_matches_estimator(rows, result):
        raise RuntimeError(
            f"{spec.label} x{multiplier}: the `registered` variant does not "
            "reproduce the estimator's own loss, so no comparison between "
            "variants on this cell is on a common footing."
        )

    gap_peak = _weighted_gap_peak(result)
    settle = reference["k_settle_exact"]
    occupancy_at_start = (
        ""
        if result.occupancy_weights is None
        else float(np.max(np.asarray(result.occupancy_weights, dtype=np.float64)[:, 0]))
    )
    common = {
        "chain_length": spec.chain_length,
        "decoy_fraction": spec.decoy_fraction,
        "slip_probability": spec.slip_probability,
        "pilot_multiplier": multiplier,
        "pilot_size": len(transitions),
        "num_trajectories": int(len(np.unique(buffer["trajectory_id"]))),
        "reward_density": reward_density(buffer),
        "terminal_reach_rate": terminal_reach_rate(buffer),
        "optimal_action_is_advance": reference["optimal_action_is_advance"],
        "k_settle_exact": "" if settle is None else settle,
        "k_flip_exact_at_start": (
            "" if reference["k_flip_exact_at_start"] is None
            else reference["k_flip_exact_at_start"]
        ),
        # The partition that decides which cells test the hypothesis at all.
        # Where depth 0 is already optimal everywhere, k_dagger = 0 is the
        # correct answer and carries no evidence about the definition.
        "depth_is_necessary": bool(settle is not None and settle > 0),
        "argmax_exact_gap_at_start": reference["argmax_exact_gap_at_start"],
        "argmax_weighted_gap": "" if gap_peak is None else gap_peak,
        "occupancy_at_start": occupancy_at_start,
    }

    for row in rows:
        row.update(common)
        k_dagger = row["k_dagger"]
        scored = k_dagger != ""
        row["multiplier_above_one"] = bool(
            row["max_multiplier"] not in ("", None)
            and float(row["max_multiplier"]) > 1.0 + 1e-9
        )
        error_settle = (
            int(k_dagger) - int(settle) if scored and settle is not None else None
        )
        error_peak = (
            int(k_dagger) - int(gap_peak) if scored and gap_peak is not None else None
        )
        row["depth_error_vs_settle"] = "" if error_settle is None else error_settle
        row["abs_depth_error_vs_settle"] = (
            "" if error_settle is None else abs(error_settle)
        )
        row["depth_error_vs_gap_peak"] = "" if error_peak is None else error_peak
        # The silent-empty-sum failure DERIVATIONS.md sec 5 records as having
        # actually happened: delta = f * V_hat above every gap in the instance.
        row["material_sum_empty"] = bool(
            row["variant"] in MATERIALITY_VARIANTS
            and row["material_states_at_k_dagger"] not in ("", None)
            and int(row["material_states_at_k_dagger"]) == 0
        )
    return rows


# --------------------------------------------------------------------------
# Summarising
# --------------------------------------------------------------------------


def constant_predictor_baseline(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How well "always answer the modal depth" does on the same cells.

    The control ``c2_correlation`` applies with ``|S||A|`` and this run needs
    its analogue.  With ``k_settle_exact`` concentrated on one value, a
    statistic that lands on it once has demonstrated nothing that a constant
    would not; without this number, a single correct hit reads as a signal.
    """

    truths = [
        int(row["k_settle_exact"])
        for row in rows
        if row["depth_is_necessary"]
        and row["oeh_status"] != "abstained"
        and row["k_settle_exact"] != ""
    ]
    if not truths:
        return {"cells": 0, "modal_depth": None, "correct": 0, "accuracy": None}
    modal, hits = Counter(truths).most_common(1)[0]
    return {
        "cells": len(truths),
        "modal_depth": modal,
        "correct": hits,
        "accuracy": hits / len(truths),
        "note": (
            "A definition must beat this to have measured anything. It uses no "
            "data at all."
        ),
    }


def argmin_attractor_diagnostic(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Why materiality-based ``k_dagger`` is drawn to depths with no material states.

    ``m_k = min{mu >= 1 : L_k(mu) <= epsilon}`` and ``m_k >= 1`` by
    construction, so a depth whose material set is empty has ``L_k == 0``,
    hence ``m_k = 1``, hence it *is* a global minimiser and wins the argmin
    outright.  ``k_dagger`` then reports the depth at which the threshold
    ``delta = f * V_hat`` happened to exclude everything, not the depth at
    which evidence is cheapest.  This is a defect of the construction, not of
    a particular ``V_hat``: correcting the value scale moves which depths are
    empty, not the fact that an empty one wins.
    """

    affected = [
        row
        for row in rows
        if row["oeh_status"] != "abstained"
        and row["variant"] in MATERIALITY_VARIANTS
        and row["material_sum_empty"]
    ]
    return {
        "mechanism": (
            "empty material set => L_k = 0 => m_k = 1 = the global floor => "
            "that depth is an argmin by default"
        ),
        "rows_with_empty_material_set_at_k_dagger": len(affected),
        "of_which_counted_as_escapes": sum(1 for row in affected if row["escaped"]),
        "of_which_counted_as_interior_argmin": sum(
            1 for row in affected if row["interior_argmin"]
        ),
        "affects": sorted({row["variant"] for row in affected}),
        "note": (
            "Both definitions Corollary 2.3 constructs to prove the admissible "
            "class is non-empty are affected. Distinct from the calibration "
            "bug in DERIVATIONS.md sec 5, which is about choosing V_hat."
        ),
    }


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, or ``None`` when it is undefined.

    Undefined includes the case that matters here: if every cell shares one
    ``k_flip_exact`` there are no ranks to correlate, and a coefficient
    computed anyway would be noise dressed as evidence.
    """

    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(xs, ys).statistic)
    return None if math.isnan(value) else value


def summarize_variant(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Escape counts and, for escaping rows only, depth alignment.

    Alignment is read on escaping rows because ``k_dagger`` on a non-escaping
    row is the argmin of a curve that never reached ``epsilon`` -- the
    registered tie-break's 0, not a measurement.

    Everything is reported twice: over all cells, and over the subset where
    ``depth_is_necessary``.  On a cell where depth 0 is already optimal at
    every state, ``k_dagger = 0`` is the *right* answer, so counting it as
    evidence that the definition is blind to depth would be counting a correct
    answer as a failure.
    """

    scored = [row for row in rows if row["oeh_status"] != "abstained"]
    # An escape whose k_dagger sits at a depth where delta = f * V_hat emptied
    # the material set is not a measurement: the loss there is 0 because the
    # sum has no terms, m_k = 1 by default, and that depth wins the argmin for
    # a reason that has nothing to do with evidence.  Counted separately rather
    # than dropped, because how often it happens is itself a result about the
    # materiality construction.
    escaped = [row for row in scored if row["escaped"] and not row["material_sum_empty"]]
    spurious = [row for row in scored if row["escaped"] and row["material_sum_empty"]]
    testable = [row for row in scored if row["depth_is_necessary"]]
    testable_escaped = [row for row in testable if row["escaped"] and not row["material_sum_empty"]]
    errors = [
        int(row["abs_depth_error_vs_settle"])
        for row in escaped
        if row["abs_depth_error_vs_settle"] != ""
    ]
    pairs = [
        (int(row["k_dagger"]), int(row["k_settle_exact"]))
        for row in escaped
        if row["k_dagger"] != "" and row["k_settle_exact"] != ""
    ]
    return {
        "rows": len(rows),
        "scored": len(scored),
        "abstained": len(rows) - len(scored),
        "loss_reaches_epsilon": sum(1 for row in scored if row["loss_reaches_epsilon"]),
        "multiplier_above_one": sum(1 for row in scored if row["multiplier_above_one"]),
        "interior_argmin": sum(1 for row in scored if row["interior_argmin"]),
        "escaped_raw": len(escaped) + len(spurious),
        "escaped": len(escaped),
        "escaped_spurious_empty_material_set": len(spurious),
        "material_sum_empty": sum(1 for row in scored if row["material_sum_empty"]),
        "k_dagger_distribution": dict(
            sorted(Counter(row["k_dagger"] for row in scored).items(), key=lambda kv: str(kv[0]))
        ),
        "depth_necessary_cells": len(testable),
        "depth_necessary_escaped": len(testable_escaped),
        "depth_necessary_k_dagger": dict(
            sorted(
                Counter(row["k_dagger"] for row in testable).items(),
                key=lambda kv: str(kv[0]),
            )
        ),
        "depth_necessary_correct": sum(
            1 for row in testable if row["abs_depth_error_vs_settle"] == 0
        ),
        "aligned_rows": len(pairs),
        "median_abs_depth_error": statistics.median(errors) if errors else None,
        "within_one_of_settle": sum(1 for value in errors if value <= 1) if errors else 0,
        "spearman_k_dagger_vs_settle": _spearman(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        ),
    }


def _reproduces_round_4b(rows: Sequence[dict[str, Any]], probe_csv: Path) -> dict[str, Any]:
    """Cross-check ``registered`` against the committed round 4b artifact.

    Same instances, same seeds, same estimator: the losses must be identical.
    If they are not, something drifted between the two runs and no comparison
    across the four definitions means anything.
    """

    if not probe_csv.exists():
        return {"checked": False, "reason": f"{probe_csv} not found"}
    probe = {
        f"{row['instance']}_x{row['pilot_multiplier']}": row
        for row in read_csv_rows(probe_csv)
    }
    mismatches: list[str] = []
    compared = 0
    for row in rows:
        if row["variant"] != "registered":
            continue
        other = probe.get(str(row["mdp_id"]))
        if other is None:
            mismatches.append(f"{row['mdp_id']}: absent from round 4b")
            continue
        if row["max_loss_at_one"] == "" or other["max_loss_at_one"] == "":
            if row["max_loss_at_one"] != other["max_loss_at_one"]:
                mismatches.append(f"{row['mdp_id']}: one side has no loss")
            continue
        compared += 1
        if not math.isclose(
            float(row["max_loss_at_one"]),
            float(other["max_loss_at_one"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            mismatches.append(
                f"{row['mdp_id']}: {row['max_loss_at_one']} vs {other['max_loss_at_one']}"
            )
    return {
        "checked": True,
        "cells_compared": compared,
        "mismatches": mismatches,
        "reproduced": not mismatches,
    }


def run(config: OEHConfig, probe_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert_registered(config)
    started = time.time()
    rows: list[dict[str, Any]] = []
    total = len(REGISTERED_GRID) * len(PILOT_MULTIPLIERS)
    done = 0
    for spec in REGISTERED_GRID:
        for multiplier in PILOT_MULTIPLIERS:
            rows.extend(evaluate_instance(spec, multiplier, config))
            done += 1
            print(f"  [{done:2}/{total}] {spec.label} x{multiplier}", flush=True)

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        fractions = (
            MATERIALITY_FRACTIONS
            if variant in MATERIALITY_VARIANTS
            else MATERIALITY_FRACTIONS[:1]
        )
        for fraction in fractions:
            subset = [
                row
                for row in rows
                if row["variant"] == variant and row["materiality_fraction"] == fraction
            ]
            by_variant[f"{variant}@{fraction}"] = summarize_variant(subset)

    registered_rows = [row for row in rows if row["variant"] == "registered"]
    settles = [
        row["k_settle_exact"] for row in registered_rows if row["k_settle_exact"] != ""
    ]
    escaping = {
        name: stats["escaped"] for name, stats in by_variant.items() if stats["escaped"]
    }
    testable_cells = sum(1 for row in registered_rows if row["depth_is_necessary"])

    report = {
        "round": "4c -- four definitions on the decoy corridor",
        "owner": "Do Trong Hung (Workstream 3)",
        "registered_in": "artifacts/week2/ARM_REGISTRY.md, round 4c, 2026-09-06",
        "scope": (
            "Definition probe, not a benchmark result. The 120-MDP evaluation "
            "split is untouched (PLAN_Final.md sec 5)."
        ),
        "question": (
            "When an admissible (degree-0, Theorem 2) definition escapes "
            "saturation, does its k_dagger land where the corridor's mechanism "
            "actually is?"
        ),
        "prior_on_disk": {
            "note": (
                "'Does any definition escape' was already answered before this "
                "run and is not what is being tested here."
            ),
            "development_escapes_at_f_0.005": {
                "registered": "0/62",
                "materiality": "15/62",
                "max_state": "27/62",
                "probability": "16/62",
            },
            "source": "artifacts/week2/alternative_definitions_2026-09-01",
        },
        "ground_truth": {
            "k_settle_exact": (
                "min{k : greedy(Q^(k))(s) optimal at every corridor state}, "
                "true model and true behaviour policy"
            ),
            "category": (
                "oracle design check, same as round 4b's "
                "optimal_action_is_advance; never enters the estimator"
            ),
            "not_b1": (
                "This is not the exact evidence curve: no R=200 resampling, no "
                "sigma. Workstream 1 owns that."
            ),
            "distribution": dict(sorted(Counter(settles).items(), key=lambda kv: str(kv[0]))),
            "cells_where_depth_is_necessary": testable_cells,
            "cells_total": total,
            "round_4b_correction": (
                "Round 4b states the depth-0 ranking on the corridor is "
                "'provably wrong'; it verified pi*(s_0) = advance on 12/12, "
                "which is the other half of that claim. Measured here: depth 0 "
                f"is already optimal at every state on {total - testable_cells} "
                f"of {total} cells, because Q^(0) is the evaluation of an "
                "epsilon-greedy-advance behaviour policy that already completes "
                "the corridor. k_dagger = 0 is the correct answer there, so "
                "those cells carry no evidence about the definition."
            ),
        },
        "epsilon": config.epsilon,
        "k_max": config.k_max,
        "materiality_fractions": list(MATERIALITY_FRACTIONS),
        "buffer_seed": BUFFER_SEED,
        "behavior_epsilon": BEHAVIOR_EPSILON,
        "instances": len(REGISTERED_GRID),
        "pilot_multipliers": list(PILOT_MULTIPLIERS),
        "cells": total,
        "rows": len(rows),
        "round_4b_reproduction": _reproduces_round_4b(rows, probe_csv),
        "constant_predictor_baseline": constant_predictor_baseline(registered_rows),
        "argmin_attractor": argmin_attractor_diagnostic(rows),
        "by_variant": by_variant,
        "any_variant_escaped": bool(escaping),
        "escaping_variants": escaping,
        "runtime_seconds": round(time.time() - started, 1),
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--probe-csv",
        type=Path,
        default=ROOT / "artifacts/week2/depth_probe_2026-09-06/depth_probe.csv",
        help="Round 4b artifact the `registered` variant must reproduce.",
    )
    args = parser.parse_args()

    if args.output_dir is not None and args.output_dir.exists() and any(
        args.output_dir.iterdir()
    ):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )

    rows, report = run(OEHConfig.load(), args.probe_csv)

    reproduction = report["round_4b_reproduction"]
    if reproduction["checked"] and not reproduction["reproduced"]:
        raise SystemExit(
            "STOP (round 4c stop rule): the `registered` variant does not "
            "reproduce depth_probe.csv. Fix the cause before reading any other "
            f"number.\n  {reproduction['mismatches'][:5]}"
        )

    print(
        f"\nround 4b reproduction: "
        f"{'OK' if reproduction.get('reproduced') else 'NOT CHECKED'} "
        f"({reproduction.get('cells_compared', 0)} cells)"
    )
    ground = report["ground_truth"]
    print(f"k_settle_exact distribution: {ground['distribution']}")
    print(
        f"cells where depth is necessary: "
        f"{ground['cells_where_depth_is_necessary']}/{ground['cells_total']}"
    )
    print(f"\n{'variant@f':22} {'scr':>4} {'L>=e':>5} {'m>1':>4} {'int':>4} "
          f"{'ESC':>4} {'(spur)':>7} | {'need':>4} {'ESC':>4} {'ok':>3} | "
          f"{'|err|':>6} {'rho':>6}")
    for name, stats in report["by_variant"].items():
        median = stats["median_abs_depth_error"]
        rho = stats["spearman_k_dagger_vs_settle"]
        print(
            f"{name:22} {stats['scored']:4} {stats['loss_reaches_epsilon']:5} "
            f"{stats['multiplier_above_one']:4} {stats['interior_argmin']:4} "
            f"{stats['escaped']:4} {stats['escaped_spurious_empty_material_set']:7} | "
            f"{stats['depth_necessary_cells']:4} "
            f"{stats['depth_necessary_escaped']:4} {stats['depth_necessary_correct']:3} | "
            f"{'-' if median is None else f'{median:6.1f}'} "
            f"{'-' if rho is None else f'{rho:6.3f}'}"
        )

    if args.output_dir is not None:
        path = args.output_dir / "corridor_definitions.csv"
        write_csv_rows(path, rows, CORRIDOR_FIELDS)
        report["artifact_sha256"] = {"corridor_definitions.csv": sha256_file(path)}
        write_json(args.output_dir / "corridor_definitions_report.json", report)
        print(f"\nwrote {path.parent}")


if __name__ == "__main__":
    main()
