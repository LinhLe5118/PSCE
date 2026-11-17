#!/usr/bin/env python
"""K2 -- scale invariance: does ``sigma`` really shrink like ``N^{-1/2}``?

``PLAN_Final.md`` sec 8.1 registers K2 as *"N * sigma_Delta^2 does not depend
on N; compute at N0, 3*N0, 10*N0; error within a factor of 2"*, and sec 8.2
makes it success criterion 2. Sec 9's 18/08-21/08 milestone says to run it
immediately, in its own words because *"a failure here is a failure of the
formula, not noise"*. It was never run. This script runs it.

Why it matters more now than it did then. The project's central theoretical
result is Theorem 1 (``docs/DERIVATIONS.md`` sec 2), and the usable form of it
is Corollary 2.2:

    L_k(1) <= 0.169971 * sigma_bar    =>    escape requires sigma_bar >= 0.2942

Everything downstream leans on that number: the feasibility tool's
``blocked_by_theorem_1`` verdict, the positive control's design target, the
"missing by a factor of 24" headline, and the negative result's claim that the
saturation is structural rather than incidental. Corollary 2.2 *assumes*
``sigma = O(n^{-1/2})``, and ``docs/DERIVATIONS.md`` sec 9.3 records that the
assumption is standard bootstrap theory but had **never been checked
numerically on this benchmark's own pilots**. An assumption load-bearing for
the main claim and verified nowhere is the sort of thing a reviewer finds.

Two levels are measured, because they answer different questions and, as it
turns out, they answer them differently.

* **Per-state (K2 as sec 8.1 words it).** ``sigma_Delta`` is the bootstrap
  dispersion of one state's margin. The clean test is a *matched* one: take the
  states supported at every pilot size, hold the depth at ``k = 0`` where the
  margin has not yet been widened by propagation, and check whether
  ``N * sigma_k(s)^2`` is constant. This tests the formula.
* **Aggregate (what Corollary 2.2 consumes).** ``sigma_bar`` is the
  occupancy-weighted sum over states. It is what the ``0.2942`` bar is stated
  against, and it moves for two reasons at once -- each state's dispersion
  shrinks, while more state-actions clear the support threshold and enter the
  sum. Testing only the per-state level would leave the quantity the corollary
  actually uses unmeasured.

What is measured. The original 2026-09-05 run used prefixes at
``N0 in {2, 5, 10} * |S||A|`` -- the axis ``EXPERIMENT_RULE.md`` registers as
``ctrl-pilot``. Week 4 adds a separately pre-registered diagnostic grid
``{2, 5, 10, 20, 40}`` on five development MDPs, a 20-fold lever that does not
change the adaptive-pilot grid, reference ``N0=5``, or the old K2 verdict.
For each requested prefix the registered estimator's

    sigma_bar = max_k sum_s w_k(s) * sigma_k(s)

is recorded, the same quantity ``scripts/analysis/run_alternative_definitions.py`` writes as
``sigma_weighted_mean``. Under ``sigma ~ c * N^{-1/2}`` the product
``N * sigma_bar^2`` is constant in ``N``, so the test statistic is that
product's max/min ratio across the requested sizes, against K2's factor-of-2
threshold. The log-log slope of ``sigma_bar`` against ``N`` is reported
alongside; the prediction is ``-0.5``.

Scope. Development and OOD splits only. The runner refuses a manifest holding
an evaluation record rather than filtering it, exactly as the other runners do
(``EXPERIMENT_RULE.md`` rule 1).

Usage::

    python -m scripts.analysis.run_scale_invariance \\
      --dataset-dir datasets/week1_development_validation \\
      --output-dir artifacts/week2/scale_invariance_<date>

For the Week 4 handoff, add ``--multipliers 2,5,10,20,40`` and use
``datasets/k2_leverage_development_2026-09-07``. The runner accepts 20/40 only
when the manifest records the extended grid and pilot-only deviations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import transitions_from_arrays  # noqa: E402
from evaluation.io import verified_dataset_path, reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from oeh_estimator import OEHConfig, OEHResult, estimate_oeh  # noqa: E402
from oeh_validation import prefix_bootstrap_rng  # noqa: E402
from scripts.analysis.run_saturation_sweep import REGISTERED_PILOT_MULTIPLIERS, pilot_records  # noqa: E402

#: K2's tolerance (``PLAN_Final.md`` sec 8.1): "error within a factor of 2".
K2_MAX_FACTOR_ERROR = 2.0

#: The slope ``sigma ~ N^{-1/2}`` predicts for ``log sigma_bar`` vs ``log N``.
PREDICTED_LOG_SLOPE = -0.5

# Week 4 pre-registration widens only the K2 diagnostic lever.  It does not
# replace Appendix A's adaptive-pilot grid or the reference N0=5.
K2_LEVERAGE_MULTIPLIERS: tuple[int, ...] = (2, 5, 10, 20, 40)

ROW_FIELDS = (
    "mdp_id",
    "policy_family",
    "pilot_multiplier",
    "pilot_size",
    "oeh_status",
    "abstain_reason",
    "sigma_weighted_mean",
    "n_times_sigma_squared",
    "valid_bootstrap_fraction",
    "num_supported_states",
)

SUMMARY_FIELDS = (
    "mdp_id",
    "policy_family",
    "sizes_measured",
    "invariant_min",
    "invariant_max",
    "invariant_ratio",
    "log_slope",
    "within_factor_two",
    "verdict",
    "matched_states",
    "matched_pass_fraction",
    "matched_median_log_slope",
    "matched_verdict",
)

PER_STATE_FIELDS = (
    "mdp_id",
    "policy_family",
    "state",
    "depth",
    "sizes_measured",
    "sigma_by_size",
    "invariant_ratio",
    "log_slope",
    "within_factor_two",
)

#: Below this, a bootstrap ``sigma`` is a numerical zero rather than a small
#: measurement. Values of order ``1e-16`` show up at the smallest pilot budget
#: where a state sits at exactly the ``n_min = 2`` support floor and every
#: replicate reproduces the same margin. Feeding those into a ratio produces
#: figures like ``5e+28`` that say nothing about scaling, so such a state is
#: excluded from the matched test and counted under ``no_dispersion`` instead.
SIGMA_FLOOR = 1e-12

#: Depth at which the matched per-state test is taken. ``Q^(0)`` is the value
#: of the restricted behavior policy, so its margin has not yet been widened by
#: the ``k``-step propagation that ``PLAN_Final.md`` sec 3.5 says accumulates
#: variance. Comparing across ``N`` at a fixed, minimal depth isolates sampling
#: error from that documented depth effect.
MATCHED_DEPTH = 0


def validate_multiplier_request(
    manifest: Mapping[str, Any], multipliers: Sequence[int]
) -> tuple[int, ...]:
    requested = tuple(int(value) for value in multipliers)
    if not requested or tuple(sorted(set(requested))) != requested:
        raise ValueError("K2 multipliers must be unique and strictly increasing.")
    outside_week4 = sorted(set(requested) - set(K2_LEVERAGE_MULTIPLIERS))
    if outside_week4:
        raise ValueError(
            f"Pilot multipliers {outside_week4} are outside the pre-registered "
            f"K2 grids {list(K2_LEVERAGE_MULTIPLIERS)}."
        )
    available = {int(value) for value in manifest.get("pilot_size_multipliers", [])}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            f"Dataset manifest does not declare requested pilot multipliers {missing}; "
            f"available={sorted(available)}."
        )
    # A request using 20 or 40 is legal only on the Week 4 diagnostic bundle,
    # where the generator records both the extended grid and pilot-only stream
    # as explicit protocol deviations.
    if set(requested) - set(REGISTERED_PILOT_MULTIPLIERS):
        deviations = {
            str(item.get("setting")): item
            for item in manifest.get("protocol_deviations", [])
        }
        required = {"collection.pilot_size_multipliers", "collection.streams"}
        if not required.issubset(deviations):
            raise ValueError(
                "Extended K2 multipliers require the Week 4 generator deviations "
                "for pilot_size_multipliers and streams."
            )
        grid_deviation = deviations["collection.pilot_size_multipliers"]
        stream_deviation = deviations["collection.streams"]
        if (
            tuple(int(value) for value in grid_deviation.get("registered", ()))
            != REGISTERED_PILOT_MULTIPLIERS
            or tuple(int(value) for value in grid_deviation.get("effective", ()))
            != K2_LEVERAGE_MULTIPLIERS
            or tuple(str(value) for value in stream_deviation.get("registered", ()))
            != ("pilot", "training")
            or tuple(str(value) for value in stream_deviation.get("effective", ()))
            != ("pilot",)
        ):
            raise ValueError(
                "Extended K2 deviation values do not match the pre-registered "
                "Week 4 grid and pilot-only scope."
            )
    return requested


def sigma_weighted_mean(result: OEHResult) -> float | None:
    """``max_k sum_s w_k(s) * sigma_k(s)``.

    Identical to ``scripts/analysis/run_alternative_definitions.py``'s ``sigma_weighted_mean``
    so the two artifacts can be compared directly rather than approximately.
    ``None`` when the estimator abstained and there is no curve to take a
    maximum over -- an abstention is a missing measurement, and substituting
    zero would report perfect precision where there was none.
    """

    if result.abstained or result.occupancy_weights is None or result.sigmas is None:
        return None
    weights = np.asarray(result.occupancy_weights, dtype=np.float64)
    sigmas = np.nan_to_num(np.asarray(result.sigmas, dtype=np.float64), nan=0.0)
    return float(np.max((weights * sigmas).sum(axis=1)))


def per_state_sigmas(result: OEHResult, *, depth: int = MATCHED_DEPTH) -> dict[int, float]:
    """``sigma_k(s)`` at one depth, for states that carry a comparable margin.

    Restricted to states with at least two supported actions: with fewer, there
    is no ``(a_(1), a_(2))`` pair, ``PLAN_Final.md`` sec 3.2 drops the state
    from ``L_k`` entirely, and any ``sigma`` recorded for it is not the
    dispersion of a margin.
    """

    if result.abstained or result.sigmas is None:
        return {}
    sigmas = np.asarray(result.sigmas, dtype=np.float64)
    if depth >= sigmas.shape[0]:
        return {}
    return {
        state: float(sigmas[depth, state])
        for state, actions in enumerate(result.supported_actions)
        if len(actions) >= 2 and np.isfinite(sigmas[depth, state])
    }


def log_slope(sizes: Sequence[int], sigmas: Sequence[float]) -> float | None:
    """Least-squares slope of ``log sigma_bar`` against ``log N``.

    ``None`` if any measurement is non-positive: ``log 0`` is undefined, and a
    zero ``sigma_bar`` means a deterministic pilot where the scaling question
    does not arise rather than a slope of minus infinity.
    """

    if len(sizes) < 2 or any(value <= 0 for value in sigmas):
        return None
    xs = [math.log(size) for size in sizes]
    ys = [math.log(value) for value in sigmas]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def assess_pilot(measurements: Sequence[tuple[int, float | None]]) -> dict[str, Any]:
    """One pilot's K2 verdict from its ``(pilot_size, sigma_bar)`` pairs.

    Three outcomes, kept distinct because they mean different things:

    * ``passes`` / ``fails`` -- the invariant was measurable at two or more
      sizes and did, or did not, hold within a factor of two;
    * ``deterministic`` -- ``sigma_bar`` is exactly zero wherever it was
      measured. The invariant ``0 = 0`` holds trivially and carries no
      information about scaling, so counting it as a pass would inflate the
      pass rate with instances that never tested anything;
    * ``insufficient_measurements`` -- fewer than two sizes produced a curve.
    """

    usable = [(size, value) for size, value in measurements if value is not None]
    if len(usable) < 2:
        return {
            "sizes_measured": len(usable),
            "invariant_min": None,
            "invariant_max": None,
            "invariant_ratio": None,
            "log_slope": None,
            "within_factor_two": None,
            "verdict": "insufficient_measurements",
        }

    if all(value == 0.0 for _, value in usable):
        return {
            "sizes_measured": len(usable),
            "invariant_min": 0.0,
            "invariant_max": 0.0,
            "invariant_ratio": None,
            "log_slope": None,
            "within_factor_two": None,
            "verdict": "deterministic",
        }

    invariants = [size * value**2 for size, value in usable]
    smallest, largest = min(invariants), max(invariants)
    ratio = float("inf") if smallest == 0 else largest / smallest
    within = ratio <= K2_MAX_FACTOR_ERROR
    return {
        "sizes_measured": len(usable),
        "invariant_min": smallest,
        "invariant_max": largest,
        "invariant_ratio": ratio,
        "log_slope": log_slope([size for size, _ in usable], [value for _, value in usable]),
        "within_factor_two": within,
        "verdict": "passes" if within else "fails",
    }


def assess_matched_states(
    by_size: Mapping[int, Mapping[int, float]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """K2 on the states supported at *every* pilot size.

    Matching is the point. An unmatched comparison mixes the dispersion of a
    state measured at three sizes with the arrival of states that had no
    support at the smallest size, and the second effect can swamp the first at
    these budgets. Only states present throughout test the scaling law.
    """

    sizes = sorted(by_size)
    if len(sizes) < 2:
        return [], {
            "matched_states": 0,
            "no_dispersion_states": 0,
            "pass_fraction": None,
            "median_log_slope": None,
        }

    shared = set(by_size[sizes[0]])
    for size in sizes[1:]:
        shared &= set(by_size[size])

    rows: list[dict[str, Any]] = []
    slopes: list[float] = []
    passes = 0
    tested = 0
    no_dispersion = 0
    for state in sorted(shared):
        values = [by_size[size][state] for size in sizes]
        if any(value <= SIGMA_FLOOR for value in values):
            # No measurable dispersion at one of the sizes -- see SIGMA_FLOOR.
            no_dispersion += 1
            continue
        invariants = [size * value**2 for size, value in zip(sizes, values)]
        ratio = max(invariants) / min(invariants)
        within = ratio <= K2_MAX_FACTOR_ERROR
        slope = log_slope(sizes, values)
        tested += 1
        passes += int(within)
        if slope is not None:
            slopes.append(slope)
        rows.append(
            {
                "state": state,
                "depth": MATCHED_DEPTH,
                "sizes_measured": len(sizes),
                "sigma_by_size": ";".join(f"{size}:{value:.6g}" for size, value in zip(sizes, values)),
                "invariant_ratio": ratio,
                "log_slope": "" if slope is None else slope,
                "within_factor_two": within,
            }
        )
    return rows, {
        "matched_states": tested,
        "no_dispersion_states": no_dispersion,
        "pass_fraction": None if tested == 0 else passes / tested,
        "median_log_slope": None if not slopes else statistics.median(slopes),
    }


def run(
    dataset_dir: Path,
    *,
    bootstrap_seed: int,
    multipliers: Sequence[int],
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = pilot_records(manifest)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("The bundle holds no 10|S||A| adaptive pilot candidates.")

    multipliers = validate_multiplier_request(manifest, multipliers)

    config = OEHConfig.load()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    per_state_rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(records) * len(multipliers)
    done = 0

    for record in records:
        mdp_id = str(record["instance_id"])
        family = str(record["policy_family"])
        num_states = int(record["num_states"])
        num_actions = int(record["num_actions"])
        initial = np.asarray(record["initial_state_distribution"], dtype=np.float64)
        with np.load(verified_dataset_path(dataset_dir, record), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        transitions = transitions_from_arrays(arrays)

        measurements: list[tuple[int, float | None]] = []
        state_sigmas_by_size: dict[int, dict[int, float]] = {}
        for multiplier in multipliers:
            size = int(multiplier) * num_states * num_actions
            if len(transitions) < size:
                raise ValueError(
                    f"{mdp_id}/{family}: pilot buffer holds {len(transitions)} transitions, "
                    f"fewer than the {size} that the {multiplier}|S||A| prefix needs."
                )
            prefix = transitions[:size]
            result = estimate_oeh(
                prefix,
                initial,
                num_states=num_states,
                num_actions=num_actions,
                config=config,
                rng=prefix_bootstrap_rng(bootstrap_seed, len(prefix), num_states, num_actions),
            )
            sigma = sigma_weighted_mean(result)
            measurements.append((size, sigma))
            state_sigmas = per_state_sigmas(result)
            if state_sigmas:
                state_sigmas_by_size[size] = state_sigmas
            rows.append(
                {
                    "mdp_id": mdp_id,
                    "policy_family": family,
                    "pilot_multiplier": multiplier,
                    "pilot_size": size,
                    "oeh_status": "abstained" if result.abstained else ("censored" if result.censored_right else "observed"),
                    "abstain_reason": result.reason or "",
                    "sigma_weighted_mean": "" if sigma is None else sigma,
                    "n_times_sigma_squared": "" if sigma is None else size * sigma**2,
                    "valid_bootstrap_fraction": result.valid_bootstrap_fraction,
                    "num_supported_states": sum(1 for actions in result.supported_actions if actions),
                }
            )
            done += 1
            print(
                f"[{done}/{total}] {mdp_id} {family} N={size} "
                f"sigma_bar={'abstained' if sigma is None else f'{sigma:.6g}'} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

        matched_rows, matched = assess_matched_states(state_sigmas_by_size)
        for row in matched_rows:
            per_state_rows.append({"mdp_id": mdp_id, "policy_family": family, **row})
        summaries.append(
            {
                "mdp_id": mdp_id,
                "policy_family": family,
                **assess_pilot(measurements),
                "matched_states": matched["matched_states"],
                "matched_pass_fraction": (
                    "" if matched["pass_fraction"] is None else matched["pass_fraction"]
                ),
                "matched_median_log_slope": (
                    "" if matched["median_log_slope"] is None else matched["median_log_slope"]
                ),
                "_no_dispersion": matched["no_dispersion_states"],
                "matched_verdict": (
                    "insufficient_matched_states"
                    if matched["pass_fraction"] is None
                    else ("passes" if matched["pass_fraction"] >= 0.5 else "fails")
                ),
            }
        )

    matched_within = [bool(row["within_factor_two"]) for row in per_state_rows]
    matched_slopes = [float(row["log_slope"]) for row in per_state_rows if row["log_slope"] != ""]
    verdicts = [summary["verdict"] for summary in summaries]
    tested = [summary for summary in summaries if summary["verdict"] in {"passes", "fails"}]
    slopes = [summary["log_slope"] for summary in tested if summary["log_slope"] is not None]
    report = {
        "criterion": (
            "K2 (PLAN_Final.md sec 8.1): N * sigma_Delta^2 is independent of N, within a factor "
            f"of {K2_MAX_FACTOR_ERROR}. sigma_bar = max_k sum_s w_k(s) sigma_k(s), the same "
            "quantity scripts/analysis/run_alternative_definitions.py writes as sigma_weighted_mean."
        ),
        "why_it_matters": (
            "The finite-pilot sigma_bar >= 0.2942 necessary condition follows from the "
            "registered loss ceiling without a sample-size scaling assumption. K2 tests "
            "the additional dispersion-scaling hypothesis relevant to asymptotic saturation; "
            "matched state scaling and a changing weighted aggregate are distinct claims."
        ),
        "bootstrap_seed": bootstrap_seed,
        "pilot_multipliers": list(multipliers),
        "predicted_log_slope": PREDICTED_LOG_SLOPE,
        "protocol_status_at_run": manifest.get("protocol_status"),
        "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "pilots": len(summaries),
        "verdict_counts": {name: verdicts.count(name) for name in sorted(set(verdicts))},
        "pass_rate_among_tested": (
            None if not tested else sum(1 for s in tested if s["verdict"] == "passes") / len(tested)
        ),
        "log_slope": (
            None
            if not slopes
            else {
                "count": len(slopes),
                "min": min(slopes),
                "median": statistics.median(slopes),
                "max": max(slopes),
            }
        ),
        "matched_per_state": {
            "depth": MATCHED_DEPTH,
            "states_tested": len(matched_within),
            "states_without_measurable_dispersion": sum(
                int(summary.get("_no_dispersion", 0)) for summary in summaries
            ),
            "pass_fraction": None if not matched_within else sum(matched_within) / len(matched_within),
            "log_slope": (
                None
                if not matched_slopes
                else {
                    "count": len(matched_slopes),
                    "min": min(matched_slopes),
                    "median": statistics.median(matched_slopes),
                    "max": max(matched_slopes),
                }
            ),
            "note": (
                "States supported at every pilot size, at depth 0. This is K2 as sec 8.1 words "
                "it -- the dispersion of one state's margin -- with the confound of newly "
                "supported states removed by matching."
            ),
        },
        "interpretation_note": (
            "A 'deterministic' verdict is reported separately from 'passes' on purpose: a pilot "
            "whose sigma_bar is exactly zero satisfies the invariant as 0 = 0 and tests nothing. "
            "Folding those into the pass rate would inflate it with instances that never "
            "exercised the scaling."
        ),
    }
    return rows, summaries, per_state_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--multipliers", type=str, default="2,5,10")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    reserve_output_dir(args.output_dir)

    multipliers = [int(value) for value in args.multipliers.split(",") if value.strip()]
    rows, summaries, per_state_rows, report = run(
        args.dataset_dir,
        bootstrap_seed=args.bootstrap_seed,
        multipliers=multipliers,
        limit=args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "scale_invariance.csv", rows, ROW_FIELDS)
    write_csv_rows(args.output_dir / "scale_invariance_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv_rows(args.output_dir / "scale_invariance_per_state.csv", per_state_rows, PER_STATE_FIELDS)
    report["artifact_sha256"] = {
        name: sha256_file(args.output_dir / name)
        for name in (
            "scale_invariance.csv",
            "scale_invariance_summary.csv",
            "scale_invariance_per_state.csv",
        )
    }
    write_json(args.output_dir / "scale_invariance_report.json", report)

    print()
    print(f"  {report['pilots']} pilots. Verdicts: {report['verdict_counts']}")
    if report["pass_rate_among_tested"] is not None:
        print(f"  K2 pass rate among tested pilots: {report['pass_rate_among_tested']:.3f}")
    if report["log_slope"] is not None:
        slope = report["log_slope"]
        print(f"  aggregate sigma_bar log-log slope: median {slope['median']:.3f} (predicted {PREDICTED_LOG_SLOPE})")
    matched = report["matched_per_state"]
    if matched["pass_fraction"] is not None:
        print(
            f"  matched per-state (depth {matched['depth']}): {matched['states_tested']} states, "
            f"K2 pass fraction {matched['pass_fraction']:.3f}"
        )
    if matched["log_slope"] is not None:
        print(
            f"  matched per-state log-log slope: median {matched['log_slope']['median']:.3f} "
            f"(predicted {PREDICTED_LOG_SLOPE})"
        )
    print(f"  -> {args.output_dir}")


if __name__ == "__main__":
    main()
