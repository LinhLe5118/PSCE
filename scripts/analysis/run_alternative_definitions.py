"""Named alternative loss definitions, run in parallel with the registered one.

The saturation sweep returned 0/600 escapes and the audit proved 40/62 pilots
cannot escape at any ``sigma``.  This runner asks the remaining question: is
that a fact about offline RL, or about how ``L_k`` is written?

The structural result that motivates every variant here.  Since
``Delta = z sigma``, each term of the registered loss is
``w sigma (z Phi(-z))``, and ``max_{z>=0} z Phi(-z) = 0.16997`` at
``z* = 0.7518``.  Therefore::

    L_k(1) <= 0.16997 * sum_s w_k(s) sigma_k(s)

The action gaps cancel out of the ceiling entirely: the registered loss is
bounded by a constant times the occupancy-weighted bootstrap dispersion.  If
that aggregate vanishes with sample size (uniform ``O(1/sqrt(n))`` on fixed
support/depth is sufficient), the loss eventually falls below every fixed
positive tolerance.  This is a conditional statement about the statistic, not
a claim that a learner cannot benefit from more data or that the rate holds
uniformly on the data-dependent pipeline.

Four named definitions, all sharing the registered ``epsilon = 0.05``,
``K_max``, ``gamma`` and the registered adaptive pilot selection:

``registered``
    ``sum_s w Delta Phi(-z sqrt(mu))`` -- occupancy-weighted expected regret,
    reproduced here and checked against the estimator's own output so the
    variants are known to be computed on the same footing.
``materiality``
    ``sum_s w 1[Delta >= delta_min] Phi(-z sqrt(mu))`` -- the probability of
    picking a *materially* worse action, PAC-MDP's usual semantics.
    Dimensionless, so it is comparable across instances, and a tied action
    costs nothing and is therefore not counted as an error.
``max_state``
    ``max_{s: Delta >= delta_min} Phi(-z sqrt(mu))`` -- the same probability at
    the worst critical state instead of averaged.  Removes the occupancy
    dilution the audit measured as binding, at the cost of being driven by a
    single state.
``probability``
    ``sum_s w Phi(-z sqrt(mu))`` -- no materiality filter.  Included to show
    *why* the registered definition multiplies by ``Delta`` at all: without a
    filter, a state whose two actions are genuinely tied contributes ``w/2``
    forever and no amount of data removes it.

``epsilon`` is never lowered anywhere in this file -- every variant is scored
against the registered 0.05.  What changes is the functional, not the
threshold.

Honesty constraint, stated in code because it governs how the output may be
used: every definition here was written *after* seeing the registered one
saturate.  An escape under a new definition is therefore a hypothesis, not a
result.  It has to be pre-registered under its own name and run once on the
evaluation split (EXPERIMENT_RULE.md rule 8) before it means anything, and the
write-up has to say that the registered definition returned 0/600 and that the
alternative was designed afterwards.  This runner reports all four side by
side, never one alone.

Usage::

    python -m scripts.analysis.run_alternative_definitions \\
        --dataset-dir datasets/<bundle> \\
        --output-dir artifacts/week2/alternative_definitions_<date>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import verified_dataset_path, reserve_output_dir, sha256_file, write_csv_rows, write_json  # noqa: E402
from oeh_estimator import OEHConfig, OEHResult  # noqa: E402
from oeh_validation import oeh_status, run_pilot  # noqa: E402
from scripts.analysis.run_saturation_sweep import (  # noqa: E402
    MULTIPLIER_ESCAPE_ATOL,
    evaluate_escape,
    pilot_records,
    validate_arm_config,
)

#: ``max_{z>=0} z Phi(-z)``, attained at ``z* = 0.7518``.  See module docstring.
MAX_Z_PHI = 0.16997

#: A gap below this fraction of the observed value scale is not worth being
#: right about, so misidentifying it is not an error worth counting.
DEFAULT_MATERIALITY = 0.01

#: Multiplier search ceiling.  Beyond this the answer is "no finite amount of
#: data", which is reported as infinity rather than a large number.
MAX_MULTIPLIER = 2.0**40

VARIANTS: tuple[str, ...] = (
    "registered",
    "materiality",
    "max_state",
    "probability",
)

#: Variants whose loss depends on the materiality threshold, so they have to
#: be re-scored at every candidate delta.  The escape rate of a variant that
#: needs a new constant is worthless without the sensitivity of that constant.
MATERIALITY_VARIANTS: frozenset[str] = frozenset({"materiality", "max_state"})

CSV_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "structural_family",
    "variant",
    "materiality_fraction",
    "oeh_status",
    "epsilon",
    "max_loss_at_one",
    "loss_reaches_epsilon",
    "max_multiplier",
    "min_multiplier",
    "n0_effective_size",
    "n_req_hat",
    "k_dagger",
    "interior_argmin",
    "escaped",
    "material_states_at_k_dagger",
    "sigma_weighted_mean",
    "structural_bound",
    "value_scale",
)


@dataclass(frozen=True)
class Curve:
    """One variant's evidence curve over depth.

    ``min_multiplier`` is ``min_k m_k``, the quantity ``N_hat_req`` is built
    from (``PLAN_Final.md`` sec 3.5).  It is kept beside ``k_dagger`` rather
    than recovered from ``multipliers[k_dagger]`` by the caller because the two
    disagree when no depth has a finite multiplier: ``k_dagger`` then falls
    back to 0 while the minimum is genuinely infinite, and collapsing that
    distinction would report "depth 0 needs m_0 data" for an instance where no
    finite amount of data suffices at any depth.
    """

    losses_at_one: np.ndarray
    multipliers: np.ndarray
    k_dagger: int
    min_multiplier: float


def n_req_hat(min_multiplier: float, pilot_size: int) -> float:
    """``N_hat_req = N_0 * min_k m_k`` (``PLAN_Final.md`` sec 3.5).

    ``N_0`` is the *effective* pilot size the adaptive selector settled on, not
    the registered ``n0_coefficient``, because that is the budget the curve was
    actually computed from.  An infinite minimum propagates: no finite amount
    of data suffices, which is a definite answer and not a missing one.
    """

    if pilot_size <= 0:
        raise ValueError("pilot_size must be a positive transition count.")
    if math.isnan(min_multiplier):
        raise ValueError("min_multiplier must be a number or infinity, not NaN.")
    if min_multiplier < 1.0:
        raise ValueError(
            "min_multiplier cannot be below the registered mu >= 1 floor."
        )
    return pilot_size * min_multiplier


def _multiplier(loss: Callable[[float], float], epsilon: float) -> float:
    """``min{mu >= 1 : loss(mu) <= epsilon}``, or ``inf`` if unreachable.

    Same search the estimator runs, generalised over the loss functional so the
    variants are scored exactly the way the registered definition is.
    """

    if loss(1.0) <= epsilon:
        return 1.0
    upper = 2.0
    while loss(upper) > epsilon:
        upper *= 2.0
        if upper > MAX_MULTIPLIER:
            return math.inf
    return float(brentq(lambda value: loss(value) - epsilon, 1.0, upper))


def _curve(
    weights: np.ndarray,
    deltas: np.ndarray,
    z_scores: np.ndarray,
    *,
    variant: str,
    epsilon: float,
    materiality: float,
) -> Curve:
    """Build one variant's loss and multiplier at every depth."""

    depths = weights.shape[0]
    losses = np.zeros(depths)
    multipliers = np.zeros(depths)
    material = deltas >= materiality
    for depth in range(depths):
        row_w = weights[depth]
        row_d = deltas[depth]
        row_z = z_scores[depth]
        row_material = material[depth]

        def loss(scale: float, row_w=row_w, row_d=row_d, row_z=row_z, row_material=row_material) -> float:
            tail = norm.cdf(-row_z * math.sqrt(scale))
            if variant == "registered":
                return float(np.sum(row_w * row_d * tail))
            if variant == "materiality":
                return float(np.sum(row_w * row_material * tail))
            if variant == "probability":
                return float(np.sum(row_w * tail))
            if variant == "max_state":
                eligible = row_material & (row_w > 0.0)
                return float(tail[eligible].max()) if eligible.any() else 0.0
            raise ValueError(f"Unknown variant {variant!r}.")

        losses[depth] = loss(1.0)
        multipliers[depth] = _multiplier(loss, epsilon)
    finite = multipliers[np.isfinite(multipliers)]
    if finite.size:
        minimum = float(finite.min())
        k_dagger = int(np.flatnonzero(np.isclose(multipliers, minimum, rtol=0.0, atol=1e-12))[0])
    else:
        minimum = math.inf
        k_dagger = 0
    return Curve(
        losses_at_one=losses,
        multipliers=multipliers,
        k_dagger=k_dagger,
        min_multiplier=minimum,
    )


def value_scale(result: OEHResult, arrays: Mapping[str, np.ndarray], gamma: float) -> float:
    """The data-only value scale materiality is measured against.

    Preferred: the spread of ``Q_hat`` over supported pairs, taken at the depth
    where it is widest.  Under sparse rewards ``r_max/(1-gamma)`` overstates the
    real value range by one to two orders of magnitude -- it says 20 where the
    estimated values span 0.5 -- which would push the materiality threshold
    above every gap in the instance and silently empty the sum.  It is kept only
    as the fallback for a result that carries no ``value_range``.

    The benchmark's own ``dynamic_range`` is deliberately not used: it is
    derived from ``V*`` and may never enter a data-only quantity.
    """

    if result.value_range is not None and np.any(result.value_range > 0.0):
        return float(np.max(result.value_range))
    rewards = np.asarray(arrays["reward"], dtype=np.float64)
    observed = float(np.max(np.abs(rewards))) if rewards.size else 0.0
    return observed / (1.0 - gamma)


def variant_rows(
    mdp_id: str,
    policy_family: str,
    structural_family: str,
    result: OEHResult,
    config: OEHConfig,
    *,
    scale: float,
    pilot_size: int,
    materiality_fractions: Sequence[float],
) -> list[dict[str, Any]]:
    """One row per (variant, materiality threshold) for a single pilot.

    Scale-free variants do not depend on the threshold, so they are scored
    once and reported against the first fraction; the materiality variants
    are re-scored at every candidate so the sensitivity of the new constant
    is visible next to the escape counts it produces.
    """

    status = oeh_status(result)
    if (
        result.deltas is None
        or result.z_scores is None
        or result.occupancy_weights is None
        or result.sigmas is None
    ):
        return [
            {
                "mdp_id": mdp_id,
                "policy_family": policy_family,
                "structural_family": structural_family,
                "variant": variant,
                "materiality_fraction": materiality_fractions[0],
                "oeh_status": status,
                "epsilon": config.epsilon,
                "max_loss_at_one": "",
                "loss_reaches_epsilon": False,
                "max_multiplier": "",
                "min_multiplier": "",
                "n0_effective_size": pilot_size,
                "n_req_hat": "",
                "k_dagger": "",
                "interior_argmin": False,
                "escaped": False,
                "material_states_at_k_dagger": "",
                "sigma_weighted_mean": "",
                "structural_bound": "",
                "value_scale": scale,
            }
            for variant in VARIANTS
        ]

    weights = np.asarray(result.occupancy_weights, dtype=np.float64)
    deltas = np.nan_to_num(np.asarray(result.deltas, dtype=np.float64), nan=0.0)
    z_scores = np.nan_to_num(np.asarray(result.z_scores, dtype=np.float64), nan=0.0)
    sigmas = np.nan_to_num(np.asarray(result.sigmas, dtype=np.float64), nan=0.0)
    sigma_weighted = float(np.max((weights * sigmas).sum(axis=1)))
    bound = MAX_Z_PHI * sigma_weighted

    combinations = [
        (variant, fraction)
        for variant in VARIANTS
        for fraction in (
            materiality_fractions
            if variant in MATERIALITY_VARIANTS
            else materiality_fractions[:1]
        )
    ]
    rows: list[dict[str, Any]] = []
    for variant, materiality_fraction in combinations:
        materiality = materiality_fraction * scale
        curve = _curve(
            weights,
            deltas,
            z_scores,
            variant=variant,
            epsilon=config.epsilon,
            materiality=materiality,
        )
        verdict = evaluate_escape(
            curve.losses_at_one,
            curve.multipliers,
            curve.k_dagger,
            epsilon=config.epsilon,
            k_max=config.k_max,
        )
        material_count = int(
            np.sum((deltas[curve.k_dagger] >= materiality) & (weights[curve.k_dagger] > 0.0))
        )
        rows.append(
            {
                "mdp_id": mdp_id,
                "policy_family": policy_family,
                "structural_family": structural_family,
                "variant": variant,
                "materiality_fraction": materiality_fraction,
                "oeh_status": status,
                "epsilon": config.epsilon,
                "max_loss_at_one": verdict.max_loss_at_one,
                "loss_reaches_epsilon": verdict.loss_reaches_epsilon,
                "max_multiplier": verdict.max_multiplier,
                "min_multiplier": curve.min_multiplier,
                "n0_effective_size": pilot_size,
                "n_req_hat": n_req_hat(curve.min_multiplier, pilot_size),
                "k_dagger": curve.k_dagger,
                "interior_argmin": verdict.interior_argmin,
                "escaped": verdict.escaped,
                "material_states_at_k_dagger": material_count,
                "sigma_weighted_mean": sigma_weighted,
                "structural_bound": bound,
                "value_scale": scale,
            }
        )
    return rows


def _check_registered_matches_estimator(
    rows: Sequence[Mapping[str, Any]], result: OEHResult
) -> bool:
    """The ``registered`` variant must reproduce the estimator's own loss.

    If it does not, every other variant is being computed on a different
    footing and none of the comparisons below mean anything.
    """

    if result.losses_at_one is None:
        return True
    row = next(row for row in rows if row["variant"] == "registered")
    if row["max_loss_at_one"] == "":
        return True
    return math.isclose(
        float(row["max_loss_at_one"]),
        float(np.nanmax(result.losses_at_one)),
        rel_tol=1e-9,
        abs_tol=1e-15,
    )


def run(
    dataset_dir: Path,
    *,
    bootstrap_seed: int,
    materiality_fractions: Sequence[float],
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = pilot_records(manifest)
    if limit is not None:
        records = records[:limit]
    config = validate_arm_config(OEHConfig.load())
    families = {
        str(item["instance_id"]): str(item["family"])
        for item in manifest.get("instances", [])
    }

    rows: list[dict[str, Any]] = []
    bound_violations: list[dict[str, Any]] = []
    registered_mismatches = 0
    started = time.time()

    for index, record in enumerate(records, start=1):
        mdp_id = str(record["instance_id"])
        policy_family = str(record["policy_family"])
        with np.load(verified_dataset_path(dataset_dir, record), allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        pilot = run_pilot(
            arrays,
            np.asarray(record["initial_state_distribution"], dtype=np.float64),
            num_states=int(record["num_states"]),
            num_actions=int(record["num_actions"]),
            bootstrap_seed=bootstrap_seed,
        )
        result = pilot.selected_result
        scale = value_scale(result, arrays, config.gamma)
        pilot_rows = variant_rows(
            mdp_id,
            policy_family,
            families.get(mdp_id, ""),
            result,
            config,
            scale=scale,
            pilot_size=int(pilot.selection.n0_effective_size),
            materiality_fractions=materiality_fractions,
        )
        if not _check_registered_matches_estimator(pilot_rows, result):
            registered_mismatches += 1
        registered = next(row for row in pilot_rows if row["variant"] == "registered")
        if (
            registered["max_loss_at_one"] != ""
            and registered["structural_bound"] != ""
            and float(registered["max_loss_at_one"])
            > float(registered["structural_bound"]) + 1e-12
        ):
            bound_violations.append(
                {
                    "mdp_id": mdp_id,
                    "policy_family": policy_family,
                    "loss": registered["max_loss_at_one"],
                    "bound": registered["structural_bound"],
                }
            )
        rows.extend(pilot_rows)
        escaped = sorted(row["variant"] for row in pilot_rows if row["escaped"])
        print(
            f"[{index}/{len(records)}] {mdp_id} {policy_family}: "
            f"escaped={escaped or 'none'} "
            f"sigma_w={registered['sigma_weighted_mean']} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    report = _summarize(
        rows,
        manifest=manifest,
        dataset_dir=dataset_dir,
        config=config,
        bootstrap_seed=bootstrap_seed,
        materiality_fractions=materiality_fractions,
        bound_violations=bound_violations,
        registered_mismatches=registered_mismatches,
        runtime=time.time() - started,
    )
    return rows, report


def _stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value != "" and math.isfinite(float(value))
    ]
    if not finite:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    dataset_dir: Path,
    config: OEHConfig,
    bootstrap_seed: int,
    materiality_fractions: Sequence[float],
    bound_violations: Sequence[Mapping[str, Any]],
    registered_mismatches: int,
    runtime: float,
) -> dict[str, Any]:
    per_variant: dict[str, Any] = {}
    keys = [
        (variant, fraction)
        for variant in VARIANTS
        for fraction in (
            materiality_fractions
            if variant in MATERIALITY_VARIANTS
            else materiality_fractions[:1]
        )
    ]
    for variant, fraction in keys:
        label = (
            f"{variant}@delta={fraction}"
            if variant in MATERIALITY_VARIANTS
            else variant
        )
        subset = [
            row
            for row in rows
            if row["variant"] == variant
            and row["materiality_fraction"] == fraction
        ]
        scored = [row for row in subset if row["max_loss_at_one"] != ""]
        per_variant[label] = {
            "variant": variant,
            "materiality_fraction": fraction,
            "pilots": len(subset),
            "pilots_scored": len(scored),
            "escape_counts": {
                "loss_reaches_epsilon": sum(1 for row in scored if row["loss_reaches_epsilon"]),
                "multiplier_above_one": sum(
                    1
                    for row in scored
                    if row["max_multiplier"] != ""
                    and float(row["max_multiplier"]) > 1.0 + MULTIPLIER_ESCAPE_ATOL
                ),
                "interior_argmin": sum(1 for row in scored if row["interior_argmin"]),
                "all_three": sum(1 for row in scored if row["escaped"]),
            },
            "k_dagger_distribution": dict(
                sorted(Counter(row["k_dagger"] for row in scored).items())
            ),
            "max_loss_at_one": _stats([row["max_loss_at_one"] for row in scored]),
            # N_hat_req is the quantity C2 scores against N_emp, so its own
            # dispersion and its infinite-count are reported next to the escape
            # counts rather than left to be recomputed from the CSV.
            "min_multiplier": _stats([row["min_multiplier"] for row in scored]),
            "n_req_hat": _stats([row["n_req_hat"] for row in scored]),
            "n_req_hat_infinite": sum(
                1
                for row in scored
                if row["n_req_hat"] != "" and not math.isfinite(float(row["n_req_hat"]))
            ),
            "escapes_by_family": dict(
                sorted(
                    Counter(
                        row["structural_family"] for row in scored if row["escaped"]
                    ).items()
                )
            ),
        }
    registered = [row for row in rows if row["variant"] == "registered" and row["max_loss_at_one"] != ""]
    return {
        "status": (
            "exploratory; alternative definitions were written after the "
            "registered one was seen to saturate, so an escape here is a "
            "hypothesis to pre-register, not a result (EXPERIMENT_RULE.md "
            "rules 2, 7 and 8)"
        ),
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "protocol_status_at_run": manifest.get("protocol_status"),
        "bootstrap_seed": bootstrap_seed,
        "epsilon": config.epsilon,
        "epsilon_note": "identical for every variant; never lowered",
        "materiality_fractions": list(materiality_fractions),
        "materiality_sensitivity_note": (
            "materiality and max_state need a threshold the registered "
            "definition does not; their escape counts are reported at every "
            "candidate delta so the sensitivity to that new constant is "
            "visible rather than hidden behind one chosen value."
        ),
        "runtime_seconds": round(runtime, 1),
        "structural_bound": {
            "statement": (
                "Delta = z sigma, so each term is w sigma (z Phi(-z)) and "
                f"max_z z Phi(-z) = {MAX_Z_PHI}. Hence L_k(1) <= "
                f"{MAX_Z_PHI} * sum_s w_k(s) sigma_k(s): the action gaps "
                "cancel out of the ceiling entirely."
            ),
            "consequence": (
                "sigma = O(1/sqrt(n)), so the registered loss shrinks as the "
                "pilot grows and m_k = 1 becomes more certain with more data. "
                "A statistic meant to report how much more data is needed "
                "cannot do so once its own loss vanishes with data."
            ),
            "sigma_weighted_mean": _stats(
                [row["sigma_weighted_mean"] for row in registered]
            ),
            "bound_value": _stats([row["structural_bound"] for row in registered]),
            "sigma_required_for_escape": config.epsilon / MAX_Z_PHI,
            "violations": list(bound_violations[:10]),
            "bound_holds_everywhere": not bound_violations,
        },
        "registered_reproduces_estimator": registered_mismatches == 0,
        "registered_mismatches": registered_mismatches,
        "variants": per_variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score named alternative loss definitions beside the registered one."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument(
        "--materiality-fractions",
        type=lambda value: [
            float(item) for item in value.split(",") if item.strip()
        ],
        default=[0.005, DEFAULT_MATERIALITY, 0.02],
        help=(
            "A gap below this fraction of the observed value scale is treated "
            "as a tie, so misidentifying it is not counted as an error."
        ),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.materiality_fractions or not all(
        0.0 <= fraction < 1.0 for fraction in args.materiality_fractions
    ):
        raise SystemExit("--materiality-fractions must all lie in [0, 1).")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )

    rows, report = run(
        args.dataset_dir,
        bootstrap_seed=args.bootstrap_seed,
        materiality_fractions=args.materiality_fractions,
        limit=args.limit,
    )
    import os
    report['numerical_backend'] = os.environ.get('DAORL_OEH_BACKEND','reference')
    report['acceleration_source_sha256'] = sha256_file(ROOT / 'code/tabular_acceleration.py')
    report['gpu_scientific_equivalence_validated'] = False
    path = args.output_dir / "alternative_definitions.csv"
    write_csv_rows(path, rows, CSV_FIELDS)
    report["artifact_sha256"] = {path.name: sha256_file(path)}
    write_json(args.output_dir / "alternative_definitions_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
