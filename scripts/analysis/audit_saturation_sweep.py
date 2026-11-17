"""Audit of a saturation sweep run: is the null result real, or an artifact?

``scripts/analysis/run_saturation_sweep.py`` reports *whether* an instance escapes saturation.
This runner asks the follow-up question an exploratory negative has to answer
before it is believed (EXPERIMENT_RULE.md rule 9): is the pipeline measuring a
property of the offline-RL problem, or is it measuring a degenerate setup?

Three independent checks, all data-only:

1. **Is the environment deterministic?**  Straight off the pilot buffers: how
   many distinct successors and how much reward spread each observed ``(s,a)``
   pair actually shows.  A pair whose successor and reward never vary makes
   every bootstrap replicate agree, so ``sigma_hat = 0``, ``z = +inf`` and
   ``Phi(-z sqrt(mu)) = 0`` exactly -- that state contributes *nothing* to
   ``L_k`` no matter how large its action gap is.  If the development split is
   mostly deterministic, a flat evidence curve says more about the benchmark
   than about the estimator.

2. **Could any amount of uncertainty reach epsilon?**  ``L_k(mu) = sum_s
   w_k(s) Delta_k(s) Phi(-z_k(s) sqrt(mu))`` is a weighted average with
   ``sum_s w_k(s) <= 1``, so it never exceeds its largest single term.  That
   gives two bounds, neither of which needs the weights the estimator does not
   return:

   * ``gap_bound = max_s Delta_k(s) / 2`` -- also the largest ``Phi`` can be.
     Under ``epsilon`` it proves that no stochasticity axis, pilot size or
     bootstrap scheme can escape: the action gaps themselves are too small.
   * ``reweighting_bound = max_{k,s} Delta_k(s) Phi(-z_k(s))`` -- the measured
     ``sigma``, best-case weights.  Under ``epsilon`` it rules out the ``w_gap``
     exclusion mechanism as the binding cause, because even handing the entire
     occupancy weight to the single most informative state falls short.

   Since the estimator now reports ``occupancy_weights``, a third and *exact*
   quantity replaces both as the decisive one:

   * ``exact_ceiling = max_k (1/2) sum_s w_k(s) Delta_k(s)`` -- the limit of
     ``L_k`` when every eligible state becomes maximally uncertain.  Below
     ``epsilon`` it proves no amount of environment stochasticity escapes on
     that instance; the two bounds above are kept because they separate the
     causes (gaps too small vs. exclusion not binding).

   This is the decisive check: its answer does not depend on which arm is run
   next.  Note the direction -- a value *below* ``epsilon`` proves
   impossibility, while a value above it merely fails to rule an escape out.

3. **Is the result stable across bootstrap seeds?**  One seeded run is one
   realisation.  Replicating the registered baseline over several declared
   seeds shows whether the null is a property of the data or an accident of a
   single draw.  Every seed is declared on the command line and every seed's
   result is reported; nothing is selected across them (EXPERIMENT_RULE.md
   rule 6 forbids choosing a seed after seeing results, not measuring spread).

The audit also re-derives the sweep's own baseline cell and compares it with
``sweep_results.csv``, so a corrupted or stale artifact cannot pass unnoticed.

Nothing exact (``P``, ``R``, ``V*``, ``Q*``, returns, the structural-family
label) is read as an input here; the family label is copied into the output
table for later scoring only, exactly as EXPERIMENT_RULE.md rule 3 permits.

Usage::

    python -m scripts.analysis.audit_saturation_sweep \\
        --dataset-dir datasets/<bundle> \\
        --sweep-dir artifacts/week2/saturation_sweep_<date> \\
        --output-dir artifacts/week2/saturation_audit_<date>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from evaluation.io import (  # noqa: E402
    read_csv_rows,
    sha256_file,
    write_csv_rows,
    write_json,
)
from oeh_estimator import OEHConfig, OEHResult  # noqa: E402
from oeh_validation import oeh_status, run_pilot  # noqa: E402
from scripts.analysis.run_saturation_sweep import (  # noqa: E402
    MULTIPLIER_ESCAPE_ATOL,
    evaluate_escape,
    pilot_records,
    validate_arm_config,
)

#: ``Phi(-z sqrt(mu)) <= Phi(0) = 1/2`` for every ``z >= 0`` and ``mu >= 1``.
MAX_TAIL_PROBABILITY = 0.5

#: A bootstrap spread this small is zero for our purposes: it is the scale on
#: which ``Q_hat`` differences are pure floating-point noise, not evidence.
SIGMA_ZERO_ATOL = 1e-12

STOCHASTICITY_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "structural_family",
    "num_states",
    "num_actions",
    "num_transitions",
    "num_trajectories",
    "observed_sa_pairs",
    "repeated_sa_pairs",
    "deterministic_successor_fraction",
    "mean_distinct_successors",
    "mean_successor_entropy_bits",
    "constant_reward_fraction",
    "mean_reward_std",
    "fully_deterministic",
)

CEILING_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "structural_family",
    "oeh_status",
    "epsilon",
    "max_delta",
    "gap_bound",
    "escape_ruled_out_by_gap_bound",
    "reweighting_bound",
    "escape_ruled_out_by_reweighting_bound",
    "exact_ceiling",
    "escape_ruled_out_by_exact_ceiling",
    "sigma_inflation_to_reach_epsilon",
    "weight_invariant_ok",
    "max_loss_at_one",
    "loss_headroom_ratio",
    "bound_consistent_with_measured_loss",
    "eligible_states_at_k_dagger",
    "zero_sigma_fraction",
    "median_z_score",
    "infinite_z_fraction",
)

REPLICATION_FIELDS: tuple[str, ...] = (
    "mdp_id",
    "policy_family",
    "bootstrap_seed",
    "oeh_status",
    "max_loss_at_one",
    "max_multiplier",
    "k_dagger",
    "escaped",
    "mean_gap_occupancy",
)


# --------------------------------------------------------------------------
# Check 1 -- how stochastic is the data the estimator actually sees?
# --------------------------------------------------------------------------


def data_stochasticity(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Successor and reward variability per observed ``(s,a)``, from raw data.

    Only pairs observed at least twice can show variability at all, so the
    fractions below are over those pairs; reporting them over all observed
    pairs would let a thin pilot masquerade as a deterministic environment.
    """

    states = np.asarray(arrays["state"], dtype=np.int64)
    actions = np.asarray(arrays["action"], dtype=np.int64)
    next_states = np.asarray(arrays["next_state"], dtype=np.int64)
    rewards = np.asarray(arrays["reward"], dtype=np.float64)
    trajectories = np.asarray(arrays["trajectory_id"], dtype=np.int64)

    pairs = np.unique(np.stack([states, actions], axis=1), axis=0)
    distinct_successors: list[int] = []
    entropies: list[float] = []
    reward_stds: list[float] = []
    repeated = 0
    deterministic = 0
    constant_reward = 0
    for state, action in pairs:
        selected = (states == state) & (actions == action)
        if int(selected.sum()) < 2:
            continue
        repeated += 1
        _, counts = np.unique(next_states[selected], return_counts=True)
        probabilities = counts / counts.sum()
        distinct_successors.append(int(len(counts)))
        entropies.append(
            float(-np.sum(probabilities * np.log2(probabilities)))
            if len(counts) > 1
            else 0.0
        )
        if len(counts) == 1:
            deterministic += 1
        spread = float(np.std(rewards[selected], ddof=0))
        reward_stds.append(spread)
        if spread <= SIGMA_ZERO_ATOL:
            constant_reward += 1

    deterministic_fraction = deterministic / repeated if repeated else None
    constant_fraction = constant_reward / repeated if repeated else None
    return {
        "num_transitions": int(states.size),
        "num_trajectories": int(np.unique(trajectories).size),
        "observed_sa_pairs": int(len(pairs)),
        "repeated_sa_pairs": repeated,
        "deterministic_successor_fraction": deterministic_fraction,
        "mean_distinct_successors": (
            float(np.mean(distinct_successors)) if distinct_successors else None
        ),
        "mean_successor_entropy_bits": (
            float(np.mean(entropies)) if entropies else None
        ),
        "constant_reward_fraction": constant_fraction,
        "mean_reward_std": float(np.mean(reward_stds)) if reward_stds else None,
        "fully_deterministic": bool(
            repeated > 0
            and deterministic == repeated
            and constant_reward == repeated
        ),
    }


# --------------------------------------------------------------------------
# Check 2 -- the ceiling the loss can never exceed
# --------------------------------------------------------------------------


def loss_ceiling(result: OEHResult, config: OEHConfig) -> dict[str, Any]:
    """Two upper bounds on ``L_k``, neither of which needs the weights.

    Both start from ``L_k(mu) = sum_s w_k(s) Delta_k(s) Phi(-z_k(s) sqrt(mu))``
    with ``Delta >= 0``, ``w >= 0`` and ``sum_s w_k(s) <= 1``, so the sum is at
    most its largest single term:

    * ``gap_bound = max_s Delta_k(s) / 2`` also takes ``Phi(-z sqrt(mu)) <=
      Phi(0) = 1/2``, i.e. the most uncertainty any ``sigma`` could ever
      express.  Below ``epsilon`` it *proves* no stochasticity, pilot-size or
      bootstrap axis can escape on this reward scale.
    * ``reweighting_bound = max_{k,s} Delta_k(s) Phi(-z_k(s))`` keeps the
      measured ``sigma`` and only moves the occupancy weight: it is the best
      ``L_k`` any redistribution of ``w`` could reach.  Below ``epsilon`` it
      rules out the ``w_gap`` exclusion mechanism as the binding cause, because
      even handing the whole weight to the single most informative state would
      not get there.

    The estimator's own ``L_k`` is never recomputed here -- it is read from
    ``losses_at_one`` -- and ``bound_consistent_with_measured_loss`` checks the
    bounds really do dominate it, so a bound derived from the wrong array
    cannot pass silently.
    """

    empty = {
        "max_delta": None,
        "gap_bound": None,
        "escape_ruled_out_by_gap_bound": None,
        "reweighting_bound": None,
        "escape_ruled_out_by_reweighting_bound": None,
        "max_loss_at_one": None,
        "loss_headroom_ratio": None,
        "bound_consistent_with_measured_loss": None,
        "eligible_states_at_k_dagger": None,
        "zero_sigma_fraction": None,
        "median_z_score": None,
        "infinite_z_fraction": None,
    }
    if result.deltas is None or result.z_scores is None:
        return empty
    deltas = np.asarray(result.deltas, dtype=np.float64)
    sigmas = np.asarray(result.sigmas, dtype=np.float64)
    z_scores = np.asarray(result.z_scores, dtype=np.float64)
    eligible = np.isfinite(deltas)
    max_delta = float(np.nanmax(deltas)) if eligible.any() else None
    gap_bound = None if max_delta is None else MAX_TAIL_PROBABILITY * max_delta
    terms = np.where(
        eligible,
        np.nan_to_num(deltas) * norm.cdf(-np.nan_to_num(z_scores, nan=0.0)),
        0.0,
    )
    reweighting_bound = float(terms.max()) if eligible.any() else None

    depth = int(result.k_dagger) if result.k_dagger is not None else 0
    row_eligible = eligible[depth]
    row_sigmas = sigmas[depth][row_eligible]
    row_z = z_scores[depth][row_eligible]
    positive_gap = deltas[depth][row_eligible] > 0.0
    finite_z = row_z[np.isfinite(row_z)]
    max_loss = (
        float(np.nanmax(result.losses_at_one))
        if result.losses_at_one is not None
        else None
    )
    return {
        "max_delta": max_delta,
        "gap_bound": gap_bound,
        "escape_ruled_out_by_gap_bound": bool(
            gap_bound is not None and gap_bound < config.epsilon
        ),
        "reweighting_bound": reweighting_bound,
        "escape_ruled_out_by_reweighting_bound": bool(
            reweighting_bound is not None and reweighting_bound < config.epsilon
        ),
        "max_loss_at_one": max_loss,
        "loss_headroom_ratio": (
            None
            if max_loss is None or not max_loss > 0.0
            else float(config.epsilon / max_loss)
        ),
        "bound_consistent_with_measured_loss": bool(
            max_loss is None
            # No state anywhere has two supported actions, so L_k is an empty
            # sum -- exactly 0, with no bound to compare it against. That is a
            # coverage outcome, not a violated inequality.
            or (gap_bound is None and reweighting_bound is None and max_loss == 0.0)
            or (
                reweighting_bound is not None
                and gap_bound is not None
                and max_loss <= reweighting_bound + 1e-12
                and max_loss <= gap_bound + 1e-12
            )
        ),
        "eligible_states_at_k_dagger": int(row_eligible.sum()),
        "zero_sigma_fraction": (
            float(np.mean(np.nan_to_num(row_sigmas) <= SIGMA_ZERO_ATOL))
            if row_sigmas.size
            else None
        ),
        "median_z_score": (
            float(np.median(finite_z)) if finite_z.size else None
        ),
        "infinite_z_fraction": (
            float(np.mean(np.isinf(row_z) & positive_gap))
            if row_z.size
            else None
        ),
    }


def exact_ceiling(result: OEHResult, config: OEHConfig) -> dict[str, Any]:
    """The exact ceiling on ``L_k``, now that ``w_k(s)`` is reported.

    ``gap_bound`` and ``reweighting_bound`` are upper bounds because the
    occupancy weights used to be unavailable.  With ``occupancy_weights`` on the
    result they stop being necessary:

    * ``exact_ceiling = max_k (1/2) sum_s w_k(s) Delta_k(s)`` is what ``L_k``
      converges to if *every* eligible state became maximally uncertain
      (``z -> 0``, ``Phi -> 1/2``).  Below ``epsilon`` this is a proof, not a
      bound: no stochasticity of any kind escapes on this instance.
    * ``sigma_inflation_to_reach_epsilon`` is the smallest ``c >= 1`` with
      ``max_k sum_s w_k(s) Delta_k(s) Phi(-z_k(s)/c) >= epsilon`` -- i.e. how
      many times larger the measured ``sigma_hat`` would have to be.  It is
      deliberately multiplicative, so a state with ``sigma_hat = 0`` exactly
      stays at zero however large ``c`` grows; that is why it is reported next
      to ``exact_ceiling`` rather than instead of it.

    ``weight_invariant_ok`` checks the estimator's own bookkeeping: at every
    depth the reported weights plus ``w_gap`` must sum to one, unless no
    critical state exists at all and both are zero.
    """

    empty = {
        "exact_ceiling": None,
        "escape_ruled_out_by_exact_ceiling": None,
        "sigma_inflation_to_reach_epsilon": None,
        "weight_invariant_ok": None,
    }
    if (
        result.occupancy_weights is None
        or result.deltas is None
        or result.z_scores is None
        or result.gap_occupancy is None
    ):
        return empty
    weights = np.asarray(result.occupancy_weights, dtype=np.float64)
    deltas = np.nan_to_num(np.asarray(result.deltas, dtype=np.float64), nan=0.0)
    z_scores = np.nan_to_num(
        np.asarray(result.z_scores, dtype=np.float64), nan=0.0
    )
    gap = np.asarray(result.gap_occupancy, dtype=np.float64)

    totals = weights.sum(axis=1)
    invariant = bool(
        np.all(
            np.isclose(totals + gap, 1.0, atol=1e-9)
            | (np.isclose(totals, 0.0, atol=1e-12) & np.isclose(gap, 0.0, atol=1e-12))
        )
    )
    ceiling = MAX_TAIL_PROBABILITY * float((weights * deltas).sum(axis=1).max())

    def loss_at(scale: float, depth: int) -> float:
        return float(
            np.sum(
                weights[depth]
                * deltas[depth]
                * norm.cdf(-z_scores[depth] / scale)
            )
        )

    inflation: float | None = None
    for depth in range(weights.shape[0]):
        if loss_at(1.0, depth) >= config.epsilon:
            inflation = 1.0
            break
        # Only states with a finite z can be moved by scaling sigma at all.
        finite = np.isfinite(z_scores[depth])
        limit = MAX_TAIL_PROBABILITY * float(
            np.sum(weights[depth][finite] * deltas[depth][finite])
        )
        if limit < config.epsilon:
            continue
        upper = 2.0
        while loss_at(upper, depth) < config.epsilon:
            upper *= 2.0
            if not np.isfinite(upper):
                break
        if not np.isfinite(upper):
            continue
        found = float(
            brentq(lambda value: loss_at(value, depth) - config.epsilon, 1.0, upper)
        )
        inflation = found if inflation is None else min(inflation, found)
    return {
        "exact_ceiling": ceiling,
        "escape_ruled_out_by_exact_ceiling": bool(ceiling < config.epsilon),
        "sigma_inflation_to_reach_epsilon": inflation,
        "weight_invariant_ok": invariant,
    }

# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    """Undefined diagnostics are empty cells, never NaN or a stand-in number."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _column(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row[field] not in ("", None)]


def run(
    dataset_dir: Path,
    *,
    sweep_dir: Path | None,
    seeds: Sequence[int],
    limit: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = pilot_records(manifest)
    if limit is not None:
        records = records[:limit]
    config = validate_arm_config(OEHConfig.load())
    families = {
        str(item["instance_id"]): str(item["family"])
        for item in manifest.get("instances", [])
    }

    stochasticity: list[dict[str, Any]] = []
    ceilings: list[dict[str, Any]] = []
    replication: list[dict[str, Any]] = []
    started = time.time()

    for index, record in enumerate(records, start=1):
        mdp_id = str(record["instance_id"])
        family = str(record["policy_family"])
        num_states = int(record["num_states"])
        num_actions = int(record["num_actions"])
        initial = np.asarray(record["initial_state_distribution"], dtype=np.float64)
        with np.load(dataset_dir / record["file"], allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}

        measured = data_stochasticity(arrays)
        stochasticity.append(
            {
                "mdp_id": mdp_id,
                "policy_family": family,
                "structural_family": families.get(mdp_id, ""),
                "num_states": num_states,
                "num_actions": num_actions,
                **{key: _cell(value) for key, value in measured.items()},
            }
        )

        for seed in seeds:
            result = run_pilot(
                arrays,
                initial,
                num_states=num_states,
                num_actions=num_actions,
                bootstrap_seed=seed,
            ).selected_result
            verdict = evaluate_escape(
                result.losses_at_one,
                result.multipliers,
                result.k_dagger,
                epsilon=config.epsilon,
                k_max=config.k_max,
            )
            replication.append(
                {
                    "mdp_id": mdp_id,
                    "policy_family": family,
                    "bootstrap_seed": seed,
                    "oeh_status": oeh_status(result),
                    "max_loss_at_one": _cell(verdict.max_loss_at_one),
                    "max_multiplier": _cell(verdict.max_multiplier),
                    "k_dagger": _cell(verdict.k_dagger),
                    "escaped": verdict.escaped,
                    "mean_gap_occupancy": _cell(
                        float(np.mean(result.gap_occupancy))
                        if result.gap_occupancy is not None
                        else None
                    ),
                }
            )
            if seed == seeds[0]:
                ceiling = {
                    **loss_ceiling(result, config),
                    **exact_ceiling(result, config),
                }
                ceilings.append(
                    {
                        "mdp_id": mdp_id,
                        "policy_family": family,
                        "structural_family": families.get(mdp_id, ""),
                        "oeh_status": oeh_status(result),
                        "epsilon": config.epsilon,
                        **{key: _cell(value) for key, value in ceiling.items()},
                    }
                )

        print(
            f"[{index}/{len(records)}] {mdp_id} {family}: "
            f"det_successor={measured['deterministic_successor_fraction']} "
            f"gap_bound={ceilings[-1]['gap_bound']} "
            f"reweight_bound={ceilings[-1]['reweighting_bound']} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    tables = {
        "audit_data_stochasticity": stochasticity,
        "audit_loss_ceiling": ceilings,
        "audit_seed_replication": replication,
    }
    report = _summarize(
        tables,
        manifest=manifest,
        dataset_dir=dataset_dir,
        sweep_dir=sweep_dir,
        seeds=seeds,
        config=config,
        runtime=time.time() - started,
    )
    return tables, report


def _cross_check_sweep(
    sweep_dir: Path, ceilings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Re-derive the sweep's baseline cell and compare it with the artifact.

    The baseline cell of the ``nmin`` arm is the registered configuration, so
    the audit's own baseline run must reproduce it.  A mismatch means the CSV
    on disk does not correspond to the code that is being audited.
    """

    path = sweep_dir / "sweep_results.csv"
    if not path.exists():
        return {"skipped": f"{path} not found"}
    rows = read_csv_rows(path)
    baseline = {
        (row["mdp_id"], row["policy_family"]): row
        for row in rows
        if row["arm_id"] == "nmin" and row["axis_value"] == "2"
    }
    audited = {
        (row["mdp_id"], row["policy_family"]): row for row in ceilings
    }
    compared = 0
    mismatches: list[dict[str, Any]] = []
    for key, row in audited.items():
        other = baseline.get(key)
        if other is None or other["max_loss_at_one"] == "" or row["max_loss_at_one"] == "":
            continue
        compared += 1
        if not math.isclose(
            float(other["max_loss_at_one"]),
            float(row["max_loss_at_one"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            mismatches.append(
                {
                    "mdp_id": key[0],
                    "policy_family": key[1],
                    "sweep": other["max_loss_at_one"],
                    "audit": row["max_loss_at_one"],
                }
            )
    return {
        "sweep_results_sha256": sha256_file(path),
        "rows_compared": compared,
        "mismatches": mismatches[:10],
        "reproduces_sweep_baseline": compared > 0 and not mismatches,
    }


def _summarize(
    tables: Mapping[str, list[dict[str, Any]]],
    *,
    manifest: Mapping[str, Any],
    dataset_dir: Path,
    sweep_dir: Path | None,
    seeds: Sequence[int],
    config: OEHConfig,
    runtime: float,
) -> dict[str, Any]:
    stochasticity = tables["audit_data_stochasticity"]
    ceilings = tables["audit_loss_ceiling"]
    replication = tables["audit_seed_replication"]

    by_family: dict[str, dict[str, Any]] = {}
    for row in stochasticity:
        family = str(row["structural_family"])
        bucket = by_family.setdefault(
            family, {"pilots": 0, "fully_deterministic": 0, "successor": [], "reward": []}
        )
        bucket["pilots"] += 1
        bucket["fully_deterministic"] += int(bool(row["fully_deterministic"]))
        if row["deterministic_successor_fraction"] != "":
            bucket["successor"].append(float(row["deterministic_successor_fraction"]))
        if row["mean_reward_std"] != "":
            bucket["reward"].append(float(row["mean_reward_std"]))
    family_summary = {
        family: {
            "pilots": bucket["pilots"],
            "fully_deterministic_pilots": bucket["fully_deterministic"],
            "deterministic_successor_fraction": _stats(bucket["successor"]),
            "mean_reward_std": _stats(bucket["reward"]),
        }
        for family, bucket in sorted(by_family.items())
    }

    spread: list[float] = []
    unstable = 0
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in replication:
        grouped.setdefault((row["mdp_id"], row["policy_family"]), []).append(row)
    for rows in grouped.values():
        losses = [float(row["max_loss_at_one"]) for row in rows if row["max_loss_at_one"] != ""]
        if len(losses) > 1:
            spread.append(max(losses) - min(losses))
        if len({row["escaped"] for row in rows}) > 1:
            unstable += 1

    return {
        "status": (
            "exploratory audit; EXPERIMENT_RULE.md rule 9 -- reported at the "
            "same detail a positive result would be"
        ),
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "protocol_status_at_run": manifest.get("protocol_status"),
        "pilots_audited": len(stochasticity),
        "bootstrap_seeds": list(seeds),
        "runtime_seconds": round(runtime, 1),
        "baseline_config": {
            "epsilon": config.epsilon,
            "n_min": config.n_min,
            "k_max": config.k_max,
            "num_bootstrap": config.num_bootstrap,
            "gamma": config.gamma,
        },
        "check_1_environment_determinism": {
            "question": (
                "Does the pilot data show any successor or reward variability? "
                "A pair that never varies gives sigma_hat=0, z=+inf and a zero "
                "contribution to L_k regardless of its action gap."
            ),
            "fully_deterministic_pilots": sum(
                1 for row in stochasticity if row["fully_deterministic"]
            ),
            "deterministic_successor_fraction": _stats(
                _column(stochasticity, "deterministic_successor_fraction")
            ),
            "mean_successor_entropy_bits": _stats(
                _column(stochasticity, "mean_successor_entropy_bits")
            ),
            "constant_reward_fraction": _stats(
                _column(stochasticity, "constant_reward_fraction")
            ),
            "by_structural_family": family_summary,
        },
        "check_2_loss_ceiling": {
            "question": (
                "Can L_k reach epsilon at all -- under the most favourable "
                "sigma (gap_bound), and under the most favourable occupancy "
                "weights at the measured sigma (reweighting_bound)?"
            ),
            "epsilon": config.epsilon,
            "pilots": len(ceilings),
            "pilots_escape_ruled_out_by_gap_bound": sum(
                1 for row in ceilings if row["escape_ruled_out_by_gap_bound"] is True
            ),
            "pilots_escape_ruled_out_by_reweighting_bound": sum(
                1
                for row in ceilings
                if row["escape_ruled_out_by_reweighting_bound"] is True
            ),
            "pilots_escape_ruled_out_by_exact_ceiling": sum(
                1 for row in ceilings if row["escape_ruled_out_by_exact_ceiling"] is True
            ),
            "pilots_with_exact_ceiling": sum(
                1 for row in ceilings if row["exact_ceiling"] != ""
            ),
            "exact_ceiling": _stats(_column(ceilings, "exact_ceiling")),
            "sigma_inflation_to_reach_epsilon": _stats(
                _column(ceilings, "sigma_inflation_to_reach_epsilon")
            ),
            "pilots_reachable_by_sigma_inflation": sum(
                1
                for row in ceilings
                if row["sigma_inflation_to_reach_epsilon"] not in ("", None)
            ),
            "estimator_weight_invariant_holds": all(
                row["weight_invariant_ok"] is not False for row in ceilings
            ),
            "gap_bound": _stats(_column(ceilings, "gap_bound")),
            "reweighting_bound": _stats(_column(ceilings, "reweighting_bound")),
            "max_delta": _stats(_column(ceilings, "max_delta")),
            "max_loss_at_one": _stats(_column(ceilings, "max_loss_at_one")),
            "loss_headroom_ratio": _stats(
                _column(ceilings, "loss_headroom_ratio")
            ),
            "zero_sigma_fraction_at_k_dagger": _stats(
                _column(ceilings, "zero_sigma_fraction")
            ),
            "infinite_z_fraction_at_k_dagger": _stats(
                _column(ceilings, "infinite_z_fraction")
            ),
            "bounds_dominate_measured_loss": all(
                row["bound_consistent_with_measured_loss"] is not False
                for row in ceilings
            ),
            "interpretation": (
                "exact_ceiling is the decisive quantity now that the estimator "
                "reports w_k(s): it is what L_k converges to if every eligible "
                "state became maximally uncertain, so below epsilon it proves "
                "no amount of environment stochasticity escapes on that "
                "instance. gap_bound and reweighting_bound are the weaker "
                "bounds that were computable before, kept because they "
                "separate the causes: gap_bound below epsilon means the action "
                "gaps are too small, reweighting_bound below epsilon means the "
                "w_gap exclusion mechanism is not the binding cause. "
                "sigma_inflation_to_reach_epsilon says how many times larger "
                "sigma_hat would have to be; it is multiplicative, so a state "
                "at sigma_hat=0 exactly can never be moved by it. A bound at "
                "or above epsilon rules nothing out."
            ),
        },
        "check_3_seed_replication": {
            "question": (
                "Is the null a property of the data or of one bootstrap draw?"
            ),
            "seeds": list(seeds),
            "instances_with_seed_dependent_escape": unstable,
            "max_loss_spread_across_seeds": _stats(spread),
            "escape_counts_by_seed": {
                str(seed): sum(
                    1
                    for row in replication
                    if row["bootstrap_seed"] == seed and row["escaped"]
                )
                for seed in seeds
            },
            "status_counts_by_seed": {
                str(seed): dict(
                    sorted(
                        Counter(
                            row["oeh_status"]
                            for row in replication
                            if row["bootstrap_seed"] == seed
                        ).items()
                    )
                )
                for seed in seeds
            },
            "note": (
                "Every declared seed is reported; none is selected. "
                "EXPERIMENT_RULE.md rule 6."
            ),
        },
        "check_4_sweep_artifact_cross_check": (
            _cross_check_sweep(sweep_dir, ceilings)
            if sweep_dir is not None
            else {"skipped": "no --sweep-dir supplied"}
        ),
        "multiplier_escape_atol": MULTIPLIER_ESCAPE_ATOL,
    }


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a saturation sweep for degenerate-setup artifacts."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Sweep artifact directory to cross-check against.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_ints,
        default=[20260824, 20260825, 20260826],
        help=(
            "Declared bootstrap seeds for the replication check; the first is "
            "the sweep's registered seed. All are reported, none selected."
        ),
    )
    parser.add_argument("--limit", type=int, help="Only the first N pilots.")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )
    if not args.seeds:
        raise SystemExit("--seeds must name at least one seed.")

    tables, report = run(
        args.dataset_dir,
        sweep_dir=args.sweep_dir,
        seeds=args.seeds,
        limit=args.limit,
    )

    fields = {
        "audit_data_stochasticity": STOCHASTICITY_FIELDS,
        "audit_loss_ceiling": CEILING_FIELDS,
        "audit_seed_replication": REPLICATION_FIELDS,
    }
    written: dict[str, str] = {}
    for name, rows in tables.items():
        path = args.output_dir / f"{name}.csv"
        write_csv_rows(path, rows, fields[name])
        written[path.name] = sha256_file(path)
    report["artifact_sha256"] = written
    write_json(args.output_dir / "saturation_audit_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
