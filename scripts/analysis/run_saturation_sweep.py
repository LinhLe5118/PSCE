"""Exploratory sweep: does any registered axis let ``L_k(1)`` reach ``epsilon``?

Measured state of play (PROTOCOL_DECISIONS_2026-08-16.md D6-D8): on the 15-MDP
development split every one of the 12,019 evidence-curve points has
``m_k = 1.0`` exactly, because every measured ``L_k(1)`` already sits under
``epsilon`` (max 0.0152, 3.3x below).  The evidence search therefore never
engages and the depth axis carries no information.  The diagnosed mechanism is
that a state with fewer than two supported actions is *excluded* from the
``L_k`` sum and its occupancy is booked into ``w_gap(k)`` instead, so shrinking
the pilot removes uncertain states from the sum rather than raising it.

This runner measures that mechanism instead of arguing about it.  Each arm is
the registered baseline with exactly one axis changed (EXPERIMENT_RULE.md
rule 4), every arm is written out whether it escapes or not (rule 5, rule 9),
and the escape criterion is the one fixed before the first run (rule 2):

    max_k L_k(1) >= epsilon   and   max_k m_k > 1 + 1e-9   and   0 < k† < K_max

All three conditions are also reported separately, so a partial escape is
visible rather than collapsed into a single ``False``.

Scope and refusals, from EXPERIMENT_RULE.md rule 1 and rule 3:

* development and OOD bundles only -- a manifest carrying any ``evaluation``
  record is refused outright, never silently filtered;
* nothing exact (``P``, ``R``, ``V*``, ``Q*``, returns, the structural-family
  label) is read as an estimator input here;
* ``epsilon`` can never go below 0.05.  Lowering it would manufacture the
  escape being tested for, which is exactly the post-hoc threshold selection
  PLAN sec 11 rule 2 forbids; the check is duplicated at the CLI boundary and
  inside the escape criterion so no code path can route around it.

This is an exploratory instrument.  Nothing it prints becomes a result before
EXPERIMENT_RULE.md rule 8 has been walked end to end.

Usage::

    python -m scripts.analysis.run_saturation_sweep --dataset-dir datasets/<bundle> \\
        --output-dir artifacts/saturation_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from adaptive_pilot import (  # noqa: E402
    select_effective_pilot,
    transitions_from_arrays,
)
from evaluation.io import sha256_file, write_csv_rows, write_json  # noqa: E402
from oeh_estimator import (  # noqa: E402
    OEHConfig,
    OEHResult,
    TransitionLike,
    estimate_oeh,
)
from oeh_validation import (  # noqa: E402
    oeh_status,
    prefix_bootstrap_rng,
    run_pilot,
)

# --------------------------------------------------------------------------
# Registered grid (EXPERIMENT_RULE.md rule 3 and the round-1 arm table)
# --------------------------------------------------------------------------

#: Hard floor on ``epsilon``.  Not a default, not a tunable: see module docstring.
EPSILON_FLOOR = 0.05

#: Values Appendix A already registers as permitted robustness settings.
REGISTERED_N_MIN: tuple[int, ...] = (2, 3, 5)
REGISTERED_EPSILON: tuple[float, ...] = (0.05, 0.10)
REGISTERED_PILOT_MULTIPLIERS: tuple[int, ...] = (2, 5, 10)

#: Float slack for comparing a configured ``epsilon`` against the registry.
EPSILON_ATOL = 1e-12

#: ``m_k`` is pinned at the ``mu >= 1`` floor; this is the "actually moved" gap.
MULTIPLIER_ESCAPE_ATOL = 1e-9

#: Splits this runner is allowed to touch while the evaluation split is sealed.
PERMITTED_SPLITS = frozenset({"development", "ood"})


@dataclass(frozen=True)
class Arm:
    """One sweep arm: the baseline with exactly one axis varied."""

    arm_id: str
    axis: str
    values: tuple[Any, ...]


ARMS: dict[str, Arm] = {
    "ctrl-pilot": Arm("ctrl-pilot", "pilot_prefix_multiplier", (2, 5, 10)),
    "nmin": Arm("nmin", "n_min", (2, 3, 5)),
    "epsilon-robust": Arm("epsilon-robust", "epsilon", (0.05, 0.10)),
}

CSV_FIELDS: tuple[str, ...] = (
    "arm_id",
    "axis",
    "axis_value",
    "mdp_id",
    "policy_family",
    "oeh_status",
    "abstain_reason",
    "max_loss_at_one",
    "epsilon",
    "loss_reaches_epsilon",
    "max_multiplier",
    "k_dagger",
    "interior_argmin",
    "escaped",
    "mean_gap_occupancy",
    "gap_occupancy_at_k_dagger",
    "valid_bootstrap_fraction",
    "num_supported_states",
    "num_single_action_states",
)


# --------------------------------------------------------------------------
# Protocol guards
# --------------------------------------------------------------------------


def check_epsilon(epsilon: float) -> float:
    """Refuse any ``epsilon`` outside the registered set, floor first.

    The floor is checked before membership so that a request for a smaller
    ``epsilon`` is answered with the reason it is forbidden rather than a
    generic "not in the grid": ``m_k = min{mu >= 1 : L_k(mu) <= epsilon}``, so
    lowering ``epsilon`` fabricates the escape this sweep is measuring.
    """

    value = float(epsilon)
    if not math.isfinite(value) or value < EPSILON_FLOOR - EPSILON_ATOL:
        raise ValueError(
            f"epsilon={epsilon!r} is below the hard floor {EPSILON_FLOOR}. "
            "Lowering epsilon manufactures an escape instead of measuring one "
            "(EXPERIMENT_RULE.md rule 3); this is refused unconditionally."
        )
    if not any(
        math.isclose(value, registered, rel_tol=0.0, abs_tol=EPSILON_ATOL)
        for registered in REGISTERED_EPSILON
    ):
        raise ValueError(
            f"epsilon={epsilon!r} is not a registered value "
            f"{REGISTERED_EPSILON}; an unregistered constant needs its own "
            "named definition run alongside the registered one."
        )
    return value


def check_n_min(n_min: int) -> int:
    """Only Appendix A's registered robustness values for ``n_min``."""

    if isinstance(n_min, bool) or not isinstance(n_min, int):
        raise ValueError(f"n_min must be an integer, got {n_min!r}.")
    if n_min not in REGISTERED_N_MIN:
        raise ValueError(
            f"n_min={n_min} is not a registered value {REGISTERED_N_MIN}."
        )
    return n_min


def check_pilot_multiplier(multiplier: int) -> int:
    """Only the registered ``N0`` grid ``{2,5,10}|S||A|``."""

    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        raise ValueError(
            f"pilot multiplier must be an integer, got {multiplier!r}."
        )
    if multiplier not in REGISTERED_PILOT_MULTIPLIERS:
        raise ValueError(
            f"pilot multiplier={multiplier} is not a registered value "
            f"{REGISTERED_PILOT_MULTIPLIERS}."
        )
    return multiplier


def validate_arm_config(config: OEHConfig) -> OEHConfig:
    """Gate every config an arm hands to the estimator through the registry."""

    check_epsilon(config.epsilon)
    check_n_min(config.n_min)
    return config


def pilot_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``10|S||A|`` pilot candidates, after refusing a sealed bundle.

    The evaluation split is sealed until EXPERIMENT_RULE.md rule 8 is walked;
    touching it ends the exploratory phase irreversibly.  A manifest that
    contains such a record is therefore an error to raise, not a set to filter:
    quietly dropping it would make an exploratory run look development-only
    while it was pointed at the sealed bundle.
    """

    splits = {
        str(item.get("benchmark_split"))
        for item in manifest["datasets"]
    } | {
        str(item["split"])
        for item in manifest.get("instances", [])
        if item.get("split") is not None
    }
    if "evaluation" in splits:
        raise ValueError(
            "This manifest contains evaluation records. The evaluation split "
            "is sealed for exploratory runs (EXPERIMENT_RULE.md rule 1); "
            "refusing rather than filtering it out."
        )
    unknown = splits - PERMITTED_SPLITS
    if unknown:
        raise ValueError(
            f"Unrecognised benchmark splits {sorted(unknown)}; expected only "
            f"{sorted(PERMITTED_SPLITS)}."
        )
    return [
        dict(item)
        for item in manifest["datasets"]
        if item["stream"] == "pilot"
        and item["budget_role"] == "adaptive_pilot_candidate"
        and item["size_multiplier"] == 10
    ]


# --------------------------------------------------------------------------
# The escape criterion, fixed before the first run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EscapeVerdict:
    """The three conditions separately, plus their conjunction.

    Kept separate on purpose: an instance whose loss finally reaches
    ``epsilon`` but whose ``k†`` sits on a boundary is a different measurement
    from one that never moved at all, and rule 9 asks for negative results at
    the same resolution as positive ones.
    """

    max_loss_at_one: float | None
    max_multiplier: float | None
    k_dagger: int | None
    loss_reaches_epsilon: bool
    multiplier_above_one: bool
    interior_argmin: bool
    escaped: bool


def _finite_max(values: Sequence[float] | None) -> float | None:
    """Maximum ignoring NaN, keeping ``inf``; ``None`` when nothing is defined.

    ``inf`` is a real outcome of the multiplier search -- the bracket never
    came back under ``epsilon`` -- so it must survive to the caller rather than
    be flattened into "undefined" like a NaN.
    """

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    defined = array[~np.isnan(array)]
    if defined.size == 0:
        return None
    return float(defined.max())


def evaluate_escape(
    losses_at_one: Sequence[float] | None,
    multipliers: Sequence[float] | None,
    k_dagger: int | None,
    *,
    epsilon: float,
    k_max: int,
) -> EscapeVerdict:
    """Apply the pre-registered escape criterion to one estimator result.

    An abstained result carries no curve, so it cannot escape; that is a
    recorded non-escape, not a missing measurement.
    """

    check_epsilon(epsilon)
    max_loss = _finite_max(losses_at_one)
    max_multiplier = _finite_max(multipliers)
    loss_reaches_epsilon = max_loss is not None and max_loss >= epsilon
    multiplier_above_one = (
        max_multiplier is not None
        and max_multiplier > 1.0 + MULTIPLIER_ESCAPE_ATOL
    )
    interior_argmin = k_dagger is not None and 0 < k_dagger < k_max
    return EscapeVerdict(
        max_loss_at_one=max_loss,
        max_multiplier=max_multiplier,
        k_dagger=k_dagger,
        loss_reaches_epsilon=loss_reaches_epsilon,
        multiplier_above_one=multiplier_above_one,
        interior_argmin=interior_argmin,
        escaped=bool(
            loss_reaches_epsilon and multiplier_above_one and interior_argmin
        ),
    )


# --------------------------------------------------------------------------
# Running one arm on one pilot buffer
# --------------------------------------------------------------------------


def arm_config(arm_id: str, axis_value: Any, base_config: OEHConfig) -> OEHConfig:
    """The config an arm hands the estimator: baseline with one field changed."""

    if arm_id == "ctrl-pilot":
        check_pilot_multiplier(axis_value)
        config = base_config
    elif arm_id == "nmin":
        config = replace(base_config, n_min=check_n_min(axis_value))
    elif arm_id == "epsilon-robust":
        config = replace(base_config, epsilon=check_epsilon(axis_value))
    else:
        raise ValueError(f"Unknown arm_id {arm_id!r}; known arms: {sorted(ARMS)}.")
    return validate_arm_config(config)


def run_arm_value(
    arm_id: str,
    axis_value: Any,
    arrays: Mapping[str, Any],
    initial_state_distribution: Sequence[float],
    *,
    num_states: int,
    num_actions: int,
    base_config: OEHConfig,
    bootstrap_seed: int,
) -> tuple[OEHResult, OEHConfig]:
    """Run one (arm, axis value) cell on one pilot buffer.

    ``ctrl-pilot`` replaces the adaptive selector with a fixed prefix, which is
    its single change from the baseline; the other arms keep the registered
    selector and change one estimator constant.  The bootstrap generator is
    rebuilt with ``prefix_bootstrap_rng`` in every path so a cell is a function
    of ``(bootstrap_seed, prefix length, |S|, |A|)`` alone and reruns reproduce.
    """

    config = arm_config(arm_id, axis_value, base_config)
    transitions = transitions_from_arrays(arrays)
    initial = np.asarray(initial_state_distribution, dtype=np.float64)

    if arm_id == "ctrl-pilot":
        size = int(axis_value) * num_states * num_actions
        if len(transitions) < size:
            raise ValueError(
                f"Pilot buffer holds {len(transitions)} transitions, fewer "
                f"than the {size} the {axis_value}|S||A| prefix needs."
            )
        prefix = transitions[:size]
        result = estimate_oeh(
            prefix,
            initial,
            num_states=num_states,
            num_actions=num_actions,
            config=config,
            rng=prefix_bootstrap_rng(
                bootstrap_seed, len(prefix), num_states, num_actions
            ),
        )
        return result, config

    if config == OEHConfig.load():
        # The unchanged baseline goes through the registered runner itself, so
        # the reference cell of every arm is the registered code path and not a
        # local re-derivation of it.  ``run_pilot`` loads the registered config
        # internally, which is why the comparison is against that config rather
        # than against whatever ``base_config`` a caller passed in.
        run = run_pilot(
            arrays,
            initial,
            num_states=num_states,
            num_actions=num_actions,
            bootstrap_seed=bootstrap_seed,
        )
        return run.selected_result, config

    def assess_prefix(
        prefix: Sequence[TransitionLike],
        prefix_initial: np.ndarray,
        states: int,
        actions: int,
    ) -> OEHResult:
        return estimate_oeh(
            prefix,
            prefix_initial,
            num_states=states,
            num_actions=actions,
            config=config,
            rng=prefix_bootstrap_rng(
                bootstrap_seed, len(prefix), states, actions
            ),
        )

    selection = select_effective_pilot(
        transitions,
        initial,
        num_states=num_states,
        num_actions=num_actions,
        assess_prefix=assess_prefix,
    )
    return selection.result, config


# --------------------------------------------------------------------------
# Rows and summaries
# --------------------------------------------------------------------------


def _cell(value: Any) -> Any:
    """Undefined diagnostics are empty cells, never NaN and never a stand-in.

    ``inf`` is deliberately *not* blanked here: an infinite multiplier is a
    measured outcome that satisfies the second escape condition, so hiding it
    would hide an escape.
    """

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _support_counts(result: OEHResult) -> tuple[int, int]:
    """States with any supported action, and states with exactly one.

    The second count is the diagnosed mechanism made countable: a single-action
    state contributes no ``Delta`` and is booked into ``w_gap`` instead of the
    ``L_k`` sum.
    """

    supported = result.supported_actions
    with_support = sum(1 for row in supported if len(row) >= 1)
    single = sum(1 for row in supported if len(row) == 1)
    return with_support, single


def sweep_row(
    arm_id: str,
    axis: str,
    axis_value: Any,
    mdp_id: str,
    policy_family: str,
    result: OEHResult,
    config: OEHConfig,
) -> dict[str, Any]:
    """One CSV row: the escape verdict plus the diagnostics that explain it."""

    verdict = evaluate_escape(
        result.losses_at_one,
        result.multipliers,
        result.k_dagger,
        epsilon=config.epsilon,
        k_max=config.k_max,
    )
    gap = result.gap_occupancy
    mean_gap = float(np.mean(gap)) if gap is not None else None
    gap_at_k = (
        float(gap[result.k_dagger])
        if gap is not None and result.k_dagger is not None
        else None
    )
    with_support, single_action = _support_counts(result)
    return {
        "arm_id": arm_id,
        "axis": axis,
        "axis_value": axis_value,
        "mdp_id": mdp_id,
        "policy_family": policy_family,
        "oeh_status": oeh_status(result),
        "abstain_reason": result.reason or "",
        "max_loss_at_one": _cell(verdict.max_loss_at_one),
        "epsilon": config.epsilon,
        "loss_reaches_epsilon": verdict.loss_reaches_epsilon,
        "max_multiplier": _cell(verdict.max_multiplier),
        "k_dagger": _cell(verdict.k_dagger),
        "interior_argmin": verdict.interior_argmin,
        "escaped": verdict.escaped,
        "mean_gap_occupancy": _cell(mean_gap),
        "gap_occupancy_at_k_dagger": _cell(gap_at_k),
        "valid_bootstrap_fraction": _cell(result.valid_bootstrap_fraction),
        "num_supported_states": with_support,
        "num_single_action_states": single_action,
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    """Count/min/median/max, with ``None`` rather than a stand-in when empty."""

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


def _numeric(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row[field] != ""]


def summarize_cell(
    rows: Sequence[Mapping[str, Any]], config: OEHConfig
) -> dict[str, Any]:
    """Per (arm, axis value) summary, negative and positive at equal detail."""

    return {
        "instances_run": len(rows),
        "oeh_status_counts": dict(
            sorted(Counter(row["oeh_status"] for row in rows).items())
        ),
        "escape_counts": {
            "loss_reaches_epsilon": sum(
                1 for row in rows if row["loss_reaches_epsilon"]
            ),
            "multiplier_above_one": sum(
                1
                for row in rows
                if row["max_multiplier"] != ""
                and float(row["max_multiplier"]) > 1.0 + MULTIPLIER_ESCAPE_ATOL
            ),
            "interior_argmin": sum(1 for row in rows if row["interior_argmin"]),
            "all_three": sum(1 for row in rows if row["escaped"]),
        },
        "max_loss_at_one": _stats(_numeric(rows, "max_loss_at_one")),
        "mean_gap_occupancy": _stats(_numeric(rows, "mean_gap_occupancy")),
        "config": asdict(config),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(
    dataset_dir: Path,
    *,
    arm_ids: Sequence[str],
    bootstrap_seed: int,
    n_min_values: Sequence[int],
    epsilon_values: Sequence[float],
    pilot_multipliers: Sequence[int],
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the requested arms over every pilot record in one bundle."""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = pilot_records(manifest)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("The bundle holds no 10|S||A| adaptive pilot candidates.")

    base_config = validate_arm_config(OEHConfig.load())
    selected_values: dict[str, tuple[Any, ...]] = {
        "ctrl-pilot": tuple(check_pilot_multiplier(int(v)) for v in pilot_multipliers),
        "nmin": tuple(check_n_min(int(v)) for v in n_min_values),
        "epsilon-robust": tuple(check_epsilon(float(v)) for v in epsilon_values),
    }
    unknown = [arm_id for arm_id in arm_ids if arm_id not in ARMS]
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; known arms: {sorted(ARMS)}.")

    rows: list[dict[str, Any]] = []
    cells: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    configs: dict[tuple[str, Any], OEHConfig] = {}
    started = time.time()
    total = len(records) * sum(len(selected_values[arm_id]) for arm_id in arm_ids)
    done = 0

    for record in records:
        mdp_id = str(record["instance_id"])
        family = str(record["policy_family"])
        num_states = int(record["num_states"])
        num_actions = int(record["num_actions"])
        initial = np.asarray(
            record["initial_state_distribution"], dtype=np.float64
        )
        with np.load(dataset_dir / record["file"], allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}

        for arm_id in arm_ids:
            arm = ARMS[arm_id]
            for axis_value in selected_values[arm_id]:
                result, config = run_arm_value(
                    arm_id,
                    axis_value,
                    arrays,
                    initial,
                    num_states=num_states,
                    num_actions=num_actions,
                    base_config=base_config,
                    bootstrap_seed=bootstrap_seed,
                )
                row = sweep_row(
                    arm_id, arm.axis, axis_value, mdp_id, family, result, config
                )
                rows.append(row)
                cells.setdefault((arm_id, axis_value), []).append(row)
                configs[(arm_id, axis_value)] = config
                done += 1
                print(
                    f"[{done}/{total}] {mdp_id} {family} "
                    f"{arm_id}={axis_value}: {row['oeh_status']} "
                    f"max_L={row['max_loss_at_one']} "
                    f"escaped={row['escaped']} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )

    arms_report: dict[str, Any] = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        arm_rows = [row for row in rows if row["arm_id"] == arm_id]
        arms_report[arm_id] = {
            "axis": arm.axis,
            "axis_values": {
                str(value): summarize_cell(
                    cells[(arm_id, value)], configs[(arm_id, value)]
                )
                for value in selected_values[arm_id]
                if (arm_id, value) in cells
            },
            "arm_totals": {
                "rows": len(arm_rows),
                "escaped": sum(1 for row in arm_rows if row["escaped"]),
                "loss_reaches_epsilon": sum(
                    1 for row in arm_rows if row["loss_reaches_epsilon"]
                ),
            },
        }

    report: dict[str, Any] = {
        "status": (
            "exploratory; not a registered result until EXPERIMENT_RULE.md "
            "rule 8 has been completed"
        ),
        "scope": (
            f"{len(records)} pilot records at 10|S||A| from "
            f"{sorted({str(item['benchmark_split']) for item in records})}"
        ),
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "protocol_status_at_run": manifest.get("protocol_status"),
        "run_mode": manifest.get("run_mode"),
        "bootstrap_seed": bootstrap_seed,
        "runtime_seconds": round(time.time() - started, 1),
        "escape_criterion": {
            "condition_1": "max_k L_k(1) >= epsilon",
            "condition_2": f"max_k m_k > 1 + {MULTIPLIER_ESCAPE_ATOL}",
            "condition_3": "0 < k_dagger < K_max",
            "rule": "an instance escapes only when all three hold",
            "fixed_before_run": "EXPERIMENT_RULE.md section 2",
        },
        "epsilon_floor": EPSILON_FLOOR,
        "baseline_config": asdict(base_config),
        "arms": arms_report,
    }
    return rows, report


def _parse_arms(value: str) -> list[str]:
    arms = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown arms {unknown}; known arms: {sorted(ARMS)}."
        )
    return arms


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the registered axes and measure which, if any, lets "
            "L_k(1) reach epsilon."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arms",
        type=_parse_arms,
        default=list(ARMS),
        help=f"Comma-separated subset of {sorted(ARMS)}; default all.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260824,
        help="Declared in the arm registry before the first run; never changed after seeing results.",
    )
    parser.add_argument(
        "--n-min-values",
        type=_parse_ints,
        default=list(REGISTERED_N_MIN),
        help=f"Subset of the registered grid {list(REGISTERED_N_MIN)}.",
    )
    parser.add_argument(
        "--epsilon-values",
        type=_parse_floats,
        default=list(REGISTERED_EPSILON),
        help=(
            f"Subset of the registered grid {list(REGISTERED_EPSILON)}. "
            f"Anything below {EPSILON_FLOOR} is refused."
        ),
    )
    parser.add_argument(
        "--pilot-multipliers",
        type=_parse_ints,
        default=list(REGISTERED_PILOT_MULTIPLIERS),
        help=f"Subset of the registered grid {list(REGISTERED_PILOT_MULTIPLIERS)}.",
    )
    parser.add_argument(
        "--limit", type=int, help="Only the first N pilot records; smoke use only."
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"{args.output_dir} is not empty; refusing to overwrite an artifact directory."
        )

    rows, report = run(
        args.dataset_dir,
        arm_ids=args.arms,
        bootstrap_seed=args.bootstrap_seed,
        n_min_values=args.n_min_values,
        epsilon_values=args.epsilon_values,
        pilot_multipliers=args.pilot_multipliers,
        limit=args.limit,
    )

    csv_path = args.output_dir / "sweep_results.csv"
    write_csv_rows(csv_path, rows, CSV_FIELDS)
    report["artifact_sha256"] = {"sweep_results.csv": sha256_file(csv_path)}
    write_json(args.output_dir / "sweep_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
