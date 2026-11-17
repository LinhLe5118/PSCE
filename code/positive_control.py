"""Positive control for the registered OEH estimator.

Every result in the Week 2 record is negative: 0 escapes in 600 sweep cells,
`m_k = 1.0` at all 12,019 development curve points.  The first question a
committee asks about a null result is *how do you know the pipeline is not
simply broken*.  A positive control answers it: an instance where the
**registered** estimator, unmodified, returns ``m > 1`` at an interior depth.
Until one exists, "the metric does not work" and "our code does not work" are
indistinguishable, and the whole negative result is worth nothing.

Two properties make this module the control rather than another experiment:

* the estimator is the registered one.  ``verify`` refuses a config that
  differs from ``OEHConfig.load()`` on any field that could manufacture the
  result — most of all ``epsilon``, since ``m_k = min{mu >= 1 : L_k(mu) <=
  epsilon}`` means lowering it fabricates exactly the outcome being sought;
* a failure is diagnosed, not merely reported.  Theorem 1 gives
  ``L_k(1) <= 0.16997 * sum_s w_k(s) sigma_k(s)``, so an instance whose
  occupancy-weighted dispersion is under ``epsilon / 0.16997 = 0.2942`` cannot
  reach ``epsilon`` at any depth, whatever else is true of it.  Saying which
  of the three conditions failed, and by how much, is what lets whoever builds
  the candidate instances converge instead of guessing.

Handover format, so the instance builder and this module do not have to agree
on anything else: an ``.npz`` carrying the eight
``adaptive_pilot.transitions_from_arrays`` fields, plus ``num_states``,
``num_actions`` and ``initial_state_distribution`` either as arrays in the same
file or in a JSON sidecar of the same stem.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from adaptive_pilot import transitions_from_arrays
from oeh_estimator import OEHConfig, OEHResult, estimate_oeh

#: ``max_{z>=0} z Phi(-z)``, attained at ``z* = 0.7518``.  Theorem 1's constant.
MAX_Z_PHI = 0.16997

#: Multipliers within this of 1.0 count as the floor, matching the sweep.
MULTIPLIER_ATOL = 1e-9

#: Fields whose alteration would change what "the registered estimator" means.
#: ``num_bootstrap`` is included because a smaller ``B`` inflates ``sigma`` and
#: could carry an instance over the bound by noise alone.
REGISTERED_FIELDS: tuple[str, ...] = (
    "gamma",
    "tau_coefficient",
    "n_min",
    "k_max",
    "num_bootstrap",
    "epsilon",
    "min_valid_bootstrap_fraction",
    "bootstrap_scheme",
    "bootstrap_dirichlet_alpha",
    "min_trajectory_support",
)


def required_sigma_bar(epsilon: float) -> float:
    """Smallest ``sum_s w_k(s) sigma_k(s)`` that could reach ``epsilon``.

    Theorem 1 bounds the registered loss by ``MAX_Z_PHI * sigma_bar``, so this
    is a necessary condition on the instance, independent of its gaps, depth,
    or transition structure.  Reported alongside every failure because it is
    the one number an instance builder can aim at.
    """

    return float(epsilon) / MAX_Z_PHI


def weighted_sigma_bar(result: OEHResult) -> float | None:
    """``max_k sum_s w_k(s) sigma_k(s)`` -- the quantity Theorem 1 bounds by.

    Named rather than inlined because ``code/feasibility.py`` needs the same
    number, and two implementations of one quantity is how two tools come to
    report different values for the same instance.
    """

    if result.sigmas is None or result.occupancy_weights is None:
        return None
    weights = np.nan_to_num(np.asarray(result.occupancy_weights, dtype=np.float64), nan=0.0)
    sigmas = np.nan_to_num(np.asarray(result.sigmas, dtype=np.float64), nan=0.0)
    return float(np.max((weights * sigmas).sum(axis=1)))


def assert_registered(config: OEHConfig) -> OEHConfig:
    """Refuse anything but the registered estimator.

    A positive control run on a tweaked estimator proves the tweak works, not
    the pipeline, so this is a hard error rather than a warning.
    """

    registered = OEHConfig.load()
    altered = [
        name
        for name in REGISTERED_FIELDS
        if getattr(config, name) != getattr(registered, name)
    ]
    if altered:
        raise ValueError(
            "A positive control must run the registered estimator; these "
            f"fields differ from protocol_config.json: {sorted(altered)}."
        )
    return config


@dataclass(frozen=True)
class PositiveControlVerdict:
    """The three registered conditions, kept separate, plus a diagnosis.

    ``passed`` is their conjunction and nothing else; a candidate that clears
    two of three has not passed, and the field that failed says which lever
    the instance builder has to move.
    """

    label: str
    passed: bool
    oeh_status: str
    max_multiplier: float | None
    k_dagger: int | None
    k_max: int
    epsilon: float
    max_loss_at_one: float | None
    sigma_bar: float | None
    required_sigma_bar: float
    structural_bound: float | None
    loss_reaches_epsilon: bool
    multiplier_above_one: bool
    interior_argmin: bool
    diagnosis: str

    @property
    def sigma_headroom(self) -> float | None:
        """How far the instance is from Theorem 1's necessary condition."""

        if self.sigma_bar is None:
            return None
        return self.sigma_bar / self.required_sigma_bar


def _diagnose(
    *,
    status: str,
    loss_reaches_epsilon: bool,
    multiplier_above_one: bool,
    interior_argmin: bool,
    sigma_bar: float | None,
    needed: float,
    k_dagger: int | None,
    k_max: int,
) -> str:
    if status == "abstained":
        return (
            "estimator abstained, so no curve exists; the instance needs "
            "supported action pairs in its critical states before it can "
            "serve as a control"
        )
    if not loss_reaches_epsilon:
        if sigma_bar is not None and sigma_bar < needed:
            return (
                f"blocked by Theorem 1: sigma_bar={sigma_bar:.4g} is below the "
                f"necessary {needed:.4g}, so L_k(1) cannot reach epsilon at any "
                "depth regardless of the gaps. Raise bootstrap dispersion "
                "(fewer trajectories, higher-variance rewards), not the gaps"
            )
        return (
            f"sigma_bar={sigma_bar:.4g} clears the necessary bound but the loss "
            "still misses epsilon, so the gaps are mis-scaled relative to it: "
            "the per-state z is far from the maximising z*=0.7518"
        )
    if not multiplier_above_one:
        return (
            "loss reaches epsilon but every multiplier is still at the mu>=1 "
            "floor, which should not happen; inspect the root search"
        )
    if not interior_argmin:
        return (
            f"m>1 achieved but k_dagger={k_dagger} sits on a boundary "
            f"(0 or {k_max}); the instance needs depth structure so that "
            "backups sharpen the gap before variance dominates"
        )
    return "all three registered conditions met"


def verify(
    arrays: Mapping[str, Any],
    initial_state_distribution: Any,
    *,
    num_states: int,
    num_actions: int,
    label: str = "candidate",
    seed: int,
    config: OEHConfig | None = None,
) -> PositiveControlVerdict:
    """Run the registered estimator on one candidate pilot buffer.

    ``seed`` is required rather than defaulted: a control that passes only
    under an unrecorded random draw is not a control.
    """

    cfg = assert_registered(config or OEHConfig.load())
    result = estimate_oeh(
        transitions_from_arrays(arrays),
        np.asarray(initial_state_distribution, dtype=np.float64),
        num_states=int(num_states),
        num_actions=int(num_actions),
        config=cfg,
        rng=np.random.default_rng(seed),
    )
    return verdict_from_result(result, cfg, label=label)


def verdict_from_result(
    result: OEHResult, config: OEHConfig, *, label: str = "candidate"
) -> PositiveControlVerdict:
    """Score an estimator result against the registered control criterion."""

    needed = required_sigma_bar(config.epsilon)
    status = (
        "abstained"
        if result.abstained
        else "censored"
        if result.censored_right
        else "observed"
    )
    sigma_bar: float | None = None
    bound: float | None = None
    sigma_bar = weighted_sigma_bar(result)
    if sigma_bar is not None:
        bound = MAX_Z_PHI * sigma_bar

    max_loss = _finite_max(result.losses_at_one)
    max_multiplier = _finite_max(result.multipliers)
    loss_reaches_epsilon = max_loss is not None and max_loss >= config.epsilon
    multiplier_above_one = (
        max_multiplier is not None and max_multiplier > 1.0 + MULTIPLIER_ATOL
    )
    interior_argmin = (
        result.k_dagger is not None and 0 < result.k_dagger < config.k_max
    )
    return PositiveControlVerdict(
        label=label,
        passed=bool(loss_reaches_epsilon and multiplier_above_one and interior_argmin),
        oeh_status=status,
        max_multiplier=max_multiplier,
        k_dagger=result.k_dagger,
        k_max=config.k_max,
        epsilon=config.epsilon,
        max_loss_at_one=max_loss,
        sigma_bar=sigma_bar,
        required_sigma_bar=needed,
        structural_bound=bound,
        loss_reaches_epsilon=loss_reaches_epsilon,
        multiplier_above_one=multiplier_above_one,
        interior_argmin=interior_argmin,
        diagnosis=_diagnose(
            status=status,
            loss_reaches_epsilon=loss_reaches_epsilon,
            multiplier_above_one=multiplier_above_one,
            interior_argmin=interior_argmin,
            sigma_bar=sigma_bar,
            needed=needed,
            k_dagger=result.k_dagger,
            k_max=config.k_max,
        ),
    )


def _finite_max(values: Any) -> float | None:
    """Maximum ignoring NaN but keeping ``inf``; ``None`` when nothing is set."""

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[~np.isnan(array)]
    return float(finite.max()) if finite.size else None


def load_candidate(path: Path) -> dict[str, Any]:
    """Read one handover buffer plus its dimensions.

    Dimensions may travel inside the ``.npz`` or in a JSON sidecar of the same
    stem.  They are never inferred from the transitions: a state the pilot
    happens not to visit still exists, and guessing ``num_states`` from the
    data would silently shrink ``tau = 1/(10|S|)``.
    """

    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    sidecar = path.with_suffix(".json")
    meta: dict[str, Any] = {}
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("num_states", "num_actions", "initial_state_distribution")
        if key not in arrays and key not in meta
    ]
    if missing:
        raise ValueError(
            f"{path.name}: candidate is missing {sorted(missing)}; supply them "
            "as arrays in the npz or in a JSON sidecar of the same stem."
        )

    def pick(key: str) -> Any:
        return meta[key] if key in meta else arrays[key]

    return {
        "arrays": {
            name: value
            for name, value in arrays.items()
            if name
            not in {"num_states", "num_actions", "initial_state_distribution"}
        },
        "num_states": int(np.asarray(pick("num_states")).reshape(()).item()),
        "num_actions": int(np.asarray(pick("num_actions")).reshape(()).item()),
        "initial_state_distribution": np.asarray(
            pick("initial_state_distribution"), dtype=np.float64
        ),
    }


#: Parameters of the reference control below, kept as one constant so the
#: fixture, its test and any report quote the same numbers.
REFERENCE_CONTROL = {
    "chain_length": 3,
    "repeats": 3,
    "advance_probability": 0.7,
    "buffer_seed": 1007,
    "estimator_seed": 20260901,
}


def stochastic_chain_control_buffer(
    *,
    chain_length: int = REFERENCE_CONTROL["chain_length"],
    repeats: int = REFERENCE_CONTROL["repeats"],
    advance_probability: float = REFERENCE_CONTROL["advance_probability"],
    seed: int = REFERENCE_CONTROL["buffer_seed"],
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    """Build a stochastic, high-variance chain pilot in handover format.

    The caller owns the parameter choice and provenance.  The internal
    reference fixture and Workstream 1's independent handover both use this
    mechanism, but use distinct registered parameter sets and seeds.

    Why the design forces a large dispersion, which Theorem 1 makes the binding
    necessary condition:

    * **Few trajectories.**  ``sigma`` of a Dirichlet-weighted mean falls like
      ``1/sqrt(T)``, so the control uses a handful of trajectories rather than
      the hundreds a benchmark pilot collects.  Two per action per state is the
      registered support minimum, and the control sits just above it.
    * **Maximally split terminal reward.**  Entering the terminal pays 0 or 1
      with equal chance rather than a fixed value, so the per-trajectory
      returns disagree as much as rewards in ``[0,1]`` permit.
    * **A chain, not a bandit.**  ``PLAN_Final.md`` sec 3.5 shows a single
      decision state gives ``k_dagger = 0`` by construction, which fails the
      interior-argmin condition however large the dispersion is.  Three
      decision states let backups sharpen the gap before variance from distant
      states dominates, which is what puts the argmin strictly inside.

    Returns the arrays, the initial distribution, ``num_states`` and
    ``num_actions``.
    """

    rng = np.random.default_rng(seed)
    rows: list[tuple[Any, ...]] = []
    trajectory = 0
    for _ in range(repeats):
        for first_action in (0, 1):
            state, step = 0, 0
            while state < chain_length and step < 2 * chain_length:
                action = (
                    first_action
                    if step == 0
                    else (0 if rng.random() < advance_probability else 1)
                )
                nxt = state + 1 if action == 0 else 0
                terminal = nxt >= chain_length
                reward = float(rng.choice([0.0, 1.0]))
                rows.append(
                    (state, action, reward, nxt, terminal, False, trajectory, step)
                )
                state, step = nxt, step + 1
                if terminal:
                    break
            trajectory += 1
    columns = list(zip(*rows))
    names = (
        "state",
        "action",
        "reward",
        "next_state",
        "terminal",
        "truncated",
        "trajectory_id",
        "timestep",
    )
    arrays = {name: np.asarray(column) for name, column in zip(names, columns)}
    num_states = chain_length + 1
    initial = np.zeros(num_states, dtype=np.float64)
    initial[0] = 1.0
    return arrays, initial, num_states, 2


def reference_control_buffer(
    *,
    chain_length: int = REFERENCE_CONTROL["chain_length"],
    repeats: int = REFERENCE_CONTROL["repeats"],
    advance_probability: float = REFERENCE_CONTROL["advance_probability"],
    seed: int = REFERENCE_CONTROL["buffer_seed"],
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    """Return the fixed internal reference fixture used by the harness."""

    return stochastic_chain_control_buffer(
        chain_length=chain_length,
        repeats=repeats,
        advance_probability=advance_probability,
        seed=seed,
    )


def verdict_row(verdict: PositiveControlVerdict) -> dict[str, Any]:
    """Flatten one verdict for CSV, empty cell for an undefined number."""

    row: dict[str, Any] = {}
    for field in fields(verdict):
        value = getattr(verdict, field.name)
        if value is None or (
            isinstance(value, float) and not math.isfinite(value) and math.isnan(value)
        ):
            row[field.name] = ""
        else:
            row[field.name] = value
    row["sigma_headroom"] = (
        "" if verdict.sigma_headroom is None else verdict.sigma_headroom
    )
    return row


VERDICT_FIELDS: tuple[str, ...] = tuple(
    [field.name for field in fields(PositiveControlVerdict)] + ["sigma_headroom"]
)


__all__ = [
    "MAX_Z_PHI",
    "PositiveControlVerdict",
    "REGISTERED_FIELDS",
    "VERDICT_FIELDS",
    "assert_registered",
    "load_candidate",
    "reference_control_buffer",
    "required_sigma_bar",
    "stochastic_chain_control_buffer",
    "verdict_from_result",
    "verdict_row",
    "verify",
]
