"""Data-only sequential selection of the effective OEH pilot budget.

The selector sees only nested transition prefixes plus public dimensions and
the initial-state distribution.  It never receives an MDP, structural-family
label, exact dynamics/rewards, or evaluation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.protocol_config import PROTOCOL_CONFIG  # noqa: E402


TransitionT = TypeVar("TransitionT")
ResultT = TypeVar("ResultT", bound="PilotAssessmentLike")


class PilotAssessmentLike(Protocol):
    """Minimum OEH result surface required by the sequential selector."""

    abstained: bool
    censored_right: bool
    reason: str | None
    valid_bootstrap_fraction: float | None


AssessPrefix = Callable[
    [Sequence[TransitionT], np.ndarray, int, int],
    ResultT,
]


@dataclass(frozen=True)
class PilotAttempt:
    multiplier: int
    size: int
    abstained: bool
    censored_right: bool
    reason: str | None
    valid_bootstrap_fraction: float | None
    accepted: bool


@dataclass(frozen=True)
class AdaptivePilotSelection(Generic[ResultT]):
    """Selected prefix and the complete pre-registered decision trace."""

    n0_effective_multiplier: int
    n0_effective_size: int
    status: str
    attempts: tuple[PilotAttempt, ...]
    result: ResultT


@dataclass(frozen=True)
class ArrayTransition:
    """Data-only transition view matching ``oeh_estimator.TransitionLike``."""

    state: int
    action: int
    reward: float
    next_state: int
    terminal: bool
    truncated: bool
    trajectory_id: int
    timestep: int


def transitions_from_arrays(
    arrays: Mapping[str, Any],
) -> tuple[ArrayTransition, ...]:
    """Convert one generated NPZ buffer into OEH transition records."""

    required = (
        "state",
        "action",
        "reward",
        "next_state",
        "terminal",
        "truncated",
        "trajectory_id",
        "timestep",
    )
    missing = set(required) - set(arrays)
    if missing:
        raise ValueError(f"Transition buffer is missing fields: {sorted(missing)}")
    values = {name: np.asarray(arrays[name]) for name in required}
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("Transition buffer fields must have equal length.")
    length = next(iter(lengths))
    return tuple(
        ArrayTransition(
            state=int(values["state"][index]),
            action=int(values["action"][index]),
            reward=float(values["reward"][index]),
            next_state=int(values["next_state"][index]),
            terminal=bool(values["terminal"][index]),
            truncated=bool(values["truncated"][index]),
            trajectory_id=int(values["trajectory_id"][index]),
            timestep=int(values["timestep"][index]),
        )
        for index in range(length)
    )


def select_effective_pilot(
    transitions: Sequence[TransitionT],
    initial_state_distribution: Sequence[float],
    *,
    num_states: int,
    num_actions: int,
    assess_prefix: AssessPrefix[TransitionT, ResultT],
) -> AdaptivePilotSelection[ResultT]:
    """Select the first usable prefix from ``{2,5,10}|S||A|``.

    A prefix is usable when OEH does not abstain, is not right-censored, and
    retains the protocol-required fraction of bootstrap replicates.  If no
    candidate is usable, the capped ``10|S||A|`` result is returned with an
    explicit terminal status; data collection is never extended beyond the
    registered grid.
    """

    if num_states <= 0 or num_actions <= 0:
        raise ValueError("num_states and num_actions must be positive.")
    initial = np.asarray(initial_state_distribution, dtype=np.float64)
    if (
        initial.shape != (num_states,)
        or np.any(initial < 0.0)
        or not np.isclose(initial.sum(), 1.0)
    ):
        raise ValueError(
            "initial_state_distribution must be a probability vector over S."
        )

    estimator = PROTOCOL_CONFIG["estimator"]
    candidates = tuple(
        int(value)
        for value in PROTOCOL_CONFIG["collection"]["pilot_size_multipliers"]
    )
    minimum_valid_fraction = float(
        estimator["min_valid_bootstrap_fraction"]
    )
    required_size = candidates[-1] * num_states * num_actions
    if len(transitions) < required_size:
        raise ValueError(
            "The pilot buffer must contain the registered maximum prefix "
            f"({required_size} transitions)."
        )

    attempts: list[PilotAttempt] = []
    latest_result: ResultT | None = None
    for multiplier in candidates:
        size = multiplier * num_states * num_actions
        result = assess_prefix(
            transitions[:size],
            initial.copy(),
            num_states,
            num_actions,
        )
        latest_result = result
        valid_fraction = result.valid_bootstrap_fraction
        bootstrap_valid = (
            valid_fraction is not None
            and np.isfinite(valid_fraction)
            and valid_fraction >= minimum_valid_fraction
        )
        accepted = bool(
            not result.abstained
            and not result.censored_right
            and bootstrap_valid
        )
        attempts.append(
            PilotAttempt(
                multiplier=multiplier,
                size=size,
                abstained=bool(result.abstained),
                censored_right=bool(result.censored_right),
                reason=result.reason,
                valid_bootstrap_fraction=(
                    float(valid_fraction) if valid_fraction is not None else None
                ),
                accepted=accepted,
            )
        )
        if accepted:
            return AdaptivePilotSelection(
                n0_effective_multiplier=multiplier,
                n0_effective_size=size,
                status="selected",
                attempts=tuple(attempts),
                result=result,
            )

    assert latest_result is not None
    final_attempt = attempts[-1]
    if final_attempt.abstained:
        status = "abstained_at_cap"
    elif final_attempt.censored_right:
        status = "censored_at_cap"
    else:
        status = "insufficient_bootstrap_at_cap"
    return AdaptivePilotSelection(
        n0_effective_multiplier=candidates[-1],
        n0_effective_size=required_size,
        status=status,
        attempts=tuple(attempts),
        result=latest_result,
    )


def select_effective_pilot_with_oeh(
    arrays: Mapping[str, Any],
    initial_state_distribution: Sequence[float],
    *,
    num_states: int,
    num_actions: int,
    bootstrap_seed: int,
) -> AdaptivePilotSelection[Any]:
    """Run the registered sequential rule with the real OEH estimator."""

    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer.")
    from oeh_estimator import estimate_oeh

    transitions = transitions_from_arrays(arrays)

    def assess_prefix(prefix, initial, states, actions):
        seed = np.random.SeedSequence(
            [bootstrap_seed % (2**32), len(prefix), states, actions]
        )
        return estimate_oeh(
            prefix,
            initial,
            num_states=states,
            num_actions=actions,
            rng=np.random.default_rng(seed),
        )

    return select_effective_pilot(
        transitions,
        initial_state_distribution,
        num_states=num_states,
        num_actions=num_actions,
        assess_prefix=assess_prefix,
    )


__all__ = [
    "AdaptivePilotSelection",
    "ArrayTransition",
    "PilotAssessmentLike",
    "PilotAttempt",
    "select_effective_pilot",
    "select_effective_pilot_with_oeh",
    "transitions_from_arrays",
]
