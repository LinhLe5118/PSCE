"""Decoy-corridor MDPs: a benchmark built to make lookahead depth matter.

Not a benchmark instance. This is a *probe* for testing the definition, the
same role ``code/probe_envs.py`` plays, and it must never be reported as a
120+15+3 benchmark number (``PLAN_Final.md`` sec 5).

Why it exists (registered in ``ARM_REGISTRY.md``, round 4b, before any run):
the development split reports ``k_dagger = 0`` at all 12,019 evidence-curve
points, and two objections to reading that as a fact about the *definition*
have never been separated from it.

* **Depth had nothing to express.** The development MDPs have 8-64 states and
  ``gamma = 0.95``. On layered and garnet structures at that size, one or two
  Bellman backups already produce the optimal action ranking, so ``k = 0``
  being the argmin is close to a foregone conclusion.
* **The pilot never saw reward.** 12 of 20 ``lock_structured`` pilots contain
  no non-zero reward at all, because a near-uniform policy reaches a
  length-``L`` combination lock's goal with probability ``~2^-L``. No
  data-only statistic can work on a buffer with no signal in it; that is not a
  finding about metrics.

The corridor is designed so neither objection applies.

**Structure.** ``L`` corridor states plus a terminal. In each corridor state:

* action 0 -- *advance*: move one step forward with probability ``1 - p``,
  slip back to the start with probability ``p``. No immediate reward.
* action 1 -- *decoy*: collect ``delta`` immediately, then return to the start.

``delta`` is **not** given directly. It is derived from ``decoy_fraction``, the
share of the corridor's own payoff that an infinite decoy loop is worth::

    delta = decoy_fraction * (1 - gamma) * gamma ** chain_length

so that the decoy loop's value ``delta / (1 - gamma) = decoy_fraction *
gamma**L`` is strictly below the corridor's ``gamma**L``. Parameterising the
absolute reward instead is what a first draft of this file did, and a
value-iteration sanity check caught it: at ``gamma = 0.95`` an absolute
``delta = 0.05`` is worth ``0.05 / 0.05 = 1.0`` forever, which beats *any*
one-off terminal reward, so the "decoy" was simply the optimal policy and the
instance did not implement its own design. Tying ``delta`` to ``gamma**L``
keeps the decoy locally attractive and globally wrong at every ``L``, which is
the property the probe needs.

The only large reward, ``1.0``, is on the transition into the terminal.

**Intended mechanism -- and a correction, 2026-09-06.** The design argument for
this file read: *under a behavior policy that mostly advances, ``Q^(0)`` is the
value of that policy, and at the near end of the corridor the decoy's immediate
``delta`` beats a discounted terminal reward still ``gamma^L`` away, so the
depth-0 ranking favours the decoy and must flip after enough backups.*

**That argument does not go through, and the claim built on it is withdrawn.**
It compares ``delta`` against the terminal reward as if ``Q^(0)(s, advance)``
did not already contain it. It does. With ``beta`` the epsilon-greedy-advance
policy used here, which completes the corridor on 83-100% of trajectories::

    Q^beta(s, decoy)   = delta + gamma * V^beta(s_0)
    Q^beta(s, advance) = gamma * [ (1 - p) * V^beta(s + 1) + p * V^beta(s_0) ]

so advance wins at depth 0 exactly when
``gamma (1-p) (V^beta(s+1) - V^beta(s_0)) > delta``.  For a policy that reaches
the terminal, ``V^beta(s+1) - V^beta(s_0)`` is of order ``(1 - gamma) gamma^L``
-- the **same order** as ``delta = decoy_fraction * (1 - gamma) * gamma^L``.
The sign is therefore decided by ``decoy_fraction`` and ``gamma``, not fixed by
the construction, and the top-two ranking at ``k = 0`` may already be correct.

What this file still delivers, and what the artifact may be cited for, is the
removal of the two grounded objections to the null: the pilot **sees reward**
(density 0.217-0.250 against 0.000 for the starved locks) and the occupancy
weight is **not** concentrated away from the states carrying the gap. What it
does **not** deliver is a family on which lookahead depth is *known* to be
required. The measurement that would settle it --
``k_settle(s) = min{k : the Q^(k) ranking agrees with pi* on the exact model}``,
by value iteration -- is registered as an open item owned by Workstream 1
(Benchmark & Oracle). See ``artifacts/week2/ARM_REGISTRY.md`` (2026-09-06
correction) and ``WEEK2_DECISION_GATE.md`` sec 10j.

**Why the pilot sees reward.** The behavior policy is epsilon-greedy around
*advance*, so it completes the corridor at a rate that stays workable as ``L``
grows, instead of the ``2^-L`` of a combination lock. Every generated buffer
reports its own non-zero-reward fraction so the claim is checked, not assumed.

**Why transitions are stochastic.** A deterministic MDP gives every bootstrap
replicate the same empirical model, so ``sigma_hat_Delta = 0`` exactly and the
depth-evidence curve is flat for a reason that has nothing to do with the
definition (``code/probe_envs.py`` says the same). ``p > 0`` gives the
resampling something real to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Action indices, fixed so the buffers and the analysis agree.
ADVANCE = 0
DECOY = 1


@dataclass(frozen=True)
class CorridorSpec:
    """One registered grid point (``ARM_REGISTRY.md`` round 4b)."""

    chain_length: int
    decoy_fraction: float
    slip_probability: float
    gamma: float = 0.95

    @property
    def decoy_reward(self) -> float:
        """Immediate decoy payoff, scaled so the decoy loop is worth
        ``decoy_fraction`` of the corridor. See the module docstring."""

        return self.decoy_fraction * (1.0 - self.gamma) * self.gamma**self.chain_length

    @property
    def label(self) -> str:
        fraction = f"{self.decoy_fraction:g}".replace(".", "p")
        slip = f"{self.slip_probability:g}".replace(".", "p")
        return f"decoy_corridor_L{self.chain_length}_f{fraction}_p{slip}"

    def validate(self) -> None:
        if self.chain_length < 2:
            raise ValueError("chain_length must be >= 2 for depth to matter.")
        if not 0.0 < self.decoy_fraction < 1.0:
            raise ValueError(
                "decoy_fraction must lie strictly inside (0, 1): at 1.0 or above the decoy "
                "loop matches or beats the corridor and stops being a decoy."
            )
        if not 0.0 <= self.slip_probability < 1.0:
            raise ValueError("slip_probability must lie in [0, 1).")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly inside (0, 1).")


def build_corridor(spec: CorridorSpec) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, ...]]:
    """Return ``(P, R, initial_state_distribution, terminal_states)``.

    ``P`` and ``R`` have shape ``(S, A, S)`` with ``S = chain_length + 1``,
    matching ``TabularMDP`` so the exact solver can score these instances
    without a separate code path.
    """

    spec.validate()
    num_states = spec.chain_length + 1
    terminal = spec.chain_length
    num_actions = 2

    P = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    R = np.zeros((num_states, num_actions, num_states), dtype=np.float64)

    for state in range(spec.chain_length):
        forward = state + 1
        P[state, ADVANCE, forward] = 1.0 - spec.slip_probability
        P[state, ADVANCE, 0] += spec.slip_probability
        if forward == terminal:
            R[state, ADVANCE, forward] = 1.0

        # The decoy pays immediately and sends the agent back to the start, so
        # it is locally attractive and globally wrong -- the whole point.
        P[state, DECOY, 0] = 1.0
        R[state, DECOY, 0] = spec.decoy_reward

    P[terminal, :, terminal] = 1.0

    initial = np.zeros(num_states, dtype=np.float64)
    initial[0] = 1.0
    return P, R, initial, (terminal,)


def collect_buffer(
    spec: CorridorSpec,
    *,
    num_transitions: int,
    epsilon: float,
    gamma: float,
    seed: int,
    horizon: int = 100,
) -> dict[str, Any]:
    """Roll an epsilon-greedy-advance policy and return ``transitions_from_arrays`` fields.

    ``epsilon`` is the probability of taking the decoy at a corridor state.
    Keeping it well below 0.5 is what makes the corridor completable, and is
    the difference between this probe and the reward-starved combination lock.

    Episodes restart from the initial distribution on termination or timeout,
    exactly as ``pilot_collector`` does, so ``trajectory_id`` counts real
    episodes and the trajectory-level bootstrap has genuine units to resample.
    """

    P, R, initial, terminals = build_corridor(spec)
    terminal_set = set(terminals)
    rng = np.random.default_rng(seed)

    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    trajectory_ids: list[int] = []
    timesteps: list[int] = []

    trajectory = 0
    while len(states) < num_transitions:
        state = int(rng.choice(len(initial), p=initial))
        for step in range(horizon):
            if len(states) >= num_transitions:
                break
            action = DECOY if rng.random() < epsilon else ADVANCE
            successor = int(rng.choice(P.shape[2], p=P[state, action]))
            reward = float(R[state, action, successor])
            is_terminal = successor in terminal_set
            at_horizon = step == horizon - 1

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(successor)
            terminated.append(is_terminal)
            truncated.append(at_horizon and not is_terminal)
            trajectory_ids.append(trajectory)
            timesteps.append(step)

            state = successor
            if is_terminal or at_horizon:
                break
        trajectory += 1

    return {
        "state": np.asarray(states, dtype=np.int64),
        "action": np.asarray(actions, dtype=np.int64),
        "reward": np.asarray(rewards, dtype=np.float64),
        "next_state": np.asarray(next_states, dtype=np.int64),
        "terminal": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "trajectory_id": np.asarray(trajectory_ids, dtype=np.int64),
        "timestep": np.asarray(timesteps, dtype=np.int64),
        "num_states": P.shape[0],
        "num_actions": P.shape[1],
        "initial_state_distribution": initial,
        "gamma": gamma,
    }


def reward_density(buffer: dict[str, Any]) -> float:
    """Fraction of transitions carrying a non-zero reward.

    Reported on every row because "the pilot saw reward" is the premise this
    probe rests on, and a premise that is assumed rather than measured is how
    the combination-lock starvation went unnoticed for three weeks.
    """

    rewards = np.asarray(buffer["reward"], dtype=np.float64)
    return float(np.mean(rewards != 0.0)) if rewards.size else 0.0


def terminal_reach_rate(buffer: dict[str, Any]) -> float:
    """Fraction of trajectories that reached the terminal state."""

    trajectories = np.asarray(buffer["trajectory_id"])
    reached = np.asarray(buffer["terminal"], dtype=bool)
    if trajectories.size == 0:
        return 0.0
    total = len(np.unique(trajectories))
    return float(len(np.unique(trajectories[reached])) / total) if total else 0.0


#: The grid registered in ``ARM_REGISTRY.md`` round 4b, with the amendment of
#: 2026-09-06: ``decoy_reward in {0.05, 0.15}`` became ``decoy_fraction in
#: {0.4, 0.7}`` after a value-iteration check showed the absolute form did not
#: build a decoy at all. No estimator had been run at that point.
REGISTERED_GRID: tuple[CorridorSpec, ...] = tuple(
    CorridorSpec(chain_length=length, decoy_fraction=fraction, slip_probability=slip)
    for length in (4, 6, 8)
    for fraction in (0.4, 0.7)
    for slip in (0.05, 0.15)
)


__all__ = [
    "ADVANCE",
    "DECOY",
    "CorridorSpec",
    "REGISTERED_GRID",
    "build_corridor",
    "collect_buffer",
    "reward_density",
    "terminal_reach_rate",
]
