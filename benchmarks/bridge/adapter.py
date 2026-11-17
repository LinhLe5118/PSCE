"""Published BRIDGE base MDP to sticky-action TabularMDP adapter.

The state records the previous executed action (or a reset sentinel). A
100-step episode cap is imposed by the collector/evaluator, as in the authors'
time-invariant tabular environment. It is not part of the dense P/R tensors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from environments.tabular_environments import TabularMDP

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/bridge_v1_1_1/mdps"


def load_sticky_mdp(name: str, *, sticky_probability: float = .25) -> TabularMDP:
    if not name.startswith("MiniGrid-") or "/" in name or "\\" in name:
        raise ValueError("Invalid published BRIDGE task name")
    if not 0 <= sticky_probability <= 1:
        raise ValueError("Sticky probability outside [0,1]")
    path = DATA / name / "consolidated.npz"
    with np.load(path, allow_pickle=False) as raw:
        transitions = raw["transitions"]
        rewards = raw["rewards"]
    if transitions.shape != rewards.shape or transitions.ndim != 2:
        raise ValueError("Bad published base arrays")
    base_states, actions = transitions.shape
    if not np.issubdtype(transitions.dtype, np.integer) or not np.all(
            (transitions >= -1) & (transitions < base_states)):
        raise ValueError("Bad published successor IDs")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Bad published rewards")
    sentinel = actions
    # The published sticky-state counts include only reachable combinations.
    # Keeping all base*(A+1) combinations would inflate the x10 pilot budget.
    reachable = [(0, sentinel)]
    index = {reachable[0]: 0}
    terminal_actions = set()
    for base, previous in reachable:
        for command in range(actions):
            executed_options = (command,) if previous == sentinel else (previous, command)
            for executed in executed_options:
                base_next = int(transitions[base, executed])
                if base_next < 0:
                    terminal_actions.add(executed)
                    continue
                key = (base_next, executed)
                if key not in index:
                    index[key] = len(reachable)
                    reachable.append(key)
    terminal_by_action = {a: len(reachable) + i for i, a in enumerate(sorted(terminal_actions))}
    terminal_states = tuple(terminal_by_action.values())
    states = len(reachable) + len(terminal_states)
    P = np.zeros((states, actions, states), dtype=np.float64)
    reward_mass = np.zeros_like(P)
    for state, (base, previous) in enumerate(reachable):
        for command in range(actions):
            choices = ((command, 1.0),) if previous == sentinel else (
                (previous, sticky_probability), (command, 1 - sticky_probability))
            for executed, chance in choices:
                if chance == 0:
                    continue
                base_next = int(transitions[base, executed])
                next_state = terminal_by_action[executed] if base_next == -1 else index[(base_next, executed)]
                P[state, command, next_state] += chance
                reward_mass[state, command, next_state] += chance * float(rewards[base, executed])
    for terminal in terminal_states:
        P[terminal, :, terminal] = 1.0
    R = np.divide(reward_mass, P, out=np.zeros_like(P), where=P > 0)
    rho = np.zeros(states, dtype=np.float64)
    rho[0] = 1.0
    mdp = TabularMDP(name + "_sticky25", P, R, rho, terminal_states, .99, {
        "source": "BRIDGE v1.1.1 published consolidated.npz",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sticky_probability": sticky_probability,
        "state_encoding": "BFS-reachable (base_state, previous_executed_action); reset previous=A",
        "reachable_sticky_states_including_terminal": states,
        "horizon": 100,
        "horizon_handling": "external episode cap; stationary observed state as in author's tabular setup",
    })
    mdp.validate()
    return mdp
