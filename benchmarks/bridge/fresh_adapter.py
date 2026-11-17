"""Adapter for the frozen Atari/Procgen BRIDGE sticky panel.

The BRIDGE MDP kernels are published; the offline logs and requirement labels
used by OEH are generated separately. The previous executed action belongs
to the Markov state. Episode length comes from the pinned author metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.tabular_environments import TabularMDP

ROOT = Path(__file__).resolve().parents[2]
PANEL = Path(__file__).resolve().parent / "fresh_panel_2026-10-08"
DATA = ROOT / "datasets/bridge_v1_1_1/mdps"


def registry() -> dict[str, dict]:
    frozen = json.loads((PANEL / "manifest.json").read_text(encoding="utf-8"))
    path = PANEL / "candidate_registry.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["registry_sha256"]:
        raise ValueError("Frozen BRIDGE registry changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["mdp_name"] for row in rows] != frozen["task_names"]:
        raise ValueError("Frozen BRIDGE task order changed")
    return {row["mdp_name"]: row for row in rows}


def load_fresh_sticky_mdp(name: str, *, sticky_probability: float = .25) -> TabularMDP:
    spec = registry().get(name)
    if spec is None or not 0 <= sticky_probability <= 1:
        raise ValueError("Task or sticky probability outside frozen panel")
    path = DATA / name / "consolidated.npz"
    with np.load(path, allow_pickle=False) as raw:
        transitions = raw["transitions"]
        rewards = raw["rewards"]
    if transitions.shape != rewards.shape or transitions.ndim != 2:
        raise ValueError("Invalid published base arrays")
    base_states, actions = transitions.shape
    if actions != int(spec["actions"]) or base_states + 1 != int(spec["base_states"]):
        raise ValueError("Published MDP dimensions differ from frozen metadata")
    if not np.issubdtype(transitions.dtype, np.integer) or not np.all(
            (transitions >= -1) & (transitions < base_states)):
        raise ValueError("Invalid successor indices")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Nonfinite published reward")
    sentinel = actions
    reachable = [(0, sentinel)]
    index = {reachable[0]: 0}
    terminal_actions: set[int] = set()
    for base, previous in reachable:
        for command in range(actions):
            executed_options = (command,) if previous == sentinel else (previous, command)
            for executed in executed_options:
                next_base = int(transitions[base, executed])
                if next_base < 0:
                    terminal_actions.add(executed)
                else:
                    key = (next_base, executed)
                    if key not in index:
                        index[key] = len(reachable)
                        reachable.append(key)
    terminal_by_action = {action: len(reachable) + i
                          for i, action in enumerate(sorted(terminal_actions))}
    states = len(reachable) + len(terminal_by_action)
    # The author summary counts a different sticky state representation. Keep
    # the reachable-state count explicit; check dynamics against the wrapper.
    if not base_states <= states <= (base_states + 1) * (actions + 1):
        raise ValueError(f"Implausible sticky state count for {name}: {states}")
    P = np.zeros((states, actions, states), dtype=np.float64)
    reward_mass = np.zeros_like(P)
    for state, (base, previous) in enumerate(reachable):
        for command in range(actions):
            choices = ((command, 1.0),) if previous == sentinel else (
                (previous, sticky_probability), (command, 1 - sticky_probability))
            for executed, chance in choices:
                if chance == 0:
                    continue
                next_base = int(transitions[base, executed])
                next_state = (terminal_by_action[executed] if next_base < 0
                              else index[(next_base, executed)])
                P[state, command, next_state] += chance
                reward_mass[state, command, next_state] += chance * float(rewards[base, executed])
    terminal_states = tuple(terminal_by_action.values())
    for terminal in terminal_states:
        P[terminal, :, terminal] = 1.0
    R = np.divide(reward_mass, P, out=np.zeros_like(P), where=P > 0)
    rho = np.zeros(states, dtype=np.float64)
    rho[0] = 1.0
    mdp = TabularMDP(name + "_sticky25", P, R, rho, terminal_states, .99, {
        "source": "BRIDGE v1.1.1 published consolidated.npz",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "origin": spec["origin"],
        "sticky_probability": sticky_probability,
        "state_encoding": "BFS-reachable (base_state, previous_executed_action); reset previous=A",
        "reachable_sticky_states_including_terminal": states,
        "published_sticky_states": int(spec["sticky_states"]),
        "published_minus_adapter_states": int(spec["sticky_states"]) - states,
        "horizon": int(spec["horizon"]),
        "horizon_handling": "external episode cap; stationary observed state as in published tabular setup",
    })
    mdp.validate()
    return mdp
