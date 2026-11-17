"""OEH-R entry point for the existing data-only trajectory array contract.

Exploratory backend; does not replace the frozen estimator or adaptive pilot
selection rule. Terminal destinations observed in this PREFIX are absorbing
zero-value states under the repository's state-based termination contract.
Truncation is a collector boundary, never an absorbing transition.
"""
from __future__ import annotations

import numpy as np

from oeh_certificate import ModelBox, confidence_box, certify_depths, pessimistic_depth_policies


def trajectory_statistics(arrays, *, num_states, num_actions, reward_bound):
    names = ("state", "action", "reward", "next_state", "terminal", "truncated",
             "trajectory_id", "timestep")
    if any(name not in arrays for name in names):
        raise ValueError("Missing trajectory contract fields.")
    a = {name: np.asarray(arrays[name]) for name in names}
    if any(v.ndim != 1 for v in a.values()) or len({len(v) for v in a.values()}) != 1:
        raise ValueError("Trajectory arrays must be one-dimensional and aligned.")
    if not len(a["state"]):
        raise ValueError("Empty pilot prefix.")
    if not isinstance(num_states, int) or not isinstance(num_actions, int) or min(num_states, num_actions) < 1:
        raise ValueError("Positive public dimensions required.")
    for name in ("state", "action", "next_state", "trajectory_id", "timestep"):
        if not np.issubdtype(a[name].dtype, np.integer) or np.any(a[name] < 0):
            raise ValueError("Integer nonnegative indices required.")
    for name in ("terminal", "truncated"):
        if a[name].dtype != np.bool_:
            raise ValueError("Boolean boundary flags required.")
    s, ac, ns, r = a["state"], a["action"], a["next_state"], a["reward"]
    if np.any(s >= num_states) or np.any(ns >= num_states) or np.any(ac >= num_actions):
        raise ValueError("Observed state/action outside public dimensions.")
    if (not np.isfinite(reward_bound) or reward_bound <= 0
            or not np.all(np.isfinite(r)) or np.any(r < 0) or np.any(r > reward_bound)):
        raise ValueError("Reward outside the public bound.")
    terminal, truncated = a["terminal"], a["truncated"]
    if np.any(terminal & truncated):
        raise ValueError("A transition cannot terminate and truncate together.")
    if "done" in arrays:
        done = np.asarray(arrays["done"])
        if done.dtype != np.bool_ or not np.array_equal(done, terminal):
            raise ValueError("Repository contract requires done == terminal.")
    terminals = np.unique(ns[terminal])
    if np.any(np.isin(s, terminals)) or np.any(np.isin(ns[~terminal], terminals)):
        raise ValueError("Contradictory state-based terminal semantics.")
    ids, times = a["trajectory_id"], a["timestep"]
    same = ids[1:] == ids[:-1]
    if times[0] != 0 or np.any(times[1:][~same] != 0):
        raise ValueError("Pilot must start at an episode boundary.")
    starts = np.r_[True, ~same]
    if len(set(ids[starts].tolist())) != np.count_nonzero(starts):
        raise ValueError("Trajectory identifiers must not recur after a boundary.")
    if (np.any(times[1:][same] != times[:-1][same] + 1)
            or np.any(s[1:][same] != ns[:-1][same])
            or np.any((terminal[:-1] | truncated[:-1])[same])
            or np.any(~(terminal[:-1] | truncated[:-1])[~same])):
        raise ValueError("Broken trajectory continuity or reset boundary.")
    counts = np.zeros((num_states, num_actions, num_states), dtype=np.int64)
    rewards, squares = np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))
    np.add.at(counts, (s, ac, ns), 1)
    np.add.at(rewards, (s, ac), r)
    np.add.at(squares, (s, ac), r*r)
    return counts, rewards, squares, terminals


def estimate_oeh_certificate(arrays, initial_state_distribution, *, num_states: int,
                             num_actions: int, gamma: float = .95, k_max: int = 100,
                             reward_bound: float = 1, epsilon: float = .05,
                             delta: float = .05, method: str = "bernstein"):
    """Return certificate plus candidate policies from an already sliced pilot.

    No oracle metadata accepted. Baseline is empirical behavior, uniform at
    unvisited states as an explicit policy choice, NOT as evidence about P/R.
    epsilon normalizes by reward_bound/(1-gamma), not benchmark NR.
    """
    counts, rs, rss, terminals = trajectory_statistics(
        arrays, num_states=num_states, num_actions=num_actions, reward_bound=reward_bound)
    box = confidence_box(counts, rs, rss, reward_bound=reward_bound, delta=delta, method=method)
    plo, phi = box.p_lower.copy(), box.p_upper.copy()
    rlo, rhi = box.r_lower.copy(), box.r_upper.copy()
    # Structural terminal semantics, not fabricated observations or pseudocounts.
    for s in terminals:
        plo[s], phi[s], rlo[s], rhi[s] = 0, 0, 0, 0
        plo[s, :, s], phi[s, :, s] = 1, 1
    box = ModelBox(plo, phi, rlo, rhi, reward_bound, delta)
    visits = counts.sum(-1)
    totals = visits.sum(-1, keepdims=True)
    baseline = np.divide(visits, totals, out=np.full(visits.shape, 1/num_actions), where=totals > 0)
    policies = pessimistic_depth_policies(box, gamma, baseline, k_max)
    result = certify_depths(box, gamma, initial_state_distribution, policies, epsilon=epsilon)
    decision = np.ones(num_states, dtype=bool)
    decision[terminals] = False
    cells = visits[decision]
    result["diagnostics"] = {
        "num_transitions": int(visits.sum()),
        "observed_terminal_states": terminals.tolist(),
        "observed_decision_sa_fraction": float(np.mean(cells > 0)) if cells.size else None,
        "median_decision_visits": float(np.median(cells)) if cells.size else None,
        "positive_reward_lower_fraction": float(np.mean(rlo[decision] > 0)) if cells.size else None,
        "full_transition_box_fraction": float(np.mean(
            np.all(plo[decision] == 0, axis=-1) & np.all(phi[decision] == 1, axis=-1))) if cells.size else None,
    }
    return result, policies
