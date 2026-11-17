"""Data-only B0--B2 features for stochastic tabular pilot trajectories."""
from __future__ import annotations

import hashlib
import math

import numpy as np


B0_NAMES = ("log_pilot_transitions", "log_state_action_count")
B1_EXTRA_NAMES = (
    "observed_sa_fraction", "observed_state_fraction",
    "normalized_visitation_entropy", "log1p_min_positive_sa_count",
    "trajectory_return_mean", "trajectory_return_std",
    "trajectory_return_iqr", "terminal_trajectory_fraction",
)
B2_EXTRA_NAMES = (
    "weighted_transition_entropy", "observed_cell_transition_entropy",
    "low_count_cell_fraction", "empirical_stochastic_cell_fraction",
    "low_coverage_entropy", "high_coverage_entropy",
    "bootstrap_return_mean_std", "bootstrap_rare_exposure_mean_std",
)


def _entropy(probabilities):
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive*np.log(positive))) if len(positive) else 0.0


def extract_stochastic_features(arrays, *, num_states, num_actions, gamma=.95,
                                bootstrap_replicates=200, seed_key=""):
    """Extract nested, finite, oracle-free features from one complete pilot."""
    required = ("state", "action", "reward", "next_state", "trajectory_id",
                "timestep", "terminal", "truncated")
    if any(name not in arrays for name in required):
        raise ValueError("Missing pilot trajectory field")
    values = {name: np.asarray(arrays[name]) for name in required}
    size = len(values["state"])
    if size == 0 or any(len(item) != size for item in values.values()):
        raise ValueError("Pilot arrays must be nonempty and aligned")
    state = values["state"].astype(int)
    action = values["action"].astype(int)
    next_state = values["next_state"].astype(int)
    reward = values["reward"].astype(float)
    if (np.any(state < 0) or np.any(state >= num_states)
            or np.any(action < 0) or np.any(action >= num_actions)
            or np.any(next_state < 0) or np.any(next_state >= num_states)
            or not np.all(np.isfinite(reward))):
        raise ValueError("Invalid pilot values")

    counts3 = np.zeros((num_states, num_actions, num_states), dtype=float)
    np.add.at(counts3, (state, action, next_state), 1)
    counts = counts3.sum(2)
    observed = counts > 0
    probabilities = counts.ravel()/size
    normalizer_sa = math.log(num_states*num_actions)
    visitation_entropy = (_entropy(probabilities)/normalizer_sa
                          if normalizer_sa > 0 else 0.0)
    positives = counts[observed]

    trajectory_ids = values["trajectory_id"].astype(int)
    unique_ids, inverse = np.unique(trajectory_ids, return_inverse=True)
    discounted = reward*np.power(gamma, values["timestep"].astype(float))
    trajectory_returns = np.zeros(len(unique_ids))
    np.add.at(trajectory_returns, inverse, discounted)
    terminal_by_trajectory = np.zeros(len(unique_ids), dtype=bool)
    np.logical_or.at(terminal_by_trajectory, inverse, values["terminal"].astype(bool))

    normalized_entropies = np.zeros_like(counts)
    if num_states > 1:
        for s, a in zip(*np.where(observed)):
            normalized_entropies[s, a] = (
                _entropy(counts3[s, a]/counts[s, a])/math.log(num_states))
    weighted_transition_entropy = float(np.sum(
        normalized_entropies*counts)/size)
    observed_entropy = float(np.mean(normalized_entropies[observed]))
    low = observed & (counts < 3)
    high = counts >= 3
    low_entropy = float(np.mean(normalized_entropies[low])) if np.any(low) else 0.0
    high_entropy = float(np.mean(normalized_entropies[high])) if np.any(high) else 0.0
    multiple_successors = np.count_nonzero(counts3 > 0, axis=2) > 1
    eligible_stochastic = counts >= 2
    stochastic_fraction = (float(np.mean(multiple_successors[eligible_stochastic]))
                           if np.any(eligible_stochastic) else 0.0)

    rare_transition = counts[state, action] <= 2
    rare_exposure = np.zeros(len(unique_ids))
    trajectory_lengths = np.zeros(len(unique_ids))
    np.add.at(rare_exposure, inverse, rare_transition.astype(float))
    np.add.at(trajectory_lengths, inverse, 1.0)
    rare_exposure = np.divide(rare_exposure, trajectory_lengths,
                              out=np.zeros_like(rare_exposure),
                              where=trajectory_lengths > 0)
    digest = hashlib.sha256(str(seed_key).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    if len(unique_ids) > 1 and bootstrap_replicates > 1:
        picks = rng.integers(0, len(unique_ids),
                             size=(bootstrap_replicates, len(unique_ids)))
        boot_return = trajectory_returns[picks].mean(1)
        boot_rare = rare_exposure[picks].mean(1)
        bootstrap_return_std = float(np.std(boot_return, ddof=1))
        bootstrap_rare_std = float(np.std(boot_rare, ddof=1))
    else:
        bootstrap_return_std = 0.0
        bootstrap_rare_std = 0.0

    result = {
        "log_pilot_transitions": math.log(size),
        "log_state_action_count": math.log(num_states*num_actions),
        "observed_sa_fraction": float(np.mean(observed)),
        "observed_state_fraction": float(len(np.unique(state))/num_states),
        "normalized_visitation_entropy": visitation_entropy,
        "log1p_min_positive_sa_count": math.log1p(float(np.min(positives))),
        "trajectory_return_mean": float(np.mean(trajectory_returns)),
        "trajectory_return_std": (float(np.std(trajectory_returns, ddof=1))
                                  if len(trajectory_returns) > 1 else 0.0),
        "trajectory_return_iqr": float(np.ptp(np.quantile(
            trajectory_returns, [.25, .75]))),
        "terminal_trajectory_fraction": float(np.mean(terminal_by_trajectory)),
        "weighted_transition_entropy": weighted_transition_entropy,
        "observed_cell_transition_entropy": observed_entropy,
        "low_count_cell_fraction": float(np.mean(low)),
        "empirical_stochastic_cell_fraction": stochastic_fraction,
        "low_coverage_entropy": low_entropy,
        "high_coverage_entropy": high_entropy,
        "bootstrap_return_mean_std": bootstrap_return_std,
        "bootstrap_rare_exposure_mean_std": bootstrap_rare_std,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite stochastic feature")
    return result


def aggregate_crossfit_oeh_profile(rows):
    """Convert cross-fit depth rows into finite B3 covariates plus missingness."""
    rows = sorted(rows, key=lambda item: int(item["qvi_index"]))
    if not rows:
        raise ValueError("Empty OEH profile")
    statuses = [str(item["status"]) for item in rows]
    finite_rows = [item for item in rows
                   if item["status"] == "identified"
                   and item.get("n_hat_eb") not in (None, "")]
    logs = np.array([math.log1p(float(item["n_hat_eb"]))
                     for item in finite_rows])
    indices = np.array([int(item["qvi_index"]) for item in finite_rows], float)
    selected = min(finite_rows,
                   key=lambda item: (float(item["n_hat_eb"]), int(item["qvi_index"])),
                   default=None)
    slope = (float(np.polyfit(indices, logs, 1)[0]) if len(logs) >= 2 else 0.0)
    curvature = (float(np.mean(np.abs(np.diff(logs, n=2))))
                 if len(logs) >= 3 else 0.0)
    count = len(rows)
    return {
        "oeh_abstained": float(selected is None),
        "oeh_identified_fraction": statuses.count("identified")/count,
        "oeh_epsilon_trivial_fraction": statuses.count("epsilon_trivial")/count,
        "oeh_unsupported_fraction": statuses.count("unsupported")/count,
        "oeh_unresolved_fraction": statuses.count("unresolved")/count,
        "oeh_not_solvable_fraction": statuses.count("not_epsilon_solvable")/count,
        "oeh_log1p_min_requirement": float(np.min(logs)) if len(logs) else 0.0,
        "oeh_selected_qvi_fraction": (int(selected["qvi_index"])/count
                                      if selected is not None else 0.0),
        "oeh_log_requirement_range": float(np.ptp(logs)) if len(logs) else 0.0,
        "oeh_profile_slope": slope,
        "oeh_profile_curvature": curvature,
    }


B3_EXTRA_NAMES = (
    "oeh_abstained", "oeh_identified_fraction", "oeh_unsupported_fraction",
    "oeh_epsilon_trivial_fraction",
    "oeh_unresolved_fraction", "oeh_not_solvable_fraction",
    "oeh_log1p_min_requirement", "oeh_selected_qvi_fraction",
    "oeh_log_requirement_range", "oeh_profile_slope", "oeh_profile_curvature",
)


def aggregate_oeh_risk_v3_profile(rows):
    """Build the frozen compact OEH-v3 block from pilot-only depth rows.

    Four variables are retained a priori: future data requirement, chosen
    planning depth, residual risk at the current pilot size, and trajectory
    half policy instability.  This avoids fitting the eleven-dimensional v2
    diagnostic block on a small number of independent MDP clusters.
    """
    rows = sorted(rows, key=lambda item: int(item["qvi_index"]))
    if not rows:
        raise ValueError("Empty OEH-v3 profile")
    finite = [item for item in rows
              if str(item["status"]) in {"identified", "epsilon_trivial"}
              and item.get("n_hat_eb") not in (None, "")]
    selected = min(
        finite,
        key=lambda item: (float(item["n_hat_eb"]), int(item["qvi_index"])),
        default=None,
    )
    if selected is None:
        # Missingness is represented by the registered maximum scales rather
        # than a fifth indicator: risk and instability equal one, while the
        # censored requirement is one order beyond the largest finite value.
        requirements = [float(item["n_hat_eb"]) for item in rows
                        if item.get("n_hat_eb") not in (None, "")]
        reference = max(requirements, default=1.0)
        return {
            "oeh3_log1p_requirement": math.log1p(10.0 * reference),
            "oeh3_selected_qvi_fraction": 1.0,
            "oeh3_risk_at_pilot": 1.0,
            "oeh3_policy_instability": 1.0,
        }
    count = len(rows)
    result = {
        "oeh3_log1p_requirement": math.log1p(float(selected["n_hat_eb"])),
        "oeh3_selected_qvi_fraction": int(selected["qvi_index"]) / count,
        "oeh3_risk_at_pilot": float(selected["risk_at_pilot"]),
        "oeh3_policy_instability": float(selected["policy_instability"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite OEH-v3 feature")
    return result


OEH_V3_EXTRA_NAMES = (
    "oeh3_log1p_requirement", "oeh3_selected_qvi_fraction",
    "oeh3_risk_at_pilot", "oeh3_policy_instability",
)


__all__ = [
    "B0_NAMES", "B1_EXTRA_NAMES", "B2_EXTRA_NAMES", "B3_EXTRA_NAMES",
    "OEH_V3_EXTRA_NAMES", "extract_stochastic_features",
    "aggregate_crossfit_oeh_profile", "aggregate_oeh_risk_v3_profile",
]
