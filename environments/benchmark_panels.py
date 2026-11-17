"""Pre-outcome dynamics panels; oracle checks never become pilot features."""
from __future__ import annotations

import numpy as np

PRIMARY_FAMILIES = ("uniform_garnet", "forward_layered", "bottleneck")
CONTROL_FAMILIES = ("lock_structured",)
SUPPORT_TOLERANCE = 1e-12


def inference_panel(family: str) -> str:
    if family in PRIMARY_FAMILIES:
        return "stochastic_primary"
    if family in CONTROL_FAMILIES:
        return "deterministic_control"
    raise ValueError(f"Unregistered inference family: {family}")


def audit_dynamics(mdp, family: str) -> dict:
    """Check the recipe against reachable nonterminal transition rows.

    Reachability takes the union over actions, independently of behavior or
    observed buffers. A stochastic row has two probabilities exceeding 1e-12.
    A mismatch is an error, never a reason to regenerate or reassign the MDP.
    """
    panel = inference_panel(family)
    mdp.validate()
    support = np.asarray(mdp.P) > SUPPORT_TOLERANCE
    reachable = np.asarray(mdp.initial_state_distribution) > SUPPORT_TOLERANCE
    terminals = np.zeros(mdp.num_states, dtype=bool)
    terminals[list(mdp.terminal_states)] = True
    while True:
        expanded = reachable | support[reachable & ~terminals].any(axis=(0, 1))
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    eligible = reachable & ~terminals
    if not np.any(eligible):
        raise ValueError("No reachable nonterminal state")
    counts = support[eligible].sum(axis=2)
    stochastic_rows = int(np.count_nonzero(counts >= 2))
    actual = stochastic_rows > 0
    if actual != (panel == "stochastic_primary"):
        raise ValueError(f"Dynamics/recipe mismatch for {family}")
    return {
        "panel": panel, "classification_source": "registered_family_verified_by_kernel",
        "support_tolerance": SUPPORT_TOLERANCE,
        "reachable_nonterminal_states": int(eligible.sum()),
        "eligible_state_action_rows": int(counts.size),
        "stochastic_state_action_rows": stochastic_rows,
        "has_stochastic_dynamics": actual,
        "oracle_audit_only": True,
    }


def partition_rows(rows, family_by_mdp):
    """Partition BEFORE folds/fitting; reject missing IDs and family mappings.

    Accepts mappings or dataclass instances used by return/survival models.
    Does not inspect features, outcomes, policy entropy, or empirical entropy.
    """
    panels = {"stochastic_primary": [], "deterministic_control": []}
    for row in rows:
        mdp_id = row["mdp_id"] if isinstance(row, dict) else row.mdp_id
        if mdp_id not in family_by_mdp:
            raise ValueError(f"Missing registered family for {mdp_id}")
        panels[inference_panel(family_by_mdp[mdp_id])].append(row)
    return panels
