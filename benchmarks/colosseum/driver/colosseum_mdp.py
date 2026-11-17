"""Colosseum kernel -> project TabularMDP (shared by bundle building and labels)."""
from __future__ import annotations

import numpy as np

from common import RUN_SRC

SOURCE = RUN_SRC


def to_mdp(mdp_id: str, TabularMDP, metadata: dict | None = None):
    with np.load(SOURCE / f"{mdp_id}.npz") as z:
        T, R, rho = z["T"], z["R"], z["rho"]
    T = T / T.sum(-1, keepdims=True)
    S, _ = R.shape
    mdp = TabularMDP(environment_id=mdp_id, P=T.astype(np.float64),
                     R=np.repeat(R[:, :, None], S, axis=2).astype(np.float64),
                     initial_state_distribution=(rho / rho.sum()).astype(np.float64),
                     terminal_states=(), gamma=0.95, metadata=dict(metadata or {}))
    mdp.validate()
    return mdp
